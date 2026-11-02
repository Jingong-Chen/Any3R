# Copyright (c) 2026 The Any3R Authors.
# Adapted from VGGT-Omega (Copyright (c) Meta Platforms, Inc. and affiliates) and PanoVGGT (MIT License,
# Copyright (c) 2026 Yijing Guo).
#
# This file is a derivative of VGGT-Omega and is licensed under the FAIR Noncommercial Research License
# found in the LICENSE-FAIR file in the root directory of this source tree. It is the only part of the
# any3r package, besides the vendored any3r/_vggt_omega, that is not under the MIT license.

"""The VGGT-Omega trunk, conditioned on the ray field:

  * AngularRope feeds the backbone's rotary position embedding with angular coordinates computed per
    view from the patch rays (any3r/angular_rope.py) instead of patch row and column;
  * RayAggregator adds an absolute position embedding alpha * MLP([sin lat, cos lat, sin lon, cos lon]),
    evaluated at each patch's mean ray, to the patch tokens before every attention block;
  * run_trunk runs the aggregator under bfloat16 autocast and the camera and dense heads in float32.
"""
import math

import torch
import torch.nn as nn

from ._vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten
from .angular_rope import banded_rope_coords


class AngularRope(nn.Module):
    """Wraps the backbone's RopePositionEmbedding (kept as `.rope`, so its `periods` buffer keeps the
    checkpoint key `rope_embed.rope.periods`) and computes its coordinates from patch rays."""

    def __init__(self, rope, band_split: int = 8):
        super().__init__()
        self.rope = rope
        self.band_split = int(band_split)

    def forward(self, H: int, W: int, patch_rays=None):
        if patch_rays is None:
            return self.rope(H=H, W=W)
        periods = self.rope.periods
        device, dtype = periods.device, self.rope.dtype
        # float64 for the angle round trip, so that on an equirectangular grid the coordinates match
        # the stock ones to float32 precision
        r = patch_rays.to(device=device, dtype=torch.float64)
        batched = r.dim() == 3
        coords = banded_rope_coords(r, H, W, int(periods.numel()), self.band_split,
                                    self.rope.normalize_coords).to(dtype)
        angles = 2 * math.pi * coords / (periods if not batched else periods[None, None, None, :])
        if batched:
            angles = angles.flatten(2, 3).tile(1, 1, 2).unsqueeze(1)
        else:
            angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        old, new = prefix + "periods", prefix + "rope.periods"
        if old in state_dict and new not in state_dict:
            state_dict[new] = state_dict.pop(old)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                             missing_keys, unexpected_keys, error_msgs)


class RayAggregator(Aggregator):
    def __init__(self, patch_size: int = 16, embed_dim: int = 1024, pos_mlp_hidden: int = 1024,
                 band_split: int = 8, **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, **kwargs)
        self.pano_pos_mlp = nn.Sequential(nn.Linear(4, pos_mlp_hidden), nn.GELU(),
                                          nn.Linear(pos_mlp_hidden, embed_dim))
        self.alpha_pos = nn.Parameter(torch.tensor(0.0))
        self.rope_embed = AngularRope(self.rope_embed, band_split=band_split)

    def _pos_add(self, ray_features, num_tokens, ref):
        """(B*S, Hp*Wp, 4) -> (B*S, N, C): zeros for the camera and register tokens."""
        pos_patch = self.pano_pos_mlp(ray_features)
        BS, _, C = pos_patch.shape
        zeros = torch.zeros(BS, self.patch_token_start, C, device=pos_patch.device, dtype=pos_patch.dtype)
        pos_full = torch.cat([zeros, pos_patch], dim=1)
        if pos_full.shape[1] != num_tokens:
            raise ValueError(f"position rows {pos_full.shape[1]} != token count {num_tokens}")
        return (self.alpha_pos.to(pos_full.dtype) * pos_full).to(ref.dtype)

    def forward(self, images, ray_features, patch_rays):
        """images (B, S, 3, H, W) in [0, 1], H and W multiples of the patch size;
        ray_features (B, S, Hp*Wp, 4); patch_rays (B*S, Hp*Wp, 3)."""
        B, S, C_in, H, W = images.shape
        if C_in != 3:
            raise ValueError(f"expected 3 input channels, got {C_in}")
        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(B * S, C_in, H, W)

        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, N, C = tokens.shape

        Hp, Wp = H // self.patch_size, W // self.patch_size
        s, c = self.rope_embed(Hp, Wp, patch_rays)
        frame_rope = (s.to(device=patch_tokens.device, dtype=torch.float32),
                      c.to(device=patch_tokens.device, dtype=torch.float32))

        rf = ray_features.reshape(B * S, -1, 4).to(tokens.dtype)
        pos_add = self._pos_add(rf, N, tokens)

        outputs = []
        for block_idx in range(self.depth):
            tokens = tokens.reshape(B * S, N, C) + pos_add
            tokens, frame_tokens = self._run_frame_block(tokens, B, S, N, C, block_idx, frame_rope)
            tokens = tokens.reshape(B * S, N, C) + pos_add
            tokens = self._run_inter_frame_attention_block(tokens, B, S, N, C, block_idx,
                                                           self.inter_frame_attention_types[block_idx])
            outputs.append(torch.cat([frame_tokens, tokens], dim=-1)
                           if block_idx in self.cached_layer_indices else None)
        return outputs, self.patch_token_start


def run_trunk(model, images, ray_features, patch_rays):
    """Aggregator under bfloat16 autocast (float16 where bf16 is unsupported), heads in float32.
    `model` holds .aggregator, .camera_head, .dense_head and .frames_chunk_size.
    Returns pose_enc (B, S, 9), depth (B, S, H, W, 1) and depth_conf (B, S, H, W)."""
    amp_dtype = torch.bfloat16 if (images.is_cuda and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.autocast(device_type=images.device.type, dtype=amp_dtype, enabled=images.is_cuda):
        tokens_list, patch_token_start = model.aggregator(images, ray_features, patch_rays)
    with torch.autocast(device_type=images.device.type, enabled=False):
        pose_enc = model.camera_head(tokens_list, patch_token_start=patch_token_start)
        depth, depth_conf = model.dense_head(tokens_list, images=images, patch_token_start=patch_token_start,
                                             frames_chunk_size=model.frames_chunk_size)
    return pose_enc, depth, depth_conf
