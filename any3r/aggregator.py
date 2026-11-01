# Copyright (c) 2026 The Any3R Authors.
# Portions adapted from PanoVGGT (MIT License, Copyright (c) 2026 Yijing Guo) and VGGT-Omega
# (Copyright (c) Meta Platforms, Inc. and affiliates, FAIR Noncommercial Research License).
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Ray-conditioned aggregator: the VGGT-Omega aggregator with
  * angular coordinates for the rotary position embedding (frame attention), per view, and
  * an absolute position embedding alpha * MLP([sin lat, cos lat, sin lon, cos lon]) evaluated at
    each patch's mean ray, added to the patch tokens before every attention block.
"""
import torch
import torch.nn as nn

from ._vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten
from .angular_rope import AngularRope


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
