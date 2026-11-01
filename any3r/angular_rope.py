# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Angular coordinates for the rotary position embedding (Sec. 3.2 of the paper).

The backbone's RoPE locates a patch by its row and column. Any3R locates it by the elevation and
azimuth of the patch's mean ray instead:

    lat = (Hp / m) * (1 - el / el_half) - 1,     lon = (Wp / m) * (az / az_half + 1) - 1,

with m = max(Hp, Wp). With el_half = pi/2 and az_half = pi this reproduces the stock coordinates
on an equirectangular grid (at patch centres), so the pretrained encoding is kept on panoramas.

The released model splits the 16 RoPE frequencies: the 8 lowest frequencies use the absolute angles
above (one coordinate unit is the same angle on every camera), and the 8 highest frequencies divide
by the view's own half extents instead, so that they span each camera's field of view.
Coordinates are computed per view, from that view's own rays.
"""
import math

import torch


def _half_extents(el, az, Hp, Wp, eps=1e-6):
    """Nominal half extents of the view from its patch rays: the observed maximum |angle| times
    n / (n - 1), which undoes the half-patch offset of patch centres."""
    kh = Hp / max(Hp - 1.0, 1.0)
    kw = Wp / max(Wp - 1.0, 1.0)
    el_half = (el.abs().amax(dim=-1, keepdim=True) * kh).clamp_min(eps)
    az_half = (az.abs().amax(dim=-1, keepdim=True) * kw).clamp_min(eps)
    return el_half.clamp_max(math.pi / 2), az_half.clamp_max(math.pi)


def rays_to_rope_coords(ray_patch, Hp, Wp, normalize_coords="max", fov_normalize=False):
    """(..., Hp*Wp, 3) patch rays -> (..., Hp*Wp, 2) (lat, lon) RoPE coordinates."""
    x, y, z = ray_patch[..., 0], ray_patch[..., 1], ray_patch[..., 2]
    n = torch.linalg.norm(ray_patch, dim=-1).clamp_min(1e-9)
    el = torch.asin((-y / n).clamp(-1.0, 1.0))
    az = torch.atan2(x, z)
    if normalize_coords == "max":
        mh = mw = float(max(Hp, Wp))
    elif normalize_coords == "min":
        mh = mw = float(min(Hp, Wp))
    else:
        mh, mw = float(Hp), float(Wp)
    if fov_normalize:
        el_half, az_half = _half_extents(el, az, Hp, Wp)
    else:
        el_half, az_half = math.pi / 2, math.pi
    lat = (Hp / mh) * (1.0 - el / el_half) - 1.0
    lon = (Wp / mw) * (az / az_half + 1.0) - 1.0
    return torch.stack([lat, lon], dim=-1)


def banded_rope_coords(ray_patch, Hp, Wp, n_periods, split, normalize_coords="max"):
    """(..., Hp*Wp, 2, P): frequency index k < split uses view-normalised coordinates, k >= split
    absolute angular coordinates (periods increase with k, so k < split are the high frequencies)."""
    lo = rays_to_rope_coords(ray_patch, Hp, Wp, normalize_coords, fov_normalize=False)
    hi = rays_to_rope_coords(ray_patch, Hp, Wp, normalize_coords, fov_normalize=True)
    out = lo.unsqueeze(-1).expand(*lo.shape, n_periods).clone()
    out[..., :split] = hi.unsqueeze(-1).expand(*hi.shape, split)
    return out


class AngularRope(torch.nn.Module):
    """Wraps the backbone's RopePositionEmbedding (kept as `.rope`, so its `periods` buffer keeps
    the checkpoint key `rope_embed.rope.periods`) and computes its coordinates from patch rays."""

    def __init__(self, rope, band_split: int = 8):
        super().__init__()
        self.rope = rope
        self.band_split = int(band_split)

    def forward(self, H: int, W: int, patch_rays=None):
        if patch_rays is None:
            return self.rope(H=H, W=W)
        periods = self.rope.periods
        device, dtype = periods.device, self.rope.dtype
        # float64 for the angle round trip, so that on an equirectangular grid the coordinates
        # match the stock ones to float32 precision
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
