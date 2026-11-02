# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
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
Coordinates are computed per view, from that view's own rays. The rotary embedding that consumes
them is in any3r/aggregator.py.
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
