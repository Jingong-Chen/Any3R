# Copyright (c) 2026 The Any3R Authors.
# Portions adapted from PanoVGGT (MIT License, Copyright (c) 2026 Yijing Guo).
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Ray-field geometry used by the model.

Camera frame convention (OpenCV): x right, y down, z along the optical axis. A ray field is an
(H, W, 3) array of unit directions, one per pixel centre.
"""
import math

import torch
import torch.nn.functional as F

_DIR_CACHE: dict = {}


def erp_directions(H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Ray field of an equirectangular (360 x 180 degree) panorama, (H, W, 3).

    Pixel centres at (j + 0.5, i + 0.5); azimuth = (u/W - 0.5) * 2pi, elevation = -(v/H - 0.5) * pi.
    This is also the ray field the model assumes when none is given.
    """
    u = torch.arange(W, device=device, dtype=dtype) + 0.5
    v = torch.arange(H, device=device, dtype=dtype) + 0.5
    azim = (u / W - 0.5) * (2.0 * math.pi)
    elev = -(v / H - 0.5) * math.pi
    el, az = torch.meshgrid(elev, azim, indexing="ij")
    return torch.stack([torch.cos(el) * torch.sin(az), -torch.sin(el), torch.cos(el) * torch.cos(az)], dim=-1)


def erp_directions_cached(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    key = (int(H), int(W), str(device), dtype)
    if key not in _DIR_CACHE:
        _DIR_CACHE[key] = erp_directions(H, W, device=device, dtype=dtype)
    return _DIR_CACHE[key]


def ray_to_spherical_features(rays: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """(..., 3) rays -> (..., 4) [sin(lat), cos(lat), sin(lon), cos(lon)], latitude positive down.

    These feed the absolute position embedding; on an equirectangular grid they equal PanoVGGT's
    patch-grid features.
    """
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    r = torch.sqrt(x * x + y * y + z * z).clamp_min(eps)
    rho = torch.sqrt(x * x + z * z).clamp_min(eps)
    return torch.stack([y / r, rho / r, x / rho, z / rho], dim=-1)


def patch_mean_rays(ray_maps: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
    """(B, S, H, W, 3) per-pixel rays -> (B*S, Hp*Wp, 3) area-averaged ray of every patch."""
    B, S, H, W, _ = ray_maps.shape
    r = ray_maps.reshape(B * S, H, W, 3).permute(0, 3, 1, 2)
    r = F.adaptive_avg_pool2d(r, (Hp, Wp))
    return r.permute(0, 2, 3, 1).reshape(B * S, Hp * Wp, 3)


def raymap_to_patch_features(ray_maps: torch.Tensor, Hp: int, Wp: int) -> torch.Tensor:
    """(B, S, H, W, 3) per-pixel rays -> (B, S, Hp*Wp, 4) spherical features of each patch's mean ray."""
    B, S, H, W, _ = ray_maps.shape
    r = ray_maps.permute(0, 1, 4, 2, 3).reshape(B * S, 3, H, W)
    r = F.adaptive_avg_pool2d(r, (Hp, Wp)).permute(0, 2, 3, 1)
    return ray_to_spherical_features(r).reshape(B, S, Hp * Wp, 4)


def homogenize_points(xyz: torch.Tensor) -> torch.Tensor:
    return torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)


def pose_encoding_to_c2w(pose_enc: torch.Tensor, quat_to_mat) -> torch.Tensor:
    """9-d pose encoding [t(3), quat_xyzw(4), fov_h, fov_w] of the world-to-camera transform
    -> (B, S, 4, 4) camera-to-world matrices. The two field-of-view entries are not used."""
    T = pose_enc[..., :3]
    R_w2c = quat_to_mat(pose_enc[..., 3:7])
    R_c2w = R_w2c.transpose(-1, -2)
    t_c2w = -torch.einsum("...ij,...j->...i", R_c2w, T)
    B, S = pose_enc.shape[:2]
    c2w = torch.eye(4, device=pose_enc.device, dtype=pose_enc.dtype).view(1, 1, 4, 4).repeat(B, S, 1, 1)
    c2w[..., :3, :3] = R_c2w
    c2w[..., :3, 3] = t_c2w
    return c2w
