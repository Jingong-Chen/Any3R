# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Equirectangular panorama <-> unit ray conversions and image sampling (numpy, float64).

Camera frame: x right, y down, z forward. A panorama pixel (row i, column j) has its centre at
u = (j + 0.5) / W, v = (i + 0.5) / H; azimuth atan2(x, z) grows with u, elevation decreases with v.
"""
import math

import numpy as np

TWO_PI = 2.0 * math.pi


def erp_directions(H, W):
    """Unit ray of every panorama pixel centre, (H, W, 3) float32."""
    u = np.arange(W, dtype=np.float64) + 0.5
    v = np.arange(H, dtype=np.float64) + 0.5
    az = (u / W - 0.5) * TWO_PI
    el = -(v / H - 0.5) * math.pi
    EL, AZ = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(EL) * np.sin(AZ), -np.sin(EL), np.cos(EL) * np.cos(AZ)], -1).astype(np.float32)


def bearing_to_uv(b, eps=1e-8):
    """Rays (..., 3), not necessarily unit -> normalised panorama coordinates (..., 2) in [0, 1]."""
    x, y, z = b[..., 0], b[..., 1], b[..., 2]
    r = np.maximum(np.linalg.norm(b, axis=-1), eps)
    u = np.arctan2(x, z) / TWO_PI + 0.5
    v = 0.5 - np.arcsin(np.clip(-y / r, -1.0, 1.0)) / math.pi
    return np.stack([u, v], -1)


def sample_erp(img, uv, mode="bilinear"):
    """Sample a panorama (H, W) or (H, W, C) at normalised coordinates uv (..., 2): periodic in
    azimuth, clamped in elevation."""
    squeeze = img.ndim == 2
    img2 = img[..., None] if squeeze else img
    H, W, _ = img2.shape
    fx = uv[..., 0] * W - 0.5
    fy = uv[..., 1] * H - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = (fx - x0)[..., None]
    wy = (fy - y0)[..., None]
    x0w, x1w = np.mod(x0, W), np.mod(x0 + 1, W)
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    if mode == "nearest":
        out = img2[np.where(wy[..., 0] < 0.5, y0c, y1c), np.where(wx[..., 0] < 0.5, x0w, x1w)]
    else:
        out = (img2[y0c, x0w] * (1 - wx) * (1 - wy) + img2[y0c, x1w] * wx * (1 - wy)
               + img2[y1c, x0w] * (1 - wx) * wy + img2[y1c, x1w] * wx * wy)
    return out[..., 0] if squeeze else out


def sample_image(img, u, v, mode="bilinear"):
    """Sample an image (H, W) or (H, W, C) at pixel coordinates u (column), v (row), clamped at the
    border (integer coordinates address pixel centres)."""
    squeeze = img.ndim == 2
    img2 = img[..., None] if squeeze else img
    H, W, _ = img2.shape
    if mode == "nearest":
        out = img2[np.clip(np.round(v).astype(np.int64), 0, H - 1), np.clip(np.round(u).astype(np.int64), 0, W - 1)]
    else:
        x0 = np.floor(u).astype(np.int64)
        y0 = np.floor(v).astype(np.int64)
        wx = (u - x0)[..., None]
        wy = (v - y0)[..., None]
        x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
        y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
        out = (img2[y0c, x0c] * (1 - wx) * (1 - wy) + img2[y0c, x1c] * wx * (1 - wy)
               + img2[y1c, x0c] * (1 - wx) * wy + img2[y1c, x1c] * wx * wy)
    return out[..., 0] if squeeze else out
