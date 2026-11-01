# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Rendering of benchmark views from source panoramas through a camera model.

Two operations, both pure numpy:
  render_view      source panorama -> the camera's native image, the ray of every pixel, the ground
                   truth range along it and the mask of scored pixels (evaluation);
  roundtrip_view   source panorama -> native image on the camera's round-trip sensor -> back onto the
                   panorama through the camera's own projection (benchmark construction). The
                   distortion profile and the resampling it implies survive the trip, so two lenses with
                   the same field of view do not hand the model identical pixels.
"""
import math

import numpy as np

from any3r import cameras as C

from . import protocol as P
from .sampling import bearing_to_uv, erp_directions, sample_erp, sample_image


def make_camera(kind, W, H):
    """The benchmark camera of a cell at its operating point on a W x H image."""
    return C.from_fov(kind, W, H, P.CAMERA_FOV[kind], principal_point=((W - 1) / 2.0, (H - 1) / 2.0),
                      circular=True)


def camera_rays(cam, W, H):
    """(rays (H, W, 3) float32, valid (H, W) bool, sampling rays (H, W, 3) float64).

    The sampling rays locate each pixel on the source panorama. They are the float32 unit rays,
    except for the cylindrical model, whose views were sampled with its unrounded float64 rays when
    the benchmark was defined; both are kept so the rendered pixels match it bit for bit."""
    rays, valid = cam.ray_map(H, W)
    if cam.kind == "cylindrical":
        u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        d64, _ = cam.pixel_to_ray(u.astype(np.float64), v.astype(np.float64))
    else:
        d64 = rays.astype(np.float64)
    return rays, np.asarray(valid), d64


def render_view(cam, W, H, erp_rgb, erp_rng, erp_msk):
    """One evaluation view.

    erp_rgb (EH, EW, 3) float32 in [0, 1], erp_rng (EH, EW) float32 ground-truth range (metres),
    erp_msk (EH, EW) bool. Returns rgb (H, W, 3) float32 (black outside the scored pixels),
    mask (H, W) bool, rays (H, W, 3) float32 and range (H, W) float32."""
    rays, valid, d64 = camera_rays(cam, W, H)
    uv = bearing_to_uv(d64)
    rgb = sample_erp(erp_rgb, uv, "bilinear")
    rad = sample_erp(erp_rng, uv, "bilinear")
    mv = sample_erp(erp_msk.astype(np.float32), uv, "nearest") > 0.5
    m = valid & mv & (rad > P.MIN_GT_RANGE)
    rgb = rgb.copy()
    rgb[~m] = 0
    return rgb.astype(np.float32), m, rays.astype(np.float32), rad.astype(np.float32)


_ERP_DIRS = {}


def _erp_dirs(H, W):
    if (H, W) not in _ERP_DIRS:
        _ERP_DIRS[(H, W)] = erp_directions(H, W)
    return _ERP_DIRS[(H, W)]


def roundtrip_view(cam, W, H, src_rgb, src_rng, src_msk):
    """One view of the benchmark construction for a camera on its W x H round-trip sensor.

    src_rgb (EH, EW, 3) float32 in [0, 1], src_rng (EH, EW) float32, src_msk (EH, EW) bool: the view's
    source panorama. Returns the panorama as seen through the lens, (EH, EW, 3) float32 (zero outside
    its support), and the support (EH, EW) bool (inside the lens, inside the sensor, scored)."""
    _rays, valid, d64 = camera_rays(cam, W, H)
    uv = bearing_to_uv(d64)
    nat_rgb = sample_erp(src_rgb, uv, "bilinear")
    nat_rad = sample_erp(src_rng, uv, "bilinear")
    nat_mv = sample_erp(src_msk.astype(np.float32), uv, "nearest") > 0.5
    nat_m = valid & nat_mv & (nat_rad > P.MIN_GT_RANGE)
    nat_rgb = nat_rgb.copy()
    nat_rgb[~nat_m] = 0
    nat_rgb = nat_rgb.astype(np.float32)

    EH, EW = src_msk.shape
    dirs = _erp_dirs(EH, EW)
    u, v, ok = cam.project(dirs.astype(np.float64))
    # the projection of some models (MEI, double sphere) is defined beyond the lens' half-angle
    tm = getattr(cam, "theta_max", None)
    if tm is not None and tm < math.pi:
        ang = np.arccos(np.clip(dirs[..., 2].astype(np.float64), -1.0, 1.0))
        ok = np.asarray(ok) & (ang <= tm + 1e-9)
    uu, vv = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
    cm = sample_image(nat_m.astype(np.float32), uu, vv, "nearest") > 0.5
    inb = (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
    support = np.asarray(ok) & inb & cm & src_msk
    rgb = sample_image(nat_rgb, uu, vv, "bilinear")
    out = np.zeros((EH, EW, 3), np.float32)
    out[support] = rgb[support]
    return out, support
