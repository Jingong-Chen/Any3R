# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Construction of one benchmark scene from its three dataset panoramas.

1. read the three frames named by the manifest (loaders.read_frame);
2. turn each view to its frozen yaw: the panorama arrays are rolled left by yaw_shift_px columns and
   the world-to-camera is composed with R_y(yaw)^T, yaw = yaw_shift_px * 360 / 1036 degrees;
3. for every camera model, round-trip each view through the lens on the model's round-trip sensor
   (render.roundtrip_view) and store the result as uint8.
"""
import math

import numpy as np

from . import protocol as P
from .loaders import read_frame
from .render import make_camera, roundtrip_view


def yaw_rotation(yaw_deg):
    """View-to-panorama rotation for a view whose optical axis lies at azimuth yaw (degrees) of the
    panorama (x right, y down, z forward): columns right, down, forward. Equals R_y(yaw)."""
    psi = math.radians(float(yaw_deg))
    fwd = np.array([math.sin(psi), 0.0, math.cos(psi)])
    up0 = np.array([0.0, -1.0, 0.0])
    right = np.cross(fwd, up0)
    right /= np.linalg.norm(right)
    down = -np.cross(right, fwd)
    return np.stack([right, down, fwd], axis=1)


def yaw_deg_of_shift(shift_px, width=P.ERP_W):
    return float(shift_px) * 360.0 / float(width)


def compose_w2c(w2c_pano, yaw_deg):
    """World-to-camera of the turned view, float32."""
    return (yaw_rotation(yaw_deg).T @ np.asarray(w2c_pano, np.float64)).astype(np.float32)


def roll_view(a, shift_px):
    """Panorama array (..., W) -> the turned view: view column j = panorama column j + shift_px."""
    s = int(shift_px)
    return a if s == 0 else np.roll(a, -s, axis=-1)


def source_arrays(ds, root, scene_rec):
    """The turned source panoramas of a scene: dict(rgb (3, 3, H, W) float32, rng (3, H, W) float32,
    msk (3, H, W) bool, ext (3, 3, 4) float32, ext_loader (3, 3, 4) float32)."""
    rgb, rng, msk, ext, ext0 = [], [], [], [], []
    for fr in scene_rec["frames"]:
        f = read_frame(ds, root, fr)
        s = fr["yaw_shift_px"]
        yaw = yaw_deg_of_shift(s)
        if yaw != fr["yaw_deg"]:
            raise ValueError(f"yaw {yaw} of shift {s} != manifest {fr['yaw_deg']}")
        rgb.append(roll_view(f["rgb"], s))
        rng.append(roll_view(f["depth"], s))
        msk.append(roll_view(f["mask"], s))
        ext0.append(f["w2c"])
        ext.append(compose_w2c(f["w2c"], yaw))
    return dict(rgb=np.stack(rgb).astype(np.float32), rng=np.stack(rng).astype(np.float32),
                msk=np.stack(msk).astype(bool), ext=np.stack(ext).astype(np.float32),
                ext_loader=np.stack(ext0).astype(np.float32))


def camera_cell(kind, src):
    """erp_rgb (3, 3, H, W) uint8 and cone (3, H, W) bool of one camera model for a scene."""
    W, H = P.roundtrip_sensor(kind)
    cam = make_camera(kind, W, H)
    views = src["rgb"].transpose(0, 2, 3, 1)
    erp = np.zeros(views.shape, np.float32)
    cone = np.zeros(src["msk"].shape, bool)
    for k in range(len(views)):
        erp[k], cone[k] = roundtrip_view(cam, W, H, views[k], src["rng"][k], src["msk"][k])
    return (np.transpose(erp, (0, 3, 1, 2)) * 255).astype(np.uint8), cone
