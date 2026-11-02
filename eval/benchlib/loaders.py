# Copyright (c) 2026 The Any3R Authors.
# Portions adapted from PanoVGGT, training/data/datasets/{matterport3d,stanford2d3ds,panocity}.py
# (https://github.com/YijingGuo-June/PanoVGGT). PanoVGGT is distributed under the MIT License:
#
#   Copyright (c) 2026 Yijing Guo
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
#   associated documentation files (the "Software"), to deal in the Software without restriction,
#   including without limitation the rights to use, copy, modify, merge, publish, distribute,
#   sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#   NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
#   OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Frame readers for the three benchmark datasets, as laid out by PanoVGGT's data preparation.

Each reader turns one panorama (RGB file, depth file, pose file; paths from the benchmark
manifest) into the arrays PanoVGGT's test-split loader produces at its fixed 518 x 1036 resolution:
  rgb    (3, H, W) float32 in [0, 1]   OpenCV decode, INTER_AREA resize, value / 255
  depth  (H, W) float32, metres        OpenCV decode, INTER_NEAREST resize, value / scale, 0 beyond
                                       the dataset's depth cut-off
  mask   (H, W) bool                   0.1 m < depth <= cut-off
  w2c    (3, 4) float32                world-to-camera in OpenCV axes (x right, y down, z forward)
Only the frames named in the manifest are read; there is no scene indexing or sampling here.
"""
import json
import os

import cv2
import numpy as np

from . import protocol as P

TARGET_HW = (P.ERP_H, P.ERP_W)


def read_rgb(path, hw=TARGET_HW):
    h, w = hw
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot decode image {path}")
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)


def read_depth(path, scale, depth_max, hw=TARGET_HW):
    h, w = hw
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(f"cannot decode depth {path}")
    if d.ndim == 3:
        if d.shape[2] == 3:
            d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        elif d.shape[2] == 4:
            d = cv2.cvtColor(d, cv2.COLOR_BGRA2GRAY)
        else:
            d = d[..., 0]
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    m = d.astype(np.float32) / float(scale)
    m = m.astype(np.float64)
    m[m > depth_max] = 0.0
    m[~np.isfinite(m)] = 0.0
    return m.astype(np.float32)


def valid_mask(depth, depth_min, depth_max):
    d = np.asarray(depth, np.float32)
    return (d > np.float32(depth_min)) & (d <= np.float32(depth_max)) & ~np.isnan(d)


def c2w_to_w2c(c2w):
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    out = np.zeros((3, 4), dtype=np.float32)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def m3d_pose(path):
    """Matterport3D pano_poses/<view>.txt: camera-to-world, world x right / y forward / z up and
    camera x right / y up / z backward -> OpenCV world-to-camera."""
    c2w = np.loadtxt(path, dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError(f"bad pose matrix {c2w.shape} in {path}")
    cam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
    world = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    return c2w_to_w2c(world @ c2w @ np.linalg.inv(cam))


def s2d3ds_pose(path):
    """Stanford2D3DS pano/pose/*_pose.json: camera_rt_matrix, world-to-camera with world x right /
    y forward / z up -> OpenCV world-to-camera."""
    with open(path, "r") as f:
        rt = np.array(json.load(f)["camera_rt_matrix"], dtype=np.float32)
    if rt.shape != (3, 4):
        raise ValueError(f"bad pose matrix {rt.shape} in {path}")
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :] = rt
    world = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    return (w2c @ np.linalg.inv(world))[:3, :]


def panocity_pose(path, frame_name):
    """PanoCity poses file: frames[].transformation_matrix, camera-to-world in OpenCV axes."""
    with open(path, "r") as f:
        frames = json.load(f).get("frames", [])
    c2w = None
    for fr in frames:                      # as PanoVGGT's reader: the last valid entry of a name wins
        if fr.get("name") != frame_name or fr.get("transformation_matrix") is None:
            continue
        m = np.array(fr["transformation_matrix"], dtype=np.float32)
        if m.shape == (4, 4) and np.isfinite(m).all():
            c2w = m
    if c2w is None:
        raise KeyError(f"no valid pose for {frame_name} in {path}")
    return c2w_to_w2c(c2w)


def read_frame(ds, root, frame):
    """One manifest frame -> dict(rgb, depth, mask, w2c). `frame` holds the file paths relative to
    the dataset root (frame['files'][rgb|depth|pose]['path']) and the frame name."""
    cfg = P.LOADERS[ds]
    files = {k: os.path.join(root, v["path"]) for k, v in frame["files"].items()}
    if ds == "M3D":
        w2c = m3d_pose(files["pose"])
    elif ds == "S2D3DS":
        w2c = s2d3ds_pose(files["pose"])
    elif ds == "PanoCity":
        w2c = panocity_pose(files["pose"], frame["name"])
    else:
        raise KeyError(ds)
    if not np.isfinite(w2c).all():
        raise ValueError(f"non-finite pose for {frame['name']}")
    rgb = read_rgb(files["rgb"])
    depth = read_depth(files["depth"], cfg["depth_scale"], cfg["depth_max"])
    return dict(rgb=rgb, depth=depth, mask=valid_mask(depth, cfg["depth_min"], cfg["depth_max"]),
                w2c=np.asarray(w2c, np.float32))
