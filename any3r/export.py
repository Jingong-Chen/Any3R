# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Turn predictions into files: a coloured point cloud (PLY / GLB), camera poses (JSON) and
per-view depth maps (NPZ)."""
import json
import math
import os
from typing import Optional

import numpy as np
import torch

from .io import Views


def _np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def depth_edges(depth: np.ndarray, rel: float = 0.03) -> np.ndarray:
    """(H, W) radial depth -> True where it jumps by more than `rel` (relative) to a 4-neighbour,
    i.e. the flying points that interpolate between a foreground edge and the background."""
    d = depth
    jx = np.abs(d[:, 1:] - d[:, :-1]) > rel * np.minimum(d[:, 1:], d[:, :-1])
    jy = np.abs(d[1:, :] - d[:-1, :]) > rel * np.minimum(d[1:, :], d[:-1, :])
    e = np.zeros(d.shape, bool)
    e[:, 1:] |= jx
    e[:, :-1] |= jx
    e[1:, :] |= jy
    e[:-1, :] |= jy
    return e


def consistency_mask(pred: dict, views: Views, tol: float = 0.02, min_views: int = 1, neighbours: int = 10,
                     edge: Optional[float] = 0.03) -> np.ndarray:
    """(S, H, W) mask of the pixels whose 3D point is confirmed by at least `min_views` other views:
    projected into another view (with that view's own camera model), the point's distance to that
    camera agrees with the depth predicted there within `tol` (relative). Each view is checked
    against its `neighbours` nearest cameras. With `edge`, pixels on depth discontinuities are
    dropped as well. Uses only the model's predictions."""
    pts = _np(pred["world_points"]).astype(np.float64)
    dep = _np(pred["depth"])[..., 0]
    c2w = _np(pred["camera_poses"]).astype(np.float64)
    valid = _np(views.valid).astype(bool) & np.isfinite(pts).all(-1)
    S = pts.shape[0]
    centres = c2w[:, :3, 3]
    out = np.zeros(valid.shape, bool)
    for i in range(S):
        m = valid[i].copy()
        if edge:
            m &= ~depth_edges(dep[i], edge)
        P = pts[i][m]
        cnt = np.zeros(len(P), np.int32)
        order = [j for j in np.argsort(np.linalg.norm(centres - centres[i], axis=1)) if j != i][:neighbours]
        for j in order:
            Xc = (P - c2w[j, :3, 3]) @ c2w[j, :3, :3]
            u, v, ok = views.cameras[j].project(Xc)
            t, l, h, w = views.boxes[j]
            ok &= np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            col = np.clip(np.floor(np.where(ok, u, 0)).astype(np.int64), 0, w - 1) + l
            row = np.clip(np.floor(np.where(ok, v, 0)).astype(np.int64), 0, h - 1) + t
            dj = dep[j, row, col]
            r = np.linalg.norm(Xc, axis=1)
            cnt += ok & valid[j, row, col] & (np.abs(r - dj) < tol * dj)
        out[i][m] = cnt >= min_views
    return out


def scene_points(pred: dict, views: Views, conf_percentile: float = 10.0, max_points: Optional[int] = None,
                 filter_black: bool = True, min_views: int = 0, seed: int = 0):
    """Fuse the per-view points of one scene.

    Keeps pixels inside a lens whose confidence is above the given percentile of the scene's
    confidences; with filter_black, drops pure-black pixels (masked skies or backgrounds in
    rendered data); with min_views > 0, keeps only points confirmed by that many other views
    (see consistency_mask). Returns xyz (N, 3) float32 in the frame of the first camera, rgb (N, 3)
    uint8 and the view index of every point."""
    pts = _np(pred["world_points"])
    conf = _np(pred["depth_conf"])
    valid = _np(views.valid).astype(bool) & np.isfinite(pts).all(-1)
    if min_views > 0 and pts.shape[0] > 1:
        valid &= consistency_mask(pred, views, min_views=min_views)
    if conf_percentile and conf_percentile > 0:
        thr = np.percentile(conf[valid], conf_percentile)
        valid &= conf >= thr
    rgb = (_np(views.images).transpose(0, 2, 3, 1) * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    if filter_black:
        valid &= rgb.max(-1) > 4
    view_idx = np.broadcast_to(np.arange(pts.shape[0])[:, None, None], valid.shape)
    xyz, col, vid = pts[valid].astype(np.float32), rgb[valid], view_idx[valid].astype(np.int32)
    if max_points is not None and len(xyz) > max_points:
        keep = np.random.default_rng(seed).choice(len(xyz), max_points, replace=False)
        keep.sort()
        xyz, col, vid = xyz[keep], col[keep], vid[keep]
    return xyz, col, vid


def save_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Binary little-endian PLY with float32 positions and uint8 colours."""
    xyz = np.asarray(xyz, np.float32)
    rgb = np.asarray(rgb, np.uint8)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def _camera_glyph(c2w: np.ndarray, half_angle: float, size: float):
    """A wireframe-free solid glyph for any camera: a pyramid for fields of view below 150 degrees,
    a cone-capped sphere otherwise. Returns (vertices, faces) in world coordinates."""
    import trimesh
    if half_angle < math.radians(75):
        t = math.tan(half_angle)
        apex = np.zeros(3)
        base = np.array([[-t, -t, 1], [t, -t, 1], [t, t, 1], [-t, t, 1]]) * size
        v = np.vstack([apex, base])
        f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
        mesh = trimesh.Trimesh(v, np.vstack([f, f[:, ::-1]]), process=False)
    else:
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.35 * size)
        axis = trimesh.creation.cone(radius=0.18 * size, height=0.9 * size)
        mesh = trimesh.util.concatenate([mesh, axis])
    mesh.apply_transform(c2w)
    return mesh


def save_glb(path: str, xyz: np.ndarray, rgb: np.ndarray, poses: Optional[np.ndarray] = None,
             cameras=None, camera_scale: float = 0.03):
    """GLB scene with the point cloud and one glyph per camera (requires trimesh). Points are
    converted from the OpenCV camera frame (y down, z forward) to glTF's (y up, z backward)."""
    import trimesh
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(np.asarray(xyz) * flip[:3, :3].diagonal(),
                                          colors=np.concatenate([rgb, np.full((len(rgb), 1), 255, np.uint8)], 1)))
    if poses is not None and len(xyz):
        extent = float(np.linalg.norm(np.percentile(xyz, 95, 0) - np.percentile(xyz, 5, 0)))
        size = camera_scale * max(extent, 1e-6)
        palette = _palette(len(poses))
        for i, c2w in enumerate(np.asarray(poses)):
            ha = getattr(cameras[i], "theta_max", None) if cameras is not None else None
            ha = math.radians(45) if ha is None else min(float(ha), math.pi)
            g = _camera_glyph(flip @ c2w, ha, size)
            g.visual.face_colors = np.array([*palette[i], 255], np.uint8)
            scene.add_geometry(g)
    scene.export(path)


def _palette(n):
    import colorsys
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i / max(n, 1)) % 1.0, 0.75, 0.95)) for i in range(n)]


def save_predictions(out_dir: str, pred: dict, views: Views, conf_percentile: float = 10.0,
                     filter_black: bool = True, min_views: int = 0, glb: bool = True,
                     max_points: Optional[int] = None) -> dict:
    """Write points.ply, scene.glb (if trimesh is installed), cameras.json and depth.npz to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    xyz, rgb, _ = scene_points(pred, views, conf_percentile, max_points, filter_black, min_views)
    files = {"points": os.path.join(out_dir, "points.ply")}
    save_ply(files["points"], xyz, rgb)
    poses = _np(pred["camera_poses"])
    if glb:
        try:
            files["glb"] = os.path.join(out_dir, "scene.glb")
            save_glb(files["glb"], xyz, rgb, poses, views.cameras)
        except ImportError:
            files.pop("glb")
    cams = [{"image": n, "camera_to_world": p.tolist(), "camera": c.to_dict(), "box_on_canvas": list(b)}
            for n, p, c, b in zip(views.names, poses, views.cameras, views.boxes)]
    files["cameras"] = os.path.join(out_dir, "cameras.json")
    with open(files["cameras"], "w") as f:
        json.dump({"convention": "OpenCV camera frame (x right, y down, z forward); camera_to_world maps it "
                                 "to the frame of the first camera; scale is arbitrary",
                   "views": cams}, f, indent=1)
    files["depth"] = os.path.join(out_dir, "depth.npz")
    depth, conf = _np(pred["depth"])[..., 0], _np(pred["depth_conf"])
    np.savez_compressed(files["depth"], **{f"depth_{i:03d}": views.crop(depth, i) for i in range(len(views.names))},
                        **{f"conf_{i:03d}": views.crop(conf, i) for i in range(len(views.names))},
                        names=np.array(views.names))
    return files
