# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Scoring of one benchmark scene from a method's own predictions.

A method outputs per-view camera-frame points p_k (its own scale) and camera-to-world poses
T_k = [R_k | c_k] from the same forward pass. Its fused cloud is X = R_k p_k + c_k. One similarity
(Umeyama on the pixel correspondences with the ground truth) places X in the ground-truth frame,
and PanoVGGT's eval_pointcloud scores it (its own Umeyama absorbs that placement; the placement only
fixes the frame the ICP starts from). Nothing else of the ground truth is used: no ground-truth
poses, no per-view scale.
"""
import statistics

import numpy as np

from . import protocol as P
from .pointcloud import eval_pointcloud
from .pose import calculate_auc_np, pose_errors


def as_c2w44(c2w):
    """(N, 3, 4) or (N, 4, 4) camera-to-world -> (N, 4, 4) float64, after checking that the rotations
    are rotations (bfloat16 heads leave ~1e-3 of non-orthogonality; larger is a wrong matrix)."""
    c = np.asarray(c2w, np.float64)
    if c.shape[-2:] == (3, 4):
        q = np.repeat(np.eye(4)[None], len(c), 0)
        q[:, :3, :4] = c
        c = q
    if not np.isfinite(c).all():
        raise ValueError("predicted poses are not finite")
    R = c[:, :3, :3]
    if np.abs(R @ R.transpose(0, 2, 1) - np.eye(3)).max() > 5e-2 or np.abs(np.linalg.det(R) - 1.0).max() > 5e-2:
        raise ValueError("predicted pose rotations are not rotations")
    return c


def w2c_apply(cam_points, w2c):
    """Camera-frame points of a view -> world, given its world-to-camera (3, 4): (x - t) R."""
    R, t = w2c[:3, :3], w2c[:3, 3]
    return (cam_points - t) @ R


def gt_world_points(rays, ranges, masks, ext_w2c):
    """Ground-truth cloud: ray x range of every scored pixel, view by view, in the world frame
    (float32, the order of the scored pixels of view 0, then view 1, ...)."""
    return np.concatenate([w2c_apply((np.asarray(rays[k]) * ranges[k][..., None]).reshape(-1, 3)
                                     [masks[k].reshape(-1)], ext_w2c[k]) for k in range(len(masks))], 0)


def masked_points(points, masks):
    """(N, H, W, 3) -> the scored pixels of every view, concatenated in view order."""
    return np.concatenate([points[k].reshape(-1, 3)[masks[k].reshape(-1)] for k in range(len(masks))], 0)


def fuse(cam_points, view_ids, c2w):
    """X = R_v p + c_v, float64."""
    p = np.asarray(cam_points, np.float64)
    v = np.asarray(view_ids, np.int64)
    return np.einsum("mij,mj->mi", c2w[v, :3, :3], p) + c2w[v, :3, 3]


def umeyama_sim3(src, dst):
    """(s, R, t) with dst ~ s R src + t (least squares, no reflection); src, dst (N, 3)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("Umeyama input is not finite")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    var_s = (S ** 2).sum() / len(src)
    if var_s <= 1e-24:
        raise ValueError("Umeyama source has zero spread")
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1.0
    R = U @ E @ Vt
    s = float((sig * np.diag(E)).sum() / var_s)
    return s, R, mu_d - s * R @ mu_s


def place(own, gt_world, max_fit=P.FIT_MAX_POINTS, seed=0):
    """The one similarity from the method's frame to the ground-truth frame, fitted on a fixed
    RandomState(seed) subset of max_fit correspondences, applied to the whole cloud."""
    own = np.asarray(own, np.float64)
    idx = np.arange(len(own))
    if len(own) > max_fit:
        idx = np.sort(np.random.RandomState(seed).choice(len(own), max_fit, replace=False))
    s, R, t = umeyama_sim3(own[idx], np.asarray(gt_world, np.float64)[idx])
    return s * (own @ np.asarray(R).T) + t, (s, R, t)


def eval_subset(n, max_points=P.MAX_EVAL_POINTS):
    """Indices of the fixed RandomState(0) subset of correspondences handed to eval_pointcloud."""
    if max_points <= 0:
        return None
    return np.random.RandomState(0).choice(n, min(max_points, n), replace=False)


def score_scene(depth, c2w, rays, masks, gt_world, ext_w2c, ds):
    """Metrics of one scene.

    depth (N, H, W) float32: predicted range along each ray; c2w (N, 4, 4): predicted
    camera-to-world, same forward pass; rays (N, H, W, 3) float32; masks (N, H, W) bool; gt_world:
    gt_world_points(...); ext_w2c (N, 3, 4) float32 ground-truth world-to-camera (pose metrics
    only). Returns {metrics of eval_pointcloud, 'pose_r', 'pose_t' (degrees per view pair)}."""
    pts = np.asarray(rays) * np.asarray(depth, np.float32)[..., None]
    pts_m = masked_points(pts, masks)
    if not np.isfinite(pts_m).all():
        raise ValueError("non-finite predicted points")
    view_ids = np.concatenate([np.full(int(m.sum()), k) for k, m in enumerate(masks)])
    c44 = as_c2w44(c2w)
    world, _sRt = place(fuse(pts_m, view_ids, c44), gt_world)
    L = len(gt_world)
    if L < 200 or len(world) != L:
        raise ValueError(f"cloud of {len(world)} points vs ground truth {L} (need equal and >= 200)")
    p = np.asarray(world, np.float64)
    g = np.asarray(gt_world, np.float64)
    idx = eval_subset(L)
    if idx is not None:
        p, g = p[idx], g[idx]
    out = {k: float(v) for k, v in eval_pointcloud(p, g, **P.ICP_CONFIG[ds]).items()}
    out["pose_r"], out["pose_t"] = pose_errors(c44, ext_w2c, P.POSE_TRANS_NORM_THRESH)
    return out


def cell_summary(per_scene):
    """Cell statistics from the per-scene records {scene: score_scene(...)}: means and medians over
    scenes of Acc / Comp / NC1, and the AUC over all view pairs of the cell."""
    names = list(per_scene)
    out = {"n_scenes": len(names)}
    for key in ("Acc-mean", "Comp-mean", "NC1-mean"):
        v = [float(per_scene[s][key]) for s in names]
        out[key.split("-")[0]] = statistics.fmean(v)
        out[key.split("-")[0] + "-median"] = statistics.median(v)
    r = np.array([x for s in names for x in per_scene[s]["pose_r"]], np.float64)
    t = np.array([x for s in names for x in per_scene[s]["pose_t"]], np.float64)
    out["n_pairs"] = int(len(r))
    for tau in (5, 15, 30):
        out[f"AUC@{tau}"] = calculate_auc_np(r, t, tau) if len(r) else 0.0
    out["R_med"] = float(np.median(r)) if len(r) else float("nan")
    out["T_med"] = float(np.median(t)) if len(t) else float("nan")
    return out
