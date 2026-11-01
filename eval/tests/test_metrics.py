# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Metric checks on synthetic data: the point-cloud metrics, the pose metrics and the scoring of a
scene must give the exact answers on exact predictions, be invariant to the similarity gauge of the
prediction, and respond to known errors with the known amount."""
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]

from benchlib import metrics as M                                       # noqa: E402
from benchlib import pointcloud as PC                                   # noqa: E402
from benchlib import pose as PO                                         # noqa: E402
from benchlib.render import make_camera                                 # noqa: E402


def rot(axis, deg):
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def box_cloud(n=6000, seed=0):
    """Points on the faces of a 4 x 3 x 2.5 m box (planar surfaces, well-defined normals)."""
    rs = np.random.RandomState(seed)
    size = np.array([4.0, 3.0, 2.5])
    pts = rs.uniform(-0.5, 0.5, (n, 3)) * size
    face = rs.randint(0, 6, n)
    ax, sign = face % 3, np.where(face < 3, -0.5, 0.5)
    pts[np.arange(n), ax] = sign * size[ax]
    return pts


def test_umeyama_recovers_similarity():
    X = box_cloud(500)
    R, s, t = rot([1, 2, 3], 37.0), 2.5, np.array([0.3, -1.0, 4.0])
    Y = s * X @ R.T + t
    c, R2, t2 = PC.umeyama(X.T, Y.T)
    assert abs(c - s) < 1e-12 and np.abs(R2 - R).max() < 1e-12 and np.abs(t2[:, 0] - t).max() < 1e-12
    s3, R3, t3 = M.umeyama_sim3(X, Y)
    assert abs(s3 - s) < 1e-12 and np.abs(R3 - R).max() < 1e-12 and np.abs(t3 - t).max() < 1e-12


@pytest.mark.parametrize("icp", [0.1, 1.0])
def test_eval_pointcloud_exact_prediction(icp):
    gt = box_cloud()
    R, s, t = rot([0, 1, 0], 25.0), 0.37, np.array([5.0, 1.0, -2.0])
    pred = s * gt @ R.T + t                       # same cloud in another similarity gauge
    m = PC.eval_pointcloud(pred, gt, icp_threshold=icp, normal_radius=2 * icp)
    assert m["Acc-mean"] < 1e-9 and m["Comp-mean"] < 1e-9
    assert m["NC1-mean"] > 0.999 and m["NC2-mean"] > 0.999


def test_eval_pointcloud_known_offset():
    """Half the points pushed off the surface by 5 cm: the median accuracy stays ~0, the mean is ~2.5 cm."""
    gt = box_cloud(20000, seed=1)
    pred = gt.copy()
    rs = np.random.RandomState(2)
    idx = rs.choice(len(gt), len(gt) // 2, replace=False)
    pred[idx] += 0.05 * rs.standard_normal((len(idx), 3)) / np.sqrt(3)
    m = PC.eval_pointcloud(pred, gt)
    assert m["Acc-med"] < 0.005
    assert 0.005 < m["Acc-mean"] < 0.05


def test_pose_errors_exact_and_known():
    rs = np.random.RandomState(0)
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    for k in range(3):
        c2w[k, :3, :3] = rot(rs.randn(3), rs.uniform(0, 90))
        c2w[k, :3, 3] = rs.randn(3)
    w2c = np.linalg.inv(c2w)[:, :3, :4]
    r, t = PO.pose_errors(c2w, w2c)
    assert len(r) == 3 and max(r) < 1e-5 and max(t) < 1e-5
    # the same poses in another similarity gauge: relative errors unchanged
    G = np.eye(4)
    G[:3, :3], G[:3, 3] = 3.0 * rot([1, 1, 0], 50.0), [1.0, 2.0, 3.0]
    c2w_g = G @ c2w
    c2w_g[:, :3, :3] /= 3.0
    c2w_g[:, :3, :3] = np.stack([np.linalg.qr(R)[0] * np.sign(np.diag(np.linalg.qr(R)[1])) for R in c2w_g[:, :3, :3]])
    r, t = PO.pose_errors(c2w_g, w2c)
    assert max(r) < 1e-4 and max(t) < 1e-4
    # view 2 turned by 10 degrees in its own frame: pairs (0,2) and (1,2) err by 10 degrees in rotation
    bad = c2w.copy()
    bad[2, :3, :3] = bad[2, :3, :3] @ rot([0, 1, 0], 10.0)
    r, _t = PO.pose_errors(bad, w2c)
    assert abs(r[0]) < 1e-5 and abs(r[1] - 10.0) < 1e-5 and abs(r[2] - 10.0) < 1e-5


def test_pose_baseline_filter_and_auc():
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    c2w[1, :3, 3] = [0.005, 0, 0]                 # 5 mm from view 0: pair (0,1) is dropped
    c2w[2, :3, 3] = [1.0, 0, 0]
    r, t = PO.pose_errors(c2w, np.linalg.inv(c2w)[:, :3, :4])
    assert len(r) == len(t) == 2
    assert PO.calculate_auc_np(np.zeros(4), np.zeros(4), 30) == 1.0
    # errors of 20 degrees count from the 21st bin on: AUC@30 = 10 / 30
    assert abs(PO.calculate_auc_np(np.full(4, 20.0), np.zeros(4), 30) - 10 / 30) < 1e-12
    assert PO.calculate_auc_np(np.full(4, 40.0), np.zeros(4), 30) == 0.0


def synthetic_scene(kind="equidistant", W=96, H=96, seed=0):
    """Three views inside a box room: exact rays and ranges, ground-truth poses."""
    cam = make_camera(kind, W, H)
    rays, valid = cam.ray_map(H, W)
    rs = np.random.RandomState(seed)
    half = np.array([2.0, 1.25, 1.5])
    c2w = np.repeat(np.eye(4)[None], 3, 0)
    depth, masks = [], []
    for k in range(3):
        c2w[k, :3, :3] = rot([0, 1, 0], 120.0 * k + rs.uniform(-10, 10))
        c2w[k, :3, 3] = rs.uniform(-0.5, 0.5, 3)
        d_w = rays.reshape(-1, 3).astype(np.float64) @ c2w[k, :3, :3].T
        o = c2w[k, :3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(d_w > 0, (half - o) / d_w, (-half - o) / d_w)
        rng = np.nanmin(np.where(tt > 0, tt, np.inf), axis=1).reshape(H, W)
        depth.append(rng.astype(np.float32))
        masks.append(valid & np.isfinite(rng) & (rng > 0.05))
    depth, masks = np.stack(depth), np.stack(masks)
    w2c = np.linalg.inv(c2w)[:, :3, :4].astype(np.float32)
    rays3 = np.stack([rays] * 3)
    gt = M.gt_world_points(rays3, depth, masks, w2c)
    return rays3, depth, masks, c2w, w2c, gt


def test_score_scene_exact_and_gauge_invariant():
    rays, depth, masks, c2w, w2c, gt = synthetic_scene()
    m = M.score_scene(depth, c2w, rays, masks, gt, w2c, "M3D")
    assert m["Acc-mean"] < 1e-5 and m["Comp-mean"] < 1e-5 and max(m["pose_r"]) < 1e-3
    # the prediction in an arbitrary similarity gauge (depth and translations scaled together)
    G = rot([1, 0, 1], 33.0)
    c2w_g = c2w.copy()
    c2w_g[:, :3, :3] = G @ c2w[:, :3, :3]
    c2w_g[:, :3, 3] = 0.25 * (c2w[:, :3, 3] @ G.T) + 7.0
    m2 = M.score_scene(depth * 0.25, c2w_g, rays, masks, gt, w2c, "M3D")
    assert m2["Acc-mean"] < 1e-5 and max(m2["pose_r"]) < 1e-3


def test_score_scene_detects_pose_error():
    rays, depth, masks, c2w, w2c, gt = synthetic_scene(seed=1)
    bad = c2w.copy()
    bad[1, :3, :3] = bad[1, :3, :3] @ rot([0, 1, 0], 5.0)
    good = M.score_scene(depth, c2w, rays, masks, gt, w2c, "M3D")
    wrong = M.score_scene(depth, bad, rays, masks, gt, w2c, "M3D")
    assert wrong["Acc-mean"] > good["Acc-mean"] + 0.01
    assert max(wrong["pose_r"]) > 4.9


def test_cell_summary():
    per = {f"s{i}": {"Acc-mean": a, "Comp-mean": 2 * a, "NC1-mean": 0.9, "pose_r": [1.0, 2.0, 20.0],
                     "pose_t": [0.5, 0.5, 0.5]} for i, a in enumerate((0.1, 0.2, 0.6))}
    s = M.cell_summary(per)
    assert abs(s["Acc"] - 0.3) < 1e-15 and s["Acc-median"] == 0.2 and s["n_pairs"] == 9
    assert abs(s["AUC@15"] - PO.calculate_auc_np(np.array([1.0, 2.0, 20.0] * 3), np.full(9, 0.5), 15)) < 1e-15
