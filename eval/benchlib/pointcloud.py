# Copyright (c) 2026 The Any3R Authors.
# Adapted from PanoVGGT, evaluation/utils/pointcloud.py (https://github.com/YijingGuo-June/PanoVGGT),
# which follows CUT3R's multi-view reconstruction evaluation. PanoVGGT is distributed under the MIT
# License:
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
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Point-cloud metrics of PanoVGGT's evaluation, unchanged in substance: Umeyama Sim(3), rigid
point-to-point ICP (Open3D), then accuracy / completion / normal consistency.

The arithmetic follows PanoVGGT's code statement by statement, so the benchmark's numbers are
reproduced to the last bit on the same libraries. Open3D's ICP and normal estimation are
multi-threaded; run with OMP_NUM_THREADS=1 (evaluate.py does) for bitwise reproducible results.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

KD_WORKERS = -1          # threads of the nearest-neighbour queries (results do not depend on it)


def umeyama(X, Y):
    """(c, R, t) with c R X + t ~ Y. X, Y: (d, n), same indexing. t is (d, 1)."""
    mu_x = X.mean(axis=1, keepdims=True)
    mu_y = Y.mean(axis=1, keepdims=True)
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(X.shape[0])
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ Vt
    t = mu_y - c * R @ mu_x
    return c, R, t


def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Reconstruction -> ground truth distances: (mean, median, nc mean, nc median, mae, rmse)."""
    dists, idx = cKDTree(gt_points).query(rec_points, workers=KD_WORKERS)
    acc_mean = float(np.mean(dists))
    acc_med = float(np.median(dists))
    rmse = float(np.sqrt(np.mean(dists ** 2)))
    if gt_normals is not None and rec_normals is not None:
        dots = np.abs(np.sum(gt_normals[idx] * rec_normals, axis=-1))
        return acc_mean, acc_med, float(np.mean(dots)), float(np.median(dots)), acc_mean, rmse
    return acc_mean, acc_med, None, None, acc_mean, rmse


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    """Ground truth -> reconstruction distances: (mean, median, nc mean, nc median)."""
    dists, idx = cKDTree(rec_points).query(gt_points, workers=KD_WORKERS)
    comp_mean = float(np.mean(dists))
    comp_med = float(np.median(dists))
    if gt_normals is not None and rec_normals is not None:
        dots = np.abs(np.sum(gt_normals * rec_normals[idx], axis=-1))
        return comp_mean, comp_med, float(np.mean(dots)), float(np.median(dots))
    return comp_mean, comp_med, None, None


METRIC_KEYS = ("Acc-mean", "Acc-med", "Comp-mean", "Comp-med", "overall_mean", "overall_med",
               "NC1-mean", "NC1-med", "NC2-mean", "NC2-med", "MAE", "RMSE")


def eval_pointcloud(pred_pts, gt_pts, icp_threshold=0.1, normal_radius=0.2):
    """Metrics of a fused point cloud against ground truth with one-to-one correspondence.

    pred_pts, gt_pts: (N, 3) float64, point i of one matching point i of the other. Umeyama Sim(3)
    on the correspondences, point-to-point ICP from there (max correspondence distance
    icp_threshold), normals estimated within normal_radius, then accuracy (prediction -> ground
    truth), completion (ground truth -> prediction) and normal consistency (NC1 / NC2)."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return {k: 0.0 for k in METRIC_KEYS}
    c, R, t = umeyama(pred_pts.T, gt_pts.T)
    pred_aligned = c * (R @ pred_pts.T).T + t.T
    pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_aligned))
    pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))
    reg = o3d.pipelines.registration.registration_icp(
        pcd_pred, pcd_gt, icp_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pcd_pred.transform(reg.transformation)
    search = o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30)
    pcd_pred.estimate_normals(search_param=search)
    pcd_gt.estimate_normals(search_param=search)
    pred_n, gt_n = np.asarray(pcd_pred.normals), np.asarray(pcd_gt.normals)
    pred_p, gt_p = np.asarray(pcd_pred.points), np.asarray(pcd_gt.points)
    acc, acc_med, nc1, nc1_med, mae, rmse = accuracy(gt_p, pred_p, gt_n, pred_n)
    comp, comp_med, nc2, nc2_med = completion(gt_p, pred_p, gt_n, pred_n)
    return {"Acc-mean": acc, "Acc-med": acc_med, "Comp-mean": comp, "Comp-med": comp_med,
            "overall_mean": (acc + comp) / 2, "overall_med": (acc_med + comp_med) / 2,
            "NC1-mean": nc1 or 0.0, "NC1-med": nc1_med or 0.0, "NC2-mean": nc2 or 0.0, "NC2-med": nc2_med or 0.0,
            "MAE": mae, "RMSE": rmse}
