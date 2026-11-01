# Copyright (c) 2026 The Any3R Authors.
# Adapted from PanoVGGT, evaluation/utils/pose.py and evaluation/utils/geometry.py
# (https://github.com/YijingGuo-June/PanoVGGT), which follow VGGSfM's relative-pose evaluation.
# PanoVGGT is distributed under the MIT License:
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

"""Relative camera-pose metrics of PanoVGGT's evaluation: per view pair, the angle of the relative
rotation error and of the relative translation direction error (sign ambiguity resolved); AUC@tau
of max(rotation, translation) errors in 1-degree bins. Pairs whose ground-truth baseline is at most
1 cm are left out. torch float64 on the CPU, as in PanoVGGT."""
import numpy as np
import torch


def mat_to_quat(rot):
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z), Shepperd's method."""
    batch = rot.shape[0]
    trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
    quat = torch.zeros(batch, 4, dtype=rot.dtype, device=rot.device)
    s = torch.sqrt((trace + 1.0).clamp(min=1e-10)) * 2
    quat[:, 0] = 0.25 * s
    quat[:, 1] = (rot[:, 2, 1] - rot[:, 1, 2]) / s
    quat[:, 2] = (rot[:, 0, 2] - rot[:, 2, 0]) / s
    quat[:, 3] = (rot[:, 1, 0] - rot[:, 0, 1]) / s
    for i in range(batch):
        if trace[i] > 0:
            continue
        R = rot[i]
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s_ = 2.0 * torch.sqrt((1.0 + R[0, 0] - R[1, 1] - R[2, 2]).clamp(min=1e-10))
            quat[i, 0] = (R[2, 1] - R[1, 2]) / s_
            quat[i, 1] = 0.25 * s_
            quat[i, 2] = (R[0, 1] + R[1, 0]) / s_
            quat[i, 3] = (R[0, 2] + R[2, 0]) / s_
        elif R[1, 1] > R[2, 2]:
            s_ = 2.0 * torch.sqrt((1.0 + R[1, 1] - R[0, 0] - R[2, 2]).clamp(min=1e-10))
            quat[i, 0] = (R[0, 2] - R[2, 0]) / s_
            quat[i, 1] = (R[0, 1] + R[1, 0]) / s_
            quat[i, 2] = 0.25 * s_
            quat[i, 3] = (R[1, 2] + R[2, 1]) / s_
        else:
            s_ = 2.0 * torch.sqrt((1.0 + R[2, 2] - R[0, 0] - R[1, 1]).clamp(min=1e-10))
            quat[i, 0] = (R[1, 0] - R[0, 1]) / s_
            quat[i, 1] = (R[0, 2] + R[2, 0]) / s_
            quat[i, 2] = (R[1, 2] + R[2, 1]) / s_
            quat[i, 3] = 0.25 * s_
    return quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-10)


def closed_form_inverse_se3(se3):
    """(N, 4, 4) or (N, 3, 4) rigid transforms -> (N, 4, 4) inverses [R^T | -R^T t]."""
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]
    R_t = R.transpose(1, 2)
    t_inv = -torch.bmm(R_t, T)
    inv = torch.eye(4, dtype=se3.dtype, device=se3.device).unsqueeze(0).expand(len(R), -1, -1).clone()
    inv[:, :3, :3] = R_t
    inv[:, :3, 3:] = t_inv
    return inv


def rotation_angle(rot_gt, rot_pred, eps=1e-15):
    """Geodesic angle (degrees) between rotation matrices (P, 3, 3)."""
    dot = (mat_to_quat(rot_pred) * mat_to_quat(rot_gt)).sum(dim=1)
    loss = (1.0 - dot ** 2).clamp(min=eps)
    return torch.arccos(1.0 - 2.0 * loss) * (180.0 / np.pi)


def translation_angle(t_gt, t_pred, eps=1e-15, default_err=1e6):
    """Angle (degrees) between translation directions (P, 3), min(err, 180 - err)."""
    t_gt = t_gt / (t_gt.norm(dim=1, keepdim=True) + eps)
    t_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + eps)
    dot2 = (t_gt * t_pred).sum(dim=1) ** 2
    err = torch.arccos(torch.sqrt(1.0 - (1.0 - dot2).clamp(min=eps)))
    err[torch.isnan(err) | torch.isinf(err)] = default_err
    deg = err * (180.0 / np.pi)
    return torch.min(deg, (180.0 - deg).abs())


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """Area under the curve of max(r_error, t_error) up to max_threshold degrees, 1-degree bins."""
    errs = np.maximum(r_error, t_error)
    hist, _ = np.histogram(errs, bins=np.arange(max_threshold + 1))
    norm = hist.astype(float) / max(len(errs), 1)
    return float(np.mean(np.cumsum(norm)))


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames, trans_norm_thresh=1e-2):
    """Rotation and translation-direction errors (degrees) of every view pair (i < j) whose
    ground-truth translation exceeds trans_norm_thresh. Poses: (N, 4, 4), both world-to-camera or
    both camera-to-world."""
    i1, i2 = torch.combinations(torch.arange(num_frames), 2, with_replacement=False).unbind(-1)
    rel_gt = gt_se3[i2].bmm(closed_form_inverse_se3(gt_se3[i1]))
    rel_pred = pred_se3[i2].bmm(closed_form_inverse_se3(pred_se3[i1]))
    r_err = rotation_angle(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_gt = rel_gt[:, :3, 3]
    valid = t_gt.norm(p=2, dim=-1) > trans_norm_thresh
    if valid.sum() == 0:
        empty = torch.empty(0, dtype=r_err.dtype, device=r_err.device)
        return empty, empty.clone()
    t_err = translation_angle(t_gt[valid], rel_pred[:, :3, 3][valid])
    return r_err[valid], t_err


def pose_errors(c2w, gt_w2c, trans_norm_thresh=1e-2):
    """Per view pair (i < j) errors in degrees, as lists: predicted camera-to-world (N, 4, 4) against
    ground-truth world-to-camera (N, 3, 4) or (N, 4, 4)."""
    N = len(c2w)
    g = np.repeat(np.eye(4)[None], N, 0)
    g[:, :3, :4] = np.asarray(gt_w2c, np.float64)[:, :3, :4]
    pred_w2c = closed_form_inverse_se3(torch.from_numpy(np.asarray(c2w, np.float64)))
    r, t = se3_to_relative_pose_error(pred_w2c, torch.from_numpy(g), N, trans_norm_thresh)
    return [float(x) for x in r.numpy()], [float(x) for x in t.numpy()]
