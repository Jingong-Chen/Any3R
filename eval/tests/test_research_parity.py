# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Bitwise parity with the code that produced the paper's benchmark numbers (maintainers only).

Skipped unless ANY3R_RESEARCH_ROOT points at the research tree. The research functions are taken
from their source files verbatim (by name, without importing the heavy modules around them) and
compared with the release on synthetic panoramas and clouds: camera rays and projections of all
13 models at both sensor sizes, the evaluation render, the benchmark round trip, and the metric
pipeline (fusion, Sim(3) placement, eval_pointcloud, pose errors)."""
import ast
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
from benchlib import protocol as P                                      # noqa: E402
from benchlib import render as R                                        # noqa: E402

ROOT = os.environ.get("ANY3R_RESEARCH_ROOT")
pytestmark = pytest.mark.skipif(not ROOT or not os.path.isdir(str(ROOT)), reason="ANY3R_RESEARCH_ROOT not set")


def _functions(path, names, ns):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), path, "exec"), ns)
        elif isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def research():
    sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "scripts", "trialign"),
                    os.path.join(ROOT, "scripts", "anycam"), os.path.join(ROOT, "src", "panovggt_modified")]
    import anycam.rays as rays
    import hetero_cameras as hc
    ns = {k: getattr(rays, k) for k in dir(rays) if not k.startswith("_")}
    ns.update(math=math, np=np, bearing_to_uv=hc.bearing_to_uv, sample_erp_np=hc.sample_erp_np,
              sample_image_np=hc.sample_image_np,
              EH=P.ERP_H, EW=P.ERP_W, DIR=hc.erp_directions(P.ERP_H, P.ERP_W))
    brc = os.path.join(ROOT, "scripts", "trialign", "bench_render_camtype.py")
    _functions(brc, ("FOV", "make_camera", "render_native", "erp_from_native"), ns)
    bench = {}
    bench.update(ns)
    _functions(os.path.join(ROOT, "scripts", "anycam", "bench_image_in.py"), ("render_native", "w2c_apply"), bench)
    import pred_fusion as PF
    from evaluation.utils import pointcloud as rpc
    from evaluation.utils import pose as rpose
    return dict(brc=ns, bench=bench, PF=PF, pc=rpc, pose=rpose)


def _random_pano(seed=0):
    rs = np.random.RandomState(seed)
    rgb = rs.uniform(0, 1, (P.ERP_H, P.ERP_W, 3)).astype(np.float32)
    rng = rs.uniform(0.0, 8.0, (P.ERP_H, P.ERP_W)).astype(np.float32)
    msk = rs.uniform(0, 1, (P.ERP_H, P.ERP_W)) > 0.1
    return rgb, rng, msk


@pytest.mark.parametrize("kind", P.CAMERA_KINDS)
def test_render_and_roundtrip(research, kind):
    ns, bench = research["brc"], research["bench"]
    assert ns["FOV"][kind] == P.CAMERA_FOV[kind]
    rgb, rng, msk = _random_pano(P.CAMERA_KINDS.index(kind))
    # evaluation render at the native size
    W, H = P.native_size(kind)
    rc, nc = ns["make_camera"](kind, W, H, P.CAMERA_FOV[kind]), R.make_camera(kind, W, H)
    a = bench["render_native"](rc, W, H, rgb, rng, msk)
    b = R.render_view(nc, W, H, rgb, rng, msk)
    for x, y in zip(a, b):
        assert x.dtype == y.dtype and np.array_equal(x, y)
    # benchmark round trip on the round-trip sensor
    W, H = P.roundtrip_sensor(kind)
    rc, nc = ns["make_camera"](kind, W, H, P.CAMERA_FOV[kind]), R.make_camera(kind, W, H)
    n_rgb, n_m = ns["render_native"](rc, W, H, rgb, rng, msk)
    e1, s1 = ns["erp_from_native"](rc, W, H, n_rgb, n_m, msk)
    e2, s2 = R.roundtrip_view(nc, W, H, rgb, rng, msk)
    assert np.array_equal(e1, e2) and np.array_equal(s1, s2)


def test_metric_pipeline(research):
    """The research pp1 'pred' path on a synthetic scene equals benchlib.metrics.score_scene."""
    PF, rpc, rpose = research["PF"], research["pc"], research["pose"]
    import torch
    from test_metrics import synthetic_scene
    rays, depth, masks, c2w, w2c, gt = synthetic_scene("kb4", 160, 160, seed=3)
    rs = np.random.RandomState(0)
    depth = (depth * (1 + 0.02 * rs.standard_normal(depth.shape))).astype(np.float32) * 0.4
    c2w = c2w.copy()
    c2w[:, :3, 3] *= 0.4
    c2w[1, :3, 3] += 0.03
    c2w = c2w.astype(np.float32)
    got = M.score_scene(depth, c2w, rays, masks, gt, w2c, "PanoCity")
    # research: bench_image_in.pp1_points (ours) + pp1_offline_eval._eval_scene (pred) + pose_errors
    pts = rays * depth[..., None]
    pts_m = np.concatenate([pts[k].reshape(-1, 3)[masks[k].reshape(-1)] for k in range(3)], 0)
    vid = np.concatenate([np.full(int(m.sum()), k) for k, m in enumerate(masks)])
    c = c2w.astype(np.float64)
    own = PF.fuse(pts_m, vid, c)
    world = PF.apply(PF.align_points(own, gt, 200_000, 0), own)
    L = len(gt)
    idx = np.random.RandomState(0).choice(L, min(40000, L), replace=False)
    want = rpc.eval_pointcloud(np.asarray(world, np.float64)[idx], np.asarray(gt, np.float64)[idx],
                               icp_threshold=1.0, normal_radius=2.0)
    for k, v in want.items():
        assert got[k] == float(v), k
    g = np.repeat(np.eye(4)[None], 3, 0)
    g[:, :3, :4] = np.asarray(w2c, np.float64)
    from evaluation.utils.geometry import closed_form_inverse_se3
    r, t = rpose.se3_to_relative_pose_error(closed_form_inverse_se3(torch.from_numpy(c)), torch.from_numpy(g), 3)
    assert got["pose_r"] == [float(x) for x in r.numpy()] and got["pose_t"] == [float(x) for x in t.numpy()]
    assert PO.calculate_auc_np(np.array(got["pose_r"]), np.array(got["pose_t"]), 15) == \
        rpose.calculate_auc_np(r.numpy(), t.numpy(), 15)
    assert PC.umeyama is not rpc.umeyama
