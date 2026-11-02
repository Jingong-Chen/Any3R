# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Evaluate Any3R on the any-camera benchmark (PP13).

    python eval/evaluate.py --benchmark data/any3r_bench --out results/any3r
    python eval/evaluate.py --benchmark data/any3r_bench --out results/any3r --datasets S2D3DS --cameras pinhole,erp
    python eval/evaluate.py --out results/any3r --verify            # compare finished results with the paper

For every cell (dataset x camera model) and scene: the three views are rendered natively through the
camera model from the scene file, Any3R reconstructs them from the images and their ray maps, and
the fused cloud (Any3R's own depths and poses) is scored with PanoVGGT's eval_pointcloud; pose
errors come from the predicted relative poses. Results go to {out}/{DS}/{camera}/scene_XXX.json
(one per scene; finished scenes are skipped on a rerun) and {out}/{DS}/{camera}.json (the cell).

--save-predictions keeps Any3R's depth and poses per scene ({out}/predictions/{DS}/{camera}/scene_XXX.npz);
--predictions DIR scores stored predictions of that layout instead of running the model.
--verify compares every finished scene with the paper's values (benchmark/paper_any3r.json).
"""
import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time

# Open3D's ICP and normal estimation are multi-threaded; one thread makes every number reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                                       # noqa: E402

from benchlib import benchmark as B                                     # noqa: E402
from benchlib import metrics as M                                       # noqa: E402
from benchlib import protocol as P                                      # noqa: E402
from benchlib.render import make_camera, render_view                    # noqa: E402

DEFAULT_REFERENCE = os.path.join(HERE, "benchmark", "paper_any3r.json")
METRIC_KEYS = ("Acc-mean", "Acc-med", "Comp-mean", "Comp-med", "NC1-mean", "NC1-med", "NC2-mean", "NC2-med", "RMSE")


# ----------------------------------------------------------------------------------------- scenes
def prepare_scene(task):
    """Render one scene of a cell: images (3, 3, H, W) float32, rays (3, H, W, 3) float32, masks,
    ground-truth cloud and extrinsics. Runs in a worker process."""
    root, ds, kind, scene, manifest_path, check = task
    man = _manifest(manifest_path)
    s = B.load_scene(root, ds, kind, scene, man, check=check)
    W, H = P.native_size(kind)
    cam = make_camera(kind, W, H)
    erp = s["erp_rgb"].astype(np.float32) / 255.0
    rng = s["erp_rng"].astype(np.float32)
    ext = s["ext_w2c"].astype(np.float32)
    imgs, rays, masks, rads = [], [], [], []
    for k in range(P.N_VIEWS):
        rgb, m, r, rad = render_view(cam, W, H, erp[k].transpose(1, 2, 0), rng[k], s["erp_msk"][k])
        imgs.append(rgb.transpose(2, 0, 1))
        rays.append(r)
        masks.append(m)
        rads.append(rad)
    masks = np.stack(masks)
    gt = M.gt_world_points(rays, rads, masks, ext)
    return dict(ds=ds, kind=kind, scene=scene, images=np.stack(imgs), rays=np.stack(rays), masks=masks,
                gt_world=gt, ext=ext)


def score_task(task):
    """Score one scene from its prediction. Runs in a worker process."""
    t0 = time.time()
    S, depth, c2w = task
    try:
        out = M.score_scene(depth, c2w, S["rays"], S["masks"], S["gt_world"], S["ext"], S["ds"])
    except Exception as ex:                                              # noqa: BLE001
        return dict(ds=S["ds"], kind=S["kind"], scene=S["scene"], error=f"{type(ex).__name__}: {ex}")
    out.update(ds=S["ds"], kind=S["kind"], scene=S["scene"], n_gt_points=int(len(S["gt_world"])),
               n_scored_pixels=[int(m.sum()) for m in S["masks"]], seconds_forward=S.get("seconds_forward"),
               seconds_score=round(time.time() - t0, 2))
    return out


_MAN = {}


def _manifest(path):
    if path not in _MAN:
        _MAN[path] = B.load_manifest(path)
    return _MAN[path]


# ------------------------------------------------------------------------------------------ model
class Runner:
    def __init__(self, weights, device):
        import torch
        from any3r import Any3R
        self.torch = torch
        self.model = Any3R.from_pretrained(weights).to(device).eval()
        self.device = device

    def __call__(self, images, rays):
        torch = self.torch
        # channels-last strides as in the paper's evaluation (infer() normalises any layout anyway)
        img = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 2, 3, 1))).permute(0, 3, 1, 2)
        out = self.model.infer(img, torch.from_numpy(np.ascontiguousarray(rays)))
        depth = out["depth"].squeeze(-1).cpu().numpy().astype(np.float32)
        poses = out["camera_poses"].cpu().numpy().astype(np.float32)
        return depth, poses


def env_record(device=None):
    import open3d
    import scipy
    rec = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
           "open3d": open3d.__version__, "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "host": platform.node()}
    try:
        import torch
        rec["torch"] = torch.__version__
        if device and str(device).startswith("cuda") and torch.cuda.is_available():
            rec["gpu"] = torch.cuda.get_device_name(torch.device(device))
            rec["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return rec


# ----------------------------------------------------------------------------------------- output
def scene_json(out, ds, kind, scene):
    return os.path.join(out, ds, kind, f"{scene}.json")


def pred_npz(root, ds, kind, scene):
    return os.path.join(root, ds, kind, f"{scene}.npz")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def write_cell(out, ds, kind, scenes):
    per = {}
    for sc in scenes:
        with open(scene_json(out, ds, kind, sc), encoding="utf-8") as f:
            per[sc] = json.load(f)
    envs = []
    for sc in scenes:                                  # the environments the scenes were evaluated in
        if per[sc].get("env") not in envs:
            envs.append(per[sc].get("env"))
    W, H = P.native_size(kind)
    cell = dict(ds=ds, camera=kind, fov_deg=P.CAMERA_FOV[kind], native_size=[W, H],
                summary=M.cell_summary(per), envs=envs,
                scenes={sc: {k: per[sc][k] for k in METRIC_KEYS + ("pose_r", "pose_t")} for sc in scenes})
    write_json(os.path.join(out, ds, f"{kind}.json"), cell)
    return cell


# ----------------------------------------------------------------------------------------- verify
def load_reference(path):
    """{(ds, camera): {"cell": {...}, "scenes": {scene: {...}}}} from paper_any3r.json or from the
    research aggregate (cells[...].arms.band8_ours)."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    ref = {}
    for key, c in d["cells"].items():
        ds, kind = key.split("|")
        if "arms" in c:                                   # research aggregate: Acc / Comp / NC1 and pooled AUC only
            a = c["arms"]["band8_ours"]
            names = [s.replace(".npz", "") for s in c["scenes"]]
            sc = {n: {"Acc-mean": a["per_scene"]["Acc"][i], "Comp-mean": a["per_scene"]["Comp"][i],
                      "NC1-mean": a["per_scene"]["NC1"][i]} for i, n in enumerate(names)}
            cell = {"Acc": a["Acc-mean"], "AUC@15": a["pose"]["pooled"]["AUC@15"],
                    "AUC@30": a["pose"]["pooled"]["AUC@30"]}
            ref[(ds, kind)] = {"cell": cell, "scenes": sc}
        else:
            ref[(ds, kind)] = {"cell": c["cell"], "scenes": c["scenes"]}
    return ref


def verify(out, reference, cells, tol):
    """Per cell: max |ours - paper| of every per-scene metric and pose error, and of the cell values.
    Returns (report, n_compared_scenes, n_failed_cells)."""
    ref = load_reference(reference)
    report, n_sc, n_bad = {}, 0, 0
    for ds, kind in cells:
        r = ref.get((ds, kind))
        cell_path = os.path.join(out, ds, f"{kind}.json")
        if r is None or not os.path.exists(cell_path):
            continue
        with open(cell_path, encoding="utf-8") as f:
            got = json.load(f)
        diffs = {}
        common = sorted(set(got["scenes"]) & set(r["scenes"]))
        for sc in common:
            g, w = got["scenes"][sc], r["scenes"][sc]
            for k, v in w.items():
                if k in ("pose_r", "pose_t"):
                    if len(v) != len(g[k]):
                        diffs[k] = float("inf")
                        continue
                    d = max((abs(a - b) for a, b in zip(g[k], v)), default=0.0)
                else:
                    d = abs(g[k] - v)
                diffs[k] = max(diffs.get(k, 0.0), d)
        cs = got["summary"]
        cell_d = {k: abs(cs[k] - r["cell"][k]) for k in ("Acc", "AUC@15", "AUC@30") if k in r["cell"]}
        worst = max(list(diffs.values()) + list(cell_d.values()) + [0.0])
        complete = len(common) == len(r["scenes"])
        ok = complete and worst <= tol
        n_bad += not ok
        n_sc += len(common)
        report[f"{ds}|{kind}"] = dict(n_scenes=len(common), complete=complete, max_abs_diff=diffs,
                                      cell_abs_diff=cell_d, worst=worst, ok=ok,
                                      ours={k: cs[k] for k in ("Acc", "AUC@15", "AUC@30")},
                                      paper={k: r["cell"][k] for k in ("Acc", "AUC@15", "AUC@30") if k in r["cell"]})
    return report, n_sc, n_bad


def print_verify(report, tol):
    print(f"\n{'cell':28s} {'n':>3s} {'Acc':>9s} {'Comp':>9s} {'NC1':>9s} {'pose_r':>9s} {'pose_t':>9s} "
          f"{'cellAcc':>9s} {'AUC15':>9s} {'AUC30':>9s}  ok")
    for key, v in report.items():
        d, c = v["max_abs_diff"], v["cell_abs_diff"]
        f = lambda x: f"{x:9.2e}" if x is not None else f"{'-':>9s}"          # noqa: E731
        print(f"{key:28s} {v['n_scenes']:3d} {f(d.get('Acc-mean'))} {f(d.get('Comp-mean'))} {f(d.get('NC1-mean'))} "
              f"{f(d.get('pose_r'))} {f(d.get('pose_t'))} {f(c.get('Acc'))} {f(c.get('AUC@15'))} {f(c.get('AUC@30'))}"
              f"  {'yes' if v['ok'] else 'NO'}")
    print(f"(max |diff| over the scenes of each cell; ok = all 24 scenes present and every diff <= {tol:g})")


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", help="built benchmark directory (build_benchmark.py --out)")
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--weights", default="Jingong/Any3R", help="Hugging Face repo id or local directory")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=None, help="comma list (default: M3D,S2D3DS,PanoCity)")
    ap.add_argument("--cameras", default=None,
                    help="comma list (default: all 13); 'main' for the eight of the main tables")
    ap.add_argument("--scenes", default=None, help="comma list of scene names (default: all 24)")
    ap.add_argument("--workers", type=int, default=4, help="CPU processes for rendering and scoring")
    ap.add_argument("--manifest", default=None, help="default: benchmark/manifest.json")
    ap.add_argument("--no-check", action="store_true", help="do not check the scene files against the manifest md5s")
    ap.add_argument("--predictions", default=None, help="score stored predictions from this directory")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="recompute scenes that already have a result")
    ap.add_argument("--verify", nargs="?", const=DEFAULT_REFERENCE, default=None, metavar="REFERENCE",
                    help="compare with the paper (default benchmark/paper_any3r.json; the research aggregate "
                         "pp1_data_v2.json is accepted too)")
    ap.add_argument("--tol", type=float, default=1e-6, help="--verify: largest acceptable |difference|")
    a = ap.parse_args()

    man_path = os.path.abspath(a.manifest or B.DEFAULT_MANIFEST)
    man = _manifest(man_path)
    dss = a.datasets.split(",") if a.datasets else list(P.DATASETS)
    if a.cameras == "main":
        cams = list(P.MAIN_CAMERAS)
    else:
        cams = a.cameras.split(",") if a.cameras else list(P.CAMERA_KINDS)
    bad = [d for d in dss if d not in P.DATASETS] + [c for c in cams if c not in P.CAMERA_KINDS]
    if bad:
        sys.exit(f"unknown datasets / cameras: {bad}")
    scenes_filter = set(a.scenes.split(",")) if a.scenes else None
    cells = [(d, c) for d in dss for c in cams]

    if a.benchmark:
        todo = []
        for ds, kind in cells:
            for s in man["datasets"][ds]["scenes"]:
                sc = s["scene"]
                if scenes_filter and sc not in scenes_filter:
                    continue
                if a.overwrite or not os.path.exists(scene_json(a.out, ds, kind, sc)):
                    todo.append((ds, kind, sc))
        print(f"{len(cells)} cells, {len(todo)} scenes to evaluate -> {a.out}", flush=True)
        env = env_record(None if a.predictions else a.device)
        if todo:
            ctx = mp.get_context("fork")
            pool = ctx.Pool(max(1, a.workers))                 # forked before CUDA is initialised
            runner = None
            if not a.predictions:
                runner = Runner(a.weights, a.device)
                env = env_record(a.device)
            env["weights"] = None if a.predictions else a.weights
            env["predictions"] = a.predictions
            t0 = time.time()
            # rendering runs ahead of the model by at most `ahead` scenes, and scoring is submitted to the
            # same pool as each prediction arrives, so memory stays bounded
            ahead = 2 * max(1, a.workers)
            prep_tasks = iter([(a.benchmark, ds, kind, sc, man_path, not a.no_check) for ds, kind, sc in todo])
            prepared = []

            def refill():
                while len(prepared) < ahead:
                    t = next(prep_tasks, None)
                    if t is None:
                        return
                    prepared.append(pool.apply_async(prepare_scene, (t,)))

            def prep_iter():
                refill()
                while prepared:
                    S = prepared.pop(0).get()
                    refill()
                    yield S

            prep = prep_iter()
            pending, n_done, n_err = [], 0, 0

            def collect(block):
                nonlocal n_done, n_err
                while pending and (block or pending[0].ready() or len(pending) > ahead):
                    res = pending.pop(0).get()
                    if "error" in res:
                        n_err += 1
                        print(f"  FAILED {res['ds']}/{res['kind']}/{res['scene']}: {res['error']}", flush=True)
                        continue
                    res["env"] = env
                    write_json(scene_json(a.out, res["ds"], res["kind"], res["scene"]), res)
                    n_done += 1
                    print(f"  {res['ds']}/{res['kind']}/{res['scene']}: Acc {res['Acc-mean']:.4f} "
                          f"Comp {res['Comp-mean']:.4f} NC1 {res['NC1-mean']:.4f}  "
                          f"rot {np.round(res['pose_r'], 2).tolist()}  [{n_done}/{len(todo)}, "
                          f"{time.time() - t0:.0f}s]", flush=True)

            for S in prep:
                ds, kind, sc = S["ds"], S["kind"], S["scene"]
                if a.predictions:
                    with np.load(pred_npz(a.predictions, ds, kind, sc)) as z:
                        depth, c2w = z["depth"].astype(np.float32), z["camera_poses"].astype(np.float32)
                else:
                    t1 = time.time()
                    depth, c2w = runner(S["images"], S["rays"])
                    if a.save_predictions:
                        p = pred_npz(os.path.join(a.out, "predictions"), ds, kind, sc)
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        np.savez_compressed(p, depth=depth, camera_poses=c2w)
                    S["seconds_forward"] = round(time.time() - t1, 2)
                S.pop("images")
                pending.append(pool.apply_async(score_task, ((S, depth, c2w),)))
                collect(block=False)
            collect(block=True)
            pool.close()
            pool.join()
            print(f"evaluated {n_done} scenes ({n_err} failed) in {time.time() - t0:.0f}s", flush=True)
            if n_err:
                sys.exit(1)
        n_cells = 0
        for ds, kind in cells:
            names = [s["scene"] for s in man["datasets"][ds]["scenes"]]
            if all(os.path.exists(scene_json(a.out, ds, kind, sc)) for sc in names):
                c = write_cell(a.out, ds, kind, names)
                n_cells += 1
                sm = c["summary"]
                print(f"{ds:9s} {kind:14s} Acc {sm['Acc']:.4f}  AUC@15 {100 * sm['AUC@15']:5.1f}  "
                      f"AUC@30 {100 * sm['AUC@30']:5.1f}")
        print(f"{n_cells} complete cells written")
    elif not a.verify:
        sys.exit("give --benchmark to evaluate (and/or --verify to compare finished results)")

    if a.verify:
        report, n_sc, n_bad = verify(a.out, a.verify, cells, a.tol)
        print_verify(report, a.tol)
        write_json(os.path.join(a.out, "verify_report.json"), dict(reference=a.verify, tol=a.tol, cells=report))
        print(f"verify: {len(report)} cells, {n_sc} scenes compared, {n_bad} cells outside tolerance "
              f"-> {os.path.join(a.out, 'verify_report.json')}")


if __name__ == "__main__":
    main()
