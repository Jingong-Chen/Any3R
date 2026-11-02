# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Build the any-camera benchmark (PP13) from the original datasets and check it against the
manifest.

    python eval/build_benchmark.py --out data/any3r_bench \
        --m3d-root /data/matterport3d_mp360 --s2d3ds-root /data/stanford2d3ds/no_xyz \
        --panocity-root /data/panocity

For every scene of benchmark/manifest.json the three named panoramas are read with PanoVGGT's
loader rules, turned to their frozen yaws and round-tripped through each of the 13 camera models;
every resulting array is compared with the md5 frozen in the manifest. The dataset files are hashed
too and compared with the reference copies (--skip-source-md5 to skip); a file may differ in its
bytes and still decode to the same arrays, so only --strict-sources makes that fatal. The process
exits with status 1 if any array differs; the report (--report, default {out}/build_report.json)
lists every comparison.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                                       # noqa: E402

from benchlib import benchmark as B                                     # noqa: E402
from benchlib import construct as K                                     # noqa: E402
from benchlib import protocol as P                                      # noqa: E402

_JOB = {}


def _md5_file(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_scene(task):
    """Build one scene for every requested camera; returns its report entry."""
    ds, scene = task
    J = _JOB
    root, out, cams, man = J["roots"][ds], J["out"], J["cameras"], J["manifest"]
    rec = B.scene_record(man, ds, scene)
    t0 = time.time()
    entry = {"ds": ds, "scene": scene, "seq_name": rec["seq_name"], "checks": {}, "ok": True}

    def check(name, arr, ref):
        ok = B.array_matches(arr, ref)
        entry["checks"][name] = ok if ok else {"got_md5": B.array_md5(arr), "want_md5": ref["md5"],
                                               "dtype": str(np.asarray(arr).dtype), "shape": list(np.shape(arr))}
        entry["ok"] &= ok
        return ok

    entry["sources"] = {}
    if not J["skip_source_md5"]:
        # a file whose bytes differ can still decode to the same arrays (e.g. a depth PNG re-encoded by
        # another libpng); the array checks below decide, unless --strict-sources
        for v, fr in enumerate(rec["frames"]):
            for kind, f in fr["files"].items():
                p = os.path.join(root, f["path"])
                got = _md5_file(p) if os.path.isfile(p) else None
                ok = got == f["md5"]
                entry["sources"][f"view{v}/{kind}"] = ok if ok else {"path": f["path"], "got_md5": got,
                                                                     "want_md5": f["md5"]}
                if J["strict_sources"]:
                    entry["ok"] &= ok
    try:
        src = K.source_arrays(ds, root, rec)
    except Exception as ex:                                              # noqa: BLE001
        entry.update(ok=False, error=f"{type(ex).__name__}: {ex}")
        return entry
    loader_ok = all(np.array_equal(src["ext_loader"][v], np.asarray(fr["ext_w2c_loader"], np.float32))
                    for v, fr in enumerate(rec["frames"]))
    entry["checks"]["ext_w2c_loader"] = loader_ok
    entry["ok"] &= loader_ok
    shared = {"erp_rng": src["rng"], "erp_msk": src["msk"], "ext_w2c": src["ext"]}
    for key, arr in shared.items():
        check(key, arr, rec["arrays"][key])
    meta = dict(seq_name=np.array(rec["seq_name"]), seq_index=np.int32(rec["seq_index"]),
                ids=np.array([fr["id"] for fr in rec["frames"]], np.int32),
                frame_names=np.array([fr["name"] for fr in rec["frames"]]),
                yaw_shift_px=np.array([fr["yaw_shift_px"] for fr in rec["frames"]], np.int32),
                yaw_deg=np.array([fr["yaw_deg"] for fr in rec["frames"]], np.float64))
    for kind in cams:
        erp_rgb, cone = K.camera_cell(kind, src)
        check(f"{kind}/erp_rgb", erp_rgb, rec["cells"][kind]["erp_rgb"])
        check(f"{kind}/cone", cone, rec["cells"][kind]["cone"])
        if out:
            path = B.scene_path(out, ds, kind, scene)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}.npz"
            np.savez_compressed(tmp, erp_rgb=erp_rgb, cone=cone, **shared, **meta, cam_kind=np.array(kind),
                                cam_fov_deg=np.float32(P.CAMERA_FOV[kind]),
                                cam_sensor=np.array(P.roundtrip_sensor(kind), np.int32))
            os.replace(tmp, path)
    entry["seconds"] = round(time.time() - t0, 1)
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output directory (omit to only check against the manifest)")
    ap.add_argument("--m3d-root", default=None, help="Matterport3D root (PanoVGGT layout: <house>/pano_*)")
    ap.add_argument("--s2d3ds-root", default=None, help="Stanford2D3DS root (the no_xyz directory: area_5b/pano/*)")
    ap.add_argument("--panocity-root", default=None, help="PanoCity root (<city>/<block>/...)")
    ap.add_argument("--manifest", default=None, help="default: benchmark/manifest.json")
    ap.add_argument("--datasets", default=None, help="comma list (default: every dataset with a root)")
    ap.add_argument("--cameras", default=None, help="comma list (default: all 13)")
    ap.add_argument("--scenes", default=None, help="comma list of scene names, e.g. scene_000,scene_005")
    ap.add_argument("--workers", type=int, default=1, help="scenes built in parallel")
    ap.add_argument("--skip-source-md5", action="store_true", help="do not hash the dataset files first")
    ap.add_argument("--strict-sources", action="store_true",
                    help="fail on a dataset file whose md5 differs from the manifest even if its arrays match")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()
    man = B.load_manifest(a.manifest)
    roots = {"M3D": a.m3d_root, "S2D3DS": a.s2d3ds_root, "PanoCity": a.panocity_root}
    dss = a.datasets.split(",") if a.datasets else [d for d in P.DATASETS if roots[d]]
    for d in dss:
        if not roots.get(d):
            sys.exit(f"no root given for {d}")
    cams = a.cameras.split(",") if a.cameras else list(P.CAMERA_KINDS)
    unknown = [c for c in cams if c not in P.CAMERA_KINDS]
    if unknown:
        sys.exit(f"unknown cameras {unknown}")
    scenes = set(a.scenes.split(",")) if a.scenes else None
    tasks = [(d, s["scene"]) for d in dss for s in man["datasets"][d]["scenes"]
             if scenes is None or s["scene"] in scenes]
    _JOB.update(roots=roots, out=a.out, cameras=cams, manifest=man, skip_source_md5=a.skip_source_md5,
                strict_sources=a.strict_sources)
    print(f"building {len(tasks)} scenes x {len(cams)} cameras -> {a.out or '(check only)'}", flush=True)
    t0 = time.time()
    if a.workers > 1:
        with mp.get_context("fork").Pool(a.workers) as pool:
            entries = []
            for e in pool.imap(build_scene, tasks):
                entries.append(e)
                print(f"  {e['ds']}/{e['scene']}: {'OK' if e['ok'] else 'MISMATCH'} {e.get('error', '')}", flush=True)
    else:
        entries = []
        for t in tasks:
            e = build_scene(t)
            entries.append(e)
            print(f"  {e['ds']}/{e['scene']}: {'OK' if e['ok'] else 'MISMATCH'} {e.get('error', '')}", flush=True)
    n_checks = sum(len(e["checks"]) for e in entries)
    n_bad = sum(1 for e in entries for v in e["checks"].values() if v is not True)
    n_src = sum(len(e.get("sources", {})) for e in entries)
    n_src_same = sum(1 for e in entries for v in e.get("sources", {}).values() if v is True)
    summary = {"scenes": len(entries), "scenes_ok": sum(e["ok"] for e in entries), "array_checks": n_checks,
               "array_checks_failed": n_bad, "source_files": n_src, "source_files_identical": n_src_same,
               "seconds": round(time.time() - t0, 1)}
    by_ds = {}
    for e in entries:
        s = by_ds.setdefault(e["ds"], {"scenes": 0, "scenes_ok": 0})
        s["scenes"] += 1
        s["scenes_ok"] += int(e["ok"])
    summary["by_dataset"] = by_ds
    report = a.report or (os.path.join(a.out, "build_report.json") if a.out else None)
    if report:
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        with open(report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "scenes": entries}, f, indent=1)
    print(json.dumps(summary))
    if n_src_same != n_src:
        print(f"note: {n_src - n_src_same} of {n_src} dataset files differ from the reference copies in their bytes "
              f"(see the report); the benchmark is defined by the array checks")
    sys.exit(0 if n_bad == 0 and summary["scenes_ok"] == len(entries) else 1)


if __name__ == "__main__":
    main()
