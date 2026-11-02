# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""The benchmark manifest and the per-cell scene files.

A built benchmark is a directory {root}/{DS}/PP13_{camera}/scene_XXX.npz, one file per dataset,
camera model and scene, holding the arrays the evaluator reads:
  erp_rgb  (3, 3, 518, 1036) uint8   source panorama of each view as seen through the camera
  cone     (3, 518, 1036) bool       its support
  erp_rng  (3, 518, 1036) float32    ground-truth range of each view (metres)
  erp_msk  (3, 518, 1036) bool       pixels with valid ground truth
  ext_w2c  (3, 3, 4) float32         ground-truth world-to-camera of each view
plus identity fields (cam_kind, cam_fov_deg, seq_name, ids, ...). The manifest freezes the md5 of
every one of these arrays.
"""
import hashlib
import json
import os

import numpy as np

from . import protocol as P

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MANIFEST = os.path.join(HERE, "..", "benchmark", "manifest.json")
SCENE_ARRAYS = ("erp_rgb", "cone", "erp_rng", "erp_msk", "ext_w2c")


def load_manifest(path=None):
    with open(path or DEFAULT_MANIFEST, encoding="utf-8") as f:
        return json.load(f)


def array_md5(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def array_matches(a, rec):
    a = np.asarray(a)
    return str(a.dtype) == rec["dtype"] and list(a.shape) == rec["shape"] and array_md5(a) == rec["md5"]


def scene_path(root, ds, kind, scene):
    return os.path.join(root, ds, f"PP13_{kind}", f"{scene}.npz")


def expected(manifest, ds, scene, key, kind=None):
    """The manifest record (dtype, shape, md5) of one array of a scene (per camera for erp_rgb / cone)."""
    s = scene_record(manifest, ds, scene)
    return s["cells"][kind][key] if key in ("erp_rgb", "cone") else s["arrays"][key]


def scene_record(manifest, ds, scene):
    for s in manifest["datasets"][ds]["scenes"]:
        if s["scene"] == scene:
            return s
    raise KeyError(f"{ds}/{scene} is not in the manifest")


def load_scene(root, ds, kind, scene, manifest=None, check=True):
    """The arrays of one cell's scene file. With check, every array must match its manifest md5."""
    path = scene_path(root, ds, kind, scene)
    with np.load(path) as z:
        if "cam_kind" in z.files and str(z["cam_kind"]) != kind:
            raise ValueError(f"{path} holds camera {z['cam_kind']}, not {kind}")
        if "cam_fov_deg" in z.files and float(z["cam_fov_deg"]) != P.CAMERA_FOV[kind]:
            raise ValueError(f"{path} holds field of view {z['cam_fov_deg']}, not {P.CAMERA_FOV[kind]}")
        out = {k: z[k] for k in SCENE_ARRAYS}
    if check:
        bad = [k for k in SCENE_ARRAYS if not array_matches(out[k], expected(manifest, ds, scene, k, kind))]
        if bad:
            raise ValueError(f"{path}: {bad} differ from the benchmark manifest (rebuild with build_benchmark.py)")
    return out


def iter_cells(manifest, datasets=None, cameras=None, scenes=None):
    """(ds, kind, scene) of the benchmark, filtered."""
    for ds in datasets or P.DATASETS:
        for kind in cameras or P.CAMERA_KINDS:
            for s in manifest["datasets"][ds]["scenes"]:
                if scenes is None or s["scene"] in scenes:
                    yield ds, kind, s["scene"]
