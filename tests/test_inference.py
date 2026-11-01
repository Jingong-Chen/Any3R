"""End-to-end inference on the bundled examples (needs a CUDA GPU and the weights; skipped otherwise).

The reference values were produced by the released weights, whose outputs are bit-identical to the
model evaluated in the paper; small differences across GPUs and library versions are tolerated."""
import json
import os

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.join(HERE, "reference_outputs.json")
WEIGHTS = os.environ.get("ANY3R_WEIGHTS", "Jingong/Any3R")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def run(name):
    from any3r import Any3R
    from any3r.io import list_images, load_cameras_json, prepare_views
    ex = os.path.join(ROOT, "examples", name)
    paths = list_images(os.path.join(ex, "images"))
    views = prepare_views(paths, load_cameras_json(os.path.join(ex, "cameras.json"), paths))
    try:
        model = Any3R.from_pretrained(WEIGHTS).cuda().eval()
    except (OSError, ValueError) as e:                 # no network / weights not reachable
        pytest.skip(f"weights not available: {type(e).__name__}")
    pred = model.infer(views.images, views.rays)
    v = views.valid.cuda()
    d = pred["depth"][..., 0]
    d = d / d[0][v[0]].median()                       # scale-free: relative to the first view
    t = pred["camera_poses"][:, :3, 3]
    t = t / t[1:].norm(dim=-1).max().clamp_min(1e-9)
    return {"depth_median": [float(d[i][v[i]].median()) for i in range(len(d))],
            "centres": t.cpu().numpy().round(6).tolist(),
            "conf_median": [float(pred["depth_conf"][i][v[i]].median()) for i in range(len(d))]}


@pytest.mark.parametrize("name", ["street_mixed", "fiord_meetingroom", "panocity_erp"])
def test_examples_match_reference(name):
    out = run(name)
    if os.environ.get("ANY3R_WRITE_REFERENCE"):
        ref = json.load(open(REF)) if os.path.exists(REF) else {}
        ref[name] = out
        json.dump(ref, open(REF, "w"), indent=1)
        pytest.skip("reference written")
    ref = json.load(open(REF))[name]
    assert np.allclose(out["depth_median"], ref["depth_median"], rtol=2e-2)
    assert np.abs(np.array(out["centres"]) - np.array(ref["centres"])).max() < 2e-2
