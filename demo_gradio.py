# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Interactive Any3R demo: upload images, describe their cameras, inspect the reconstruction.

    pip install -e ".[demo]"
    python demo_gradio.py [--model Jingong/Any3R] [--share]
"""
import argparse
import glob
import json
import os
import shutil
import tempfile
import time

import gradio as gr
import numpy as np
import torch

from any3r import Any3R
from any3r.cameras import CAMERA_KINDS, ERP, from_fov
from any3r.export import save_glb, save_ply, scene_points
from any3r.io import load_cameras_json, load_colmap_cameras, load_image, prepare_views

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, "examples")
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LENS_LABELS = {
    "pinhole": "Pinhole (rectilinear)", "brown_conrady": "Brown-Conrady (OpenCV, radial-tangential)",
    "equidistant": "Equidistant fisheye", "kb4": "Kannala-Brandt fisheye (OpenCV fisheye)",
    "mei": "MEI / unified omnidirectional", "double_sphere": "Double sphere", "eucm": "Extended unified (EUCM)",
    "equisolid": "Equisolid fisheye", "stereographic": "Stereographic fisheye", "fov": "FOV (Devernay-Faugeras)",
    "cylindrical": "Cylindrical panorama (360 x 90)", "erp": "Equirectangular panorama (360 x 180)",
    "erp_crop": "Equirectangular window",
}
DEFAULT_FOV = {"pinhole": 60, "brown_conrady": 90, "equidistant": 180, "kb4": 200, "mei": 200,
               "double_sphere": 200, "eucm": 200, "equisolid": 180, "stereographic": 180, "fov": 150,
               "cylindrical": 360, "erp": 360, "erp_crop": 180}


def get_model(name):
    global MODEL
    if MODEL is None:
        MODEL = Any3R.from_pretrained(name).to(DEVICE).eval()
    return MODEL


def _paths(files):
    return [f if isinstance(f, str) else f.name for f in (files or [])]


def _colorize(depth, valid):
    import matplotlib
    d = depth.copy()
    v = valid & np.isfinite(d)
    if not v.any():
        return np.zeros(d.shape + (3,), np.uint8)
    lo, hi = np.percentile(d[v], [2, 98])
    t = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = (matplotlib.colormaps["turbo_r"](t)[..., :3] * 255).astype(np.uint8)
    rgb[~v] = 0
    return rgb


def build_cameras(paths, mode, lens, fov, cam_file, colmap_files):
    if mode == "Same lens for every image":
        if lens == "erp":
            return [ERP() for _ in paths]
        cams = []
        for p in paths:
            h, w = load_image(p).shape[:2]
            cams.append(from_fov(lens, w, h, float(fov)))
        return cams
    if mode == "Per-image cameras (JSON)":
        if cam_file is None:
            raise gr.Error("Upload a cameras.json (see docs/cameras.md for the format).")
        return load_cameras_json(_paths([cam_file])[0], paths)
    colmap = _paths(colmap_files)
    if not colmap:
        raise gr.Error("Upload COLMAP cameras.(txt|bin) and images.(txt|bin).")
    d = tempfile.mkdtemp(prefix="any3r_colmap_")
    for f in colmap:
        shutil.copy(f, os.path.join(d, os.path.basename(f)))
    return load_colmap_cameras(d, paths)


def export_scene(state, conf, filter_black, show_cams, consistent=False):
    if not state:
        return None, None
    views, pred = state["views"], state["pred"]
    xyz, rgb, _ = scene_points(pred, views, conf_percentile=conf, filter_black=filter_black,
                               min_views=1 if consistent else 0, max_points=2_000_000)
    out = state["dir"]
    glb = os.path.join(out, f"scene_{int(conf)}_{int(filter_black)}_{int(show_cams)}_{int(consistent)}.glb")
    save_glb(glb, xyz, rgb, pred["camera_poses"].cpu().numpy() if show_cams else None, views.cameras)
    ply = os.path.join(out, "points.ply")
    save_ply(ply, xyz, rgb)
    return glb, [ply, state["cameras_json"]]


def reconstruct(files, mode, lens, fov, cam_file, colmap_files, size, conf, filter_black, show_cams, consistent,
                model_name):
    paths = _paths(files)
    if not paths:
        raise gr.Error("Upload at least one image.")
    cams = build_cameras(paths, mode, lens, fov, cam_file, colmap_files)
    views = prepare_views(paths, cams, size=int(size))
    model = get_model(model_name)
    t0 = time.time()
    pred = model.infer(views.images, views.rays)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    pred = {k: v.cpu() for k, v in pred.items()}
    state = {"views": views, "pred": pred, "dir": tempfile.mkdtemp(prefix="any3r_")}
    state["cameras_json"] = os.path.join(state["dir"], "cameras.json")
    with open(state["cameras_json"], "w") as f:
        json.dump([{"image": n, "camera_to_world": p.tolist(), "camera": c.to_dict()}
                   for n, p, c in zip(views.names, pred["camera_poses"].numpy(), views.cameras)], f, indent=1)
    glb, downloads = export_scene(state, conf, filter_black, show_cams, consistent)

    depth = pred["depth"][..., 0].numpy()
    valid = views.valid.numpy()
    gallery = []
    for i, name in enumerate(views.names):
        img = (views.crop(views.images.permute(0, 2, 3, 1).numpy(), i) * 255).astype(np.uint8)
        dep = _colorize(views.crop(depth, i), views.crop(valid, i))
        gallery.append((np.concatenate([img, dep], 1), f"{name} · {views.cameras[i].kind}"))
    kinds = sorted({c.kind for c in views.cameras})
    info = (f"**{len(paths)} views** ({', '.join(kinds)}) on a {views.images.shape[-1]}×{views.images.shape[-2]} canvas · "
            f"inference {dt:.2f} s on {torch.cuda.get_device_name() if DEVICE == 'cuda' else 'CPU'}")
    return state, glb, gallery, downloads, info


def load_example(name):
    d = os.path.join(EXAMPLES, name)
    imgs = sorted(glob.glob(os.path.join(d, "images", "*")))
    return imgs, "Per-image cameras (JSON)", os.path.join(d, "cameras.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Jingong/Any3R")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    with gr.Blocks(title="Any3R", theme=theme, css=".gradio-container {max-width: 1280px !important}") as demo:
        gr.Markdown("# Any3R: 3D reconstruction from any camera\n"
                    "Upload images taken with any central camera (pinhole, fisheye beyond 180°, panoramas, or a mix), "
                    "say what the cameras are, and get depth, camera poses and a fused point cloud in one forward pass. "
                    "[Project page](https://jingong-chen.github.io/Any3R/) · "
                    "[Code](https://github.com/Jingong-Chen/Any3R)")
        state = gr.State(None)
        model_name = gr.State(args.model)
        with gr.Row():
            with gr.Column(scale=4):
                files = gr.File(label="Images", file_count="multiple", file_types=["image"])
                mode = gr.Radio(["Same lens for every image", "Per-image cameras (JSON)", "COLMAP calibration"],
                                value="Same lens for every image", label="Cameras")
                with gr.Group(visible=True) as same_box:
                    lens = gr.Dropdown([(LENS_LABELS[k], k) for k in CAMERA_KINDS], value="equidistant",
                                       label="Camera model")
                    fov = gr.Slider(10, 360, value=180, step=1, label="Field of view across the image height (°)")
                cam_file = gr.File(label="cameras.json", file_types=[".json"], visible=False)
                colmap_files = gr.File(label="COLMAP cameras + images (.txt or .bin)", file_count="multiple",
                                       visible=False)
                with gr.Accordion("Options", open=False):
                    size = gr.Slider(256, 1024, value=512, step=16, label="Max image side (px)")
                    conf = gr.Slider(0, 90, value=10, step=1, label="Drop points below confidence percentile")
                    filter_black = gr.Checkbox(True, label="Drop pure-black pixels (masked sky / background)")
                    consistent = gr.Checkbox(False, label="Keep only points confirmed by another view")
                    show_cams = gr.Checkbox(True, label="Show cameras")
                run = gr.Button("Reconstruct", variant="primary")
                info = gr.Markdown()
            with gr.Column(scale=7):
                viewer = gr.Model3D(label="Reconstruction", height=560, clear_color=[1, 1, 1, 1])
                downloads = gr.File(label="Downloads", file_count="multiple")
        gallery = gr.Gallery(label="Input · predicted depth", columns=4, height="auto", object_fit="contain")

        def on_mode(m):
            return (gr.update(visible=m == "Same lens for every image"),
                    gr.update(visible=m == "Per-image cameras (JSON)"),
                    gr.update(visible=m == "COLMAP calibration"))
        mode.change(on_mode, mode, [same_box, cam_file, colmap_files])
        lens.change(lambda k: gr.update(value=DEFAULT_FOV[k], interactive=k not in ("erp", "cylindrical")), lens, fov)

        gr.Markdown("### Examples")
        with gr.Row():
            for name, label in [("street_mixed", "Street · 8 camera models"),
                                ("fiord_meetingroom", "Real 214° fisheye (FIORD)"),
                                ("panocity_erp", "360° panoramas")]:
                if os.path.isdir(os.path.join(EXAMPLES, name)):
                    gr.Button(label, size="sm").click(lambda n=name: load_example(n), None, [files, mode, cam_file]) \
                        .then(on_mode, mode, [same_box, cam_file, colmap_files])

        inputs = [files, mode, lens, fov, cam_file, colmap_files, size, conf, filter_black, show_cams, consistent,
                  model_name]
        run.click(reconstruct, inputs, [state, viewer, gallery, downloads, info])
        refilter = [state, conf, filter_black, show_cams, consistent]
        conf.release(export_scene, refilter, [viewer, downloads])
        for c in (filter_black, show_cams, consistent):
            c.change(export_scene, refilter, [viewer, downloads])
    demo.queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
