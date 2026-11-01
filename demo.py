# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Reconstruct a scene from images taken with any central camera.

Examples
--------
    # mixed cameras, one calibration per image in a JSON file
    python demo.py --images examples/street_mixed/images --cameras examples/street_mixed/cameras.json

    # a COLMAP calibration (cameras.bin / images.bin)
    python demo.py --images my_scene/images --colmap my_scene/sparse/0

    # every image from the same lens, known only by model and field of view
    python demo.py --images my_fisheye/ --camera equidistant --fov 190

    # equirectangular panoramas
    python demo.py --images my_panos/ --camera erp

Writes points.ply, scene.glb, cameras.json and depth.npz to --out.
"""
import argparse
import os
import sys
import time

import torch

from any3r import Any3R
from any3r.cameras import CAMERA_KINDS, from_fov, ERP
from any3r.export import save_predictions
from any3r.io import list_images, load_cameras_json, load_colmap_cameras, load_image, prepare_views


def build_cameras(args, paths):
    if args.cameras:
        return load_cameras_json(args.cameras, paths)
    if args.colmap:
        return load_colmap_cameras(args.colmap, paths)
    if args.camera == "erp":
        return [ERP() for _ in paths]
    if args.camera:
        if args.fov is None and args.camera not in ("cylindrical",):
            sys.exit(f"--camera {args.camera} needs --fov (the field of view across the image height, in degrees)")
        cams = []
        for p in paths:
            h, w = load_image(p).shape[:2]
            cams.append(from_fov(args.camera, w, h, args.fov if args.fov is not None else 360.0))
        return cams
    sys.exit("Any3R needs each image's camera: give --cameras (JSON), --colmap (sparse model) or "
             "--camera MODEL --fov DEG. See docs/cameras.md.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, nargs="+", help="image files or one folder")
    g = ap.add_argument_group("camera (choose one)")
    g.add_argument("--cameras", help="JSON file with the camera of every image (see docs/cameras.md)")
    g.add_argument("--colmap", help="COLMAP sparse model directory holding cameras and images files")
    g.add_argument("--camera", choices=CAMERA_KINDS, help="one camera model for all images")
    g.add_argument("--fov", type=float, help="with --camera: field of view across the image height (degrees)")
    ap.add_argument("--out", default="outputs/demo", help="output directory")
    ap.add_argument("--model", default="Jingong/Any3R", help="Hugging Face repo id or local directory")
    ap.add_argument("--size", type=int, default=512, help="downscale views whose longer side exceeds this")
    ap.add_argument("--conf", type=float, default=10.0, help="drop points below this confidence percentile")
    ap.add_argument("--keep-black", action="store_true", help="keep points of pure-black pixels (dropped by default)")
    ap.add_argument("--min-views", type=int, default=0,
                    help="keep only points confirmed by this many other views (multi-view consistency; 0 = off)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = list_images(args.images[0]) if len(args.images) == 1 and os.path.isdir(args.images[0]) else args.images
    if not paths:
        sys.exit(f"no images found in {args.images}")
    cameras = build_cameras(args, paths)
    views = prepare_views(paths, cameras, size=args.size)
    print(f"{len(paths)} views on a {views.images.shape[-1]}x{views.images.shape[-2]} canvas: "
          + ", ".join(sorted({c.kind for c in views.cameras})))

    model = Any3R.from_pretrained(args.model).to(args.device).eval()
    t0 = time.time()
    pred = model.infer(views.images, views.rays)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"inference {time.time() - t0:.2f} s")
    files = save_predictions(args.out, pred, views, conf_percentile=args.conf, filter_black=not args.keep_black,
                             min_views=args.min_views)
    for k, v in files.items():
        print(f"  {k:8s} {v}")


if __name__ == "__main__":
    main()
