# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Input preparation: images and cameras -> the (images, ray_maps) pair Any3R takes.

Any3R processes all views of a scene on one H x W canvas. `prepare_views` downscales every view whose
longer side exceeds `size` pixels (the camera is rescaled with it, exactly), centres it on a
canvas that holds the largest view, and fills the margin with black pixels and the ray (0, 0, 1),
which is how the model saw pixels outside a lens during training. Pixels outside a fisheye's image
circle are blacked out the same way.
"""
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from .cameras import CameraModel, ERP, camera_from_dict

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def load_image(path_or_array) -> np.ndarray:
    """RGB image as float32 (H, W, 3) in [0, 1] from a file path, a PIL image or an array."""
    if isinstance(path_or_array, np.ndarray):
        im = path_or_array
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, -1)
        im = im[..., :3]
        return (im.astype(np.float32) / 255.0) if im.dtype == np.uint8 else im.astype(np.float32)
    from PIL import Image, ImageOps
    im = path_or_array if isinstance(path_or_array, Image.Image) else Image.open(path_or_array)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im, np.float32) / 255.0


def list_images(folder: str) -> List[str]:
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if img.shape[1] == w and img.shape[0] == h:
        return img
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False,
                                        antialias=(w < img.shape[1] or h < img.shape[0]))
    return t[0].permute(1, 2, 0).clamp(0, 1).numpy()


def _ceil_to(x: int, m: int) -> int:
    return int(math.ceil(x / m) * m)


@dataclass
class Views:
    """Network-ready views of one scene.

    images (S, 3, H, W) float32 in [0, 1] and rays (S, H, W, 3) go to Any3R.infer; valid (S, H, W)
    marks pixels that belong to a lens; boxes[i] = (top, left, height, width) is where view i sits
    on the canvas; cameras[i] is view i's camera at its resized size."""
    images: torch.Tensor
    rays: torch.Tensor
    valid: torch.Tensor
    boxes: List[tuple]
    cameras: List[CameraModel]
    names: List[str] = field(default_factory=list)

    def crop(self, x: Union[torch.Tensor, np.ndarray], i: int):
        """Cut view i out of a per-view canvas array (S, H, W, ...) -> (h, w, ...)."""
        t, l, h, w = self.boxes[i]
        return x[i, t:t + h, l:l + w]


def prepare_views(images: Sequence, cameras: Sequence, size: Optional[int] = 512, multiple: int = 16,
                  upscale: bool = False, names: Optional[Sequence[str]] = None) -> Views:
    """Resize, ray-map and pad a scene's views onto one canvas.

    Args:
        images: file paths, PIL images or arrays, one per view.
        cameras: one CameraModel (or camera dict, see any3r.cameras.camera_from_dict) per view,
            calibrated at the image's own resolution. A camera without a stored size is assumed to
            match its image; a camera calibrated at another resolution of the same aspect ratio is
            rescaled.
        size: views whose longer side exceeds this many pixels are downscaled to it (None keeps the
            input size; upscale=True also enlarges smaller views).
        multiple: canvas sides are rounded up to a multiple of this (the network's patch size).
    """
    if len(images) != len(cameras):
        raise ValueError(f"{len(images)} images but {len(cameras)} cameras")
    if len(images) == 0:
        raise ValueError("no views")
    imgs, cams = [], []
    for k, (im, cam) in enumerate(zip(images, cameras)):
        im = load_image(im)
        cam = camera_from_dict(cam) if isinstance(cam, dict) else cam
        h, w = im.shape[:2]
        if cam.width is None or cam.height is None:
            cam = cam._copy(); cam.width, cam.height = w, h
            if hasattr(cam, "_bind"):
                cam._bind(h, w)
        elif (cam.width, cam.height) != (w, h):
            if abs(cam.width / cam.height - w / h) > 1e-3:
                raise ValueError(f"view {k}: camera is calibrated for {cam.width}x{cam.height}, image is {w}x{h}")
            cam = cam.resized(w, h)
        if size is not None and (max(h, w) > size or upscale):
            s = size / max(h, w)
            nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            im, cam = _resize(im, nw, nh), cam.resized(nw, nh)
        imgs.append(im); cams.append(cam)

    Hc = _ceil_to(max(im.shape[0] for im in imgs), multiple)
    Wc = _ceil_to(max(im.shape[1] for im in imgs), multiple)
    S = len(imgs)
    out_img = np.zeros((S, Hc, Wc, 3), np.float32)
    out_ray = np.zeros((S, Hc, Wc, 3), np.float32); out_ray[..., 2] = 1.0
    out_val = np.zeros((S, Hc, Wc), bool)
    boxes = []
    for k, (im, cam) in enumerate(zip(imgs, cams)):
        h, w = im.shape[:2]
        rays, valid = cam.ray_map(h, w)
        t, l = (Hc - h) // 2, (Wc - w) // 2
        out_img[k, t:t + h, l:l + w] = np.where(valid[..., None], im, 0.0)
        out_ray[k, t:t + h, l:l + w] = rays
        out_val[k, t:t + h, l:l + w] = valid
        boxes.append((t, l, h, w))
    names = list(names) if names is not None else [
        os.path.basename(p) if isinstance(p, str) else f"view_{k:03d}" for k, p in enumerate(images)]
    return Views(images=torch.from_numpy(out_img).permute(0, 3, 1, 2).contiguous(),
                 rays=torch.from_numpy(out_ray), valid=torch.from_numpy(out_val),
                 boxes=boxes, cameras=cams, names=names)


def load_cameras_json(path: str, image_paths: Sequence[str]) -> List[CameraModel]:
    """Cameras for a list of images from a JSON file. Accepted layouts:

        {"camera": {...}}                                one camera for every image
        {"cameras": {"img_000.jpg": {...}, ...}}          per image, keyed by file name
        {"cameras": [{...}, {...}, ...]}                  per image, in order
        {"cameras": {"front": {...}}, "images": {"img_000.jpg": "front", ...}}   shared rigs

    Each camera is a dict for any3r.cameras.camera_from_dict. A string value "erp" is shorthand for
    an equirectangular panorama."""
    with open(path) as f:
        spec = json.load(f)

    def mk(c):
        return ERP() if c == "erp" else camera_from_dict(c)

    names = [os.path.basename(p) for p in image_paths]
    if "camera" in spec:
        return [mk(spec["camera"]) for _ in names]
    cams = spec["cameras"]
    if isinstance(cams, list):
        if len(cams) != len(names):
            raise ValueError(f"{path}: {len(cams)} cameras for {len(names)} images")
        return [mk(c) for c in cams]
    if "images" in spec:
        return [mk(cams[spec["images"][n]]) for n in names]
    missing = [n for n in names if n not in cams]
    if missing:
        raise ValueError(f"{path}: no camera for {missing[:3]}{'...' if len(missing) > 3 else ''}")
    return [mk(cams[n]) for n in names]


def _read_colmap_image_camera_ids(path: str) -> dict:
    """images.txt / images.bin of a COLMAP model -> {image name: camera_id}."""
    import struct
    out = {}
    if path.endswith(".bin"):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            for _ in range(n):
                _img_id, *_qt, cam_id = struct.unpack("<i7di", f.read(64))
                name = b""
                while (c := f.read(1)) != b"\x00":
                    name += c
                n2d = struct.unpack("<Q", f.read(8))[0]
                f.seek(24 * n2d, 1)
                out[name.decode("utf-8")] = cam_id
    else:
        with open(path) as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        for l in lines[0::2]:                       # every image takes two lines
            t = l.split()
            out[" ".join(t[9:])] = int(t[8])
    return out


def load_colmap_cameras(model_dir: str, image_paths: Sequence[str]) -> List[CameraModel]:
    """Cameras for a list of images from a COLMAP sparse model directory (cameras + images files).
    Images are matched by file name (or by their path relative to COLMAP's image root)."""
    from .cameras import read_colmap_cameras
    cams = read_colmap_cameras(model_dir)
    for name in ("images.bin", "images.txt"):
        if os.path.exists(os.path.join(model_dir, name)):
            ids = _read_colmap_image_camera_ids(os.path.join(model_dir, name))
            break
    else:
        if len(cams) == 1:
            return [next(iter(cams.values())) for _ in image_paths]
        raise FileNotFoundError(f"{model_dir} has several cameras but no images.bin / images.txt")
    by_base = {os.path.basename(k): v for k, v in ids.items()}
    out = []
    for p in image_paths:
        key = next((k for k in ids if p.replace(os.sep, "/").endswith(k)), None)
        cid = ids[key] if key is not None else by_base.get(os.path.basename(p))
        if cid is None:
            raise ValueError(f"{p} is not registered in {model_dir}")
        out.append(cams[cid])
    return out
