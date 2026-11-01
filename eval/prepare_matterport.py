# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.

"""Prepare the Matterport3D panoramas of the benchmark from the official 'mp360' task data.

Matterport3D's own release has no equirectangular panoramas. The benchmark uses the 2048 x 1024
panoramas with depth and pose stitched by the authors of 360MonoDepth (Rey-Area et al., 2022)
and distributed through the official download script (`download_mp.py --task_data mp360`, seven
archives data_00.zip ... data_06.zip), converted to the layout PanoVGGT's Matterport3D loader reads:

    data/<house>/<view>_rgb.png    -> <house>/pano_skybox_color/<view>.png   (copied unchanged)
    data/<house>/<view>_depth.dpt  -> <house>/pano_depth/<view>.png          (uint16, metres x 4000)
    data/<house>/<view>_pose.txt   -> <house>/pano_poses/<view>.txt          (4x4 camera-to-world)

    python eval/prepare_matterport.py --mp360 /data/matterport/v1/tasks/mp3d_360 --out /data/matterport3d

--mp360 is the directory of the archives (or of their extracted data/ folder). By default only the
frames of the benchmark are converted; --all converts every panorama. The poses are written with six
significant digits and the depth maps are re-encoded, as in the preparation the benchmark was built
from, so build_benchmark.py reproduces every array of the benchmark from the result.
"""
import argparse
import glob
import os
import struct
import sys
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from benchlib import benchmark as B                                     # noqa: E402

DEPTH_SCALE = 4000.0


def read_dpt(buf):
    """Sintel .dpt bytes: 'PIEH', width, height (int32), then float32 depth in metres."""
    if buf[:4] != b"PIEH":
        raise ValueError("not a .dpt depth file")
    W, H = struct.unpack("<ii", buf[4:12])
    return np.frombuffer(buf[12:12 + 4 * W * H], dtype=np.float32).reshape(H, W)


def depth_to_png16(depth_m):
    d = np.where(np.isfinite(depth_m), depth_m, np.float32(0)).astype(np.float32)
    return np.clip(d * np.float32(DEPTH_SCALE), np.float32(0), np.float32(65535)).astype(np.uint16)


def pose_to_c2w(text):
    """mp360 pose: tx ty tz qx qy qz qw (camera-to-world, Matterport world, OpenGL camera axes)."""
    from scipy.spatial.transform import Rotation
    v = np.array([float(x) for x in text.split()], np.float64)
    if v.shape != (7,):
        raise ValueError(f"pose has {v.size} values, expected 7")
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = Rotation.from_quat(v[3:]).as_matrix()
    c2w[:3, 3] = v[:3]
    return c2w


class Source:
    """The mp360 files, from the archives or from an extracted data/ folder."""

    def __init__(self, path):
        self.dir = None
        self.index = {}
        zips = sorted(glob.glob(os.path.join(path, "data_*.zip")))
        if zips:
            for z in zips:
                zf = zipfile.ZipFile(z)
                for n in zf.namelist():
                    parts = n.split("/")
                    if len(parts) == 3 and parts[2]:
                        self.index[(parts[1], parts[2])] = (zf, n)
        else:
            self.dir = os.path.join(path, "data") if os.path.isdir(os.path.join(path, "data")) else path

    def read(self, house, name):
        if self.dir is not None:
            with open(os.path.join(self.dir, house, name), "rb") as f:
                return f.read()
        zf, n = self.index[(house, name)]
        return zf.read(n)

    def views(self, house=None):
        if self.dir is not None:
            houses = [house] if house else sorted(os.listdir(self.dir))
            return [(h, f[:-8]) for h in houses for f in sorted(os.listdir(os.path.join(self.dir, h)))
                    if f.endswith("_rgb.png")]
        return sorted((h, f[:-8]) for h, f in self.index if f.endswith("_rgb.png") and (house is None or h == house))


def convert(src, house, view, out):
    rgb = os.path.join(out, house, "pano_skybox_color", f"{view}.png")
    dep = os.path.join(out, house, "pano_depth", f"{view}.png")
    pose = os.path.join(out, house, "pano_poses", f"{view}.txt")
    for p in (rgb, dep, pose):
        os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(rgb, "wb") as f:
        f.write(src.read(house, f"{view}_rgb.png"))
    if not cv2.imwrite(dep, depth_to_png16(read_dpt(src.read(house, f"{view}_depth.dpt")))):
        raise IOError(f"cannot write {dep}")
    np.savetxt(pose, pose_to_c2w(src.read(house, f"{view}_pose.txt").decode()), fmt="%g")
    return rgb, dep, pose


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mp360", required=True, help="directory of data_XX.zip (or of the extracted data/ folder)")
    ap.add_argument("--out", required=True, help="Matterport3D root for build_benchmark.py --m3d-root")
    ap.add_argument("--all", action="store_true", help="convert every panorama, not only the benchmark's")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    src = Source(a.mp360)
    if a.all:
        todo = src.views()
    else:
        man = B.load_manifest(a.manifest)
        todo = sorted({(fr["files"]["rgb"]["path"].split("/")[0], fr["name"])
                       for s in man["datasets"]["M3D"]["scenes"] for fr in s["frames"]})
    for i, (house, view) in enumerate(todo):
        convert(src, house, view, a.out)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"  {i + 1}/{len(todo)} panoramas", flush=True)
    print(f"wrote {len(todo)} panoramas to {a.out}")
    if not a.all:
        # the benchmark's color directory is pano_skybox_color for every house
        bad = [fr["files"]["rgb"]["path"] for s in B.load_manifest(a.manifest)["datasets"]["M3D"]["scenes"]
               for fr in s["frames"] if "/pano_skybox_color/" not in fr["files"]["rgb"]["path"]]
        if bad:
            print(f"note: {len(bad)} manifest frames expect another color directory: {bad[:3]}")


if __name__ == "__main__":
    main()
