"""Camera-model self-checks: projection inverts unprojection, the COLMAP / OpenCV readers agree with
the reference implementations, and the factories produce what they document."""
import math
import os
import struct

import numpy as np
import pytest

from any3r import cameras as C

SIZES = [(64, 64), (48, 96)]


@pytest.mark.parametrize("kind", C.CAMERA_KINDS)
@pytest.mark.parametrize("fov", [90.0, 150.0, 200.0])
def test_roundtrip(kind, fov):
    if kind in ("pinhole", "brown_conrady", "fov") and fov >= 180:
        with pytest.raises(ValueError):
            C.from_fov(kind, 64, 64, fov)
        return
    for H, W in SIZES:
        cam = C.from_fov(kind, W, H, fov)
        rays, valid = cam.ray_map()
        assert rays.shape == (H, W, 3) and rays.dtype == np.float32 and valid.any()
        assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0, atol=1e-6)
        assert np.all(rays[~valid] == np.array([0, 0, 1], np.float32))
        err, n = cam.roundtrip_error()
        assert n > 0 and err < 1e-4, (kind, fov, err)


@pytest.mark.parametrize("kind", [k for k in C.CAMERA_KINDS if k not in ("erp", "erp_crop", "cylindrical", "brown_conrady")])
def test_fov_is_honoured(kind):
    fov = 120.0 if kind in ("pinhole", "fov") else 190.0
    cam = C.from_fov(kind, 257, 257, fov)
    # the top-centre edge of the image (half a pixel above the first row centre) lies at fov / 2
    d, _ = cam.pixel_to_ray(np.array([cam.cx]), np.array([0.0]))
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    theta = math.degrees(math.acos(float(np.clip(d[0, 2], -1, 1))))
    assert abs(theta - fov / 2) < 1e-6, (kind, theta)


def test_erp_matches_model_default():
    import torch
    from any3r.geometry import erp_directions
    r, v = C.ERP(width=96, height=48).ray_map()
    assert v.all()
    assert np.abs(r - erp_directions(48, 96).numpy()).max() < 1e-6


def test_resized_is_exact():
    cam = C.from_fov("kb4", 200, 100, 200.0)
    small = cam.resized(100, 50)
    u, v = np.array([10.0, 40.0]), np.array([20.0, 30.0])
    d1, _ = cam.pixel_to_ray(u, v)
    d2, _ = small.pixel_to_ray(u / 2, v / 2)
    assert np.abs(d1 / np.linalg.norm(d1, axis=-1, keepdims=True)
                  - d2 / np.linalg.norm(d2, axis=-1, keepdims=True)).max() < 1e-12


def _write_colmap(tmp, cams):
    with open(os.path.join(tmp, "cameras.txt"), "w") as f:
        f.write("# Camera list\n")
        for cid, (name, w, h, p) in cams.items():
            f.write(f"{cid} {name} {w} {h} " + " ".join(repr(x) for x in p) + "\n")
    with open(os.path.join(tmp, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, (name, w, h, p) in cams.items():
            f.write(struct.pack("<iiQQ", cid, C._COLMAP[name][0], w, h))
            f.write(struct.pack("<" + "d" * len(p), *p))


def test_colmap_txt_bin(tmp_path):
    cams = {1: ("PINHOLE", 640, 480, [500.0, 510.0, 320.5, 240.5]),
            2: ("OPENCV_FISHEYE", 800, 800, [300.0, 300.0, 400.0, 400.0, 0.05, -0.01, 0.002, -1e-4]),
            3: ("SIMPLE_RADIAL", 320, 240, [250.0, 160.0, 120.0, -0.1]),
            4: ("FOV", 640, 480, [280.0, 280.0, 320.0, 240.0, 0.9]),
            5: ("FULL_OPENCV", 640, 480, [400., 400., 320., 240., -0.2, 0.05, 1e-3, -1e-3, 0.0, 0.01, 0.0, 0.0])}
    _write_colmap(str(tmp_path), cams)
    a = C.read_colmap_cameras(os.path.join(tmp_path, "cameras.txt"))
    b = C.read_colmap_cameras(os.path.join(tmp_path, "cameras.bin"))
    for cid in cams:
        ra, _ = a[cid].ray_map()
        rb, _ = b[cid].ray_map()
        assert np.array_equal(ra, rb)
        err, n = a[cid].roundtrip_error()
        assert n > 0 and err < 1e-4, (cams[cid][0], err)


def test_opencv_agrees_with_cv2():
    cv2 = pytest.importorskip("cv2")
    K = np.array([[420.0, 0, 319.3], [0, 418.0, 241.7], [0, 0, 1]])
    pts = np.random.default_rng(0).normal(size=(200, 3)) * [0.4, 0.3, 0.0] + [0, 0, 1.0]
    dist = np.array([-0.21, 0.04, 1e-3, -8e-4, 0.01])
    cam = C.from_opencv(K, dist, 640, 480)
    u, v, _ = cam.project(pts)
    ref, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    assert np.abs(np.stack([u - 0.5, v - 0.5], -1) - ref[:, 0]).max() < 1e-6
    kd = np.array([0.03, -0.01, 0.002, -2e-4])
    fe = C.from_opencv(K, kd, 640, 480, fisheye=True)
    u, v, _ = fe.project(pts)
    ref, _ = cv2.fisheye.projectPoints(pts[:, None], np.zeros(3), np.zeros(3), K, kd)
    assert np.abs(np.stack([u - 0.5, v - 0.5], -1) - ref[:, 0]).max() < 1e-6


def test_camera_from_dict():
    a = C.camera_from_dict({"model": "OPENCV_FISHEYE", "width": 64, "height": 64,
                            "params": [20, 20, 32, 32, 0.01, 0, 0, 0]})
    b = C.camera_from_dict({"model": "kb4", "width": 64, "height": 64, "fx": 20, "fy": 20, "cx": 32, "cy": 32,
                            "k1": 0.01})
    assert np.array_equal(a.ray_map()[0], b.ray_map()[0])
    c = C.camera_from_dict({"model": "equidistant", "width": 64, "height": 64, "fov_deg": 190})
    assert abs(c.theta_max - math.radians(95)) < 1e-12
    d = C.camera_from_dict(c.to_dict())
    assert np.array_equal(c.ray_map()[0], d.ray_map()[0])
    e = C.camera_from_dict({"model": "erp", "width": 64, "height": 32})
    assert e.ray_map()[1].all()


def test_strong_barrel_distortion_inverts():
    """The fixed-point undistortion diverges for strong barrel distortion; the Newton pass must
    recover those pixels exactly (GoPro-like wide lens, k1 = -0.35)."""
    cam = C.BrownConrady(180.0, 180.0, 160.0, 120.0, k1=-0.35, k2=0.08, p1=1e-3, p2=-5e-4, width=320, height=240)
    rays, valid = cam.ray_map()
    assert valid.mean() > 0.99, valid.mean()
    err, n = cam.roundtrip_error()
    assert err < 1e-4, err


def test_circular_default():
    assert C.from_fov("pinhole", 64, 48, 90.0).ray_map()[1].all()
    assert not C.from_fov("pinhole", 64, 48, 90.0, circular=True).ray_map()[1].all()
    assert not C.from_fov("equidistant", 64, 64, 180.0).ray_map()[1].all()
    full = C.from_fov("equidistant", 64, 64, 180.0, circular=False)
    assert full.ray_map()[1].all() and full.roundtrip_error()[0] < 1e-4
