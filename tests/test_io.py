"""Input preparation: canvas layout, padding, rescaled cameras."""
import numpy as np
import pytest

from any3r import cameras as C
from any3r.io import prepare_views, load_cameras_json


def _img(h, w, seed=0):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def test_mixed_sizes_share_one_canvas():
    imgs = [_img(300, 400), _img(512, 512, 1), _img(256, 512, 2)]
    cams = [C.from_fov("pinhole", 400, 300, 60), C.from_fov("equidistant", 512, 512, 190), C.ERP()]
    v = prepare_views(imgs, cams, size=256)
    S, _, H, W = v.images.shape
    assert S == 3 and H % 16 == 0 and W % 16 == 0
    assert v.rays.shape == (3, H, W, 3) and v.valid.shape == (3, H, W)
    for i, (t, l, h, w) in enumerate(v.boxes):
        assert max(h, w) == 256
        assert v.cameras[i].width == w and v.cameras[i].height == h
        # outside the view's box: black pixels and the (0, 0, 1) ray
        m = np.ones((H, W), bool); m[t:t + h, l:l + w] = False
        assert np.all(v.images[i].numpy()[:, m] == 0)
        assert np.all(v.rays[i].numpy()[m] == np.array([0, 0, 1], np.float32))
        assert not v.valid[i].numpy()[m].any()
    # pixels outside the fisheye circle are blacked out
    t, l, h, w = v.boxes[1]
    inside = v.valid[1].numpy()[t:t + h, l:l + w]
    assert (~inside).any() and np.all(v.images[1].numpy()[:, t:t + h, l:l + w][:, ~inside] == 0)


def test_rescaled_camera_matches_resampled_rays():
    cam = C.from_fov("kb4", 640, 640, 200)
    v = prepare_views([_img(640, 640)], [cam], size=320)
    small = v.cameras[0]
    r_small, _ = small.ray_map()
    r_full, ok = cam.ray_map()
    # the centre of small pixel (i, j) is the centre of the 2x2 block of full pixels
    blk = r_full.reshape(320, 2, 320, 2, 3).mean((1, 3))
    blk /= np.linalg.norm(blk, axis=-1, keepdims=True)
    good = ok.reshape(320, 2, 320, 2).all((1, 3))
    assert np.degrees(np.arccos(np.clip((blk * r_small).sum(-1)[good], -1, 1))).max() < 0.05


def test_no_upscaling_by_default():
    v = prepare_views([_img(100, 120)], [C.from_fov("pinhole", 120, 100, 60)], size=512)
    assert v.boxes[0][2:] == (100, 120)
    v = prepare_views([_img(100, 120)], [C.from_fov("pinhole", 120, 100, 60)], size=240, upscale=True)
    assert v.boxes[0][2:] == (200, 240)


def test_camera_calibrated_at_other_resolution(tmp_path):
    p = tmp_path / "cams.json"
    p.write_text('{"camera": {"model": "OPENCV_FISHEYE", "width": 1000, "height": 1000, '
                 '"params": [300, 300, 500, 500, 0.01, 0, 0, 0], "fov_deg": 190}}')
    cams = load_cameras_json(str(p), ["a.png"])
    v = prepare_views([_img(500, 500)], cams, size=None)
    assert v.cameras[0].fx == pytest.approx(150.0) and v.cameras[0].cx == pytest.approx(250.0)
    with pytest.raises(ValueError):
        prepare_views([_img(500, 400)], cams, size=None)
