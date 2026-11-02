# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""The frozen benchmark definition is complete and consistent with the protocol constants."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]

from benchlib import benchmark as B                                     # noqa: E402
from benchlib import construct as K                                     # noqa: E402
from benchlib import protocol as P                                      # noqa: E402

MAN = B.load_manifest()


def test_protocol_sizes():
    assert P.native_size("pinhole") == (294, 294) and P.native_size("erp") == (1176, 588)
    assert P.native_size("kb4") == (644, 644) and P.native_size("cylindrical") == (1176, 588)
    assert P.roundtrip_sensor("equidistant") == (294, 294) and P.roundtrip_sensor("erp") == (520, 260)
    for kind in P.CAMERA_KINDS:
        rec = MAN["cameras"][kind]
        assert rec["native_size"] == list(P.native_size(kind))
        assert rec["roundtrip_sensor"] == list(P.roundtrip_sensor(kind))
        assert rec["fov_deg"] == P.CAMERA_FOV[kind]


def test_manifest_complete():
    n_cells = 0
    for ds in P.DATASETS:
        scenes = MAN["datasets"][ds]["scenes"]
        assert [s["scene"] for s in scenes] == [f"scene_{k:03d}" for k in range(P.N_SCENES)]
        for s in scenes:
            assert len(s["frames"]) == P.N_VIEWS and len({f["id"] for f in s["frames"]}) == P.N_VIEWS
            assert set(s["arrays"]) == {"erp_rng", "erp_msk", "ext_w2c"}
            assert set(s["cells"]) == set(P.CAMERA_KINDS)
            for f in s["frames"]:
                assert set(f["files"]) == {"rgb", "depth", "pose"}
                assert all(len(v["md5"]) == 32 and v["path"] for v in f["files"].values())
            n_cells += len(s["cells"])
    assert n_cells == len(P.DATASETS) * P.N_SCENES * len(P.CAMERA_KINDS) == 936


def test_yaws_and_extrinsics():
    """The view extrinsics in the manifest are the loader extrinsics composed with the frozen yaw."""
    n = 0
    for ds in P.DATASETS:
        for s in MAN["datasets"][ds]["scenes"]:
            for f in s["frames"]:
                yaw = K.yaw_deg_of_shift(f["yaw_shift_px"])
                assert yaw == f["yaw_deg"]
                ext = K.compose_w2c(np.asarray(f["ext_w2c_loader"], np.float32), yaw)
                assert np.array_equal(ext, np.asarray(f["ext_w2c"], np.float32))
                n += 1
    assert n == len(P.DATASETS) * P.N_SCENES * P.N_VIEWS


def test_yaw_rotation_is_ry():
    for yaw in (0.0, 29.88416988, 119.88416988, 300.0):
        R = K.yaw_rotation(yaw)
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        assert np.abs(R - np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])).max() < 1e-12
        assert abs(np.linalg.det(R) - 1) < 1e-12
