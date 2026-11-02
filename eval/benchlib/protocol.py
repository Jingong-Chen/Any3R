# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree.

"""Frozen constants of the any-camera benchmark (PP13).

Three datasets x thirteen central camera models x 24 scenes x 3 views. A scene is three panoramas of
one room (Matterport3D, Stanford2D3DS) or one street segment (PanoCity), each turned to a fixed yaw.
Every camera cell of a scene sees the same frames and yaws; only the lens changes.
"""
import math

DATASETS = ("M3D", "S2D3DS", "PanoCity")
DATASET_NAMES = {"M3D": "Matterport3D", "S2D3DS": "Stanford2D3DS", "PanoCity": "PanoCity"}
DOMAINS = (("Indoor", ("M3D", "S2D3DS")), ("Outdoor", ("PanoCity",)))
N_SCENES = 24
N_VIEWS = 3

# camera model -> field of view (degrees) at its benchmark operating point, ordered by field of view
CAMERA_FOV = {"pinhole": 90.0, "brown_conrady": 90.0, "fov": 150.0, "equidistant": 180.0,
              "stereographic": 180.0, "erp_crop": 180.0, "mei": 180.0, "kb4": 200.0, "equisolid": 200.0,
              "double_sphere": 200.0, "eucm": 200.0, "erp": 360.0, "cylindrical": 360.0}
CAMERA_KINDS = tuple(CAMERA_FOV)
# the eight camera models of the main tables (the models used in training)
MAIN_CAMERAS = ("pinhole", "brown_conrady", "equidistant", "erp_crop", "mei", "kb4", "double_sphere", "erp")
CAMERA_LABELS = {"pinhole": "Pinhole", "brown_conrady": "Brown-Conrady", "fov": "FOV/ATAN",
                 "equidistant": "Equidistant", "stereographic": "Stereographic", "erp_crop": "ERP-crop",
                 "mei": "MEI", "kb4": "KB4", "equisolid": "Equisolid", "double_sphere": "Double sphere",
                 "eucm": "EUCM", "erp": "ERP", "cylindrical": "Cylindrical"}
PANORAMIC = ("erp", "cylindrical")           # 2:1 sensors covering 360 degrees of azimuth

# source panoramas (the dataset loaders' fixed resolution)
ERP_H, ERP_W = 518, 1036

# evaluation render: every camera at the same angular pixel density
PX_PER_DEG = 294.0 / 90.0
PATCH = 14


def native_size(kind):
    """(W, H) of the image the evaluated model receives for a camera cell: 360 x 180 degrees (2:1)
    for erp / cylindrical, field of view x field of view otherwise, at PX_PER_DEG, each side
    floored to a multiple of PATCH."""
    fov = CAMERA_FOV[kind]
    if kind in PANORAMIC:
        w0, h0 = int(round(PX_PER_DEG * 360)), int(round(PX_PER_DEG * 180))
    else:
        w0 = h0 = int(round(PX_PER_DEG * fov))
    return max(PATCH, (w0 // PATCH) * PATCH), max(PATCH, (h0 // PATCH) * PATCH)


# benchmark construction: each view is round-tripped through its camera (source panorama -> native
# sensor -> back onto the sphere) on a sensor with the same pixels per steradian for every model,
# 180 degrees on 294 x 294 pixels being the reference
REF_SENSOR, REF_FOV = 294.0, 180.0


def solid_angle(kind):
    fov = CAMERA_FOV[kind]
    if kind == "erp":
        return 4 * math.pi
    if kind == "cylindrical":                 # 360 x 90 degrees
        return 4 * math.pi * math.sin(math.radians(45.0))
    if kind == "erp_crop":                    # fov x 90 degrees
        return 4 * math.radians(fov / 2) * math.sin(math.radians(45.0))
    return 2 * math.pi * (1 - math.cos(math.radians(fov / 2)))


PX_PER_SR = (math.pi / 4 * REF_SENSOR ** 2) / (2 * math.pi * (1 - math.cos(math.radians(REF_FOV / 2))))


def roundtrip_sensor(kind):
    """(W, H) of the sensor a view is round-tripped through when the benchmark is built."""
    px = PX_PER_SR * solid_angle(kind)
    if kind in PANORAMIC:
        h = int(round(math.sqrt(px / 2) / 2) * 2)
        return 2 * h, h
    s = int(round(math.sqrt(px / (math.pi / 4)) / 2) * 2)
    return s, s


# metrics: PanoVGGT's eval_pointcloud with its per-dataset ICP configuration
ICP_CONFIG = {"M3D": {"icp_threshold": 0.1, "normal_radius": 0.2},
              "S2D3DS": {"icp_threshold": 0.1, "normal_radius": 0.2},
              "PanoCity": {"icp_threshold": 1.0, "normal_radius": 2.0}}
MAX_EVAL_POINTS = 40000          # fixed RandomState(0) subset of the masked pixels handed to eval_pointcloud
FIT_MAX_POINTS = 200000          # RandomState(0) subset for the Sim(3) placement of the fused cloud
POSE_TRANS_NORM_THRESH = 1e-2    # view pairs with a smaller ground-truth baseline are left out of the pose metrics
MIN_GT_RANGE = 0.05              # pixels whose ground-truth range is below this are not scored

# dataset loaders (PanoVGGT's): file layout, depth encoding and the depth cut-off of the validity mask
LOADERS = {
    "M3D": dict(cache="cache/matterport3d_test_index.json", split="test_final", depth_scale=4000.0,
                depth_max=10.0, depth_min=0.1,
                pose="<house>/pano_poses/<view>.txt: 4x4 camera-to-world, Matterport axes"),
    "S2D3DS": dict(cache="cache/2d3ds_test_index.json", split="test_final (area_5b)", depth_scale=512.0,
                   depth_max=10.0, depth_min=0.1,
                   pose="area_5b/pano/pose/camera_<id>_<room>_frame_equirectangular_domain_pose.json: "
                        "camera_rt_matrix, 3x4 world-to-camera, Stanford axes"),
    "PanoCity": dict(cache="cache/PanoCity_test_index.json", split="test_final", depth_scale=100.0,
                     depth_max=240.0, depth_min=0.1,
                     pose="<city>/<city>_block<N>/<city>_Pano_block<N>_poses.json: frames[].transformation_matrix, "
                          "4x4 camera-to-world, OpenCV axes"),
}


def protocol_record():
    return dict(
        n_scenes=N_SCENES, n_views=N_VIEWS, source_panorama=[ERP_H, ERP_W],
        render=dict(px_per_deg=PX_PER_DEG, patch=PATCH,
                    rule="erp and cylindrical: W = 360 deg, H = 180 deg; other cameras square, side = fov; "
                         "at px_per_deg, each side floored to a multiple of patch",
                    camera="any3r.cameras.from_fov(kind, W, H, fov, principal_point=((W-1)/2, (H-1)/2), "
                           "circular=True)",
                    gt_mask="lens valid & source mask (nearest) & source range (bilinear) > 0.05 m"),
        roundtrip=dict(ref_sensor=REF_SENSOR, ref_fov=REF_FOV, px_per_sr=PX_PER_SR,
                       rule="sensor with px_per_sr pixels per steradian of the camera's solid angle; "
                            "2:1 for erp and cylindrical, square otherwise; even sides"),
        metrics=dict(icp=ICP_CONFIG, max_eval_points=MAX_EVAL_POINTS, fit_max_points=FIT_MAX_POINTS,
                     pose_trans_norm_thresh=POSE_TRANS_NORM_THRESH,
                     acc="cell mean over scenes of PanoVGGT eval_pointcloud Acc-mean",
                     auc="AUC over all view pairs of the cell of max(rotation, translation-direction error), "
                         "1-degree bins"))


def camera_record(kind):
    W, H = native_size(kind)
    Wr, Hr = roundtrip_sensor(kind)
    return dict(fov_deg=CAMERA_FOV[kind], label=CAMERA_LABELS[kind], main=kind in MAIN_CAMERAS,
                native_size=[W, H], roundtrip_sensor=[Wr, Hr], principal_point="((W-1)/2, (H-1)/2)",
                circular=True)
