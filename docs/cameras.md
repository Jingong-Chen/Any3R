# Describing cameras

Any3R does not need to know a lens type. It needs, for every view, the **ray field**: the unit direction of every pixel in the camera frame. `any3r.cameras` computes ray fields from the calibrations people usually have. You can also pass a ray field you computed yourself.

## Conventions

| | |
|---|---|
| Camera frame | OpenCV: +x right, +y down, +z along the optical axis |
| Pixel coordinates | COLMAP: the top-left corner of the image is (0, 0), so the centre of pixel (row *i*, column *j*) is (*j* + 0.5, *i* + 0.5) |
| Outside the lens | pixels that no ray reaches (the corners of a circular fisheye) are marked invalid, get the ray (0, 0, 1) and are blacked out in the image, which is how Any3R saw them in training |
| Output poses | camera-to-world, in the frame of the first view; the scene scale is arbitrary |

OpenCV calibrations place pixel centres at integer coordinates; `from_opencv` adds the half pixel for you.

## Four ways to give a camera

### 1. A calibration you already have

```python
from any3r import cameras

# COLMAP model name + parameter list (the same convention as COLMAP, no conversion needed)
cam = cameras.from_colmap("OPENCV_FISHEYE", 3264, 3264, [fx, fy, cx, cy, k1, k2, k3, k4], fov_deg=213.6)

# a whole COLMAP sparse model: {camera_id: CameraModel}
cams = cameras.read_colmap_cameras("my_scene/sparse/0")

# an OpenCV calibration (cv2.calibrateCamera or cv2.fisheye.calibrate)
cam = cameras.from_opencv(K, dist, width, height)                 # radial-tangential / rational
cam = cameras.from_opencv(K, D, width, height, fisheye=True)      # Kannala-Brandt

# or construct a model directly
cam = cameras.DoubleSphere(fx, fy, cx, cy, xi=-0.2, alpha=0.6, width=1280, height=1024)
```

For fisheye calibrations, pass `fov_deg` if the polynomial is only valid inside the image circle: pixels beyond that half-angle are then treated as outside the lens.

### 2. Only the lens type and field of view

```python
cam = cameras.from_fov("equidistant", width, height, fov_deg=190)   # a 190-degree circular fisheye
cam = cameras.from_fov("pinhole", width, height, fov_deg=60)
```

`fov_deg` is the field of view across the image **height**. Fisheye models default to `circular=True` (the image circle touches the top and bottom edges); rectilinear models use the full rectangle.

### 3. A JSON file (for the demos)

```json
{"camera": {"model": "OPENCV_FISHEYE", "width": 3264, "height": 3264,
            "params": [fx, fy, cx, cy, k1, k2, k3, k4], "fov_deg": 213.6}}
```

applies one camera to every image. Per-image cameras:

```json
{"cameras": {
  "front.jpg": {"model": "pinhole", "width": 1920, "height": 1080, "fx": 1100, "fy": 1100, "cx": 960, "cy": 540},
  "fisheye.jpg": {"model": "equidistant", "width": 1024, "height": 1024, "fov_deg": 190},
  "pano.jpg": {"model": "erp"}
}}
```

A camera entry is either one of the model names below with keyword parameters, a COLMAP model name with `params`, or a model name with only `fov_deg` (see `from_fov`). Rigs can share entries: `{"cameras": {"left": {...}, "right": {...}}, "images": {"0001_l.jpg": "left", ...}}`. A camera calibrated at another resolution of the same aspect ratio is rescaled to the image automatically.

### 4. Your own ray field

`Any3R.infer(images, ray_maps)` takes any `(S, H, W, 3)` tensor of unit rays, so any central camera works, including models not listed here (OCamCalib polynomials, look-up tables, learned calibrations). Set rays outside the lens to (0, 0, 1) and black out those pixels.

## Supported models

| Name | Class | Parameters | Notes |
|---|---|---|---|
| `pinhole` | `Pinhole` | fx, fy, cx, cy | COLMAP `SIMPLE_PINHOLE`, `PINHOLE` |
| `brown_conrady` | `BrownConrady` (= `OpenCV`) | fx, fy, cx, cy, k1, k2, p1, p2, k3[, k4, k5, k6] | COLMAP `SIMPLE_RADIAL`, `RADIAL`, `OPENCV`, `FULL_OPENCV`; rational model when k4..k6 are set |
| `kb4` | `KB4` (= `OpenCVFisheye`, `KannalaBrandt`) | fx, fy, cx, cy, k1..k4 | COLMAP `OPENCV_FISHEYE`, `SIMPLE_RADIAL_FISHEYE`, `RADIAL_FISHEYE`; beyond 180 degrees |
| `equidistant` | `Equidistant` | fx, fy, cx, cy | r = f θ |
| `equisolid` | `Equisolid` | fx, fy, cx, cy | r = 2f sin(θ/2) |
| `stereographic` | `Stereographic` | fx, fy, cx, cy | r = 2f tan(θ/2) |
| `fov` | `FOV` | fx, fy, cx, cy, w | Devernay–Faugeras; COLMAP `FOV` |
| `mei` | `MEI` (= `UnifiedCamera`) | fx, fy, cx, cy, xi | unified omnidirectional model |
| `double_sphere` | `DoubleSphere` | fx, fy, cx, cy, xi, alpha | Usenko et al. |
| `eucm` | `EUCM` | fx, fy, cx, cy, alpha, beta | extended unified model |
| `erp` | `ERP` (= `Equirectangular`) | — | 360 × 180 degree panorama |
| `erp_crop` | `ERPCrop` | az_max, el_max (radians) | equirectangular window around the optical axis |
| `cylindrical` | `Cylindrical` | v_fov_deg | 360-degree strip panorama |

Every model offers `ray_map(height, width)`, `project(rays)`, `resized(width, height)` and `roundtrip_error()` (largest pixel error of project ∘ unproject over valid pixels; below 1e-4 px for every model). Iterative inversions (Brown–Conrady, Kannala–Brandt) check convergence and mark pixels that do not invert as outside the lens.

## Mixing cameras

Views in one call can come from different cameras and have different sizes. `any3r.io.prepare_views` downscales views larger than `size`, centres every view on one canvas (the largest view, rounded up to a multiple of 16) and fills the margins with black pixels and (0, 0, 1) rays. Nothing else is needed.
