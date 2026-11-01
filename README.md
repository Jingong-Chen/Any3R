<div align="center">

# Any3R: Feedforward 3D Reconstruction from Any Camera Type

**Jingong Chen**<sup>1,2</sup> · **Qingwen Zhang**<sup>3†</sup> · **Youming Deng**<sup>4†</sup> · **Chulwoo Pack**<sup>1</sup> · **Olaf Wysocki**<sup>5</sup> · **Boris Jutzi**<sup>6</sup> · **Kwanghee Won**<sup>1*</sup> · **Kyle Gao**<sup>2</sup>

<sup>1</sup>South Dakota State University &nbsp; <sup>2</sup>Aalto University &nbsp; <sup>3</sup>KTH Royal Institute of Technology<br>
<sup>4</sup>Cornell University &nbsp; <sup>5</sup>University of Cambridge &nbsp; <sup>6</sup>Karlsruhe Institute of Technology<br>
<sup>†</sup>Equal contribution &nbsp; <sup>*</sup>Corresponding author

<a href="https://jingong-chen.github.io/Any3R/"><img src="https://img.shields.io/badge/Project-Page-4b6bfb?logo=googlechrome&logoColor=white" alt="Project Page"></a> <a href="https://arxiv.org/abs/2610.36150"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a> <a href="https://huggingface.co/Jingong/Any3R"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-ffc107" alt="Hugging Face"></a> <a href="LICENSE"><img src="https://img.shields.io/badge/License-FAIR%20Noncommercial-6c757d" alt="License"></a>

<img src="assets/teaser.gif" width="92%" alt="Any3R reconstructs a street from twenty views taken with eight different camera models">

</div>

**Any3R** reconstructs a scene from images taken with **any central camera**, in a single forward pass: pinhole and distorted lenses, fisheyes beyond 180°, 360° panoramas, and any mix of them in one scene. Each view is described by its **ray field**, the direction of every pixel, instead of a lens type, so one set of weights covers every lens with no undistortion and no retraining. Any3R predicts per-view depth along each ray, camera poses and a fused point cloud.

## News

- **2026-10-31** · Inference code, model weights, interactive demo and evaluation code are released.

## Contents

[Highlights](#highlights) · [Installation](#installation) · [Quick start](#quick-start) · [Cameras](#describing-cameras) · [Demos](#demos) · [Model](#model) · [Results](#results) · [Evaluation](#evaluation) · [Limitations](#limitations) · [Citation](#citation)

## Highlights

- **One model for every central camera.** Thirteen camera models are built in (pinhole, Brown–Conrady, Kannala–Brandt, equidistant, equisolid, stereographic, FOV, MEI, double sphere, EUCM, cylindrical, equirectangular and equirectangular windows), and any other central camera works through its ray field.
- **Mixed cameras in one scene.** Views from different cameras, with different sizes and fields of view, are reconstructed together into one consistent point cloud.
- **Calibrations as you have them.** COLMAP models, OpenCV calibrations, a JSON file, or just "a 190° fisheye".
- **Fast.** 32 views at 512 × 512 in 1.1 s and 128 views in 9 s (17.7 GB) on one RTX 5090.

<p align="center"><img src="assets/method.png" width="96%" alt="Any3R architecture"></p>

The ray field enters the network in two places. *Angular coordinates* give the rotary position embedding the measured elevation and azimuth of every patch, per view, in place of its row and column; *solid-angle weighting* scales the training loss by the area each pixel covers on the sphere. Neither depends on the lens.

## Installation

```bash
git clone https://github.com/Jingong-Chen/Any3R.git
cd Any3R
conda create -n any3r python=3.11 -y && conda activate any3r
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the build for your CUDA
pip install -e .              # the any3r package
pip install -e ".[demo]"      # optional: Gradio demo and GLB export
```

The weights (4.6 GB) are downloaded from Hugging Face on first use. A GPU with bfloat16 support is recommended; inference also runs in float16 on older GPUs and on the CPU.

## Quick start

```python
import torch
from any3r import Any3R, cameras
from any3r.io import prepare_views

model = Any3R.from_pretrained("Jingong/Any3R").to("cuda").eval()

images = ["phone.jpg", "fisheye.jpg", "panorama.jpg"]
cams = [
    cameras.from_fov("pinhole", 4032, 3024, fov_deg=55),                 # any calibration works, see below
    cameras.from_colmap("OPENCV_FISHEYE", 1920, 1920, [fx, fy, cx, cy, k1, k2, k3, k4]),
    cameras.ERP(),                                                        # equirectangular 360° panorama
]
views = prepare_views(images, cams, size=512)    # resize, compute ray maps, put the views on one canvas

pred = model.infer(views.images, views.rays)     # bfloat16 autocast inside
pred["depth"]          # (S, H, W, 1)  distance along each pixel's ray
pred["depth_conf"]     # (S, H, W)     confidence
pred["camera_poses"]   # (S, 4, 4)     camera-to-world, in the frame of the first view
pred["world_points"]   # (S, H, W, 3)  fused points; the scale is arbitrary
```

`model.infer` takes the images `(S, 3, H, W)` in [0, 1] and the ray maps `(S, H, W, 3)`, so you can also pass ray fields computed by your own code. `any3r.export.save_predictions(out_dir, pred, views)` writes a PLY point cloud, a GLB scene with the cameras, the poses as JSON and the depth maps.

## Describing cameras

Any3R needs the ray field of every view, which `any3r.cameras` computes from a calibration:

| You have | Use |
|:---|:---|
| a COLMAP sparse model | `cameras.read_colmap_cameras("sparse/0")` or `demo.py --colmap sparse/0` |
| an OpenCV calibration | `cameras.from_opencv(K, dist, W, H)`, with `fisheye=True` for `cv2.fisheye` |
| model parameters | `cameras.KB4(fx, fy, cx, cy, k1, k2, k3, k4)`, `cameras.DoubleSphere(...)`, ... |
| only the lens type and field of view | `cameras.from_fov("equidistant", W, H, fov_deg=190)` |
| an equirectangular panorama | `cameras.ERP()` |
| your own ray directions | pass them to `model.infer` directly |

Conventions, the JSON format used by the demos and the parameters of every model are in **[docs/cameras.md](docs/cameras.md)**.

## Demos

**Command line.** Three example scenes ship with the code:

```bash
# 20 views of a street through 8 different camera models (pinhole to 360°)
python demo.py --images examples/street_mixed/images --cameras examples/street_mixed/cameras.json --out outputs/street

# a real 214° fisheye with its COLMAP calibration (FIORD)
python demo.py --images examples/fiord_meetingroom/images --cameras examples/fiord_meetingroom/cameras.json --out outputs/fiord

# eight equirectangular panoramas
python demo.py --images examples/panocity_erp/images --camera erp --out outputs/panos

# your own data
python demo.py --images my_scene/images --colmap my_scene/sparse/0
python demo.py --images my_fisheye_folder --camera equidistant --fov 190
```

Each run writes `points.ply`, `scene.glb` (points and cameras), `cameras.json` (poses and calibrations) and `depth.npz`.

**Interactive.** `python demo_gradio.py` opens a browser app: upload images, pick a camera model or upload a calibration, and inspect the point cloud, cameras and depth maps. The examples load with one click.

## Model

| Model | Parameters | Input | Weights |
|:---|:---:|:---|:---|
| Any3R | 1.15 B | any number of views from any central cameras, mixed freely | [🤗 Jingong/Any3R](https://huggingface.co/Jingong/Any3R) |

Any3R is fine-tuned from VGGT-Omega for 40 epochs on Matterport3D, Stanford2D3DS, Structured3D, PanoCity and DL3DV videos. Half of the panoramic training samples are native panoramas and half are views resampled through a random camera, drawn from eight central models (pinhole, Brown–Conrady, equidistant, Kannala–Brandt, MEI, double sphere, equirectangular and cropped equirectangular) with fields of view from 30° to 360°. All test scenes are held out.

## Results

Reconstruction accuracy on the paper's benchmark: three datasets, each scene imaged through the same views by every camera model, and every method fuses its points with **its own predicted camera poses**. Lower is better; the best value of each column is in bold. Pose accuracy (AUC), five more camera models and zero-shot results on real fisheye lenses (ScanNet++, FIORD) are in the paper and on the [project page](https://jingong-chen.github.io/Any3R/).

<b>Indoor (Matterport3D + Stanford2D3DS)</b> · Acc ↓ (m)

| Method | Pinhole 90° | B–C 90° | Equi. 180° | ERP-crop 180° | MEI 180° | KB4 200° | DS 200° | ERP 360° |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OmniVGGT | 0.250 | 0.201 | 0.326 | 0.318 | 0.329 | 0.355 | 0.350 | 0.406 |
| Wid3R | 0.144 | -- | 0.141 | -- | 0.146 | -- | -- | 0.101 |
| PanoVGGT | 0.282 | 0.272 | 0.259 | 0.348 | 0.265 | 0.251 | 0.265 | 0.072 |
| MapA-ray | 0.092 | 0.146 | 0.277 | 0.254 | 0.239 | 0.274 | 0.258 | 0.405 |
| π³ | 0.084 | 0.119 | 0.171 | 0.287 | 0.164 | 0.189 | 0.183 | 0.414 |
| MapAnything | 0.092 | 0.149 | 0.273 | 0.269 | 0.267 | 0.283 | 0.294 | 0.386 |
| VGGT-Ω | 0.149 | 0.161 | 0.252 | 0.265 | 0.250 | 0.281 | 0.302 | 0.403 |
| **Any3R** | **0.076** | **0.091** | **0.058** | **0.052** | **0.059** | **0.056** | **0.060** | **0.054** |

<b>Outdoor (PanoCity)</b> · Acc ↓ (m)

| Method | Pinhole 90° | B–C 90° | Equi. 180° | ERP-crop 180° | MEI 180° | KB4 200° | DS 200° | ERP 360° |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OmniVGGT | 2.875 | 2.676 | 3.313 | 2.804 | 3.703 | 3.941 | 3.670 | 1.479 |
| Wid3R | 2.278 | -- | 2.121 | -- | 2.252 | -- | -- | 2.330 |
| PanoVGGT | 2.256 | 2.708 | 2.498 | 3.633 | 2.433 | 2.405 | 2.473 | 0.751 |
| MapA-ray | 1.484 | 1.558 | 2.368 | 2.686 | 2.029 | 2.523 | 2.579 | 1.894 |
| π³ | 1.479 | 1.711 | 2.093 | 2.426 | 2.036 | 2.394 | 2.271 | 1.628 |
| MapAnything | 1.485 | 1.633 | 2.066 | 2.861 | 2.236 | 3.297 | 2.789 | 1.833 |
| VGGT-Ω | 1.711 | 1.572 | 2.162 | 2.455 | 2.490 | 2.680 | 2.544 | 1.876 |
| **Any3R** | **1.160** | **1.243** | **0.866** | **0.811** | **0.854** | **0.863** | **0.901** | **0.663** |

Wid3R receives the true ray field and supports the cameras it is defined for (-- otherwise). MapA-ray is MapAnything with the true ray field fed to its ray encoder.

## Evaluation

[`eval/`](eval/README.md) reproduces the paper's benchmark end to end: it rebuilds the 936 scene files from the original datasets (every array is checked against a frozen manifest), renders each view through its camera model, runs Any3R and computes the paper's metrics (Acc, Comp, NC, and pose AUC with predicted poses).

```bash
pip install -r eval/requirements.txt
python eval/build_benchmark.py --out data/any3r_bench --m3d-root ... --s2d3ds-root ... --panocity-root ...
python eval/evaluate.py --benchmark data/any3r_bench --out results/any3r --verify
python eval/make_table.py --results results/any3r --compare-paper
```

On the paper's configuration (A100, PyTorch 2.5.1) the released code and weights reproduce every per-scene result of the paper exactly; on other GPUs bfloat16 kernels shift the numbers slightly (on an RTX 5090, at most 0.004 m Acc and 0.7 AUC points in the main table). The full benchmark takes about 5 minutes on one RTX 5090.

## Limitations

- **Central cameras.** Every pixel's ray must pass through one centre. Non-central optics (rolling-shutter motion, push-broom sensors, strongly non-central catadioptric systems) are outside the model.
- **A calibration is needed.** Any3R uses the ray field it is given and does not estimate intrinsics. Without a calibration, `from_fov` with the nominal lens type and field of view is a reasonable start.
- **Scale and dynamics.** Reconstructions have an arbitrary scale, and the scene is assumed static.

## Citation

```bibtex
@article{chen2026any3r,
  title   = {Any3R: Feedforward 3D Reconstruction from Any Camera Type},
  author  = {Chen, Jingong and Zhang, Qingwen and Deng, Youming and Pack, Chulwoo and
             Wysocki, Olaf and Jutzi, Boris and Won, Kwanghee and Gao, Kyle},
  journal = {arXiv preprint arXiv:2610.36150},
  year    = {2026}
}
```

## Acknowledgments

Any3R builds on [VGGT-Omega](https://github.com/facebookresearch/vggt) and [PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT); we thank their authors for releasing code and weights. The example scenes come from [PanoCity](https://huggingface.co/datasets/YijingGuo/PanoCity) and [FIORD](https://zenodo.org/records/15181976).

This work was supported by the U.S. National Science Foundation under Grant No. 2316400 and by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. The computations and data handling were enabled by the Arrhenius HPC resource provided by the National Academic Infrastructure for Supercomputing in Sweden, and by the South Dakota State University High Performance Computing (SDSU HPC) team.

## License

The code and the weights are released under the [FAIR Noncommercial Research License](LICENSE), inherited from VGGT-Omega, for noncommercial research use. Parts adapted from PanoVGGT are under the MIT License, and the example images keep their dataset licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
