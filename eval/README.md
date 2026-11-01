# Evaluation: the any-camera benchmark

This directory reproduces the predicted-pose geometry benchmark of the paper (the comparison tables on the
eight training cameras and the 13-camera tables of the appendix) for Any3R:

```
eval/
  prepare_matterport.py     Matterport3D panoramas (official mp360 task data) -> the layout the builder reads
  build_benchmark.py        build the benchmark from the original datasets and check it against the manifest
  evaluate.py               run Any3R on every cell, score it, compare with the paper (--verify)
  make_table.py             the paper's tables from the results
  benchmark/manifest.json   the frozen benchmark: frames, source-file md5s, yaws, extrinsics, md5 of every array
  benchmark/paper_any3r.json  the paper's Any3R results, per scene (reference for --verify)
  benchlib/                 protocol constants, dataset readers, rendering, metrics
  tests/                    metric checks on synthetic data, manifest consistency, parity with the original code
```

No dataset image, depth map or pose is shipped. Matterport3D and Stanford2D3DS do not allow
redistribution; PanoCity (CC BY 4.0) is treated the same way for consistency. The manifest only names
the files and freezes their md5 and the md5 of everything derived from them.

## The benchmark

* **Datasets.** Matterport3D (indoor), Stanford2D3DS (indoor, area 5b) and PanoCity (outdoor), the
  test splits of PanoVGGT. 24 scenes per dataset; a scene is three panoramas of one room or one
  street segment, drawn without replacement. Each panorama is turned to a fixed yaw (one of 12 steps
  of 30°), chosen once for all cameras to maximise the overlap of the three views as seen by the
  narrowest camera, a 90° pinhole (the weakest link of their maximum spanning tree of covisibility).
* **Cameras.** 13 central camera models; the eight marked * are those of the main tables. Every view
  of a scene is seen through every model, so the cells of a scene differ only by the lens.

  | model | FoV | input (W x H) | | model | FoV | input (W x H) |
  |---|:-:|:-:|---|---|:-:|:-:|
  | pinhole * | 90° | 294 x 294 | | KB4 * | 200° | 644 x 644 |
  | Brown-Conrady * | 90° | 294 x 294 | | equisolid | 200° | 644 x 644 |
  | FOV (ATAN) | 150° | 490 x 490 | | double sphere * | 200° | 644 x 644 |
  | equidistant * | 180° | 588 x 588 | | EUCM | 200° | 644 x 644 |
  | stereographic | 180° | 588 x 588 | | ERP * | 360° | 1176 x 588 |
  | ERP-crop * | 180° x 90° | 588 x 588 | | cylindrical | 360° x 90° | 1176 x 588 |
  | MEI * | 180° | 588 x 588 | | | | |

  The camera of a cell is `any3r.cameras.from_fov(kind, W, H, fov, principal_point=((W-1)/2, (H-1)/2),
  circular=True)` (lens parameters in `any3r/cameras.py`); every model is rendered at the same
  angular density, 294 px per 90°.
* **Construction** (`build_benchmark.py`). The three panoramas are read at 518 x 1036 with PanoVGGT's
  loader rules (RGB resized with INTER_AREA, depth with INTER_NEAREST and cut at 10 m indoors and
  240 m outdoors). For each camera model, each view is round-tripped through the lens on a sensor
  with the same number of pixels per steradian for every model (source panorama -> native image ->
  back onto the panorama through the model's own projection) and stored as 8-bit RGB. The lens's
  distortion and the resampling it implies survive the trip, so lenses with the same field of view do
  not hand the model the same pixels.
* **Evaluation** (`evaluate.py`). Each view is rendered from its scene file through the cell's camera
  at the size above; Any3R receives the three images and their ray maps. A pixel is scored when it
  lies inside the lens and its ground-truth range exceeds 5 cm.
* **Metrics.** Any3R's own depths and poses from one forward pass give the fused cloud. One
  similarity (Umeyama on up to 200k pixel correspondences with the ground truth) places it, and
  PanoVGGT's `eval_pointcloud` scores a fixed subset of 40,000 correspondences: Umeyama, rigid
  point-to-point ICP (0.1 m indoors, 1.0 m on PanoCity; normals within 0.2 m / 2.0 m), then
  accuracy (Acc, prediction to ground truth), completion (Comp) and normal consistency (NC). Pose:
  for the three view pairs, the angle of the relative rotation error and of the relative translation
  direction error (pairs with a ground-truth baseline of at most 1 cm are left out); AUC@15 and AUC@30
  of their maximum in 1° bins over all view pairs of a cell. In the tables, Acc is the mean over the
  24 scenes of the per-scene Acc, Indoor is the mean of the Matterport3D and Stanford2D3DS cell values
  and Outdoor is PanoCity.

## Setup

```bash
pip install -e .                       # the any3r package (from the repository root)
pip install -r eval/requirements.txt   # numpy, scipy, open3d 0.19.0, opencv-python, pytest
```

Commands below are run from the repository root. Any3R needs about 8 GB of GPU memory here (7.3 GiB
allocated at peak, on the 1176 x 588 panoramas); rendering and scoring run in CPU processes
(`--workers`).

## 1. Datasets

Obtain the datasets from their owners and keep the directory layout read by PanoVGGT's loaders.
Only the 216 files per dataset named in the manifest are opened, and their md5 is checked before use.

| dataset | where | root argument | files per frame |
|---|---|---|---|
| Matterport3D | [niessner.github.io/Matterport](https://niessner.github.io/Matterport/) (terms of use): the `mp360` task data of the official download script, converted with `eval/prepare_matterport.py` (below) | `--m3d-root` | `<house>/pano_skybox_color/<view>.png` (H x W = 1024 x 2048), `<house>/pano_depth/<view>.png` (uint16, 1/4000 m), `<house>/pano_poses/<view>.txt` (4x4 camera-to-world) |
| Stanford2D3DS | [buildingparser.stanford.edu](http://buildingparser.stanford.edu/dataset.html) (terms of use), the `no_xyz` release | `--s2d3ds-root` | `area_5b/pano/{rgb,depth,pose}/camera_<id>_<room>_frame_equirectangular_domain_{rgb.png,depth.png,pose.json}` (H x W = 2048 x 4096, depth uint16 1/512 m) |
| PanoCity | [huggingface.co/datasets/YijingGuo/PanoCity](https://huggingface.co/datasets/YijingGuo/PanoCity) (CC BY 4.0) | `--panocity-root` | `<city>/<city>_block<N>/pano_images/pano_<k>.png`, `.../panodepth_images/pano_depth_<k>.png` (uint16, cm), `<city>/<city>_block<N>/<city>_Pano_block<N>_poses.json` |

The exact relative path and md5 of every file are in `benchmark/manifest.json`
(`datasets.<DS>.scenes[].frames[].files`).

Matterport3D itself has no equirectangular panoramas. The benchmark uses those stitched by the
authors of 360MonoDepth and distributed with Matterport3D (`download_mp.py --task_data mp360`, seven
archives `data_00.zip` ... `data_06.zip`, about 82 GB). Convert the benchmark's panoramas (or all of
them with `--all`) to the layout above:

```bash
python eval/prepare_matterport.py --mp360 /data/matterport/v1/tasks/mp3d_360 --out /data/matterport3d
```

## 2. Build the benchmark

```bash
python eval/build_benchmark.py --out data/any3r_bench --workers 8 \
    --m3d-root /data/matterport3d --s2d3ds-root /data/stanford2d3ds/no_xyz --panocity-root /data/panocity
```

This writes `data/any3r_bench/{M3D,S2D3DS,PanoCity}/PP13_<camera>/scene_XXX.npz` (936 files, about
7 GB) and `build_report.json`. Every array is checked against the manifest md5 (2,160 checks: the
loader extrinsics, the ground-truth range, mask and extrinsics of every scene, and the round-tripped
image and support of every camera cell), and the command exits with status 1 if any differs. The 648
dataset files are compared with the reference copies too; a file can differ in its bytes and still
decode to the same arrays (a re-encoded PNG), which is reported but not fatal (`--strict-sources` makes
it fatal). About 2 minutes with 8 processes. A dataset can be built alone (`--datasets PanoCity`);
`--out` can be omitted to check only.

The benchmark is rebuilt bit for bit from the original files: on two machines (an AMD workstation
with Python 3.12 / numpy 2.4 and a cluster node with Python 3.10 / numpy 2.2, both OpenCV 4.11) all
2,160 array checks pass for all three datasets, with all 648 dataset files identical to the
reference copies. If the arrays differ (another release or preparation of a dataset), the numbers are
not comparable with the paper.

## 3. Evaluate

```bash
python eval/evaluate.py --benchmark data/any3r_bench --out results/any3r --workers 8
python eval/evaluate.py --benchmark data/any3r_bench --out results/any3r --cameras main   # the 8 main cameras
python eval/evaluate.py --benchmark data/any3r_bench --out results/any3r --datasets S2D3DS --cameras pinhole,erp
```

`--weights` takes a Hugging Face repo id (default `Jingong/Any3R`) or a local directory. Each
scene gives `results/any3r/<DS>/<camera>/scene_XXX.json` (all eval_pointcloud metrics and the
per-pair pose errors); finished scenes are skipped when the command is rerun, and each complete cell
gets `results/any3r/<DS>/<camera>.json` with its summary. The scene files are checked against the
manifest before use (`--no-check` skips this). `--save-predictions` keeps Any3R's depths and poses;
`--predictions DIR` scores stored predictions instead of running the model. The 936 scenes take
about 5 minutes on an RTX 5090 with 10 workers and 7 minutes on an A100 with 7 workers.

The evaluation runs Open3D single-threaded (`OMP_NUM_THREADS=1`, set by the script unless already
set): its ICP and normal estimation are multi-threaded and not bitwise reproducible otherwise.

## 4. Tables

```bash
python eval/make_table.py --results results/any3r                   # main table + 13 cameras per dataset
python eval/make_table.py --results results/any3r --compare-paper   # the paper's values in parentheses
python eval/make_table.py --paper                                   # the paper's values alone
```

## 5. Comparison with the paper

```bash
python eval/evaluate.py --out results/any3r --verify
```

compares every finished scene with `benchmark/paper_any3r.json` and prints, per cell, the largest
absolute difference of each per-scene metric and pose error and of the cell's Acc / AUC values
(`verify_report.json`; `--tol`, default 1e-6).

**What agreement to expect.** Any3R runs in bfloat16, whose GPU kernels differ between GPU
architectures and PyTorch builds. The metrics are a deterministic function of the predictions.

* On the paper's configuration (NVIDIA A100 80GB, PyTorch 2.5.1+cu121, numpy 2.2.6, Open3D 0.19.0)
  this code reproduces the paper's numbers exactly: for all 936 scenes Any3R's depths and poses are
  bit-identical to those of the paper's run, and every per-scene Acc, Comp, NC and pose error and
  every cell value is identical (`--verify --tol 0` passes).
* Scoring the paper run's own predictions with this code on another machine (RTX 5090 workstation,
  numpy 2.4.4) agrees to 3e-13 on Acc and Comp, 1e-11 degrees on pose errors, exactly on AUC; NC
  agrees to 4e-5 (normal estimation reacts to last-bit differences of its input between CPUs and
  library builds).
* Running the model on another GPU changes the predictions slightly (RTX 5090 with PyTorch 2.11:
  depth by ~0.1% (median), rotations by up to ~0.5°, more on the few scenes whose pose is unstable).
  Per-scene Acc then moves by 0.3% (median; 7% at the 99th percentile). In the main table 31 of the
  48 printed values change, by at most 0.004 m in Acc and 0.7 points in AUC.

## Expected results (paper)

Main table, Any3R:

| Indoor | Pin. 90° | B-C 90° | Equi. 180° | ERP-C 180° | MEI 180° | KB4 200° | DS 200° | ERP 360° |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Acc (m) | 0.076 | 0.091 | 0.058 | 0.052 | 0.059 | 0.056 | 0.060 | 0.054 |
| AUC@15 (%) | 74.8 | 64.8 | 90.6 | 86.0 | 90.5 | 81.2 | 80.5 | 94.5 |
| AUC@30 (%) | 85.0 | 78.3 | 93.9 | 91.6 | 94.6 | 89.6 | 88.5 | 96.6 |

| Outdoor | Pin. 90° | B-C 90° | Equi. 180° | ERP-C 180° | MEI 180° | KB4 200° | DS 200° | ERP 360° |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Acc (m) | 1.160 | 1.243 | 0.866 | 0.811 | 0.854 | 0.863 | 0.901 | 0.663 |
| AUC@15 (%) | 61.5 | 51.0 | 85.0 | 87.4 | 85.2 | 79.5 | 81.0 | 97.2 |
| AUC@30 (%) | 80.7 | 74.2 | 92.5 | 93.7 | 92.6 | 89.4 | 90.5 | 98.6 |

Every cell (Acc in m / AUC@15 / AUC@30 in %; * = main tables):

| Camera (FoV) | Matterport3D | Stanford2D3DS | PanoCity |
|---|:-:|:-:|:-:|
| Pinhole (90°) * | 0.0690 / 75.6 / 86.4 | 0.0827 / 73.9 / 83.5 | 1.1604 / 61.5 / 80.7 |
| Brown-Conrady (90°) * | 0.0962 / 64.9 / 78.2 | 0.0850 / 64.6 / 78.4 | 1.2434 / 51.0 / 74.2 |
| FOV/ATAN (150°) | 0.0508 / 88.9 / 94.4 | 0.0530 / 90.6 / 93.9 | 0.9289 / 81.6 / 90.8 |
| Equidistant (180°) * | 0.0560 / 89.8 / 93.5 | 0.0609 / 91.5 / 94.4 | 0.8664 / 85.0 / 92.5 |
| Stereographic (180°) | 0.0597 / 87.0 / 92.1 | 0.0694 / 88.0 / 92.6 | 0.9170 / 81.9 / 90.9 |
| ERP-crop (180°) * | 0.0509 / 84.7 / 91.0 | 0.0529 / 87.3 / 92.3 | 0.8111 / 87.4 / 93.7 |
| MEI (180°) * | 0.0536 / 89.9 / 95.0 | 0.0638 / 91.1 / 94.2 | 0.8540 / 85.2 / 92.6 |
| KB4 (200°) * | 0.0502 / 80.6 / 89.7 | 0.0615 / 81.9 / 89.5 | 0.8633 / 79.5 / 89.4 |
| Equisolid (200°) | 0.0562 / 77.4 / 87.3 | 0.0554 / 82.7 / 91.3 | 0.8929 / 78.6 / 89.3 |
| Double sphere (200°) * | 0.0549 / 79.4 / 87.6 | 0.0654 / 81.6 / 89.4 | 0.9013 / 81.0 / 90.5 |
| EUCM (200°) | 0.0542 / 80.8 / 89.0 | 0.0671 / 81.6 / 89.4 | 0.8962 / 81.3 / 90.6 |
| ERP (360°) * | 0.0447 / 94.9 / 97.5 | 0.0641 / 94.1 / 95.6 | 0.6630 / 97.2 / 98.6 |
| Cylindrical (360°) | 0.0818 / 84.4 / 92.2 | 0.0979 / 87.0 / 93.5 | 1.2977 / 95.4 / 97.7 |

Not covered here: the baselines of the paper's tables, the mixed-camera scenes (three different
cameras in one scene) and the Structured3D rows of the appendix.

## Tests

```bash
python -m pytest eval/tests
```

`test_metrics.py` checks the metrics on synthetic scenes (exact predictions score zero, scores do not
depend on the similarity gauge of the prediction, known errors give known values); `test_manifest.py`
checks the manifest against the protocol constants. `test_research_parity.py` compares rendering and
metrics bit for bit with the original research code and runs only when `ANY3R_RESEARCH_ROOT` points
to it.

## Acknowledgements

The dataset readers and the point-cloud and pose metrics are adapted from
[PanoVGGT](https://github.com/YijingGuo-June/PanoVGGT) (MIT License; see the notices in
`benchlib/loaders.py`, `benchlib/pointcloud.py`, `benchlib/pose.py` and `THIRD_PARTY_NOTICES.md`),
whose evaluation follows CUT3R and VGGSfM.
