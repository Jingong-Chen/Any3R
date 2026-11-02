# Copyright (c) 2026 The Any3R Authors.
#
# This source code is licensed under the MIT license found in the LICENSE file in the root directory
# of this source tree. It builds the network from components under the FAIR Noncommercial Research
# License (any3r/aggregator.py and the vendored any3r/_vggt_omega, see LICENSE-FAIR), and the model
# weights are under that license too.

"""Any3R: feed-forward 3D reconstruction from views of any central camera.

The camera of every view is given as its ray field, the unit direction of every pixel in the camera
frame (any3r.cameras computes it from a calibration). The network predicts, per view, a radial
distance d along every ray and a camera-to-world pose; points are X = d * R in the camera frame.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._vggt_omega.models.heads import CameraHead, DenseHead
from ._vggt_omega.utils.rotation import quat_to_mat
from .aggregator import RayAggregator, run_trunk
from .geometry import (erp_directions_cached, homogenize_points, patch_mean_rays,
                       pose_encoding_to_c2w, raymap_to_patch_features)

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:                                    # huggingface_hub is optional for local weights
    class PyTorchModelHubMixin:                        # noqa: D401
        """Placeholder when huggingface_hub is not installed."""
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()


class Any3R(nn.Module, PyTorchModelHubMixin,
            library_name="any3r", repo_url="https://github.com/Jingong-Chen/Any3R",
            license="other", tags=["3d-reconstruction", "camera-pose-estimation", "fisheye", "panorama"]):
    def __init__(self, patch_size: int = 16, embed_dim: int = 1024, band_split: int = 8,
                 frames_chunk_size: int = 8):
        super().__init__()
        if patch_size != 16:
            raise ValueError("Any3R uses the VGGT-Omega trunk with patch size 16")
        self.patch_size = patch_size
        self.frames_chunk_size = frames_chunk_size
        self.aggregator = RayAggregator(patch_size=patch_size, embed_dim=embed_dim, band_split=band_split)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size)

    def forward(self, images: torch.Tensor, ray_maps: torch.Tensor = None) -> dict:
        """
        Args:
            images: (B, S, 3, H, W) or (S, 3, H, W), RGB in [0, 1]. Views may come from different
                cameras but share one H x W canvas (pad smaller views; see any3r.io).
            ray_maps: (B, S, H, W, 3) or (S, H, W, 3), unit rays in the camera frame (x right,
                y down, z forward). Pixels outside the lens should hold (0, 0, 1). If None, every
                view is read as an equirectangular panorama.
        Returns a dict with
            depth (B, S, H, W, 1): radial distance along each ray (scene scale is arbitrary),
            depth_conf (B, S, H, W): confidence,
            camera_poses (B, S, 4, 4): camera-to-world, in the frame of the first view,
            local_points (B, S, H, W, 3): depth * ray, in each camera's frame,
            world_points (B, S, H, W, 3): local points mapped by camera_poses,
            pose_enc (B, S, 9): raw camera-head output.
        """
        if images.dim() == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.shape
        if ray_maps is None:
            ray_maps = erp_directions_cached(H, W, images.device, torch.float32).view(1, 1, H, W, 3) \
                .expand(B, S, H, W, 3)
        elif ray_maps.dim() == 4:
            ray_maps = ray_maps.unsqueeze(0)
        if tuple(ray_maps.shape[:4]) != (B, S, H, W):
            raise ValueError(f"ray_maps {tuple(ray_maps.shape)} do not match images {tuple(images.shape)}")

        # the trunk needs multiples of the patch size; the rays stay at the input resolution
        H16, W16 = (H // self.patch_size) * self.patch_size, (W // self.patch_size) * self.patch_size
        images_in = images
        if (H16, W16) != (H, W):
            x = images.flatten(0, 1)
            images_in = torch.cat([F.interpolate(x[i:i + 4], size=(H16, W16), mode="bilinear",
                                                 align_corners=False, antialias=True)
                                   for i in range(0, x.shape[0], 4)], 0).view(B, S, -1, H16, W16)
        Hp, Wp = H16 // self.patch_size, W16 // self.patch_size
        ray_features = raymap_to_patch_features(ray_maps, Hp, Wp)
        patch_rays = patch_mean_rays(ray_maps, Hp, Wp)

        pose_enc, depth, depth_conf = run_trunk(self, images_in, ray_features, patch_rays)

        out = {}
        with torch.autocast(device_type=images.device.type, enabled=False):
            out["pose_enc"] = pose_enc
            out["camera_poses"] = pose_encoding_to_c2w(pose_enc, quat_to_mat)
            if depth.shape[2:4] != (H, W):
                d = F.interpolate(depth[..., 0].flatten(0, 1).unsqueeze(1), size=(H, W), mode="bilinear",
                                  align_corners=False)
                depth = d.squeeze(1).view(B, S, H, W, 1)
                c = F.interpolate(depth_conf.flatten(0, 1).unsqueeze(1), size=(H, W), mode="bilinear",
                                  align_corners=False)
                depth_conf = c.squeeze(1).view(B, S, H, W)
            local_points = ray_maps.to(depth.dtype) * depth
            out.update(depth=depth, depth_conf=depth_conf, local_points=local_points)
            out["world_points"] = torch.einsum("bsij,bshwj->bshwi", out["camera_poses"],
                                               homogenize_points(local_points))[..., :3]
        return out

    @torch.no_grad()
    def infer(self, images: torch.Tensor, ray_maps: torch.Tensor = None) -> dict:
        """One scene: images (S, 3, H, W) in [0, 1], ray_maps (S, H, W, 3) or None.

        Runs forward() under no_grad and bfloat16 autocast (fp16 where bf16 is unsupported), the
        precision Any3R was evaluated in, and returns float32 tensors without the batch axis.
        The images are put in channels-last memory format first: under bfloat16 the convolution
        kernels, and with them the low bits of the output, depend on the input's memory layout, and
        channels-last is the layout of the paper's evaluation, so results are reproducible and
        identical to it whatever layout the caller passes."""
        if images.dim() != 4:
            raise ValueError("infer() takes one scene, images of shape (S, 3, H, W); use forward() for batches")
        dev = next(self.parameters()).device
        images = images.to(dev).float().contiguous(memory_format=torch.channels_last)
        ray_maps = None if ray_maps is None else ray_maps.to(dev).float()
        amp_dtype = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=dev.type == "cuda"):
            out = self.forward(images, ray_maps)
        return {k: v[0].float() if torch.is_tensor(v) and v.dim() > 1 else v for k, v in out.items()}
