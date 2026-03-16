# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
import time
import gc
from contextlib import nullcontext
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
    viewmatrix,
)
from fussim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from gsplat import export_splats, import_splats
from gsplat.compression import PngCompression
from gsplat.compression.sort import sort_splats
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove
from gsplat.utils import log_transform
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
import kornia
from datasets.read_write_model import read_model, write_model, rotmat2qvec
from scipy.spatial.transform import Rotation as R
from datasets.normalize import transform_points
from datasets.visual_hull import VisualHull
import open3d as o3d
import open3d.core as o3c


def append_jsonl(path: str, payload: Dict[str, object]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def open3d_has_cuda_support() -> bool:
    build_config = getattr(o3d, "_build_config", {})
    return bool(build_config.get("BUILD_CUDA_MODULE", False)) and o3d.core.cuda.is_available()


def resolve_open3d_device(cuda_index: int) -> tuple[o3d.core.Device, bool]:
    if open3d_has_cuda_support():
        device = o3d.core.Device(f"CUDA:{cuda_index}")
        print(f"Using Open3D device: {device}")
        return device, True

    build_config = getattr(o3d, "_build_config", {})
    if not build_config.get("BUILD_CUDA_MODULE", False):
        print("Warning: Open3D was installed without CUDA module support; reconstruction will run on CPU.")
    else:
        print("Warning: Open3D CUDA module is present but no CUDA device is available; reconstruction will run on CPU.")
    return o3d.core.Device("CPU:0"), False


def open3d_tensor_from_torch(
    tensor: Tensor,
    *,
    o3d_device: o3d.core.Device,
    use_cuda: bool,
) -> o3d.core.Tensor:
    tensor = tensor.detach().contiguous()
    if use_cuda:
        # Clone after DLPack import so Open3D owns the CUDA buffer lifetime
        # instead of sharing it with Torch during interpreter teardown.
        return o3d.core.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(tensor)
        ).clone()
    return o3d.core.Tensor(tensor.cpu().numpy()).to(o3d_device)


def open3d_image_from_torch(
    tensor: Tensor,
    *,
    o3d_device: o3d.core.Device,
    use_cuda: bool,
) -> o3d.t.geometry.Image:
    return o3d.t.geometry.Image(
        open3d_tensor_from_torch(tensor, o3d_device=o3d_device, use_cuda=use_cuda)
    )


@dataclass(frozen=True)
class SparsifySchedule:
    base_steps: int
    sparsify_steps: int
    refine_steps: int

    @property
    def enabled(self) -> bool:
        return self.sparsify_steps > 0

    @property
    def total_steps(self) -> int:
        if not self.enabled:
            return self.base_steps
        return self.base_steps + self.sparsify_steps + self.refine_steps

    @property
    def prune_step(self) -> Optional[int]:
        if not self.enabled:
            return None
        return self.base_steps + self.sparsify_steps - 1

    def phase(self, step: int) -> Literal["base", "sparsify", "refine"]:
        if not self.enabled or step < self.base_steps:
            return "base"
        if step < self.base_steps + self.sparsify_steps:
            return "sparsify"
        return "refine"

    def step_in_phase(self, step: int) -> int:
        phase = self.phase(step)
        if phase == "base":
            return step
        if phase == "sparsify":
            return step - self.base_steps
        return step - self.base_steps - self.sparsify_steps

    def boundary_steps_1based(self) -> set[int]:
        steps = {self.base_steps}
        if self.enabled:
            steps.add(self.base_steps + self.sparsify_steps)
            steps.add(self.total_steps)
        return {step for step in steps if step > 0}


class ADMMSparsifier:
    """Opacity sparsifier modeled after the post-train ADMM stage in LichtFeld-Studio."""

    def __init__(self, *, rho: float, prune_ratio: float) -> None:
        self.rho = rho
        self.prune_ratio = prune_ratio
        self.z: Optional[Tensor] = None
        self.u: Optional[Tensor] = None

    @property
    def initialized(self) -> bool:
        return self.z is not None and self.u is not None

    def initialize(self, opacities: Tensor) -> None:
        opa = torch.sigmoid(opacities.detach())
        self.u = torch.zeros_like(opa)
        self.z = self._threshold(opa)

    def penalty(self, opacities: Tensor) -> Tensor:
        if not self.initialized:
            self.initialize(opacities)
        assert self.z is not None and self.u is not None
        opa = torch.sigmoid(opacities)
        diff = opa - self.z + self.u
        return 0.5 * self.rho * diff.square().sum()

    @torch.no_grad()
    def update_state(self, opacities: Tensor) -> None:
        if not self.initialized:
            self.initialize(opacities)
        assert self.z is not None and self.u is not None
        opa = torch.sigmoid(opacities.detach())
        self.z = self._threshold(opa + self.u)
        self.u = self.u + opa - self.z

    @torch.no_grad()
    def build_prune_mask(self, opacities: Tensor) -> Tensor:
        opa = torch.sigmoid(opacities.detach()).flatten()
        if opa.numel() <= 1:
            return torch.zeros_like(opa, dtype=torch.bool)
        n_prune = min(
            int(self.prune_ratio * opa.numel()),
            opa.numel() - 1,
        )
        if n_prune <= 0:
            return torch.zeros_like(opa, dtype=torch.bool)
        prune_indices = torch.argsort(opa, descending=False)[:n_prune]
        mask = torch.zeros_like(opa, dtype=torch.bool)
        mask[prune_indices] = True
        return mask

    @torch.no_grad()
    def _threshold(self, values: Tensor) -> Tensor:
        flat = values.flatten()
        if flat.numel() == 0:
            return torch.zeros_like(values)
        prune_index = min(
            int(self.prune_ratio * flat.numel()),
            flat.numel() - 1,
        )
        if prune_index <= 0:
            return torch.zeros_like(values)
        threshold = flat.sort(descending=False).values[prune_index - 1]
        return torch.where(values > threshold, values, torch.zeros_like(values))


def total_optimization_steps(cfg: "Config") -> int:
    if not cfg.enable_sparsify:
        return cfg.max_steps
    return cfg.max_steps + cfg.sparsify_steps + cfg.sparsify_refine_steps


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Path to the .ply file to initialize splats. If provide, it will skip SFM initialization.
    ply_file: Optional[str] = None
    # Whether to evaluate or train
    eval_only: bool = False
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"
    # Image name for centering and creating a camera shake video
    center_and_shake_image_name: Optional[str] = "cam_5/17.png"
    # Radius for the camera shake
    shake_radius: float = 0.2
    # Number of frames for the camera shake video
    shake_frames: int = 120
    # Factor to move the camera closer to the object (1.0 is original distance)
    shake_distance_factor: float = 0.15

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "/data/shared/aly/data/filter_on_sponge/"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "/data/shared/aly/results/videos/"
    # Every N images there is a test image
    test_every: int = 0
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = False

    # Port for the viewer server
    port: int = 8085

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 50_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 50_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 50_000])
    # Refresh the tqdm loss/phase display every N steps. Lower values are more
    # responsive but force more GPU-to-CPU scalar syncs.
    progress_refresh_every: int = 10
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False
    # Whether to render all training views
    should_render_depths: bool = False

    # Whether to pre-load all images into memory
    load_images_in_memory: bool = False
    # Whether to directly load them to gpu, only used if load_images_in_memory is True
    load_images_to_gpu: bool = False
    # Whether to optimize by cropping to foreground bounding box
    optimize_foreground: bool = False
    # Margin for foreground optimization
    foreground_margin: float = 0.1
    # Image exclusion prefixes
    exclude_prefixes: List[str] | None = None

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Voxel size for visual hull initialization
    hull_voxel_size: int = 256
    # --- Visual Hull Sampling Config ---
    # Number of evenly spaced images per camera to use for visual hull. 0 means use all.
    hull_images_per_camera: int = 2
    # Stride to skip cameras for the hull (1=use all, 2=use every second).
    hull_camera_stride: int = 2
    # Whether to sample only from the first quarter of images for each camera.
    hull_sample_from_first_quarter: bool = False
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Randomize initial colors while keeping the chosen initialization geometry.
    randomize_init_colors: bool = False
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # Enable a post-training sparsification stage.
    enable_sparsify: bool = False
    # Number of optimization steps in the sparsification stage.
    sparsify_steps: int = 2_000
    # Number of short refinement steps to run after pruning.
    sparsify_refine_steps: int = 500
    # ADMM update interval during sparsification.
    sparsify_update_every: int = 50
    # ADMM penalty coefficient.
    sparsify_rho: float = 5e-4
    # Fraction of Gaussians to prune at the end of the sparsify stage.
    sparsify_prune_ratio: float = 0.6

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # PPISP controller activation step. Negative values use the LichtFeld-style
    # default of the final 5k optimization steps.
    ppisp_controller_activation_num_steps: int = -1
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False
    # Record CUDA-synchronized per-phase timings for each training step.
    profile_detailed_timing: bool = False
    # Export a short torch profiler trace and operator summaries.
    profile_trace: bool = False
    profile_trace_warmup_steps: int = 2
    profile_trace_active_steps: int = 4

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether to undistort COLMAP input (disable for fisheye with 3DGUT)
    undistort_colmap_input: bool = True

    # Whether to use masks
    use_masks: bool = False

    # Enable erank loss. (experimental)
    use_erank_loss: bool = False
    # Start step for erank loss
    erank_start_step: int = 7000
    # Weight for erank loss
    erank_lambda: float = 0.05

    # Whether to presort the splats at initialization
    use_sort: bool = False
    sort_kernel_size: int = 5
    sort_sigma: float = 3.0
    sort_lambda: float = 1.0

    use_rade: bool = False
    rade_lambda: float = 0.05
    rade_step: int = 15_000

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sparsify_steps = int(self.sparsify_steps * factor)
        self.sparsify_refine_steps = int(self.sparsify_refine_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    dataset: Dataset,
    init_type: str = "sfm",
    ply_file: Optional[str] = None,
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    randomize_init_colors: bool = False,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    use_sort: bool = False,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    hull_voxel_size: int = 128,
    hull_images_per_camera: int = 4,
    hull_camera_stride: int = 1,
    hull_sample_from_first_quarter: bool = False,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    shNs = None
    if ply_file:
        print(f"Initializing splats from {ply_file}")
        points, scales, quats, opacities, sh0s_ply, shNs = import_splats(
            ply_file, device=device
        )
        print(f"Loaded {points.shape[0]} splats.")
        sh0s = sh0s_ply.squeeze(1)  # (N, 1, 3) -> (N, 3)
    else:
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "visual_hull":
            bounds = (parser.points.min(axis=0), parser.points.max(axis=0))
            points_np = generate_points_from_visual_hull(
                dataset,
                bounds,
                hull_voxel_size,
                hull_images_per_camera,
                hull_camera_stride,
                hull_sample_from_first_quarter,
            )
            points_sfm = torch.from_numpy(parser.points).float()
            rgbs_sfm = torch.from_numpy(parser.points_rgb / 255.0).float()
            points = torch.cat([points_sfm, torch.from_numpy(points_np).float()], dim=0)
            rgbs = torch.cat([rgbs_sfm, torch.zeros((points_np.shape[0], 3))], dim=0)
            # points = torch.from_numpy(points_np).float()
            # rgbs = torch.zeros((points_np.shape[0], 3))
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError("Please specify a correct init_type: sfm or random")

        if randomize_init_colors:
            rgbs = torch.rand_like(rgbs)

        # Initialize the GS size to be the average dist of the 3 nearest neighbors
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

        if use_sort:
            print("Pre-sorting splats at initialization...")
            # 1. Group initial data into a dictionary
            N_total = points.shape[0]
            splat_tensors = {
                "means": points,
                "scales": scales,
                "opacities": torch.logit(torch.full((N_total,), init_opacity)),
                "quats": torch.rand((N_total, 4)),
                "sh0": rgb_to_sh(rgbs),
                "rgbs": rgbs,  # Also carry rgbs for feature_dim case
            }

            # 2. Crop to the nearest perfect square number
            n_gs = N_total
            n_sidelen = int(n_gs**0.5)
            n_square = n_sidelen**2
            if n_gs != n_square:
                print(
                    f"Cropping splats from {n_gs} to {n_square} to make it a perfect square."
                )
                # We can just truncate, as initial opacities are identical
                for k, v in splat_tensors.items():
                    splat_tensors[k] = v[:n_square]

            # 3. Prepare a temporary copy for sorting
            splats_for_sorting = {k: v.clone() for k, v in splat_tensors.items() if k != "rgbs"}
            splats_for_sorting["means"] = log_transform(splats_for_sorting["means"])
            splats_for_sorting["quats"] = F.normalize(
                splats_for_sorting["quats"], dim=-1
            )

            # 4. Get the sorting indices
            _, sort_indices = sort_splats(splats_for_sorting, verbose=False)

            # 5. Apply sorting to original tensors
            for k, v in splat_tensors.items():
                splat_tensors[k] = v[sort_indices]

            # 6. Unpack the sorted tensors to be used by the rest of the function
            points = splat_tensors["means"]
            scales = splat_tensors["scales"]
            opacities = splat_tensors["opacities"]  # Already in logit scale
            quats = splat_tensors["quats"]
            sh0s = splat_tensors["sh0"]
            rgbs = splat_tensors["rgbs"]
        else:
            # Original path
            opacities = torch.logit(torch.full((points.shape[0],), init_opacity))
            quats = torch.rand((points.shape[0], 4))
            sh0s = rgb_to_sh(rgbs)

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    scales = scales[world_rank::world_size]
    quats = quats[world_rank::world_size]
    opacities = opacities[world_rank::world_size]
    sh0s = sh0s[world_rank::world_size]
    if shNs is not None:
        shNs = shNs[world_rank::world_size]

    N = points.shape[0]
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=device)  # [N, K, 3]
        colors[:, 0, :] = sh0s
        if shNs is not None:
            num_sh_bands_in_ns = shNs.shape[1]
            # Ensure we don't exceed the allocated size for colors
            num_sh_bands_to_copy = min(num_sh_bands_in_ns, colors.shape[1] - 1)
            if num_sh_bands_in_ns > num_sh_bands_to_copy:
                print(
                    f"Warning: PLY file has more SH bands ({num_sh_bands_in_ns}) "
                    f"than configured ({colors.shape[1] - 1}). Truncating."
                )
            colors[:, 1 : 1 + num_sh_bands_to_copy, :] = shNs[
                :, :num_sh_bands_to_copy, :
            ]
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :].contiguous()), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :].contiguous()), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True,
        )
        for name, _, lr in params
    }
    return splats, optimizers

def generate_points_from_visual_hull(
    dataset: Dataset,
    bounds: Tuple[np.ndarray, np.ndarray],
    voxel_size: int,
    images_per_camera: int,
    camera_stride: int,
    sample_from_first_quarter: bool,
) -> np.ndarray:
    """Generates a point cloud from the surface of a visual hull."""
    print("Generating points from visual hull...")
    from collections import defaultdict

    if images_per_camera <= 0:
        print("Using all available training images for visual hull.")
        subset = dataset
    else:
        print(f"Subsampling views for visual hull...")
        # 1. Group dataset indices by camera ID
        camera_to_indices = defaultdict(list)
        for i in range(len(dataset)):
            original_index = dataset.indices[i]
            image_name = dataset.parser.image_names[original_index]
            camera_id = image_name.split("/")[0]
            camera_to_indices[camera_id].append(i)

        # 2. Subsample cameras using the stride
        all_camera_ids = sorted(camera_to_indices.keys())
        selected_camera_ids = all_camera_ids[::camera_stride]
        print(
            f"Selected {len(selected_camera_ids)} out of {len(all_camera_ids)} cameras."
        )

        # 3. Sample indices from each selected camera group
        sampled_indices = []
        for camera_id in selected_camera_ids:
            indices_for_cam = camera_to_indices[camera_id]
            num_images_in_cam = len(indices_for_cam)

            if num_images_in_cam <= images_per_camera:
                sampled_indices.extend(indices_for_cam)
                continue

            # Determine the sampling range
            if sample_from_first_quarter:
                # Sample from the first 25% of images
                end_index = num_images_in_cam // 4
            else:
                # Sample from all images
                end_index = num_images_in_cam

            # Ensure we don't try to sample more images than available in the range
            num_to_sample = min(images_per_camera, end_index)

            if num_to_sample > 0:
                # np.linspace is inclusive, so we sample from [0, end_index-1]
                positions = np.linspace(
                    0, end_index - 1, num_to_sample, dtype=int
                )
                sampled_indices.extend([indices_for_cam[p] for p in positions])

        print(
            f"Using {len(sampled_indices)} images from {len(selected_camera_ids)} cameras for visual hull."
        )
        subset = torch.utils.data.Subset(dataset, sampled_indices)

    # Use a DataLoader to iterate through the dataset subset
    data_loader = torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False)

    camera_matrices = []
    masks = []

    for data in tqdm.tqdm(data_loader, desc="Loading data for Visual Hull"):
        camtoworld = data["camtoworld"].squeeze(0).cpu().numpy()
        K = data["K"].squeeze(0).cpu().numpy()
        mask_tensor = data.get("segmentation_mask")

        if mask_tensor is not None:
            w2c = np.linalg.inv(camtoworld)
            P = K @ w2c[:3, :]
            camera_matrices.append(P)

            mask_np = mask_tensor.squeeze(0).cpu().numpy()
            if mask_np.ndim == 3:
                mask_np = mask_np.squeeze(-1)
            masks.append(mask_np > 0.5)
        else:
            image_name = data["image_name"][0]
            print(f"Warning: No mask for image {image_name}, skipping for visual hull.")
            continue

    if not masks:
        raise ValueError("No valid masks found to generate visual hull.")

    # Get scene bounds from arguments
    min_bound, max_bound = bounds

    # 3. Carve the visual hull
    hull = VisualHull(voxel_size=voxel_size, bounds=(min_bound, max_bound))
    hull.process_all_views(camera_matrices, masks)

    # 4. Extract surface points from the hull
    surface_points = hull.get_surface_points()

    print(f"Generated {len(surface_points)} points from visual hull surface.")

    # if len(surface_points) == 0:
    #     raise ValueError(
    #         "Visual hull resulted in 0 points. Check your masks or scene bounds."
    #     )

    return surface_points

def erode_masks(masks, kernel_size=3, iterations=1):
    """Apply erosion to masks using max pooling with negative values."""
    import torch.nn.functional as F

    # Create padding to maintain size
    padding = kernel_size // 2
    eroded = masks

    for _ in range(iterations):
        # Invert mask (1->0, 0->1), apply max pooling, then invert back
        inverted = 1 - eroded
        pooled = F.max_pool2d(inverted.float(), kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1 - pooled

    return eroded

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.sparsify_schedule = SparsifySchedule(
            base_steps=cfg.max_steps,
            sparsify_steps=cfg.sparsify_steps if cfg.enable_sparsify else 0,
            refine_steps=cfg.sparsify_refine_steps if cfg.enable_sparsify else 0,
        )
        if cfg.enable_sparsify:
            if cfg.sparsify_steps <= 0:
                raise ValueError("enable_sparsify requires sparsify_steps > 0.")
            if not 0.0 < cfg.sparsify_prune_ratio < 1.0:
                raise ValueError(
                    "sparsify_prune_ratio must be between 0 and 1."
                )

        if cfg.load_images_to_gpu and cfg.optimize_foreground and cfg.batch_size > 1:
            raise ValueError(
                "Using 'load_images_to_gpu' with 'optimize_foreground' is only supported for batch_size=1, "
                "as it results in variable image sizes that cannot be batched."
            )

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        if cfg.save_ply:
            self.ply_dir = f"{cfg.result_dir}/ply"
            os.makedirs(self.ply_dir, exist_ok=True)
        if cfg.pose_opt:
            self.save_colmap_path = f"{cfg.data_dir}/sparse/pose_opt_colmap"

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        should_load_exposure = (
            cfg.load_exposure
            or cfg.app_opt
            or cfg.post_processing == "ppisp"
        )
        if should_load_exposure and not cfg.load_exposure:
            print("[Parser] Enabling EXIF exposure loading because appearance correction is active.")
        cfg.load_exposure = should_load_exposure

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            undistort_input=cfg.undistort_colmap_input,
            use_masks=cfg.use_masks,
            load_images_in_memory=cfg.load_images_in_memory,
            load_images_to_gpu=cfg.load_images_to_gpu,
            optimize_foreground=cfg.optimize_foreground,
            foreground_margin=cfg.foreground_margin,
            exclude_prefixes=cfg.exclude_prefixes,
            load_exposure=should_load_exposure,
        )
        
        # Auto-detect fisheye cameras and adjust config accordingly
        if not cfg.undistort_colmap_input and hasattr(self.parser, 'camtype_dict'):
            # Check if any camera is fisheye
            is_fisheye = any(camtype == "fisheye" for camtype in self.parser.camtype_dict.values())
            if is_fisheye:
                print("Fisheye cameras detected. Setting camera_model to 'fisheye' and enabling 3DGUT flags.")
                cfg.camera_model = "fisheye"
                cfg.with_ut = True
                cfg.with_eval3d = True
        
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            device=self.device,
        )
        self.valset = Dataset(self.parser, split="val", device=self.device)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if self.parser.num_cameras > 1 and cfg.batch_size != 1:
            raise ValueError(
                f"When using multiple cameras ({self.parser.num_cameras} found), batch_size must be 1, "
                f"but got batch_size={cfg.batch_size}."
            )
        if cfg.post_processing == "ppisp" and cfg.batch_size != 1:
            raise ValueError(
                f"PPISP post-processing requires batch_size=1, got batch_size={cfg.batch_size}"
            )
        if cfg.post_processing is not None and world_size > 1:
            raise ValueError(
                f"Post-processing ({cfg.post_processing}) requires single-GPU training, "
                f"but world_size={world_size}."
            )
        if cfg.post_processing == "ppisp" and isinstance(cfg.strategy, DefaultStrategy):
            raise ValueError(
                f"PPISP post-processing requires MCMCStrategy at the moment."
            )

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            self.trainset,
            init_type=cfg.init_type,
            ply_file=cfg.ply_file,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            randomize_init_colors=cfg.randomize_init_colors,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            use_sort=cfg.use_sort,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            hull_voxel_size=cfg.hull_voxel_size,
            hull_images_per_camera=cfg.hull_images_per_camera,
            hull_camera_stride=cfg.hull_camera_stride,
            hull_sample_from_first_quarter=cfg.hull_sample_from_first_quarter,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Unify sorting flag
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.sort = self.cfg.use_sort

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.post_processing_module = None
        self.ppisp_controller_activation_step: Optional[int] = None
        self.ppisp_controller_activation_uses_default = False
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_module = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
        elif cfg.post_processing == "ppisp":
            total_steps = total_optimization_steps(cfg)
            controller_activation_step = cfg.ppisp_controller_activation_num_steps
            self.ppisp_controller_activation_uses_default = (
                controller_activation_step < 0
            )
            if controller_activation_step < 0:
                controller_activation_step = max(0, total_steps - 5_000)
            self.ppisp_controller_activation_step = controller_activation_step
            ppisp_config = PPISPConfig(
                use_controller=cfg.ppisp_use_controller,
                controller_distillation=cfg.ppisp_controller_distillation,
                controller_activation_ratio=controller_activation_step / total_steps,
            )
            self.post_processing_module = PPISP(
                num_cameras=self.parser.num_cameras,
                num_frames=len(self.trainset),
                config=ppisp_config,
            ).to(self.device)

        self.post_processing_optimizers = []
        if cfg.post_processing == "bilateral_grid":
            self.post_processing_optimizers = [
                torch.optim.Adam(
                    self.post_processing_module.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]
        elif cfg.post_processing == "ppisp":
            self.post_processing_optimizers = (
                self.post_processing_module.create_optimizers()
            )

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        self.sparsifier = None
        if cfg.enable_sparsify:
            self.sparsifier = ADMMSparsifier(
                rho=cfg.sparsify_rho,
                prune_ratio=cfg.sparsify_prune_ratio,
            )

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def freeze_gaussians(self):
        """Freeze all Gaussian parameters for controller distillation.

        This prevents Gaussians from being updated by any loss (including regularization)
        while the controller learns to predict per-frame corrections.
        """
        if self._gaussians_frozen:
            return

        for name, param in self.splats.items():
            param.requires_grad = False

        self._gaussians_frozen = True
        print("[Distillation] Gaussian parameters frozen")

    def unfreeze_gaussians(self):
        """Re-enable Gaussian updates after controller distillation or pruning."""
        if not self._gaussians_frozen:
            return

        for _, param in self.splats.items():
            param.requires_grad = True

        self._gaussians_frozen = False
        print("[Distillation] Gaussian parameters unfrozen")

    @torch.no_grad()
    def apply_sparsify_prune(self, step: int) -> None:
        if self.sparsifier is None:
            return

        mask = self.sparsifier.build_prune_mask(self.splats["opacities"])
        n_before = len(self.splats["means"])
        n_prune = int(mask.sum().item())
        if n_prune <= 0:
            print("[Sparsify] No Gaussians selected for pruning")
            return

        remove(params=self.splats, optimizers=self.optimizers, state={}, mask=mask)
        n_after = len(self.splats["means"])
        stats = {
            "step": step,
            "num_GS_before": n_before,
            "num_GS_after": n_after,
            "num_GS_pruned": n_before - n_after,
            "prune_ratio_realized": (n_before - n_after) / max(n_before, 1),
        }
        print(
            f"[Sparsify] Pruned {stats['num_GS_pruned']} GSs "
            f"({n_before} -> {n_after})"
        )
        with open(
            f"{self.stats_dir}/sparsify_step{step:04d}_rank{self.world_rank}.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(stats, handle)
        if self.world_rank == 0:
            self.writer.add_scalar(
                "sparsify/pruned_gaussians", stats["num_GS_pruned"], step
            )
            self.writer.add_scalar(
                "sparsify/prune_ratio_realized",
                stats["prune_ratio_realized"],
                step,
            )
            self.writer.flush()

    def load_training_checkpoint(self, ckpt_files: List[str]) -> int:
        """Load model state from checkpoint(s) and return the next step index."""
        ckpts = [
            torch.load(file, map_location=self.device, weights_only=True)
            for file in ckpt_files
        ]
        if not ckpts:
            raise ValueError("No checkpoint files were provided.")

        for key in self.splats.keys():
            self.splats[key].data = torch.cat([ckpt["splats"][key] for ckpt in ckpts])

        primary = ckpts[0]
        if self.cfg.pose_opt and "pose_adjust" in primary:
            module = self.pose_adjust.module if self.world_size > 1 else self.pose_adjust
            module.load_state_dict(primary["pose_adjust"])
        if self.cfg.app_opt and "app_module" in primary:
            module = self.app_module.module if self.world_size > 1 else self.app_module
            module.load_state_dict(primary["app_module"])
        if self.post_processing_module is not None:
            pp_state = primary.get("post_processing")
            if pp_state is not None:
                self.post_processing_module.load_state_dict(pp_state)

        next_step = int(primary["step"]) + 1
        print(
            f"[Checkpoint] Loaded {len(ckpts)} checkpoint(s); "
            f"resuming from step {next_step} with {len(self.splats['means'])} GS"
        )
        return next_step

    def save_training_checkpoint(self, step: int, global_tic: float) -> None:
        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellipse_time": time.time() - global_tic,
            "num_GS": len(self.splats["means"]),
        }
        print("Step: ", step, stats)
        with open(
            f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(stats, f)
        data = {"step": step, "splats": self.splats.state_dict()}
        if self.cfg.pose_opt:
            if self.world_size > 1:
                data["pose_adjust"] = self.pose_adjust.module.state_dict()
            else:
                data["pose_adjust"] = self.pose_adjust.state_dict()
        if self.cfg.app_opt:
            if self.world_size > 1:
                data["app_module"] = self.app_module.module.state_dict()
            else:
                data["app_module"] = self.app_module.state_dict()
        if self.post_processing_module is not None:
            data["post_processing"] = self.post_processing_module.state_dict()
        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

    def export_current_ply(self, step: int, sh_degree_to_use: int) -> None:
        if self.cfg.app_opt:
            # Eval at origin to bake appearance correction into exported colors.
            rgb = self.app_module(
                features=self.splats["features"],
                embed_ids=None,
                dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                sh_degree=sh_degree_to_use,
            )
            rgb = rgb + self.splats["colors"]
            rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
            sh0 = rgb_to_sh(rgb)
            shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
        else:
            sh0 = self.splats["sh0"]
            shN = self.splats["shN"]

        export_splats(
            means=self.splats["means"],
            scales=self.splats["scales"],
            quats=self.splats["quats"],
            opacities=self.splats["opacities"],
            sh0=sh0,
            shN=shN,
            format="ply",
            save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
        )

    def _is_ppisp_controller_phase(self, step: int) -> bool:
        return (
            self.cfg.post_processing == "ppisp"
            and self.cfg.ppisp_use_controller
            and self.cfg.ppisp_controller_distillation
            and self.ppisp_controller_activation_step is not None
            and step >= self.ppisp_controller_activation_step
        )

    def _refresh_ppisp_controller_activation_step(
        self,
        *,
        init_step: int,
        base_max_steps: int,
        total_steps: int,
        optimizer_schedule_steps: int,
    ) -> None:
        if self.cfg.post_processing != "ppisp" or self.post_processing_module is None:
            return
        if self.ppisp_controller_activation_step is None:
            return

        activation_step = self.ppisp_controller_activation_step
        if self.ppisp_controller_activation_uses_default:
            activation_step = max(0, total_steps - 5_000)
            # Resumed post-base runs (for example optimize-prune rounds resuming from
            # ckpt_29999) should not default into controller-only mode immediately.
            if init_step >= base_max_steps:
                activation_step = total_steps + 1

        self.ppisp_controller_activation_step = activation_step
        self.post_processing_module.config.controller_activation_ratio = (
            activation_step / max(optimizer_schedule_steps, 1)
        )

    def _crop_to_foreground_for_ppisp(
        self,
        colors: Tensor,
        pixels: Tensor,
        segmentation_masks: Optional[Tensor],
        *,
        padding: int = 8,
    ) -> Tuple[Tensor, Tensor]:
        if segmentation_masks is None or colors.shape[0] != 1:
            return colors, pixels

        mask = segmentation_masks[0] >= 0.5
        if not torch.any(mask):
            return colors, pixels

        ys, xs = torch.where(mask)
        y0 = max(int(ys.min().item()) - padding, 0)
        y1 = min(int(ys.max().item()) + padding + 1, colors.shape[1])
        x0 = max(int(xs.min().item()) - padding, 0)
        x1 = min(int(xs.max().item()) + padding + 1, colors.shape[2])

        if y1 <= y0 or x1 <= x0:
            return colors, pixels

        return (
            colors[:, y0:y1, x0:x1, :],
            pixels[:, y0:y1, x0:x1, :],
        )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        frame_idcs: Optional[Tensor] = None,
        camera_idcs: Optional[Tensor] = None,
        exposure: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, expected_depths, median_depths, expected_normals, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0

        if self.cfg.post_processing is not None:
            # Create pixel coordinates [H, W, 2] with +0.5 center offset
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(height, device=self.device) + 0.5,
                torch.arange(width, device=self.device) + 0.5,
                indexing="ij",
            )
            pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1)  # [H, W, 2]

            # Split RGB from extra channels (e.g. depth) for post-processing
            rgb = render_colors[..., :3]
            extra = render_colors[..., 3:] if render_colors.shape[-1] > 3 else None

            if self.cfg.post_processing == "bilateral_grid":
                if frame_idcs is not None:
                    grid_xy = (
                        pixel_coords / torch.tensor([width, height], device=self.device)
                    ).unsqueeze(0)
                    rgb = slice(
                        self.post_processing_module,
                        grid_xy.expand(rgb.shape[0], -1, -1, -1),
                        rgb,
                        frame_idcs.unsqueeze(-1),
                    )["rgb"]
            elif self.cfg.post_processing == "ppisp":
                camera_idx = camera_idcs.item() if camera_idcs is not None else None
                frame_idx = frame_idcs.item() if frame_idcs is not None else None
                rgb = self.post_processing_module(
                    rgb=rgb,
                    pixel_coords=pixel_coords,
                    resolution=(width, height),
                    camera_idx=camera_idx,
                    frame_idx=frame_idx,
                    exposure_prior=exposure,
                )

            render_colors = (
                torch.cat([rgb, extra], dim=-1) if extra is not None else rgb
            )

        return render_colors, render_alphas, expected_depths, median_depths, expected_normals, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        schedule = self.sparsify_schedule
        base_max_steps = cfg.max_steps
        max_steps = schedule.total_steps
        init_step = 0
        if cfg.ckpt is not None:
            init_step = self.load_training_checkpoint(cfg.ckpt)
        optimizer_schedule_steps = max_steps if schedule.enabled else base_max_steps
        if init_step >= base_max_steps:
            optimizer_schedule_steps = base_max_steps
        self._refresh_ppisp_controller_activation_step(
            init_step=init_step,
            base_max_steps=base_max_steps,
            total_steps=max_steps,
            optimizer_schedule_steps=optimizer_schedule_steps,
        )

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"],
                gamma=0.01 ** (1.0 / optimizer_schedule_steps),
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0],
                    gamma=0.01 ** (1.0 / optimizer_schedule_steps),
                )
            )
        ppisp_schedulers: List[torch.optim.lr_scheduler.LRScheduler] = []
        # Post-processing module has a learning rate schedule
        if cfg.post_processing == "bilateral_grid":
            # Linear warmup + exponential decay
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.post_processing_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.post_processing_optimizers[0],
                            gamma=0.01 ** (1.0 / optimizer_schedule_steps),
                        ),
                    ]
                )
            )
        elif cfg.post_processing == "ppisp":
            ppisp_schedulers = self.post_processing_module.create_schedulers(
                self.post_processing_optimizers,
                max_optimization_iters=optimizer_schedule_steps,
            )
            schedulers.extend(ppisp_schedulers)
        ppisp_controller_scheduler = None
        if (
            cfg.post_processing == "ppisp"
            and cfg.ppisp_use_controller
            and len(self.post_processing_optimizers) > 1
            and self.ppisp_controller_activation_step is not None
        ):
            controller_total_iters = max(
                optimizer_schedule_steps - self.ppisp_controller_activation_step,
                1,
            )
            controller_warmup_iters = min(100, controller_total_iters)
            if controller_warmup_iters < controller_total_iters:
                controller_schedulers: List[torch.optim.lr_scheduler.LRScheduler] = [
                    torch.optim.lr_scheduler.LinearLR(
                        self.post_processing_optimizers[1],
                        start_factor=0.1,
                        total_iters=controller_warmup_iters,
                    ),
                    torch.optim.lr_scheduler.ExponentialLR(
                        self.post_processing_optimizers[1],
                        gamma=0.01
                        ** (1.0 / max(controller_total_iters - controller_warmup_iters, 1)),
                    ),
                ]
                ppisp_controller_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.post_processing_optimizers[1],
                    schedulers=controller_schedulers,
                    milestones=[controller_warmup_iters],
                )
            else:
                ppisp_controller_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.post_processing_optimizers[1],
                    start_factor=0.1,
                    total_iters=controller_warmup_iters,
                )

        if init_step > 0:
            for _ in range(init_step):
                for scheduler in schedulers:
                    scheduler.step()
            if ppisp_controller_scheduler is not None and self.ppisp_controller_activation_step is not None:
                controller_init_steps = max(
                    min(
                        init_step - self.ppisp_controller_activation_step,
                        optimizer_schedule_steps - self.ppisp_controller_activation_step,
                    ),
                    0,
                )
                for _ in range(controller_init_steps):
                    ppisp_controller_scheduler.step()

        trainloader_num_workers = 0 if self.trainset.on_gpu else (1 if cfg.load_images_in_memory else 4)
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=trainloader_num_workers,
            persistent_workers=trainloader_num_workers > 0,
            pin_memory=not self.trainset.on_gpu,
        )
        trainloader_iter = iter(trainloader)

        # Pre-compute step sets to avoid rebuilding lists every iteration
        def normalize_step_set(
            configured_steps: List[int],
            *,
            extra_steps: set[int],
        ) -> set[int]:
            merged = set(configured_steps) | extra_steps
            return {
                step - 1
                for step in merged
                if step > 0 and step <= max_steps
            }

        boundary_steps = schedule.boundary_steps_1based()
        save_steps_set = normalize_step_set(cfg.save_steps, extra_steps=boundary_steps)
        ply_steps_set = normalize_step_set(cfg.ply_steps, extra_steps=set())
        eval_steps_set = normalize_step_set(cfg.eval_steps, extra_steps=set())

        # Training loop.
        global_tic = time.time()
        speed_profile_path = f"{self.stats_dir}/speed_profile_rank{self.world_rank}.jsonl"
        detailed_phase_timing = cfg.profile_detailed_timing and str(device).startswith("cuda")
        profile_trace_enabled = cfg.profile_trace
        trace_profiler = None
        if profile_trace_enabled:
            trace_dir = os.path.join(
                self.stats_dir, f"torch_profile_rank{self.world_rank}"
            )
            os.makedirs(trace_dir, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if str(device).startswith("cuda"):
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            def on_trace_ready(profile: torch.profiler.profile) -> None:
                trace_path = os.path.join(trace_dir, "trace.json")
                profile.export_chrome_trace(trace_path)
                key_averages = profile.key_averages()
                summary_specs = (
                    ("summary_self_cuda.txt", "self_cuda_time_total"),
                    ("summary_cuda_total.txt", "cuda_time_total"),
                    ("summary_self_cpu.txt", "self_cpu_time_total"),
                )
                for filename, sort_key in summary_specs:
                    try:
                        table = key_averages.table(sort_by=sort_key, row_limit=100)
                    except RuntimeError:
                        continue
                    with open(os.path.join(trace_dir, filename), "w", encoding="utf-8") as handle:
                        handle.write(table)
                        handle.write("\n")

            trace_profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=cfg.profile_trace_warmup_steps,
                    active=cfg.profile_trace_active_steps,
                    repeat=1,
                ),
                on_trace_ready=on_trace_ready,
                record_shapes=False,
                profile_memory=True,
                with_stack=False,
            )
            trace_profiler.start()

        def begin_timed_phase() -> float:
            if detailed_phase_timing:
                torch.cuda.synchronize(device)
            return time.perf_counter()

        def end_timed_phase(started_at: float) -> float:
            if detailed_phase_timing:
                torch.cuda.synchronize(device)
            return time.perf_counter() - started_at

        def record_region(name: str):
            if profile_trace_enabled:
                return torch.autograd.profiler.record_function(name)
            return nullcontext()

        pbar = tqdm.tqdm(range(init_step, max_steps))
        progress_refresh_every = max(int(cfg.progress_refresh_every), 1)
        last_loss_value = float("nan")
        previous_phase: Optional[str] = None
        for step in pbar:
            phase = schedule.phase(step)
            phase_step = schedule.step_in_phase(step)
            if phase != previous_phase:
                if phase == "sparsify":
                    print(
                        f"[Sparsify] Entering sparsify phase at step {step} "
                        f"with {len(self.splats['means'])} GS"
                    )
                elif phase == "refine":
                    print(
                        f"[Sparsify] Entering refine phase at step {step} "
                        f"with {len(self.splats['means'])} GS"
                    )
                previous_phase = phase
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()
            step_started_at = time.perf_counter()
            checkpoint_time_sec = 0.0
            ply_export_time_sec = 0.0
            eval_time_sec = 0.0
            render_depths_time_sec = 0.0
            recon_time_sec = 0.0
            compression_time_sec = 0.0
            viewer_time_sec = 0.0
            data_prep_time_sec = 0.0
            rasterize_time_sec = 0.0
            strategy_pre_backward_time_sec = 0.0
            loss_assembly_time_sec = 0.0
            backward_time_sec = 0.0
            optimizer_and_strategy_time_sec = 0.0
            sparsify_loss_value = 0.0
            ppisp_controller_phase = self._is_ppisp_controller_phase(step)

            data_prep_started_at = begin_timed_phase()
            with record_region("train/data_prep"):
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)

                if not self.trainset.on_gpu:
                    camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
                    Ks = data["K"].to(device)  # [1, 3, 3]
                    pixels = data["image"].to(device)  # [1, H, W, 3]
                    image_ids = data["image_id"].to(device)
                    undistort_masks = data["undistort_mask"].to(device) if "undistort_mask" in data else None  # [1, H, W]
                    segmentation_masks = data["segmentation_mask"].to(device) if "segmentation_mask" in data else None  # [1, H, W]
                    distortion_params = data.get("distortion_params", None)
                    if distortion_params is not None:
                        distortion_params = distortion_params.to(device)
                    if cfg.depth_loss:
                        points = data["points"].to(device)  # [1, M, 2]
                        depths_gt = data["depths"].to(device)  # [1, M]
                else:
                    camtoworlds = camtoworlds_gt = data["camtoworld"]
                    Ks = data["K"]
                    pixels = data["image"]
                    image_ids = data["image_id"]
                    undistort_masks = data.get("undistort_mask")
                    segmentation_masks = data.get("segmentation_mask")
                    distortion_params = data.get("distortion_params")
                    # Depth loss is not supported with GPU loading for now
                    if cfg.depth_loss:
                        raise NotImplementedError("Depth loss is not supported when load_images_to_gpu is True.")

                num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                )
                exposure = (
                    data["exposure"].to(device) if "exposure" in data else None
                )  # [B,]

                height, width = pixels.shape[1:3]

                if cfg.pose_noise:
                    camtoworlds = self.pose_perturb(camtoworlds, image_ids)

                if cfg.pose_opt:
                    camtoworlds = self.pose_adjust(camtoworlds, image_ids)

                # sh schedule
                sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

                # Prepare distortion coefficients for rasterization
                radial_coeffs_to_pass = None
                tangential_coeffs_to_pass = None
                # thin_prism_coeffs_to_pass = None  # Not typically used in COLMAP

                if not cfg.undistort_colmap_input and distortion_params is not None:
                    if cfg.camera_model == "fisheye":
                        # For OPENCV_FISHEYE, COLMAP params are [k1, k2, k3, k4]
                        # The rasterization function expects fisheye radial_coeffs as [batch_size, 4]
                        if distortion_params.shape[-1] == 4:
                            radial_coeffs_to_pass = distortion_params  # Should be [batch_size, 4]
                        else:
                            print(f"Warning: Fisheye model expects 4 distortion params, got {distortion_params.shape[-1]}")
                    elif cfg.camera_model == "pinhole":
                        # For pinhole with distortion (OPENCV, RADIAL, SIMPLE_RADIAL)
                        # rasterization expects radial_coeffs [..., C, 6] and tangential_coeffs [..., C, 2]
                        num_params = distortion_params.shape[-1]
                        if num_params >= 1:
                            # Prepare radial coefficients (pad to 6 elements)
                            rad_params = torch.zeros(distortion_params.shape[0], 6, device=distortion_params.device)
                            
                            if num_params == 4:  # OPENCV: [k1, k2, p1, p2]
                                rad_params[:, 0] = distortion_params[:, 0]  # k1
                                rad_params[:, 1] = distortion_params[:, 1]  # k2
                                # k3-k6 remain zero
                                tangential_coeffs_to_pass = distortion_params[:, [2, 3]].unsqueeze(0)  # [1, C, 2] p1, p2
                            elif num_params == 2:  # RADIAL: [k1, k2, 0, 0] -> extract [k1, k2]
                                rad_params[:, 0] = distortion_params[:, 0]  # k1
                                rad_params[:, 1] = distortion_params[:, 1]  # k2
                            elif num_params == 1:  # SIMPLE_RADIAL: [k1, 0, 0, 0] -> extract [k1]
                                rad_params[:, 0] = distortion_params[:, 0]  # k1
                            else:
                                print(f"Warning: Unexpected number of distortion parameters: {num_params}")
                                
                            radial_coeffs_to_pass = rad_params.unsqueeze(0)  # [1, C, 6]

            data_prep_time_sec = end_timed_phase(data_prep_started_at)

            # forward
            rasterize_started_at = begin_timed_phase()
            with record_region("train/rasterize"):
                renders, alphas, expected_depths, median_depths, expected_normals, info = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                    masks=undistort_masks,
                    radial_coeffs=radial_coeffs_to_pass,
                    tangential_coeffs=tangential_coeffs_to_pass,
                    frame_idcs=image_ids,
                    camera_idcs=data["camera_idx"].to(device),
                    exposure=exposure,
                )
            rasterize_time_sec = end_timed_phase(rasterize_started_at)
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if cfg.use_masks and segmentation_masks is not None:
                colors[segmentation_masks<0.5] = 0.0
                pixels[segmentation_masks<0.5] = 0.0
                if expected_depths is not None:
                    expected_depths[segmentation_masks<0.5] = 0.0
                if median_depths is not None:
                    median_depths[segmentation_masks<0.5] = 0.0
                if expected_normals is not None:
                    expected_normals[segmentation_masks<0.5] = 0.0

            if phase == "base" and not ppisp_controller_phase:
                strategy_pre_backward_started_at = begin_timed_phase()
                with record_region("train/strategy_pre_backward"):
                    self.cfg.strategy.step_pre_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=info,
                    )
                strategy_pre_backward_time_sec = end_timed_phase(strategy_pre_backward_started_at)

            # loss
            loss_assembly_started_at = begin_timed_phase()
            with record_region("train/loss"):
                zero_scalar = torch.zeros((), device=device)
                depthloss = zero_scalar
                post_processing_reg_loss = zero_scalar
                segmentation_loss = zero_scalar
                sort_loss_val = zero_scalar
                erank_loss = zero_scalar
                loss_colors = colors
                loss_pixels = pixels

                l1loss = F.l1_loss(loss_colors, loss_pixels)
                ssimloss = 1.0 - fused_ssim(
                    loss_colors.permute(0, 3, 1, 2),
                    loss_pixels.permute(0, 3, 1, 2),
                    padding="valid",
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
                if cfg.depth_loss and not ppisp_controller_phase:
                    # query depths from depth map
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    depths = F.grid_sample(
                        depths.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths = depths.squeeze(3).squeeze(1)  # [1, M]
                    # calculate loss in disparity space
                    disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss * cfg.depth_lambda
                if cfg.post_processing == "bilateral_grid" and not ppisp_controller_phase:
                    post_processing_reg_loss = 10 * total_variation_loss(
                        self.post_processing_module.grids
                    )
                    loss += post_processing_reg_loss
                elif cfg.post_processing == "ppisp" and not ppisp_controller_phase:
                    post_processing_reg_loss = (
                        self.post_processing_module.get_regularization_loss()
                    )
                    loss += post_processing_reg_loss

                # regularizations
                if cfg.opacity_reg > 0.0 and not ppisp_controller_phase:
                    loss = (
                        loss
                        + cfg.opacity_reg
                        * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                    )
                if cfg.scale_reg > 0.0 and not ppisp_controller_phase:
                    loss = (
                        loss
                        + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                    )
                if phase == "sparsify" and self.sparsifier is not None:
                    sparsify_loss = self.sparsifier.penalty(self.splats["opacities"])
                    loss = loss + sparsify_loss
                    sparsify_loss_value = float(sparsify_loss.detach().item())

                eroded_masks = None
                if segmentation_masks is not None and cfg.use_masks and not ppisp_controller_phase:
                    segmentation_loss = torch.sum(alphas * (1.0 - segmentation_masks.unsqueeze(-1))) / ((1.0 - segmentation_masks).sum())
                    eroded_masks = erode_masks(segmentation_masks, kernel_size=3, iterations=1)
                    foreground_loss = 0.1 * torch.sum((1.0 - alphas) * eroded_masks.unsqueeze(-1)) / eroded_masks.sum()
                    loss += segmentation_loss + foreground_loss

                if self.cfg.sort_lambda > 0.0 and cfg.use_sort and not ppisp_controller_phase:
                    sort_loss_val = self.sort_loss()
                    loss += sort_loss_val

                # erank loss
                if cfg.use_erank_loss and step > cfg.erank_start_step and not ppisp_controller_phase:
                    original_scales = torch.exp(self.splats["scales"])
                    s = original_scales * original_scales
                    S = torch.sum(s, dim=-1)
                    q = torch.div(s, S.unsqueeze(dim=-1))
                    H = -torch.sum(q * torch.log(q + 1e-8), dim=-1)
                    erank = torch.exp(H)
                    erank_loss = torch.sum(
                        cfg.erank_lambda * torch.maximum(-torch.log(erank - 1 + 1e-5), torch.zeros_like(erank))
                        + torch.min(original_scales, dim=-1)[0]
                    )
                    loss += erank_loss

                if cfg.use_rade and step> self.cfg.rade_step and not(cfg.with_eval3d or cfg.with_ut) and not ppisp_controller_phase:
                    grid_x, grid_y = torch.meshgrid(torch.arange(width, device=self.device) + 0.5, torch.arange(height, device=self.device) + 0.5, indexing="xy")
                    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(1, -1, 3).float()
                    rays_d = points @ torch.linalg.inv(Ks.transpose(2, 1))  # 1, M, 3
                    points_e = expected_depths.reshape(Ks.shape[0], -1, 1) * rays_d
                    points_m = median_depths.reshape(Ks.shape[0], -1, 1) * rays_d
                    points_e = points_e.reshape_as(expected_normals)
                    points_m = points_m.reshape_as(expected_normals)
                    normal_map_e = torch.zeros_like(points_e)
                    dx = points_e[..., 2:, 1:-1, :] - points_e[..., :-2, 1:-1, :]
                    dy = points_e[..., 1:-1, 2:, :] - points_e[..., 1:-1, :-2, :]
                    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
                    normal_map_e[..., 1:-1, 1:-1, :] = normal_map
                    normal_map_m = torch.zeros_like(points_m)
                    dx = points_m[..., 2:, 1:-1, :] - points_m[..., :-2, 1:-1, :]
                    dy = points_m[..., 1:-1, 2:, :] - points_m[..., 1:-1, :-2, :]
                    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
                    normal_map_m[..., 1:-1, 1:-1, :] = normal_map
                    normal_error_map_e = 1 - (expected_normals * normal_map_e).sum(dim=-1)
                    normal_error_map_m = 1 - (expected_normals * normal_map_m).sum(dim=-1)
                    eroded_masks_binary = eroded_masks > 0.5 if eroded_masks is not None else None
                    if eroded_masks_binary is not None:
                        # Multiply the error map by the safe mask.
                        masked_error_e = normal_error_map_e * eroded_masks_binary
                        masked_error_m = normal_error_map_m * eroded_masks_binary

                        # Compute the mean loss, being careful to divide only by the number of valid pixels.
                        # Add a small epsilon to avoid division by zero if the mask is empty.
                        loss_e = masked_error_e.sum() / (eroded_masks_binary.sum() + 1e-8)
                        loss_m = masked_error_m.sum() / (eroded_masks_binary.sum() + 1e-8)
                    else:
                        loss_e = normal_error_map_e.mean()
                        loss_m = normal_error_map_m.mean()

                    loss += cfg.rade_lambda * (0.4 * loss_e + 0.6 * loss_m)

            loss_assembly_time_sec = end_timed_phase(loss_assembly_started_at)
            backward_started_at = begin_timed_phase()
            with record_region("train/backward"):
                loss.backward()
            backward_time_sec = end_timed_phase(backward_started_at)
            should_refresh_progress = (
                step == init_step
                or step == max_steps - 1
                or (step - init_step) % progress_refresh_every == 0
            )
            if should_refresh_progress:
                loss_value = float(loss.detach().item())
                last_loss_value = loss_value
                desc = (
                    f"phase={phase}| "
                    f"loss={loss_value:.3f}| "
                    f"sh degree={sh_degree_to_use}| "
                )
                if cfg.depth_loss:
                    desc += f"depth loss={depthloss.item():.6f}| "
                if cfg.pose_opt and cfg.pose_noise:
                    # monitor the pose error if we inject noise
                    pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                    desc += f"pose err={pose_err.item():.6f}| "
                if phase == "sparsify":
                    desc += f"sparse loss={sparsify_loss_value:.6f}| "
                pbar.set_description(desc)
            else:
                loss_value = last_loss_value

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.add_scalar(
                    "train/phase",
                    {"base": 0.0, "sparsify": 1.0, "refine": 2.0}[phase],
                    step,
                )
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.post_processing is not None:
                    self.writer.add_scalar(
                        "train/post_processing_reg_loss",
                        post_processing_reg_loss.item(),
                        step,
                    )
                if cfg.use_masks:
                    self.writer.add_scalar("train/segmentation_loss", segmentation_loss.item(), step)
                if cfg.use_erank_loss and step > cfg.erank_start_step:
                    self.writer.add_scalar("train/erank_loss", erank_loss.item(), step)
                if cfg.sort_lambda > 0.0 and cfg.use_sort:
                    self.writer.add_scalar("train/sort_loss", sort_loss_val.item(), step)
                if phase == "sparsify":
                    self.writer.add_scalar(
                        "train/sparsify_loss",
                        sparsify_loss_value,
                        step,
                    )
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            optimizer_started_at = begin_timed_phase()

            with record_region("train/optimize"):
                # Turn Gradients into Sparse Tensor before running optimizer
                if cfg.sparse_grad:
                    assert cfg.packed, "Sparse gradients only work with packed mode."
                    gaussian_ids = info["gaussian_ids"]
                    for k in self.splats.keys():
                        grad = self.splats[k].grad
                        if grad is None or grad.is_sparse:
                            continue
                        self.splats[k].grad = torch.sparse_coo_tensor(
                            indices=gaussian_ids[None],  # [1, nnz]
                            values=grad[gaussian_ids],  # [nnz, ...]
                            size=self.splats[k].size(),  # [N, ...]
                            is_coalesced=len(Ks) == 1,
                        )

                if cfg.visible_adam:
                    gaussian_cnt = self.splats.means.shape[0]
                    if cfg.packed:
                        visibility_mask = torch.zeros_like(
                            self.splats["opacities"], dtype=bool
                        )
                        visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                    else:
                        visibility_mask = (info["radii"] > 0).all(-1).any(0)

                # optimize
                if ppisp_controller_phase:
                    for optimizer in self.optimizers.values():
                        optimizer.zero_grad(set_to_none=True)
                    for optimizer in self.pose_optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    for optimizer in self.app_optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    if cfg.post_processing == "ppisp" and len(self.post_processing_optimizers) > 1:
                        self.post_processing_optimizers[0].zero_grad(set_to_none=True)
                        self.post_processing_optimizers[1].step()
                        for scheduler in ppisp_schedulers:
                            scheduler.step()
                        if ppisp_controller_scheduler is not None:
                            ppisp_controller_scheduler.step()
                        self.post_processing_optimizers[1].zero_grad(set_to_none=True)
                    else:
                        for optimizer in self.post_processing_optimizers:
                            optimizer.zero_grad(set_to_none=True)
                else:
                    for optimizer in self.optimizers.values():
                        if cfg.visible_adam:
                            optimizer.step(visibility_mask)
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    for optimizer in self.pose_optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    for optimizer in self.app_optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    for optimizer in self.post_processing_optimizers:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                    for scheduler in schedulers:
                        scheduler.step()

                # Run post-backward steps after backward and optimizer
                if phase == "base" and not ppisp_controller_phase:
                    if isinstance(self.cfg.strategy, DefaultStrategy):
                        self.cfg.strategy.step_post_backward(
                            params=self.splats,
                            optimizers=self.optimizers,
                            state=self.strategy_state,
                            step=step,
                            info=info,
                            packed=cfg.packed,
                        )
                    elif isinstance(self.cfg.strategy, MCMCStrategy):
                        self.cfg.strategy.step_post_backward(
                            params=self.splats,
                            optimizers=self.optimizers,
                            state=self.strategy_state,
                            step=step,
                            info=info,
                            lr=schedulers[0].get_last_lr()[0],
                        )
                    else:
                        assert_never(self.cfg.strategy)
                elif (
                    phase == "sparsify"
                    and self.sparsifier is not None
                    and phase_step > 0
                    and phase_step % cfg.sparsify_update_every == 0
                ):
                    self.sparsifier.update_state(self.splats["opacities"])

                if schedule.prune_step is not None and step == schedule.prune_step:
                    self.apply_sparsify_prune(step)

            optimizer_and_strategy_time_sec = end_timed_phase(optimizer_started_at)

            if step in save_steps_set or step == max_steps - 1:
                save_started_at = time.perf_counter()
                self.save_training_checkpoint(step, global_tic)
                checkpoint_time_sec = time.perf_counter() - save_started_at

            if (step in ply_steps_set or step == max_steps - 1) and cfg.save_ply:
                ply_started_at = time.perf_counter()
                self.export_current_ply(step, sh_degree_to_use)
                ply_export_time_sec = time.perf_counter() - ply_started_at

            train_iteration_time_sec = time.perf_counter() - step_started_at
            train_iteration_accounted_time_sec = (
                data_prep_time_sec
                + rasterize_time_sec
                + strategy_pre_backward_time_sec
                + loss_assembly_time_sec
                + backward_time_sec
                + optimizer_and_strategy_time_sec
            )
            train_iteration_residual_time_sec = max(
                train_iteration_time_sec - train_iteration_accounted_time_sec, 0.0
            )

            # eval the full set
            if step in eval_steps_set:
                eval_started_at = time.perf_counter()
                self.eval(step)
                eval_time_sec = time.perf_counter() - eval_started_at
                # self.render_traj(step)
                if cfg.should_render_depths:
                    render_depths_started_at = time.perf_counter()
                    self.render_depths(step)
                    render_depths_time_sec = time.perf_counter() - render_depths_started_at
                if cfg.use_rade:
                    recon_started_at = time.perf_counter()
                    self.recon(step)
                    recon_time_sec = time.perf_counter() - recon_started_at

            # run compression
            if cfg.compression is not None and step in eval_steps_set:
                compression_started_at = time.perf_counter()
                self.run_compression(step=step)
                compression_time_sec = time.perf_counter() - compression_started_at

            if not cfg.disable_viewer:
                viewer_started_at = time.perf_counter()
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
                viewer_time_sec = time.perf_counter() - viewer_started_at

            total_step_time_sec = time.perf_counter() - step_started_at
            mem_gb = torch.cuda.max_memory_allocated() / 1024**3
            step_profile = {
                "step": step,
                "phase": phase,
                "phase_step": phase_step,
                "loss": loss_value,
                "sparsify_loss": sparsify_loss_value,
                "num_GS": len(self.splats["means"]),
                "num_train_rays": int(num_train_rays_per_step),
                "mem_gb": mem_gb,
                "elapsed_time_sec": time.time() - global_tic,
                "train_iteration_time_sec": train_iteration_time_sec,
                "data_prep_time_sec": data_prep_time_sec,
                "rasterize_time_sec": rasterize_time_sec,
                "strategy_pre_backward_time_sec": strategy_pre_backward_time_sec,
                "loss_assembly_time_sec": loss_assembly_time_sec,
                "backward_time_sec": backward_time_sec,
                "optimizer_and_strategy_time_sec": optimizer_and_strategy_time_sec,
                "train_iteration_accounted_time_sec": train_iteration_accounted_time_sec,
                "train_iteration_residual_time_sec": train_iteration_residual_time_sec,
                "checkpoint_time_sec": checkpoint_time_sec,
                "ply_export_time_sec": ply_export_time_sec,
                "eval_time_sec": eval_time_sec,
                "render_depths_time_sec": render_depths_time_sec,
                "recon_time_sec": recon_time_sec,
                "compression_time_sec": compression_time_sec,
                "viewer_time_sec": viewer_time_sec,
                "total_step_time_sec": total_step_time_sec,
                "train_steps_per_sec": 1.0 / max(train_iteration_time_sec, 1e-10),
                "train_rays_per_sec": num_train_rays_per_step / max(train_iteration_time_sec, 1e-10),
                "overall_steps_per_sec": 1.0 / max(total_step_time_sec, 1e-10),
                "overall_rays_per_sec": num_train_rays_per_step / max(total_step_time_sec, 1e-10),
            }
            append_jsonl(speed_profile_path, step_profile)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/train_iteration_time_sec", train_iteration_time_sec, step)
                self.writer.add_scalar("train/data_prep_time_sec", data_prep_time_sec, step)
                self.writer.add_scalar("train/rasterize_time_sec", rasterize_time_sec, step)
                self.writer.add_scalar(
                    "train/strategy_pre_backward_time_sec",
                    strategy_pre_backward_time_sec,
                    step,
                )
                self.writer.add_scalar("train/loss_assembly_time_sec", loss_assembly_time_sec, step)
                self.writer.add_scalar("train/backward_time_sec", backward_time_sec, step)
                self.writer.add_scalar(
                    "train/optimizer_and_strategy_time_sec",
                    optimizer_and_strategy_time_sec,
                    step,
                )
                self.writer.add_scalar(
                    "train/train_iteration_residual_time_sec",
                    train_iteration_residual_time_sec,
                    step,
                )
                self.writer.add_scalar("train/total_step_time_sec", total_step_time_sec, step)
                self.writer.add_scalar("train/train_steps_per_sec", step_profile["train_steps_per_sec"], step)
                self.writer.add_scalar("train/train_rays_per_sec", step_profile["train_rays_per_sec"], step)
                self.writer.add_scalar("train/overall_steps_per_sec", step_profile["overall_steps_per_sec"], step)
                self.writer.add_scalar("train/overall_rays_per_sec", step_profile["overall_rays_per_sec"], step)
                self.writer.flush()

            if trace_profiler is not None:
                trace_profiler.step()

        if trace_profiler is not None:
            trace_profiler.stop()

    def sort_loss(self) -> torch.Tensor:
        """Compute sorting loss."""
        if not self.cfg.use_sort or self.cfg.sort_lambda == 0.0:
            return torch.tensor(0.0)

        n_gs = self.splats["means"].shape[0]
        n_sidelen = int(n_gs**0.5)
        if n_sidelen * n_sidelen != n_gs:
            return torch.tensor(0.0)

        keys_to_regularize = ["means", "scales", "quats", "opacities", "sh0"]
        total_loss = torch.tensor(0.0, device=self.device)

        for key in keys_to_regularize:
            tensor = self.splats[key].data

            if key == "means":
                preprocessed_tensor = log_transform(tensor)
            elif key == "quats":
                preprocessed_tensor = F.normalize(tensor, dim=-1)
            elif key == "sh0":
                preprocessed_tensor = tensor.squeeze(1)
            else:
                preprocessed_tensor = tensor
            
            # Ensure tensor is 2D for consistent processing before reshaping
            if preprocessed_tensor.ndim == 1:
                preprocessed_tensor = preprocessed_tensor.unsqueeze(-1)
            
            # Reshape to grid. The -1 will correctly infer the feature dimension.
            grid_img = preprocessed_tensor.reshape(n_sidelen, n_sidelen, -1)
            grid_img = grid_img.permute(2, 0, 1) # (D, H, W)
            grid_img_batched = grid_img.unsqueeze(0) # (1, D, H, W)

            # Blur along X dimension with 'circular' padding
            blurred_x = kornia.filters.gaussian_blur2d(
                grid_img_batched,
                kernel_size=(1, self.cfg.sort_kernel_size),
                sigma=(self.cfg.sort_sigma, self.cfg.sort_sigma),
                border_type="circular",
            )
            # Blur along Y dimension with 'reflect' padding
            blurred_img = kornia.filters.gaussian_blur2d(
                blurred_x,
                kernel_size=(self.cfg.sort_kernel_size, 1),
                sigma=(self.cfg.sort_sigma, self.cfg.sort_sigma),
                border_type="reflect",
            )
            
            loss = F.huber_loss(blurred_img, grid_img_batched)
            total_loss += loss

        return total_loss * self.cfg.sort_lambda / float(len(keys_to_regularize))

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        
        valloader_dataset = self.valset
        if cfg.test_every <= 0 and len(self.valset) == 0:
            print(f"Evaluating on {len(self.trainset)} training images")
            valloader_dataset = self.trainset
        else:
             print(f"Evaluating on {len(self.valset)} validation images")
        
        valloader = torch.utils.data.DataLoader(
            valloader_dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0 if valloader_dataset.on_gpu else 1,
            pin_memory=not valloader_dataset.on_gpu
        )
        
        
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):

            if not valloader_dataset.on_gpu:
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) 
                undistort_masks = data["undistort_mask"].to(device) if "undistort_mask" in data else None
                segmentation_masks = data["segmentation_mask"].to(device) if "segmentation_mask" in data else None
                distortion_params = data.get("distortion_params", None)
                if distortion_params is not None:
                    distortion_params = distortion_params.to(device)
            else:
                camtoworlds = data["camtoworld"]
                Ks = data["K"]
                pixels = data["image"]
                undistort_masks = data.get("undistort_mask")
                segmentation_masks = data.get("segmentation_mask")
                distortion_params = data.get("distortion_params")

            height, width = pixels.shape[1:3]

            # Extract distortion parameters if provided by the dataset
            camera_type_from_data = data.get("camera_type", None)

            # Prepare distortion coefficients for rasterization
            radial_coeffs_to_pass = None
            tangential_coeffs_to_pass = None

            if (cfg.test_every <= 0 or len(self.valset) == 0) and cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, data["image_id"].to(device))

            if not cfg.undistort_colmap_input and distortion_params is not None:
                if cfg.camera_model == "fisheye":
                    # For OPENCV_FISHEYE, COLMAP params are [k1, k2, k3, k4]
                    # The rasterization function expects fisheye radial_coeffs as [batch_size, 4]
                    if distortion_params.shape[-1] == 4:
                        radial_coeffs_to_pass = distortion_params  # Should be [batch_size, 4]
                    else:
                        print(f"Warning: Fisheye model expects 4 distortion params, got {distortion_params.shape[-1]}")
                elif cfg.camera_model == "pinhole":
                    # For pinhole with distortion (OPENCV, RADIAL, SIMPLE_RADIAL)
                    # rasterization expects radial_coeffs [..., C, 6] and tangential_coeffs [..., C, 2]
                    num_params = distortion_params.shape[-1]
                    if num_params >= 1:
                        # Prepare radial coefficients (pad to 6 elements)
                        rad_params = torch.zeros(distortion_params.shape[0], 6, device=distortion_params.device)

                        if num_params == 4:  # OPENCV: [k1, k2, p1, p2]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                            rad_params[:, 1] = distortion_params[:, 1]  # k2
                            # k3-k6 remain zero
                            tangential_coeffs_to_pass = distortion_params[:, [2, 3]].unsqueeze(0)  # [1, C, 2] p1, p2
                        elif num_params == 2:  # RADIAL: [k1, k2, 0, 0] -> extract [k1, k2]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                            rad_params[:, 1] = distortion_params[:, 1]  # k2
                        elif num_params == 1:  # SIMPLE_RADIAL: [k1, 0, 0, 0] -> extract [k1]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                        else:
                            print(f"Warning: Unexpected number of distortion parameters: {num_params}")

                        radial_coeffs_to_pass = rad_params.unsqueeze(0)  # [1, C, 6]

            # Exposure metadata is available for any image with EXIF data (train or val)
            exposure = data["exposure"].to(device) if "exposure" in data else None
            frame_idcs = data["image_id"].to(device)

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=undistort_masks,
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
                frame_idcs=frame_idcs,
                camera_idcs=data["camera_idx"].to(device),
                exposure=exposure,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            
            if segmentation_masks is not None:
                pixels[segmentation_masks<0.5] = 0.0

            canvas_list = [pixels, colors]

            if world_rank == 0:
                

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                psnr_val = self.psnr(colors_p, pixels_p)
                ssim_val = self.ssim(colors_p, pixels_p)
                lpips_val = self.lpips(colors_p, pixels_p)
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["lpips"].append(lpips_val)
                if cfg.test_every > 0 and len(self.valset) > 0:
                    # write images
                    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)

                    # Add metrics text to image
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.fromarray(canvas)
                    draw = ImageDraw.Draw(img)
                    metrics_text = f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}\nLPIPS: {lpips_val:.3f}"
                    # Use a larger font size (48 pixels)
                    font = ImageFont.load_default(size=48)
                    draw.text((10, 10), metrics_text, fill=(255,255,255), font=font)

                    img.save(f"{self.render_dir}/{stage}_step{step}_{i:04d}.png")
                # Compute color-corrected metrics for fair comparison across methods
                if cfg.use_color_correction_metric:
                    if cfg.color_correct_method == "affine":
                        cc_colors = color_correct_affine(colors, pixels)
                    else:
                        cc_colors = color_correct_quadratic(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_color_correction_metric:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()
            if (cfg.test_every <= 0 or len(self.valset) == 0):
                from PIL import Image, ImageDraw, ImageFont
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                img = Image.fromarray(canvas)
                draw = ImageDraw.Draw(img)
                metrics_text = f"PSNR: {stats['psnr']:.2f}\nSSIM: {stats['ssim']:.3f}\nLPIPS: {stats['lpips']:.3f}"
                # Use a larger font size (48 pixels)
                font = ImageFont.load_default(size=48)
                draw.text((10, 10), metrics_text, fill=(255,255,255), font=font)
                
                img.save(f"{self.render_dir}/{stage}_step{step}.png")

    @torch.no_grad()
    def recon(self, step: int):
        """Entry for reconstrution."""
        tic = time.time()
        print("Running reconstrution...")
        cfg = self.cfg
        device = self.device

        trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=0 if self.trainset.on_gpu else 1,
            pin_memory=not self.trainset.on_gpu
        )

        points = self.splats["means"].cpu().numpy()
        max_bound = np.max(points, axis=0)
        min_bound = np.min(points, axis=0)
        size = np.max(max_bound - min_bound)
        desired_resolution = 512
        voxel_size = size / desired_resolution
        print(f"Voxel size: {voxel_size}")
        
        o3d_device, use_open3d_cuda = resolve_open3d_device(self.local_rank)
        
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=voxel_size,
            block_resolution=16,
            block_count=50000,
            device=o3d_device,
        )

        for data in tqdm.tqdm(trainloader, desc="Reconstructing"):
            if not self.trainset.on_gpu:
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device)
                undistort_masks = data["undistort_mask"].to(device) if "undistort_mask" in data else None
                segmentation_masks = data["segmentation_mask"].to(device) if "segmentation_mask" in data else None
            else:
                camtoworlds = data["camtoworld"]
                Ks = data["K"]
                pixels = data["image"]
                undistort_masks = data.get("undistort_mask")
                segmentation_masks = data.get("segmentation_mask")
            
            height, width = pixels.shape[1:3]
            renders, alphas, expected_depths, median_depths, normals, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=undistort_masks,
            )  # [1, H, W, 3]

            depth = median_depths
            depth[alphas < 0.5] = 0
            if segmentation_masks is not None:
                depth[segmentation_masks < 0.5] = 0
            depth = open3d_image_from_torch(
                depth[0, ..., 0].to(dtype=torch.float32),
                o3d_device=o3d_device,
                use_cuda=use_open3d_cuda,
            )
            color = open3d_image_from_torch(
                torch.clamp(renders, min=0, max=1.0)[0].to(dtype=torch.float32),
                o3d_device=o3d_device,
                use_cuda=use_open3d_cuda,
            )
            camera_matrix_device = o3d.core.Device("CPU:0")
            intrinsic = open3d_tensor_from_torch(
                Ks[0].to(dtype=torch.float64),
                o3d_device=camera_matrix_device,
                use_cuda=False,
            )
            extrinsic = open3d_tensor_from_torch(
                torch.linalg.inv(camtoworlds[0]).to(dtype=torch.float64),
                o3d_device=camera_matrix_device,
                use_cuda=False,
            )
            frustum_block_coords = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic, 1.0, 8.0)
            vbg.integrate(frustum_block_coords, depth, color, intrinsic, extrinsic, 1.0, 8.0)

        if use_open3d_cuda:
            o3d.core.cuda.synchronize(o3d_device)
        mesh = vbg.extract_triangle_mesh()
        if use_open3d_cuda:
            o3d.core.cuda.synchronize(o3d_device)
            mesh = mesh.cpu()
            del vbg
            gc.collect()
            o3d.core.cuda.release_cache()
        mesh.compute_vertex_normals()
        legacy_mesh = mesh.to_legacy()
        cluster_ids, num_triangles, _ = legacy_mesh.cluster_connected_triangles()
        if len(num_triangles) == 0:
            print("Warning: Reconstruction produced an empty mesh.")
            return
        cluster_ids = np.asarray(cluster_ids)
        
        # Find the ID of the largest cluster
        largest_cluster_idx = np.argmax(np.asarray(num_triangles))
        
        # Create a boolean mask for triangles that are NOT part of the largest cluster
        triangles_to_remove_mask = (cluster_ids != largest_cluster_idx)
        
        # Use the robust remove_triangles_by_mask method
        legacy_mesh.remove_triangles_by_mask(triangles_to_remove_mask)
        component_mesh = legacy_mesh.remove_unreferenced_vertices()
        component_mesh = component_mesh.compute_vertex_normals()
            

        mesh_dir = f"{cfg.result_dir}/mesh"
        os.makedirs(mesh_dir, exist_ok=True)
        o3d.io.write_triangle_mesh(f"{mesh_dir}/recon_{step}.ply", component_mesh)
        if use_open3d_cuda:
            o3d.core.cuda.synchronize()
            del mesh
            del legacy_mesh
            del component_mesh
            gc.collect()
            o3d.core.cuda.release_cache()
        print(f"Time taken: {time.time() - tic:.2f} seconds")
        print("done!")

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # Get distortion parameters for novel view synthesis (use first camera's params)
        radial_coeffs_to_pass = None
        tangential_coeffs_to_pass = None
        
        if not cfg.undistort_colmap_input and hasattr(self.parser, 'params_dict') and len(self.parser.params_dict) > 0:
            first_camera_id = list(self.parser.params_dict.keys())[0]
            distortion_params = torch.from_numpy(self.parser.params_dict[first_camera_id]).float().to(device)
            
            if cfg.camera_model == "fisheye":
                # For OPENCV_FISHEYE, COLMAP params are [k1, k2, k3, k4]
                if distortion_params.shape[-1] == 4:
                    radial_coeffs_to_pass = distortion_params.unsqueeze(0)  # [1, 4] for single camera
            elif cfg.camera_model == "pinhole":
                # For pinhole with distortion (OPENCV, RADIAL, SIMPLE_RADIAL)
                num_params = distortion_params.shape[-1]
                if num_params >= 1:
                    # Prepare radial coefficients (pad to 6 elements)
                    rad_params = torch.zeros(1, 6, device=distortion_params.device)
                    
                    if num_params == 4:  # OPENCV: [k1, k2, p1, p2]
                        rad_params[0] = distortion_params[0]  # k1
                        rad_params[1] = distortion_params[1]  # k2
                        # k3-k6 remain zero
                        tangential_coeffs_to_pass = distortion_params[[2, 3]].unsqueeze(0)  # [1, 2] p1, p2
                    elif num_params == 2:  # RADIAL: [k1, k2, 0, 0] -> extract [k1, k2]
                        rad_params[0] = distortion_params[0]  # k1
                        rad_params[1] = distortion_params[1]  # k2
                    elif num_params == 1:  # SIMPLE_RADIAL: [k1, 0, 0, 0] -> extract [k1]
                        rad_params[0] = distortion_params[0]  # k1
                        
                    radial_coeffs_to_pass = rad_params.unsqueeze(0)  # [1, 1, 6]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def render_video_traj(self, step: int):
        """Generates a video with a camera shake effect centered on the object."""
        if self.cfg.disable_video:
            return

        image_name = self.cfg.center_and_shake_image_name
        if not image_name:
            print("Error: Please provide an image name for the centering and shake video.")
            return

        print("Running centering and camera shake video rendering...")
        cfg = self.cfg
        device = self.device

        # 1. Calculate the centroid of the Gaussian splats
        means = self.splats["means"].cpu().numpy()
        centroid = np.mean(means, axis=0)

        # 2. Get the initial camera pose to determine the camera's position
        try:
            image_idx = self.parser.image_names.index(image_name)
            initial_pose = self.parser.camtoworlds[image_idx]
        except ValueError as e:
            print(f"Error: {e}")
            print("Please provide a valid image name from the dataset.")
            return

        # 3. Define the new centered camera orientation
        camera_position = initial_pose[:3, 3]

        # Move the camera closer to the object
        camera_position = centroid + cfg.shake_distance_factor * (camera_position - centroid)

        look_dir = centroid - camera_position
        
        # Use a stable 'up' vector by finding the world axis closest to the original 'up'
        initial_up_vector = initial_pose[:3, 1]
        up_index = np.argmax(np.abs(initial_up_vector))
        up_vector = np.eye(3)[up_index] * np.sign(initial_up_vector[up_index])
        
        # This is the central pose for the shake animation
        centered_pose_3x4 = viewmatrix(look_dir, up_vector, camera_position)
        centered_pose = np.eye(4)
        centered_pose[:3, :] = centered_pose_3x4

        # 4. Generate the camera shake path
        shake_poses = []
        radius = cfg.shake_radius
        n_frames = cfg.shake_frames
        
        base_position = centered_pose[:3, 3]
        right_vector = centered_pose[:3, 0]
        up_vector_centered = centered_pose[:3, 1]

        for i in range(n_frames):
            theta = 2.0 * np.pi * i / n_frames
            
            # Calculate the circular offset in the camera's local XY plane
            offset = radius * (np.cos(theta) * right_vector + np.sin(theta) * up_vector_centered)
            new_position = base_position + offset
            
            # Always look at the centroid
            new_look_dir = centroid - new_position
            
            pose_3x4 = viewmatrix(new_look_dir, up_vector, new_position)
            pose_4x4 = np.eye(4)
            pose_4x4[:3, :] = pose_3x4
            shake_poses.append(pose_4x4)
            
        camtoworlds_all = np.stack(shake_poses, axis=0)

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # 5. Render the video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/centered_shake_{step}.mp4", fps=30)

        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering camera shake"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, alphas, _, _, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            renders = renders + (1-alphas) * 1.0
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            canvas = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(canvas)

        writer.close()
        print(f"Video saved to {video_dir}/centered_shake_{step}.mp4")

    @torch.no_grad()
    def render_depths(self, step: int):
        """Renders all training views with depth and normal maps."""
        print("Rendering all training views with depth and normal maps...")
        cfg = self.cfg
        device = self.device

        # New parser and dataset for full resolution images
        render_parser = Parser(
            data_dir=cfg.data_dir,
            factor=1,
            normalize=cfg.normalize_world_space,
            test_every=0,
            undistort_input=cfg.undistort_colmap_input,
            use_masks=cfg.use_masks,
            load_images_in_memory=False,
            optimize_foreground=False,
            exclude_prefixes=cfg.exclude_prefixes,
        )
        render_dataset = Dataset(render_parser, split="train")
        render_loader = torch.utils.data.DataLoader(
            render_dataset, batch_size=1, shuffle=False, num_workers=1
        )

        # render_output_dir = f"{self.render_dir}/train_view_renders_{step}"
        depths_dir = os.path.join(cfg.result_dir, "depths")
        depth_images_dir = os.path.join(cfg.result_dir, "depth_images")
        normals_raw_dir = os.path.join(cfg.result_dir, "normals")
        normal_images_dir = os.path.join(cfg.result_dir, "normal_images")
        os.makedirs(depths_dir, exist_ok=True)
        os.makedirs(depth_images_dir, exist_ok=True)
        os.makedirs(normals_raw_dir, exist_ok=True)
        os.makedirs(normal_images_dir, exist_ok=True)

        for i, data in enumerate(
            tqdm.tqdm(render_loader, desc="Rendering training views")
        ):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device)
            image_name = data["image_name"][0]
            undistort_masks = data["undistort_mask"].to(device) if "undistort_mask" in data else None
            segmentation_masks = data["segmentation_mask"].to(device) if "segmentation_mask" in data else None

            # Get dimensions from actual image data to ensure correctness
            height, width = pixels.shape[1:3]

            distortion_params = data.get("distortion_params", None)
            if distortion_params is not None:
                distortion_params = distortion_params.to(device)

            radial_coeffs_to_pass = None
            tangential_coeffs_to_pass = None

            if not cfg.undistort_colmap_input and distortion_params is not None:
                if cfg.camera_model == "fisheye":
                    if distortion_params.shape[-1] == 4:
                        radial_coeffs_to_pass = distortion_params
                    else:
                        print(
                            f"Warning: Fisheye model expects 4 distortion params, got {distortion_params.shape[-1]}"
                        )
                elif cfg.camera_model == "pinhole":
                    num_params = distortion_params.shape[-1]
                    if num_params >= 1:
                        rad_params = torch.zeros(
                            distortion_params.shape[0], 6, device=distortion_params.device
                        )
                        if num_params == 4:
                            rad_params[:, 0] = distortion_params[:, 0]
                            rad_params[:, 1] = distortion_params[:, 1]
                            tangential_coeffs_to_pass = distortion_params[
                                :, [2, 3]
                            ].unsqueeze(0)
                        elif num_params == 2:
                            rad_params[:, 0] = distortion_params[:, 0]
                            rad_params[:, 1] = distortion_params[:, 1]
                        elif num_params == 1:
                            rad_params[:, 0] = distortion_params[:, 0]
                        else:
                            print(
                                f"Warning: Unexpected number of distortion parameters: {num_params}"
                            )
                        radial_coeffs_to_pass = rad_params.unsqueeze(0)

            _, alphas, _, median_depths, expected_normals, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
                masks=undistort_masks,
            )
            median_depths = median_depths * (alphas > 0.5)
            expected_normals = expected_normals * (alphas > 0.5)
            if cfg.use_masks:
                expected_normals[segmentation_masks<0.5] = 0.0
                median_depths[segmentation_masks<0.5] = 0.0
                

            # Save depth as .npy
            depth_img = median_depths.squeeze(0).cpu().numpy()
            depth_filename_npy = os.path.splitext(image_name)[0] + ".npy"
            depth_output_path_npy = os.path.join(depths_dir, depth_filename_npy)
            os.makedirs(os.path.dirname(depth_output_path_npy), exist_ok=True)
            np.save(depth_output_path_npy, depth_img)

            # Save colormapped depth
            # Normalize between 0 and 95th percentile to handle outliers
            valid_depths = depth_img[depth_img > 0]
            if valid_depths.size > 0:
                p95 = np.percentile(valid_depths, 95)
                depth_img_normalized = (depth_img - valid_depths.min()) / (
                    p95 - valid_depths.min() + 1e-10
                )
                depth_img_normalized = np.clip(depth_img_normalized, 0, 1)
            else:
                depth_img_normalized = np.zeros_like(depth_img)

            depth_colormap = apply_float_colormap(
                torch.from_numpy(depth_img_normalized), "viridis"
            ).numpy()
            depth_filename_jpg = os.path.splitext(image_name)[0] + ".jpg"
            depth_output_path_jpg = os.path.join(depth_images_dir, depth_filename_jpg)
            os.makedirs(os.path.dirname(depth_output_path_jpg), exist_ok=True)
            imageio.imwrite(
                depth_output_path_jpg,
                (depth_colormap * 255).astype(np.uint8),
                quality=90,
            )

            # Save raw normal map as .npy
            normals_img = expected_normals.squeeze(0).cpu().numpy()
            normal_filename_npy = os.path.splitext(image_name)[0] + ".npy"
            normal_output_path_npy = os.path.join(normals_raw_dir, normal_filename_npy)
            os.makedirs(os.path.dirname(normal_output_path_npy), exist_ok=True)
            np.save(normal_output_path_npy, normals_img)

            # Save normalized normal map as image
            normals_img_to_save = normals_img * 0.5 + 0.5
            normal_filename_jpg = os.path.splitext(image_name)[0] + ".jpg"
            normal_output_path_jpg = os.path.join(
                normal_images_dir, normal_filename_jpg
            )
            os.makedirs(os.path.dirname(normal_output_path_jpg), exist_ok=True)
            imageio.imwrite(
                normal_output_path_jpg,
                (normals_img_to_save * 255).astype(np.uint8),
                quality=90,
            )

        print(f"Training views rendered and saved to {cfg.result_dir}")

    @torch.no_grad()
    def export_ppisp_reports(self) -> None:
        """Export PPISP visualization reports (PDF) and parameter JSON."""
        if self.cfg.post_processing != "ppisp":
            return
        print("Exporting PPISP reports...")

        # Compute frames per camera from training dataset
        num_cameras = self.parser.num_cameras
        frames_per_camera = [0] * num_cameras
        for idx in self.trainset.indices:
            cam_idx = self.parser.camera_indices[idx]
            frames_per_camera[cam_idx] += 1

        # Generate camera names from COLMAP camera IDs
        # camera_id_to_idx maps COLMAP ID -> 0-based index
        idx_to_camera_id = {v: k for k, v in self.parser.camera_id_to_idx.items()}
        camera_names = [f"camera_{idx_to_camera_id[i]}" for i in range(num_cameras)]

        # Export reports
        output_dir = Path(self.cfg.result_dir) / "ppisp_reports"
        pdf_paths = export_ppisp_report(
            self.post_processing_module,
            frames_per_camera,
            output_dir,
            camera_names=camera_names,
        )
        print(f"PPISP reports saved to {output_dir}")
        for path in pdf_paths:
            print(f"  - {path.name}")

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, _, _, _, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders

    @torch.no_grad()
    def save_colmap_reconstruction(self):
        """Save the refined camera poses to a new COLMAP reconstruction."""
        

        colmap_dir = os.path.join(self.parser.data_dir, "sparse/pose_opt_colmap/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(self.parser.data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(self.parser.data_dir, "sparse")

        # Read the original model as a base
        cameras, images, points3D = read_model(path=Path(colmap_dir))

        # Filter the model to match the one used by the parser (respecting exclusions etc.)
        parser_image_names = set(self.parser.image_names)
        images_in_parser = {
            img_id: img
            for img_id, img in images.items()
            if img.name in parser_image_names
        }
        kept_image_ids = {img.id for img in images_in_parser.values()}

        points3D_in_parser = {}
        for p_id, p in points3D.items():
            obs_by_kept = [img_id for img_id in p.image_ids if img_id in kept_image_ids]
            if len(obs_by_kept) > 0:
                new_point2D_idxs = [
                    p.point2D_idxs[i]
                    for i, img_id in enumerate(p.image_ids)
                    if img_id in kept_image_ids
                ]
                points3D_in_parser[p_id] = p._replace(
                    image_ids=np.array(obs_by_kept),
                    point2D_idxs=np.array(new_point2D_idxs),
                )

        used_camera_ids = {img.camera_id for img in images_in_parser.values()}
        cameras_in_parser = {
            cam_id: cam for cam_id, cam in cameras.items() if cam_id in used_camera_ids
        }

        # If the scene was normalized, transform the point cloud to the normalized space
        # to match the camera poses.
        if self.parser.normalize:
            p_xyz = np.array([p.xyz for p in points3D_in_parser.values()])
            p_xyz_transformed = transform_points(self.parser.transform, p_xyz)
            for i, p_id in enumerate(points3D_in_parser.keys()):
                points3D_in_parser[p_id] = points3D_in_parser[p_id]._replace(
                    xyz=p_xyz_transformed[i]
                )

        # Create a mapping from image name to COLMAP image ID
        name_to_id = {im.name: im.id for im in images_in_parser.values()}

        updated_images = images_in_parser.copy()

        pose_adjust_model = (
            self.pose_adjust.module if self.world_size > 1 else self.pose_adjust
        )

        print("Updating camera poses for training images...")
        for i in tqdm.tqdm(range(len(self.trainset))):
            data = self.trainset[i]

            # Get original camera-to-world matrix and the optimizer id
            original_c2w = data["camtoworld"].to(self.device)
            optimizer_id = torch.tensor([data["image_id"]], device=self.device)

            # Get refined camera pose (which is already in the normalized space)
            refined_c2w = pose_adjust_model(original_c2w.unsqueeze(0), optimizer_id)
            refined_c2w = refined_c2w.squeeze(0).cpu().numpy()

            # Convert back to world-to-camera quaternion and translation vector
            w2c_mat = np.linalg.inv(refined_c2w)
            R = w2c_mat[:3, :3]
            t = w2c_mat[:3, 3]
            qvec = rotmat2qvec(R)
            tvec = t

            # Find the corresponding image in the COLMAP model
            global_idx = self.trainset.indices[i]
            image_name = self.parser.image_names[global_idx]
            if image_name in name_to_id:
                colmap_id = name_to_id[image_name]
                image_data = updated_images[colmap_id]

                # Update the image data
                updated_images[colmap_id] = image_data._replace(qvec=qvec, tvec=tvec)
            else:
                print(f"Warning: could not find image {image_name} in COLMAP model.")

        print("Updating camera poses for validation images...")
        for i in tqdm.tqdm(range(len(self.valset))):
            global_idx = self.valset.indices[i]
            image_name = self.parser.image_names[global_idx]

            if image_name in name_to_id:
                colmap_id = name_to_id[image_name]
                # The camera poses from the parser are already normalized.
                c2w = self.parser.camtoworlds[global_idx]
                if torch.is_tensor(c2w):
                    c2w = c2w.cpu().numpy()
                
                w2c_mat = np.linalg.inv(c2w)
                R = w2c_mat[:3, :3]
                t = w2c_mat[:3, 3]
                qvec = rotmat2qvec(R)
                tvec = t
                
                image_data = updated_images[colmap_id]
                updated_images[colmap_id] = image_data._replace(qvec=qvec, tvec=tvec)

                
        save_path = self.save_colmap_path
        print(f"Saving COLMAP reconstruction with refined poses to {save_path}")
        os.makedirs(save_path, exist_ok=True)

        print(f"Saving updated COLMAP model to {save_path}")
        # Save as text for easy inspection
        write_model(
            cameras_in_parser,
            updated_images,
            points3D_in_parser,
            path=Path(save_path),
        )
        print("Done.")


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    # Import post-processing modules based on configuration
    # These imports must be here (not in __main__) for distributed workers
    if cfg.post_processing == "bilateral_grid":
        global BilateralGrid, slice, total_variation_loss
        if cfg.bilateral_grid_fused:
            from fused_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
        else:
            from lib_bilagrid import (
                BilateralGrid,
                slice,
                total_variation_loss,
            )
    elif cfg.post_processing == "ppisp":
        global PPISP, PPISPConfig, export_ppisp_report
        from ppisp import PPISP, PPISPConfig
        from ppisp.report import export_ppisp_report

    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.eval_only and (cfg.ckpt is not None or cfg.ply_file is not None):
        # run eval only
        if cfg.ckpt is not None:
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=True)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            if runner.post_processing_module is not None:
                pp_state = ckpts[0].get("post_processing")
                if pp_state is not None:
                    runner.post_processing_module.load_state_dict(pp_state)
            step = ckpts[0]["step"]
        else:
            step = 29999
        # runner.eval(step=step)
        if cfg.center_and_shake_image_name:
            runner.render_video_traj(step)
        else:
            runner.render_traj(step=step)
        if cfg.should_render_depths:
            runner.render_depths(step=step)
        if cfg.use_rade:
            runner.recon(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()
        if runner.cfg.pose_opt:
            runner.save_colmap_reconstruction()
        runner.export_ppisp_reports()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.ply_file and not os.path.exists(cfg.ply_file):
        raise FileNotFoundError(f"PLY file not found: {cfg.ply_file}")

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
