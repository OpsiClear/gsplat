"""
4DGS Trainer with frozen Static + trainable Dynamic Gaussians.

Extends simple_trainer_4dgs.py to support two separate PLY inputs:

  static_ply_path  — Background Gaussians loaded once, NEVER trained.
                     Rendered at every step with no gradient.
  dynamic_ply_path — Foreground Gaussians trained with deformation field.
                     Densification applies only to these.

Architecture:
  Static Gaussians (pre-activated, frozen)  <- loaded from static_ply_path
  + Dynamic Gaussians (canonical, trainable) <- loaded from dynamic_ply_path
  + DeformationField(dynamic_xyz, t) -> (delta_xyz, delta_rot, delta_scale)
  -> Combined rasterization [dynamic | static] -> RGB
  -> Backprop only through dynamic path

Key design:
  - Static Gaussians stored outside self.splats as plain GPU tensors (.detach()).
  - Dynamic Gaussians placed FIRST in concatenated array so strategy info slicing
    [..., :N_dynamic, :] is always correct.
  - DefaultStrategy (densification) only sees N_dynamic entries via sliced info dict.

Usage:
  python simple_trainer_static_dynamic.py \\
      --data_dir /data/scene/ \\
      --result_dir results/scene_static_dynamic/ \\
      --static_ply_path /path/to/static.ply \\
      --dynamic_ply_path /path/to/dynamic.ply \\
      --coarse_iters 0 \\
      --num_frames 300
"""

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from fused_ssim import fused_ssim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from datasets.colmap import Dataset, Parser
from datasets.traj import generate_interpolated_path
from utils import knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.io_ply import import_splats, sh2rgb
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

from deformation import (
    DeformationField,
    apply_deformation,
    plane_tv_loss,
    time_smoothness_loss,
    time_smoothness_loss_1st,
    l1_time_planes_loss,
)


# ---------------------------------------------------------------------------
# Dataset utilities for dynamic scenes
# ---------------------------------------------------------------------------

def _parse_frame_idx(image_name: str) -> int:
    """Extract frame index from image name like '002-002/000042.jpg' → 42."""
    import re
    stem = Path(image_name).stem
    # Bare number stem: '000042' → 42
    if re.match(r"^\d+$", stem):
        return int(stem)
    # frame_NNNNNN pattern
    m = re.search(r"frame_(\d+)", image_name)
    if m:
        return int(m.group(1))
    return 0


class DynamicDataset(Dataset):
    """
    Extends the standard colmap Dataset to return normalized timestamps.

    Timestamps are derived from the frame number in the image filename.
    E.g., '002-002/000042.jpg' → frame_idx=42.

    Args:
        parser: ColmapParser instance (loaded with frame_num=None to include all frames).
        split: "train" or "val".
        num_frames: Total unique timesteps for normalization (t = frame_idx / (N-1)).
        selected_frames: If given, only include images whose frame_idx is in this set.
        patch_size: Optional random crop size.
    """

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        num_frames: int = 1,
        selected_frames: Optional[List[int]] = None,
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        super().__init__(parser, split=split, patch_size=patch_size, load_depths=load_depths)
        self.num_frames = max(num_frames, 1)

        # If frame selection requested, filter self.indices
        if selected_frames is not None:
            frame_set = set(selected_frames)
            self.indices = np.array([
                idx for idx in self.indices
                if _parse_frame_idx(parser.image_names[idx]) in frame_set
            ])
            print(f"[DynamicDataset] {split}: {len(self.indices)} images after frame filter "
                  f"({len(selected_frames)} frames × cameras)")

    def __getitem__(self, item: int) -> Dict[str, Any]:
        data = super().__getitem__(item)
        index = self.indices[item]
        image_name = self.parser.image_names[index]
        frame_idx = _parse_frame_idx(image_name)
        # Centered time: [-0.5, 0.5] so canonical frame is mid-sequence
        t = float(np.clip(frame_idx / max(self.num_frames - 1, 1), 0.0, 1.0)) - 0.5
        data["timestamp"] = torch.tensor(t, dtype=torch.float32)
        return data


class DynamicRigDataset(torch.utils.data.Dataset):
    """
    Dataset for a fixed multi-camera rig with dynamic content.

    Camera poses are taken from a reference-frame Parser (frame_num=0 or similar).
    Images are loaded from any selected frame for each camera.
    Timestamps are assigned as t = frame_rank / (num_selected_frames - 1) - 0.5
    (centered at 0 so the canonical/keyframe is mid-sequence).

    This is the right approach for rigs like the elly dataset where:
      - 64 cameras have fixed poses (calibrated once from COLMAP)
      - 900 frames exist per camera
      - We select a subset of frames for 4DGS training

    Args:
        ref_parser: ColmapParser loaded with frame_num=0 (or reference frame).
                    Provides camera poses and intrinsics for all 64 cameras.
        image_dir: Path to the images root directory (contains camera subdirs).
        selected_frames: Sorted list of frame indices to use (e.g. [0,10,20,...,890]).
        split: "train" or "val".
        test_every: 1-in-N cameras held out for validation (set high to keep most for train).
        factor: Image downscale factor.
        patch_size: Optional random crop.
    """

    def __init__(
        self,
        ref_parser: Parser,
        image_dir: str,
        selected_frames: List[int],
        split: str = "train",
        test_every: int = 0,
        factor: int = 1,
        patch_size: Optional[int] = None,
        mask_dir: Optional[str] = None,
        invert_masks: bool = False,
        val_num_cameras: int = 0,
        val_num_frames: int = 0,
    ):
        import imageio.v2 as iio
        self.ref_parser = ref_parser
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.invert_masks = invert_masks
        self.selected_frames = sorted(selected_frames)
        self.num_frames = len(selected_frames)
        self.factor = factor
        self.patch_size = patch_size
        self._iio = iio

        # Build list of (camera_idx, frame_rank) pairs.
        # Train always uses ALL cameras × ALL frames.
        # Val picks a small subset: val_num_cameras evenly-spaced cameras ×
        # val_num_frames random frames, so eval is fast (e.g. 5×5 = 25 images).
        num_cameras = len(ref_parser.image_names)
        cam_indices = np.arange(num_cameras)

        if split == "val" and val_num_cameras > 0 and val_num_cameras < num_cameras:
            # Pick evenly-spaced cameras
            cam_indices = np.linspace(0, num_cameras - 1, val_num_cameras).astype(int)
        elif test_every > 1 and split == "val":
            cam_indices = cam_indices[cam_indices % test_every == 0]

        # Select frames for this split
        if split == "val" and val_num_frames > 0 and val_num_frames < self.num_frames:
            # Pick evenly-spaced frames from the selected set
            frame_indices = np.linspace(0, self.num_frames - 1, val_num_frames).astype(int)
            val_frames = [(int(frame_indices[i]), self.selected_frames[frame_indices[i]])
                          for i in range(len(frame_indices))]
        else:
            val_frames = list(enumerate(self.selected_frames))

        # All combinations of selected cameras × selected frames
        self.items = []
        for cam_idx in cam_indices:
            for frame_rank, frame_idx in val_frames:
                self.items.append((int(cam_idx), frame_rank, frame_idx))

        n_cams = len(cam_indices)
        n_frames = len(val_frames)
        print(f"[DynamicRigDataset] {split}: {n_cams} cameras × "
              f"{n_frames} frames = {len(self.items)} images")

    def _detect_frame_format(self):
        """Auto-detect frame filename format from COLMAP image names.

        COLMAP stores full paths like 'cam00/00001.png' or 'take_18_cam_01/000001.jpg'.
        We extract the basename to determine digit count and extension.
        """
        import re
        # Use COLMAP image name to determine extension and digit format
        cam_name = self.ref_parser.image_names[0]  # e.g. "cam00/00001.png"
        basename = os.path.basename(cam_name)        # e.g. "00001.png"
        name, ext = os.path.splitext(basename)
        # Check if basename is purely numeric (frame file) or a reference name
        if re.fullmatch(r'\d+', name):
            n_digits = len(name)
            self._frame_fmt = f"0{n_digits}d"
            self._frame_ext = ext
        else:
            # Reference frame name isn't numeric (e.g. "cam00.png").
            # Scan the actual image directory to detect format.
            cam_dir = os.path.dirname(cam_name)
            sample_dir = os.path.join(self.image_dir, cam_dir)
            files = sorted(f for f in os.listdir(sample_dir) if re.match(r'\d+\.', f))
            if files:
                fname, fext = os.path.splitext(files[0])
                self._frame_fmt = f"0{len(fname)}d"
                self._frame_ext = fext
            else:
                self._frame_fmt = "06d"
                self._frame_ext = ext
        print(f"[DynamicRigDataset] Frame format: {self._frame_fmt} ext={self._frame_ext}")

    def indices_within_time_radius(self, radius: float) -> list:
        """Return dataset indices for items whose |timestamp| <= radius.
        Symmetric expansion from center t=0 (old/default behavior)."""
        indices = []
        for i, (cam_idx, frame_rank, frame_idx) in enumerate(self.items):
            t = frame_rank / max(self.num_frames - 1, 1) - 0.5
            if abs(t) <= radius:
                indices.append(i)
        return indices

    def indices_within_time_window(self, t_start: float, t_end: float) -> list:
        """Return dataset indices for items whose timestamp is in [t_start, t_end].
        Used for forward-only progressive sampling starting from frame 0."""
        indices = []
        for i, (cam_idx, frame_rank, frame_idx) in enumerate(self.items):
            t = frame_rank / max(self.num_frames - 1, 1) - 0.5
            if t_start <= t <= t_end:
                indices.append(i)
        return indices

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        cam_idx, frame_rank, frame_idx = self.items[item]

        # Camera intrinsics and pose from reference parser
        cam_name = self.ref_parser.image_names[cam_idx]
        cam_dir = os.path.dirname(cam_name)  # e.g. "002-002"

        # Auto-detect frame filename format on first access
        if not hasattr(self, "_frame_fmt"):
            self._detect_frame_format()
        ext = self._frame_ext
        frame_name = f"{frame_idx:{self._frame_fmt}}{ext}"

        image_path = os.path.join(self.image_dir, cam_dir, frame_name)

        # Load image
        image = self._iio.imread(image_path)[..., :3].astype(np.float32) / 255.0

        if self.factor > 1:
            from PIL import Image as PILImage
            h, w = image.shape[:2]
            new_h, new_w = h // self.factor, w // self.factor
            image = np.array(
                PILImage.fromarray((image * 255).astype(np.uint8)).resize(
                    (new_w, new_h), PILImage.BICUBIC
                )
            ).astype(np.float32) / 255.0

        # Load mask if mask_dir is provided
        # White (255) = object/keep, Black (0) = ignore
        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, cam_dir, frame_name)
            if os.path.exists(mask_path):
                mask = self._iio.imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[..., 0]  # take first channel if RGB
                mask = (mask > 127).astype(np.float32)  # binary: 1=keep, 0=ignore
                if self.invert_masks:
                    mask = 1.0 - mask
                if self.factor > 1:
                    from PIL import Image as PILImage
                    h, w = mask.shape[:2]
                    new_h, new_w = h // self.factor, w // self.factor
                    mask = np.array(
                        PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
                            (new_w, new_h), PILImage.NEAREST
                        )
                    ).astype(np.float32) / 255.0

        # Random patch crop
        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, w - self.patch_size + 1)
            y = np.random.randint(0, h - self.patch_size + 1)
            image = image[y:y+self.patch_size, x:x+self.patch_size]
            if mask is not None:
                mask = mask[y:y+self.patch_size, x:x+self.patch_size]

        # Camera extrinsics (c2w) from reference parser
        camtoworld = self.ref_parser.camtoworlds[cam_idx]

        # Camera intrinsics
        cam_id = self.ref_parser.camera_ids[cam_idx]
        K = self.ref_parser.Ks_dict[cam_id].copy()
        if self.factor > 1:
            K[:2] /= self.factor

        # Timestamp: centered at 0 so canonical frame is mid-sequence
        # Range: [-0.5, 0.5] instead of [0, 1]
        t = frame_rank / max(self.num_frames - 1, 1) - 0.5

        result = {
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "K": torch.from_numpy(K).float(),
            "image": torch.from_numpy(image).float(),
            "timestamp": torch.tensor(t, dtype=torch.float32),
            "image_id": torch.tensor(item, dtype=torch.long),
            "cam_idx": torch.tensor(cam_idx, dtype=torch.long),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.long),
        }
        if mask is not None:
            result["mask"] = torch.from_numpy(mask).float()  # [H, W]
        return result


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # ---- Output / Checkpointing ----
    disable_viewer: bool = True
    ckpt: Optional[List[str]] = None
    resume_deform_ckpt: Optional[str] = None
    render_traj_path: str = "interp"

    # ---- Data ----
    data_dir: str = "data/scene"
    data_factor: int = 1
    result_dir: str = "results/scene_4dgs"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = False
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    port: int = 8080
    load_images_in_memory: bool = False
    undistort_colmap_input: bool = True
    use_masks: bool = False
    # Path to mask directory (mirrors image_dir structure: mask_dir/cam_dir/frame.jpg).
    # White=keep, Black=ignore. Loss computed only on white pixels.
    mask_dir: Optional[str] = None
    # Invert masks (Black=keep, White=ignore). Use when mask convention is opposite.
    invert_masks: bool = False

    # Total number of timesteps in the video (used for timestamp normalization).
    # Set to the number of frames extracted from your video.
    num_frames: int = 1

    # ---- Training Schedule ----
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    save_ply: bool = True
    ply_steps: List[int] = field(default_factory=lambda: [15_000, 30_000])
    export_per_frame_ply: bool = False
    disable_video: bool = False
    render_train_views: bool = False

    # ---- Gaussian Initialization ----
    # Options: "sfm" (COLMAP sparse points, default/recommended),
    #          "random" (random points within extent),
    #          "ply" (load pre-built PLY — use coarse_iters=0 to skip coarse phase)
    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    ply_path: Optional[str] = None
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0

    # ---- Loss ----
    ssim_lambda: float = 0.0
    near_plane: float = 0.01
    far_plane: float = 1e10
    random_bkgd: bool = False

    # ---- Gaussian Optimizer LRs ----
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # ---- Densification ----
    # Reference 4DGS: densify iters 500-15K across both phases, hard cap at 360K Gaussians.
    # We densify through fine phase too so dynamic regions get proper coverage.
    strategy: DefaultStrategy = field(default_factory=lambda: DefaultStrategy(
        refine_stop_iter=15_000,
        reset_every=3_000,
    ))
    # Max number of Gaussians (reference 4DGS caps at 360K)
    max_num_gaussians: int = 360_000
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False
    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    # ---- 4DGS Deformation Field ----
    use_deformation: bool = True
    # Number of coarse-phase iters where deformation is OFF (static 3DGS warm-up)
    coarse_iters: int = 3_000
    # During coarse phase, train on single frame (True) or all frames (False).
    # Reference 4DGS default: False (train all frames, no deformation → learns "average" pose).
    # This makes canonical Gaussians a centroid so deformation only needs small deltas.
    # True (zerostamp_init): only canonical frame → avoids ghosting but biases to one pose.
    coarse_single_frame: bool = False
    # AABB margin: fraction of point-cloud extent added on each side.
    # Default 0.1 → AABB = 1.2× point-cloud. Set to 1.3 to triple the AABB.
    aabb_margin: float = 0.1
    # HexPlane grid config
    deform_grid_resolution: int = 64
    deform_time_resolution: int = 150
    deform_feature_dim: int = 32
    deform_multires: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # MLP config. Reference DyNeRF uses depth=0 (no backbone, grid→heads directly), width=128.
    # depth=0 works well with high multires. depth=1+ adds a shared backbone.
    deform_net_width: int = 128
    deform_net_depth: int = 0
    # Deformation head flags (always: Δxyz, Δrot, Δscale)
    enable_opacity_deform: bool = False   # Δopacity head (usually OFF)
    enable_sh_deform: bool = False        # ΔSH head (usually OFF, expensive)
    # SH degree for the deformation SH head (-1 = use cfg.sh_degree).
    # Use a lower degree than sh_degree to save memory/compute while still
    # capturing low-frequency time-varying color changes.
    # 0→3 dims, 1→12 dims, 2→27 dims, 3→48 dims (for sh_degree=3 rendering)
    deform_sh_degree: int = -1
    # Time positional encoding bands (0=disabled, 6=12 extra dims for high-freq time signal)
    deform_time_pe_bands: int = 0
    # Per-head activation functions. Choices: relu, elu, sine, sinerelu, leakyrelu, siren
    deform_act_xyz: str = "relu"
    deform_act_rot: str = "relu"
    deform_act_scale: str = "relu"
    deform_act_sh: str = "relu"
    # Deformation optimizer LRs
    deform_lr: float = 1.6e-4
    grid_lr: float = 1.6e-3
    # Regularization (active only after coarse_iters)
    plane_tv_weight: float = 0.0001
    time_smooth_weight: float = 0.01
    time_smooth_weight_final: float = -1.0  # Final weight after annealing (-1=no annealing)
    time_smooth_order: int = 2            # 1=velocity penalty (allows fast motion), 2=acceleration penalty
    l1_time_planes_weight: float = 0.0001  # Keeps temporal planes near 1 (identity for product)
    # LR warmup: ramp deformation LR from lr_delay_mult * lr to lr over first warmup steps
    deform_lr_delay_mult: float = 0.01    # Match reference 4DGS: start at 1% of target LR
    deform_lr_warmup_steps: int = 1000    # Warmup over first 1000 steps of fine phase
    # Standard constraint: at t=0, penalize deformation deviating from identity.
    # Reference 4DGS applies (deformed - canonical)^2 at t=0 to keep canonical pose stable.
    # Annealed from weight_constraint_init → weight_constraint_after over weight_constraint_decay iters.
    weight_constraint_init: float = 1.0
    weight_constraint_after: float = 0.2
    weight_constraint_decay_iters: int = 5000

    # ---- Progressive frame sampling ----
    # During fine phase, start training on frames near the keyframe (t=0)
    # and gradually expand to the full time range. This avoids the difficulty
    # of predicting long-range motions at the start of deformation learning.
    # 0 = disabled (use all frames immediately).
    progressive_time_warmup: int = 0
    # Initial time radius around keyframe (t=0). 0.1 = ±10% of sequence.
    progressive_time_initial: float = 0.1
    # If True: start from frame 0 (t=-0.5) and expand forward only.
    # Window grows from [−0.5, −0.5+initial] → [−0.5, +0.5] over warmup steps.
    # Use when canonical PLY = frame 0, so the model first masters the canonical
    # then gradually learns forward deformations.
    # If False (default): symmetric expansion from center t=0 (old behavior).
    progressive_time_forward: bool = False

    # ---- Dataset mode ----
    # "standard": use ColmapParser + DynamicDataset (frame_num=None loads all frames)
    # "rig": use ColmapParser with frame_num=0 + DynamicRigDataset (fixed-camera rig)
    #        Best for multi-camera rigs like elly where cameras don't move.
    dataset_mode: Literal["standard", "rig"] = "standard"
    # For "rig" mode: stride between selected frames (stride=10 → every 10th frame)
    frame_stride: int = 10
    # For "rig" mode: starting frame index
    frame_start: int = 0
    # For "rig" mode: cam/val split frequency (0=all train, 8=every 8th cam is val)
    rig_test_every: int = 8
    # Validation subset size: N cameras × M frames (0 = use all)
    val_num_cameras: int = 5
    val_num_frames: int = 5

    # ---- Logging ----
    tb_every: int = 100
    # Log GT+render comparison image to tensorboard every N steps (0 = disabled)
    tb_image_every: int = 200
    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncentered transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # ---- Static + Dynamic PLY ----
    # static_ply_path: frozen background Gaussians, never trained, always rendered.
    # dynamic_ply_path: trainable foreground Gaussians (overrides init_type/ply_path).
    #   Set coarse_iters=0 when using pre-trained PLYs (no warm-up needed).
    static_ply_path: Optional[str] = None
    dynamic_ply_path: Optional[str] = None

    # ---- RobustNeRF inlier mask (Sabour et al., CVPR 2023 variant) ----
    # Learnable per-pixel outlier rejection active after max_steps // 2.
    # τ (outlier fraction) is a trainable scalar; threshold ρ is quantile(R_dyn, 1−τ).
    # Hard mask forward, soft sigmoid backward (straight-through estimator).
    # Applied only within the projected dynamic AABB (static pixels always kept).
    mask_lr: float = 1e-4             # LR for τ_raw parameter
    mask_max_tau: float = 0.10        # Cap on τ: at most 10% outliers
    mask_beta: float = 50.0           # Sigmoid sharpness for STE (higher → closer to hard)
    mask_coverage_target: float = 0.97  # Minimum inlier fraction to enforce
    mask_coverage_lambda: float = 1.0   # Coverage penalty weight
    # Step at which RobustNeRF mask activates. -1 = max_steps // 2 (default).
    mask_start_step: int = -1

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.coarse_iters = int(self.coarse_iters * factor)
        s = self.strategy
        s.refine_start_iter = int(s.refine_start_iter * factor)
        s.refine_stop_iter = int(s.refine_stop_iter * factor)
        s.reset_every = int(s.reset_every * factor)
        s.refine_every = int(s.refine_every * factor)


# ---------------------------------------------------------------------------
# Splat initialization (identical to simple_trainer.py)
# ---------------------------------------------------------------------------

def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
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
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
    ply_path: Optional[str] = None,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "ply":
        if ply_path is None:
            raise ValueError("ply_path must be provided when init_type='ply'")
        print(f"[4DGS] Loading canonical Gaussians from PLY: {ply_path}")
        points, scales, quats, opacities, sh0, shN = import_splats(ply_path, device)
        points = points[world_rank::world_size]
        scales = scales[world_rank::world_size]
        quats = quats[world_rank::world_size]
        opacities = opacities[world_rank::world_size]
        sh0 = sh0[world_rank::world_size]
        shN = shN[world_rank::world_size]
        N = points.shape[0]
        # Preserve all SH coefficients from PLY, matching target sh_degree
        target_shN_count = (sh_degree + 1) ** 2 - 1
        loaded_shN_count = shN.shape[1]
        if loaded_shN_count >= target_shN_count:
            shN_use = shN[:, :target_shN_count, :]
        else:
            shN_use = torch.zeros((N, target_shN_count, 3), device=device)
            shN_use[:, :loaded_shN_count, :] = shN
        print(f"[4DGS] PLY loaded: {N} Gaussians, SH bands: {loaded_shN_count} loaded / {target_shN_count} target")
        params = [
            ("means", torch.nn.Parameter(points), means_lr * scene_scale),
            ("scales", torch.nn.Parameter(scales), scales_lr),
            ("quats", torch.nn.Parameter(quats), quats_lr),
            ("opacities", torch.nn.Parameter(opacities), opacities_lr),
            ("sh0", torch.nn.Parameter(sh0), sh0_lr),
            ("shN", torch.nn.Parameter(shN_use), shN_lr),
        ]
    else:
        if init_type == "sfm":
            points = torch.from_numpy(parser.points).float()
            rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        elif init_type == "random":
            points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
            rgbs = torch.rand((init_num_pts, 3))
        else:
            raise ValueError(f"Unknown init_type: {init_type}. Use 'sfm', 'random', or 'ply'.")

        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
        opacities = torch.logit(torch.full((points.shape[0],), init_opacity))
        quats = torch.rand((points.shape[0], 4))
        sh0s = rgb_to_sh(rgbs)

        points = points[world_rank::world_size]
        scales = scales[world_rank::world_size]
        quats = quats[world_rank::world_size]
        opacities = opacities[world_rank::world_size]
        sh0s = sh0s[world_rank::world_size]

        N = points.shape[0]
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = sh0s
        params = [
            ("means", torch.nn.Parameter(points), means_lr * scene_scale),
            ("scales", torch.nn.Parameter(scales), scales_lr),
            ("quats", torch.nn.Parameter(quats), quats_lr),
            ("opacities", torch.nn.Parameter(opacities), opacities_lr),
            ("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr),
            ("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr),
        ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    BS = batch_size * world_size
    optimizer_class = SelectiveAdam if visible_adam else torch.optim.Adam
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """4DGS training and evaluation engine."""

    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: Config):
        set_random_seed(42 + local_rank)
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        self.stats_dir = f"{cfg.result_dir}/stats"
        self.render_dir = f"{cfg.result_dir}/renders"
        for d in [self.ckpt_dir, self.stats_dir, self.render_dir]:
            os.makedirs(d, exist_ok=True)
        if cfg.save_ply:
            self.ply_dir = f"{cfg.result_dir}/ply"
            os.makedirs(self.ply_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # ---- Dataset ----
        if cfg.dataset_mode == "rig":
            # Fixed multi-camera rig mode (e.g. elly dataset).
            # Load camera poses from reference frame (frame 0), then build
            # DynamicRigDataset covering all cameras × selected frames.
            print("[4DGS] Dataset mode: fixed-camera rig")
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=0,   # no filtering — we need all cameras as reference
                undistort_input=cfg.undistort_colmap_input,
                frame_num=cfg.frame_start,    # load reference frame poses (matches first frame)
            )
            # Build selected frame list: frame_start, frame_start+stride, ...
            selected_frames = list(range(
                cfg.frame_start,
                cfg.frame_start + cfg.num_frames * cfg.frame_stride,
                cfg.frame_stride,
            ))
            print(f"[4DGS] Selected frames: {selected_frames[0]}..{selected_frames[-1]} "
                  f"({len(selected_frames)} frames, stride={cfg.frame_stride})")
            image_dir = os.path.join(cfg.data_dir, "images")
            # Full dataset (used for fine phase and eval)
            self.trainset = DynamicRigDataset(
                ref_parser=self.parser,
                image_dir=image_dir,
                selected_frames=selected_frames,
                split="train",
                test_every=cfg.rig_test_every,
                factor=cfg.data_factor,
                patch_size=cfg.patch_size,
                mask_dir=cfg.mask_dir,
                invert_masks=cfg.invert_masks,
            )
            self.valset = DynamicRigDataset(
                ref_parser=self.parser,
                image_dir=image_dir,
                selected_frames=selected_frames,
                split="val",
                test_every=cfg.rig_test_every,
                factor=cfg.data_factor,
                mask_dir=cfg.mask_dir,
                invert_masks=cfg.invert_masks,
                val_num_cameras=cfg.val_num_cameras,
                val_num_frames=cfg.val_num_frames,
            )
            # Coarse-phase dataset: only the reference frame (no ghosting).
            # Matches reference 4DGS zerostamp_init: during coarse, only train
            # on a single canonical frame so Gaussians don't average across time.
            if cfg.coarse_single_frame:
                self.coarse_trainset = DynamicRigDataset(
                    ref_parser=self.parser,
                    image_dir=image_dir,
                    selected_frames=[selected_frames[0]],  # just the first frame
                    split="train",
                    test_every=0,  # use ALL cameras for coarse
                    factor=cfg.data_factor,
                    patch_size=cfg.patch_size,
                    mask_dir=cfg.mask_dir,
                    invert_masks=cfg.invert_masks,
                )
            else:
                self.coarse_trainset = self.trainset
        else:
            # Standard mode: load all frames from COLMAP images.bin (frame_num=None)
            print("[4DGS] Dataset mode: standard (all frames from COLMAP)")
            self.parser = Parser(
                data_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=cfg.normalize_world_space,
                test_every=cfg.test_every,
                undistort_input=cfg.undistort_colmap_input,
                use_masks=cfg.use_masks,
                load_images_in_memory=cfg.load_images_in_memory,
            )
            # Compute selected frame indices for filtering
            if cfg.frame_stride > 1 or cfg.frame_start > 0:
                selected_frames = list(range(
                    cfg.frame_start,
                    cfg.frame_start + cfg.num_frames * cfg.frame_stride,
                    cfg.frame_stride,
                ))
            else:
                selected_frames = None  # use all
            self.trainset = DynamicDataset(
                self.parser,
                split="train",
                num_frames=cfg.num_frames,
                selected_frames=selected_frames,
                patch_size=cfg.patch_size,
                load_depths=False,
            )
            self.valset = DynamicDataset(
                self.parser,
                split="val",
                num_frames=cfg.num_frames,
                selected_frames=selected_frames,
            )

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"[4DGS] Scene scale: {self.scene_scale:.4f}")
        print(f"[4DGS] Train images: {len(self.trainset)}, Val images: {len(self.valset)}")

        # ---- Scene AABB for HexPlane normalization ----
        # Estimate from COLMAP points
        if hasattr(self.parser, "points") and len(self.parser.points) > 0:
            pts = torch.from_numpy(self.parser.points).float()
            aabb_min = pts.min(0).values
            aabb_max = pts.max(0).values
            margin = (aabb_max - aabb_min) * cfg.aabb_margin
            aabb = torch.stack([aabb_min - margin, aabb_max + margin])
        else:
            # Fallback: use camera center extent
            c2w = torch.from_numpy(self.parser.camtoworlds).float()
            centers = c2w[:, :3, 3]
            margin = centers.std(0).mean().clamp(min=1.0) * 3.0
            aabb = torch.stack([
                centers.min(0).values - margin,
                centers.max(0).values + margin,
            ])
        print(f"[4DGS] Scene AABB: {aabb[0].tolist()} → {aabb[1].tolist()}")
        aabb_half = (aabb[1] - aabb[0]) / 2.0
        print(f"[4DGS] AABB half-extent: {aabb_half.tolist()} | scene_scale={self.parser.scene_scale:.3f}")
        self.aabb = aabb.to(self.device)

        # If dynamic_ply_path is given, use it as the PLY init for trainable Gaussians
        if cfg.dynamic_ply_path is not None:
            cfg.init_type = "ply"
            cfg.ply_path = cfg.dynamic_ply_path
            print(f"[4DGS] Dynamic init: loading Gaussians from {cfg.dynamic_ply_path}")

        # ---- Dynamic (trainable) Gaussians ----
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
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
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
            ply_path=cfg.ply_path,
        )
        self._n_dynamic = len(self.splats["means"])
        print(f"[4DGS] Dynamic Gaussians: {self._n_dynamic}")

        # ---- Static (frozen background) Gaussians ----
        # Loaded once, pre-activated, stored as plain tensors outside self.splats.
        # Never updated, no optimizer, no densification.
        self.static_activated = None
        if cfg.static_ply_path is not None:
            s_means, s_scales, s_quats, s_opacs, s_sh0, s_shN = import_splats(
                cfg.static_ply_path, self.device
            )
            # Static Gaussians always render at SH=0 (DC only, view-independent).
            # Zero out all higher-order SH bands so rasterization at any sh_degree
            # evaluates to DC only for static — no view-dependent color variation.
            # Shape must match dynamic colors [N, (sh_degree+1)^2, 3] for concatenation.
            target_shN = (cfg.sh_degree + 1) ** 2 - 1
            self.static_activated = {
                "means":     s_means,                                                         # [N_s, 3]
                "scales":    torch.exp(s_scales),                                             # [N_s, 3] — activated
                "quats":     s_quats,                                                         # [N_s, 4]
                "opacities": torch.sigmoid(s_opacs),                                         # [N_s]   — activated
                "colors":    torch.cat([s_sh0,                                               # [N_s, 1, 3] DC term
                                        torch.zeros(s_sh0.shape[0], target_shN, 3,
                                                    device=self.device)], dim=1),            # [N_s, K, 3] shN=0
            }
            print(f"[4DGS] Static Gaussians loaded: {s_means.shape[0]} (frozen, SH=0/DC-only)")

        # ---- Dynamic AABB (kept for legacy/RobustNeRF quantile region) ----
        _d_means = self.splats["means"].detach()
        _dyn_aabb_min = _d_means.min(0).values
        _dyn_aabb_max = _d_means.max(0).values
        _dyn_margin = (_dyn_aabb_max - _dyn_aabb_min) * 0.05
        self.dynamic_aabb_min = (_dyn_aabb_min - _dyn_margin).to(self.device)
        self.dynamic_aabb_max = (_dyn_aabb_max + _dyn_margin).to(self.device)

        # ---- Static alpha cache (lazy per-camera, never invalidated) ----
        # Stores [H, W] bool: True = non-static pixel (safe for loss).
        self._static_alpha_cache: dict = {}

        # ---- RobustNeRF learnable inlier mask ----
        # sigmoid(-3.48) * 0.10 ≈ 0.03  →  initial τ ≈ 3% outliers
        self.mask_tau_raw = torch.nn.Parameter(
            torch.tensor(-3.48, device=self.device)
        )
        self.mask_optimizer = torch.optim.Adam([self.mask_tau_raw], lr=cfg.mask_lr)

        # ---- Densification Strategy ----
        cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = cfg.strategy.initialize_state(scene_scale=self.scene_scale)

        # ---- Deformation Field ----
        self.deform_field = None
        self.deform_optimizers = {}
        self.deform_schedulers = []
        if cfg.use_deformation:
            # Effective SH degree for the deformation head:
            # deform_sh_degree=-1 means use full sh_degree (default behaviour).
            eff_deform_sh_degree = cfg.deform_sh_degree if cfg.deform_sh_degree >= 0 else cfg.sh_degree
            self.deform_field = DeformationField(
                grid_resolution=cfg.deform_grid_resolution,
                time_resolution=cfg.deform_time_resolution,
                feature_dim=cfg.deform_feature_dim,
                multires=cfg.deform_multires,
                net_width=cfg.deform_net_width,
                defor_depth=cfg.deform_net_depth,
                aabb=aabb,
                enable_opacity_deform=cfg.enable_opacity_deform,
                enable_sh_deform=cfg.enable_sh_deform,
                sh_degree=eff_deform_sh_degree,
                time_pe_bands=cfg.deform_time_pe_bands,
                act_xyz=cfg.deform_act_xyz,
                act_rot=cfg.deform_act_rot,
                act_scale=cfg.deform_act_scale,
                act_sh=cfg.deform_act_sh,
            ).to(self.device)
            if cfg.enable_sh_deform:
                print(f"[4DGS] SH deform enabled: degree={eff_deform_sh_degree} "
                      f"({(eff_deform_sh_degree + 1) ** 2 * 3} output dims)")
            BS = cfg.batch_size * world_size
            self.deform_optimizers = {
                "deform_mlp": torch.optim.Adam(
                    [p for n, p in self.deform_field.named_parameters()
                     if "hexplane" not in n],
                    lr=cfg.deform_lr * math.sqrt(BS),
                    eps=1e-15 / math.sqrt(BS),
                    betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                ),
                "deform_grid": torch.optim.Adam(
                    self.deform_field.hexplane.parameters(),
                    lr=cfg.grid_lr * math.sqrt(BS),
                    eps=1e-15 / math.sqrt(BS),
                    betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                ),
            }
            print(
                f"[4DGS] DeformationField: grid={cfg.deform_grid_resolution}^3 x {cfg.deform_time_resolution}t, "
                f"feat={cfg.deform_feature_dim}, width={cfg.deform_net_width}, depth={cfg.deform_net_depth}"
            )

            # Optionally load deformation weights from a previous checkpoint
            # (keeps splats/points fresh, only warms up the deformation network)
            if cfg.resume_deform_ckpt is not None:
                ckpt_data = torch.load(cfg.resume_deform_ckpt, map_location=self.device)
                if "deform_field" in ckpt_data:
                    self.deform_field.load_state_dict(ckpt_data["deform_field"])
                    print(f"[4DGS] Loaded deformation weights from {cfg.resume_deform_ckpt}")
                else:
                    print(f"[4DGS] WARNING: no deform_field in {cfg.resume_deform_ckpt}, skipping")

        # ---- Metrics ----
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type=cfg.lpips_net, normalize=(cfg.lpips_net == "alex")
        ).to(self.device)

        # ---- Viewer ----
        if not cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    # -----------------------------------------------------------------------
    # Core rasterization (applies deformation when active)
    # -----------------------------------------------------------------------

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        timestamp: Optional[float] = None,
        sh_degree: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize Gaussians. If deformation is active, applies deformation at `timestamp`.

        Args:
            camtoworlds: [C, 4, 4] camera-to-world matrices.
            Ks: [C, 3, 3] camera intrinsics.
            width, height: image dimensions.
            timestamp: normalized time in [-0.5, 0.5]. None → no deformation.
            sh_degree: SH degree to evaluate. None → cfg.sh_degree.
        """
        cfg = self.cfg
        if sh_degree is None:
            sh_degree = cfg.sh_degree

        deformation_active = (
            self.deform_field is not None
            and timestamp is not None
        )

        if deformation_active:
            # Apply deformation: pass means WITH gradients (no detach) so the
            # deformation field can learn from position gradients, matching the
            # original 4DGS implementation (hustvl/4DGaussians).
            deltas = self.deform_field(
                self.splats["means"], timestamp
            )
            self._last_deform_mag = deltas.delta_xyz.detach().norm(dim=-1).mean().item()  # for TB (normalized units)
            means, quats, scales, opacities, colors = apply_deformation(
                self.splats, deltas, aabb=self.aabb
            )
        else:
            means = self.splats["means"]
            quats = self.splats["quats"]
            scales = torch.exp(self.splats["scales"])
            opacities = torch.sigmoid(self.splats["opacities"])
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)

        # Track N_dynamic (may change after densification steps)
        self._n_dynamic = means.shape[0]

        # Concatenate static (frozen background) Gaussians AFTER dynamic.
        # Dynamic first => info["means2d"][..., :N_dynamic, :] is always dynamic.
        # Static tensors are detached so no gradient flows into them.
        if self.static_activated is not None:
            sa = self.static_activated
            means     = torch.cat([means,     sa["means"].detach()],     dim=0)
            quats     = torch.cat([quats,     sa["quats"].detach()],     dim=0)
            scales    = torch.cat([scales,    sa["scales"].detach()],    dim=0)
            opacities = torch.cat([opacities, sa["opacities"].detach()], dim=0)
            colors    = torch.cat([colors,    sa["colors"].detach()],    dim=0)

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            packed=cfg.packed,
            absgrad=(
                cfg.strategy.absgrad
                if isinstance(cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=cfg.sparse_grad,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
            distributed=self.world_size > 1,
            camera_model=cfg.camera_model,
            with_ut=cfg.with_ut,
            with_eval3d=cfg.with_eval3d,
            **kwargs,
        )
        return render_colors, render_alphas, info

    # -----------------------------------------------------------------------
    # RobustNeRF helper
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def _get_nonstatic_mask(
        self,
        camera_idx: int,
        camtoworld: Tensor,  # [4, 4]
        K: Tensor,           # [3, 3]
        height: int,
        width: int,
        alpha_threshold: float = 0.5,
    ) -> Tensor:
        """Return [H, W] bool: True = non-static pixel (include in loss).

        Renders static Gaussians once per camera (lazy cache) and excludes
        pixels where static alpha > threshold. Dynamic Gaussians can densify
        and grow anywhere — not limited to the AABB projection.
        """
        cache_key = (camera_idx, height, width)
        if cache_key not in self._static_alpha_cache:
            if self.static_activated is None:
                self._static_alpha_cache[cache_key] = torch.ones(
                    height, width, dtype=torch.bool, device=self.device
                )
            else:
                sa = self.static_activated
                _, alphas, _ = rasterization(
                    means=sa["means"],
                    quats=sa["quats"],
                    scales=sa["scales"],
                    opacities=sa["opacities"],
                    colors=sa["colors"][:, :1, :],  # DC only
                    viewmats=torch.linalg.inv(camtoworld)[None],
                    Ks=K[None],
                    width=width,
                    height=height,
                    sh_degree=0,
                    packed=True,
                )
                static_alpha = alphas[0, :, :, 0]  # [H, W]
                self._static_alpha_cache[cache_key] = static_alpha < alpha_threshold
        return self._static_alpha_cache[cache_key]

    def _dynamic_pixel_mask(
        self,
        viewmat: Tensor,   # [4, 4] world-to-camera
        K: Tensor,         # [3, 3] intrinsics
        height: int,
        width: int,
    ) -> Optional[Tensor]:
        """Return a [H, W] bool mask of pixels covered by the dynamic 3D AABB.

        Projects the 8 AABB corners to image space and returns their 2D bounding
        box.  Returns None if no AABB is set or all corners are behind the camera.
        """
        if self.dynamic_aabb_min is None:
            return None

        mn, mx = self.dynamic_aabb_min, self.dynamic_aabb_max
        corners = torch.stack([
            torch.stack([mn[0], mn[1], mn[2]]),
            torch.stack([mx[0], mn[1], mn[2]]),
            torch.stack([mn[0], mx[1], mn[2]]),
            torch.stack([mx[0], mx[1], mn[2]]),
            torch.stack([mn[0], mn[1], mx[2]]),
            torch.stack([mx[0], mn[1], mx[2]]),
            torch.stack([mn[0], mx[1], mx[2]]),
            torch.stack([mx[0], mx[1], mx[2]]),
        ])  # [8, 3]

        # World → camera space
        R, t = viewmat[:3, :3], viewmat[:3, 3]
        p_cam = (R @ corners.T).T + t  # [8, 3]

        valid = p_cam[:, 2] > 0.01
        if valid.sum() == 0:
            return None
        p_cam = p_cam[valid]

        z = p_cam[:, 2]
        u = (K[0, 0] * p_cam[:, 0] / z + K[0, 2]).long()
        v = (K[1, 1] * p_cam[:, 1] / z + K[1, 2]).long()

        u_min = u.min().clamp(0, width  - 1)
        u_max = u.max().clamp(0, width  - 1)
        v_min = v.min().clamp(0, height - 1)
        v_max = v.max().clamp(0, height - 1)

        mask = torch.zeros(height, width, dtype=torch.bool, device=self.device)
        mask[v_min:v_max + 1, u_min:u_max + 1] = True
        return mask  # [H, W]

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps

        # LR schedulers — only Gaussian means here.
        # Deformation schedulers are created at fine phase start (matching reference
        # 4DGS which re-creates the optimizer for the fine phase).
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        deform_schedulers = []  # populated at coarse→fine transition
        mask_start_step = cfg.mask_start_step if cfg.mask_start_step >= 0 else max_steps // 2

        # Start with coarse dataset (single frame if coarse_single_frame=True)
        coarse_ds = getattr(self, 'coarse_trainset', self.trainset)
        trainloader = torch.utils.data.DataLoader(
            coarse_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)   # [B, 4, 4]
            Ks = data["K"].to(device)                     # [B, 3, 3]
            pixels = data["image"].to(device)             # [B, H, W, 3]
            timestamps = data["timestamp"].to(device)     # [B]
            image_ids = data["image_id"].to(device)       # [B]
            # Masks: DynamicRigDataset uses "mask", base Dataset uses "segmentation_mask"
            if "mask" in data:
                masks = data["mask"].to(device)  # [B, H, W]
            elif "segmentation_mask" in data:
                masks = data["segmentation_mask"].to(device)  # [B, H, W]
            else:
                masks = None
            height, width = pixels.shape[1:3]

            # SH schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Deformation active after coarse warm-up
            deform_active = (
                cfg.use_deformation
                and self.deform_field is not None
                and step >= cfg.coarse_iters
            )

            # One-time: create deformation optimizers + schedulers at fine phase start.
            # Matches reference 4DGS which runs coarse and fine as separate training
            # loops with fresh optimizer state. Fresh Adam momentum + clean LR schedule.
            if deform_active and step == cfg.coarse_iters and self.deform_field is not None:
                BS = cfg.batch_size * self.world_size
                self.deform_optimizers = {
                    "deform_mlp": torch.optim.Adam(
                        [p for n, p in self.deform_field.named_parameters()
                         if "hexplane" not in n],
                        lr=cfg.deform_lr * math.sqrt(BS),
                        eps=1e-15 / math.sqrt(BS),
                        betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                    ),
                    "deform_grid": torch.optim.Adam(
                        self.deform_field.hexplane.parameters(),
                        lr=cfg.grid_lr * math.sqrt(BS),
                        eps=1e-15 / math.sqrt(BS),
                        betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
                    ),
                }
                fine_max = max(max_steps - cfg.coarse_iters, 1)
                def _make_fine_lr_lambda(base_lr, final_lr, delay_mult, warmup_steps):
                    """LR schedule for fine phase: cosine warmup then log-linear decay."""
                    import math as _math
                    def lr_lambda(fine_step):
                        # Warmup
                        if fine_step < warmup_steps and warmup_steps > 0:
                            t = fine_step / warmup_steps
                            wf = delay_mult + (1.0 - delay_mult) * (
                                0.5 * (1.0 + _math.cos(_math.pi * (1.0 - t)))
                            )
                        else:
                            wf = 1.0
                        # Decay over fine phase duration
                        td = min(fine_step / fine_max, 1.0)
                        df = (final_lr / base_lr) ** td if fine_step > 0 else 1.0
                        return wf * df
                    return lr_lambda
                deform_schedulers = [
                    torch.optim.lr_scheduler.LambdaLR(
                        self.deform_optimizers["deform_mlp"],
                        lr_lambda=_make_fine_lr_lambda(
                            cfg.deform_lr, cfg.deform_lr * 0.1,
                            cfg.deform_lr_delay_mult, cfg.deform_lr_warmup_steps,
                        ),
                    ),
                    torch.optim.lr_scheduler.LambdaLR(
                        self.deform_optimizers["deform_grid"],
                        lr_lambda=_make_fine_lr_lambda(
                            cfg.grid_lr, cfg.grid_lr * 0.1,
                            cfg.deform_lr_delay_mult, cfg.deform_lr_warmup_steps,
                        ),
                    ),
                ]
                # Switch to fine-phase dataset
                if cfg.progressive_time_warmup > 0 and hasattr(self.trainset, 'indices_within_time_radius'):
                    _prog_radius = cfg.progressive_time_initial  # window size (used in both modes)
                    if cfg.progressive_time_forward:
                        # Forward-only: start from t=-0.5 (frame 0), window = initial size
                        _prog_indices = self.trainset.indices_within_time_window(-0.5, -0.5 + _prog_radius)
                        print(f"[4DGS] Fine phase start: forward progressive sampling, "
                              f"window=[-0.5, {-0.5 + _prog_radius:.2f}], {len(_prog_indices)} images")
                    else:
                        # Symmetric: start with frames near center t=0
                        _prog_indices = self.trainset.indices_within_time_radius(_prog_radius)
                        print(f"[4DGS] Fine phase start: progressive sampling (symmetric), "
                              f"time_radius={_prog_radius:.2f}, {len(_prog_indices)} images")
                    _prog_subset = torch.utils.data.Subset(self.trainset, _prog_indices)
                    trainloader = torch.utils.data.DataLoader(
                        _prog_subset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        num_workers=4,
                        persistent_workers=False,
                        pin_memory=True,
                    )
                else:
                    _prog_radius = 0.5  # full range
                    trainloader = torch.utils.data.DataLoader(
                        self.trainset,
                        batch_size=cfg.batch_size,
                        shuffle=True,
                        num_workers=4,
                        persistent_workers=True,
                        pin_memory=True,
                    )
                    print(f"[4DGS] Fine phase start: full dataset")
                trainloader_iter = iter(trainloader)
                print(f"[4DGS] Fresh deformation optimizers + schedulers")

            # Use first timestamp in batch (batch_size=1 for 4DGS training)
            t = timestamps[0].item() if deform_active else None

            # Debug: log camera/frame/timestamp mapping for first few fine steps
            if deform_active and step < cfg.coarse_iters + 5:
                cam_idx_dbg = data["cam_idx"][0].item()
                frame_idx_dbg = data["frame_idx"][0].item()
                print(f"  [DBG] step={step} cam={cam_idx_dbg} frame={frame_idx_dbg} t={t:.4f}")

            # Forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                timestamp=t,
                sh_degree=sh_degree_to_use,
            )
            colors_render = renders  # [B, H, W, 3]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors_render = colors_render + bkgd * (1.0 - alphas)

            # Pre-backward hook: pass FULL info so retain_grad() is called on the
            # full means2d tensor. retain_grad on a slice/view does NOT work —
            # grad stays None after backward. We slice for post_backward instead.
            n_dyn = self._n_dynamic
            cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # ---- Losses ----
            # Supervise all pixels. Static Gaussians are frozen so background
            # pixels have near-zero error and don't hurt dynamic learning.
            if masks is not None:
                active_mask = masks.unsqueeze(-1).float()  # [B, H, W, 1]
            else:
                active_mask = torch.ones(1, height, width, 1, device=device)

            # SSIM: compute only on non-static pixels
            colors_m = colors_render * active_mask
            pixels_m = pixels * active_mask
            ssimloss = 1.0 - fused_ssim(
                colors_m.permute(0, 3, 1, 2),
                pixels_m.permute(0, 3, 1, 2),
                padding="valid",
            )

            # L1: RobustNeRF inlier mask after mask_start_step
            if step > mask_start_step:
                # RobustNeRF-style learnable outlier rejection (no dilation variant).
                # τ = sigmoid(τ_raw) * max_tau  ∈ (0, mask_max_tau)
                # ρ = quantile(R_nonstatic, 1−τ) — differentiable w.r.t. τ via torch.quantile
                # M = hard mask forward, soft sigmoid backward (straight-through)
                diff      = (colors_render - pixels).abs()         # [1, H, W, 3]
                pixel_err = diff.mean(dim=-1, keepdim=True)        # [1, H, W, 1]

                tau = torch.sigmoid(self.mask_tau_raw) * cfg.mask_max_tau
                q   = (1.0 - tau).clamp(0.0, 1.0)

                rho = torch.quantile(pixel_err.flatten().detach(), q)

                M_soft = torch.sigmoid(cfg.mask_beta * (rho - pixel_err))  # [1, H, W, 1]
                M_hard = (pixel_err <= rho).float()
                M_ste  = M_hard + (M_soft - M_soft.detach())     # straight-through

                robustnerf_mask = M_ste * active_mask
                coverage = M_soft.mean()

                coverage_penalty = (
                    F.relu(cfg.mask_coverage_target - coverage) ** 2
                    * cfg.mask_coverage_lambda
                )
                l1loss = (diff * robustnerf_mask).sum() / (robustnerf_mask.sum() * 3).clamp(min=1)

                _log_tau              = tau.detach()
                _log_rho              = rho.detach()
                _log_coverage         = coverage.detach()
                _log_coverage_penalty = coverage_penalty.detach()
            else:
                # Standard L1 for first half of training
                coverage_penalty = torch.tensor(0.0, device=device)
                _log_tau = _log_rho = _log_coverage = _log_coverage_penalty = None

                # L1 on non-static pixels only
                l1loss = (torch.abs(colors_render - pixels) * active_mask).sum() / (active_mask.sum() * 3).clamp(min=1)

            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda + coverage_penalty

            # Alpha penalty in masked (background) regions: discourage Gaussians there
            if masks is not None:
                masked_region = 1.0 - masks  # [B, H, W], 1 where background
                alpha_in_bg = (alphas.squeeze(-1) * masked_region).sum() / masked_region.sum().clamp(min=1.0)
                loss = loss + 0.1 * alpha_in_bg

            # Gaussian regularizations
            if cfg.opacity_reg > 0.0:
                loss = loss + cfg.opacity_reg * torch.abs(
                    torch.sigmoid(self.splats["opacities"])
                ).mean()
            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * torch.abs(
                    torch.exp(self.splats["scales"])
                ).mean()

            # Deformation field regularizations (only after coarse phase)
            deform_reg = torch.tensor(0.0, device=device)
            if deform_active and self.deform_field is not None:
                if cfg.plane_tv_weight > 0.0:
                    deform_reg = deform_reg + cfg.plane_tv_weight * plane_tv_loss(
                        self.deform_field.hexplane
                    )
                if cfg.time_smooth_weight > 0.0:
                    # Optionally anneal temporal smoothness from start to final weight
                    if cfg.time_smooth_weight_final >= 0.0:
                        progress = (step - cfg.coarse_iters) / max(max_steps - cfg.coarse_iters, 1)
                        progress = min(max(progress, 0.0), 1.0)
                        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                        tsw = cfg.time_smooth_weight_final + (cfg.time_smooth_weight - cfg.time_smooth_weight_final) * cos_decay
                    else:
                        tsw = cfg.time_smooth_weight
                    time_loss_fn = time_smoothness_loss_1st if cfg.time_smooth_order == 1 else time_smoothness_loss
                    deform_reg = deform_reg + tsw * time_loss_fn(
                        self.deform_field.hexplane
                    )
                if cfg.l1_time_planes_weight > 0.0:
                    deform_reg = deform_reg + cfg.l1_time_planes_weight * l1_time_planes_loss(
                        self.deform_field.hexplane
                    )
                # Standard constraint: at t=0, deformation should be identity.
                # Reference 4DGS: (deformed_xyz - canonical_xyz)^2 + (deformed_rot - canonical_rot)^2 + ...
                # Annealed from weight_constraint_init → weight_constraint_after.
                if cfg.weight_constraint_init > 0.0:
                    fine_step = step - cfg.coarse_iters
                    wc_progress = min(fine_step / max(cfg.weight_constraint_decay_iters, 1), 1.0)
                    wc = cfg.weight_constraint_init + (cfg.weight_constraint_after - cfg.weight_constraint_init) * wc_progress
                    if wc > 0.0:
                        # Evaluate deformation at t=-0.5 (frame 0 = PLY init time).
                        # Canonical Gaussians are initialized from dynamic.ply which
                        # represents frame 0. With centered-time [-0.5, 0.5], t=0.0
                        # is the MID-SEQUENCE frame (frame 75), NOT frame 0 — so
                        # constraining at t=0.0 created a conflict with the PLY init.
                        canonical_means = self.splats["means"]
                        n_sub = min(len(canonical_means), 100000)
                        idx = torch.randperm(len(canonical_means), device=device)[:n_sub]
                        deform_t0 = self.deform_field(canonical_means[idx], -0.5)
                        constraint_loss = (
                            deform_t0.delta_xyz.pow(2).mean()
                            + deform_t0.delta_rot.pow(2).mean()
                            + deform_t0.delta_scale.pow(2).mean()
                        )
                        deform_reg = deform_reg + wc * constraint_loss
                loss = loss + deform_reg

            loss.backward()

            # ---- Progress bar ----
            desc = (
                f"loss={loss.item():.3f} "
                f"l1={l1loss.item():.3f} "
                f"sh={sh_degree_to_use} "
                f"deform={'on' if deform_active else 'off'} "
                f"N={len(self.splats['means'])}"
            )
            if deform_active:
                desc += f" reg={deform_reg.item():.5f}"
            pbar.set_description(desc)

            # ---- Tensorboard ----
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/active_mask_frac", active_mask.mean().item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem_GB", mem, step)
                self.writer.add_scalar("train/timestamp", t if t is not None else 0.0, step)
                if deform_active:
                    self.writer.add_scalar("train/deform_reg", deform_reg.item(), step)
                    if hasattr(self, "_last_deform_mag"):
                        self.writer.add_scalar("train/deform_xyz_mag", self._last_deform_mag, step)

                # RobustNeRF mask scalars (only logged after first half of training)
                if _log_tau is not None:
                    self.writer.add_scalar("train/mask_tau",              _log_tau.item(),              step)
                    self.writer.add_scalar("train/mask_rho",              _log_rho.item(),              step)
                    self.writer.add_scalar("train/mask_coverage",         _log_coverage.item(),         step)
                    self.writer.add_scalar("train/mask_coverage_penalty", _log_coverage_penalty.item(), step)

                # GT + rendered concat image every tb_image_every steps
                if cfg.tb_image_every > 0 and step % cfg.tb_image_every == 0:
                    with torch.no_grad():
                        gt = pixels[0].clamp(0, 1)            # [H, W, 3]
                        pred = colors_render[0].clamp(0, 1)   # [H, W, 3]
                        # Show active mask as a gray image so we can verify coverage
                        mask_vis = active_mask[0].expand(-1, -1, 3)  # [H, W, 3]
                        canvas = torch.cat([gt, pred, mask_vis], dim=1)  # [H, 3W, 3]
                        self.writer.add_image(
                            "train/gt_vs_render",
                            canvas.permute(2, 0, 1),           # [3, H, 2W]
                            step,
                        )

                self.writer.flush()

            # ---- Checkpoint ----
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                    "step": step,
                }
                print(f"[4DGS] Step {step}: {stats}")
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{world_rank}.json", "w"
                ) as f:
                    json.dump(stats, f)

                ckpt_data = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                    "config": vars(cfg),
                    "aabb": self.aabb.cpu(),
                    "mask_tau_raw": self.mask_tau_raw.data.cpu(),
                }
                if self.deform_field is not None:
                    ckpt_data["deform_field"] = self.deform_field.state_dict()
                torch.save(
                    ckpt_data,
                    f"{self.ckpt_dir}/ckpt_{step}_rank{world_rank}.pt",
                )

            # ---- PLY export ----
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                export_splats(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"],
                    sh0=self.splats["sh0"],
                    shN=self.splats["shN"],
                    format="ply",
                    save_to=f"{self.ply_dir}/canonical_{step}.ply",
                )

            # ---- Optimizer step ----
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients require packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )

            # Gradient clipping BEFORE optimizer steps (critical for stability).
            # During fine phase, deformation gradients flow into canonical means,
            # so we clip Gaussian grads too.
            if deform_active:
                torch.nn.utils.clip_grad_norm_(
                    self.deform_field.parameters(), max_norm=1.0
                )
                # Also clip canonical Gaussian means to prevent position explosion
                if self.splats["means"].grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [self.splats["means"]], max_norm=1.0
                    )

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if deform_active:
                for optimizer in self.deform_optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # Mask τ optimizer (always step, grad is zero before mask activates)
            self.mask_optimizer.step()
            self.mask_optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()
            for scheduler in deform_schedulers:
                scheduler.step()

            # ---- Progressive frame sampling: expand time window ----
            if (
                cfg.progressive_time_warmup > 0
                and deform_active
                and hasattr(self.trainset, 'indices_within_time_radius')
                and step > cfg.coarse_iters
                and (step - cfg.coarse_iters) % 1000 == 0
            ):
                fine_step = step - cfg.coarse_iters
                progress = min(fine_step / cfg.progressive_time_warmup, 1.0)
                # window_size: grows from initial → 1.0 (full range)
                new_radius = cfg.progressive_time_initial + (1.0 - cfg.progressive_time_initial) * progress
                if new_radius > _prog_radius + 0.01:  # only rebuild if meaningful change
                    _prog_radius = new_radius
                    if _prog_radius >= 0.99:
                        # Full range — use entire trainset
                        trainloader = torch.utils.data.DataLoader(
                            self.trainset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True,
                        )
                        print(f"[4DGS] Progressive: full range, {len(self.trainset)} images")
                    elif cfg.progressive_time_forward:
                        # Forward-only: window = [-0.5, -0.5 + window_size]
                        t_end = min(-0.5 + _prog_radius, 0.5)
                        _prog_indices = self.trainset.indices_within_time_window(-0.5, t_end)
                        _prog_subset = torch.utils.data.Subset(self.trainset, _prog_indices)
                        trainloader = torch.utils.data.DataLoader(
                            _prog_subset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=4,
                            persistent_workers=False,
                            pin_memory=True,
                        )
                        print(f"[4DGS] Progressive (forward): window=[-0.5, {t_end:.2f}], {len(_prog_indices)} images")
                    else:
                        # Symmetric: |t| <= radius (radius = window_size / 2)
                        sym_radius = _prog_radius / 2.0
                        _prog_indices = self.trainset.indices_within_time_radius(sym_radius)
                        _prog_subset = torch.utils.data.Subset(self.trainset, _prog_indices)
                        trainloader = torch.utils.data.DataLoader(
                            _prog_subset,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            num_workers=4,
                            persistent_workers=False,
                            pin_memory=True,
                        )
                        print(f"[4DGS] Progressive (symmetric): time_radius={sym_radius:.2f}, {len(_prog_indices)} images")
                    trainloader_iter = iter(trainloader)

            # ---- Densification (dynamic Gaussians only) ----
            # Build post-backward info sliced to dynamic Gaussians.
            # We need a proper leaf tensor with .grad set from the full gradient,
            # because slices/views don't accumulate grad via retain_grad.
            if self.static_activated is not None and "means2d" in info and info["means2d"].shape[-2] > n_dyn:
                full_grad = info["means2d"].grad  # set because retain_grad on full tensor
                means2d_dyn = info["means2d"][..., :n_dyn, :].detach().clone()
                means2d_dyn.requires_grad_(True)
                if full_grad is not None:
                    means2d_dyn.grad = full_grad[..., :n_dyn, :].contiguous().clone()
                radii_dyn = (
                    info["radii"][..., :n_dyn, :]
                    if info["radii"].dim() >= 3
                    else info["radii"][..., :n_dyn]
                )
                info_post = {**info, "means2d": means2d_dyn, "radii": radii_dyn}
            else:
                info_post = info
            cfg.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info_post,
                packed=cfg.packed,
            )
            del info, renders, alphas

            # ---- Gaussian count cap ----
            n_gs = len(self.splats["means"])
            if cfg.max_num_gaussians > 0 and n_gs > cfg.max_num_gaussians:
                # Keep the most opaque Gaussians
                opacities = torch.sigmoid(self.splats["opacities"].detach())
                keep_mask = torch.zeros(n_gs, dtype=torch.bool, device=device)
                _, top_idx = opacities.topk(cfg.max_num_gaussians)
                keep_mask[top_idx] = True
                for k in self.splats.keys():
                    self.splats[k] = torch.nn.Parameter(
                        self.splats[k][keep_mask]
                    )
                for opt_name, opt in self.optimizers.items():
                    for group in opt.param_groups:
                        group["params"] = [self.splats[group["name"]]]
                    opt_state = opt.state
                    if opt_state:
                        old_param = list(opt_state.keys())[0]
                        state = opt_state.pop(old_param)
                        for state_key, state_val in state.items():
                            if isinstance(state_val, torch.Tensor) and state_val.dim() > 0 and state_val.shape[0] == n_gs:
                                state[state_key] = state_val[keep_mask]
                        opt_state[self.splats[group["name"]]] = state
                # Reset strategy state to match new Gaussian count
                for state_key in ["grad2d", "count", "radii"]:
                    if state_key in self.strategy_state and self.strategy_state[state_key] is not None:
                        self.strategy_state[state_key] = self.strategy_state[state_key][keep_mask]
                print(f"[4DGS] Capped Gaussians: {n_gs} → {cfg.max_num_gaussians}")

            # ---- Evaluation ----
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / max(time.time() - tic, 1e-10)
                num_train_rays_per_step = pixels.shape[0] * height * width
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_steps_per_sec * num_train_rays_per_step
                )
                self.viewer.update(step, num_train_rays_per_step)

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def eval(self, step: int):
        """Evaluate on validation set, compute PSNR/SSIM/LPIPS."""
        print(f"[4DGS] Evaluating at step {step}...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset if len(self.valset) > 0 else self.trainset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )

        ellipse_time = 0
        metrics = defaultdict(list)

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device)
            timestamps = data["timestamp"].to(device)
            # Masks: DynamicRigDataset uses "mask", base Dataset uses "segmentation_mask"
            if "mask" in data:
                masks = data["mask"].to(device)  # [B, H, W]
            elif "segmentation_mask" in data:
                masks = data["segmentation_mask"].to(device)  # [B, H, W]
            else:
                masks = None
            height, width = pixels.shape[1:3]

            deform_active = (
                self.deform_field is not None and step >= cfg.coarse_iters
            )
            t = timestamps[0].item() if deform_active else None

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                timestamp=t,
                sh_degree=cfg.sh_degree,
                backgrounds=torch.zeros(1, 3, device=device),
            )
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(renders, 0.0, 1.0)

            # Apply masks to metrics only (not visualization)
            if masks is not None:
                mask_3d = masks.unsqueeze(-1)  # [B, H, W, 1]
                colors_masked = colors * mask_3d
                pixels_masked = pixels * mask_3d
            else:
                colors_masked = colors
                pixels_masked = pixels

            colors_p = colors_masked.permute(0, 3, 1, 2)
            pixels_p = pixels_masked.permute(0, 3, 1, 2)

            metrics["psnr"].append(self.psnr(colors_p, pixels_p))
            metrics["ssim"].append(fused_ssim(colors_p, pixels_p, padding="valid"))
            metrics["lpips"].append(self.lpips(colors_p, pixels_p))

            if self.world_rank == 0 and i < 10:
                cam_idx = data["cam_idx"][0].item()
                frame_idx = data["frame_idx"][0].item()
                print(f"  val{i:03d}: cam={cam_idx} frame={frame_idx} t={t:.4f}" if t is not None
                      else f"  val{i:03d}: cam={cam_idx} frame={frame_idx} t=None")
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/step{step:05d}_val{i:03d}.png", canvas
                )

        if self.world_rank == 0:
            avg_psnr = torch.stack(metrics["psnr"]).mean().item()
            avg_ssim = torch.stack(metrics["ssim"]).mean().item()
            avg_lpips = torch.stack(metrics["lpips"]).mean().item()
            fps = len(valloader) / ellipse_time
            print(
                f"[4DGS] Step {step}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} "
                f"LPIPS={avg_lpips:.4f} FPS={fps:.1f}"
            )
            stats = {
                "step": step,
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "lpips": avg_lpips,
                "fps": fps,
            }
            with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            self.writer.add_scalar("val/psnr", avg_psnr, step)
            self.writer.add_scalar("val/ssim", avg_ssim, step)
            self.writer.add_scalar("val/lpips", avg_lpips, step)

    # -----------------------------------------------------------------------
    # Per-frame PLY export (bake deformation into static PLYs)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def export_per_frame_plys(self):
        """Export one PLY per frame with deformation baked in.

        For each timestamp in [0, num_frames), queries the deformation field,
        applies deformation, converts back to pre-activation space (log-scales,
        logit-opacities), and writes a standard PLY via export_splats().
        """
        cfg = self.cfg
        if self.deform_field is None:
            print("[4DGS] No deformation field — skipping per-frame PLY export.")
            return

        out_dir = f"{cfg.result_dir}/ply_per_frame"
        os.makedirs(out_dir, exist_ok=True)

        num_frames = cfg.num_frames
        print(f"[4DGS] Exporting {num_frames} per-frame PLYs to {out_dir}")

        for frame_rank in range(num_frames):
            # Same timestamp normalization as DynamicRigDataset
            t = frame_rank / max(num_frames - 1, 1) - 0.5

            deltas = self.deform_field(self.splats["means"], t)
            means_d, quats_d, scales_d, opacs_d, colors_d = apply_deformation(
                self.splats, deltas, aabb=self.aabb
            )

            # Convert activated values back to pre-activation for PLY export:
            # export_splats expects raw (pre-activation) scales and opacities
            scales_log = torch.log(scales_d)
            opacs_logit = torch.logit(opacs_d.clamp(1e-6, 1 - 1e-6))

            # Split colors [N, K, 3] back to sh0 [N, 1, 3] + shN [N, K-1, 3]
            sh0_d = colors_d[:, :1, :]
            shN_d = colors_d[:, 1:, :]

            # Compute actual frame index for naming (accounts for frame_start/stride)
            frame_idx = cfg.frame_start + frame_rank * cfg.frame_stride

            export_splats(
                means=means_d,
                scales=scales_log,
                quats=quats_d,
                opacities=opacs_logit,
                sh0=sh0_d,
                shN=shN_d,
                format="ply",
                save_to=f"{out_dir}/frame_{frame_idx:06d}.ply",
            )

        print(f"[4DGS] Exported {num_frames} PLYs to {out_dir}")

    # -----------------------------------------------------------------------
    # Viewer render function
    # -----------------------------------------------------------------------

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

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)

        # Time slider value (if available in render_tab_state, else t=0)
        t = getattr(render_tab_state, "time", 0.0)

        renders, alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            timestamp=t if self.deform_field is not None else None,
            sh_degree=min(
                getattr(render_tab_state, "max_sh_degree", self.cfg.sh_degree),
                self.cfg.sh_degree,
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor(
                [render_tab_state.backgrounds], device=self.device
            ) / 255.0,
            render_mode={"rgb": "RGB", "depth(accumulated)": "D", "depth(expected)": "ED",
                         "alpha": "RGB"}.get(render_tab_state.render_mode, "RGB"),
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            return renders[0, ..., :3].clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = renders[0, ..., :1]
            if render_tab_state.normalize_nearfar:
                near, far = render_tab_state.near_plane, render_tab_state.far_plane
            else:
                near, far = depth.min(), depth.max()
            d = (depth - near) / (far - near + 1e-10)
            d = torch.clip(d, 0, 1)
            if render_tab_state.inverse:
                d = 1 - d
            return apply_float_colormap(d, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            return apply_float_colormap(alphas[0], render_tab_state.colormap).cpu().numpy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(local_rank: int, world_rank: int, world_size: int, args):
    torch.manual_seed(42)
    runner = Runner(local_rank, world_rank, world_size, args)

    if args.ckpt is not None:
        # Eval-only mode: load checkpoint and run evaluation
        for ckpt_path in args.ckpt:
            ckpt = torch.load(ckpt_path, map_location=runner.device)
            runner.splats.load_state_dict(ckpt["splats"])
            if "deform_field" in ckpt and runner.deform_field is not None:
                runner.deform_field.load_state_dict(ckpt["deform_field"])
            step = ckpt.get("step", 0)
            print(f"[4DGS] Loaded checkpoint at step {step}")
        runner.eval(step)
        if args.export_per_frame_ply:
            runner.export_per_frame_plys()
        if not args.disable_viewer:
            print("[4DGS] Viewer running... Ctrl+C to exit.")
            time.sleep(100_000)
    else:
        runner.train()
        if args.export_per_frame_ply:
            runner.export_per_frame_plys()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
