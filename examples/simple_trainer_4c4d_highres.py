"""
4C4D: 4 Camera 4D Gaussian Splatting Trainer

Implements the 4C4D method (CVPR 2026) on top of gsplat:
  - 4D Gaussians with temporal attributes (mu_t, scale_t)
  - Neural Decaying Function: MLP that predicts per-Gaussian opacity decay
  - Visibility-aware separate decay strategy
  - Standard 3DGS densification on canonical Gaussians

Key equations from the paper:
  Position at time t:  sigma(t) = sigma + Sigma_{1:3,4} * Sigma_{4,4}^{-1} * (t - mu_t)
  Temporal factor:     omega(t) = exp(-0.5 * (t - mu_t)^2 / Sigma_{4,4})
  Neural decay:        tau = f_theta(x, y, z, o, r)
  Final opacity:       o(t) = tau * omega(t) * o

Usage:
  python simple_trainer_4c4d.py default \\
      --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted \\
      --result_dir results/face_4c4d \\
      --data_factor 15 \\
      --num_frames 100
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
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn as nn
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

from datasets.colmap import Parser
from datasets.traj import generate_interpolated_path, generate_ellipse_path_z
from utils import knn, rgb_to_sh, set_random_seed

from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


# ---------------------------------------------------------------------------
# 4D Rotation and Covariance (Section 3.1 — SO(4) isoclinic decomposition)
# ---------------------------------------------------------------------------

def build_rotation_4d(l: Tensor, r: Tensor) -> Tensor:
    """Build 4×4 SO(4) rotation from two quaternions (left/right isoclinic).

    Args:
        l: [N, 4] left quaternion (raw, will be normalized)
        r: [N, 4] right quaternion (raw, will be normalized)
    Returns:
        [N, 4, 4] rotation matrix in SO(4)
    """
    q_l = F.normalize(l, dim=-1)
    q_r = F.normalize(r, dim=-1)
    a, b, c, d = q_l.unbind(-1)
    p, q, r, s = q_r.unbind(-1)
    M_l = torch.stack([
        a, -b, -c, -d,
        b,  a, -d,  c,
        c,  d,  a, -b,
        d, -c,  b,  a,
    ]).view(4, 4, -1).permute(2, 0, 1)  # [N, 4, 4]
    M_r = torch.stack([
         p,  q,  r,  s,
        -q,  p, -s,  r,
        -r,  s,  p, -q,
        -s, -r,  q,  p,
    ]).view(4, 4, -1).permute(2, 0, 1)  # [N, 4, 4]
    return (M_l @ M_r).flip(1, 2)  # [N, 4, 4]


def build_scaling_rotation_4d(s: Tensor, l: Tensor, r: Tensor) -> Tensor:
    """Build L = R_4d @ S_4d matrix for 4D covariance construction.

    Args:
        s: [N, 4] activated scales (sx, sy, sz, st)
        l: [N, 4] left quaternion
        r: [N, 4] right quaternion
    Returns:
        [N, 4, 4] L matrix such that Sigma = L @ L^T
    """
    R = build_rotation_4d(l, r)  # [N, 4, 4]
    S = torch.zeros(s.shape[0], 4, 4, device=s.device, dtype=s.dtype)
    S[:, 0, 0] = s[:, 0]
    S[:, 1, 1] = s[:, 1]
    S[:, 2, 2] = s[:, 2]
    S[:, 3, 3] = s[:, 3]
    return R @ S  # [N, 4, 4]


def load_ply_splats(filename: str, device: str = "cuda") -> Dict[str, Tensor]:
    """Load Gaussian splat params from PLY. Returns dict with means/scales/quats/opacities/sh0/shN."""
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    v = plydata["vertex"].data
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T

    f_dc = [n for n in v.dtype.names if n.startswith("f_dc_")]
    f_rest = [n for n in v.dtype.names if n.startswith("f_rest_")]
    sc = [n for n in v.dtype.names if n.startswith("scale_")]
    rt = [n for n in v.dtype.names if n.startswith("rot_")]

    dc = np.vstack([v[n] for n in f_dc]).T if f_dc else np.zeros((len(xyz), 3))
    rest = np.vstack([v[n] for n in f_rest]).T if f_rest else None
    scales = np.vstack([v[n] for n in sc]).T  # log space
    quats = np.vstack([v[n] for n in rt]).T
    opacities = v["opacity"]  # logit space

    result = {
        "means": torch.tensor(xyz, dtype=torch.float32, device=device),
        "scales": torch.tensor(scales, dtype=torch.float32, device=device),
        "quats": torch.tensor(quats, dtype=torch.float32, device=device),
        "opacities": torch.tensor(opacities, dtype=torch.float32, device=device),
    }

    # SH coefficients
    sh0 = dc.reshape(-1, 3, 1).transpose(0, 2, 1)  # [N, 1, 3]
    result["sh0"] = torch.tensor(sh0, dtype=torch.float32, device=device)
    if rest is not None and len(f_rest) > 0:
        rest_k = len(f_rest) // 3
        shN = rest.reshape(-1, 3, rest_k).transpose(0, 2, 1)  # [N, K, 3]
        result["shN"] = torch.tensor(shN, dtype=torch.float32, device=device)
    else:
        result["shN"] = torch.zeros((len(xyz), 0, 3), dtype=torch.float32, device=device)

    print(f"  Loaded {len(xyz):,} Gaussians from {filename}")
    return result


# ---------------------------------------------------------------------------
# Neural Decaying Function / Coefficient (Section 3.2)
# Matches the author's Coefficient class from 4C4D.
# ---------------------------------------------------------------------------

class NeuralDecayCoefficient(nn.Module):
    """
    Neural network predicting per-Gaussian opacity decay coefficient.

    Input: (opacity[1], 4D_positions[4], 4D_scales[4]) = 9 dims
    All inputs are normalized before feeding to the network.
    Output: coefficient in (0, 1) via sigmoid.

    Applied as: new_opacity = opacity * (f_min + (f_max - f_min) * coefficient)
    """

    def __init__(self, hidden_dim: int = 32, dropout_rate: float = 0.1):
        super().__init__()
        input_dim = 9  # opacity(1) + xyzt(4) + scales_xyzt(4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        opacity: Tensor,       # [N, 1] in [0, 1]
        positions: Tensor,     # [N, 4] (x, y, z, mu_t)
        scales: Tensor,        # [N, 4] (sx, sy, sz, st) activated
    ) -> Tensor:
        """Returns decay coefficient [N, 1] in (0, 1)."""
        opa = opacity * 2 - 1  # Scale opacity to [-1, 1]

        pos = positions - positions.mean(0, keepdim=True)
        pos = pos / (positions.std(0, keepdim=True, unbiased=False) + 1e-6)

        sca = torch.log(scales + 1e-6)
        sca = sca - sca.mean(0, keepdim=True)
        sca = sca / (sca.std(0, keepdim=True, unbiased=False) + 1e-6)

        x = torch.cat([opa, pos, sca], dim=1)  # [N, 9]
        return self.net(x)  # [N, 1]


# ---------------------------------------------------------------------------
# Dataset: Multi-frame blocks from a fixed camera rig
# ---------------------------------------------------------------------------

class MultiFrameDataset(torch.utils.data.Dataset):
    """
    Each item returns ONE camera at N consecutive frames (a time block).

    Items are laid out as:
        [block_0/cam_0, block_0/cam_1, ..., block_0/cam_44,
         block_1/cam_0, block_1/cam_1, ..., block_1/cam_44, ...]

    With batch_size=num_cameras, one DataLoader batch gives ALL cameras
    for a single time block — maximizing spatial consensus per step.

    Returns:
        images:     [N, H, W, 3]   — N consecutive frames from one camera
        timestamps: [N]             — normalized time for each frame
        camtoworld: [4, 4]          — camera pose (fixed rig)
        K:          [3, 3]          — intrinsics
    """

    def __init__(
        self,
        ref_parser: Parser,
        image_dir: str,
        selected_frames: List[int],
        frames_per_step: int = 5,
        split: str = "train",
        test_every: int = 0,
        factor: int = 1,
        patch_size: Optional[int] = None,
        mask_dir: Optional[str] = None,
        invert_masks: bool = False,
        val_num_cameras: int = 5,
        val_num_frames: int = 5,
    ):
        self.ref_parser = ref_parser
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.invert_masks = invert_masks
        self.selected_frames = sorted(selected_frames)
        self.total_frames = len(selected_frames)
        self.frames_per_step = frames_per_step
        self.factor = factor
        self.patch_size = patch_size

        num_cameras = len(ref_parser.image_names)
        cam_indices = np.arange(num_cameras)

        if split == "val" and val_num_cameras > 0 and val_num_cameras < num_cameras:
            cam_indices = np.linspace(0, num_cameras - 1, val_num_cameras).astype(int)
        elif test_every > 1 and split == "val":
            cam_indices = cam_indices[cam_indices % test_every == 0]

        self.cam_indices = cam_indices
        self.num_cameras = len(cam_indices)

        # Build time blocks: non-overlapping blocks of frames_per_step
        if split == "val":
            # Val: pick evenly-spaced single frames, each block is 1 frame
            if val_num_frames > 0 and val_num_frames < self.total_frames:
                frame_ranks = np.linspace(0, self.total_frames - 1, val_num_frames).astype(int)
            else:
                frame_ranks = np.arange(self.total_frames)
            self.time_blocks = [[int(r)] for r in frame_ranks]
        else:
            # Train: consecutive non-overlapping blocks of frames_per_step
            self.time_blocks = []
            for start in range(0, self.total_frames - frames_per_step + 1, frames_per_step):
                block = list(range(start, start + frames_per_step))
                self.time_blocks.append(block)

        self.num_blocks = len(self.time_blocks)

        # Flat item list: sorted by [block, camera]
        # items[i] = (cam_idx, block_idx)
        self.items = []
        for block_idx in range(self.num_blocks):
            for cam_idx in cam_indices:
                self.items.append((int(cam_idx), block_idx))

        # Image cache: populated by preload_to_device()
        # List[Tensor], one per camera: each is [total_frames, H_i, W_i, 3] uint8 on GPU
        # (cameras have different resolutions due to undistortion)
        self._image_cache = None

        print(f"[MultiFrameDataset] {split}: {self.num_cameras} cameras x "
              f"{self.num_blocks} blocks ({frames_per_step} frames/block) "
              f"= {len(self.items)} items")

    def preload_to_device(self, device: str = "cuda"):
        """
        Preload ALL images as uint8 tensors on GPU — one tensor per camera.

        Uses disk cache + threaded I/O:
        - First run: parallel JPEG reads (16 threads) + save .pt cache
        - Subsequent runs: load .pt cache in ~2s
        """
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image as PILImage

        if not hasattr(self, "_frame_fmt"):
            self._detect_frame_format()

        # Disk cache: <data_dir>/cache_f<factor>_<num_frames>f.pt
        cache_name = f"cache_f{self.factor}_{self.total_frames}f.pt"
        cache_path = os.path.join(os.path.dirname(self.image_dir), cache_name)

        if os.path.exists(cache_path):
            print(f"[Preload] Loading cache: {cache_path}")
            saved = torch.load(cache_path, map_location="cpu", weights_only=True)
            # HIGH-RES: keep on CPU as uint8 to save GPU memory
            cache = saved["images"]  # list of uint8 tensors on CPU
            self._cam_sizes = saved["cam_sizes"]
            self._image_cache = cache
            self._cache_on_cpu = True
            total_gb = sum(t.nelement() for t in cache) / 1024**3
            print(f"[Preload] Done: {len(cache)} cameras, {total_gb:.2f} GB (uint8 on CPU)")
        else:
            print(f"[Preload] Building cache: {self.num_cameras} cams × {self.total_frames} frames (threaded)...")

            def _load_one(args):
                """Load + resize a single image. Returns None if file missing."""
                cam_dir, frame_idx, target_w, target_h = args
                frame_name = f"{frame_idx:{self._frame_fmt}}{self._frame_ext}"
                path = os.path.join(self.image_dir, cam_dir, frame_name)
                if not os.path.exists(path):
                    return None  # missing frame — will be filled with zeros
                try:
                    img = iio.imread(path)[..., :3]
                    if self.factor > 1:
                        img = np.array(PILImage.fromarray(img).resize(
                            (target_w, target_h), PILImage.BICUBIC))
                    return img
                except Exception as e:
                    print(f"[Warning] Failed to load {path}: {e}")
                    return None

            cache = []
            self._cam_sizes = []

            for ci, cam_idx in enumerate(tqdm.tqdm(self.cam_indices, desc="Preload")):
                cam_name = self.ref_parser.image_names[cam_idx]
                cam_dir = os.path.dirname(cam_name)

                # Get target size from first available frame
                tw, th = None, None
                for fi in self.selected_frames:
                    fpath = os.path.join(self.image_dir, cam_dir,
                        f"{fi:{self._frame_fmt}}{self._frame_ext}")
                    if os.path.exists(fpath):
                        h0, w0 = iio.imread(fpath).shape[:2]
                        tw, th = (w0 // self.factor, h0 // self.factor) if self.factor > 1 else (w0, h0)
                        break
                if tw is None:
                    print(f"[Warning] No frames found for camera {cam_dir}, skipping")
                    continue
                self._cam_sizes.append((tw, th))

                # Parallel load all frames for this camera
                tasks = [(cam_dir, fi, tw, th) for fi in self.selected_frames]
                cam_tensor = torch.zeros((self.total_frames, th, tw, 3), dtype=torch.uint8)
                n_missing = 0
                with ThreadPoolExecutor(max_workers=16) as pool:
                    for fi, img in enumerate(pool.map(_load_one, tasks)):
                        if img is not None:
                            cam_tensor[fi] = torch.from_numpy(img)
                        else:
                            n_missing += 1
                if n_missing > 0:
                    print(f"  cam {cam_dir}: {n_missing}/{self.total_frames} frames missing (filled with zeros)")

                cache.append(cam_tensor)

            print(f"[Preload] Saving cache → {cache_path}")
            torch.save({"images": cache, "cam_sizes": self._cam_sizes}, cache_path)

            # HIGH-RES: keep on CPU as uint8
            self._image_cache = cache  # uint8 on CPU
            self._cache_on_cpu = True
            total_gb = sum(t.nelement() for t in cache) / 1024**3
            print(f"[Preload] Done: {len(cache)} cameras, {total_gb:.2f} GB (uint8 on CPU)")

        ws = [s[0] for s in self._cam_sizes]
        hs = [s[1] for s in self._cam_sizes]
        print(f"  Resolution range: W=[{min(ws)},{max(ws)}], H=[{min(hs)},{max(hs)}]")

    def _detect_frame_format(self):
        """Auto-detect frame filename format."""
        import re
        cam_name = self.ref_parser.image_names[0]
        basename = os.path.basename(cam_name)
        name, ext = os.path.splitext(basename)
        if re.fullmatch(r'\d+', name):
            self._frame_fmt = f"0{len(name)}d"
            self._frame_ext = ext
        else:
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
        print(f"[MultiFrameDataset] Frame format: {self._frame_fmt} ext={self._frame_ext}")

    def _get_target_size(self) -> Tuple[int, int]:
        """Get consistent (width, height) for all cameras after downsampling."""
        if not hasattr(self, "_target_wh"):
            # Use COLMAP's imsize_dict (already divided by factor in Parser)
            cam_id = self.ref_parser.camera_ids[0]
            self._target_wh = self.ref_parser.imsize_dict[cam_id]  # (w, h)
        return self._target_wh

    def _load_image(self, cam_idx: int, frame_rank: int) -> np.ndarray:
        """Load and resize a single image to a consistent target size."""
        cam_name = self.ref_parser.image_names[cam_idx]
        cam_dir = os.path.dirname(cam_name)

        if not hasattr(self, "_frame_fmt"):
            self._detect_frame_format()

        frame_idx = self.selected_frames[frame_rank]
        frame_name = f"{frame_idx:{self._frame_fmt}}{self._frame_ext}"
        image_path = os.path.join(self.image_dir, cam_dir, frame_name)
        image = iio.imread(image_path)[..., :3].astype(np.float32) / 255.0

        if self.factor > 1:
            from PIL import Image as PILImage
            target_w, target_h = self._get_target_size()
            image = np.array(
                PILImage.fromarray((image * 255).astype(np.uint8)).resize(
                    (target_w, target_h), PILImage.BICUBIC
                )
            ).astype(np.float32) / 255.0
        return image

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        cam_idx, block_idx = self.items[item]
        frame_ranks = self.time_blocks[block_idx]

        # Camera index within our cam_indices array
        ci = int(np.searchsorted(self.cam_indices, cam_idx))

        if self._image_cache is not None:
            # Fast path: slice from preloaded per-camera GPU tensor
            images = self._image_cache[ci][frame_ranks].float() / 255.0  # uint8 CPU → float
        else:
            # Slow path: load from disk
            frame_images = [self._load_image(cam_idx, r) for r in frame_ranks]
            if self.patch_size is not None:
                h, w = frame_images[0].shape[:2]
                x = np.random.randint(0, max(w - self.patch_size, 1))
                y = np.random.randint(0, max(h - self.patch_size, 1))
                frame_images = [
                    img[y:y + self.patch_size, x:x + self.patch_size]
                    for img in frame_images
                ]
            images = torch.from_numpy(np.stack(frame_images, axis=0)).float()

        # Camera pose & intrinsics (fixed rig — same for all frames)
        camtoworld = self.ref_parser.camtoworlds[cam_idx]
        cam_id = self.ref_parser.camera_ids[cam_idx]
        K = self.ref_parser.Ks_dict[cam_id].copy()
        # Note: K is already scaled by factor inside Parser

        # Normalized timestamps in [0, 1]
        timestamps = np.array(
            [r / max(self.total_frames - 1, 1) for r in frame_ranks],
            dtype=np.float32,
        )

        # Frame numbers for logging (actual file frame indices)
        frame_indices = [self.selected_frames[r] for r in frame_ranks]

        return {
            "camtoworld": torch.from_numpy(camtoworld).float(),         # [4, 4]
            "K": torch.from_numpy(K).float(),                            # [3, 3]
            "images": images,                                             # [N, H, W, 3]
            "timestamps": torch.from_numpy(timestamps).float(),         # [N]
            "image_id": torch.tensor(item, dtype=torch.long),
            "cam_idx": torch.tensor(cam_idx, dtype=torch.long),
            "block_idx": torch.tensor(block_idx, dtype=torch.long),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),  # [N]
        }


class ProgressiveTimeSampler(torch.utils.data.Sampler):
    """
    Sweeps through time blocks progressively, shuffles cameras within each.

    Layout: dataset.items sorted by [block, camera].
    indices [b * num_cameras : (b+1) * num_cameras] belong to block b.

    One full sweep = num_blocks steps (at batch_size=num_cameras).
    After exhausting all blocks, the DataLoader restarts a new sweep.
    """

    def __init__(self, dataset: MultiFrameDataset):
        self.num_cameras = dataset.num_cameras
        self.num_blocks = dataset.num_blocks
        self.total = len(dataset)

    def __iter__(self):
        for b in range(self.num_blocks):
            start = b * self.num_cameras
            cam_perm = torch.randperm(self.num_cameras)
            for c in cam_perm:
                yield start + c.item()

    def __len__(self):
        return self.total


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    disable_viewer: bool = True
    ckpt: Optional[List[str]] = None
    render_traj_path: str = "interp"

    # ---- Data ----
    data_dir: str = "data/scene"
    data_factor: int = 4
    result_dir: str = "results/4c4d"
    test_every: int = 0  # 0 = all cameras for train
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"
    port: int = 8080
    use_masks: bool = False
    mask_dir: Optional[str] = None
    invert_masks: bool = False

    # ---- 4D / Temporal ----
    num_frames: int = 300
    # Consecutive frames per time block per training step.
    # With batch_size=45 (all cameras), each step renders 45 × frames_per_step images.
    frames_per_step: int = 5
    # Sampling mode: "progressive" = sweep time blocks (our default),
    # "random" = random (camera, frame) pairs like the author's DataLoader
    sampling: str = "random"
    frame_start: int = 1
    frame_step: int = 1
    val_num_cameras: int = 5
    val_num_frames: int = 5

    # ---- Training ----
    batch_size: int = 1
    steps_scaler: float = 1.0
    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # ---- GS Init ----
    init_type: str = "sfm"  # "sfm", "random", or "ply"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    # PLY-based init
    static_ply: Optional[str] = None   # path to background PLY
    dynamic_ply: Optional[str] = None  # path to foreground PLY
    freeze_static: bool = False        # True = freeze static PLY; False = all trainable

    # ---- Loss ----
    ssim_lambda: float = 0.2
    near_plane: float = 0.01
    far_plane: float = 1e10

    # ---- Strategy ----
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = False
    sparse_grad: bool = False
    antialiased: bool = False
    random_bkgd: bool = False

    # ---- Regularization ----
    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    # Max cameras per rasterization call. Bounds peak VRAM for the rasterizer.
    # All 45 cameras are still used per step, but rendered in chunks.
    # Set to 0 to render all cameras at once (needs more VRAM).
    render_batch_size: int = 8

    # ---- 4C4D Specific ----
    # Neural Decaying Function (matches author's Coefficient)
    decay_warmup: int = 500         # Start NDF after this many steps
    decay_mlp_lr: float = 1e-3     # Learning rate for the NDF MLP
    decay_mlp_hidden: int = 32     # Hidden dim (author's default)
    decay_dropout: float = 0.1     # Dropout rate in NDF MLP
    decay_f_min: float = 0.996     # Min decay factor
    decay_f_max: float = 0.998     # Max decay factor

    # Temporal attribute learning rates
    temporal_lr: float = 1e-3      # LR for mu_t, scale_t, and velocity
    # Number of Fourier frequencies for 4D SH (Eq. 4: cos(2*pi*n/T * t) * Y_lm)
    # 0 = standard time-invariant SH; 1+ = time-varying color
    num_fourier_freqs: int = 1

    # Tensorboard
    tb_every: int = 100
    # Log GT vs rendered comparison images every N steps (0 = off)
    tb_image_every: int = 200
    # Number of random cameras to show in TB comparisons
    tb_image_num_views: int = 3

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
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


# ---------------------------------------------------------------------------
# Splat initialization
# ---------------------------------------------------------------------------

def create_splats_with_optimizers(
    parser: Parser,
    cfg: Config,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create 4D Gaussian splats with temporal attributes."""
    init_type = cfg.init_type
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        if len(points) == 0:
            print("[WARNING] SFM produced 0 points! Falling back to random init.")
            print("  Consider using MAST3R for initialization in sparse-view settings.")
            init_type = "random"
    if init_type == "random":
        scene_scale = parser.scene_scale * 1.1 * cfg.global_scale
        points = cfg.init_extent * scene_scale * (torch.rand((cfg.init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((cfg.init_num_pts, 3))
    elif init_type != "sfm":
        raise ValueError(f"Unknown init_type: {init_type}")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    scene_scale = parser.scene_scale * 1.1 * cfg.global_scale

    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), cfg.init_opa))

    # Temporal attributes (matching author: random centers, scale covers ~1/5 of range)
    means_t = torch.rand((N,))            # random temporal centers in [0, 1]
    scales_t = torch.full((N,), math.log(math.sqrt(0.2)))  # covers ~1/5 of [0,1] range
    # Right quaternion for SO(4) rotation (identity = no 4D rotation initially)
    rotation_r = torch.zeros((N, 4))
    rotation_r[:, 0] = 1.0  # identity quaternion [1, 0, 0, 0]

    # SH coefficients
    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    params = [
        ("means", nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", nn.Parameter(scales), 5e-3),
        ("quats", nn.Parameter(quats), 1e-3),
        ("opacities", nn.Parameter(opacities), 5e-2),
        ("sh0", nn.Parameter(colors[:, :1, :]), 2.5e-3),
        ("shN", nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
        ("means_t", nn.Parameter(means_t), cfg.temporal_lr),
        ("scales_t", nn.Parameter(scales_t), cfg.temporal_lr),
        ("rotation_r", nn.Parameter(rotation_r), cfg.temporal_lr),
    ]

    splats = nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # No batch-size scaling: batch_size = num cameras rendered per step,
    # NOT a data-parallel batch. Loss is already averaged over all renders.
    optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name, _, lr in params
    }
    return splats, optimizers


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class Runner:
    """4C4D Training engine."""

    def __init__(
        self, local_rank: int, world_rank: int, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)
        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.step = 0

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load COLMAP with frame_num=cfg.frame_start (reference frame for poses)
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every if cfg.test_every > 0 else 1000000,
            frame_num=cfg.frame_start,
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"Scene scale: {self.scene_scale}")
        print(f"Number of cameras: {len(self.parser.image_names)}")

        # Build frame list
        selected_frames = list(range(
            cfg.frame_start,
            cfg.frame_start + cfg.num_frames * cfg.frame_step,
            cfg.frame_step,
        ))
        print(f"Selected frames: {len(selected_frames)} "
              f"(from {selected_frames[0]} to {selected_frames[-1]})")

        image_dir = os.path.join(cfg.data_dir, "images")
        self.trainset = MultiFrameDataset(
            ref_parser=self.parser,
            image_dir=image_dir,
            selected_frames=selected_frames,
            frames_per_step=cfg.frames_per_step,
            split="train",
            test_every=cfg.test_every,
            factor=cfg.data_factor,
            patch_size=cfg.patch_size,
            mask_dir=cfg.mask_dir,
            invert_masks=cfg.invert_masks,
            val_num_cameras=0,
            val_num_frames=0,
        )
        self.valset = MultiFrameDataset(
            ref_parser=self.parser,
            image_dir=image_dir,
            selected_frames=selected_frames,
            frames_per_step=1,  # val: single frames
            split="val",
            factor=cfg.data_factor,
            val_num_cameras=cfg.val_num_cameras,
            val_num_frames=cfg.val_num_frames,
        )

        # Preload all images to GPU for zero-I/O training
        self.trainset.preload_to_device(self.device)

        # Model: Load from PLY or SFM
        self.static_splats = None

        if cfg.static_ply and cfg.dynamic_ply:
            # --- Load both PLYs ---
            print(f"Loading static PLY: {cfg.static_ply}")
            static_data = load_ply_splats(cfg.static_ply, device=self.device)
            N_s = static_data["means"].shape[0]
            if "sh0" not in static_data or static_data["sh0"].shape[1] == 0:
                static_data["sh0"] = torch.zeros((N_s, 1, 3), device=self.device)
            if "shN" not in static_data or static_data["shN"].shape[1] == 0:
                static_data["shN"] = torch.zeros((N_s, 0, 3), device=self.device)

            print(f"Loading dynamic PLY: {cfg.dynamic_ply}")
            dyn_data = load_ply_splats(cfg.dynamic_ply, device=self.device)
            N_d = dyn_data["means"].shape[0]
            if "sh0" not in dyn_data or dyn_data["sh0"].shape[1] == 0:
                dyn_data["sh0"] = torch.zeros((N_d, 1, 3), device=self.device)
            if "shN" not in dyn_data or dyn_data["shN"].shape[1] == 0:
                dyn_data["shN"] = torch.zeros((N_d, 0, 3), device=self.device)

            # Truncate or pad SH to match sh_degree
            target_sh_coeffs = (cfg.sh_degree + 1) ** 2
            for data in [static_data, dyn_data]:
                N = data["means"].shape[0]
                cur = data["sh0"].shape[1] + data["shN"].shape[1]
                if cur < target_sh_coeffs:
                    need = target_sh_coeffs - data["sh0"].shape[1]
                    data["shN"] = torch.zeros((N, need, 3), device=self.device)
                elif cur > target_sh_coeffs:
                    # Truncate higher SH to save memory
                    need = target_sh_coeffs - data["sh0"].shape[1]
                    data["shN"] = data["shN"][:, :max(need, 0), :]

            # Apply normalization transform to BOTH PLYs (match cameras)
            def _normalize_ply(data, label):
                if cfg.normalize_world_space and hasattr(self.parser, 'transform'):
                    T = self.parser.transform
                    means_np = data["means"].cpu().numpy()
                    data["means"] = torch.tensor(
                        means_np @ T[:3, :3].T + T[:3, 3],
                        dtype=torch.float32, device=self.device)
                    scale_factor = np.cbrt(np.linalg.det(T[:3, :3]))
                    if abs(scale_factor - 1.0) > 1e-6:
                        data["scales"] = data["scales"] + math.log(scale_factor)
                    print(f"  {label}: normalized (scale={scale_factor:.4f})")
            _normalize_ply(static_data, "Static")
            _normalize_ply(dyn_data, "Dynamic")

            if cfg.freeze_static:
                # --- Frozen static + trainable dynamic ---
                self.static_splats = static_data  # plain dict, no grad
                print(f"  Static: {N_s:,} (FROZEN)")

                N_trainable = N_d
                trainable_data = dyn_data
            else:
                # --- Merge all into one trainable set ---
                trainable_data = {}
                for key in ["means", "scales", "quats", "opacities"]:
                    trainable_data[key] = torch.cat([static_data[key], dyn_data[key]], 0)
                trainable_data["sh0"] = torch.cat([static_data["sh0"], dyn_data["sh0"]], 0)
                trainable_data["shN"] = torch.cat([static_data["shN"], dyn_data["shN"]], 0)
                N_trainable = N_s + N_d
                print(f"  Merged: {N_s:,} + {N_d:,} = {N_trainable:,} ALL trainable")

            # Temporal attributes: PLY trained at frame 0
            means_t = torch.full((N_trainable,), 0.0, device=self.device)
            scales_t = torch.full((N_trainable,), math.log(math.sqrt(0.2)), device=self.device)
            rotation_r = torch.zeros((N_trainable, 4), device=self.device)
            rotation_r[:, 0] = 1.0  # identity quaternion

            params = [
                ("means", nn.Parameter(trainable_data["means"]), 1.6e-4 * self.scene_scale),
                ("scales", nn.Parameter(trainable_data["scales"]), 5e-3),
                ("quats", nn.Parameter(trainable_data["quats"]), 1e-3),
                ("opacities", nn.Parameter(trainable_data["opacities"]), 5e-2),
                ("sh0", nn.Parameter(trainable_data["sh0"]), 2.5e-3),
                ("shN", nn.Parameter(trainable_data["shN"]), 2.5e-3 / 20),
                ("means_t", nn.Parameter(means_t), cfg.temporal_lr),
                ("scales_t", nn.Parameter(scales_t), cfg.temporal_lr),
                ("rotation_r", nn.Parameter(rotation_r), cfg.temporal_lr),
            ]
            self.splats = nn.ParameterDict({n: v for n, v, _ in params}).to(self.device)
            self.optimizers = {
                name: torch.optim.Adam(
                    [{"params": self.splats[name], "lr": lr, "name": name}],
                    eps=1e-15,
                    betas=(0.9, 0.999),
                )
                for name, _, lr in params
            }
        else:
            # --- Standard init from SFM/random ---
            self.splats, self.optimizers = create_splats_with_optimizers(
                self.parser,
                cfg=cfg,
                device=self.device,
                world_rank=world_rank,
                world_size=world_size,
            )
        print(f"Model initialized. Dynamic GS: {len(self.splats['means'])}")

        # Neural Decaying Coefficient (matches author's Coefficient)
        self.decay_mlp = NeuralDecayCoefficient(
            hidden_dim=cfg.decay_mlp_hidden,
            dropout_rate=cfg.decay_dropout,
        ).to(self.device)
        self.decay_optimizer = torch.optim.Adam(
            self.decay_mlp.parameters(),
            lr=cfg.decay_mlp_lr,
            weight_decay=1e-4,
        )

        # Densification strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()

        # Losses & Metrics
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        else:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            import nerfview
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def compute_4d_gaussians(
        self,
        timestamp: float,
        base_cache: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Compute time-dependent Gaussian attributes using full 4D covariance.

        Builds 4D covariance from two quaternions + 4D scales, then takes the
        Schur complement to get a time-conditioned 3D covariance. This allows
        Gaussian shape to change over time (stretch, rotate, compress).

        Args:
            timestamp: normalized time in [0, 1]
            base_cache: pre-computed activations to avoid redundant work.

        Returns: (means_t, covars_3d, opacities_t, colors, marginal_t)
            means_t:    [N, 3]    time-deformed positions
            covars_3d:  [N, 3, 3] conditional 3D covariance (Schur complement)
            opacities_t:[N]       time-modulated opacity
            colors:     [N, K, 3] SH coefficients
            marginal_t: [N]       temporal weight (for prefiltering)
        """
        means = self.splats["means"]            # [N, 3]
        quats = self.splats["quats"]             # [N, 4] left quaternion
        rotation_r = self.splats["rotation_r"]   # [N, 4] right quaternion
        mu_t = self.splats["means_t"]            # [N]

        # Use cached activations if available
        if base_cache is not None:
            scales = base_cache["scales"]        # [N, 3]
            opacities = base_cache["opacities"]  # [N]
            scale_t = base_cache["scale_t"]      # [N]
            tau = base_cache.get("tau")
            colors = base_cache["colors"]        # [N, K, 3]
        else:
            scales = torch.exp(self.splats["scales"])
            opacities = torch.sigmoid(self.splats["opacities"])
            scale_t = torch.exp(self.splats["scales_t"])
            tau = None
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        dt = timestamp - mu_t  # [N]

        # --- Build full 4D covariance via SO(4) rotation ---
        # L = R_4d @ S_4d, Sigma = L @ L^T
        scales_4d = torch.cat([scales, scale_t.unsqueeze(-1)], dim=1)  # [N, 4]
        L = build_scaling_rotation_4d(scales_4d, quats, rotation_r)    # [N, 4, 4]
        Sigma = L @ L.transpose(1, 2)                                  # [N, 4, 4]

        # Partition the 4D covariance
        cov_11 = Sigma[:, :3, :3]       # [N, 3, 3] spatial
        cov_12 = Sigma[:, :3, 3:4]      # [N, 3, 1] spatial-temporal cross
        cov_t = Sigma[:, 3:4, 3:4]      # [N, 1, 1] temporal variance

        # --- Schur complement: conditional 3D covariance at time t ---
        # cov_3d(t) = cov_11 - cov_12 @ cov_12^T / cov_t
        covars_3d = cov_11 - cov_12 @ cov_12.transpose(1, 2) / (cov_t + 1e-8)  # [N, 3, 3]

        # --- Eq 1: Mean offset from 4D cross-correlation ---
        # delta_mean = cov_12 / cov_t * (t - mu_t)
        mean_offset = (cov_12.squeeze(-1) / (cov_t.squeeze(-1) + 1e-8)) * dt.unsqueeze(-1)  # [N, 3]
        means_t = means + mean_offset  # [N, 3]

        # --- Eq 2-3: Temporal opacity (marginal_t) ---
        # marginal_t = exp(-0.5 * dt^2 / Sigma_{4,4})
        cov_t_scalar = cov_t.squeeze()  # [N]
        marginal_t = torch.exp(-0.5 * dt ** 2 / (cov_t_scalar + 1e-8))  # [N]
        opacities_t = opacities * marginal_t

        # --- Neural Decaying Function ---
        if tau is not None:
            opacities_t = opacities_t * tau

        # --- Eq 4: 4D SH — Fourier-modulated time-varying color ---
        if self.cfg.num_fourier_freqs > 0:
            sh0 = self.splats["sh0"]
            shN = self.splats["shN"]
            K = shN.shape[1]
            if K > 0:
                if not hasattr(self, '_freq_idx') or self._freq_idx.shape[0] != K:
                    freq_per_coeff = torch.arange(K, device=shN.device).float()
                    self._freq_idx = (freq_per_coeff % self.cfg.num_fourier_freqs) + 1
                modulation = torch.cos(
                    2 * math.pi * self._freq_idx.unsqueeze(0) * timestamp
                )
                shN_modulated = shN * modulation.unsqueeze(-1)
                colors = torch.cat([sh0, shN_modulated], 1)

        return means_t, covars_3d, opacities_t, colors, marginal_t

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        timestamp: Optional[float] = None,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        # --- Dynamic Gaussians (with temporal deformation) ---
        use_covars = False
        if timestamp is not None:
            dyn_means, dyn_covars, dyn_opacities, dyn_colors, dyn_marginal = \
                self.compute_4d_gaussians(timestamp)
            use_covars = True
            # Temporal prefilter: skip Gaussians with negligible temporal contribution
            mask = dyn_marginal > 0.05
            if mask.sum() < dyn_means.shape[0]:
                dyn_means = dyn_means[mask]
                dyn_covars = dyn_covars[mask]
                dyn_opacities = dyn_opacities[mask]
                dyn_colors = dyn_colors[mask]
        else:
            dyn_means = self.splats["means"]
            dyn_quats = self.splats["quats"]
            dyn_scales = torch.exp(self.splats["scales"])
            dyn_opacities = torch.sigmoid(self.splats["opacities"])
            dyn_colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        # --- Merge with static Gaussians (frozen, no grad) ---
        if self.static_splats is not None:
            sc = self._static_cache if hasattr(self, '_static_cache') else None
            if sc is None:
                s = self.static_splats
                sc = {
                    "means": s["means"],
                    "quats": s["quats"],
                    "scales": torch.exp(s["scales"]),
                    "opacities": torch.sigmoid(s["opacities"]),
                    "colors": torch.cat([s["sh0"], s["shN"]], 1) if s["shN"].shape[1] > 0 else s["sh0"],
                }

            d_sh = dyn_colors.shape[1]
            s_sh = sc["colors"].shape[1]
            max_sh = max(d_sh, s_sh)
            dc = dyn_colors if d_sh == max_sh else torch.cat([dyn_colors, torch.zeros(dyn_colors.shape[0], max_sh - d_sh, 3, device=dyn_colors.device)], 1)
            s_col = sc["colors"] if s_sh == max_sh else torch.cat([sc["colors"], torch.zeros(sc["colors"].shape[0], max_sh - s_sh, 3, device=sc["colors"].device)], 1)

            if use_covars:
                # Build covars for static Gaussians from their quats/scales
                s_L = torch.zeros(sc["scales"].shape[0], 3, 3, device=sc["means"].device)
                s_R = torch.zeros_like(s_L)
                # Simple: build covariance from scales and quats for static
                s_q = F.normalize(sc["quats"], dim=-1)
                r, x, y, z = s_q[:, 0], s_q[:, 1], s_q[:, 2], s_q[:, 3]
                R00 = 1 - 2*(y*y + z*z); R01 = 2*(x*y - r*z); R02 = 2*(x*z + r*y)
                R10 = 2*(x*y + r*z); R11 = 1 - 2*(x*x + z*z); R12 = 2*(y*z - r*x)
                R20 = 2*(x*z - r*y); R21 = 2*(y*z + r*x); R22 = 1 - 2*(x*x + y*y)
                s_R = torch.stack([R00,R01,R02,R10,R11,R12,R20,R21,R22], -1).reshape(-1, 3, 3)
                s_S = torch.diag_embed(sc["scales"])  # [N, 3, 3]
                s_L = s_R @ s_S
                s_covars = s_L @ s_L.transpose(1, 2)  # [N, 3, 3]

                means = torch.cat([dyn_means, sc["means"]], 0)
                covars = torch.cat([dyn_covars, s_covars], 0)
                opacities = torch.cat([dyn_opacities, sc["opacities"]], 0)
                colors = torch.cat([dc, s_col], 0)
            else:
                means = torch.cat([dyn_means, sc["means"]], 0)
                quats = torch.cat([dyn_quats, sc["quats"]], 0)
                scales = torch.cat([dyn_scales, sc["scales"]], 0)
                opacities = torch.cat([dyn_opacities, sc["opacities"]], 0)
                colors = torch.cat([dc, s_col], 0)
        else:
            means = dyn_means
            opacities = dyn_opacities
            colors = dyn_colors
            if use_covars:
                covars = dyn_covars
            else:
                quats = dyn_quats
                scales = dyn_scales

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        raster_kwargs = dict(
            means=means,
            quats=None if use_covars else quats,
            scales=None if use_covars else scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
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
        )
        if use_covars:
            raster_kwargs["covars"] = covars
        raster_kwargs.update(kwargs)

        render_colors, render_alphas, info = rasterization(**raster_kwargs)
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # Direct GPU iteration — data preloaded, no DataLoader needed.
        ds = self.trainset

        def _make_block_iter():
            """Yields (block_idx) for one progressive sweep."""
            for b in range(ds.num_blocks):
                yield b

        block_iter = _make_block_iter()

        # Pre-compute per-camera tensors on device (reused every step)
        all_camtoworlds = torch.from_numpy(
            np.stack([self.parser.camtoworlds[ci] for ci in ds.cam_indices])
        ).float().to(device)  # [num_cameras, 4, 4]
        all_Ks = torch.stack([
            torch.from_numpy(self.parser.Ks_dict[self.parser.camera_ids[ci]].copy()).float()
            for ci in ds.cam_indices
        ]).to(device)  # [num_cameras, 3, 3]
        # Note: Ks_dict already scaled by factor inside Parser — do NOT scale again
        all_viewmats = torch.linalg.inv(all_camtoworlds)  # [num_cameras, 4, 4] — pre-computed once

        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            self.step = step

            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # ---- Sample (camera, frame) pairs for this step ----
            all_cam_indices = list(range(ds.num_cameras))

            if cfg.sampling == "random":
                # Pick a random adjacent time window, ALL cameras at each timestamp
                # → max batching (C=num_cameras) + temporal coherence + diversity
                n_times = cfg.frames_per_step
                max_start = max(ds.total_frames - n_times, 0)
                start_rank = torch.randint(0, max_start + 1, (1,)).item()
                window_ranks = list(range(start_rank, min(start_rank + n_times, ds.total_frames)))
                samples_by_frame = {fr: list(all_cam_indices) for fr in window_ranks}
            else:
                # Progressive sweep through time blocks, all cameras
                try:
                    block_idx = next(block_iter)
                except StopIteration:
                    block_iter = _make_block_iter()
                    block_idx = next(block_iter)
                frame_ranks_block = ds.time_blocks[block_idx]
                samples_by_frame = {fr: list(all_cam_indices) for fr in frame_ranks_block}

            unique_frame_ranks = sorted(samples_by_frame.keys())
            all_timestamps = torch.tensor(
                [r / max(ds.total_frames - 1, 1) for r in unique_frame_ranks],
                device=device,
            )
            frame_indices = torch.tensor(
                [ds.selected_frames[r] for r in unique_frame_ranks], device=device,
            )

            # SH schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # --- Pre-compute constants for this step ---
            rasterize_mode = "antialiased" if cfg.antialiased else "classic"
            use_absgrad = (cfg.strategy.absgrad
                           if isinstance(cfg.strategy, DefaultStrategy) else False)

            # Cache static Gaussian activations (frozen — same every step)
            if self.static_splats is not None and not hasattr(self, '_static_cache'):
                s = self.static_splats
                self._static_cache = {
                    "means": s["means"],
                    "quats": s["quats"],
                    "scales": torch.exp(s["scales"]),
                    "opacities": torch.sigmoid(s["opacities"]),
                    "colors": torch.cat([s["sh0"], s["shN"]], 1) if s["shN"].shape[1] > 0 else s["sh0"],
                }

            # Cache base param activations ONCE per step (same across all timestamps)
            base_scales = torch.exp(self.splats["scales"])
            base_opacities = torch.sigmoid(self.splats["opacities"])
            base_scale_t = torch.exp(self.splats["scales_t"])
            base_colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

            # Compute decay MLP ONCE per step (inputs don't change across timestamps)
            if step >= cfg.decay_warmup:
                positions_4d = torch.cat([
                    self.splats["means"],
                    self.splats["means_t"].unsqueeze(-1),
                ], dim=1)
                scales_4d = torch.cat([base_scales, base_scale_t.unsqueeze(-1)], dim=1)
                coef = self.decay_mlp(
                    base_opacities.unsqueeze(-1), positions_4d, scales_4d,
                ).squeeze(-1)
                base_tau = cfg.decay_f_min + (cfg.decay_f_max - cfg.decay_f_min) * coef
            else:
                base_tau = None

            base_cache = {
                "scales": base_scales,
                "opacities": base_opacities,
                "scale_t": base_scale_t,
                "tau": base_tau,
                "colors": base_colors,
            }

            # Build per-camera resolution map for batched rasterization
            cam_sizes = ds._cam_sizes  # list of (w, h) per camera index

            # --- Render: group cameras by resolution, batch each group ---
            loss = 0.0
            last_info = None
            total_renders = 0

            for t_idx, frame_rank in enumerate(unique_frame_ranks):
                timestamp = all_timestamps[t_idx].item()
                cams_for_this_frame = samples_by_frame[frame_rank]

                # Compute 4D Gaussians ONCE per timestamp (with Schur complement)
                dyn_means, dyn_covars, dyn_opacities, dyn_colors, dyn_marginal = \
                    self.compute_4d_gaussians(timestamp, base_cache=base_cache)

                # Temporal prefilter: skip Gaussians with negligible contribution
                tmask = dyn_marginal > 0.05
                if tmask.sum() < dyn_means.shape[0]:
                    dyn_means = dyn_means[tmask]
                    dyn_covars = dyn_covars[tmask]
                    dyn_opacities = dyn_opacities[tmask]
                    dyn_colors = dyn_colors[tmask]

                # Merge with static (build covars for static from cached quats/scales)
                if self.static_splats is not None:
                    sc = self._static_cache
                    # Build 3D covariance for static Gaussians (time-invariant)
                    if not hasattr(self, '_static_covars'):
                        s_q = F.normalize(sc["quats"], dim=-1)
                        rr, xx, yy, zz = s_q[:, 0], s_q[:, 1], s_q[:, 2], s_q[:, 3]
                        R = torch.stack([
                            1-2*(yy*yy+zz*zz), 2*(xx*yy-rr*zz), 2*(xx*zz+rr*yy),
                            2*(xx*yy+rr*zz), 1-2*(xx*xx+zz*zz), 2*(yy*zz-rr*xx),
                            2*(xx*zz-rr*yy), 2*(yy*zz+rr*xx), 1-2*(xx*xx+yy*yy),
                        ], -1).reshape(-1, 3, 3)
                        S = torch.diag_embed(sc["scales"])
                        LS = R @ S
                        self._static_covars = LS @ LS.transpose(1, 2)  # [N_s, 3, 3]

                    d_sh, s_sh = dyn_colors.shape[1], sc["colors"].shape[1]
                    max_sh = max(d_sh, s_sh)
                    dc = dyn_colors if d_sh == max_sh else torch.cat([dyn_colors, torch.zeros(dyn_colors.shape[0], max_sh - d_sh, 3, device=device)], 1)
                    s_col = sc["colors"] if s_sh == max_sh else torch.cat([sc["colors"], torch.zeros(sc["colors"].shape[0], max_sh - s_sh, 3, device=device)], 1)

                    gs_means = torch.cat([dyn_means, sc["means"]], 0)
                    gs_covars = torch.cat([dyn_covars, self._static_covars], 0)
                    gs_opacities = torch.cat([dyn_opacities, sc["opacities"]], 0)
                    gs_colors = torch.cat([dc, s_col], 0)
                else:
                    gs_means = dyn_means
                    gs_covars = dyn_covars
                    gs_opacities = dyn_opacities
                    gs_colors = dyn_colors

                # Group cameras by (width, height) for batched rasterization
                res_groups = {}
                for ci in cams_for_this_frame:
                    wh = cam_sizes[ci]
                    res_groups.setdefault(wh, []).append(ci)

                for (w_grp, h_grp), grp_cams in res_groups.items():
                    C = len(grp_cams)
                    grp_viewmats = all_viewmats[grp_cams]
                    grp_Ks = all_Ks[grp_cams]
                    # HIGH-RES: images on CPU (uint8), move batch to GPU on the fly
                    grp_pixels = torch.stack([
                        ds._image_cache[ci][frame_rank] for ci in grp_cams
                    ]).to(device=device, dtype=torch.float32).div_(255.0)

                    render_colors, render_alphas, info = rasterization(
                        means=gs_means,
                        quats=None,   # using covars instead
                        scales=None,
                        opacities=gs_opacities,
                        colors=gs_colors,
                        viewmats=grp_viewmats,
                        Ks=grp_Ks,
                        width=w_grp,
                        height=h_grp,
                        packed=cfg.packed,
                        absgrad=use_absgrad,
                        sparse_grad=cfg.sparse_grad,
                        rasterize_mode=rasterize_mode,
                        distributed=self.world_size > 1,
                        camera_model=cfg.camera_model,
                        sh_degree=sh_degree_to_use,
                        near_plane=cfg.near_plane,
                        far_plane=cfg.far_plane,
                        covars=gs_covars,
                    )
                    colors = render_colors[..., 0:3] if render_colors.shape[-1] == 4 else render_colors
                    # colors: [C, H, W, 3], grp_pixels: [C, H, W, 3]

                    l1loss = F.l1_loss(colors, grp_pixels)
                    ssimloss = 1.0 - fused_ssim(
                        colors.permute(0, 3, 1, 2),
                        grp_pixels.permute(0, 3, 1, 2),
                        padding="valid",
                    )
                    loss = loss + (l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda) * C
                    total_renders += C
                    last_info = info

            # Average over all (timestamp × camera) renders
            loss = loss / total_renders

            # Regularizations
            if cfg.opacity_reg > 0.0:
                loss += cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            if cfg.scale_reg > 0.0:
                loss += cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()

            # Strategy pre-backward (skip when using PLY — IDs span static+dynamic)
            if self.static_splats is None:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=last_info,
                )

            # Single backward pass — gradients averaged across entire time block
            loss.backward()

            t_start = all_timestamps[0].item()
            t_end = all_timestamps[-1].item()
            desc = (f"loss={loss.item():.3f}| sh={sh_degree_to_use}| "
                    f"t=[{t_start:.3f}→{t_end:.3f}]| GS={len(self.splats['means'])}")
            pbar.set_description(desc)

            # Tensorboard scalars
            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.add_scalar("train/block_t_start", t_start, step)
                self.writer.add_scalar("train/block_t_end", t_end, step)
                self.writer.flush()

            # Tensorboard GT vs Rendered comparison images (per-camera rendering)
            if (world_rank == 0 and cfg.tb_image_every > 0
                    and step % cfg.tb_image_every == 0 and step > 0):
                with torch.no_grad():
                    n_views = min(cfg.tb_image_num_views, ds.num_cameras)
                    view_cams = list(range(n_views))
                    n_ts = len(unique_frame_ranks)
                    mid_t = all_timestamps[n_ts // 2].item()
                    mid_frame_idx = frame_indices[n_ts // 2].item()
                    mid_rank = unique_frame_ranks[n_ts // 2]

                    for ci in view_cams:
                        c2w_i = all_camtoworlds[ci:ci + 1]
                        K_i = all_Ks[ci:ci + 1]
                        gt_i = ds._image_cache[ci][mid_rank:mid_rank + 1].to(device=device, dtype=torch.float32).div_(255.0)
                        h_i, w_i = gt_i.shape[1:3]

                        rend_i, _, _ = self.rasterize_splats(
                            camtoworlds=c2w_i, Ks=K_i,
                            width=w_i, height=h_i,
                            timestamp=mid_t,
                            sh_degree=sh_degree_to_use,
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                        )
                        rend_i = torch.clamp(rend_i[..., 0:3], 0.0, 1.0)

                        # Upscale 4x for visibility
                        scale = 4
                        rend_up = F.interpolate(
                            rend_i.permute(0, 3, 1, 2), scale_factor=scale,
                            mode="bilinear", align_corners=False,
                        )[0]  # [3, H*4, W*4]
                        gt_up = F.interpolate(
                            gt_i.permute(0, 3, 1, 2), scale_factor=scale,
                            mode="bilinear", align_corners=False,
                        )[0]
                        pair = torch.cat([gt_up, rend_up], dim=2)  # [3, H*4, W*8]
                        self.writer.add_image(
                            f"train_cmp/cam{ci}_frame{mid_frame_idx}",
                            pair, step,
                        )
                    self.writer.flush()

            # Save checkpoint
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "elapsed_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print(f"Step: {step}", stats)
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)

                data_to_save = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                    "decay_mlp": self.decay_mlp.state_dict(),
                }
                torch.save(
                    data_to_save,
                    f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt",
                )

            # Optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Optimize NDF MLP (after warmup)
            if step >= cfg.decay_warmup:
                self.decay_optimizer.step()
                self.decay_optimizer.zero_grad(set_to_none=True)

            # Note: NO persistent opacity decay here. The coefficient in
            # compute_4d_gaussians already modulates rendered opacity (with gradients
            # for MLP training). Persistent decay on top would double-apply tau,
            # causing opacity to collapse to zero over many steps (0.997^20K ≈ 0).

            for scheduler in schedulers:
                scheduler.step()

            # Strategy post-backward (skip for PLY mode)
            if self.static_splats is None:
                if isinstance(self.cfg.strategy, DefaultStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=last_info,
                        packed=cfg.packed,
                    )
                elif isinstance(self.cfg.strategy, MCMCStrategy):
                    self.cfg.strategy.step_post_backward(
                        params=self.splats,
                        optimizers=self.optimizers,
                        state=self.strategy_state,
                        step=step,
                        info=last_info,
                        lr=schedulers[0].get_last_lr()[0],
                    )

            # Eval
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_rays_per_step = pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluate on validation set."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)

        img_counter = 0
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)    # [1, 4, 4]
            Ks = data["K"].to(device)                       # [1, 3, 3]
            all_images = data["images"].to(device)          # [1, 2, H, W, 3]
            all_timestamps = data["timestamps"].to(device)  # [1, 2]
            height, width = all_images.shape[2:4]

            # Evaluate on first frame of the pair (val pairs are same-frame duplicates)
            timestamp = all_timestamps[0, 0].item()
            pixels = all_images[:, 0]  # [1, H, W, 3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                timestamp=timestamp,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)

            if self.world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{img_counter:04d}.png", canvas
                )
                img_counter += 1

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if self.world_rank == 0:
            ellipse_time /= max(len(valloader), 1)
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            })
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, "
                f"LPIPS: {stats['lpips']:.3f} Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int, stage: str = "val"):
        """Render trajectory video at mid-sequence timestamp."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[:len(self.parser.camtoworlds) // 2]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(camtoworlds_all, 6)
        elif cfg.render_traj_path == "ellipse":
            height_val = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(camtoworlds_all, height=height_val)

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0),
            ],
            axis=1,
        )
        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        n_cam_poses = len(camtoworlds_all)
        num_t = cfg.num_frames
        render_fps = 30

        # Single 4D video: ~90 frames (3 sec at 30fps)
        # Camera smoothly moves along trajectory while time advances over a random interval
        n_video_frames = min(render_fps * 3, n_cam_poses)  # 3 seconds
        cam_indices = np.linspace(0, n_cam_poses - 1, n_video_frames).astype(int)
        # Pick a random 3-sec window in the time range [0, 1]
        t_window = min(n_video_frames / max(num_t, 1), 0.5)  # cover at most half the sequence
        t_start = np.random.uniform(0, max(1.0 - t_window, 0.01))
        t_end = t_start + t_window
        timestamps = np.linspace(t_start, t_end, n_video_frames)
        print(f"  4D video: {n_video_frames} frames, time [{t_start:.2f}→{t_end:.2f}]")

        # Pre-compute all viewmats for trajectory
        traj_viewmats = torch.linalg.inv(camtoworlds_all)

        writer = imageio.get_writer(
            f"{video_dir}/{stage}_4d_{step}.mp4",
            fps=render_fps, format="FFMPEG", codec="libx264",
        )

        # Batch render: group consecutive frames that share the same timestamp
        # For max speed, render in chunks using batched rasterization
        chunk_size = min(n_video_frames, 8)  # render up to 8 cameras at once
        for start in tqdm.trange(0, n_video_frames, chunk_size, desc="Rendering 4D video"):
            end = min(start + chunk_size, n_video_frames)
            for i in range(start, end):
                ci = cam_indices[i]
                renders, _, _ = self.rasterize_splats(
                    camtoworlds=camtoworlds_all[ci:ci + 1], Ks=K[None],
                    width=width, height=height,
                    timestamp=float(timestamps[i]),
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                )
                canvas = torch.clamp(renders[..., 0:3], 0, 1).squeeze(0).cpu().numpy()
                writer.append_data((canvas * 255).astype(np.uint8))
        writer.close()
        print(f"4D video ({n_video_frames} frames, {n_video_frames/render_fps:.1f}s): "
              f"{video_dir}/{stage}_4d_{step}.mp4")

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state, img_wh):
        """Callable for interactive viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            timestamp=0.5,  # render at mid-sequence
            sh_degree=self.cfg.sh_degree,
            radius_clip=3.0,
        )
        return render_colors[0].cpu().numpy()


    @torch.no_grad()
    def export_per_frame_ply(self, step: int):
        """Export a standard 3DGS PLY for each timestamp.

        Decomposes the time-conditioned 3D covariance (Schur complement) back
        to scales + quaternions via eigendecomposition, so each PLY can be
        viewed in any standard 3DGS viewer.
        """
        from plyfile import PlyData, PlyElement
        cfg = self.cfg
        device = self.device
        num_t = cfg.num_frames

        ply_dir = f"{cfg.result_dir}/ply_frames"
        os.makedirs(ply_dir, exist_ok=True)

        print(f"Exporting {num_t} per-frame PLY files...")
        for fi in tqdm.trange(num_t, desc="Exporting PLYs"):
            timestamp = fi / max(num_t - 1, 1)

            means_t, covars_3d, opacities_t, colors, marginal_t = \
                self.compute_4d_gaussians(timestamp)

            # Filter by temporal relevance
            mask = marginal_t > 0.01
            means_t = means_t[mask]
            covars_3d = covars_3d[mask]
            opacities_t = opacities_t[mask]
            colors = colors[mask]

            # Merge with static if present
            if self.static_splats is not None and hasattr(self, '_static_cache'):
                sc = self._static_cache
                if hasattr(self, '_static_covars'):
                    s_covars = self._static_covars
                else:
                    s_q = F.normalize(sc["quats"], dim=-1)
                    rr, xx, yy, zz = s_q[:, 0], s_q[:, 1], s_q[:, 2], s_q[:, 3]
                    R = torch.stack([
                        1-2*(yy*yy+zz*zz), 2*(xx*yy-rr*zz), 2*(xx*zz+rr*yy),
                        2*(xx*yy+rr*zz), 1-2*(xx*xx+zz*zz), 2*(yy*zz-rr*xx),
                        2*(xx*zz-rr*yy), 2*(yy*zz+rr*xx), 1-2*(xx*xx+yy*yy),
                    ], -1).reshape(-1, 3, 3)
                    S = torch.diag_embed(sc["scales"])
                    LS = R @ S
                    s_covars = LS @ LS.transpose(1, 2)
                means_t = torch.cat([means_t, sc["means"]], 0)
                covars_3d = torch.cat([covars_3d, s_covars], 0)
                opacities_t = torch.cat([opacities_t, sc["opacities"]], 0)
                d_sh, s_sh = colors.shape[1], sc["colors"].shape[1]
                max_sh = max(d_sh, s_sh)
                if d_sh < max_sh:
                    colors = torch.cat([colors, torch.zeros(colors.shape[0], max_sh - d_sh, 3, device=device)], 1)
                s_col = sc["colors"] if s_sh == max_sh else torch.cat([sc["colors"], torch.zeros(sc["colors"].shape[0], max_sh - s_sh, 3, device=device)], 1)
                colors = torch.cat([colors, s_col], 0)

            N = means_t.shape[0]

            # Decompose covariance → scales + quaternions via eigendecomposition
            # Sigma = R @ diag(eigenvalues) @ R^T → scales = sqrt(eigenvalues), quats from R
            eigenvalues, eigenvectors = torch.linalg.eigh(covars_3d)  # [N, 3], [N, 3, 3]
            scales = torch.sqrt(torch.clamp(eigenvalues, min=1e-8))  # [N, 3]
            scales_log = torch.log(scales)  # log-space for PLY

            # Convert rotation matrix to quaternion
            R = eigenvectors  # [N, 3, 3]
            trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
            quats = torch.zeros(N, 4, device=device)
            s = torch.sqrt(torch.clamp(trace + 1, min=1e-8)) * 2  # s = 4*w
            quats[:, 0] = 0.25 * s
            quats[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (s + 1e-8)
            quats[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (s + 1e-8)
            quats[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (s + 1e-8)
            quats = F.normalize(quats, dim=-1)

            # Opacity: convert to logit-space for PLY
            opacities_logit = torch.logit(opacities_t.clamp(1e-7, 1 - 1e-7))

            # Build PLY arrays
            xyz = means_t.cpu().numpy()
            normals = np.zeros_like(xyz)
            sh0 = colors[:, :1, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            shN = colors[:, 1:, :].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
            opa = opacities_logit.unsqueeze(1).cpu().numpy()
            sc_np = scales_log.cpu().numpy()
            qt = quats.cpu().numpy()

            attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            for i in range(sh0.shape[1]):
                attrs.append(f'f_dc_{i}')
            for i in range(shN.shape[1]):
                attrs.append(f'f_rest_{i}')
            attrs.append('opacity')
            for i in range(3):
                attrs.append(f'scale_{i}')
            for i in range(4):
                attrs.append(f'rot_{i}')

            dtype_full = [(a, 'f4') for a in attrs]
            elements = np.empty(N, dtype=dtype_full)
            elements[:] = list(map(tuple, np.concatenate(
                [xyz, normals, sh0, shN, opa, sc_np, qt], axis=1
            )))

            el = PlyElement.describe(elements, 'vertex')
            frame_idx = fi + cfg.frame_start
            PlyData([el]).write(f"{ply_dir}/frame_{frame_idx:06d}.ply")

        print(f"Exported {num_t} PLY files to {ply_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        if "decay_mlp" in ckpts[0]:
            runner.decay_mlp.load_state_dict(ckpts[0]["decay_mlp"])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()
        # Export per-frame PLY files after training
        runner.export_per_frame_ply(step=cfg.max_steps - 1)

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Train 4C4D on multi-frame data
    python simple_trainer_4c4d.py default \\
        --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted \\
        --result_dir results/face_4c4d \\
        --data_factor 15 \\
        --num_frames 100

    # With MCMC strategy
    python simple_trainer_4c4d.py mcmc \\
        --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted \\
        --result_dir results/face_4c4d_mcmc \\
        --data_factor 15 \\
        --num_frames 100
    ```
    """

    configs = {
        "default": (
            "4C4D training with DefaultStrategy densification.",
            Config(
                strategy=DefaultStrategy(
                    verbose=True,
                    reset_every=1000000,  # disable opacity reset — neural decay handles it
                ),
            ),
        ),
        "mcmc": (
            "4C4D training with MCMC densification strategy.",
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

    cli(main, cfg, verbose=True)
