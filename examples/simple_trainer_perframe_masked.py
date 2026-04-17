"""
Per-Frame Masked Gaussian Splatting Trainer

Trains Gaussians frame-by-frame using SAM2 binary masks to separate
static (frozen) and dynamic (trainable) regions.

Two experiments:
  A) mask_only:    Keep only Gaussians in mask area, train from scratch per frame
  B) freeze_split: Freeze static Gaussians, train dynamic ones, loss on object only

Usage:
  python simple_trainer_perframe_masked.py \
      --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \
      --ply_path /data/shared/elaheh/elly_static_v2/ply/point_cloud_29999.ply \
      --mask_base_dir tracking_experiment \
      --result_dir results/elly_perframe \
      --num_frames 5 --mode freeze_split
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import imageio
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from fused_ssim import fused_ssim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio

from datasets.colmap import Parser
from utils import set_random_seed

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data_dir: str = ""
    ply_path: str = ""                  # first-frame trained PLY
    mask_base_dir: str = "tracking_experiment"  # relative to data_dir
    mask_subfolder: str = "sam2"        # subfolder inside each camera dir
    result_dir: str = "results/perframe"
    data_factor: int = 1
    num_frames: int = 5
    frame_start: int = 0
    first_frame_steps: int = 29000   # more steps for first frame (from scratch)
    post_split_steps: int = 1000    # fine-tune dynamic after split on frame 0
    steps_per_frame: int = 7000     # steps for subsequent frames (fine-tune)
    densify_every: int = 700
    densify_start: int = 500
    densify_stop: int = 5000
    sh_degree: int = 3
    batch_size: int = 1                 # cameras per step
    normalize_world_space: bool = True
    mode: str = "freeze_split"          # "freeze_split" or "mask_only"
    ssim_lambda: float = 0.2
    # Early stopping: stop frame training when loss < threshold AND psnr > threshold
    early_stop_loss: float = 0.015
    early_stop_psnr: float = 31.0
    # Render only dynamic Gaussians during training (skip static for speed)
    render_dynamic_only: bool = False
    # Anti-halo weight (penalize alpha outside mask)
    alpha_outside_weight: float = 0.5
    # Keyframe stride: reset init every N frames (0 = sequential, no reset)
    keyframe_stride: int = 0
    # Previous frame index for motion map (parallel mode). -1 = auto (sequential)
    prev_frame_idx: int = -1
    near_plane: float = 0.01
    far_plane: float = 1e10
    # Learning rates
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4


# ---------------------------------------------------------------------------
# PLY I/O using gsplat's exporter
# ---------------------------------------------------------------------------

def load_ply(path: str, device: str = "cuda") -> nn.ParameterDict:
    """Load PLY into a ParameterDict matching simple_trainer conventions."""
    from plyfile import PlyData
    plydata = PlyData.read(path)
    v = plydata["vertex"].data
    N = len(v)

    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    f_dc = [n for n in v.dtype.names if n.startswith("f_dc_")]
    f_rest = [n for n in v.dtype.names if n.startswith("f_rest_")]
    sc = sorted([n for n in v.dtype.names if n.startswith("scale_")])
    rt = sorted([n for n in v.dtype.names if n.startswith("rot_")])

    dc = np.vstack([v[n] for n in sorted(f_dc)]).T if f_dc else np.zeros((N, 3))
    rest = np.vstack([v[n] for n in sorted(f_rest)]).T if f_rest else None
    scales = np.vstack([v[n] for n in sc]).T
    quats = np.vstack([v[n] for n in rt]).T
    opacities = np.array(v["opacity"])

    sh0 = dc.reshape(-1, 3, 1).transpose(0, 2, 1)  # [N, 1, 3]
    if rest is not None and len(f_rest) > 0:
        rest_k = len(f_rest) // 3
        shN = rest.reshape(-1, 3, rest_k).transpose(0, 2, 1)  # [N, K, 3]
    else:
        shN = np.zeros((N, 0, 3))

    splats = nn.ParameterDict({
        "means": nn.Parameter(torch.tensor(xyz, dtype=torch.float32, device=device)),
        "scales": nn.Parameter(torch.tensor(scales, dtype=torch.float32, device=device)),
        "quats": nn.Parameter(torch.tensor(quats, dtype=torch.float32, device=device)),
        "opacities": nn.Parameter(torch.tensor(opacities, dtype=torch.float32, device=device)),
        "sh0": nn.Parameter(torch.tensor(sh0, dtype=torch.float32, device=device)),
        "shN": nn.Parameter(torch.tensor(shN, dtype=torch.float32, device=device)),
    })
    print(f"Loaded {N:,} Gaussians from {path}")
    return splats


def save_ply(splats: nn.ParameterDict, path: str, ref_ply_path: Optional[str] = None):
    """Save ParameterDict to PLY.

    If ref_ply_path is provided, uses the exact same PLY format/dtype as the reference.
    Otherwise falls back to standard 3DGS format.
    """
    from plyfile import PlyData, PlyElement
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    if ref_ply_path and os.path.exists(ref_ply_path):
        # Match exact format of reference PLY
        ref = PlyData.read(ref_ply_path)
        ref_dtype = ref["vertex"].data.dtype
        N = splats["means"].shape[0]

        # Build structured array with same dtype
        data = np.empty(N, dtype=ref_dtype)
        means = splats["means"].detach().cpu().numpy()
        data["x"], data["y"], data["z"] = means[:, 0], means[:, 1], means[:, 2]
        if "nx" in ref_dtype.names:
            data["nx"] = data["ny"] = data["nz"] = 0.0

        # SH: f_dc and f_rest
        sh0 = splats["sh0"].detach().cpu().numpy()  # [N, 1, 3]
        sh0_flat = sh0.reshape(N, -1)  # [N, 3]
        for i, name in enumerate(n for n in ref_dtype.names if n.startswith("f_dc_")):
            data[name] = sh0_flat[:, i] if i < sh0_flat.shape[1] else 0.0
        shN = splats["shN"].detach().cpu().numpy()  # [N, K, 3]
        shN_flat = shN.transpose(0, 2, 1).reshape(N, -1)  # [N, 3*K]
        for i, name in enumerate(n for n in ref_dtype.names if n.startswith("f_rest_")):
            data[name] = shN_flat[:, i] if i < shN_flat.shape[1] else 0.0

        data["opacity"] = splats["opacities"].detach().cpu().numpy()
        scales = splats["scales"].detach().cpu().numpy()
        for i, name in enumerate(sorted(n for n in ref_dtype.names if n.startswith("scale_"))):
            data[name] = scales[:, i]
        quats = splats["quats"].detach().cpu().numpy()
        for i, name in enumerate(sorted(n for n in ref_dtype.names if n.startswith("rot_"))):
            data[name] = quats[:, i]

        PlyData([PlyElement.describe(data, "vertex")]).write(path)
    else:
        # Fallback: standard format
        means = splats["means"].detach().cpu().numpy()
        normals = np.zeros_like(means)
        sh0 = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        shN = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = splats["opacities"].detach().unsqueeze(1).cpu().numpy()
        scales = splats["scales"].detach().cpu().numpy()
        quats = splats["quats"].detach().cpu().numpy()

        attrs = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(sh0.shape[1]):
            attrs.append(f'f_dc_{i}')
        for i in range(shN.shape[1]):
            attrs.append(f'f_rest_{i}')
        attrs.append('opacity')
        for i in range(scales.shape[1]):
            attrs.append(f'scale_{i}')
        for i in range(quats.shape[1]):
            attrs.append(f'rot_{i}')

        dtype_full = [(a, 'f4') for a in attrs]
        elements = np.empty(means.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, np.concatenate(
            [means, normals, sh0, shN, opacities, scales, quats], axis=1)))
        PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

    print(f"Saved {splats['means'].shape[0]:,} Gaussians to {path}")


# ---------------------------------------------------------------------------
# Mask loading
# ---------------------------------------------------------------------------

def load_mask(mask_dir: str, cam_name: str, frame_idx: int,
              frame_fmt: str = "06d", ext: str = ".png",
              factor: int = 1, device: str = "cuda") -> Tensor:
    """Load binary mask. White (255) = object/dynamic."""
    path = os.path.join(mask_dir, cam_name, f"{frame_idx:{frame_fmt}}{ext}")
    if not os.path.exists(path):
        return None
    mask = iio.imread(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.float32)
    if factor > 1:
        from PIL import Image as PILImage
        h, w = mask.shape
        mask = np.array(PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
            (w // factor, h // factor), PILImage.NEAREST)).astype(np.float32) / 255.0
    return torch.from_numpy(mask).float().to(device)


def load_image(image_dir: str, cam_name: str, frame_idx: int,
               frame_fmt: str = "06d", ext: str = ".jpg",
               factor: int = 1, device: str = "cuda") -> Tensor:
    """Load image as [H, W, 3] float32 tensor."""
    path = os.path.join(image_dir, cam_name, f"{frame_idx:{frame_fmt}}{ext}")
    if not os.path.exists(path):
        return None
    img = iio.imread(path)[..., :3].astype(np.float32) / 255.0
    if factor > 1:
        from PIL import Image as PILImage
        h, w = img.shape[:2]
        img = np.array(PILImage.fromarray((img * 255).astype(np.uint8)).resize(
            (w // factor, h // factor), PILImage.BICUBIC)).astype(np.float32) / 255.0
    return torch.from_numpy(img).float().to(device)


# ---------------------------------------------------------------------------
# Gaussian classification
# ---------------------------------------------------------------------------

@torch.no_grad()
def classify_gaussians(
    splats: nn.ParameterDict,
    masks: List[Tensor],           # [num_cams] each [H, W]
    viewmats: Tensor,             # [C, 4, 4]
    Ks: Tensor,                   # [C, 3, 3]
    widths: List[int],
    heights: List[int],
    device: str = "cuda",
    threshold: float = 0.85,       # fraction of views where Gaussian must be in mask
) -> Tensor:
    """
    Classify Gaussians as dynamic (True) or static (False) based on mask projection.

    A Gaussian is dynamic if it projects into the white (object) mask area
    in at least `threshold` fraction of camera views that see it.

    Returns: [N] bool tensor — True = dynamic/trainable
    """
    N = splats["means"].shape[0]
    in_mask_count = torch.zeros(N, dtype=torch.float32, device=device)
    visible_count = torch.zeros(N, dtype=torch.float32, device=device)

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = splats["sh0"]  # minimal SH for classification

    for ci in tqdm.trange(len(masks), desc="Classifying", leave=False):
        if masks[ci] is None:
            continue
        w, h = widths[ci], heights[ci]
        mask = masks[ci]  # [H, W]

        _, _, meta = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmats[ci:ci + 1], Ks=Ks[ci:ci + 1],
            width=w, height=h,
            sh_degree=0,
            near_plane=0.01, far_plane=1e10,
            packed=True,
        )
        # Packed mode: gaussian_ids and means2d are [nnz]
        gs_ids = meta["gaussian_ids"]   # [nnz] — which Gaussian each entry belongs to
        means2d = meta["means2d"]       # [nnz, 2] — 2D projected positions

        # Sample mask at projected positions
        x = means2d[:, 0].long().clamp(0, w - 1)
        y = means2d[:, 1].long().clamp(0, h - 1)
        in_mask = mask[y, x] > 0.5  # white = object

        # Count per-Gaussian: how many views see it in mask vs total visible
        unique_gs = gs_ids.unique()
        visible_count.index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))
        in_mask_count.index_add_(0, gs_ids[in_mask], torch.ones(in_mask.sum(), dtype=torch.float32, device=device))

    # Dynamic if seen in mask in >= threshold fraction of views
    frac = in_mask_count / visible_count.clamp(min=1)
    is_dynamic = frac >= threshold

    n_dyn = is_dynamic.sum().item()
    n_static = N - n_dyn
    print(f"  Classification: {n_dyn:,} dynamic, {n_static:,} static "
          f"(threshold={threshold:.0%}, {(visible_count > 0).sum().item():,} visible)")
    return is_dynamic


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

def train_frame(
    splats: nn.ParameterDict,
    parser: Parser,
    frame_idx: int,
    cfg: Config,
    is_dynamic: Optional[Tensor],  # [N] bool, None for mask_only mode
    device: str = "cuda",
    writer: Optional[SummaryWriter] = None,
    global_step_offset: int = 0,
    is_first_frame: bool = False,
):
    """Train Gaussians for a single frame."""
    image_dir = os.path.join(cfg.data_dir, "images")
    mask_dir = os.path.join(cfg.data_dir, cfg.mask_base_dir)

    # Detect frame format
    cam_names = [os.path.dirname(n) for n in parser.image_names]
    # Deduplicate while preserving order
    seen = set()
    unique_cams = []
    for c in cam_names:
        if c not in seen:
            seen.add(c)
            unique_cams.append(c)
    cam_names = unique_cams

    num_cameras = len(cam_names)
    print(f"  Frame {frame_idx}: {num_cameras} cameras")

    # Pre-compute camera data
    all_camtoworlds = torch.from_numpy(parser.camtoworlds).float().to(device)
    all_viewmats = torch.linalg.inv(all_camtoworlds)
    all_Ks = torch.stack([
        torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]].copy()).float()
        for i in range(num_cameras)
    ]).to(device)

    # Build optimizers
    scene_scale = parser.scene_scale * 1.1
    param_lrs = [
        ("means", cfg.lr_means * scene_scale),
        ("scales", cfg.lr_scales),
        ("quats", cfg.lr_quats),
        ("opacities", cfg.lr_opacities),
        ("sh0", cfg.lr_sh0),
        ("shN", cfg.lr_shN),
    ]
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15, betas=(0.9, 0.999),
        )
        for name, lr in param_lrs
    }

    # Strategy for densification — disable pruning (prune_opa=0)
    # Frozen Gaussians must never be removed
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=cfg.densify_start,
        refine_stop_iter=cfg.densify_stop,
        refine_every=cfg.densify_every,
        reset_every=1000000,  # disable opacity reset
        prune_opa=0.0,        # disable opacity pruning — frozen GS must stay
    )
    strategy.check_sanity(splats, optimizers)
    strategy_state = strategy.initialize_state(scene_scale=scene_scale)

    # LR scheduler
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / cfg.steps_per_frame)
        ),
    ]

    # Preload all images + masks for this frame to GPU (eliminates per-step I/O)
    preloaded_images = []
    preloaded_masks = []
    for ci in range(num_cameras):
        cam_name = cam_names[ci]
        img = load_image(image_dir, cam_name, frame_idx,
                         factor=cfg.data_factor, device=device)
        msk = load_mask(mask_dir, os.path.join(cam_name, cfg.mask_subfolder),
                        frame_idx, factor=cfg.data_factor, device=device)
        preloaded_images.append(img)
        preloaded_masks.append(msk)

    # Gaussian-level freeze: find which Gaussians are already correct
    # Render all cameras, check error, freeze Gaussians with low error in all views
    # Freeze check: run when loading from PLY and not the original first frame
    # In parallel mode: --ply_path is set and frame_start > 0
    # Gaussian-level freeze: render current Gaussians, compare vs GT within mask.
    # Gaussians that render correctly → freeze (no gradient, no strategy).
    # Gaussians with high error → trainable (strategy can densify/prune).
    do_freeze = False  # using motion-guided loss instead of freezing
    if do_freeze:
        with torch.no_grad():
            N = splats["means"].shape[0]
            max_error_per_gs = torch.zeros(N, device=device)
            gs_seen_count = torch.zeros(N, device=device)

            means_3d = splats["means"].detach()  # [N, 3]
            ones = torch.ones(N, 1, device=device)
            means_h = torch.cat([means_3d, ones], dim=1)  # [N, 4]

            for ci in range(num_cameras):
                if preloaded_images[ci] is None or preloaded_masks[ci] is None:
                    continue
                gt_img = preloaded_images[ci]
                msk = preloaded_masks[ci]
                H, W = gt_img.shape[:2]

                # Render to get pixel error
                means = splats["means"]; quats = splats["quats"]
                scales = torch.exp(splats["scales"])
                opacities = torch.sigmoid(splats["opacities"])
                colors = torch.cat([splats["sh0"], splats["shN"]], 1)
                renders, _, _ = rasterization(
                    means=means, quats=quats, scales=scales,
                    opacities=opacities, colors=colors,
                    viewmats=all_viewmats[ci:ci + 1], Ks=all_Ks[ci:ci + 1],
                    width=W, height=H, sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane, far_plane=cfg.far_plane, packed=True,
                )
                pixel_error = (renders[0, ..., :3] - gt_img).abs().mean(dim=-1)  # [H, W]
                pixel_error = pixel_error * msk  # zero outside mask

                # Project Gaussian centers to 2D (simple, no packed mode needed)
                vm = all_viewmats[ci]  # [4, 4]
                K_cam = all_Ks[ci]     # [3, 3]
                cam_pts = (vm[:3, :3] @ means_3d.T + vm[:3, 3:4]).T  # [N, 3]
                z = cam_pts[:, 2].clamp(min=0.01)
                px = (K_cam[0, 0] * cam_pts[:, 0] / z + K_cam[0, 2]).long().clamp(0, W - 1)
                py = (K_cam[1, 1] * cam_pts[:, 1] / z + K_cam[1, 2]).long().clamp(0, H - 1)
                in_front = cam_pts[:, 2] > 0.1

                # Sample error at each Gaussian's projected position
                gs_error = pixel_error[py, px] * in_front.float()
                max_error_per_gs = torch.max(max_error_per_gs, gs_error)
                gs_seen_count += in_front.float()

            # Freeze: low error across all views AND visible in at least 1 view
            is_gs_correct = (max_error_per_gs < 0.05) & (gs_seen_count > 0)
            n_correct = is_gs_correct.sum().item()
            n_needs_fix = N - n_correct
            print(f"  Gaussian freeze: {n_correct:,} correct (frozen), "
                  f"{n_needs_fix:,} needs fixing (trainable)")

            # Split: frozen (plain tensors) vs trainable (nn.Parameters)
            frozen_data = {k: splats[k].data[is_gs_correct].detach() for k in splats}
            splats = nn.ParameterDict({
                k: nn.Parameter(splats[k].data[~is_gs_correct].clone())
                for k in splats
            }).to(device)

            # Rebuild optimizers and strategy for trainable only
            optimizers = {
                name: torch.optim.Adam(
                    [{"params": splats[name], "lr": lr, "name": name}],
                    eps=1e-15, betas=(0.9, 0.999),
                )
                for name, lr in param_lrs
            }
            strategy.check_sanity(splats, optimizers)
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        frozen_data = None

    # Training loop
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    pbar = tqdm.trange(cfg.steps_per_frame, desc=f"Frame {frame_idx}")

    for step in pbar:
        global_step = global_step_offset + step

        # Random camera — use preloaded data (zero I/O)
        ci = torch.randint(0, num_cameras, (1,)).item()
        gt_image = preloaded_images[ci]
        if gt_image is None:
            continue
        mask = preloaded_masks[ci]

        H, W = gt_image.shape[:2]
        gt = gt_image.unsqueeze(0)  # [1, H, W, 3]

        # SH degree: use full degree when fine-tuning from PLY, ramp when from scratch
        if cfg.ply_path:
            sh_degree = cfg.sh_degree  # PLY already trained with full SH
        else:
            sh_degree = min(step // 1000, cfg.sh_degree)

        # Render: trainable Gaussians (nn.Parameters) + frozen (plain tensors)
        t_means = splats["means"]
        t_quats = splats["quats"]
        t_scales = torch.exp(splats["scales"])
        t_opacities = torch.sigmoid(splats["opacities"])
        t_colors = torch.cat([splats["sh0"], splats["shN"]], 1)

        if frozen_data is not None:
            # Merge frozen (no grad) + trainable for rendering
            f = frozen_data
            f_colors = torch.cat([f["sh0"], f["shN"]], 1)
            means = torch.cat([t_means, f["means"]], 0)
            quats = torch.cat([t_quats, f["quats"]], 0)
            scales = torch.cat([t_scales, torch.exp(f["scales"])], 0)
            opacities = torch.cat([t_opacities, torch.sigmoid(f["opacities"])], 0)
            colors = torch.cat([t_colors, f_colors], 0)
        else:
            means = t_means
            quats = t_quats
            scales = t_scales
            opacities = t_opacities
            colors = t_colors

        renders, alphas, info = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=all_viewmats[ci:ci + 1],
            Ks=all_Ks[ci:ci + 1],
            width=W, height=H,
            sh_degree=sh_degree,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            packed=True,
        )
        rendered = renders[..., :3]  # [1, H, W, 3]

        # Motion-guided masked loss
        if mask is not None:
            obj_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]

            # Motion map: compare current vs previous frame mask
            prev_fidx = cfg.prev_frame_idx if cfg.prev_frame_idx >= 0 else (frame_idx - 1)
            if prev_fidx >= 0 and prev_fidx != frame_idx:
                prev_m = load_mask(mask_dir, os.path.join(cam_names[ci], cfg.mask_subfolder),
                                   prev_fidx, factor=cfg.data_factor, device=device)
                if prev_m is not None:
                    motion = (mask - prev_m).abs().unsqueeze(0).unsqueeze(-1)
                    weight_map = 0.1 + 0.9 * motion
                else:
                    weight_map = torch.ones_like(obj_mask)
            else:
                weight_map = torch.ones_like(obj_mask)

            weighted_mask = obj_mask * weight_map
            mask_sum = weighted_mask.sum().clamp(min=1.0)

            l1 = (torch.abs(rendered - gt) * weighted_mask).sum() / mask_sum / 3
            ssim_loss = 1.0 - fused_ssim(
                (rendered * obj_mask).permute(0, 3, 1, 2),
                (gt * obj_mask).permute(0, 3, 1, 2),
                padding="valid",
            )
        else:
            l1 = F.l1_loss(rendered, gt)
            ssim_loss = 1.0 - fused_ssim(
                rendered.permute(0, 3, 1, 2),
                gt.permute(0, 3, 1, 2),
                padding="valid",
            )

        loss = l1 * (1 - cfg.ssim_lambda) + ssim_loss * cfg.ssim_lambda

        # Distance-weighted anti-halo: gentle near mask edge, harsh far away
        # Allows cloth to extend slightly beyond mask, kills head halo
        if mask is not None:
            outside_mask = 1.0 - obj_mask  # [1, H, W, 1]
            # Compute distance from mask edge (dilate mask = distance map)
            with torch.no_grad():
                # Approximate distance: dilate mask by 10px = "near edge" zone
                dilated = F.max_pool2d(mask.unsqueeze(0).unsqueeze(0), 21, stride=1, padding=10)
                near_edge = (dilated.squeeze() > 0.5) & (mask < 0.5)  # outside mask but within 10px
                near_edge = near_edge.unsqueeze(0).unsqueeze(-1).float()  # [1, H, W, 1]
                far_from_edge = outside_mask - near_edge  # far outside

            # Near edge: weak penalty (0.1x) — let cloth extend
            # Far from edge: full penalty — kill halo
            alpha_near = (alphas * near_edge).sum() / near_edge.sum().clamp(min=1.0)
            color_near = (rendered * near_edge).abs().sum() / (near_edge.sum() * 3).clamp(min=1.0)
            alpha_far = (alphas * far_from_edge).sum() / far_from_edge.sum().clamp(min=1.0)
            color_far = (rendered * far_from_edge).abs().sum() / (far_from_edge.sum() * 3).clamp(min=1.0)

            loss = loss + cfg.alpha_outside_weight * (
                0.0 * (alpha_near + color_near) +   # no penalty near edge — cloth extends
                1.0 * (alpha_far + color_far)        # harsh far away — kills halo
            )

        # Strategy pre-backward
        strategy.step_pre_backward(
            params=splats, optimizers=optimizers,
            state=strategy_state, step=step, info=info,
        )

        loss.backward()

        # Freeze static Gaussians (zero their gradients)
        if is_dynamic is not None:
            for name in splats:
                if splats[name].grad is not None:
                    splats[name].grad[~is_dynamic] = 0
        # Note: frozen Gaussians are plain tensors (not nn.Parameters),
        # so they have no gradients and can't be pruned by strategy.

        # Step optimizers
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        for sched in schedulers:
            sched.step()

        # Strategy post-backward (densification/pruning)
        # Filter info to only include trainable Gaussians (frozen have higher IDs)
        if frozen_data is not None and "gaussian_ids" in info:
            N_train = splats["means"].shape[0]
            mask_train = info["gaussian_ids"] < N_train
            filtered_info = {k: v for k, v in info.items()}
            for k in ["gaussian_ids", "means2d", "radii", "depths", "conics", "opacities"]:
                if k in filtered_info and filtered_info[k] is not None:
                    filtered_info[k] = filtered_info[k][mask_train]
        else:
            filtered_info = info
        strategy.step_post_backward(
            params=splats, optimizers=optimizers,
            state=strategy_state, step=step, info=filtered_info,
            packed=True,
        )

        # Update is_dynamic mask if Gaussians were added/removed
        if is_dynamic is not None and len(splats["means"]) != len(is_dynamic):
            new_N = len(splats["means"])
            old_N = len(is_dynamic)
            if new_N > old_N:
                # New Gaussians from densification → mark as dynamic
                pad = torch.ones(new_N - old_N, dtype=torch.bool, device=device)
                is_dynamic = torch.cat([is_dynamic, pad])
            else:
                # Gaussians pruned → truncate mask
                is_dynamic = is_dynamic[:new_N]

        # No active mask pruning — single-camera projection is unreliable.
        # Use filter_dynamic_ply.py for post-processing instead.

        # Logging
        if step % 100 == 0:
            with torch.no_grad():
                if mask is not None:
                    # PSNR on masked (object) area only
                    m = obj_mask.permute(0, 3, 1, 2)  # [1, 1, H, W]
                    rendered_m = rendered.permute(0, 3, 1, 2) * m
                    gt_m = gt.permute(0, 3, 1, 2) * m
                    mse = ((rendered_m - gt_m) ** 2).sum() / (m.sum() * 3).clamp(min=1)
                    psnr_val = (-10 * torch.log10(mse)).item()
                else:
                    psnr_val = psnr_metric(
                        rendered.permute(0, 3, 1, 2),
                        gt.permute(0, 3, 1, 2),
                    ).item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr_val:.1f}",
                             gs=len(splats["means"]))
            if writer:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/psnr", psnr_val, global_step)
                writer.add_scalar("train/num_gs", len(splats["means"]), global_step)

            # Early stopping (minimum 2000 steps)
            if (step >= 2000 and
                    loss.item() < cfg.early_stop_loss and
                    psnr_val > cfg.early_stop_psnr):
                print(f"  Early stop at step {step}: loss={loss.item():.4f}, psnr={psnr_val:.1f}")
                break

        # TB image comparison — show GT, rendered, and mask overlay
        if step % 500 == 0 and step > 0 and writer:
            with torch.no_grad():
                rendered_clamped = torch.clamp(rendered, 0, 1)
                if mask is not None:
                    # Show: GT | Rendered | Masked-GT | Masked-Rendered
                    mask_vis = obj_mask.expand_as(gt)  # [1, H, W, 3]
                    canvas = torch.cat([
                        gt * mask_vis,
                        rendered_clamped * mask_vis,
                    ], dim=2)
                else:
                    canvas = torch.cat([gt, rendered_clamped], dim=2)
                canvas = canvas.squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                writer.add_image(f"frame{frame_idx}/cam{ci}",
                                 canvas.transpose(2, 0, 1), global_step)

        # Eval at specified steps
        if is_first_frame:
            eval_at = {7000, 15000, 29000, cfg.steps_per_frame - 1}
        else:
            eval_at = {cfg.steps_per_frame - 1}
        if step in eval_at:
            eval_frame(splats, parser, frame_idx, cfg,
                       device=device, writer=writer, global_step=global_step)

    return is_dynamic, splats, frozen_data


@torch.no_grad()
def eval_frame(
    splats: nn.ParameterDict,
    parser: Parser,
    frame_idx: int,
    cfg: Config,
    device: str = "cuda",
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0,
):
    """Evaluate on all cameras — render, compute masked PSNR, save images."""
    image_dir = os.path.join(cfg.data_dir, "images")
    mask_dir = os.path.join(cfg.data_dir, cfg.mask_base_dir)
    cam_names = list(dict.fromkeys(os.path.dirname(n) for n in parser.image_names))
    num_cameras = len(cam_names)

    all_viewmats = torch.linalg.inv(
        torch.from_numpy(parser.camtoworlds).float().to(device))
    all_Ks = torch.stack([
        torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]].copy()).float()
        for i in range(num_cameras)
    ]).to(device)

    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)

    psnr_vals = []
    render_dir = f"{cfg.result_dir}/renders/frame{frame_idx:06d}_step{global_step}"
    os.makedirs(render_dir, exist_ok=True)

    for ci in range(num_cameras):
        cam_name = cam_names[ci]
        gt_image = load_image(image_dir, cam_name, frame_idx,
                              factor=cfg.data_factor, device=device)
        if gt_image is None:
            continue
        mask = load_mask(mask_dir, os.path.join(cam_name, cfg.mask_subfolder),
                         frame_idx, factor=cfg.data_factor, device=device)
        H, W = gt_image.shape[:2]

        renders, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=all_viewmats[ci:ci + 1], Ks=all_Ks[ci:ci + 1],
            width=W, height=H, sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane, packed=True,
        )
        rendered = torch.clamp(renders[..., :3], 0, 1)
        gt = gt_image.unsqueeze(0)

        if mask is not None:
            m = mask.unsqueeze(0).unsqueeze(-1)
            mse = ((rendered * m - gt * m) ** 2).sum() / (m.sum() * 3).clamp(min=1)
        else:
            mse = ((rendered - gt) ** 2).mean()
        psnr_vals.append((-10 * torch.log10(mse)).item())

        if ci % 8 == 0:
            m_vis = mask.unsqueeze(0).unsqueeze(-1).expand_as(gt) if mask is not None else torch.ones_like(gt)
            canvas = torch.cat([gt * m_vis, rendered * m_vis], dim=2)
            imageio.imwrite(f"{render_dir}/cam{ci:02d}.png",
                            (canvas.squeeze(0).cpu().numpy() * 255).astype(np.uint8))

    avg_psnr = np.mean(psnr_vals) if psnr_vals else 0
    print(f"  Eval frame {frame_idx} step {global_step}: PSNR={avg_psnr:.2f} ({len(psnr_vals)} cams)")
    if writer:
        writer.add_scalar("eval/psnr", avg_psnr, global_step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = tyro.cli(Config)
    set_random_seed(42)
    device = "cuda"

    os.makedirs(cfg.result_dir, exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/ply_frames/dynamic", exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/ply_static", exist_ok=True)
    os.makedirs(f"{cfg.result_dir}/renders", exist_ok=True)

    writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

    with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
        yaml.dump(vars(cfg), f)

    # Load COLMAP parser
    parser = Parser(
        data_dir=cfg.data_dir,
        factor=cfg.data_factor,
        normalize=cfg.normalize_world_space,
        test_every=1000000,
        frame_num=cfg.frame_start,
    )
    print(f"Scene scale: {parser.scene_scale}")
    print(f"Cameras: {len(parser.image_names)}")

    # Camera info
    cam_names = list(dict.fromkeys(os.path.dirname(n) for n in parser.image_names))
    num_cameras = len(cam_names)
    image_dir = os.path.join(cfg.data_dir, "images")
    mask_dir = os.path.join(cfg.data_dir, cfg.mask_base_dir)

    # Pre-compute camera data
    all_camtoworlds = torch.from_numpy(parser.camtoworlds).float().to(device)
    all_viewmats = torch.linalg.inv(all_camtoworlds)
    all_Ks = torch.stack([
        torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]].copy()).float()
        for i in range(num_cameras)
    ]).to(device)

    # Initialize Gaussians: from PLY or SFM
    if cfg.ply_path:
        splats = load_ply(cfg.ply_path, device=device)
        # Apply normalization transform if needed
        if cfg.normalize_world_space and hasattr(parser, 'transform'):
            T = parser.transform
            means_np = splats["means"].detach().cpu().numpy()
            means_transformed = means_np @ T[:3, :3].T + T[:3, 3]
            splats["means"].data = torch.tensor(
                means_transformed, dtype=torch.float32, device=device)
            scale_factor = np.cbrt(np.linalg.det(T[:3, :3]))
            if abs(scale_factor - 1.0) > 1e-6:
                splats["scales"].data = splats["scales"].data + math.log(scale_factor)
            print(f"Applied normalization (scale={scale_factor:.4f})")
    else:
        # SFM init from COLMAP points3D
        from utils import knn, rgb_to_sh
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        N = points.shape[0]
        print(f"SFM init: {N:,} points from COLMAP")

        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg)
        scales = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)
        quats = torch.rand((N, 4))
        opacities = torch.logit(torch.full((N,), 0.1))
        colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)

        splats = nn.ParameterDict({
            "means": nn.Parameter(points),
            "scales": nn.Parameter(scales),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(opacities),
            "sh0": nn.Parameter(colors[:, :1, :]),
            "shN": nn.Parameter(colors[:, 1:, :]),
        }).to(device)

    # Truncate/pad SH to match sh_degree
    target_sh = (cfg.sh_degree + 1) ** 2
    cur_sh = splats["sh0"].shape[1] + splats["shN"].shape[1]
    if cur_sh > target_sh:
        need = target_sh - splats["sh0"].shape[1]
        splats["shN"] = nn.Parameter(splats["shN"][:, :max(need, 0), :])
        print(f"Truncated SH: {cur_sh} → {target_sh}")
    elif cur_sh < target_sh:
        need = target_sh - splats["sh0"].shape[1]
        pad = torch.zeros(splats["means"].shape[0], need, 3, device=device)
        splats["shN"] = nn.Parameter(torch.cat([splats["shN"].data, pad], 1))
        print(f"Padded SH: {cur_sh} → {target_sh}")

    print(f"Initial Gaussians: {len(splats['means']):,}")

    # Per-frame training loop
    for fi in range(cfg.num_frames):
        frame_idx = cfg.frame_start + fi
        print(f"\n{'='*60}")
        print(f"Frame {frame_idx} ({fi+1}/{cfg.num_frames})")
        print(f"{'='*60}")

        # Skip if already completed (dynamic PLY exists)
        dyn_ply = f"{cfg.result_dir}/ply_frames/dynamic/{frame_idx:06d}.ply"
        if os.path.exists(dyn_ply):
            print(f"  Frame {frame_idx} already done, loading and skipping...")
            splats = load_ply(dyn_ply, device=device)
            # Pad SH if needed
            cur = splats["sh0"].shape[1] + splats["shN"].shape[1]
            if cur < target_sh:
                need = target_sh - splats["sh0"].shape[1]
                splats["shN"] = nn.Parameter(torch.cat([splats["shN"].data,
                    torch.zeros(splats["means"].shape[0], need, 3, device=device)], 1))
            continue

        # Load masks for this frame (all cameras)
        masks = []
        widths, heights = [], []
        for ci in range(num_cameras):
            cam_name = cam_names[ci]
            mask = load_mask(mask_dir, os.path.join(cam_name, cfg.mask_subfolder),
                             frame_idx, factor=cfg.data_factor, device=device)
            masks.append(mask)
            # Get image size
            w, h = list(parser.imsize_dict.values())[ci] if ci < len(parser.imsize_dict) else list(parser.imsize_dict.values())[0]
            widths.append(w)
            heights.append(h)

        # Classify Gaussians with two thresholds:
        # 1) Dynamic = projects to mask in >=threshold of views (strict, clean object)
        # 2) Remove from static = projects to mask in >0% of views (no person remnants)
        # Skip classification for SFM init on frame 0 — just train all with masked loss
        if cfg.mode in ("freeze_split", "mask_only") and not (fi == 0 and not cfg.ply_path):
            # Only do border discard on the very first frame (initial split)
            # For subsequent frames (including parallel subprocesses), skip discard
            if frame_idx == 0:
                is_dynamic = classify_gaussians(
                    splats, masks, all_viewmats, all_Ks,
                    widths, heights, device=device,
                )
                touches_mask = classify_gaussians(
                    splats, masks, all_viewmats, all_Ks,
                    widths, heights, device=device, threshold=0.01,
                )
                n_discarded = (touches_mask & ~is_dynamic).sum().item()
                print(f"  Discarding {n_discarded:,} border Gaussians")
                keep = is_dynamic | ~touches_mask
                splats = nn.ParameterDict({
                    k: nn.Parameter(v.data[keep]) for k, v in splats.items()
                }).to(device)
                is_dynamic = is_dynamic[keep]
                if cfg.mode == "mask_only":
                    splats = nn.ParameterDict({
                        k: nn.Parameter(v.data[is_dynamic]) for k, v in splats.items()
                    }).to(device)
                    print(f"  mask_only: kept {splats['means'].shape[0]:,} dynamic")
                    is_dynamic = None
            else:
                # Frame > 0: no discard, no reclassification. Just train all.
                is_dynamic = None
        else:
            is_dynamic = None

        # Determine steps for this frame
        if frame_idx == 0 and cfg.ply_path:
            frame_steps = cfg.post_split_steps   # Frame 0 with PLY: 0 = just save, >0 = fine-tune
        elif frame_idx == 0 and not cfg.ply_path:
            frame_steps = cfg.first_frame_steps  # Frame 0 SFM from scratch: full training
        else:
            frame_steps = cfg.steps_per_frame    # All other frames: fine-tune
        global_offset = sum(
            (cfg.first_frame_steps if j == 0 and not cfg.ply_path else
             cfg.post_split_steps if j == 0 and cfg.ply_path else
             cfg.steps_per_frame) for j in range(fi)
        )

        # Train this frame (skip if 0 steps — just save the split)
        if frame_steps > 0:
            orig_steps = cfg.steps_per_frame
            cfg.steps_per_frame = frame_steps
            is_dynamic, splats, frozen_data = train_frame(
                splats, parser, frame_idx, cfg,
                is_dynamic=is_dynamic,
                device=device, writer=writer,
                global_step_offset=global_offset,
                is_first_frame=(fi == 0),
            )
            cfg.steps_per_frame = orig_steps
        else:
            print(f"  Skipping training (0 steps) — saving split as-is")
            frozen_data = None

        # Merge frozen + trainable back together for saving
        if frozen_data is not None:
            merged_splats = nn.ParameterDict({
                k: nn.Parameter(torch.cat([splats[k].data, frozen_data[k]], 0))
                for k in splats
            })
        else:
            merged_splats = splats

        # For save: extract dynamic only if in freeze_split mode
        if is_dynamic is not None:
            dyn_splats = nn.ParameterDict({
                k: nn.Parameter(v.data[is_dynamic]) for k, v in merged_splats.items()
            })
        else:
            dyn_splats = merged_splats
        dyn_splats_clean = dyn_splats

        # Save: use direct PLY row filter for untrained frames (preserves exact format),
        # fall back to save_ply for trained frames
        out_ply = f"{cfg.result_dir}/ply_frames/dynamic/{frame_idx:06d}.ply"
        if frame_steps == 0 and cfg.ply_path and is_dynamic is not None:
            # Direct filter from original PLY — preserves exact byte format + colors
            from plyfile import PlyData, PlyElement
            orig_ply = PlyData.read(cfg.ply_path)
            # is_dynamic indexes into the 'keep' mask (after border discard)
            # We need the mask relative to the original PLY
            keep_np = (is_dynamic & torch.ones_like(is_dynamic)).cpu().numpy()
            # But splats was already filtered by 'keep' earlier, so is_dynamic
            # indexes into the filtered set. We need to map back.
            # Simpler: just save from the filtered original rows
            orig_v = orig_ply["vertex"].data
            keep_all = (is_dynamic | ~is_dynamic).cpu().numpy()  # all True in filtered set
            # Actually, the splats were already filtered. Re-read original and apply both masks.
            all_keep = torch.zeros(len(orig_v), dtype=torch.bool)
            # touches_mask and is_dynamic were computed on the filtered set.
            # Just use the saved split_output approach instead:
            print(f"  Saving frame 0 dynamic via direct PLY filter")
            # Re-classify on the ORIGINAL PLY
            orig_splats = load_ply(cfg.ply_path, device=device)
            dyn_mask = classify_gaussians(
                orig_splats, masks, all_viewmats, all_Ks,
                widths, heights, device=device, threshold=0.85,
            ).cpu().numpy()
            PlyData([PlyElement.describe(orig_v[dyn_mask], "vertex")]).write(out_ply)
            print(f"  Saved {dyn_mask.sum():,} Gaussians (direct filter)")
        else:
            save_ply(dyn_splats_clean, out_ply, ref_ply_path=cfg.ply_path)

        if is_dynamic is not None:
            # freeze_split: rebuild splats with cleaned dynamic + unchanged static
            splats = nn.ParameterDict({
                k: nn.Parameter(torch.cat([
                    dyn_splats_clean[k].data,
                    splats[k].data[~is_dynamic],
                ])) for k in splats
            }).to(device)
            n_clean = dyn_splats_clean["means"].shape[0]
            n_static = (~is_dynamic).sum().item()
            is_dynamic = torch.cat([
                torch.ones(n_clean, dtype=torch.bool, device=device),
                torch.zeros(n_static, dtype=torch.bool, device=device),
            ])
            if fi == 0:
                static_splats = nn.ParameterDict({
                    k: nn.Parameter(splats[k].data[~is_dynamic]) for k in splats
                })
                save_ply(static_splats, f"{cfg.result_dir}/ply_static/static.ply",
                         ref_ply_path=cfg.ply_path)
        else:
            # mask_only: carry forward merged (trainable + frozen) for next frame
            splats = merged_splats

        # Keyframe stride: save keyframe, reset init for next group
        if cfg.keyframe_stride > 0:
            frames_since_keyframe = (fi % cfg.keyframe_stride)
            if frames_since_keyframe == 0:
                # This is a keyframe — save as reference for next group
                self_keyframe_splats = nn.ParameterDict({
                    k: nn.Parameter(splats[k].data.clone()) for k in splats
                })
                print(f"  Keyframe saved (stride={cfg.keyframe_stride})")
            elif frames_since_keyframe < cfg.keyframe_stride - 1:
                # Mid-group: reset to keyframe for next frame (don't accumulate drift)
                splats = nn.ParameterDict({
                    k: nn.Parameter(self_keyframe_splats[k].data.clone()) for k in self_keyframe_splats
                }).to(device)

        if is_dynamic is not None:
            print(f"  Final: {len(splats['means']):,} Gaussians "
                  f"({is_dynamic.sum().item():,} dynamic, {(~is_dynamic).sum().item():,} static)")
        else:
            print(f"  Final: {len(splats['means']):,} Gaussians (all dynamic)")

    writer.close()
    print(f"\nDone! Results at {cfg.result_dir}")


if __name__ == "__main__":
    main()
