"""Fast multi-frame training: single process, shared static cache, pre-loaded images.

Eliminates per-frame overhead:
- Python/CUDA startup (2s) → done once
- Static PLY loading (1s) → done once
- Static cache rendering (0.1s) → done once
- Image loading (3s/frame) → all frames pre-loaded at startup
- CUDA JIT compilation → done once on frame 1
"""

import argparse
import json
import math
import os
import shutil
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from gsplat.exporter import load_ply_gaussian
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import relocate, sample_add, inject_noise_to_position, reset_opa, split


class DynamicRegionMask:
    """Voxel-based spatial mask for constraining gaussians to the dynamic region."""

    def __init__(self, voxel_labels, grid_bounds, device="cuda"):
        # Dynamic region = dynamic voxels (2) + occupied_empty (3) for padding
        self.allowed = torch.from_numpy((voxel_labels == 2) | (voxel_labels == 3)).to(device)
        self.grid_min = torch.from_numpy(grid_bounds[0]).float().to(device)
        self.grid_max = torch.from_numpy(grid_bounds[1]).float().to(device)
        self.R = voxel_labels.shape[0]
        n_dyn = (voxel_labels == 2).sum()
        n_allowed = self.allowed.sum().item()
        print(f"[ROI] Voxel grid {self.R}^3, dynamic={n_dyn}, allowed={n_allowed}")

    def is_inside(self, points):
        """(N,3) -> (N,) bool: True if in dynamic voxel region."""
        norm = (points - self.grid_min) / (self.grid_max - self.grid_min)
        idx = (norm * self.R).long().clamp(0, self.R - 1)
        in_grid = (norm >= 0).all(dim=1) & (norm <= 1).all(dim=1)
        in_dynamic = self.allowed[idx[:, 0], idx[:, 1], idx[:, 2]]
        return in_grid & in_dynamic

    def is_fully_inside(self, means, scales):
        """(N,3), (N,3) -> (N,) bool: True if the entire gaussian extent
        (mean ± scale in each axis) is inside the ROI.
        Even if part sticks out, we exclude it."""
        s = torch.exp(scales)  # log-scale -> world-scale
        # Check all 8 corners of the bounding box: mean ± scale per axis
        # Sufficient to check the two extreme corners
        corner_min = means - s
        corner_max = means + s
        inside_min = self.is_inside(corner_min)
        inside_max = self.is_inside(corner_max)
        return inside_min & inside_max


class ROIAwareMCMCStrategy(MCMCStrategy):
    """MCMC strategy with ROI-aware densification.

    - Gaussians outside ROI (including extent) are treated as dead → relocated
    - New gaussians are sampled only from ROI-interior parents
    - Tiny dot gaussians and low-contribution ones are killed
    """

    def __init__(self, region_mask, prune_small_scale=0.0,
                 prune_contribution=0.0, **kwargs):
        super().__init__(**kwargs)
        self.region_mask = region_mask
        self.prune_small_scale = prune_small_scale
        self.prune_contribution = prune_contribution

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms):
        """Override: dead = low_opacity OR outside_ROI (extent) OR tiny dots."""
        opacities = torch.sigmoid(params["opacities"].flatten())
        # Extent-based ROI: mean ± scale must be fully inside
        outside_roi = ~self.region_mask.is_fully_inside(
            params["means"], params["scales"])
        dead_mask = (opacities <= self.min_opacity) | outside_roi
        # Kill tiny dot gaussians
        if self.prune_small_scale > 0:
            max_scale = torch.exp(params["scales"]).max(dim=-1).values
            dead_mask = dead_mask | (max_scale < self.prune_small_scale)
        # Kill low-contribution: opacity * max_scale too low
        if self.prune_contribution > 0:
            max_scale = torch.exp(params["scales"]).max(dim=-1).values
            dead_mask = dead_mask | (opacities * max_scale < self.prune_contribution)
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params, optimizers=optimizers, state={},
                mask=dead_mask, binoms=binoms, min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(self, params, optimizers, binoms):
        """Override: sample new gaussians only from ROI-interior parents."""
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            inside = self.region_mask.is_inside(params["means"])
            if inside.any():
                orig_opacities = params["opacities"].data.clone()
                params["opacities"].data[~inside] = -100.0
                sample_add(
                    params=params, optimizers=optimizers, state={},
                    n=n_gs, binoms=binoms, min_opacity=self.min_opacity,
                )
                params["opacities"].data[:current_n_points] = orig_opacities
        return n_gs


class ROIAwareDefaultStrategy(DefaultStrategy):
    """DefaultStrategy that prunes out-of-ROI, tiny, and low-contribution Gaussians."""

    def __init__(self, region_mask, prune_small_scale=0.0,
                 prune_contribution=0.0, **kwargs):
        super().__init__(**kwargs)
        self.region_mask = region_mask
        self.prune_small_scale = prune_small_scale
        self.prune_contribution = prune_contribution

    @torch.no_grad()
    def _prune_gs(self, params, optimizers, state, step):
        """Override: mark out-of-ROI, too-small, and low-contribution gaussians for pruning."""
        # Kill out-of-ROI gaussians (extent-based: mean ± scale must be fully inside)
        outside_roi = ~self.region_mask.is_fully_inside(
            params["means"], params["scales"])
        if outside_roi.any():
            params["opacities"].data[outside_roi] = -100.0

        scales = torch.exp(params["scales"])
        max_scale = scales.max(dim=-1).values
        opa = torch.sigmoid(params["opacities"].flatten())

        # Kill tiny dot gaussians (max scale below threshold)
        if self.prune_small_scale > 0:
            too_small = max_scale < self.prune_small_scale
            if too_small.any():
                params["opacities"].data[too_small] = -100.0

        # Kill low-contribution gaussians: opacity * max_scale too low
        # These are the noisy floaters — somewhat transparent AND small
        if self.prune_contribution > 0:
            contribution = opa * max_scale
            low_contrib = contribution < self.prune_contribution
            if low_contrib.any():
                params["opacities"].data[low_contrib] = -100.0

        return super()._prune_gs(params, optimizers, state, step)


@torch.no_grad()
def adaptive_split_large(splats, optimizers, n_target, region_mask=None):
    """Split the largest-scale Gaussians to replenish the population.
    Only splits visible (opacity > 0.1), inside-ROI Gaussians.
    Returns number of splits performed."""
    n_current = len(splats["means"])
    n_need = n_target - n_current
    if n_need <= 0:
        return 0

    n_to_split = min(n_need, n_current // 2)
    if n_to_split <= 0:
        return 0

    scales = torch.exp(splats["scales"])
    max_scale = scales.max(dim=-1).values.clone()

    # Only split visible gaussians
    opa = torch.sigmoid(splats["opacities"].flatten())
    max_scale[opa < 0.1] = 0.0

    # Only split inside ROI
    if region_mask is not None:
        inside = region_mask.is_inside(splats["means"])
        max_scale[~inside] = 0.0

    _, top_idx = torch.topk(max_scale, k=min(n_to_split, (max_scale > 0).sum().item()))
    mask = torch.zeros(n_current, dtype=torch.bool, device=splats["means"].device)
    mask[top_idx] = True

    split(params=splats, optimizers=optimizers, state={}, mask=mask)
    return mask.sum().item()


def call_step_post_backward(strategy, params, optimizers, state, step, info, lr):
    """Dispatch step_post_backward with correct kwargs for strategy type."""
    if isinstance(strategy, MCMCStrategy):
        strategy.step_post_backward(
            params=params, optimizers=optimizers, state=state,
            step=step, info=info, lr=lr,
        )
    else:
        strategy.step_post_backward(
            params=params, optimizers=optimizers, state=state,
            step=step, info=info, packed=True,
        )


def auto_data_factor(data_dir, target_min_dim=100):
    """Read one image from the dataset and compute data_factor so smallest dim ~ target_min_dim."""
    from PIL import Image as PILImage
    image_dir = os.path.join(data_dir, "images")
    # Find the first image file
    for cam_dir in sorted(os.listdir(image_dir)):
        cam_path = os.path.join(image_dir, cam_dir)
        if not os.path.isdir(cam_path):
            continue
        for fname in sorted(os.listdir(cam_path)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img = PILImage.open(os.path.join(cam_path, fname))
                w, h = img.size
                factor = max(1, round(min(w, h) / target_min_dim))
                new_w, new_h = round(w / factor), round(h / factor)
                print(f"[AutoFactor] Original: {w}x{h} → factor={factor} → {new_w}x{new_h} "
                      f"(min dim {min(new_w, new_h)}, target ~{target_min_dim})")
                return factor
    raise RuntimeError(f"No images found in {image_dir}")


class FrameImageLoader:
    """Threaded prefetcher: loads upcoming frames in background while current frame trains."""

    def __init__(self, data_dir, factor, cam_dirs, frame_fmt, ext):
        self.image_dir = os.path.join(data_dir, "images")
        self.factor = factor
        self.cam_dirs = cam_dirs
        self.frame_fmt = frame_fmt
        self.ext = ext
        self._cache = {}  # frame_num -> dict[cam_dir -> tensor]
        self._lock = __import__('threading').Lock()
        self._prefetch_thread = None

    def _load_single_frame(self, frame_num):
        """Load and resize all camera images for a single frame."""
        from PIL import Image as PILImage
        frame_str = f"{frame_num:{self.frame_fmt}}"
        images = {}
        for cam_dir in self.cam_dirs:
            path = os.path.join(self.image_dir, cam_dir, frame_str + self.ext)
            img = imageio.imread(path)[..., :3]
            if self.factor > 1:
                h, w = img.shape[:2]
                new_w, new_h = round(w / self.factor), round(h / self.factor)
                img = np.array(PILImage.fromarray(img).resize((new_w, new_h), PILImage.BICUBIC))
            images[cam_dir] = torch.from_numpy(img.astype(np.float32) / 255.0)
        return images

    def prefetch(self, frame_num):
        """Start loading a frame in the background."""
        if frame_num in self._cache:
            return
        def _load():
            imgs = self._load_single_frame(frame_num)
            with self._lock:
                self._cache[frame_num] = imgs
        import threading
        self._prefetch_thread = threading.Thread(target=_load, daemon=True)
        self._prefetch_thread.start()

    def get(self, frame_num):
        """Get a frame's images, loading if not cached. Blocks until ready."""
        if frame_num not in self._cache:
            if self._prefetch_thread and self._prefetch_thread.is_alive():
                self._prefetch_thread.join()
            if frame_num not in self._cache:
                self._cache[frame_num] = self._load_single_frame(frame_num)
        return self._cache[frame_num]

    def evict(self, frame_num):
        """Free memory for a frame we're done with."""
        with self._lock:
            self._cache.pop(frame_num, None)


def load_static_and_cache(static_ply_path, parser, trainset, device, sh_degree, cfg_antialiased):
    """Load static PLY, pre-activate, render cache for all training views."""
    print(f"[Static] Loading {static_ply_path}")
    s_means, s_scales, s_quats, s_opacs, s_sh0, s_shN = load_ply_gaussian(static_ply_path, device="cpu")
    s_colors = torch.cat([s_sh0, s_shN], dim=1)
    # Pad SH to target
    target_K = (sh_degree + 1) ** 2
    if s_colors.shape[1] < target_K:
        pad = torch.zeros(len(s_means), target_K - s_colors.shape[1], 3)
        s_colors = torch.cat([s_colors, pad], dim=1)
    elif s_colors.shape[1] > target_K:
        s_colors = s_colors[:, :target_K, :]

    static = {
        "means": s_means.to(device),
        "scales": torch.exp(s_scales).to(device),
        "quats": s_quats.to(device),
        "opacities": torch.sigmoid(s_opacs).to(device),
        "colors": s_colors.to(device),
    }
    print(f"[Static] {len(s_means):,} Gaussians loaded")

    # Get training camera params
    all_c2w, all_Ks = [], []
    for idx in trainset.indices:
        all_c2w.append(parser.camtoworlds[idx])
        cam_id = parser.camera_ids[idx]
        all_Ks.append(parser.Ks_dict[cam_id].copy())
    all_c2w = torch.from_numpy(np.stack(all_c2w)).float().to(device)
    all_Ks = torch.from_numpy(np.stack(all_Ks)).float().to(device)

    # Get resolution from first image
    w, h = parser.imsize_dict[parser.camera_ids[trainset.indices[0]]]

    # Render static cache
    print(f"[Static] Pre-rendering cache for {len(trainset)} views at {w}x{h}...")
    with torch.no_grad():
        static_colors, static_alphas, _ = rasterization(
            means=static["means"], quats=static["quats"],
            scales=static["scales"], opacities=static["opacities"],
            colors=static["colors"],
            viewmats=torch.linalg.inv(all_c2w), Ks=all_Ks,
            width=w, height=h, sh_degree=sh_degree,
            packed=True,
            rasterize_mode="antialiased" if cfg_antialiased else "classic",
        )
    cache = static_colors.clamp(0, 1).detach()
    mem_mb = cache.nelement() * 4 / 1024**2
    print(f"[Static] Cache ready: {cache.shape} ({mem_mb:.1f} MB)")
    return static, cache


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--static_ply_path", required=True)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int, default=5)
    parser.add_argument("--data_factor", type=int, default=0, help="0 = auto-compute from target_min_dim")
    parser.add_argument("--target_min_dim", type=int, default=100, help="Target smallest image dimension (used when data_factor=0)")
    parser.add_argument("--gpu", type=int, default=0)
    # Frame 1 settings
    parser.add_argument("--frame1_steps", type=int, default=3000)
    parser.add_argument("--frame1_cap", type=int, default=10000)
    # Fine-tune settings
    parser.add_argument("--ftune_steps", type=int, default=1500)
    parser.add_argument("--ftune_cap", type=int, default=10000)
    # Init from PLY (e.g. inside.ply from separation)
    parser.add_argument("--init_ply", default=None,
                        help="PLY file to init frame 1 from (e.g. inside.ply). Skips SFM init.")
    # ROI-constrained training
    parser.add_argument("--separation_dir", default=None,
                        help="Path to static_dynamic output dir (has voxel_labels.npy, grid_bounds.npy). "
                             "If set, dynamic gaussians are constrained to the dynamic voxel region.")
    parser.add_argument("--roi_padding", type=float, default=0.05,
                        help="Padding around dynamic voxels (in world units) via dilation")
    parser.add_argument("--noise_lr", type=float, default=1e4,
                        help="MCMC noise injection LR for frame 1 only (frames 2+ always use 0)")
    parser.add_argument("--strategy", choices=["mcmc", "default"], default="mcmc",
                        help="Fine-tune strategy: 'mcmc' (disabled, pure gradient) or 'default' (split/duplicate/prune)")
    parser.add_argument("--scale_reg", type=float, default=0.0001,
                        help="Scale regularization weight")
    parser.add_argument("--opacity_reg", type=float, default=0.001,
                        help="Opacity regularization weight")
    parser.add_argument("--needle_reg", type=float, default=0.001,
                        help="Needle regularization: penalizes max_scale/min_scale ratio")
    parser.add_argument("--small_scale_reg", type=float, default=0.01,
                        help="Penalize gaussians smaller than --min_scale (pushes dots to grow or die)")
    parser.add_argument("--min_scale", type=float, default=0.002,
                        help="Minimum gaussian scale; below this, small_scale_reg kicks in")
    parser.add_argument("--prune_small_scale", type=float, default=0.0005,
                        help="Hard-prune gaussians with max scale below this (0=disable)")
    parser.add_argument("--prune_opa", type=float, default=0.005,
                        help="Prune gaussians with opacity below this (DefaultStrategy)")
    parser.add_argument("--prune_contribution", type=float, default=0.0,
                        help="Prune gaussians where opacity*max_scale < this (0=disable)")
    parser.add_argument("--opacity_lr", type=float, default=5e-2,
                        help="Learning rate for opacities (lower = slower recovery after reset)")
    parser.add_argument("--reset_opacity_between_frames", type=float, default=0.0,
                        help="Reset opacities to this value at start of each frame (0=disable, e.g. 0.05)")
    # Adaptive densification & intermittent pruning
    parser.add_argument("--prune_every", type=int, default=5,
                        help="Only prune every N frames (1=every frame)")
    parser.add_argument("--densify_min_ratio", type=float, default=0.8,
                        help="Densify when GS count < this ratio of previous prune frame count")
    parser.add_argument("--densify_burn_in", type=int, default=100,
                        help="Training steps after densification to let split children learn")
    # Drift correction
    parser.add_argument("--drift_threshold", type=float, default=1.5,
                        help="Correct frame if loss > baseline * this (0=disable)")
    parser.add_argument("--correction_steps", type=int, default=1500,
                        help="Extra optimization steps when drift detected")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"

    # Auto-compute data_factor if not specified
    if args.data_factor <= 0:
        args.data_factor = auto_data_factor(args.data_dir, args.target_min_dim)

    frame_range = list(range(args.frame_start, args.frame_end + 1))
    os.makedirs(args.result_dir, exist_ok=True)

    # Load dynamic region mask if separation_dir provided
    region_mask = None
    if args.separation_dir:
        voxel_labels = np.load(os.path.join(args.separation_dir, "voxel_labels.npy"))
        grid_bounds = np.load(os.path.join(args.separation_dir, "grid_bounds.npy"))
        region_mask = DynamicRegionMask(voxel_labels, grid_bounds, device=device)

    # =========================================================
    # Phase 1: Setup parser (for camera params, SFM points)
    # =========================================================
    from datasets.colmap import Parser as ColmapParser, Dataset
    colmap_parser = ColmapParser(
        data_dir=args.data_dir, factor=args.data_factor, normalize=False,
        test_every=100000, frame_num=frame_range[0],
        load_images_in_memory=True, skip_points3d=False,
    )
    trainset = Dataset(colmap_parser, split="train")
    valset = Dataset(colmap_parser, split="val")

    # Uniform resolution
    sizes = set(colmap_parser.imsize_dict.values())
    if len(sizes) > 1:
        max_w = max(w for w, h in sizes)
        max_h = max(h for w, h in sizes)
        print(f"[Setup] Resizing {len(sizes)} resolutions to {max_w}x{max_h}")
        for cam_id in colmap_parser.imsize_dict:
            old_w, old_h = colmap_parser.imsize_dict[cam_id]
            if (old_w, old_h) != (max_w, max_h):
                sx, sy = max_w / old_w, max_h / old_h
                K = colmap_parser.Ks_dict[cam_id]
                K[0, :] *= sx
                K[1, :] *= sy
                colmap_parser.imsize_dict[cam_id] = (max_w, max_h)
        target_w, target_h = max_w, max_h
    else:
        target_w, target_h = next(iter(sizes))

    # Setup threaded frame image loader (loads per-frame on demand, prefetches next)
    cam_dirs_list = [os.path.dirname(n) for n in colmap_parser.image_names]
    # Detect frame format
    import re
    sample_cam = cam_dirs_list[0]
    image_dir = os.path.join(args.data_dir, "images")
    fnames = sorted(f for f in os.listdir(os.path.join(image_dir, sample_cam)) if re.match(r'\d+\.', f))
    stem, ext = os.path.splitext(fnames[0])
    frame_fmt = f"0{len(stem)}d"
    frame_loader = FrameImageLoader(args.data_dir, args.data_factor, cam_dirs_list, frame_fmt, ext)
    print(f"[Loader] On-demand loading: {len(cam_dirs_list)} cameras, factor={args.data_factor}")

    # =========================================================
    # Phase 3: Load static PLY + build cache (one-time)
    # =========================================================
    static_activated, static_cache = load_static_and_cache(
        args.static_ply_path, colmap_parser, trainset, device,
        sh_degree=3, cfg_antialiased=True,
    )

    # =========================================================
    # Phase 4: Pre-collate camera params on GPU (one-time)
    # =========================================================
    train_indices = trainset.indices
    cam_Ks = []
    cam_c2ws = []
    cam_dirs_order = []
    for idx in train_indices:
        cam_id = colmap_parser.camera_ids[idx]
        cam_Ks.append(colmap_parser.Ks_dict[cam_id].copy())
        cam_c2ws.append(colmap_parser.camtoworlds[idx])
        cam_dirs_order.append(os.path.dirname(colmap_parser.image_names[idx]))
    cam_Ks = torch.from_numpy(np.stack(cam_Ks)).float().to(device)
    cam_c2ws = torch.from_numpy(np.stack(cam_c2ws)).float().to(device)
    image_ids = torch.arange(len(train_indices), device=device)

    # =========================================================
    # Phase 5: Train each frame
    # =========================================================
    from gsplat import export_splats
    from gsplat.optimizers import SelectiveAdam
    from fused_ssim import fused_ssim
    from utils import knn, rgb_to_sh, set_random_seed

    scene_scale = colmap_parser.scene_scale * 1.1
    prev_ply = None
    results = []
    pipeline_start = time.time()
    needle_threshold = float('inf')  # set from frame 2
    needle_threshold_base = float('inf')
    last_prune_gs_count = None  # GS count after last prune frame

    for fi, frame_num in enumerate(frame_range):
        set_random_seed(42)
        frame_start = time.time()
        is_first = (fi == 0)
        max_steps = args.frame1_steps if is_first else args.ftune_steps
        cap_max = args.frame1_cap if is_first else args.ftune_cap

        frame_dir = os.path.join(args.result_dir, f"frame_{frame_num:03d}")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(os.path.join(frame_dir, "ply"), exist_ok=True)

        # Load this frame's images (prefetch next frame in background)
        frame_images = frame_loader.get(frame_num)
        next_frame = frame_range[fi + 1] if fi + 1 < len(frame_range) else None
        if next_frame is not None:
            frame_loader.prefetch(next_frame)

        pixels_list = []
        for cam_dir in cam_dirs_order:
            img = frame_images[cam_dir]
            h, w = img.shape[:2]
            if (w, h) != (target_w, target_h):
                img = F.interpolate(
                    img.permute(2, 0, 1).unsqueeze(0),
                    size=(target_h, target_w), mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
            pixels_list.append(img.to(device))
        pixels_all = torch.stack(pixels_list)  # [C, H, W, 3]

        # Free previous frame's images
        if fi > 0:
            frame_loader.evict(frame_range[fi - 1])

        # Initialize Gaussians
        if is_first:
            # Determine init PLY: explicit --init_ply, or inside.ply from separation_dir
            init_ply = args.init_ply
            if init_ply is None and args.separation_dir:
                init_ply = os.path.join(args.separation_dir, "inside.ply")

            if init_ply:
                print(f"\n[Frame {frame_num}] Init from PLY: {init_ply}")
                i_means, i_scales, i_quats, i_opacs, i_sh0, i_shN = load_ply_gaussian(init_ply, device="cpu")
                # Pad/trim SH to 16 coefficients
                i_colors = torch.cat([i_sh0, i_shN], dim=1)
                target_K = 16
                if i_colors.shape[1] < target_K:
                    pad = torch.zeros(len(i_means), target_K - i_colors.shape[1], 3)
                    i_colors = torch.cat([i_colors, pad], dim=1)
                elif i_colors.shape[1] > target_K:
                    i_colors = i_colors[:, :target_K, :]

                splats = torch.nn.ParameterDict({
                    "means": torch.nn.Parameter(i_means),
                    "scales": torch.nn.Parameter(i_scales),
                    "quats": torch.nn.Parameter(i_quats),
                    "opacities": torch.nn.Parameter(i_opacs),
                    "sh0": torch.nn.Parameter(i_colors[:, :1, :]),
                    "shN": torch.nn.Parameter(i_colors[:, 1:, :]),
                }).to(device)
                print(f"[Frame {frame_num}] Loaded {len(i_means)} dynamic GS, {max_steps} steps, cap={cap_max}")
            else:
                # From SFM points — fresh init
                points = torch.from_numpy(colmap_parser.points).float()
                rgbs = torch.from_numpy(colmap_parser.points_rgb / 255.0).float()
                n_pts = 5000
                if len(points) > n_pts:
                    idx = torch.randperm(len(points))[:n_pts]
                    points = points[idx]
                    rgbs = rgbs[idx]
                print(f"\n[Frame {frame_num}] From scratch: {len(points)} SFM seeds, {max_steps} steps, cap={cap_max}")

                dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
                dist_avg = torch.sqrt(dist2_avg)
                init_scales = torch.log(dist_avg * 1.0).unsqueeze(-1).repeat(1, 3)
                N = len(points)
                init_quats = torch.rand((N, 4))
                init_opacities = torch.logit(torch.full((N,), 0.5))
                init_colors = torch.zeros((N, 16, 3))
                init_colors[:, 0, :] = rgb_to_sh(rgbs)

                splats = torch.nn.ParameterDict({
                    "means": torch.nn.Parameter(points),
                    "scales": torch.nn.Parameter(init_scales),
                    "quats": torch.nn.Parameter(init_quats),
                    "opacities": torch.nn.Parameter(init_opacities),
                    "sh0": torch.nn.Parameter(init_colors[:, :1, :]),
                    "shN": torch.nn.Parameter(init_colors[:, 1:, :]),
                }).to(device)
        else:
            # Reuse previous frame's splats in-place (no save/reload!)
            # Just detach and re-wrap as fresh parameters to reset optimizer state
            print(f"\n[Frame {frame_num}] Fine-tune from frame {frame_range[fi-1]}: "
                  f"{len(splats['means'])} GS in memory, {max_steps} steps")
            new_splats = torch.nn.ParameterDict({
                name: torch.nn.Parameter(splats[name].detach().clone())
                for name in splats
            }).to(device)
            # Reset opacities between frames: keep geometry, force opacity low
            # so only gaussians useful for the new frame recover during fine-tuning
            if args.reset_opacity_between_frames > 0:
                reset_val = args.reset_opacity_between_frames
                new_splats["opacities"].data.clamp_(
                    max=torch.logit(torch.tensor(reset_val)).item())
                print(f"  [FRAME RESET] Opacities clamped to {reset_val:.3f} for new frame")
            splats = new_splats

        # Optimizers (capped BS scaling) — always fresh per frame
        BS_scale = min(len(train_indices), 10)
        beta1 = 1 - BS_scale * (1 - 0.9)
        beta2 = 1 - BS_scale * (1 - 0.999)
        lr_params = [
            ("means", 1.6e-4 * scene_scale), ("scales", 5e-3), ("quats", 1e-3),
            ("opacities", 3e-2), ("sh0", 2.5e-3), ("shN", 2.5e-3 / 20),
        ]
        optimizers = {}
        for name, lr in lr_params:
            optimizers[name] = torch.optim.Adam(
                [{"params": splats[name], "lr": lr * math.sqrt(BS_scale), "name": name}],
                eps=1e-15 / math.sqrt(BS_scale), betas=(beta1, beta2),
            )

        # Strategy — all frames: pure fine-tuning (no MCMC ops) unless --strategy default
        if args.strategy == "default" and not is_first:
            # Frames 2+: gradient-based split/duplicate/prune
            default_kwargs = dict(
                refine_every=300, refine_start_iter=50,
                refine_stop_iter=int(max_steps * 0.8),
                reset_every=999999, prune_opa=args.prune_opa,
                grow_grad2d=0.0002, grow_scale3d=0.01,
                verbose=True,
            )
            if region_mask is not None:
                strategy = ROIAwareDefaultStrategy(
                    region_mask=region_mask,
                    prune_small_scale=args.prune_small_scale,
                    prune_contribution=args.prune_contribution,
                    **default_kwargs,
                )
            else:
                strategy = DefaultStrategy(**default_kwargs)
            strategy.check_sanity(splats, optimizers)
            strategy_state = strategy.initialize_state(scene_scale=scene_scale)
        else:
            # Frames 2+ mcmc: disable all MCMC ops, pure gradient fine-tuning
            mcmc_kwargs = dict(
                cap_max=len(splats["means"]),
                refine_every=999999, verbose=False,
                refine_start_iter=999999, refine_stop_iter=0,
                noise_lr=0.0, min_opacity=0.005,
            )
            if region_mask is not None:
                strategy = ROIAwareMCMCStrategy(
                    region_mask=region_mask,
                    prune_small_scale=args.prune_small_scale,
                    prune_contribution=args.prune_contribution,
                    **mcmc_kwargs,
                )
            else:
                strategy = MCMCStrategy(**mcmc_kwargs)
            strategy.check_sanity(splats, optimizers)
            strategy_state = strategy.initialize_state()

        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        )

        # Training loop
        train_start = time.time()
        for step in range(max_steps):
            # Forward: render dynamic only
            means = splats["means"]
            quats_r = splats["quats"]
            scales_r = torch.exp(splats["scales"])
            opacities_r = torch.sigmoid(splats["opacities"])
            colors_r = torch.cat([splats["sh0"], splats["shN"]], 1)

            sh_degree = min(step, 3)
            renders, alphas, info = rasterization(
                means=means, quats=quats_r, scales=scales_r,
                opacities=opacities_r, colors=colors_r,
                viewmats=torch.linalg.inv(cam_c2ws), Ks=cam_Ks,
                width=target_w, height=target_h,
                sh_degree=sh_degree, near_plane=0.01, far_plane=1e10,
                rasterize_mode="antialiased",
            )

            # Composite static background
            dyn_colors = renders[..., :3]
            final_colors = dyn_colors + static_cache[image_ids] * (1.0 - alphas)

            # Strategy pre-backward
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
            )

            # Loss
            l1loss = F.l1_loss(final_colors, pixels_all)
            loss = l1loss
            # Regularization
            if args.opacity_reg > 0:
                loss = loss + args.opacity_reg * torch.abs(torch.sigmoid(splats["opacities"])).mean()
            if args.scale_reg > 0:
                loss = loss + args.scale_reg * torch.abs(torch.exp(splats["scales"])).mean()
            # Needle regularization: only penalize aspect ratios ABOVE 10
            if args.needle_reg > 0:
                s = torch.exp(splats["scales"])
                aspect = s.max(dim=1).values / s.min(dim=1).values.clamp(min=1e-6)
                excess = torch.clamp(aspect - 10.0, min=0.0)
                loss = loss + args.needle_reg * excess.mean()
            # Bright spot regularization: only penalize very high opacity (> 0.9)
            if args.opacity_reg > 0:
                opa = torch.sigmoid(splats["opacities"])
                bright_excess = torch.clamp(opa - 0.9, min=0.0)
                loss = loss + args.opacity_reg * bright_excess.mean()

            loss.backward()

            # Optimize
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            scheduler.step()

            # Strategy post-backward
            call_step_post_backward(
                strategy, splats, optimizers, strategy_state, step, info,
                lr=scheduler.get_last_lr()[0],
            )


        train_time = time.time() - train_start
        final_loss = loss.item()

        # Set baseline on first frame
        if is_first:
            baseline_loss = final_loss
            print(f"[Loss] Baseline set: {baseline_loss:.4f}")

        # Drift correction: if loss deviated too much, train extra steps to fix it
        if args.drift_threshold > 0 and not is_first and final_loss > baseline_loss * args.drift_threshold:
            print(f"[DRIFT] Frame {frame_num}: loss={final_loss:.4f} > "
                  f"threshold={baseline_loss * args.drift_threshold:.4f} "
                  f"({final_loss/baseline_loss:.2f}x baseline). Correcting...")
            corr_start = time.time()
            for step in range(args.correction_steps):
                means = splats["means"]
                quats_r = splats["quats"]
                scales_r = torch.exp(splats["scales"])
                opacities_r = torch.sigmoid(splats["opacities"])
                colors_r = torch.cat([splats["sh0"], splats["shN"]], 1)
                sh_degree = 3
                renders, alphas, info = rasterization(
                    means=means, quats=quats_r, scales=scales_r,
                    opacities=opacities_r, colors=colors_r,
                    viewmats=torch.linalg.inv(cam_c2ws), Ks=cam_Ks,
                    width=target_w, height=target_h,
                    sh_degree=sh_degree, near_plane=0.01, far_plane=1e10,
                    rasterize_mode="antialiased",
                )
                dyn_colors = renders[..., :3]
                final_colors = dyn_colors + static_cache[image_ids] * (1.0 - alphas)
                # retain_grad needed for DefaultStrategy (no-op for MCMC)
                strategy.step_pre_backward(
                    params=splats, optimizers=optimizers,
                    state=strategy_state, step=step, info=info,
                )
                l1loss = F.l1_loss(final_colors, pixels_all)
                loss = l1loss
                if args.opacity_reg > 0:
                    loss = loss + args.opacity_reg * torch.abs(torch.sigmoid(splats["opacities"])).mean()
                if args.scale_reg > 0:
                    loss = loss + args.scale_reg * torch.abs(torch.exp(splats["scales"])).mean()
                loss.backward()
                for opt in optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            corr_time = time.time() - corr_start
            train_time += corr_time
            print(f"[DRIFT] Corrected: {final_loss:.4f} → {loss.item():.4f} (+{corr_time:.1f}s)")
            final_loss = loss.item()

        n_gs = len(splats["means"])

        # Post-training cleanup: intermittent (every N frames) + adaptive densification
        # Bbox filter always runs (gaussians must stay in ROI).
        # Opacity + needle pruning only every prune_every frames.
        is_prune_frame = is_first or (fi % args.prune_every == 0)

        with torch.no_grad():
            keep = torch.ones(n_gs, dtype=torch.bool, device=device)
            n_bbox = 0
            n_opa = 0
            n_needle = 0

            # Opacity + needle pruning — prune hard early, relax over time
            # Progress: 0.0 at start, 1.0 at end
            progress = fi / max(len(frame_range) - 1, 1)
            # Opacity threshold: 0.005 early → 0.0005 late
            opa_threshold = 0.005 * (1.0 - 0.9 * progress)

            if is_prune_frame:
                opa = torch.sigmoid(splats["opacities"].flatten())
                low_opa = opa < opa_threshold
                n_opa = low_opa.sum().item()
                keep &= ~low_opa

                if not is_first:
                    s = torch.exp(splats["scales"])
                    aspect = s.max(dim=1).values / s.min(dim=1).values.clamp(min=1e-6)
                    if fi == 1:
                        sorted_aspect, _ = torch.sort(aspect, descending=True)
                        top1_idx = max(1, int(0.01 * len(aspect)))
                        top1_val = sorted_aspect[top1_idx - 1].item()
                        needle_threshold_base = max(top1_val, 100.0)
                        print(f"  [NEEDLE] Base threshold from frame 2: {needle_threshold_base:.1f}")
                    # Relax needle threshold over time: base early → base*3 late
                    effective_needle = needle_threshold_base * (1.0 + 2.0 * progress)
                    needles = aspect > effective_needle
                    n_needle = needles.sum().item()
                    keep &= ~needles

            if not keep.all():
                for k in splats:
                    splats[k] = torch.nn.Parameter(
                        splats[k].data[keep])
                n_gs = len(splats["means"])
                print(f"  [SAVE] Removed {(~keep).sum().item()} GS "
                      f"(low opa: {n_opa}, needles: {n_needle})")

            # Track initial GS count (frame 1 after cleanup)
            if is_first:
                last_prune_gs_count = n_gs

        # Final bbox check — always, right before save
        if region_mask is not None:
            with torch.no_grad():
                s = torch.exp(splats["scales"])
                lo = splats["means"] - s
                hi = splats["means"] + s
                inside = (lo >= region_mask.grid_min).all(dim=1) & \
                         (hi <= region_mask.grid_max).all(dim=1)
                if not inside.all():
                    n_out = (~inside).sum().item()
                    for k in splats:
                        splats[k] = torch.nn.Parameter(
                            splats[k].data[inside])
                    n_gs = len(splats["means"])

        # Save dynamic PLY
        ply_path = os.path.join(frame_dir, "ply", f"point_cloud_{max_steps-1}.ply")
        with torch.no_grad():
            export_splats(
                means=splats["means"], scales=splats["scales"],
                quats=splats["quats"], opacities=splats["opacities"],
                sh0=splats["sh0"], shN=splats["shN"],
                format="ply", save_to=ply_path,
            )
            # Combined PLY
            sa = static_activated
            s_scales_log = torch.log(sa["scales"])
            s_opacities_logit = torch.logit(sa["opacities"].clamp(1e-6, 1 - 1e-6))
            s_sh0 = sa["colors"][:, :1, :]
            s_shN_raw = sa["colors"][:, 1:, :]
            dyn_shN_K = splats["shN"].shape[1]
            if s_shN_raw.shape[1] < dyn_shN_K:
                pad = torch.zeros(len(sa["means"]), dyn_shN_K - s_shN_raw.shape[1], 3, device=device)
                s_shN_raw = torch.cat([s_shN_raw, pad], dim=1)
            elif s_shN_raw.shape[1] > dyn_shN_K:
                s_shN_raw = s_shN_raw[:, :dyn_shN_K, :]
            combined_path = os.path.join(frame_dir, "ply", f"point_cloud_combined_{max_steps-1}.ply")
            export_splats(
                means=torch.cat([splats["means"], sa["means"]]),
                scales=torch.cat([splats["scales"], s_scales_log]),
                quats=torch.cat([splats["quats"], sa["quats"]]),
                opacities=torch.cat([splats["opacities"], s_opacities_logit]),
                sh0=torch.cat([splats["sh0"], s_sh0]),
                shN=torch.cat([splats["shN"], s_shN_raw]),
                format="ply", save_to=combined_path,
            )

        # splats stays in GPU memory for next frame (no reload needed)
        frame_total = time.time() - frame_start
        print(f"[Frame {frame_num}] Train: {train_time:.1f}s | Total: {frame_total:.1f}s | "
              f"Loss: {loss.item():.4f} | #GS: {n_gs}")
        results.append({
            "frame": frame_num, "train_time": train_time,
            "total_time": frame_total, "loss": final_loss, "num_gs": n_gs,
        })


    pipeline_total = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {len(frame_range)} frames in {pipeline_total:.1f}s")
    print(f"  Avg per frame: {pipeline_total/len(frame_range):.1f}s")
    print(f"{'='*60}")
    print(f"\n| Frame | Train | Total | Loss | #GS |")
    print(f"|---|---|---|---|---|")
    for r in results:
        print(f"| {r['frame']} | {r['train_time']:.1f}s | {r['total_time']:.1f}s | {r['loss']:.4f} | {r['num_gs']} |")

    # Save summary
    with open(os.path.join(args.result_dir, "summary.json"), "w") as f:
        json.dump({"results": results, "total_time": pipeline_total}, f, indent=2)

    # Collect all dynamic PLYs into all_ply/ folder
    ply_out = os.path.join(args.result_dir, "all_ply")
    os.makedirs(ply_out, exist_ok=True)
    collected = 0
    for fi, frame_num in enumerate(frame_range):
        frame_ply_dir = os.path.join(args.result_dir, f"frame_{frame_num:03d}", "ply")
        if not os.path.isdir(frame_ply_dir):
            continue
        for f in sorted(os.listdir(frame_ply_dir)):
            if f.startswith("point_cloud_") and "combined" not in f:
                src = os.path.join(frame_ply_dir, f)
                dst = os.path.join(ply_out, f"{fi:04d}.ply")
                shutil.copy2(src, dst)
                collected += 1
                break
    print(f"\nCollected {collected} dynamic PLYs -> {ply_out}/")


if __name__ == "__main__":
    main()
