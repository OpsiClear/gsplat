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
import time

import imageio
import numpy as np
import torch
import torch.nn.functional as F

from gsplat.exporter import load_ply_gaussian
from gsplat.rendering import rasterization
from gsplat.strategy import MCMCStrategy
from gsplat.strategy.ops import relocate, sample_add, inject_noise_to_position


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


class ROIAwareMCMCStrategy(MCMCStrategy):
    """MCMC strategy that confines all densification to the dynamic ROI.

    - Gaussians outside ROI are treated as dead → relocated INTO the ROI
    - New gaussians are sampled only from ROI-interior parents
    - No post-hoc filtering needed, zero overhead
    """

    def __init__(self, region_mask, **kwargs):
        super().__init__(**kwargs)
        self.region_mask = region_mask

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms):
        """Override: dead = low_opacity OR outside_ROI."""
        opacities = torch.sigmoid(params["opacities"].flatten())
        outside_roi = ~self.region_mask.is_inside(params["means"])
        dead_mask = (opacities <= self.min_opacity) | outside_roi
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
            # Only sample from gaussians inside ROI
            inside = self.region_mask.is_inside(params["means"])
            if inside.any():
                # Temporarily zero out opacity of outside gaussians so they're not sampled
                orig_opacities = params["opacities"].data.clone()
                params["opacities"].data[~inside] = -100.0  # sigmoid(-100) ≈ 0
                sample_add(
                    params=params, optimizers=optimizers, state={},
                    n=n_gs, binoms=binoms, min_opacity=self.min_opacity,
                )
                # Restore original opacities (only for the old gaussians)
                params["opacities"].data[:current_n_points] = orig_opacities
        return n_gs


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


def preload_all_frames(data_dir, frame_range, factor, cache_file=None):
    """Pre-load and resize images for all frames. Returns dict[frame_num] -> dict[cam_dir] -> tensor."""
    if cache_file and os.path.exists(cache_file):
        print(f"[Preload] Loading all frames from cache: {cache_file}")
        return torch.load(cache_file, weights_only=True)

    from datasets.colmap import Parser
    # Use frame 1 to get camera list
    parser = Parser(
        data_dir=data_dir, factor=factor, normalize=False,
        test_every=100000, frame_num=frame_range[0],
        load_images_in_memory=False, skip_points3d=True,
    )
    cam_dirs = []
    for name in parser.image_names:
        cam_dirs.append(os.path.dirname(name))

    image_dir = os.path.join(data_dir, "images")
    # Detect frame format
    import re
    sample_cam = cam_dirs[0]
    fnames = sorted(f for f in os.listdir(os.path.join(image_dir, sample_cam)) if re.match(r'\d+\.', f))
    stem, ext = os.path.splitext(fnames[0])
    frame_fmt = f"0{len(stem)}d"

    all_frames = {}
    total_start = time.time()
    for frame_num in frame_range:
        frame_str = f"{frame_num:{frame_fmt}}"
        images = {}
        for cam_dir in cam_dirs:
            path = os.path.join(image_dir, cam_dir, frame_str + ext)
            img = imageio.imread(path)[..., :3]
            # Resize
            if factor > 1:
                h, w = img.shape[:2]
                new_w, new_h = round(w / factor), round(h / factor)
                from PIL import Image
                img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BICUBIC))
            images[cam_dir] = torch.from_numpy(img.astype(np.float32) / 255.0)
        all_frames[frame_num] = images
    elapsed = time.time() - total_start
    print(f"[Preload] Loaded {len(frame_range)} frames x {len(cam_dirs)} cameras in {elapsed:.1f}s")

    if cache_file:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(all_frames, cache_file)
        print(f"[Preload] Saved cache: {cache_file}")

    return all_frames


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
    # ROI-constrained training
    parser.add_argument("--separation_dir", default=None,
                        help="Path to static_dynamic output dir (has voxel_labels.npy, grid_bounds.npy). "
                             "If set, dynamic gaussians are constrained to the dynamic voxel region.")
    parser.add_argument("--roi_padding", type=float, default=0.05,
                        help="Padding around dynamic voxels (in world units) via dilation")
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
    # Phase 1: Pre-load ALL frames' images (one-time cost)
    # =========================================================
    cache_file = os.path.join(args.result_dir, f"all_frames_f{args.data_factor}.pt")
    all_frame_images = preload_all_frames(
        args.data_dir, frame_range, args.data_factor, cache_file=cache_file
    )

    # =========================================================
    # Phase 2: Setup parser (for camera params, SFM points)
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

    # Resize all frame images to uniform size
    for frame_num in all_frame_images:
        for cam_dir in all_frame_images[frame_num]:
            img = all_frame_images[frame_num][cam_dir]
            h, w = img.shape[:2]
            if (w, h) != (target_w, target_h):
                img = F.interpolate(
                    img.permute(2, 0, 1).unsqueeze(0),
                    size=(target_h, target_w), mode="bilinear", align_corners=False,
                ).squeeze(0).permute(1, 2, 0)
                all_frame_images[frame_num][cam_dir] = img

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
    from gsplat.strategy import MCMCStrategy
    from gsplat.optimizers import SelectiveAdam
    from fused_ssim import fused_ssim
    from utils import knn, rgb_to_sh, set_random_seed

    scene_scale = colmap_parser.scene_scale * 1.1
    prev_ply = None
    results = []
    pipeline_start = time.time()

    for fi, frame_num in enumerate(frame_range):
        set_random_seed(42)
        frame_start = time.time()
        is_first = (fi == 0)
        max_steps = args.frame1_steps if is_first else args.ftune_steps
        cap_max = args.frame1_cap if is_first else args.ftune_cap

        frame_dir = os.path.join(args.result_dir, f"frame_{frame_num:03d}")
        os.makedirs(frame_dir, exist_ok=True)
        os.makedirs(os.path.join(frame_dir, "ply"), exist_ok=True)

        # Swap training images for this frame
        frame_images = all_frame_images[frame_num]
        pixels_list = []
        for cam_dir in cam_dirs_order:
            img = frame_images[cam_dir].to(device)
            pixels_list.append(img)
        pixels_all = torch.stack(pixels_list)  # [C, H, W, 3]

        # Initialize Gaussians
        if is_first:
            # From SFM points — fresh init
            points = torch.from_numpy(colmap_parser.points).float()
            rgbs = torch.from_numpy(colmap_parser.points_rgb / 255.0).float()

            # Filter to dynamic region if ROI mask available
            if region_mask is not None:
                inside = region_mask.is_inside(points.to(device)).cpu()
                points = points[inside]
                rgbs = rgbs[inside]
                print(f"[ROI] Filtered SFM points to dynamic region: {inside.sum()}/{len(inside)}")

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
            splats = new_splats

        # Optimizers (capped BS scaling) — always fresh per frame
        BS_scale = min(len(train_indices), 10)
        beta1 = 1 - BS_scale * (1 - 0.9)
        beta2 = 1 - BS_scale * (1 - 0.999)
        lr_params = [
            ("means", 1.6e-4 * scene_scale), ("scales", 5e-3), ("quats", 1e-3),
            ("opacities", 5e-2), ("sh0", 2.5e-3), ("shN", 2.5e-3 / 20),
        ]
        optimizers = {}
        for name, lr in lr_params:
            optimizers[name] = torch.optim.Adam(
                [{"params": splats[name], "lr": lr * math.sqrt(BS_scale), "name": name}],
                eps=1e-15 / math.sqrt(BS_scale), betas=(beta1, beta2),
            )

        # Strategy
        strategy = MCMCStrategy(
            cap_max=cap_max, refine_every=100, verbose=True,
            refine_start_iter=100 if is_first else 50,
            refine_stop_iter=int(max_steps * 0.8),
            noise_lr=1e4, min_opacity=0.005,
        )
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
            loss = loss + 0.001 * torch.sigmoid(splats["opacities"]).mean()
            loss = loss + 0.0001 * torch.exp(splats["scales"]).mean()

            loss.backward()

            # Optimize
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            scheduler.step()

            # Strategy post-backward (MCMC may add/relocate here)
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
                lr=scheduler.get_last_lr()[0],
            )

            # Kill escaped gaussians: set opacity very low so MCMC relocates them
            # O(N) voxel lookup, no optimizer rebuild, ~0 overhead
            if region_mask is not None and step % strategy.refine_every == 0 and step > 0:
                n_killed = region_mask.kill_escaped(splats)
                if n_killed > 0 and strategy.verbose:
                    print(f"  [ROI] Step {step}: killed {n_killed} escaped gaussians (will be relocated)")

        train_time = time.time() - train_start
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
            "total_time": frame_total, "loss": loss.item(), "num_gs": n_gs,
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


if __name__ == "__main__":
    main()
