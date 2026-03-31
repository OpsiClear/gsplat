"""
Sequential per-frame 3DGS training with optical flow peak detection.

For a multi-camera rig dataset, trains 3DGS frame-by-frame:
  - Frame 0: full training from SFM init (20k steps, densification)
  - Non-peak frames: fine-tune from previous frame (5k steps, no densification)
  - Peak frames: fine-tune from previous frame (10k steps, MCMC relocation)

Gaussians are carried forward between frames to avoid flickering.

Usage:
    CUDA_VISIBLE_DEVICES=1 python simple_trainer_sequential.py \\
        --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \\
        --result_dir /data/shared/elaheh/4D_demo/elly_sequential_30f \\
        --flow_path /data/shared/elaheh/4D_demo/outdoor/elly/flow_results/077-002/flow_magnitudes.npy \\
        --mask_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted/masks \\
        --start_frame 0 --num_frames 30 --num_peaks 50
"""

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from datasets.colmap import Dataset, Parser
from fused_ssim import fused_ssim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Data
    data_dir: str = ""
    result_dir: str = ""
    data_factor: int = 1
    mask_dir: Optional[str] = None

    # Frame range
    start_frame: int = 0
    num_frames: int = 30

    # Optical flow peak detection
    flow_path: str = ""
    num_peaks: int = 50
    min_peak_separation: int = 10

    # Training steps per frame type
    first_frame_steps: int = 20_000
    normal_frame_steps: int = 5_000
    peak_frame_steps: int = 10_000

    # SH
    sh_degree: int = 3
    sh_degree_interval: int = 1000

    # Init
    init_type: str = "sfm"
    init_opa: float = 0.5
    init_scale: float = 0.1

    # Learning rates (frame 0)
    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # Fine-tune LR multipliers
    finetune_lr_scale: float = 0.1
    peak_lr_scale: float = 0.2

    # Loss
    ssim_lambda: float = 0.2
    opacity_reg: float = 0.0
    scale_reg: float = 0.01

    # Rasterization
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    antialiased: bool = False

    # Densification (frame 0) — MCMC with hard cap
    first_frame_cap: int = 1_000_000
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000

    # MCMC settings for fine-tune frames
    # Low noise — we're fine-tuning a good model, not exploring from scratch
    mcmc_noise_lr: float = 1e3
    mcmc_refine_every: int = 100
    # Peak frames: allow Gaussian count to grow by this fraction (e.g. 0.2 = +20%)
    peak_cap_growth: float = 0.2

    # Dataset
    test_every: int = 8
    batch_size: int = 1
    random_bkgd: bool = False
    global_scale: float = 1.0

    # Output
    save_ply: bool = True
    tb_every: int = 100
    eval_every_n_frames: int = 5

    # Misc
    port: int = 8080
    disable_viewer: bool = True


# ---------------------------------------------------------------------------
# Peak detection (reuse logic from optical_flow_segments.py)
# ---------------------------------------------------------------------------
def find_peaks(magnitudes: np.ndarray, num_peaks: int = 50,
               min_separation: int = 10) -> List[int]:
    """Find top-K peaks in magnitude signal with minimum separation."""
    sorted_indices = np.argsort(magnitudes)[::-1]
    peaks = []
    for idx in sorted_indices:
        if len(peaks) >= num_peaks:
            break
        too_close = any(abs(int(idx) - p) < min_separation for p in peaks)
        if not too_close:
            peaks.append(int(idx))
    peaks.sort()
    return peaks


def classify_frames(cfg: Config) -> Dict[int, str]:
    """Classify each frame as 'first', 'peak', or 'normal'."""
    start = cfg.start_frame
    end = start + cfg.num_frames  # exclusive

    classification = {}
    peak_set = set()

    if cfg.flow_path and os.path.exists(cfg.flow_path):
        magnitudes = np.load(cfg.flow_path)
        print(f"[Flow] Loaded magnitudes: shape={magnitudes.shape}, "
              f"min={magnitudes.min():.3f}, max={magnitudes.max():.3f}")
        # Peak at index i means motion between frame i and i+1
        # Treat frame i+1 as the peak frame
        all_peaks = find_peaks(magnitudes, num_peaks=cfg.num_peaks,
                               min_separation=cfg.min_peak_separation)
        peak_frames = set(p + 1 for p in all_peaks)  # shift to the frame that changed
        peak_set = peak_frames & set(range(start, end))
        print(f"[Flow] {len(all_peaks)} peaks over full range, "
              f"{len(peak_set)} within [{start}, {end})")
    else:
        print("[Flow] No flow path provided — all frames treated as normal.")

    for f in range(start, end):
        if f == start:
            classification[f] = "first"
        elif f in peak_set:
            classification[f] = "peak"
        else:
            classification[f] = "normal"

    return classification


# ---------------------------------------------------------------------------
# Splat creation helpers
# ---------------------------------------------------------------------------
def create_splats_from_sfm(
    parser: Parser,
    cfg: Config,
    scene_scale: float,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create splats + optimizers from SFM points (frame 0)."""
    points = torch.from_numpy(parser.points).float()
    rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

    N = points.shape[0]
    opacities = torch.logit(torch.full((N,), cfg.init_opa))
    quats = torch.rand((N, 4))
    sh0s = rgb_to_sh(rgbs)

    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = sh0s

    params = [
        ("means", torch.nn.Parameter(points), cfg.means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.shN_lr),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr, "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def create_splats_from_checkpoint(
    carried_state: Dict[str, Tensor],
    cfg: Config,
    scene_scale: float,
    lr_scale: float,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """Create splats + fresh optimizers from carried-forward state dict."""
    lr_map = {
        "means": cfg.means_lr * scene_scale * lr_scale,
        "scales": cfg.scales_lr * lr_scale,
        "quats": cfg.quats_lr * lr_scale,
        "opacities": cfg.opacities_lr * lr_scale,
        "sh0": cfg.sh0_lr * lr_scale,
        "shN": cfg.shN_lr * lr_scale,
    }

    splats = torch.nn.ParameterDict({
        name: torch.nn.Parameter(tensor.clone().to(device))
        for name, tensor in carried_state.items()
    })

    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr_map[name], "name": name}],
            eps=1e-15,
            betas=(0.9, 0.999),
        )
        for name in splats.keys()
    }
    return splats, optimizers


# ---------------------------------------------------------------------------
# Rasterize helper
# ---------------------------------------------------------------------------
def rasterize_splats(
    splats: torch.nn.ParameterDict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    cfg: Config,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    means = splats["means"]
    quats = splats["quats"]
    scales = torch.exp(splats["scales"])
    opacities = torch.sigmoid(splats["opacities"])
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)

    rasterize_mode = "antialiased" if cfg.antialiased else "classic"
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
        packed=cfg.packed,
        absgrad=True,
        rasterize_mode=rasterize_mode,
        **kwargs,
    )
    return render_colors, render_alphas, info


# ---------------------------------------------------------------------------
# Mask loading helper
# ---------------------------------------------------------------------------
def load_mask(mask_dir: str, cam_name: str, frame_num: int,
              frame_fmt: str, ext: str, factor: int = 1) -> Optional[Tensor]:
    """Load a binary mask for a specific camera and frame."""
    frame_str = f"{frame_num:{frame_fmt}}"
    mask_path = os.path.join(mask_dir, cam_name, frame_str + ext)
    if not os.path.exists(mask_path):
        return None
    import imageio.v3 as iio
    mask = iio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = (mask > 127).astype(np.float32)
    if factor > 1:
        from PIL import Image as PILImage
        h, w = mask.shape[:2]
        new_h, new_w = h // factor, w // factor
        mask = np.array(
            PILImage.fromarray((mask * 255).astype(np.uint8)).resize(
                (new_w, new_h), PILImage.NEAREST
            )
        ).astype(np.float32) / 255.0
    return torch.from_numpy(mask).float()


# ---------------------------------------------------------------------------
# Single-frame training
# ---------------------------------------------------------------------------
def train_one_frame(
    frame_num: int,
    frame_type: str,
    splats: torch.nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    parser: Parser,
    cfg: Config,
    writer: SummaryWriter,
    global_step_offset: int,
    device: str = "cuda",
) -> int:
    """Train on a single frame. Returns number of steps taken."""

    scene_scale = parser.scene_scale * 1.1 * cfg.global_scale

    # Determine max_steps
    if frame_type == "first":
        max_steps = cfg.first_frame_steps
    elif frame_type == "peak":
        max_steps = cfg.peak_frame_steps
    else:
        max_steps = cfg.normal_frame_steps

    # Strategy
    if frame_type == "first":
        strategy = MCMCStrategy(
            cap_max=cfg.first_frame_cap,
            noise_lr=cfg.mcmc_noise_lr,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=min(cfg.refine_stop_iter, max_steps - 500),
            refine_every=cfg.mcmc_refine_every,
            verbose=True,
        )
        print(f"  [First] MCMC cap={cfg.first_frame_cap}")
    elif frame_type == "peak":
        # MCMC: relocate dead Gaussians + densify up to cap
        # Allow count to grow to handle new content from large motion
        n_gs = len(splats["means"])
        cap = int(n_gs * (1.0 + cfg.peak_cap_growth))
        strategy = MCMCStrategy(
            cap_max=cap,
            noise_lr=cfg.mcmc_noise_lr,
            refine_start_iter=100,
            refine_stop_iter=max_steps - 500,
            refine_every=cfg.mcmc_refine_every,
            verbose=True,
        )
        print(f"  [Peak] Densification enabled: {n_gs} → cap {cap} (+{cfg.peak_cap_growth*100:.0f}%)")
    else:
        # Normal: MCMC with relocation enabled (less aggressive than peak)
        # Keeps count fixed via cap_max but relocates dead Gaussians
        # so the model can adapt to subtle motion
        n_gs = len(splats["means"])
        strategy = MCMCStrategy(
            cap_max=n_gs,
            noise_lr=cfg.mcmc_noise_lr * 0.1,
            refine_start_iter=100,
            refine_stop_iter=max_steps - 500,
            refine_every=cfg.mcmc_refine_every * 2,  # less frequent than peak
            verbose=False,
        )

    strategy.check_sanity(splats, optimizers)

    if isinstance(strategy, DefaultStrategy):
        strategy_state = strategy.initialize_state(scene_scale=scene_scale)
    else:
        strategy_state = strategy.initialize_state()

    # Dataset & dataloader
    trainset = Dataset(parser, split="train", patch_size=None, load_depths=False)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )
    trainloader_iter = iter(trainloader)

    # LR scheduler
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
            optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
        ),
    ]

    # Detect frame format for mask loading
    frame_fmt = "06d"
    mask_ext = ".jpg"
    if cfg.mask_dir:
        # Auto-detect from first camera
        image_dir = os.path.join(cfg.data_dir, "images")
        cam_dirs = sorted(d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d)))
        if cam_dirs:
            import re
            fnames = sorted(f for f in os.listdir(os.path.join(image_dir, cam_dirs[0])) if re.match(r'\d+\.', f))
            if fnames:
                stem, ext = os.path.splitext(fnames[0])
                frame_fmt = f"0{len(stem)}d"
                mask_ext = ext

    # Training loop
    pbar = tqdm.tqdm(range(max_steps), desc=f"Frame {frame_num} ({frame_type})")
    for step in pbar:
        try:
            data = next(trainloader_iter)
        except StopIteration:
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)

        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device)
        height, width = pixels.shape[1:3]

        # Load mask if available
        masks = None
        if cfg.mask_dir:
            seg_masks = data.get("segmentation_mask", None)
            if seg_masks is not None:
                masks = seg_masks.to(device)

        # SH schedule
        if frame_type == "first":
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        else:
            sh_degree_to_use = cfg.sh_degree  # full SH from start for fine-tuning

        # Forward
        colors_render, alphas, info = rasterize_splats(
            splats, camtoworlds, Ks, width, height, cfg,
            sh_degree=sh_degree_to_use,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
        )

        if cfg.random_bkgd:
            bkgd = torch.rand(1, 3, device=device)
            colors_render = colors_render + bkgd * (1.0 - alphas)

        # Strategy pre-backward
        if isinstance(strategy, DefaultStrategy):
            strategy.step_pre_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
            )

        # --- Loss ---
        if masks is not None:
            keep_mask = (masks > 0.5).float().unsqueeze(-1)  # [B, H, W, 1]
            l1loss = (torch.abs(colors_render - pixels) * keep_mask).sum() / (keep_mask.sum() * 3).clamp(min=1.0)
            colors_m = colors_render * keep_mask
            pixels_m = pixels * keep_mask
            ssimloss = 1.0 - fused_ssim(
                colors_m.permute(0, 3, 1, 2),
                pixels_m.permute(0, 3, 1, 2),
                padding="valid",
            )
        else:
            l1loss = F.l1_loss(colors_render, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors_render.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )

        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

        # Alpha penalty in masked background
        if masks is not None:
            masked_region = 1.0 - masks
            alpha_in_bg = (alphas.squeeze(-1) * masked_region).sum() / masked_region.sum().clamp(min=1.0)
            loss = loss + 0.1 * alpha_in_bg

        # Regularizations
        if cfg.opacity_reg > 0.0:
            loss += cfg.opacity_reg * torch.abs(torch.sigmoid(splats["opacities"])).mean()
        if cfg.scale_reg > 0.0:
            loss += cfg.scale_reg * torch.abs(torch.exp(splats["scales"])).mean()

        loss.backward()

        desc = (f"Frame {frame_num} ({frame_type}) "
                f"loss={loss.item():.4f} l1={l1loss.item():.4f} "
                f"sh={sh_degree_to_use} N={len(splats['means'])}")
        pbar.set_description(desc)

        # Tensorboard
        gs = global_step_offset + step
        if cfg.tb_every > 0 and step % cfg.tb_every == 0:
            writer.add_scalar(f"train/loss", loss.item(), gs)
            writer.add_scalar(f"train/l1loss", l1loss.item(), gs)
            writer.add_scalar(f"train/ssimloss", ssimloss.item(), gs)
            writer.add_scalar(f"train/num_GS", len(splats["means"]), gs)
            writer.add_scalar(f"train/frame", frame_num, gs)
            writer.flush()

        # Optimize
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        for scheduler in schedulers:
            scheduler.step()

        # Strategy post-backward
        if isinstance(strategy, DefaultStrategy):
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
                packed=cfg.packed,
            )
        elif isinstance(strategy, MCMCStrategy):
            strategy.step_post_backward(
                params=splats, optimizers=optimizers,
                state=strategy_state, step=step, info=info,
                lr=schedulers[0].get_last_lr()[0],
            )

    return max_steps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_frame(
    frame_num: int,
    splats: torch.nn.ParameterDict,
    parser: Parser,
    cfg: Config,
    writer: SummaryWriter,
    global_step: int,
    device: str = "cuda",
):
    """Evaluate on validation images for a given frame."""
    valset = Dataset(parser, split="val")
    if len(valset) == 0:
        valset = Dataset(parser, split="train")
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)

    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    metrics = defaultdict(list)
    render_dir = os.path.join(cfg.result_dir, "renders", f"frame_{frame_num:06d}")
    os.makedirs(render_dir, exist_ok=True)

    for i, data in enumerate(valloader):
        camtoworlds = data["camtoworld"].to(device)
        Ks = data["K"].to(device)
        pixels = data["image"].to(device)
        height, width = pixels.shape[1:3]

        masks = None
        seg_masks = data.get("segmentation_mask", None)
        if seg_masks is not None and cfg.mask_dir:
            masks = seg_masks.to(device)

        colors, _, _ = rasterize_splats(
            splats, camtoworlds, Ks, width, height, cfg,
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
        )
        colors = torch.clamp(colors, 0.0, 1.0)

        if masks is not None:
            keep = (masks > 0.5).float().unsqueeze(-1)
            colors_eval = colors * keep
            pixels_eval = pixels * keep
        else:
            colors_eval = colors
            pixels_eval = pixels

        colors_p = colors_eval.permute(0, 3, 1, 2)
        pixels_p = pixels_eval.permute(0, 3, 1, 2)
        metrics["psnr"].append(psnr_fn(colors_p, pixels_p))
        metrics["ssim"].append(ssim_fn(colors_p, pixels_p))

        # Save first few images
        if i < 4:
            canvas = torch.cat([pixels_eval, colors_eval], dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            imageio.imwrite(f"{render_dir}/val_{i:04d}.png", canvas)

    stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
    stats["num_GS"] = len(splats["means"])
    print(f"  [Eval] Frame {frame_num}: PSNR={stats['psnr']:.2f}, "
          f"SSIM={stats['ssim']:.4f}, GS={stats['num_GS']}")

    writer.add_scalar("eval/psnr", stats["psnr"], global_step)
    writer.add_scalar("eval/ssim", stats["ssim"], global_step)
    writer.flush()

    with open(f"{render_dir}/metrics.json", "w") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = tyro.cli(Config)
    set_random_seed(42)
    device = "cuda"

    os.makedirs(cfg.result_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.result_dir, "ckpts")
    ply_dir = os.path.join(cfg.result_dir, "ply")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(ply_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(cfg.result_dir, "tb"))

    # Save config
    with open(os.path.join(cfg.result_dir, "cfg.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # Classify frames
    frame_classes = classify_frames(cfg)
    start = cfg.start_frame
    end = start + cfg.num_frames

    print(f"\n{'='*70}")
    print(f"Sequential 3DGS Training")
    print(f"  Frames: {start} → {end-1} ({cfg.num_frames} frames)")
    print(f"  Steps: first={cfg.first_frame_steps}, "
          f"normal={cfg.normal_frame_steps}, peak={cfg.peak_frame_steps}")
    n_peaks = sum(1 for v in frame_classes.values() if v == "peak")
    n_normal = sum(1 for v in frame_classes.values() if v == "normal")
    print(f"  Classification: 1 first + {n_peaks} peaks + {n_normal} normal")
    print(f"{'='*70}\n")

    # Track cumulative steps for tensorboard
    global_step = 0
    carried_state = None
    scene_scale = None

    for frame_num in range(start, end):
        frame_type = frame_classes[frame_num]
        print(f"\n{'='*70}")
        print(f"Frame {frame_num} ({frame_type})")
        print(f"{'='*70}")

        # Create parser for this frame
        parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=False,
            test_every=cfg.test_every,
            frame_num=frame_num,
            use_masks=cfg.mask_dir is not None,
        )

        if scene_scale is None:
            scene_scale = parser.scene_scale * 1.1 * cfg.global_scale

        # Create splats
        if frame_type == "first":
            splats, optimizers = create_splats_from_sfm(
                parser, cfg, scene_scale, device
            )
            print(f"  SFM init: {len(splats['means'])} Gaussians")
        else:
            lr_scale = cfg.peak_lr_scale if frame_type == "peak" else cfg.finetune_lr_scale
            splats, optimizers = create_splats_from_checkpoint(
                carried_state, cfg, scene_scale, lr_scale, device
            )
            print(f"  Carry-forward: {len(splats['means'])} Gaussians, "
                  f"lr_scale={lr_scale}")

        # Train
        t0 = time.time()
        steps_taken = train_one_frame(
            frame_num=frame_num,
            frame_type=frame_type,
            splats=splats,
            optimizers=optimizers,
            parser=parser,
            cfg=cfg,
            writer=writer,
            global_step_offset=global_step,
            device=device,
        )
        elapsed = time.time() - t0
        print(f"  Trained {steps_taken} steps in {elapsed:.1f}s "
              f"({steps_taken/elapsed:.1f} it/s)")
        global_step += steps_taken

        # Carry forward
        carried_state = {k: v.detach().clone().cpu() for k, v in splats.items()}

        # Save checkpoint
        torch.save(
            {"frame": frame_num, "splats": {k: v.cpu() for k, v in carried_state.items()}},
            os.path.join(ckpt_dir, f"frame_{frame_num:06d}.pt"),
        )

        # Save PLY
        if cfg.save_ply:
            export_splats(
                means=splats["means"],
                scales=splats["scales"],
                quats=splats["quats"],
                opacities=splats["opacities"],
                sh0=splats["sh0"],
                shN=splats["shN"],
                format="ply",
                save_to=os.path.join(ply_dir, f"frame_{frame_num:06d}.ply"),
            )

        # Eval periodically
        if frame_num % cfg.eval_every_n_frames == 0 or frame_num == end - 1:
            eval_frame(frame_num, splats, parser, cfg, writer, global_step, device)

        # Free GPU memory between frames
        del splats, optimizers
        torch.cuda.empty_cache()

    writer.close()
    print(f"\n{'='*70}")
    print(f"Done! Results in {cfg.result_dir}")
    print(f"  Checkpoints: {ckpt_dir}/")
    print(f"  PLY files: {ply_dir}/")
    print(f"  TensorBoard: tensorboard --logdir {cfg.result_dir}/tb")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
