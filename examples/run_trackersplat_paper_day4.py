"""
Day 4 — one-frame TrackerSplat pipeline on thenewface.

Runs motion_fusion → PWI-LS → ISVD translation-only compensation on a single
(source_frame → target_frame) pair and renders before/after via FasterGS.

Outputs land under $OUT_DIR (default /data/shared/elaheh/4D_demo/new_data/
trackersplat_paper_day4/):
  cam{N}/gt_target.png        ground-truth target-frame image
  cam{N}/render_before.png    FasterGS render of the baseframe PLY
  cam{N}/render_after.png     FasterGS render after compensate(Motion)
  cam{N}/diff_before.png      5× |gt - before|
  cam{N}/diff_after.png       5× |gt - after|
  panel_cam{N}.png            4-up: gt | before | after | 5·|after−gt|
  metrics.json                per-cam + overall PSNR/SSIM/L1 before vs after
  motion_stats.json           #Gaussians solved, translation norm stats

Usage:
    conda activate gsplat_fastergs
    python examples/run_trackersplat_paper_day4.py \\
        --target_frame 3 --n_views 5 --n_gaussians_max 50000

Tunables default to fast first-run values; bump them up for the full pipeline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
torch.set_grad_enabled(False)
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from trackersplat_dataset import build_thenewface_video_dataset, Camera  # noqa
from trackersplat_paper.motion import Motion, compensate                  # noqa
from trackersplat_paper.pipeline import (                                  # noqa
    compute_translation_motion,
    compute_translation_motion_per_track,
)


# -----------------------------------------------------------------------------
# alltrackerxx loader — matches the schema in trackersplat_trainer_fastgs
# -----------------------------------------------------------------------------
def _load_alltrackerxx_tracks(
    cotracker_dir: str, cam_name: str, device: torch.device,
    W_render: int, H_render: int,
):
    """Return (tracks [T, N, 2], vis [T, N])   or (None, None) if absent."""
    path = Path(cotracker_dir) / f"{cam_name}.npz"
    if not path.exists():
        return None, None
    d = np.load(str(path))
    if "trajs" in d.files:
        tracks = torch.from_numpy(d["trajs"]).float().to(device)
        vis = torch.from_numpy(d["vis"] > 0.5).bool().to(device)
        W_tr = int(d["image_w"]); H_tr = int(d["image_h"])
        sx = W_render / W_tr; sy = H_render / H_tr
        tracks = tracks * torch.tensor([sx, sy], device=device)
    elif "tracks" in d.files:
        tracks = torch.from_numpy(d["tracks"].squeeze(0)).float().to(device)
        vis = torch.from_numpy(d["visibility"].squeeze(0)).bool().to(device)
        df = float(d["downsample_factor"])
        # tracks are at full-res, rescale to render resolution
        tracks = tracks * df
    else:
        return None, None
    return tracks, vis


# -----------------------------------------------------------------------------
# Tiny Gaussian container our pipeline expects (duck-typed)
# -----------------------------------------------------------------------------
class _Gaussians:
    def __init__(self, means, rotations, raw_log_scales, raw_opacity,
                 sh0=None, shN=None):
        self._xyz = torch.nn.Parameter(means)
        self._rotation = torch.nn.Parameter(rotations)
        self._scaling = torch.nn.Parameter(raw_log_scales)
        self._opacity = torch.nn.Parameter(raw_opacity)
        # sh0/shN carried opaquely so compensate() preserves them
        self._sh0 = sh0
        self._shN = shN


def _load_gaussians_from_ply(ply_path: str, device: torch.device, n_max: int = -1):
    """Load a PLY via gsplat.io_ply.import_splats; optional subsample-to-topN."""
    from gsplat.io_ply import import_splats
    means, log_scales, quats, raw_opac, sh0, shN = import_splats(ply_path, device)
    if raw_opac.dim() > 1:
        raw_opac = raw_opac.squeeze(-1)
    N = means.shape[0]
    if n_max > 0 and N > n_max:
        topk = torch.topk(raw_opac, n_max).indices
        means = means[topk]; log_scales = log_scales[topk]
        quats = quats[topk]; raw_opac = raw_opac[topk]
        sh0 = sh0[topk]; shN = shN[topk]
        print(f"[load] subsampled from {N} to {n_max} Gaussians "
              f"(top-opacity)  from {ply_path}")
    else:
        print(f"[load] N={N} from {ply_path}")
    return _Gaussians(means, quats, log_scales, raw_opac, sh0=sh0, shN=shN)


def _compose_gaussians(static_g: _Gaussians, dynamic_g: _Gaussians) -> _Gaussians:
    """Concat static + dynamic splats into a single renderable container.
    Static always first so any downstream indexing by offset is predictable."""
    def _cat(a, b): return torch.cat([a.detach(), b.detach()], dim=0)
    return _Gaussians(
        means=_cat(static_g._xyz, dynamic_g._xyz),
        rotations=_cat(static_g._rotation, dynamic_g._rotation),
        raw_log_scales=_cat(static_g._scaling, dynamic_g._scaling),
        raw_opacity=_cat(static_g._opacity, dynamic_g._opacity),
        sh0=_cat(static_g._sh0, dynamic_g._sh0),
        shN=_cat(static_g._shN, dynamic_g._shN),
    )


# -----------------------------------------------------------------------------
# FasterGS render helper
# -----------------------------------------------------------------------------
def _render_via_fastergs(gauss: _Gaussians, cam: Camera, device: torch.device) -> Tensor:
    """Render gauss → (H, W, 3) via FasterGSCudaBackend.torch_bindings.rasterize.
    Matches the convention used by trackersplat_trainer_fastgs: RAW log_scales,
    RAW (N,1) opacities, (w, x, y, z) quats, sh0 (N,1,3), shN (N,K-1,3)."""
    from FasterGSCudaBackend.torch_bindings import rasterize, RasterizerSettings
    N = gauss._xyz.shape[0]
    raw_opac = gauss._opacity.detach()
    if raw_opac.dim() == 1:
        raw_opac = raw_opac.unsqueeze(-1)
    sh0 = gauss._sh0 if gauss._sh0 is not None \
          else torch.full((N, 1, 3), 0.5, device=device)
    shN = gauss._shN if gauss._shN is not None \
          else torch.zeros((N, 0, 3), device=device)
    active_sh_bases = sh0.shape[1] + shN.shape[1]

    settings = RasterizerSettings(
        w2c=cam.w2c.contiguous(),
        cam_position=cam.cam_position.contiguous(),
        bg_color=torch.zeros(3, device=device),
        active_sh_bases=int(active_sh_bases),
        width=int(cam.image_width), height=int(cam.image_height),
        focal_x=float(cam.focal_x), focal_y=float(cam.focal_y),
        center_x=float(cam.center_x), center_y=float(cam.center_y),
        near_plane=0.01, far_plane=1e10, proper_antialiasing=False,
    )
    img = rasterize(
        gauss._xyz.detach().contiguous(),
        gauss._scaling.detach().contiguous(),
        gauss._rotation.detach().contiguous(),
        raw_opac.contiguous(),
        sh0.detach().contiguous(),
        shN.detach().contiguous(),
        settings, to_chw=True,
    )
    return img.permute(1, 2, 0).clamp(0, 1)  # (H, W, 3)


def _psnr(a: Tensor, b: Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _save(img: Tensor, path: str):
    arr = (img.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
    imageio.imwrite(path, arr)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/")
    ap.add_argument("--ply_path", default="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/inside.ply")
    ap.add_argument("--static_ply_path", default="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/outside.ply")
    ap.add_argument("--cotracker_dir", default="/data/shared/elaheh/4D_demo/new_data/thenewface/alltrackerxx_out/")
    ap.add_argument("--out_dir", default="/data/shared/elaheh/4D_demo/new_data/trackersplat_paper_day4/")
    ap.add_argument("--target_frame", type=int, default=3, help="frame index (0 = first)")
    ap.add_argument("--source_frame", type=int, default=0)
    ap.add_argument("--frame_step", type=int, default=6)
    ap.add_argument("--n_views", type=int, default=5, help="number of cameras to use for PWI-LS")
    ap.add_argument("--n_gaussians_max", type=int, default=30000, help="subsample cap on dynamic PLY")
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--eval_cam_stride", type=int, default=1, help="render panels for every Nth camera")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--solver", choices=["pwils", "per_track"], default="per_track",
                    help="pwils = paper's PWI-LS (needs dense tracks, e.g. DOT); "
                         "per_track = direct triangulation (works with sparse tracks like alltrackerxx)")
    ap.add_argument("--max_track_pixel_distance", type=float, default=30.0,
                    help="per_track only: drop tracks whose nearest Gaussian is "
                         "further than this many px in frame 0")
    args = ap.parse_args()

    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)
    tic_total = time.perf_counter()

    # 1. video dataset
    print(f"[1/5] Loading dataset at {args.data_dir} ...")
    video = build_thenewface_video_dataset(
        data_dir=args.data_dir, n_frames=args.target_frame + 1,
        frame_step=args.frame_step, data_factor=args.data_factor,
        device="cuda",
    )
    print(f"      {len(video)} frames × {len(video[0])} cameras")

    # 2. dynamic + static PLYs
    print(f"[2/5] Loading dynamic  (inside.ply) and static (outside.ply) ...")
    gauss_dyn = _load_gaussians_from_ply(args.ply_path, device, args.n_gaussians_max)
    gauss_static = _load_gaussians_from_ply(args.static_ply_path, device, n_max=-1)
    N = gauss_dyn._xyz.shape[0]
    N_static = gauss_static._xyz.shape[0]
    print(f"      dynamic N={N}   static N={N_static}")

    # 3. tracks per view
    print(f"[3/5] Loading alltrackerxx tracks for {args.n_views} cameras ...")
    cameras_used, tracks_used, vis_used = [], [], []
    skipped = 0
    for cam in video[0]:
        tracks, vis = _load_alltrackerxx_tracks(
            args.cotracker_dir, cam.cam_name, device,
            int(cam.image_width), int(cam.image_height),
        )
        if tracks is None:
            skipped += 1
            continue
        cameras_used.append(cam)
        tracks_used.append(tracks)
        vis_used.append(vis)
        if len(cameras_used) >= args.n_views:
            break
    print(f"      selected {len(cameras_used)} cameras "
          f"({skipped} skipped for missing NPZ); tracks per cam: "
          f"{tracks_used[0].shape if tracks_used else 'N/A'}")
    assert len(cameras_used) >= 2, "need at least 2 cameras with tracks for triangulation"

    # 4. solve Motion (translation only)
    print(f"[4/5] Running motion_fusion + PWI-LS + ISVD "
          f"(source frame {args.source_frame} → target {args.target_frame}) ...")
    tic = time.perf_counter()
    if args.solver == "pwils":
        print(f"      solver = PWI-LS (paper)")
        motion = compute_translation_motion(
            gauss_dyn, cameras_used, tracks_used, vis_used,
            target_frame_idx=args.target_frame,
            source_frame_idx=args.source_frame,
            verbose=True,
        )
    else:
        print(f"      solver = direct per-track triangulation (sparse-friendly, "
              f"max_track_pixel_distance={args.max_track_pixel_distance})")
        motion = compute_translation_motion_per_track(
            gauss_dyn, cameras_used, tracks_used, vis_used,
            target_frame_idx=args.target_frame,
            source_frame_idx=args.source_frame,
            max_track_pixel_distance=args.max_track_pixel_distance,
            verbose=True,
        )
    solve_time = time.perf_counter() - tic
    n_solved = int(motion.motion_mask_mean.sum()) if motion.motion_mask_mean is not None else 0
    print(f"      solved in {solve_time:.1f}s   "
          f"{n_solved}/{N} Gaussians got a translation")
    motion_stats = {
        "n_gaussians": N,
        "n_solved": n_solved,
        "solve_time_sec": round(solve_time, 3),
        "n_views_used": len(cameras_used),
        "source_frame": args.source_frame,
        "target_frame": args.target_frame,
    }
    if n_solved > 0 and motion.translation_vector is not None:
        norms = motion.translation_vector.norm(dim=-1)
        motion_stats.update({
            "translation_norm_mean": round(float(norms.mean()), 6),
            "translation_norm_max": round(float(norms.max()), 6),
            "translation_norm_p50": round(float(norms.median()), 6),
        })
    with open(os.path.join(args.out_dir, "motion_stats.json"), "w") as f:
        json.dump(motion_stats, f, indent=2)

    # 5. render before/after per eval camera + compute metrics
    #    render = static (frozen) ⊕ dynamic (before or after motion compensation)
    print(f"[5/5] Rendering before/after comparisons "
          f"(static {N_static} + dynamic {N} = {N + N_static} Gaussians) ...")
    gauss_dyn_after = compensate(gauss_dyn, motion)
    gauss_composite_before = _compose_gaussians(gauss_static, gauss_dyn)
    gauss_composite_after  = _compose_gaussians(gauss_static, gauss_dyn_after)
    per_cam_metrics = []
    frame_target = video[args.target_frame]

    # Render for every eval_cam_stride-th camera in the frame
    for ci in range(0, len(frame_target), args.eval_cam_stride):
        cam = frame_target[ci]
        H, W = int(cam.image_height), int(cam.image_width)
        gt = cam.load_image().to(device)                      # (H, W, 3)
        render_before = _render_via_fastergs(gauss_composite_before, cam, device)
        render_after  = _render_via_fastergs(gauss_composite_after,  cam, device)
        # crop to minimum shared size
        h = min(gt.shape[0], render_before.shape[0])
        w = min(gt.shape[1], render_before.shape[1])
        gt = gt[:h, :w]
        render_before = render_before[:h, :w]
        render_after = render_after[:h, :w]

        p_before = _psnr(render_before, gt)
        p_after = _psnr(render_after, gt)
        per_cam_metrics.append({
            "cam_idx": ci, "cam_name": cam.cam_name,
            "psnr_before": round(p_before, 3),
            "psnr_after": round(p_after, 3),
            "delta_psnr": round(p_after - p_before, 3),
        })

        cam_dir = os.path.join(args.out_dir, f"cam{ci:02d}_{cam.cam_name}")
        os.makedirs(cam_dir, exist_ok=True)
        _save(gt,              os.path.join(cam_dir, "gt_target.png"))
        _save(render_before,   os.path.join(cam_dir, "render_before.png"))
        _save(render_after,    os.path.join(cam_dir, "render_after.png"))
        _save((render_after - gt).abs() * 5, os.path.join(cam_dir, "diff_after.png"))
        _save((render_before - gt).abs() * 5, os.path.join(cam_dir, "diff_before.png"))
        # 4-panel side-by-side
        panel = torch.cat([gt, render_before, render_after,
                           ((render_after - gt).abs() * 5).clamp(0, 1)], dim=1)
        _save(panel, os.path.join(args.out_dir, f"panel_cam{ci:02d}.png"))

        print(f"      cam{ci:02d} {cam.cam_name:20s}  "
              f"PSNR before={p_before:5.2f}  after={p_after:5.2f}  "
              f"Δ={p_after - p_before:+.2f}")

    overall = {
        "mean_psnr_before": round(float(np.mean([m["psnr_before"] for m in per_cam_metrics])), 3),
        "mean_psnr_after":  round(float(np.mean([m["psnr_after"]  for m in per_cam_metrics])), 3),
        "mean_delta_psnr":  round(float(np.mean([m["delta_psnr"]  for m in per_cam_metrics])), 3),
        "per_cam": per_cam_metrics,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print(f"[done] mean PSNR  before={overall['mean_psnr_before']:.2f}  "
          f"after={overall['mean_psnr_after']:.2f}  "
          f"Δ={overall['mean_delta_psnr']:+.2f}")
    print(f"       total time: {time.perf_counter() - tic_total:.1f}s")
    print(f"       results at: {args.out_dir}")


if __name__ == "__main__":
    main()
