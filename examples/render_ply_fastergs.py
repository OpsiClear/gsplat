"""
Render a pre-trained gsplat PLY through FasterGSCudaBackend against every
camera's frame-0 ground-truth image — no training, no motion, no tracks.

Answers: is the PLY in the normalised or raw colmap frame? Which Parser
settings match? What PSNR can we expect before training even starts?

Usage:
    conda activate gsplat_fastergs
    python examples/render_ply_fastergs.py \\
        --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/ \\
        --ply_path /data/shared/elaheh/4D_demo/thenewface_multiframe_fast/frame_001/ply/point_cloud_combined_2999.ply \\
        --out_dir /tmp/render_ply_fastergs \\
        --sweep_normalize True

--sweep_normalize=True tries (normalize=True, skip_points3d=False) and
(normalize=False), reports mean PSNR across every camera for each, and picks
the winner automatically.
"""
from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from torchvision.utils import save_image

from datasets.colmap import Parser
from gsplat.io_ply import import_splats
from FasterGSCudaBackend.torch_bindings import rasterize, RasterizerSettings


@dataclass
class Config:
    data_dir: str
    ply_path: str
    out_dir: str = "/tmp/render_ply_fastergs"

    data_factor: int = 4
    sh_degree: int = 3
    frame_num: int = 1
    frame_step: int = 6
    frame_idx: int = 0           # CoTracker-style frame index

    near_plane: float = 0.01
    far_plane: float = 1e10
    proper_antialiasing: bool = False

    sweep_normalize: bool = True
    # Used only when sweep_normalize=False:
    normalize: bool = True
    skip_points3d: bool = False

    # Save rendered panels for up to this many cameras (per config). Others
    # get numeric PSNR only.
    max_saved_panels: int = 45


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a - b).pow(2).mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _render_cam_fastergs(
    cam_idx: int, parser: Parser,
    means, log_scales, quats, raw_opac, sh0, shN,
    cfg: Config, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Returns (gt, fastergs_img, cam_name). Both images (H, W, 3) in [0, 1]."""
    cam_id = parser.camera_ids[cam_idx]
    cam_name = os.path.dirname(parser.image_names[cam_idx])
    K = torch.from_numpy(parser.Ks_dict[cam_id]).float().to(device)
    W_img, H_img = parser.imsize_dict[cam_id]
    c2w = torch.from_numpy(parser.camtoworlds[cam_idx]).float().to(device)
    viewmat = torch.linalg.inv(c2w)

    # GT image
    image_num = cfg.frame_idx * cfg.frame_step + 1
    gt_path = os.path.join(cfg.data_dir, "images", cam_name, f"{image_num:06d}.jpg")
    gt = torch.from_numpy(imageio.imread(gt_path)).float().to(device) / 255.0
    if cfg.data_factor > 1:
        gt = gt[::cfg.data_factor, ::cfg.data_factor]

    # FasterGS render — RAW log_scales, RAW opacity reshaped to (N, 1)
    settings = RasterizerSettings(
        w2c=viewmat.contiguous(),
        cam_position=c2w[:3, 3].contiguous(),
        bg_color=torch.zeros(3, device=device),
        active_sh_bases=(cfg.sh_degree + 1) ** 2,
        width=int(W_img), height=int(H_img),
        focal_x=float(K[0, 0]), focal_y=float(K[1, 1]),
        center_x=float(K[0, 2]), center_y=float(K[1, 2]),
        near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        proper_antialiasing=cfg.proper_antialiasing,
    )
    with torch.no_grad():
        img = rasterize(
            means, log_scales, quats, raw_opac.unsqueeze(-1),
            sh0, shN, settings, to_chw=True,
        ).permute(1, 2, 0).clamp(0, 1)

    h = min(gt.shape[0], img.shape[0])
    w = min(gt.shape[1], img.shape[1])
    return gt[:h, :w], img[:h, :w], cam_name


def _run_config(
    cfg: Config, normalize: bool, skip_points3d: bool,
    means, log_scales, quats, raw_opac, sh0, shN,
    out_subdir: str, device: torch.device,
) -> float:
    """Render every camera for one (normalize, skip_points3d) combo; save
    panels + a per-camera CSV; return mean PSNR vs GT."""
    parser = Parser(
        data_dir=cfg.data_dir, factor=cfg.data_factor,
        normalize=normalize, test_every=9999,
        frame_num=cfg.frame_num, skip_points3d=skip_points3d,
    )
    out_dir = os.path.join(cfg.out_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    n = len(parser.image_names)
    psnrs: List[float] = []
    rows = ["cam_idx,cam_name,psnr_db,mean_gt,mean_render"]
    for cam_idx in range(n):
        gt, r, cam_name = _render_cam_fastergs(
            cam_idx, parser, means, log_scales, quats, raw_opac, sh0, shN,
            cfg, device,
        )
        p = _psnr(r, gt)
        psnrs.append(p)
        rows.append(f"{cam_idx},{cam_name},{p:.3f},{gt.mean():.4f},{r.mean():.4f}")
        if cam_idx < cfg.max_saved_panels:
            diff = (gt - r).abs().clamp(0, 1) * 5.0
            panel = torch.cat([gt.permute(2, 0, 1),
                               r.permute(2, 0, 1),
                               diff.permute(2, 0, 1)], dim=-1)
            save_image(panel, os.path.join(out_dir, f"cam{cam_idx:02d}_{cam_name}.png"))
    with open(os.path.join(out_dir, "psnr.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")

    mean = float(np.mean(psnrs))
    worst = float(np.min(psnrs))
    best = float(np.max(psnrs))
    print(f"[{out_subdir:40s}] cams={n}  mean PSNR = {mean:5.2f} dB  "
          f"(min {worst:.2f} / max {best:.2f})")
    return mean


def main():
    cfg = tyro.cli(Config)
    device = torch.device("cuda")

    if os.path.isdir(cfg.out_dir):
        shutil.rmtree(cfg.out_dir)
    os.makedirs(cfg.out_dir, exist_ok=True)

    means, log_scales, quats, raw_opac, sh0, shN = import_splats(cfg.ply_path, device)
    print(f"[ply] {cfg.ply_path}")
    print(f"[ply] {means.shape[0]} Gaussians  "
          f"means range x:[{means[:,0].min():.2f},{means[:,0].max():.2f}]  "
          f"y:[{means[:,1].min():.2f},{means[:,1].max():.2f}]  "
          f"z:[{means[:,2].min():.2f},{means[:,2].max():.2f}]")
    print(f"[ply] log_scales[min,max]=[{log_scales.min():.2f},{log_scales.max():.2f}]  "
          f"raw_opac[min,max]=[{raw_opac.min():.2f},{raw_opac.max():.2f}]  "
          f"sh0[min,max]=[{sh0.min():.3f},{sh0.max():.3f}]")

    if cfg.sweep_normalize:
        results = {}
        for tag, norm, skip in [
            ("norm=True_skip3d=False",  True,  False),
            ("norm=True_skip3d=True",   True,  True),
            ("norm=False_skip3d=False", False, False),
            ("norm=False_skip3d=True",  False, True),
        ]:
            try:
                results[tag] = _run_config(
                    cfg, norm, skip,
                    means, log_scales, quats, raw_opac, sh0, shN,
                    out_subdir=tag, device=device,
                )
            except Exception as e:
                print(f"[{tag}] FAILED: {e!s}")
                results[tag] = -1.0
        best_tag = max(results, key=results.get)
        print("\n=====================================================")
        print(f"[winner] {best_tag}  (mean PSNR = {results[best_tag]:.2f} dB)")
        print(f"[winner] use these Parser settings in your trainer:")
        norm, skip = results[best_tag], None
        for tag in results:
            if tag == best_tag:
                parts = tag.split("_")
                print(f"         normalize = {parts[0].split('=')[1]}")
                print(f"         skip_points3d = {parts[1].split('=')[1]}")
        print("=====================================================")
    else:
        _run_config(
            cfg, cfg.normalize, cfg.skip_points3d,
            means, log_scales, quats, raw_opac, sh0, shN,
            out_subdir=f"norm={cfg.normalize}_skip3d={cfg.skip_points3d}",
            device=device,
        )

    print(f"\npanels saved to {cfg.out_dir}/")


if __name__ == "__main__":
    main()
