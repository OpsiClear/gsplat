"""Evaluate PSNR / SSIM / L1 for TrackerSplat checkpoints.

Loads a cfg.json + every ckpt_*.pt in <result_dir>/ckpts, renders each
CoTracker camera at each of the num_cotracker_frames timesteps, compares to
the downsampled GT image, and writes eval_psnr_<step>.json.

Usage:
    conda activate gsplat
    python examples/eval_trackersplat_psnr.py \
        --result_dir /data/shared/elaheh/4D_demo/new_data/trackersplat_results/run_50k_fixed
"""

import json
import math
import os
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn.functional as F
import tqdm
import tyro
from fused_ssim import fused_ssim

from trackersplat_trainer import Config, TrackerSplatRunner


def load_cfg(result_dir: str) -> Config:
    with open(os.path.join(result_dir, "cfg.json")) as f:
        raw = json.load(f)
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in valid})


@torch.no_grad()
def eval_ckpt(runner: TrackerSplatRunner, ckpt_path: str) -> dict:
    runner.load_checkpoint(ckpt_path)
    cfg = runner.cfg

    total_mse = 0.0
    total_l1 = 0.0
    total_ssim = 0.0
    n = 0

    for ct_idx in tqdm.tqdm(range(len(runner.ct_to_cam)),
                            desc=Path(ckpt_path).name):
        cam_idx = runner.ct_to_cam[ct_idx]
        for t in range(cfg.num_cotracker_frames):
            rendered, _, _ = runner.rasterize_splats(cam_idx, t)
            gt = runner.load_image(cam_idx, t).unsqueeze(0)  # (1, H, W, 3)

            h = min(rendered.shape[1], gt.shape[1])
            w = min(rendered.shape[2], gt.shape[2])
            rendered = rendered[:, :h, :w, :].clamp(0, 1)
            gt = gt[:, :h, :w, :]

            mse = F.mse_loss(rendered, gt).item()
            l1 = F.l1_loss(rendered, gt).item()
            ssim = fused_ssim(
                rendered.permute(0, 3, 1, 2),
                gt.permute(0, 3, 1, 2),
                padding="valid",
            ).item()

            total_mse += mse
            total_l1 += l1
            total_ssim += ssim
            n += 1

    mean_mse = total_mse / n
    psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))
    return {
        "psnr": psnr,
        "ssim": total_ssim / n,
        "l1": total_l1 / n,
        "mse": mean_mse,
        "num_views": n,
    }


def main(result_dir: str):
    cfg = load_cfg(result_dir)
    cfg.result_dir = result_dir
    runner = TrackerSplatRunner(cfg)

    ckpt_dir = os.path.join(result_dir, "ckpts")
    ckpts = sorted(
        Path(ckpt_dir).glob("ckpt_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    print(f"Found {len(ckpts)} checkpoints in {ckpt_dir}")

    summary = {}
    for ckpt in ckpts:
        step = int(ckpt.stem.split("_")[1])
        metrics = eval_ckpt(runner, str(ckpt))
        summary[step] = metrics
        out_path = os.path.join(result_dir, f"eval_psnr_{step}.json")
        with open(out_path, "w") as f:
            json.dump({"step": step, **metrics}, f, indent=2)
        print(
            f"[step {step}] PSNR={metrics['psnr']:.3f}  "
            f"SSIM={metrics['ssim']:.4f}  L1={metrics['l1']:.4f}  "
            f"({metrics['num_views']} views)"
        )

    with open(os.path.join(result_dir, "eval_psnr_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {result_dir}/eval_psnr_summary.json")


if __name__ == "__main__":
    tyro.cli(main)
