"""Export a Plotly HTML of 4D Gaussian anchor trajectories from a TrackerSplat run.

Loads cfg.json + a checkpoint from <result_dir> and calls the runner's
visualize_3d_trajectories() to build an interactive 3D scatter with a
frame-slider for time.

Usage:
    conda activate gsplat
    CUDA_VISIBLE_DEVICES=0 python examples/viz_trackersplat_trajectories.py \\
        --result_dir /data/shared/elaheh/4D_demo/new_data/trackersplat_results/run_50k_fixed
    # Optional: --ckpt <path> --subsample 5 --save_path traj.html
"""

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Optional

import tyro

from trackersplat_trainer import Config, TrackerSplatRunner


def load_cfg(result_dir: str) -> Config:
    with open(os.path.join(result_dir, "cfg.json")) as f:
        raw = json.load(f)
    valid = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in raw.items() if k in valid})


def pick_latest_ckpt(result_dir: str) -> str:
    ckpts = sorted(
        Path(result_dir, "ckpts").glob("ckpt_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not ckpts:
        raise FileNotFoundError(f"No ckpt_*.pt in {result_dir}/ckpts")
    return str(ckpts[-1])


def main(
    result_dir: str,
    ckpt: Optional[str] = None,
    subsample: int = 1,
    density_keep_pct: float = 0.5,
    max_anchors: int = 300,
    scene_max_points: int = 20000,
    marker_size: int = 6,
    scene_marker_size: int = 2,
    clip_to_camera_rig: bool = True,
    rig_margin: float = 0.05,
    trajectory_color: str = "red",
    trajectory_width: float = 1.5,
    save_path: Optional[str] = None,
):
    cfg = load_cfg(result_dir)
    cfg.result_dir = result_dir
    runner = TrackerSplatRunner(cfg)

    ckpt_path = ckpt or pick_latest_ckpt(result_dir)
    runner.load_checkpoint(ckpt_path)

    step = int(Path(ckpt_path).stem.split("_")[1])
    if save_path is None:
        save_path = os.path.join(result_dir, f"trajectories_3d_step_{step}.html")
    runner.visualize_3d_trajectories(
        save_path=save_path,
        subsample=subsample,
        density_keep_pct=density_keep_pct,
        max_anchors=max_anchors,
        scene_max_points=scene_max_points,
        marker_size=marker_size,
        scene_marker_size=scene_marker_size,
        clip_to_camera_rig=clip_to_camera_rig,
        rig_margin=rig_margin,
        trajectory_color=trajectory_color,
        trajectory_width=trajectory_width,
    )
    print(f"\nHTML: {save_path}")


if __name__ == "__main__":
    tyro.cli(main)
