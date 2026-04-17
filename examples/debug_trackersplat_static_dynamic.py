"""Debug: verify static/dynamic split + cotracker→dynamic binding before full training.

Renders for a few sample cameras at frame 0:
  - composite (static + dynamic)
  - dynamic-only (static masked out via runner.static = None)
  - dynamic alpha
  - GT image masked by dynamic alpha (background blacked out)
  - GT image with cotracker points colored: GREEN = bound to dynamic (kept),
    RED = bound to static (dropped as background).

Usage:
    conda activate gsplat
    python examples/debug_trackersplat_static_dynamic.py
"""

import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import tyro

from trackersplat_trainer import Config, TrackerSplatRunner


DATA_DIR = "/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/"
DYNAMIC_PLY = "/data/shared/elaheh/thenewface_static_v2/dynamic.ply"
STATIC_PLY = "/data/shared/elaheh/thenewface_static_v2/static.ply"
COTRACKER_DIR = "/data/shared/elaheh/4D_demo/new_data/thenewface/cotracker_out/"
OUT_DIR = "/data/shared/elaheh/4D_demo/new_data/trackersplat_results/debug_static_dynamic"


def main(
    out_dir: str = OUT_DIR,
    data_factor: int = 4,
    sample_cams: tuple = (0, 22, 44),
):
    cfg = Config(
        data_dir=DATA_DIR,
        ply_path=DYNAMIC_PLY,
        static_ply_path=STATIC_PLY,
        cotracker_dir=COTRACKER_DIR,
        result_dir=out_dir,
        data_factor=data_factor,
        sh_degree=0,
        normalize_world_space=False,
    )
    runner = TrackerSplatRunner(cfg)

    debug_dir = os.path.join(out_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    summary_lines = []

    for cam_idx in sample_cams:
        if cam_idx >= runner.num_cameras:
            continue
        ct_idx = runner.cam_to_ct[cam_idx]
        if ct_idx < 0:
            print(f"cam {cam_idx} has no cotracker data, skipping")
            continue
        cam_name = runner.cam_names[cam_idx]
        print(f"\n=== {cam_name} (cam_idx={cam_idx}) ===")

        with torch.no_grad():
            # 1. Composite render (static + dynamic)
            comp, _, _ = runner.rasterize_splats(cam_idx, 0)
            comp_img = (comp[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

            # 2. Dynamic-only (temporarily disable static)
            saved_static = runner.static
            runner.static = None
            dyn_rgb, dyn_alpha, _ = runner.rasterize_splats(cam_idx, 0)
            runner.static = saved_static
            dyn_img = (dyn_rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            alpha = dyn_alpha[0, ..., 0].clamp(0, 1).cpu().numpy()
            alpha_u8 = (alpha * 255).astype(np.uint8)

            # 3. GT image
            gt_t = runner.load_image(cam_idx, 0)
            gt_img = (gt_t.cpu().numpy() * 255).astype(np.uint8)

        # Crop to common size
        h = min(comp_img.shape[0], dyn_img.shape[0], gt_img.shape[0])
        w = min(comp_img.shape[1], dyn_img.shape[1], gt_img.shape[1])
        comp_img = comp_img[:h, :w]
        dyn_img = dyn_img[:h, :w]
        alpha = alpha[:h, :w]
        alpha_u8 = alpha_u8[:h, :w]
        gt_img = gt_img[:h, :w]

        # 4. GT masked by dynamic alpha (background blacked out)
        gt_masked = (gt_img.astype(np.float32) * alpha[..., None]).astype(np.uint8)

        # 5. GT with cotracker classification (green=dynamic, red=background)
        gt_overlay = gt_img.copy()
        bgr = cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR)
        gt_pts = runner.gt_tracks[ct_idx, 0].cpu().numpy()
        valid = runner.binding_valid[ct_idx].cpu().numpy()
        vis = runner.gt_vis[ct_idx, 0].cpu().numpy()
        H, W = bgr.shape[:2]
        n_kept, n_dropped, n_invisible = 0, 0, 0
        for i in range(len(gt_pts)):
            if not vis[i]:
                n_invisible += 1
                continue
            x, y = int(gt_pts[i, 0]), int(gt_pts[i, 1])
            if not (0 <= x < W and 0 <= y < H):
                continue
            if valid[i]:
                cv2.circle(bgr, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
                n_kept += 1
            else:
                cv2.circle(bgr, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
                n_dropped += 1
        cls_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Save individual files
        prefix = os.path.join(debug_dir, cam_name)
        imageio.imwrite(f"{prefix}_01_gt.png", gt_img)
        imageio.imwrite(f"{prefix}_02_composite.png", comp_img)
        imageio.imwrite(f"{prefix}_03_dynamic_only.png", dyn_img)
        imageio.imwrite(f"{prefix}_04_dynamic_alpha.png", alpha_u8)
        imageio.imwrite(f"{prefix}_05_gt_masked_by_dyn.png", gt_masked)
        imageio.imwrite(f"{prefix}_06_cotracker_classified.png", cls_img)

        # Side-by-side panel: [GT | composite | dyn_only | gt_masked | classified]
        def label(img, txt):
            out = img.copy()
            cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 0), 1, cv2.LINE_AA)
            return out

        panel = np.concatenate([
            label(gt_img, "GT"),
            label(comp_img, "composite"),
            label(dyn_img, "dynamic-only"),
            label(gt_masked, "GT masked by dyn alpha"),
            label(cls_img, "cotracker (green=dyn, red=bg)"),
        ], axis=1)
        imageio.imwrite(f"{prefix}_panel.png", panel)

        line = (f"{cam_name}: kept={n_kept}  dropped_bg={n_dropped}  "
                f"invisible={n_invisible}  "
                f"alpha_mean={alpha.mean():.3f}  alpha_cover={(alpha>0.5).mean():.3%}")
        print(line)
        summary_lines.append(line)

    with open(os.path.join(debug_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nDebug images saved to {debug_dir}/")


if __name__ == "__main__":
    tyro.cli(main)
