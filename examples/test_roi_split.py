#!/usr/bin/env python
"""
Sanity-test ROI-based static/dynamic split for the per-frame PLYs.

1. Verify inside.ply ⊂ ROI and outside.ply ∩ ROI ≈ ∅ (leakage stats).
2. Sample 5 uniformly-spaced frames, extract the in-ROI portion of each
   per-frame PLY, and save them so you can eyeball in MeshLab / viser.

Usage:
    python test_roi_split.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsplat.io_ply import import_splats
from gsplat.exporter import export_splats


ROI_PATH   = "/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/undistorted/static_dynamic_output/roi_bounds.npy"
INSIDE_PLY = "/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/undistorted/static_dynamic_output/inside.ply"
OUTSIDE_PLY = "/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/undistorted/static_dynamic_output/outside.ply"
PER_FRAME_DIR = "/data/shared/elaheh/final_4d_results/merge_ply_all_scenes/yehe_tech/ply_sequence_merged_35000_merged"
OUT_DIR     = "/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/undistorted/static_dynamic_output/roi_split_test"

N_FRAMES  = 300
N_SAMPLES = 5


def inside_roi(pts: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Boolean mask: True = point lies inside closed AABB roi=[min, max]."""
    return np.all((pts >= roi[0]) & (pts <= roi[1]), axis=1)


def stats(name: str, ply_path: str, roi: np.ndarray):
    means, *_ = import_splats(ply_path, device="cpu")
    mu = means.numpy()
    n = mu.shape[0]
    m = inside_roi(mu, roi)
    n_in, n_out = int(m.sum()), int((~m).sum())
    print(f"{name:12s}: {n:>8,} pts   inside_ROI={n_in:>8,} ({100*n_in/n:6.2f}%)   outside={n_out:>8,} ({100*n_out/n:6.2f}%)")
    return mu, m


def main():
    roi = np.load(ROI_PATH)
    print(f"ROI: min={roi[0]}  max={roi[1]}  size={roi[1]-roi[0]}\n")

    # -- Sanity: inside.ply ⊂ ROI, outside.ply ∩ ROI ≈ ∅ --
    print("== ROI leakage check ==")
    _, m_in = stats("inside.ply",  INSIDE_PLY,  roi)
    _, m_out = stats("outside.ply", OUTSIDE_PLY, roi)

    # -- Sample 5 frames uniformly and save only their in-ROI splats --
    os.makedirs(OUT_DIR, exist_ok=True)
    sample_idxs = np.linspace(0, N_FRAMES - 1, N_SAMPLES, dtype=int)
    print(f"\n== Per-frame split (saving in-ROI subset to {OUT_DIR}) ==")
    print(f"Sample frames: {sample_idxs.tolist()}")

    for fi in sample_idxs:
        ply_path = os.path.join(PER_FRAME_DIR, f"{fi:04d}.ply")
        if not os.path.exists(ply_path):
            print(f"  frame {fi:04d}: MISSING ({ply_path})")
            continue

        means, scales, quats, opacities, sh0, shN = import_splats(ply_path, device="cpu")
        mu = means.numpy()
        n = mu.shape[0]
        mask = inside_roi(mu, roi)
        n_in = int(mask.sum())
        idx = torch.from_numpy(np.where(mask)[0])

        out_path = os.path.join(OUT_DIR, f"frame_{fi:04d}_inroi.ply")
        export_splats(
            means=means[idx], scales=scales[idx], quats=quats[idx],
            opacities=opacities[idx], sh0=sh0[idx], shN=shN[idx],
            format="ply", save_to=out_path,
        )
        print(f"  frame {fi:04d}: {n:>8,} total  |  in-ROI {n_in:>8,} ({100*n_in/n:5.2f}%)  -> {out_path}")

    print("\nDone. Open the files under roi_split_test/ in your viewer.")


if __name__ == "__main__":
    main()
