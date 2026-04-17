#!/usr/bin/env python
"""
Voxel-based floater removal for Gaussian splats.

Idea: divide scene into large voxels. Remove splats that are:
  (a) inside a voxel containing only a few splats (sparse/isolated floaters), OR
  (b) abnormally large compared to their neighborhood (oversized floaters).

Good for killing chunks that hang inside the camera rig after simplification.

Usage:
    python voxel_clean_gaussians.py --input <in.ply> [-o <out.ply>]
                                    [--voxel_size 0.5] [--min_count 20]
                                    [--max_scale_factor 8.0]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsplat.io_ply import import_splats
from gsplat.exporter import export_splats


def voxel_clean(
    mu: np.ndarray,
    max_scale: np.ndarray,
    voxel_size: float,
    min_count: int,
    max_scale_factor: float,
) -> np.ndarray:
    """Return boolean keep-mask. mu: (N,3), max_scale: (N,) linear scale max."""
    N = mu.shape[0]

    # Voxel index per point
    vox = np.floor(mu / voxel_size).astype(np.int64)

    # Hash voxels to unique IDs and count splats per voxel
    _, inverse, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    per_point_count = counts[inverse]
    keep_dense = per_point_count >= min_count

    # Oversized filter: globally vs median, and locally per voxel
    med_global = float(np.median(max_scale))
    keep_global_scale = max_scale <= max_scale_factor * med_global

    # Per-voxel median scale (only where voxel has enough points to trust median)
    # Use np.bincount to compute per-voxel median approximation via mean of logs
    # (Exact median per voxel is expensive; mean-of-log is a robust proxy.)
    log_s = np.log(np.maximum(max_scale, 1e-12))
    sums = np.bincount(inverse, weights=log_s, minlength=counts.size)
    voxel_mean_log = sums / np.maximum(counts, 1)
    voxel_typical = np.exp(voxel_mean_log)[inverse]
    keep_local_scale = max_scale <= max_scale_factor * voxel_typical

    keep = keep_dense & keep_global_scale & keep_local_scale

    n_sparse = int((~keep_dense).sum())
    n_big_g = int((~keep_global_scale).sum())
    n_big_l = int((~keep_local_scale).sum())
    n_removed = int((~keep).sum())

    print(f"Voxel cleaning {N} splats  (voxel_size={voxel_size:g})")
    print(f"  Unique voxels           : {counts.size}")
    print(f"  Median splats / voxel   : {np.median(counts):.1f}")
    print(f"  Sparse voxels (<{min_count})   : "
          f"{int((counts < min_count).sum())}  → {n_sparse} splats removed")
    print(f"  Oversized (global > {max_scale_factor}x med): {n_big_g} removed")
    print(f"  Oversized (local  > {max_scale_factor}x vox): {n_big_l} removed")
    print(f"  Total removed           : {n_removed} ({100*n_removed/N:.1f}%)")
    print(f"  Remaining               : {N - n_removed}")
    return keep


def main():
    ap = argparse.ArgumentParser(
        description="Voxel-based floater removal for Gaussian splats."
    )
    ap.add_argument("--input", required=True, help="Input PLY")
    ap.add_argument("-o", "--output", default=None, help="Output PLY")
    ap.add_argument(
        "--voxel_size", type=float, default=None,
        help="Voxel edge length (world units). Default: 2%% of scene diagonal."
    )
    ap.add_argument(
        "--min_count", type=int, default=20,
        help="Remove all splats in voxels with fewer than this many splats."
    )
    ap.add_argument(
        "--max_scale_factor", type=float, default=8.0,
        help="Remove splats whose max-axis scale exceeds factor × median "
             "(global and per-voxel)."
    )
    args = ap.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_voxclean{ext}"

    print(f"Loading: {args.input}")
    means, scales, quats, opacities, sh0, shN = import_splats(args.input, device="cpu")
    N = means.shape[0]
    print(f"Loaded {N} splats")

    mu = means.numpy().astype(np.float32)
    # linear max scale per splat
    max_scale = np.exp(np.clip(scales.numpy(), -30, 30)).max(axis=1).astype(np.float32)

    if args.voxel_size is None:
        extent = mu.max(axis=0) - mu.min(axis=0)
        diag = float(np.linalg.norm(extent))
        args.voxel_size = max(diag * 0.02, 1e-6)
        print(f"Auto voxel_size = {args.voxel_size:.4f}  (scene diag {diag:.3f})")

    keep = voxel_clean(
        mu, max_scale,
        voxel_size=args.voxel_size,
        min_count=args.min_count,
        max_scale_factor=args.max_scale_factor,
    )
    idx = torch.from_numpy(np.where(keep)[0])

    print(f"\nSaving: {args.output}")
    export_splats(
        means=means[idx], scales=scales[idx], quats=quats[idx],
        opacities=opacities[idx], sh0=sh0[idx], shN=shN[idx],
        format="ply", save_to=args.output,
    )
    print("Done!")


if __name__ == "__main__":
    main()
