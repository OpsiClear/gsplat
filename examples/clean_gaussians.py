"""Remove noisy/outlier Gaussians before simplification.

Removes:
1. Low opacity Gaussians (ghost splats)
2. Spatially isolated Gaussians (floaters far from neighbors)
3. Oversized Gaussians (scale outliers)

Usage:
    python clean_gaussians.py --input inside.ply --output inside_clean.ply
"""

import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsplat.exporter import load_ply_gaussian, export_splats


def _connected_components(pts, k=10, radius_mult=2.0):
    """Union-find connected components on KNN graph with adaptive radius."""
    N = len(pts)
    tree = cKDTree(pts)
    dists, indices = tree.query(pts, k=k + 1, workers=-1)
    median_dist = np.median(dists[:, 1])
    radius = radius_mult * median_dist
    print(f"  Median NN dist: {median_dist:.6f}, connection radius: {radius:.6f}")

    parent = np.arange(N, dtype=np.int64)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(N):
        for j_idx in range(1, k + 1):
            if dists[i, j_idx] <= radius:
                union(i, indices[i, j_idx])

    labels = np.array([find(i) for i in range(N)])
    unique_labels, counts = np.unique(labels, return_counts=True)
    return labels, unique_labels, counts


def clean_gaussians(
    means, scales, quats, opacities, sh0, shN,
    min_opacity=0.05,
    isolation_k=8,
    isolation_factor=5.0,
    max_scale_factor=10.0,
    max_aspect_ratio=20.0,
    min_cluster_frac=0.01,
    min_cluster_size=50,
    cc_k=10,
    cc_radius_mult=2.0,
):
    """Remove noisy Gaussians. All inputs are raw (log-scale, logit-opacity)."""
    N = len(means)
    opacity = torch.sigmoid(opacities).flatten().numpy()
    scale = torch.exp(scales).numpy()
    max_scale = scale.max(axis=1)
    min_scale = scale.min(axis=1)

    # 1. Opacity filter
    keep_opa = opacity >= min_opacity

    # 2. Isolation filter (KNN distance)
    pts = means.numpy()
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=isolation_k + 1)
    nn_dist = dists[:, 1:].mean(axis=1)
    med_nn = np.median(nn_dist)
    keep_iso = nn_dist <= isolation_factor * med_nn

    # 3. Scale filter
    med_scale = np.median(max_scale)
    keep_scale = max_scale <= max_scale_factor * med_scale

    # 4. Needle filter (aspect ratio = max_scale / min_scale)
    aspect = max_scale / np.maximum(min_scale, 1e-12)
    keep_needle = aspect <= max_aspect_ratio

    # 5. Connected component filter — remove small floating clusters
    print("  Finding connected components...")
    labels, unique_labels, counts = _connected_components(
        pts, k=cc_k, radius_mult=cc_radius_mult)
    order = np.argsort(-counts)
    unique_labels = unique_labels[order]
    counts = counts[order]
    threshold = max(min_cluster_size, int(min_cluster_frac * N))
    keep_labels = set(unique_labels[counts >= threshold])
    keep_cc = np.array([labels[i] in keep_labels for i in range(N)])
    n_clusters_kept = len(keep_labels)
    n_clusters_total = len(counts)
    print(f"  {n_clusters_total} clusters found, keeping {n_clusters_kept} (>= {threshold} pts)")
    for i in range(min(5, len(counts))):
        print(f"    Cluster {i}: {counts[i]} pts ({100*counts[i]/N:.1f}%)")

    keep = keep_opa & keep_iso & keep_scale & keep_needle & keep_cc
    n_removed = N - keep.sum()

    print(f"Cleaning {N} Gaussians:")
    print(f"  Low opacity (<{min_opacity}): {(~keep_opa).sum()} removed")
    print(f"  Isolated (>{isolation_factor}x median NN): {(~keep_iso).sum()} removed")
    print(f"  Oversized (>{max_scale_factor}x median scale): {(~keep_scale).sum()} removed")
    print(f"  Needles (aspect ratio >{max_aspect_ratio}): {(~keep_needle).sum()} removed")
    print(f"  Small clusters (<{threshold} pts): {(~keep_cc).sum()} removed")
    print(f"  Total removed: {n_removed} ({100*n_removed/N:.1f}%)")
    print(f"  Remaining: {keep.sum()}")

    idx = torch.from_numpy(np.where(keep)[0])
    return (
        means[idx], scales[idx], quats[idx], opacities[idx],
        sh0[idx], shN[idx],
    )


def main():
    ap = argparse.ArgumentParser(description="Clean noisy Gaussians from PLY")
    ap.add_argument("--input", required=True, help="Input PLY file")
    ap.add_argument("--output", default=None, help="Output PLY file (default: input_clean.ply)")
    ap.add_argument("--min_opacity", type=float, default=0.05)
    ap.add_argument("--isolation_factor", type=float, default=5.0)
    ap.add_argument("--max_scale_factor", type=float, default=10.0)
    ap.add_argument("--max_aspect_ratio", type=float, default=20.0,
                    help="Remove needle Gaussians with aspect ratio above this")
    ap.add_argument("--min_cluster_frac", type=float, default=0.01,
                    help="Remove clusters smaller than this fraction of total")
    ap.add_argument("--min_cluster_size", type=int, default=50,
                    help="Absolute minimum cluster size to keep")
    ap.add_argument("--cc_k", type=int, default=10,
                    help="KNN neighbors for connected component graph")
    ap.add_argument("--cc_radius_mult", type=float, default=2.0,
                    help="Connection radius = mult * median_NN_dist")
    args = ap.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_clean{ext}"

    print(f"Loading: {args.input}")
    means, scales, quats, opacities, sh0, shN = load_ply_gaussian(args.input, device="cpu")

    means, scales, quats, opacities, sh0, shN = clean_gaussians(
        means, scales, quats, opacities, sh0, shN,
        min_opacity=args.min_opacity,
        isolation_factor=args.isolation_factor,
        max_scale_factor=args.max_scale_factor,
        max_aspect_ratio=args.max_aspect_ratio,
        min_cluster_frac=args.min_cluster_frac,
        min_cluster_size=args.min_cluster_size,
        cc_k=args.cc_k,
        cc_radius_mult=args.cc_radius_mult,
    )

    print(f"Saving: {args.output}")
    export_splats(
        means=means, scales=scales, quats=quats,
        opacities=opacities, sh0=sh0, shN=shN,
        format="ply", save_to=args.output,
    )
    print("Done!")


if __name__ == "__main__":
    main()
