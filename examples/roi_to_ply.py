#!/usr/bin/env python
"""
Render an AABB ROI as 12 red "lines" of anisotropic gaussians along its edges.

Each edge is tiled with N gaussians whose long axis is aligned to that edge,
so rendered output shows solid red lines outlining the ROI for overlay with a
trained model PLY.

Usage:
    python roi_to_ply.py --roi_bounds /.../roi_bounds.npy -o /.../roi_viz.ply
"""

import argparse
import os

import numpy as np
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def sh_dc_from_rgb(rgb):
    return ((np.asarray(rgb, dtype=np.float32) - 0.5) / C0)


def quat_from_x_to_dir(d):
    """Unit quaternion (w, x, y, z) rotating the +X axis to `d`.  Shape: d (3,) -> (4,)."""
    d = d / max(np.linalg.norm(d), 1e-12)
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dot = float(np.clip(np.dot(x, d), -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if dot < -0.999999:
        # 180° around any perpendicular axis (use Y)
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    axis = np.cross(x, d)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi_bounds", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--per_edge", type=int, default=60, help="gaussians per edge")
    ap.add_argument("--thickness", type=float, default=0.01,
                    help="linear scale of the line (perpendicular axes), world units")
    ap.add_argument("--sh_degree", type=int, default=3,
                    help="pad shN with zeros so it can drop into any deg>=1 viewer")
    ap.add_argument("--rgb", nargs=3, type=float, default=[1.0, 0.0, 0.0],
                    help="RGB color in [0,1]")
    args = ap.parse_args()

    roi = np.load(args.roi_bounds)
    assert roi.shape == (2, 3), roi.shape
    mn, mx = roi[0].astype(np.float32), roi[1].astype(np.float32)

    # 8 corners
    corners = np.stack(np.meshgrid(
        [mn[0], mx[0]], [mn[1], mx[1]], [mn[2], mx[2]], indexing="ij"
    ), -1).reshape(-1, 3).astype(np.float32)

    # 12 edges — pairs of corners differing in exactly one coordinate
    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(corners[i] != corners[j])) == 1:
                edges.append((corners[i], corners[j]))
    assert len(edges) == 12

    per_edge = args.per_edge
    thickness = args.thickness
    means_list, scales_list, quats_list = [], [], []

    for a, b in edges:
        direction = b - a
        length = float(np.linalg.norm(direction))
        # Sample centers so gaussians cover the full edge without gaps.
        # Each gaussian is length/per_edge long (half-extent = 0.5 * length/per_edge),
        # and scales are 1-sigma → use sigma = 0.5*segment as a visually reasonable size.
        centers = np.linspace(0.0, 1.0, per_edge, endpoint=True, dtype=np.float32)
        pts = a[None, :] * (1 - centers[:, None]) + b[None, :] * centers[:, None]
        means_list.append(pts)

        segment = length / max(per_edge - 1, 1)
        sigma_long = 0.5 * segment * 1.5        # slight overlap
        scales_list.append(np.tile([sigma_long, thickness, thickness],
                                   (per_edge, 1)).astype(np.float32))
        q = quat_from_x_to_dir(direction)
        quats_list.append(np.tile(q, (per_edge, 1)).astype(np.float32))

    means = np.concatenate(means_list, 0)
    scales_lin = np.concatenate(scales_list, 0)
    quats = np.concatenate(quats_list, 0)
    N = means.shape[0]
    print(f"ROI: {mn} → {mx}  |  edges={len(edges)}  per_edge={per_edge}  total={N:,}")

    # Log-space scales & logit opacity
    scales = np.log(np.clip(scales_lin, 1e-6, None)).astype(np.float32)
    opacities = np.full((N,), float(np.log(0.99 / 0.01)), dtype=np.float32)    # high opacity

    # SH: DC = chosen RGB; rest = 0
    sh_dc = np.tile(sh_dc_from_rgb(args.rgb), (N, 1)).astype(np.float32)
    total_K = (args.sh_degree + 1) ** 2
    shN_K = total_K - 1
    f_rest = np.zeros((N, 3 * shN_K), dtype=np.float32)

    # Build structured array
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    attrs += [f"f_dc_{i}" for i in range(3)]
    attrs += [f"f_rest_{i}" for i in range(3 * shN_K)]
    attrs += ["opacity"]
    attrs += [f"scale_{i}" for i in range(3)]
    attrs += [f"rot_{i}" for i in range(4)]
    dtype = [(a, "f4") for a in attrs]

    el = np.empty(N, dtype=dtype)
    el["x"], el["y"], el["z"] = means[:, 0], means[:, 1], means[:, 2]
    el["nx"] = 0; el["ny"] = 0; el["nz"] = 0
    for i in range(3):
        el[f"f_dc_{i}"] = sh_dc[:, i]
    for i in range(3 * shN_K):
        el[f"f_rest_{i}"] = f_rest[:, i]
    el["opacity"] = opacities
    for i in range(3):
        el[f"scale_{i}"] = scales[:, i]
    for i in range(4):
        el[f"rot_{i}"] = quats[:, i]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    PlyData([PlyElement.describe(el, "vertex")]).write(args.output)
    print(f"Saved {N:,} red-line gaussians → {args.output}")


if __name__ == "__main__":
    main()
