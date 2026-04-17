"""
Crop PLY sequence to a sphere around scene center.

Computes sphere from camera positions, multiplies by a factor,
and removes Gaussians outside. Applies same radius to all frames.

Usage:
    python examples/crop_sphere_ply.py \
        --data_dir /path/to/undistorted \
        --ply_dir /path/to/ply_frames/dynamic \
        --output_dir /path/to/ply_frames/dynamic_cropped \
        --factor 0.8
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from datasets.colmap import Parser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="COLMAP dataset dir")
    parser.add_argument("--ply_dir", required=True, help="Input PLY folder")
    parser.add_argument("--output_dir", required=True, help="Output cropped PLY folder")
    parser.add_argument("--factor", type=float, default=0.8, help="Radius multiplier (default 0.8)")
    parser.add_argument("--data_factor", type=int, default=1, help="Image downscale factor for Parser")
    parser.add_argument("--frame_start", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load camera poses to compute scene center and radius
    colmap_parser = Parser(
        data_dir=args.data_dir, factor=args.data_factor,
        normalize=False, test_every=1000000, frame_num=args.frame_start,
    )
    cam_positions = colmap_parser.camtoworlds[:, :3, 3]  # [C, 3]
    center = cam_positions.mean(axis=0)
    distances = np.linalg.norm(cam_positions - center, axis=1)
    max_dist = distances.max()
    radius = max_dist * args.factor

    print(f"Scene center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"Max camera distance: {max_dist:.2f}")
    print(f"Crop radius: {radius:.2f} (factor={args.factor})")
    print()

    # Process all PLY files
    ply_files = sorted(f for f in os.listdir(args.ply_dir) if f.endswith('.ply'))
    print(f"Processing {len(ply_files)} PLY files...")

    center_t = torch.tensor(center, dtype=torch.float32)

    for fname in ply_files:
        ply_path = os.path.join(args.ply_dir, fname)
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        n_before = v.count

        # Get positions
        xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
        dists = np.linalg.norm(xyz - center, axis=1)
        keep = dists <= radius

        n_after = keep.sum()
        filtered = v.data[keep]

        out_path = os.path.join(args.output_dir, fname)
        PlyData([PlyElement.describe(filtered, "vertex")]).write(out_path)

        print(f"  {fname}: {n_before:,} → {n_after:,} (removed {n_before - n_after:,})")

    print(f"\nDone! Cropped PLYs at {args.output_dir}")


if __name__ == "__main__":
    main()
