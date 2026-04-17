"""
Post-process dynamic PLY files: re-filter through masks at a new threshold.

Usage:
  python filter_dynamic_ply.py \
      --data_dir /data/shared/elaheh/4D_demo/completed_indoor/elaheh_tech/undistorted \
      --ply_dir /data/shared/elaheh/4D_demo/completed_indoor/elaheh_tech/results_perframe_fast/ply_frames/dynamic \
      --output_dir /data/shared/elaheh/4D_demo/completed_indoor/elaheh_tech/results_perframe_fast/ply_frames/dynamic_t100 \
      --threshold 1.0 \
      --data_factor 4 \
      --num_frames 20 --frame_start 0
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData, PlyElement
from datasets.colmap import Parser
from simple_trainer_perframe_masked import load_ply, load_mask, classify_gaussians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ply_dir", required=True, help="Input dynamic PLY folder")
    parser.add_argument("--output_dir", required=True, help="Output filtered PLY folder")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--mask_base_dir", default="tracking_experiment")
    parser.add_argument("--mask_subfolder", default="sam2")
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load parser for camera info
    colmap_parser = Parser(
        data_dir=args.data_dir, factor=args.data_factor,
        normalize=False, test_every=1000000, frame_num=args.frame_start,
    )
    cam_names = list(dict.fromkeys(os.path.dirname(n) for n in colmap_parser.image_names))
    num_cameras = len(cam_names)

    all_viewmats = torch.linalg.inv(
        torch.from_numpy(colmap_parser.camtoworlds).float().to(device))
    all_Ks = torch.stack([
        torch.from_numpy(colmap_parser.Ks_dict[colmap_parser.camera_ids[i]].copy()).float()
        for i in range(num_cameras)
    ]).to(device)

    widths, heights = [], []
    for ci in range(num_cameras):
        w, h = list(colmap_parser.imsize_dict.values())[ci] if ci < len(colmap_parser.imsize_dict) else list(colmap_parser.imsize_dict.values())[0]
        widths.append(w)
        heights.append(h)

    mask_dir = os.path.join(args.data_dir, args.mask_base_dir)

    print(f"Filtering {args.num_frames} frames at threshold={args.threshold:.0%}")
    print(f"Input:  {args.ply_dir}")
    print(f"Output: {args.output_dir}")
    print()

    for fi in range(args.num_frames):
        frame_idx = args.frame_start + fi
        ply_path = os.path.join(args.ply_dir, f"{frame_idx:06d}.ply")

        if not os.path.exists(ply_path):
            print(f"Frame {frame_idx}: PLY not found, skipping")
            continue

        # Load as ParameterDict for classification
        splats = load_ply(ply_path, device=device)
        n_before = splats["means"].shape[0]

        # Load masks for this frame
        masks = []
        for ci in range(num_cameras):
            cam_name = cam_names[ci]
            mask = load_mask(mask_dir, os.path.join(cam_name, args.mask_subfolder),
                             frame_idx, factor=args.data_factor, device=device)
            masks.append(mask)

        # Classify
        keep_mask = classify_gaussians(
            splats, masks, all_viewmats, all_Ks,
            widths, heights, device=device, threshold=args.threshold,
        )

        # Filter original PLY directly (preserves exact format)
        plydata = PlyData.read(ply_path)
        v = plydata["vertex"]
        keep_np = keep_mask.cpu().numpy()
        filtered = v.data[keep_np]

        out_path = os.path.join(args.output_dir, f"{frame_idx:06d}.ply")
        PlyData([PlyElement.describe(filtered, "vertex")]).write(out_path)

        n_after = keep_mask.sum().item()
        print(f"Frame {frame_idx}: {n_before:,} → {n_after:,} "
              f"(removed {n_before - n_after:,})")

    print(f"\nDone! Filtered PLYs at {args.output_dir}")


if __name__ == "__main__":
    main()
