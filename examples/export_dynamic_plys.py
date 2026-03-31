"""
Export per-frame dynamic PLY files from a 4DGS checkpoint.

Loads the checkpoint, applies the deformation field at each frame's timestamp,
and saves a PLY with the deformed Gaussians (pre-activation scales/opacities).

Usage:
    python export_dynamic_plys.py \
        --ckpt /path/to/ckpt_49999_rank0.pt \
        --out_dir /path/to/output/ply_sequences

    # Override num_frames or frame indexing if needed:
    python export_dynamic_plys.py \
        --ckpt /path/to/ckpt.pt \
        --out_dir /path/to/output \
        --num_frames 300 --frame_start 1 --frame_stride 1
"""

import argparse
import math
import os
import sys
import time

import torch

# Ensure local imports work (deformation module lives next to this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deformation import DeformationField, apply_deformation

from gsplat import export_splats


@torch.no_grad()
def load_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load a 4DGS checkpoint — mirrors viewer_4dgs.py logic."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", 0)
    config = ckpt.get("config", {})

    # Reconstruct splats
    splats = torch.nn.ParameterDict()
    for key, val in ckpt["splats"].items():
        splats[key] = torch.nn.Parameter(val.to(device), requires_grad=False)

    n_gs = splats["means"].shape[0]
    sh_dim = splats["sh0"].shape[1] + splats["shN"].shape[1]
    sh_degree = int(math.sqrt(sh_dim)) - 1

    print(f"  Step: {step}")
    print(f"  Gaussians: {n_gs:,}")
    print(f"  SH degree: {sh_degree}")

    # Reconstruct deformation field
    deform_field = None
    aabb = ckpt.get("aabb", None)

    if "deform_field" in ckpt:
        grid_res = config.get("deform_grid_resolution", 64)
        time_res = config.get("deform_time_resolution", 150)
        feat_dim = config.get("deform_feature_dim", 32)
        multires = config.get("deform_multires", [1, 2, 4, 8])
        net_width = config.get("deform_net_width", 128)
        net_depth = config.get("deform_net_depth", 0)
        enable_opacity = config.get("enable_opacity_deform", False)
        enable_sh = config.get("enable_sh_deform", False)
        time_pe_bands = config.get("deform_time_pe_bands", 0)
        raw_deform_sh = config.get("deform_sh_degree", -1)
        eff_deform_sh = raw_deform_sh if raw_deform_sh >= 0 else sh_degree

        if aabb is None:
            raise ValueError("Checkpoint has no 'aabb' — cannot reconstruct deformation field.")

        deform_field = DeformationField(
            grid_resolution=grid_res,
            time_resolution=time_res,
            feature_dim=feat_dim,
            multires=multires,
            net_width=net_width,
            defor_depth=net_depth,
            aabb=aabb,
            enable_opacity_deform=enable_opacity,
            enable_sh_deform=enable_sh,
            sh_degree=eff_deform_sh,
            time_pe_bands=time_pe_bands,
            act_xyz=config.get("deform_act_xyz", "relu"),
            act_rot=config.get("deform_act_rot", "relu"),
            act_scale=config.get("deform_act_scale", "relu"),
            act_sh=config.get("deform_act_sh", "relu"),
        ).to(device)
        deform_field.load_state_dict(ckpt["deform_field"])
        deform_field.eval()
        print(f"  Deformation: grid={grid_res}, time={time_res}, "
              f"width={net_width}, depth={net_depth}")
    else:
        raise ValueError("Checkpoint has no deformation field — nothing dynamic to export.")

    if aabb is not None:
        aabb = aabb.to(device)

    return splats, deform_field, aabb, config, step


@torch.no_grad()
def export_per_frame_plys(
    splats,
    deform_field,
    aabb,
    config,
    out_dir: str,
    num_frames: int,
    frame_start: int,
    frame_stride: int,
):
    """Export one PLY per frame with deformation baked in."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nExporting {num_frames} per-frame dynamic PLYs to {out_dir}")

    t0 = time.time()
    for frame_rank in range(num_frames):
        # Same timestamp normalization as training: t in [-0.5, 0.5]
        t = frame_rank / max(num_frames - 1, 1) - 0.5

        deltas = deform_field(splats["means"], t)
        means_d, quats_d, scales_d, opacs_d, colors_d = apply_deformation(
            splats, deltas, aabb=aabb
        )

        # Convert activated values back to pre-activation for PLY export
        scales_log = torch.log(scales_d)
        opacs_logit = torch.logit(opacs_d.clamp(1e-6, 1 - 1e-6))

        # Split colors [N, K, 3] -> sh0 [N, 1, 3] + shN [N, K-1, 3]
        sh0_d = colors_d[:, :1, :]
        shN_d = colors_d[:, 1:, :]

        frame_idx = frame_start + frame_rank * frame_stride

        export_splats(
            means=means_d,
            scales=scales_log,
            quats=quats_d,
            opacities=opacs_logit,
            sh0=sh0_d,
            shN=shN_d,
            format="ply",
            save_to=f"{out_dir}/frame_{frame_idx:06d}.ply",
        )

        if (frame_rank + 1) % 50 == 0 or frame_rank == num_frames - 1:
            elapsed = time.time() - t0
            print(f"  {frame_rank + 1}/{num_frames}  ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nDone: {num_frames} PLYs in {elapsed:.1f}s "
          f"({elapsed / num_frames * 1000:.1f}ms/frame)")
    print(f"Output: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export per-frame dynamic PLYs from a 4DGS checkpoint"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to 4DGS checkpoint (.pt file)",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Directory to save per-frame PLY files",
    )
    parser.add_argument(
        "--num_frames", type=int, default=0,
        help="Number of frames (0 = read from checkpoint config)",
    )
    parser.add_argument(
        "--frame_start", type=int, default=None,
        help="Starting frame index for naming (default: from config)",
    )
    parser.add_argument(
        "--frame_stride", type=int, default=None,
        help="Frame stride for naming (default: from config)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index (e.g. --gpu 3)",
    )
    args = parser.parse_args()

    device = "cuda"
    if args.gpu is not None:
        device = f"cuda:{args.gpu}"

    splats, deform_field, aabb, config, step = load_checkpoint(args.ckpt, device)

    num_frames = args.num_frames or config.get("num_frames", 1)
    frame_start = args.frame_start if args.frame_start is not None else config.get("frame_start", 0)
    frame_stride = args.frame_stride if args.frame_stride is not None else config.get("frame_stride", 1)

    print(f"  Num frames: {num_frames}")
    print(f"  Frame start: {frame_start}, stride: {frame_stride}")

    export_per_frame_plys(
        splats, deform_field, aabb, config,
        out_dir=args.out_dir,
        num_frames=num_frames,
        frame_start=frame_start,
        frame_stride=frame_stride,
    )


if __name__ == "__main__":
    main()
