"""
Quick diagnostic: Load 4DGS checkpoint, render a few frames, and print
deformation statistics to help identify rendering issues.
"""
import sys, os, math, argparse
import torch
import numpy as np
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))
from deformation import DeformationField, apply_deformation
from gsplat.rendering import rasterization
from gsplat.io_ply import import_splats


def load_ckpt(path, device="cuda"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    splats = torch.nn.ParameterDict()
    for k, v in ckpt["splats"].items():
        splats[k] = torch.nn.Parameter(v.to(device), requires_grad=False)
    config = ckpt.get("config", {})
    aabb = ckpt.get("aabb", None)
    if aabb is not None:
        aabb = aabb.to(device)

    sh_dim = splats["sh0"].shape[1] + splats["shN"].shape[1]
    sh_degree = int(math.sqrt(sh_dim)) - 1

    deform_field = None
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
        deform_field = DeformationField(
            grid_resolution=grid_res, time_resolution=time_res,
            feature_dim=feat_dim, multires=multires,
            net_width=net_width, defor_depth=net_depth,
            aabb=aabb, enable_opacity_deform=enable_opacity,
            enable_sh_deform=enable_sh, sh_degree=eff_deform_sh,
            time_pe_bands=time_pe_bands,
            act_xyz=config.get("deform_act_xyz", "relu"),
            act_rot=config.get("deform_act_rot", "relu"),
            act_scale=config.get("deform_act_scale", "relu"),
            act_sh=config.get("deform_act_sh", "relu"),
        ).to(device)
        deform_field.load_state_dict(ckpt["deform_field"])
        deform_field.eval()
    return splats, deform_field, aabb, config, sh_degree


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--out", default="/tmp/debug_4dgs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device
    splats, deform_field, aabb, config, sh_degree = load_ckpt(args.ckpt, device)

    N = splats["means"].shape[0]
    num_frames = config.get("num_frames", 150)

    print(f"=== 4DGS Diagnostic ===")
    print(f"Dynamic Gaussians: {N:,}")
    print(f"SH degree: {sh_degree}")
    print(f"Num frames: {num_frames}")
    print(f"AABB: {aabb[0].tolist()} → {aabb[1].tolist()}" if aabb is not None else "AABB: None")
    print(f"Deformation field: {'Yes' if deform_field is not None else 'No'}")

    # 1) Check canonical means bounding box
    means = splats["means"]
    print(f"\n--- Canonical means ---")
    print(f"  min: {means.min(0).values.tolist()}")
    print(f"  max: {means.max(0).values.tolist()}")
    print(f"  std: {means.std(0).tolist()}")

    # 2) Check if means are inside the AABB
    if aabb is not None:
        inside = ((means >= aabb[0]) & (means <= aabb[1])).all(dim=-1)
        print(f"  Inside AABB: {inside.sum().item()}/{N} ({inside.float().mean()*100:.1f}%)")

    # 3) Sample deformation at several timestamps
    if deform_field is not None:
        test_times = [-0.5, -0.25, 0.0, 0.25, 0.499]
        print(f"\n--- Deformation stats at various timestamps ---")
        for t in test_times:
            deltas = deform_field(means, t)

            # delta_xyz stats (in normalized space, before AABB scaling)
            dx = deltas.delta_xyz
            print(f"  t={t:+.3f}: delta_xyz norm mean={dx.norm(dim=-1).mean():.6f}, "
                  f"max={dx.norm(dim=-1).max():.6f}")

            # If AABB provided, show world-space displacement
            if aabb is not None:
                half = (aabb[1] - aabb[0]) / 2.0
                dx_world = dx * half
                print(f"           delta_xyz_world mean={dx_world.norm(dim=-1).mean():.6f}, "
                      f"max={dx_world.norm(dim=-1).max():.6f}")

            # delta_rot stats
            dr = deltas.delta_rot
            print(f"           delta_rot norm mean={dr.norm(dim=-1).mean():.6f}, "
                  f"max={dr.norm(dim=-1).max():.6f}")

            # delta_scale stats
            ds = deltas.delta_scale
            print(f"           delta_scale norm mean={ds.norm(dim=-1).mean():.6f}, "
                  f"max={ds.norm(dim=-1).max():.6f}")

        # 4) Check deformed means at t=-0.5 (should be ≈ canonical if constraint worked)
        deltas_t0 = deform_field(means, -0.5)
        m_d, q_d, s_d, o_d, c_d = apply_deformation(splats, deltas_t0, aabb=aabb)
        diff_at_t0 = (m_d - means).norm(dim=-1)
        print(f"\n--- Identity constraint check at t=-0.5 ---")
        print(f"  ||deformed_mean - canonical_mean|| : mean={diff_at_t0.mean():.6f}, "
              f"max={diff_at_t0.max():.6f}")

        # 5) Check deformation range: max displacement across all frames
        max_disp = 0.0
        for fi in range(num_frames):
            t = fi / max(num_frames - 1, 1) - 0.5
            d = deform_field(means, t)
            _, _, _, _, _ = apply_deformation(splats, d, aabb=aabb)
            # Use means + delta (world space)
            if aabb is not None:
                half = (aabb[1] - aabb[0]) / 2.0
                disp = (d.delta_xyz * half).norm(dim=-1).max().item()
            else:
                disp = d.delta_xyz.norm(dim=-1).max().item()
            max_disp = max(max_disp, disp)
        print(f"\n--- Max displacement across all {num_frames} frames ---")
        print(f"  {max_disp:.6f} world units")

    # 6) Print opacity and scale statistics
    scales = torch.exp(splats["scales"])
    opacs = torch.sigmoid(splats["opacities"])
    print(f"\n--- Scale/opacity stats ---")
    print(f"  scales: mean={scales.mean():.6f}, max={scales.max():.6f}, min={scales.min():.6f}")
    print(f"  opacities: mean={opacs.mean():.4f}, >0.5: {(opacs>0.5).sum().item()}/{N}")

    # 7) Static PLY info
    static_path = config.get("static_ply_path", None)
    if static_path and os.path.exists(static_path):
        s_means, s_scales, s_quats, s_opacs, s_sh0, s_shN = import_splats(static_path, device)
        print(f"\n--- Static PLY ---")
        print(f"  N: {s_means.shape[0]:,}")
        print(f"  SH bands: sh0={s_sh0.shape}, shN={s_shN.shape}")
        print(f"  means min: {s_means.min(0).values.tolist()}")
        print(f"  means max: {s_means.max(0).values.tolist()}")

    print(f"\nDone! Check {args.out}/ for any debug images.")


if __name__ == "__main__":
    main()
