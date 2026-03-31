"""
4D Gaussian Splatting Viewer.

Extends simple_viewer.py with a time slider for scrubbing through the
temporal deformation field. Loads a 4DGS checkpoint (splats + deform_field)
and renders interactively using gsplat.rasterization().

Usage:
  python simple_viewer_4dgs.py \\
      --ckpt results/scene_4dgs/ckpts/ckpt_29999_rank0.pt \\
      --num_frames 50 \\
      --port 8082
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import viser

from gsplat.rendering import rasterization
from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState

from deformation import DeformationField, apply_deformation


def main(args):
    torch.manual_seed(42)
    device = torch.device("cuda", 0)

    # ---- Load checkpoint ----
    print(f"[4DGS Viewer] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    splats_state = ckpt["splats"]
    aabb = ckpt.get("aabb", None)
    if aabb is not None:
        aabb = aabb.to(device)
        print(f"[4DGS Viewer] AABB half-extent: {((aabb[1]-aabb[0])/2).tolist()}")

    # Reconstruct splat parameters
    means = splats_state["means"].to(device)       # [N, 3]
    log_scales = splats_state["scales"].to(device)  # [N, 3] log-space
    raw_quats = splats_state["quats"].to(device)    # [N, 4]
    raw_opacs = splats_state["opacities"].to(device)  # [N]
    sh0 = splats_state["sh0"].to(device)            # [N, 1, 3]
    shN = splats_state["shN"].to(device)            # [N, K-1, 3]
    sh_degree = int(math.sqrt(sh0.shape[1] + shN.shape[1]) - 1)

    print(f"[4DGS Viewer] Gaussians: {len(means)}, SH degree: {sh_degree}")

    # ---- Load deformation field (if present in checkpoint) ----
    deform_field = None
    if "deform_field" in ckpt:
        cfg = ckpt.get("config", {})
        deform_field = DeformationField(
            grid_resolution=cfg.get("deform_grid_resolution", 64),
            time_resolution=cfg.get("deform_time_resolution", 25),
            feature_dim=cfg.get("deform_feature_dim", 16),
            multires=cfg.get("deform_multires", [1, 2, 4, 8]),
            net_width=cfg.get("deform_net_width", 64),
            defor_depth=cfg.get("deform_net_depth", 1),
            enable_opacity_deform=cfg.get("enable_opacity_deform", False),
            enable_sh_deform=cfg.get("enable_sh_deform", False),
            sh_degree=sh_degree,
        ).to(device)
        deform_field.load_state_dict(ckpt["deform_field"])
        deform_field.eval()
        print("[4DGS Viewer] Deformation field loaded.")
    else:
        print("[4DGS Viewer] No deformation field in checkpoint — static 3DGS mode.")

    # Wrap splats in a dict for apply_deformation
    splats = {
        "means": means,
        "scales": log_scales,
        "quats": raw_quats,
        "opacities": raw_opacs,
        "sh0": sh0,
        "shN": shN,
    }

    # ---- Viewer render function ----
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)

        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = torch.linalg.inv(c2w)

        # Get current time from slider
        t = getattr(render_tab_state, "time", 0.0)

        # Apply deformation if field is loaded
        if deform_field is not None:
            deltas = deform_field(means.detach(), t)
            means_d, quats_d, scales_d, opacs_d, colors_d = apply_deformation(
                splats, deltas, aabb=aabb
            )
        else:
            means_d = means
            quats_d = F.normalize(raw_quats, p=2, dim=-1)
            scales_d = torch.exp(log_scales)
            opacs_d = torch.sigmoid(raw_opacs)
            colors_d = torch.cat([sh0, shN], dim=1)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means=means_d,
            quats=quats_d,
            scales=scales_d,
            opacities=opacs_d,
            colors=colors_d,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(
                getattr(render_tab_state, "max_sh_degree", sh_degree),
                sh_degree,
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor(
                [render_tab_state.backgrounds], device=device
            ) / 255.0,
            render_mode=RENDER_MODE_MAP.get(render_tab_state.render_mode, "RGB"),
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
        )

        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth = render_colors[0, ..., :1]
            if render_tab_state.normalize_nearfar:
                near = render_tab_state.near_plane
                far = render_tab_state.far_plane
            else:
                near = depth.min()
                far = depth.max()
            depth_norm = (depth - near) / (far - near + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            return apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            return apply_float_colormap(
                render_alphas[0, ..., :1], render_tab_state.colormap
            ).cpu().numpy()

    # ---- Start viewer ----
    server = viser.ViserServer(port=args.port, verbose=False)

    # Add time slider if deformation field is present
    if deform_field is not None:
        time_slider = server.gui.add_slider(
            label="Time",
            min=0.0,
            max=1.0,
            step=1.0 / max(args.num_frames - 1, 1),
            initial_value=0.0,
        )

        @time_slider.on_update
        def _on_time_update(_):
            pass  # GsplatViewer polls render_tab_state.time below

        # Monkey-patch render_tab_state to expose time from slider
        # (GsplatRenderTabState doesn't natively have a time field)
        class _TimeProxy:
            """Forwards attribute access to the wrapped state but adds .time."""
            def __init__(self, wrapped):
                object.__setattr__(self, "_w", wrapped)
            def __getattr__(self, name):
                if name == "time":
                    return time_slider.value
                return getattr(object.__getattribute__(self, "_w"), name)
            def __setattr__(self, name, value):
                if name == "_w":
                    object.__setattr__(self, "_w", value)
                else:
                    setattr(object.__getattribute__(self, "_w"), name, value)

        _original_render_fn = viewer_render_fn

        def _wrapped_render_fn(camera_state, render_tab_state):
            return _original_render_fn(camera_state, _TimeProxy(render_tab_state))

        viewer_render_fn_final = _wrapped_render_fn
        print(f"[4DGS Viewer] Time slider: 0.0 – 1.0 ({args.num_frames} frames)")
    else:
        viewer_render_fn_final = viewer_render_fn

    output_dir = Path(os.path.dirname(args.ckpt))
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn_final,
        output_dir=output_dir,
        mode="rendering",
    )

    print(f"[4DGS Viewer] Running at http://localhost:{args.port}")
    print("[4DGS Viewer] Ctrl+C to exit.")
    time.sleep(100_000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4DGS interactive viewer")
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to 4DGS checkpoint (.pt) from simple_trainer_4dgs.py",
    )
    parser.add_argument(
        "--port", type=int, default=8082,
        help="Port for the viser viewer server",
    )
    parser.add_argument(
        "--num_frames", type=int, default=50,
        help="Total number of frames in the scene (used for time slider step size)",
    )
    args = parser.parse_args()
    main(args)
