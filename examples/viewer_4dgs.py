"""
4D Gaussian Splatting (4DGS) Interactive Viewer

Loads a 4DGS checkpoint (canonical Gaussians + HexPlane deformation field)
and renders interactively with time scrubbing via viser + nerfview.

Features:
  - Time slider + auto-play for frame scrubbing
  - Deformation field applied per-frame
  - Spatial filtering (percentile-based outlier removal)
  - SH degree control, near/far plane, background color
  - Depth and alpha render modes

Usage:
    python viewer_4dgs.py --ckpt /path/to/ckpt_59999_rank0.pt --port 8080

    # Override num_frames if not in checkpoint config:
    python viewer_4dgs.py --ckpt /path/to/ckpt.pt --num-frames 300
"""

import argparse
import math
import os
import sys
import threading
import time

import numpy as np
import torch
import viser

import nerfview
from nerfview import CameraState

from gsplat.rendering import rasterization
from gsplat.io_ply import import_splats

# Import deformation modules from the local package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deformation import DeformationField, apply_deformation


@torch.no_grad()
def load_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load a 4DGS checkpoint and reconstruct the model.

    Returns:
        splats: nn.ParameterDict with canonical Gaussian parameters
        deform_field: DeformationField (or None if no deformation)
        config: dict of training config from checkpoint
        step: training step at which checkpoint was saved
    """
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    step = ckpt.get("step", 0)
    config = ckpt.get("config", {})

    # Reconstruct splats as ParameterDict
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
    if "deform_field" in ckpt:
        # Get deformation config from training config
        grid_res = config.get("deform_grid_resolution", 64)
        time_res = config.get("deform_time_resolution", 150)
        feat_dim = config.get("deform_feature_dim", 32)
        multires = config.get("deform_multires", [1, 2, 4, 8])
        net_width = config.get("deform_net_width", 128)
        net_depth = config.get("deform_net_depth", 0)
        enable_opacity = config.get("enable_opacity_deform", False)
        enable_sh = config.get("enable_sh_deform", False)
        time_pe_bands = config.get("deform_time_pe_bands", 0)
        # deform_sh_degree may differ from rendering sh_degree (-1 means use sh_degree)
        raw_deform_sh = config.get("deform_sh_degree", -1)
        eff_deform_sh = raw_deform_sh if raw_deform_sh >= 0 else sh_degree

        # AABB: use saved checkpoint AABB so HexPlane normalization matches training exactly.
        # Do NOT recompute from splat means — densification changes mean distribution,
        # giving a different aabb → wrong grid coordinates → deformation sends face off-screen.
        aabb = ckpt.get("aabb", None)
        if aabb is None:
            raise ValueError("Checkpoint has no 'aabb' key — cannot reconstruct deformation field correctly.")

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
              f"width={net_width}, depth={net_depth}, multires={multires}")
        print(f"  AABB (from ckpt): {aabb[0].tolist()} → {aabb[1].tolist()}")
    else:
        print("  Deformation: None (static model)")
        aabb = ckpt.get("aabb", None)

    if aabb is not None:
        aabb = aabb.to(device)

    return splats, deform_field, aabb, config, step, sh_degree


class Viewer4DGS:
    """Interactive 4DGS viewer with time scrubbing."""

    def __init__(
        self,
        ckpt_path: str,
        port: int = 8080,
        device: str = "cuda",
        num_frames_override: int = 0,
        precompute: bool = False,
        static_ply_path: str = None,
    ):
        self.device = device

        # Load model
        self.splats, self.deform_field, self.aabb, self.config, self.step, self.sh_degree = \
            load_checkpoint(ckpt_path, device)

        # Determine number of frames
        self.num_frames = num_frames_override or self.config.get("num_frames", 1)
        print(f"  Num frames: {self.num_frames}")

        # Load static (background) PLY — explicit arg takes priority, else auto-load from config
        self.static_splats = None
        if static_ply_path is None:
            static_ply_path = self.config.get("static_ply_path", None)
            if static_ply_path is not None:
                print(f"  Static PLY (from checkpoint config): {static_ply_path}")
        if static_ply_path is not None and not os.path.exists(static_ply_path):
            print(f"  Warning: static PLY not found at {static_ply_path}, skipping")
            static_ply_path = None
        if static_ply_path is not None:
            print(f"  Loading static PLY: {static_ply_path}")
            s_means, s_scales, s_quats, s_opacs, s_sh0, s_shN = import_splats(
                static_ply_path, device
            )
            # Static Gaussians always render at SH=0 (DC only, view-independent).
            # Zero out all higher-order bands — shape matches dynamic colors for concatenation.
            target_shN = (self.sh_degree + 1) ** 2 - 1
            self.static_splats = {
                "means":     s_means,
                "scales":    torch.exp(s_scales),
                "quats":     s_quats,
                "opacities": torch.sigmoid(s_opacs),
                "colors":    torch.cat([s_sh0,
                                        torch.zeros(s_sh0.shape[0], target_shN, 3,
                                                    device=device)], dim=1),
            }
            print(f"  Static Gaussians: {s_means.shape[0]:,} (SH=0/DC-only)")

        # Spatial filtering: compute distances for outlier removal.
        # Use combined (static + dynamic) extent when static PLY is loaded so the
        # 0.85 slider is relative to scene scale, not just the dynamic model's radius.
        # This prevents face Gaussians near the scene boundary from being clipped.
        pts_np = self.splats["means"].data.cpu().numpy()
        if self.static_splats is not None:
            all_pts = np.concatenate(
                [pts_np, self.static_splats["means"].cpu().numpy()], axis=0
            )
        else:
            all_pts = pts_np
        center = all_pts.mean(axis=0)
        self.distances = torch.from_numpy(
            np.linalg.norm(pts_np - center, axis=1)
        ).float().to(device)
        self.max_radius = float(
            np.percentile(np.linalg.norm(all_pts - center, axis=1), 99)
        )

        # Animation state
        self.current_frame = 0
        self.auto_play = False
        self.play_fps = 15.0
        self._last_frame_time = time.time()

        # Deformation cache: frame_idx -> (means, quats, scales, opacities, colors).
        # Keyed by frame only — spatial filtering is applied on top as a cheap post-process.
        # Filled lazily on first visit; never evicted so second loop onwards is instant.
        self._frame_cache: dict = {}

        # Render lock: prevents overlapping GPU renders that cause checkerboard artifacts
        self._render_lock = threading.Lock()
        self._last_rendered_img = None  # fallback when render is skipped
        self._render_busy = False  # flag for autoplay to check
        self._frame_loading = False  # True while deformation is being computed (cache miss)

        # Viser server + UI
        self.server = viser.ViserServer(port=port, verbose=False)
        self._setup_ui()

        # Initialize viewer camera: point at dynamic Gaussian centroid.
        _centroid = self.splats["means"].data.mean(0).cpu().numpy()  # [3]
        _dyn_dists = np.linalg.norm(pts_np - _centroid, axis=1)
        _extent = float(np.percentile(_dyn_dists, 90))
        self.server.scene.set_up_direction("+y")
        _pos_tuple     = tuple((_centroid + np.array([0, 0, _extent * 2])).tolist())
        _look_at_tuple = tuple(_centroid.tolist())

        @self.server.on_client_connect
        def _on_client_connect(client):
            client.camera.position = _pos_tuple
            client.camera.look_at  = _look_at_tuple

        # nerfview Viewer (uses new API: render_fn(camera_state, render_tab_state))
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._render_fn,
            mode="rendering",
        )

        # Eagerly precompute all frames if requested (--precompute flag)
        if precompute and self.deform_field is not None:
            self._precompute_all_frames()

        print(f"\n  Viewer running at http://localhost:{port}")

    def _setup_ui(self):
        """Build viser GUI controls."""
        server = self.server

        # --- Animation ---
        with server.gui.add_folder("Animation"):
            self.frame_slider = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(self.num_frames - 1, 1),
                step=1,
                initial_value=0,
            )
            self.auto_play_cb = server.gui.add_checkbox(
                "Auto Play", initial_value=False
            )
            self.play_fps_slider = server.gui.add_slider(
                "Play FPS", min=1, max=60, step=1, initial_value=15
            )
            self.loop_cb = server.gui.add_checkbox("Loop", initial_value=True)

            @self.frame_slider.on_update
            def _(_):
                self.current_frame = int(self.frame_slider.value)
                self._update_stats()
                self.viewer.rerender(None)

            @self.auto_play_cb.on_update
            def _(_):
                self.auto_play = self.auto_play_cb.value

            @self.play_fps_slider.on_update
            def _(_):
                self.play_fps = self.play_fps_slider.value

        # --- Filtering ---
        with server.gui.add_folder("Filtering"):
            self.spatial_slider = server.gui.add_slider(
                "Spatial Radius",
                min=0.1,
                max=1.0,
                step=0.01,
                initial_value=0.85,
            )
            self.sh_degree_slider = server.gui.add_slider(
                "SH Degree",
                min=0,
                max=self.sh_degree,
                step=1,
                initial_value=self.sh_degree,
            )

            @self.spatial_slider.on_update
            def _(_):
                self.viewer.rerender(None)

            @self.sh_degree_slider.on_update
            def _(_):
                self.viewer.rerender(None)

        # --- Rendering ---
        with server.gui.add_folder("Rendering"):
            self.render_scale_slider = server.gui.add_slider(
                "Render Scale",
                min=0.25,
                max=1.0,
                step=0.05,
                initial_value=1.0,
            )
            self.autoplay_scale_slider = server.gui.add_slider(
                "Autoplay Scale",
                min=0.25,
                max=1.0,
                step=0.05,
                initial_value=1.0,
            )
            self.bg_color = server.gui.add_rgb(
                "Background", initial_value=(0, 0, 0)
            )
            self.near_plane_slider = server.gui.add_number(
                "Near Plane", initial_value=0.01, min=0.001, max=10.0, step=0.01
            )
            self.far_plane_slider = server.gui.add_number(
                "Far Plane", initial_value=1000.0, min=1.0, max=10000.0, step=1.0
            )

            @self.bg_color.on_update
            def _(_):
                self.viewer.rerender(None)

            @self.near_plane_slider.on_update
            def _(_):
                self.viewer.rerender(None)

            @self.far_plane_slider.on_update
            def _(_):
                self.viewer.rerender(None)

        # --- Static / Dynamic Visibility ---
        if self.static_splats is not None:
            with server.gui.add_folder("Layers"):
                self.show_dynamic_cb = server.gui.add_checkbox(
                    "Show Dynamic", initial_value=True
                )
                self.show_static_cb = server.gui.add_checkbox(
                    "Show Static", initial_value=True
                )

                @self.show_dynamic_cb.on_update
                def _(_):
                    self.viewer.rerender(None)

                @self.show_static_cb.on_update
                def _(_):
                    self.viewer.rerender(None)
        else:
            self.show_dynamic_cb = None
            self.show_static_cb = None

        # --- Export ---
        if self.deform_field is not None:
            with server.gui.add_folder("Export"):
                self.export_dir_text = server.gui.add_text(
                    "Output Dir",
                    initial_value=self.config.get("result_dir", "/tmp/4dgs_export"),
                )
                export_btn = server.gui.add_button("Export Per-Frame PLYs (dynamic)")

                @export_btn.on_click
                def _(_):
                    out_dir = self.export_dir_text.value.rstrip("/") + "/ply_per_frame"
                    threading.Thread(
                        target=self._export_per_frame_plys, args=(out_dir,), daemon=True
                    ).start()

        # --- Stats ---
        with server.gui.add_folder("Stats"):
            self.stats_text = server.gui.add_markdown("Loading...")

        self._update_stats()

    def _update_stats(self):
        t = self.current_frame / max(self.num_frames - 1, 1) - 0.5
        n_dyn = self.splats["means"].shape[0]
        n_static = self.static_splats["means"].shape[0] if self.static_splats is not None else 0
        has_deform = self.deform_field is not None
        cached = len(self._frame_cache)
        self.stats_text.content = (
            f"**Frame:** {self.current_frame} / {self.num_frames - 1} (t={t:.3f})\n\n"
            f"**Dynamic Gaussians:** {n_dyn:,}\n\n"
            + (f"**Static Gaussians:** {n_static:,}\n\n" if n_static else "")
            + f"**Deformation:** {'active' if has_deform else 'static'}\n\n"
            f"**Cached:** {cached}/{self.num_frames} frames\n\n"
            f"**Step:** {self.step:,}"
        )

    @torch.no_grad()
    def _precompute_all_frames(self):
        """Eagerly precompute deformed Gaussians for every frame (no spatial filtering).

        Stores on GPU. Use --precompute at startup, or call before autoplay.
        """
        print(f"  Pre-computing deformations for {self.num_frames} frames...")
        t0 = time.time()
        for frame in range(self.num_frames):
            if frame in self._frame_cache:
                continue
            t = frame / max(self.num_frames - 1, 1) - 0.5
            deltas = self.deform_field(self.splats["means"], t)
            self._frame_cache[frame] = apply_deformation(self.splats, deltas, aabb=self.aabb)
            if (frame + 1) % 50 == 0 or frame == self.num_frames - 1:
                print(f"    Frame {frame + 1}/{self.num_frames}")
        elapsed = time.time() - t0
        print(f"  Pre-compute done in {elapsed:.1f}s "
              f"({elapsed / self.num_frames * 1000:.1f}ms/frame)")

    @torch.no_grad()
    def _export_per_frame_plys(self, out_dir: str):
        """Export one PLY per frame with deformation baked in (dynamic only)."""
        from gsplat import export_splats as _export_splats
        import os as _os
        _os.makedirs(out_dir, exist_ok=True)
        print(f"  Exporting {self.num_frames} per-frame PLYs to {out_dir} ...")
        frame_start = self.config.get("frame_start", 0)
        frame_stride = self.config.get("frame_stride", 1)
        t0 = time.time()
        for frame_rank in range(self.num_frames):
            t = frame_rank / max(self.num_frames - 1, 1) - 0.5
            deltas = self.deform_field(self.splats["means"], t)
            means_d, quats_d, scales_d, opacs_d, colors_d = apply_deformation(
                self.splats, deltas, aabb=self.aabb
            )
            sh0_d = colors_d[:, :1, :]
            shN_d = colors_d[:, 1:, :]
            frame_idx = frame_start + frame_rank * frame_stride
            _export_splats(
                means=means_d,
                scales=torch.log(scales_d),
                quats=quats_d,
                opacities=torch.logit(opacs_d.clamp(1e-6, 1 - 1e-6)),
                sh0=sh0_d,
                shN=shN_d,
                format="ply",
                save_to=f"{out_dir}/frame_{frame_idx:06d}.ply",
            )
            if (frame_rank + 1) % 50 == 0 or frame_rank == self.num_frames - 1:
                print(f"    {frame_rank + 1}/{self.num_frames}")
        print(f"  Done in {time.time()-t0:.1f}s → {out_dir}")

    def _get_deformed(self, spatial_factor: float):
        """Return deformed Gaussians for the current frame.

        Deformation is cached per frame (expensive HexPlane+MLP forward).
        Spatial filtering is applied on top as a cheap index op — no cache
        invalidation needed when the spatial slider changes.
        """
        frame = self.current_frame

        # --- Deformation (cached per frame) ---
        if frame not in self._frame_cache:
            self._frame_loading = True
            try:
                if self.deform_field is not None:
                    t = frame / max(self.num_frames - 1, 1) - 0.5
                    with torch.no_grad():
                        deltas = self.deform_field(self.splats["means"], t)
                        self._frame_cache[frame] = apply_deformation(self.splats, deltas, aabb=self.aabb)
                else:
                    self._frame_cache[frame] = (
                        self.splats["means"],
                        self.splats["quats"],
                        torch.exp(self.splats["scales"]),
                        torch.sigmoid(self.splats["opacities"]),
                        torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1),
                    )
            finally:
                self._frame_loading = False

        means, quats, scales, opacities, colors = self._frame_cache[frame]

        # --- Spatial filtering (cheap, not cached) ---
        if spatial_factor < 1.0:
            mask = self.distances <= (spatial_factor * self.max_radius)
            if not mask.all():
                return means[mask], quats[mask], scales[mask], opacities[mask], colors[mask]

        return means, quats, scales, opacities, colors

    @torch.no_grad()
    def _render_fn(self, camera_state: CameraState, img_wh):
        """Render function called by nerfview on each frame.

        Uses a lock to prevent overlapping GPU renders that cause
        checkerboard artifacts and memory issues at high FPS.
        """
        # Handle both old API (img_wh tuple) and new API (RenderTabState)
        if isinstance(img_wh, tuple):
            width, height = img_wh
        else:
            render_tab = img_wh
            width = render_tab.viewer_width
            height = render_tab.viewer_height

        # If another render is in progress, return the last good frame
        if not self._render_lock.acquire(blocking=False):
            if self._last_rendered_img is not None:
                h, w = self._last_rendered_img.shape[:2]
                if h == height and w == width:
                    return self._last_rendered_img
            return np.zeros((height, width, 3), dtype=np.float32)

        try:
            self._render_busy = True

            # Resolution scaling: lower during autoplay for smoother playback
            scale = (self.autoplay_scale_slider.value if self.auto_play
                     else self.render_scale_slider.value)
            if scale < 1.0:
                render_w = max(int(width * scale), 1)
                render_h = max(int(height * scale), 1)
            else:
                render_w, render_h = width, height

            c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
            K = torch.from_numpy(
                camera_state.get_K((render_w, render_h))
            ).float().to(self.device)

            spatial_factor = self.spatial_slider.value

            # Collect layers to render
            all_means, all_quats, all_scales, all_opacities, all_colors = [], [], [], [], []

            show_dyn = self.show_dynamic_cb is None or self.show_dynamic_cb.value
            if show_dyn:
                d_means, d_quats, d_scales, d_opacities, d_colors = self._get_deformed(spatial_factor)
                all_means.append(d_means)
                all_quats.append(d_quats)
                all_scales.append(d_scales)
                all_opacities.append(d_opacities)
                all_colors.append(d_colors)

            show_static = self.show_static_cb is not None and self.show_static_cb.value
            if show_static and self.static_splats is not None:
                sa = self.static_splats
                all_means.append(sa["means"])
                all_quats.append(sa["quats"])
                all_scales.append(sa["scales"])
                all_opacities.append(sa["opacities"])
                all_colors.append(sa["colors"])

            if not all_means:
                img = np.zeros((height, width, 3), dtype=np.float32)
                self._last_rendered_img = img
                return img

            means    = torch.cat(all_means,    dim=0) if len(all_means)    > 1 else all_means[0]
            quats    = torch.cat(all_quats,    dim=0) if len(all_quats)    > 1 else all_quats[0]
            scales   = torch.cat(all_scales,   dim=0) if len(all_scales)   > 1 else all_scales[0]
            opacities = torch.cat(all_opacities, dim=0) if len(all_opacities) > 1 else all_opacities[0]
            colors   = torch.cat(all_colors,   dim=0) if len(all_colors)   > 1 else all_colors[0]

            if means.shape[0] == 0:
                img = np.zeros((height, width, 3), dtype=np.float32)
                self._last_rendered_img = img
                return img

            sh_deg = int(self.sh_degree_slider.value)
            bg = torch.tensor(
                [c / 255.0 for c in self.bg_color.value],
                device=self.device,
            )

            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(c2w)[None],
                Ks=K[None],
                width=render_w,
                height=render_h,
                sh_degree=sh_deg,
                near_plane=self.near_plane_slider.value,
                far_plane=self.far_plane_slider.value,
                packed=True,
                backgrounds=bg,
            )

            # Wait for GPU to finish before touching the output
            torch.cuda.synchronize()

            out = render_colors[0, ..., :3].clamp(0, 1)

            # Upscale back to original resolution if we rendered smaller
            if render_w != width or render_h != height:
                out = out.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
                out = torch.nn.functional.interpolate(
                    out, size=(height, width), mode="bilinear", align_corners=False
                )
                out = out.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

            img = out.cpu().numpy()
            self._last_rendered_img = img
            return img
        finally:
            self._render_busy = False
            self._render_lock.release()

    def run(self):
        """Main loop: handles auto-play and keeps the server alive."""
        try:
            while True:
                if self.auto_play:
                    now = time.time()
                    elapsed = now - self._last_frame_time
                    # Only advance if: enough time passed, GPU not busy, and current
                    # frame's deformation is done (not a cache miss still computing).
                    if (elapsed >= 1.0 / self.play_fps
                            and not self._render_busy
                            and not self._frame_loading):
                        next_frame = self.current_frame + 1
                        if next_frame >= self.num_frames:
                            next_frame = 0 if self.loop_cb.value else self.num_frames - 1
                        if next_frame != self.current_frame:
                            self.current_frame = next_frame
                            self.frame_slider.value = self.current_frame
                            self._update_stats()
                            self.viewer.rerender(None)
                        self._last_frame_time = now
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("\nViewer stopped.")


def main():
    parser = argparse.ArgumentParser(description="4DGS Interactive Viewer")
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to 4DGS checkpoint (.pt file)",
    )
    parser.add_argument("--port", type=int, default=8080, help="Viewer port")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num-frames", type=int, default=0,
        help="Override number of frames (0 = read from checkpoint config)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU index to use (e.g. --gpu 3)",
    )
    parser.add_argument(
        "--precompute", action="store_true",
        help="Pre-compute deformations for all frames at startup (uses more VRAM)",
    )
    parser.add_argument(
        "--static-ply", type=str, default=None,
        help="Path to static background PLY to overlay with the dynamic model",
    )
    parser.add_argument(
        "--export-ply", type=str, default=None, metavar="OUT_DIR",
        help="Export per-frame dynamic PLYs to OUT_DIR and exit (no viewer launched)",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    viewer = Viewer4DGS(
        ckpt_path=args.ckpt,
        port=args.port,
        device=args.device,
        num_frames_override=args.num_frames,
        precompute=args.precompute,
        static_ply_path=args.static_ply,
    )

    if args.export_ply:
        viewer._export_per_frame_plys(args.export_ply)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
