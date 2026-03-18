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

        # AABB: reconstruct from splat means
        pts = splats["means"].data
        aabb_min = pts.min(0).values
        aabb_max = pts.max(0).values
        margin = (aabb_max - aabb_min) * 0.1
        aabb = torch.stack([aabb_min - margin, aabb_max + margin]).cpu()

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
            sh_degree=sh_degree,
            time_pe_bands=time_pe_bands,
        ).to(device)
        deform_field.load_state_dict(ckpt["deform_field"])
        deform_field.eval()
        print(f"  Deformation: grid={grid_res}, time={time_res}, "
              f"width={net_width}, depth={net_depth}, multires={multires}")
    else:
        print("  Deformation: None (static model)")

    return splats, deform_field, config, step, sh_degree


class Viewer4DGS:
    """Interactive 4DGS viewer with time scrubbing."""

    def __init__(
        self,
        ckpt_path: str,
        port: int = 8080,
        device: str = "cuda",
        num_frames_override: int = 0,
        precompute: bool = False,
    ):
        self.device = device

        # Load model
        self.splats, self.deform_field, self.config, self.step, self.sh_degree = \
            load_checkpoint(ckpt_path, device)

        # Determine number of frames
        self.num_frames = num_frames_override or self.config.get("num_frames", 1)
        print(f"  Num frames: {self.num_frames}")

        # Spatial filtering: compute distances for outlier removal
        pts_np = self.splats["means"].data.cpu().numpy()
        center = pts_np.mean(axis=0)
        self.distances = torch.from_numpy(
            np.linalg.norm(pts_np - center, axis=1)
        ).float().to(device)
        self.max_radius = float(np.percentile(self.distances.cpu().numpy(), 99))

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

        # --- Stats ---
        with server.gui.add_folder("Stats"):
            self.stats_text = server.gui.add_markdown("Loading...")

        self._update_stats()

    def _update_stats(self):
        t = self.current_frame / max(self.num_frames - 1, 1) - 0.5
        n_gs = self.splats["means"].shape[0]
        has_deform = self.deform_field is not None
        cached = len(self._frame_cache)
        self.stats_text.content = (
            f"**Frame:** {self.current_frame} / {self.num_frames - 1} (t={t:.3f})\n\n"
            f"**Gaussians:** {n_gs:,}\n\n"
            f"**Deformation:** {'active' if has_deform else 'static'}\n\n"
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
            self._frame_cache[frame] = apply_deformation(self.splats, deltas)
            if (frame + 1) % 50 == 0 or frame == self.num_frames - 1:
                print(f"    Frame {frame + 1}/{self.num_frames}")
        elapsed = time.time() - t0
        print(f"  Pre-compute done in {elapsed:.1f}s "
              f"({elapsed / self.num_frames * 1000:.1f}ms/frame)")

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
                        self._frame_cache[frame] = apply_deformation(self.splats, deltas)
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
            means, quats, scales, opacities, colors = self._get_deformed(spatial_factor)

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
    args = parser.parse_args()

    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    viewer = Viewer4DGS(
        ckpt_path=args.ckpt,
        port=args.port,
        device=args.device,
        num_frames_override=args.num_frames,
        precompute=args.precompute,
    )
    viewer.run()


if __name__ == "__main__":
    main()
