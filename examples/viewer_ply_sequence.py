"""
PLY Sequence Viewer
===================

Interactive viewer for per-frame PLY files (e.g., from finetune_per_frame.py).

Loads a directory of PLY files and provides:
1. Frame scrubbing with slider
2. Auto-play with adjustable FPS
3. Interactive camera via viser/nerfview
4. Render-to-disk for all frames or current frame

Usage:
    python src/viewer_ply_sequence.py \
        --ply-dir /path/to/refined_ply \
        --port 8080

    # Render all frames to disk from a specific camera pose:
    python src/viewer_ply_sequence.py \
        --ply-dir /path/to/refined_ply \
        --render-dir /path/to/renders \
        --render-width 1920 --render-height 1080
"""

import argparse
import glob
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import viser
from gsplat.rendering import rasterization
from plyfile import PlyData

import nerfview


def load_ply(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Load a standard 3DGS PLY file into tensors."""
    plydata = PlyData.read(path)
    v = plydata['vertex']

    means = torch.tensor(np.stack([v['x'], v['y'], v['z']], axis=-1), dtype=torch.float32, device=device)

    # Quaternions
    quats = torch.tensor(
        np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=-1),
        dtype=torch.float32, device=device,
    )

    # Scales (log space)
    scales = torch.tensor(
        np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=-1),
        dtype=torch.float32, device=device,
    )

    # Opacity (logit space)
    opacities = torch.tensor(np.array(v['opacity']), dtype=torch.float32, device=device)

    # SH DC
    dc = torch.tensor(
        np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=-1),
        dtype=torch.float32, device=device,
    )
    sh0 = dc.unsqueeze(1)  # [N, 1, 3]

    # SH rest
    sh_rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith('f_rest_')],
        key=lambda x: int(x.split('_')[-1]),
    )
    if sh_rest_names:
        sh_rest = torch.tensor(
            np.stack([np.array(v[n]) for n in sh_rest_names], axis=-1),
            dtype=torch.float32, device=device,
        )
        n_rest = len(sh_rest_names)
        # PLY stores SH as [N, 3*K] in order (r0,r1,...,g0,g1,...,b0,b1,...)
        # gsplat expects [N, K, 3] in order (r0,g0,b0, r1,g1,b1, ...)
        shN = sh_rest.reshape(-1, 3, n_rest // 3).transpose(1, 2).contiguous()  # [N, K, 3]
    else:
        shN = torch.zeros(means.shape[0], 0, 3, dtype=torch.float32, device=device)

    colors = torch.cat([sh0, shN], dim=1)  # [N, 1+K, 3]

    return {
        "means": means,
        "quats": quats,
        "scales_linear": torch.exp(scales),
        "opacities_sigmoid": torch.sigmoid(opacities),
        "colors": colors,
        "n": means.shape[0],
    }


class PLYSequenceViewer:
    """Interactive viewer for a sequence of PLY files with optional static overlay."""

    def __init__(self, ply_dir: str, port: int = 8080, device: str = "cuda",
                 sh_degree: int = 3, packed: bool = True,
                 near_plane: float = 0.01, far_plane: float = 1e10,
                 static_ply: str = None, start_frame: int = None, end_frame: int = None):
        self.device = torch.device(device)
        self.sh_degree = sh_degree
        self.packed = packed
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.port = port
        self._first_render = True

        # Load static PLY if provided
        self.static_frame = None
        if static_ply and os.path.exists(static_ply):
            print(f"Loading static PLY: {static_ply}")
            self.static_frame = load_ply(static_ply, torch.device(device))
            print(f"  Static: {self.static_frame['n']:,} Gaussians")

        # Discover PLY files, optionally filter by start/end frame
        ply_files = sorted(glob.glob(os.path.join(ply_dir, "*.ply")))
        if not ply_files:
            raise FileNotFoundError(f"No PLY files found in {ply_dir}")
        if start_frame is not None or end_frame is not None:
            filtered = []
            for f in ply_files:
                # Extract frame number from filename (e.g., 000005.ply → 5)
                name = os.path.splitext(os.path.basename(f))[0]
                try:
                    num = int(re.sub(r'[^0-9]', '', name))
                except ValueError:
                    continue
                if start_frame is not None and num < start_frame:
                    continue
                if end_frame is not None and num > end_frame:
                    continue
                filtered.append(f)
            ply_files = filtered
            if not ply_files:
                raise FileNotFoundError(f"No PLY files in range [{start_frame}, {end_frame}]")

        self.ply_dir = ply_dir
        self.ply_files = ply_files
        self.total_frames = len(ply_files)

        # Extract frame numbers from filenames
        self.frame_names = [os.path.splitext(os.path.basename(f))[0] for f in ply_files]

        print(f"\n{'='*60}")
        print(f"PLY Sequence Viewer")
        print(f"{'='*60}")
        print(f"  Dynamic dir: {ply_dir}")
        print(f"  Static PLY:  {static_ply or 'None'}")
        print(f"  PLY files:   {self.total_frames}")
        print(f"  First: {self.frame_names[0]}")
        print(f"  Last:  {self.frame_names[-1]}")

        # Lazy loading: only keep current frame on GPU to avoid OOM
        # Load first frame to get SH degree, then cache on demand
        print(f"\nLazy loading mode (frames loaded on demand)")
        first_frame = load_ply(ply_files[0], torch.device(device))
        print(f"  First frame: {first_frame['n']:,} Gaussians")

        n_sh_coeffs = first_frame["colors"].shape[1]
        self.actual_sh_degree = int(n_sh_coeffs ** 0.5) - 1
        print(f"  SH degree: {self.actual_sh_degree}")

        # Cache: only current frame stays on GPU
        self._cached_frame_idx = 0
        self._cached_frame = first_frame
        print(f"{'='*60}\n")

        # Viewer state
        self.current_frame = 0
        self.auto_play = False
        self.play_speed = 10.0

        # Setup viser + nerfview
        self.server = viser.ViserServer(port=port, verbose=False)
        self._setup_ui()

        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._render_fn,
            mode="rendering",
        )

    def _setup_ui(self):
        with self.server.gui.add_folder("Animation"):
            self.frame_slider = self.server.gui.add_slider(
                "Frame", min=0, max=self.total_frames - 1, step=1, initial_value=0,
            )
            self.auto_play_cb = self.server.gui.add_checkbox("Auto Play", initial_value=False)
            self.speed_slider = self.server.gui.add_slider(
                "Play Speed (FPS)", min=1, max=60, step=1, initial_value=10,
            )

        with self.server.gui.add_folder("Rendering"):
            self.bg_color_dropdown = self.server.gui.add_dropdown(
                "Background", options=["White", "Black"], initial_value="White",
            )
            self.show_static_cb = self.server.gui.add_checkbox(
                "Show Static", initial_value=self.static_frame is not None,
            )

        with self.server.gui.add_folder("Stats"):
            self.stats_text = self.server.gui.add_markdown("...")

        @self.frame_slider.on_update
        def _(_):
            self.current_frame = int(self.frame_slider.value)
            self._update_stats()
            if hasattr(self, 'viewer'):
                self.viewer.rerender(None)

        @self.auto_play_cb.on_update
        def _(_):
            self.auto_play = self.auto_play_cb.value

        @self.speed_slider.on_update
        def _(_):
            self.play_speed = self.speed_slider.value

        @self.bg_color_dropdown.on_update
        def _(_):
            if hasattr(self, 'viewer'):
                self.viewer.rerender(None)

        @self.show_static_cb.on_update
        def _(_):
            if hasattr(self, 'viewer'):
                self.viewer.rerender(None)

    def _get_frame(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lazy load frame — only one frame on GPU at a time."""
        if self._cached_frame_idx != idx:
            # Free old frame
            if hasattr(self, '_cached_frame') and self._cached_frame is not None:
                del self._cached_frame
                torch.cuda.empty_cache()
            # Load new
            self._cached_frame = load_ply(self.ply_files[idx], self.device)
            self._cached_frame_idx = idx
        return self._cached_frame

    def _update_stats(self):
        idx = self.current_frame
        frame = self._get_frame(idx)
        self.stats_text.content = f"""
**Frame:** {idx} / {self.total_frames - 1} ({self.frame_names[idx]})

**Gaussians:** {frame['n']:,}
"""

    @torch.no_grad()
    def _render_fn(self, camera_state, img_wh):
        # img_wh can be tuple (W, H) or RenderTabState
        if isinstance(img_wh, (tuple, list)):
            W, H = img_wh
        else:
            W, H = img_wh.render_width, img_wh.render_height
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = torch.from_numpy(camera_state.get_K((W, H))).float().to(self.device)
        width, height = W, H
        viewmat = c2w.inverse()

        frame = self._get_frame(self.current_frame)

        sh_degree = min(self.sh_degree, self.actual_sh_degree)

        bg_val = 1.0 if self.bg_color_dropdown.value == "White" else 0.0

        # Merge static + dynamic if static overlay is enabled
        show_static = self.show_static_cb.value and self.static_frame is not None
        if show_static:
            s = self.static_frame
            # Match SH dimensions
            d_sh = frame["colors"].shape[1]
            s_sh = s["colors"].shape[1]
            max_sh = max(d_sh, s_sh)
            d_colors = frame["colors"] if d_sh == max_sh else torch.cat(
                [frame["colors"], torch.zeros(frame["n"], max_sh - d_sh, 3, device=self.device)], 1)
            s_colors = s["colors"] if s_sh == max_sh else torch.cat(
                [s["colors"], torch.zeros(s["n"], max_sh - s_sh, 3, device=self.device)], 1)

            means = torch.cat([frame["means"], s["means"]], 0)
            quats = torch.cat([frame["quats"], s["quats"]], 0)
            scales = torch.cat([frame["scales_linear"], s["scales_linear"]], 0)
            opacities = torch.cat([frame["opacities_sigmoid"], s["opacities_sigmoid"]], 0)
            colors = torch.cat([d_colors, s_colors], 0)
            total_n = frame["n"] + s["n"]
        else:
            means = frame["means"]
            quats = frame["quats"]
            scales = frame["scales_linear"]
            opacities = frame["opacities_sigmoid"]
            colors = frame["colors"]
            total_n = frame["n"]

        if self._first_render:
            self._first_render = False
            print(f"  First render: frame={self.current_frame}, "
                  f"{total_n:,} Gaussians{' (+static)' if show_static else ''}, "
                  f"{width}x{height}, sh={sh_degree}")

        render_colors, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            packed=self.packed,
        )

        return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()

    def run(self):
        print(f"PLY Sequence Viewer running at http://localhost:{self.port}")
        self._update_stats()

        last_time = time.time()
        try:
            while True:
                if self.auto_play:
                    now = time.time()
                    if now - last_time >= 1.0 / self.play_speed:
                        self.current_frame = (self.current_frame + 1) % self.total_frames
                        self.frame_slider.value = self.current_frame
                        self._update_stats()
                        self.viewer.rerender(None)
                        last_time = now
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopped.")


def main():
    parser = argparse.ArgumentParser(description="PLY Sequence Viewer")
    parser.add_argument("--ply-dir", type=str, required=True,
                        help="Directory containing per-frame PLY files")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--packed", action="store_true", default=True)
    parser.add_argument("--static-ply", type=str, default=None,
                        help="Optional static PLY to overlay on all frames")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="Start frame index (default: first PLY found)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="End frame index inclusive (default: last PLY found)")

    args = parser.parse_args()

    viewer = PLYSequenceViewer(
        ply_dir=args.ply_dir,
        port=args.port,
        device=args.device,
        sh_degree=args.sh_degree,
        packed=args.packed,
        static_ply=args.static_ply,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    viewer.run()


if __name__ == "__main__":
    main()
