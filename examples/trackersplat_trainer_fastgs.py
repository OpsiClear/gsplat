"""
TrackerSplat 4D Gaussian Tracking Trainer — Faster-GS backend.

Same training logic as examples/trackersplat_trainer.py, but the CUDA rasterizer
is swapped from gsplat.rendering.rasterization (Apache-2.0, nerfstudio) to
FasterGSCudaBackend.torch_bindings.diff_rasterize (Apache-2.0, nerficg-project).
PLY IO is kept via gsplat.io_ply (Apache-2.0, pure Python — not a rasterizer).

Usage:
    conda activate gsplat_fastergs           # env with FasterGSCudaBackend installed
    python examples/trackersplat_trainer_fastgs.py \
        --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/ \
        --ply_path /data/shared/elaheh/4D_demo/thenewface_multiframe_fast/frame_001/ply/point_cloud_combined_2999.ply \
        --cotracker_dir /data/shared/elaheh/4D_demo/new_data/thenewface/cotracker_out/ \
        --max_steps 5000
"""

import json
import math
import os
import random
import time

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from datasets.colmap import Parser
from fused_ssim import fused_ssim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from utils import rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.io_ply import import_splats, sh2rgb

# --- Faster-GS CUDA backend ---------------------------------------------------
from FasterGSCudaBackend.torch_bindings import (
    FusedAdam,
    RasterizerSettings,
    add_noise,
    diff_rasterize,
    rasterize,
    relocation_adjustment,
    update_pruning_scores,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Paths
    data_dir: str = ""
    ply_path: str = ""
    static_ply_path: str = ""
    cotracker_dir: str = ""
    result_dir: str = "/data/shared/elaheh/4D_demo/new_data/trackersplat_results_fastgs"

    # 4D parameters
    num_cotracker_frames: int = 50
    frame_step: int = 6
    sh_degree: int = 3

    # Learning rates
    motion_lr: float = 1e-3
    means_lr: float = 1e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # Loss weights
    track_loss_weight: float = 1.0
    photo_loss_weight: float = 1.0
    temporal_smooth_weight: float = 0.1
    spatial_rigid_weight: float = 0.05
    ssim_lambda: float = 0.2

    # Schedule
    freeze_appearance_steps: int = 1000
    max_steps: int = 5000
    eval_steps: List[int] = field(default_factory=lambda: [1000, 3000, 5000])
    save_steps: List[int] = field(default_factory=lambda: [5000])
    ply_save_steps: List[int] = field(default_factory=lambda: [7000, 10000, 15000, 25000, 30000, 40000, 50000])

    # Rendering
    near_plane: float = 0.01
    far_plane: float = 1e10
    data_factor: int = 15
    normalize_world_space: bool = True
    # Whether Parser skips points3D.bin. The PLY was trained with
    # skip_points3d=False (normalization uses point-cloud centroid); setting
    # this to True produces a different world frame and drops render PSNR by
    # ~7 dB before training even starts. Always keep False unless you know
    # the PLY was exported under a camera-only normalization.
    skip_points3d: bool = False
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    proper_antialiasing: bool = False

    # Optimizer
    use_fused_adam: bool = True  # FusedAdam from FasterGS — else torch.optim.Adam

    # KNN for rigidity
    knn_k: int = 5

    # Misc
    seed: int = 42
    render_video_cams: List[int] = field(default_factory=lambda: [0, 22, 44])

    # TensorBoard
    tb_image_every: int = 200
    # Sample a small subset of (cam, frame) pairs every `tb_image_every`
    # steps and push eval/psnr, eval/ssim, eval/l1, eval/tracking_error_px
    # to TB. Cheap (~3×3 rasterizations) vs the full eval_tracking.
    tb_eval_sample_cams: int = 3
    tb_eval_sample_frames: int = 3

    # -----------------------------------------------------------------------
    # MCMC densification / pruning (paper §5.2 refinement + densify + prune)
    # -----------------------------------------------------------------------
    # Anchor preservation: dynamic splats bound to cotracker tracks are never
    # pruned and never reordered, so binding_map/knn_indices stay valid.
    #
    # Stage 1 (0 .. max_steps - refinement_steps):   MCMC noise + prune/grow
    # Stage 2 (refinement_steps at the end):          no noise, no struct-change
    #
    # MCMC noise (FasterGS add_noise) applied to NON-ANCHOR splats every step.
    # Off by default — matches the original trainer's behaviour. Opt in via
    # --use_mcmc_noise. Anchors (cotracker-bound splats) are skipped either way.
    use_mcmc_noise: bool = False
    mcmc_noise_start: int = 500
    mcmc_noise_lr: float = 5e-5

    # Structural updates (prune + grow). Off by default. Opt in via
    # --use_densify_prune. Anchors are never pruned/reordered.
    use_densify_prune: bool = False
    densify_start: int = 2000
    densify_end: int = 45000
    densify_interval: int = 500
    densify_score_views: int = 8              # how many views to accumulate scores over
    prune_score_quantile: float = 0.05        # prune bottom q of NON-anchor splats
    grow_score_quantile: float = 0.95         # duplicate top q of NON-anchor splats
    grow_max_fraction: float = 0.01           # grow at most 1% of non-anchors per event
    max_dyn_gaussians: int = 800_000
    min_dyn_gaussians: int = 5_000
    jitter_scale_world: float = 0.01          # position jitter for duplicated splats

    # Refinement phase — last N steps have structural changes disabled.
    # Zero by default (old behaviour); the paper uses 1000.
    refinement_steps: int = 0

    # Paper §5.2 "unsolvable Gaussians" regularization: every N steps,
    # identify dynamic splats that are (a) visible in < min_views_visible
    # random views, or (b) have pruning-score contribution below
    # min_prune_score_unsolvable. For those splats, copy parameters from
    # their 8 nearest *solvable* neighbours. Only applied to non-anchors
    # so track-anchored splats keep their exact bound positions.
    use_unsolvable_reg: bool = False
    unsolvable_start: int = 2000
    unsolvable_end: int = 50000
    unsolvable_interval: int = 1000
    unsolvable_score_views: int = 8       # how many random views to sample
    min_views_visible: int = 2            # paper threshold (<2 → unsolvable)
    min_prune_score_unsolvable: float = 1e-3
    unsolvable_k_nn: int = 8              # 8 nearest neighbours (paper)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def project_points(means_3d: Tensor, viewmat: Tensor, K: Tensor) -> Tuple[Tensor, Tensor]:
    """Project 3D points to 2D pixel coordinates (differentiable)."""
    R = viewmat[:3, :3]
    t = viewmat[:3, 3]
    cam_pts = means_3d @ R.T + t[None, :]
    z = cam_pts[:, 2:3].clamp(min=0.01)
    uv = cam_pts[:, :2] / z
    px = K[0, 0] * uv[:, 0] + K[0, 2]
    py = K[1, 1] * uv[:, 1] + K[1, 2]
    return torch.stack([px, py], dim=-1), z.squeeze(-1)


def compute_knn(points: Tensor, k: int) -> Tensor:
    """k-NN via chunked cdist on GPU."""
    N = points.shape[0]
    chunk = 4096
    all_indices = []
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        dists = torch.cdist(points[i:end], points)
        for j in range(end - i):
            dists[j, i + j] = float("inf")
        _, idx = dists.topk(k, dim=-1, largest=False)
        all_indices.append(idx)
    return torch.cat(all_indices, dim=0)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class TrackerSplatRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")
        set_random_seed(cfg.seed)

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.render_dir = os.path.join(cfg.result_dir, "renders")
        self.ckpt_dir = os.path.join(cfg.result_dir, "ckpts")
        self.video_dir = os.path.join(cfg.result_dir, "videos")
        self.tb_dir = os.path.join(cfg.result_dir, "tb")
        for d in (self.render_dir, self.ckpt_dir, self.video_dir, self.tb_dir):
            os.makedirs(d, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        print("[Init] Loading camera poses...")
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=9999,
            frame_num=1,
            skip_points3d=cfg.skip_points3d,
        )
        self.num_cameras = len(self.parser.image_names)
        print(f"[Init] {self.num_cameras} cameras loaded")

        self.camtoworlds = torch.from_numpy(self.parser.camtoworlds).float().to(self.device)
        self.viewmats = torch.linalg.inv(self.camtoworlds)

        self.Ks: List[Tensor] = []
        self.img_sizes: List[Tuple[int, int]] = []
        self.cam_names: List[str] = []
        for i, name in enumerate(self.parser.image_names):
            cam_dir = os.path.dirname(name)
            self.cam_names.append(cam_dir)
            cam_id = self.parser.camera_ids[i]
            K = torch.from_numpy(self.parser.Ks_dict[cam_id]).float().to(self.device)
            self.Ks.append(K)
            self.img_sizes.append(self.parser.imsize_dict[cam_id])
        self.Ks_tensor = torch.stack(self.Ks, dim=0)

        print("[Init] Loading CoTracker tracks...")
        self._load_cotracker_data()

        print(f"[Init] Loading PLY from {cfg.ply_path}...")
        self._init_splats()

        print("[Init] Building KNN graph...")
        with torch.no_grad():
            self.knn_indices = compute_knn(
                self.splats["means"].detach(), cfg.knn_k
            )
        print(f"[Init] KNN done: {self.knn_indices.shape}")

        print("[Init] Building track-to-Gaussian bindings...")
        self._build_bindings()

        # Anchor splats = any dynamic splat referenced by a *valid* binding in
        # any camera. These are the cotracker-bound primitives. Prune/grow
        # never touches them.
        N_dyn = self.splats["means"].shape[0]
        anchor_mask = torch.zeros(N_dyn, dtype=torch.bool, device=self.device)
        for ct_idx, idx in self.binding_map.items():
            valid = self.binding_valid[ct_idx]
            anchor_mask[idx[valid]] = True

        # Reorder all dynamic splats so anchors occupy indices [0 : n_anchors)
        # and non-anchors occupy [n_anchors : N). Then MCMC noise / pruning /
        # growth operate purely on the non-anchor tail and binding_map never
        # points into that region — it stays valid across all structural edits.
        anchor_idx    = torch.where( anchor_mask)[0]
        nonanchor_idx = torch.where(~anchor_mask)[0]
        perm = torch.cat([anchor_idx, nonanchor_idx], dim=0)
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN",
                     "motion_offsets"):
            self.splats[name] = torch.nn.Parameter(
                self.splats[name].data[perm].contiguous()
            )
        inverse = torch.empty(N_dyn, dtype=torch.long, device=self.device)
        inverse[perm] = torch.arange(N_dyn, device=self.device)
        for ct_idx in list(self.binding_map.keys()):
            self.binding_map[ct_idx] = inverse[self.binding_map[ct_idx]]
        self.n_anchors = int(anchor_mask.sum())
        self.anchor_mask = torch.cat([
            torch.ones(self.n_anchors, dtype=torch.bool, device=self.device),
            torch.zeros(N_dyn - self.n_anchors, dtype=torch.bool, device=self.device),
        ])
        print(f"[Anchors] reordered: anchors [0:{self.n_anchors})  "
              f"non-anchors [{self.n_anchors}:{N_dyn})")

        self._setup_optimizers()

        # Pre-allocate per-primitive buffers consumed by diff_rasterize. Sizes
        # must grow with N if we ever densify; for this trainer N is fixed.
        total_N = self.splats["means"].shape[0] + (
            self.static["means"].shape[0] if self.static is not None else 0
        )
        self._dinfo = torch.zeros(total_N, device=self.device)

        # Constant camera / background color tensors reused every step
        self._bg_color = torch.tensor(cfg.bg_color, device=self.device)
        self._active_sh_bases = (cfg.sh_degree + 1) ** 2

        print(
            f"[Init] Ready. dyn={self.splats['means'].shape[0]} + "
            f"static={0 if self.static is None else self.static['means'].shape[0]} "
            f"Gaussians, {self.num_cameras} cameras, {cfg.num_cotracker_frames} frames, "
            f"active_sh_bases={self._active_sh_bases}"
        )

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    def _load_cotracker_data(self):
        """Loads per-camera tracks. Handles two NPZ schemas:

        cotracker_out:     keys = {tracks (1,T,N,2), visibility (1,T,N) bool,
                                   downsample_factor}. Coords are in the
                                   tracker's internal resolution and must be
                                   rescaled: coord_render = coord * df / data_factor.
        alltrackerxx_out:  keys = {trajs (T,N,2), vis (T,N) float[0,1],
                                   image_w, image_h, start_frame, stride}.
                                   Coords are already at image_w×image_h; we
                                   rescale only if that differs from the
                                   render resolution parser.imsize_dict.
        """
        cfg = self.cfg
        npz_files = sorted(Path(cfg.cotracker_dir).glob("*.npz"))
        assert len(npz_files) > 0, f"No .npz files in {cfg.cotracker_dir}"

        npz_by_cam = {f.stem: f for f in npz_files}

        all_tracks, all_vis = [], []
        self.cotracker_cam_indices: List[int] = []
        schema = None
        scale_report = 1.0

        for cam_idx, cam_name in enumerate(self.cam_names):
            if cam_name not in npz_by_cam:
                continue
            data = np.load(npz_by_cam[cam_name])
            keys = set(data.files)

            if "tracks" in keys:                # cotracker schema
                tracks = data["tracks"].squeeze(0)          # (T, N, 2)
                vis = data["visibility"].squeeze(0).astype(np.bool_)
                df = float(data["downsample_factor"])
                tracks = tracks * (df / cfg.data_factor)
                schema = schema or "cotracker"
                scale_report = df
            elif "trajs" in keys:               # alltrackerxx schema
                tracks = data["trajs"]                      # (T, N, 2)
                vis = (data["vis"] > 0.5)                   # float → bool
                # Rescale from tracker resolution to render resolution.
                W_render, H_render = self.img_sizes[cam_idx]
                W_tr = int(data["image_w"]); H_tr = int(data["image_h"])
                sx = W_render / W_tr
                sy = H_render / H_tr
                tracks = tracks * np.array([sx, sy], dtype=np.float32)
                schema = schema or "alltrackerxx"
                scale_report = (sx + sy) / 2.0
            else:
                raise ValueError(
                    f"unknown NPZ schema for {cam_name}: keys={sorted(keys)}"
                )

            all_tracks.append(torch.from_numpy(tracks).float())
            all_vis.append(torch.from_numpy(vis).bool())
            self.cotracker_cam_indices.append(cam_idx)

        # Pad to a common number of points across cameras (smallest N wins),
        # since alltrackerxx can produce slightly different N per camera.
        min_N = min(t.shape[1] for t in all_tracks)
        all_tracks = [t[:, :min_N, :] for t in all_tracks]
        all_vis    = [v[:, :min_N]    for v in all_vis]

        self.gt_tracks = torch.stack(all_tracks, dim=0).to(self.device)
        self.gt_vis    = torch.stack(all_vis,    dim=0).to(self.device)

        self.ct_to_cam = self.cotracker_cam_indices
        self.cam_to_ct = [-1] * self.num_cameras
        for ct_idx, cam_idx in enumerate(self.ct_to_cam):
            self.cam_to_ct[cam_idx] = ct_idx

        print(
            f"[Tracks] schema={schema}  {self.gt_tracks.shape[0]} cameras, "
            f"tracks shape: {self.gt_tracks.shape}  (scale={scale_report:.2f})"
        )

    def _init_splats(self):
        cfg = self.cfg
        device = self.device

        means, scales, quats, opacities, sh0, shN = import_splats(cfg.ply_path, device)
        N = means.shape[0]
        motion_offsets = torch.zeros((N, cfg.num_cotracker_frames, 3), device=device)

        self.splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
            "motion_offsets": torch.nn.Parameter(motion_offsets),
        }).to(device)

        # Verify SH layout matches cfg.sh_degree
        expected_K_minus_1 = (cfg.sh_degree + 1) ** 2 - 1
        if shN.shape[1] != expected_K_minus_1:
            raise ValueError(
                f"PLY sh_degree mismatch: got shN {shN.shape}, "
                f"expected shN[:, {expected_K_minus_1}, :] for sh_degree={cfg.sh_degree}"
            )
        print(
            f"[Splats/dynamic] {N} Gaussians  sh0 {tuple(sh0.shape)} shN {tuple(shN.shape)}  "
            f"motion_offsets {tuple(motion_offsets.shape)}  "
            f"VRAM {motion_offsets.numel() * 4 / 1e6:.1f}MB"
        )

        self.static = None
        if cfg.static_ply_path:
            s_m, s_s, s_q, s_o, s_sh0, s_shN = import_splats(cfg.static_ply_path, device)
            if s_shN.shape[1] != shN.shape[1]:
                raise ValueError(
                    f"SH degree mismatch: static shN={s_shN.shape}, "
                    f"dynamic shN={shN.shape}. Re-export with the same sh_degree."
                )
            self.static = {
                "means": s_m.detach(),
                "scales": s_s.detach(),
                "quats": s_q.detach(),
                "opacities": s_o.detach(),
                "sh0": s_sh0.detach(),
                "shN": s_shN.detach(),
            }
            print(f"[Splats/static]  {s_m.shape[0]} Gaussians (frozen)")

    def _build_bindings(self):
        dyn_means = self.splats["means"].detach()
        stat_means = self.static["means"] if self.static is not None else None

        self.binding_map: Dict[int, Tensor] = {}
        self.binding_valid: Dict[int, Tensor] = {}

        total_tracks = 0
        total_valid = 0
        for ct_idx, cam_idx in enumerate(self.ct_to_cam):
            viewmat = self.viewmats[cam_idx]
            K = self.Ks[cam_idx]

            proj_dyn, _ = project_points(dyn_means, viewmat, K)
            gt_t0 = self.gt_tracks[ct_idx, 0]

            dist_dyn = torch.cdist(gt_t0, proj_dyn)
            min_dyn, idx_dyn = dist_dyn.min(dim=-1)

            if stat_means is not None:
                proj_stat, _ = project_points(stat_means, viewmat, K)
                dist_stat = torch.cdist(gt_t0, proj_stat)
                min_stat, _ = dist_stat.min(dim=-1)
                valid = min_dyn < min_stat
            else:
                valid = torch.ones_like(min_dyn, dtype=torch.bool)

            # Invalid tracks get index 0 (always an anchor after reorder) so
            # proj_2d[bound_idx] never goes out-of-bounds after pruning of
            # non-anchor splats. `binding_valid` already masks them out of
            # the tracking MSE.
            safe_idx = torch.where(valid, idx_dyn, torch.zeros_like(idx_dyn))
            self.binding_map[ct_idx] = safe_idx
            self.binding_valid[ct_idx] = valid
            total_tracks += valid.numel()
            total_valid += int(valid.sum().item())

        if self.static is not None:
            pct = 100.0 * total_valid / max(total_tracks, 1)
            print(
                f"[Binding] {len(self.binding_map)} cameras, "
                f"{total_valid}/{total_tracks} tracks kept ({pct:.1f}%), "
                f"rest dropped as background"
            )
        else:
            print(f"[Binding] Built bindings for {len(self.binding_map)} cameras")

    def _setup_optimizers(self):
        cfg = self.cfg
        # FasterGS FusedAdam requires exactly one tensor per param_group.
        OptCls = FusedAdam if cfg.use_fused_adam else torch.optim.Adam

        self.motion_optimizer = OptCls(
            [{"params": self.splats["motion_offsets"]}], lr=cfg.motion_lr, eps=1e-15,
        )

        self.appearance_optimizers: Dict[str, torch.optim.Optimizer] = {}
        for name, lr in [
            ("means", cfg.means_lr),
            ("scales", cfg.scales_lr),
            ("quats", cfg.quats_lr),
            ("opacities", cfg.opacities_lr),
            ("sh0", cfg.sh0_lr),
            ("shN", cfg.shN_lr),
        ]:
            self.appearance_optimizers[name] = OptCls(
                [{"params": self.splats[name]}], lr=lr, eps=1e-15,
            )

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------
    def load_image(self, cam_idx: int, frame_idx: int) -> Tensor:
        image_num = frame_idx * self.cfg.frame_step + 1
        cam_name = self.cam_names[cam_idx]
        img_path = os.path.join(
            self.cfg.data_dir, "images", cam_name, f"{image_num:06d}.jpg"
        )
        img = imageio.imread(img_path)
        img = torch.from_numpy(img).float().to(self.device) / 255.0
        if self.cfg.data_factor > 1:
            f = self.cfg.data_factor
            img = img[::f, ::f]
        return img

    # -----------------------------------------------------------------------
    # Forward pass — FasterGS diff_rasterize
    # -----------------------------------------------------------------------
    def _make_settings(self, cam_idx: int) -> RasterizerSettings:
        K = self.Ks[cam_idx]
        W, H = self.img_sizes[cam_idx]
        cam_pos = self.camtoworlds[cam_idx, :3, 3].contiguous()
        return RasterizerSettings(
            w2c=self.viewmats[cam_idx].contiguous(),
            cam_position=cam_pos,
            bg_color=self._bg_color,
            active_sh_bases=self._active_sh_bases,
            width=int(W),
            height=int(H),
            focal_x=float(K[0, 0]),
            focal_y=float(K[1, 1]),
            center_x=float(K[0, 2]),
            center_y=float(K[1, 2]),
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            proper_antialiasing=self.cfg.proper_antialiasing,
        )

    def rasterize_splats(
        self, cam_idx: int, frame_idx: int
    ) -> Tuple[Tensor, Tensor, dict]:
        """Render composite (static frozen + dynamic with motion) via FasterGS.

        CONVENTIONS — confirmed from
        FasterGSCudaBackend/rasterization/include/kernels_forward.cuh:
          * scales are RAW log-scales; kernel applies variance = exp(2 * raw).
          * opacities are RAW pre-sigmoid scalars; kernel applies sigmoid().
          * quats are (w, x, y, z), any norm (kernel normalises internally).
          * sh0 shape (N, 1, 3), shN shape (N, K-1, 3); DC uses result = 0.5 + C0 * sh0
            — matches gsplat's import_splats layout, no re-encoding needed.

        Returns (render_colors, render_alphas, info) with render_colors shaped
        (1, H, W, 3) so downstream photometric / overlay code is unchanged.
        The alphas tensor is a dummy all-ones — diff_rasterize does not expose
        a separate alpha channel; downstream callers only rely on colors.
        """
        dyn_means = self.splats["means"] + self.splats["motion_offsets"][:, frame_idx, :]
        dyn_quats = self.splats["quats"]
        dyn_log_scales = self.splats["scales"]       # RAW log-scales
        dyn_raw_opac = self.splats["opacities"]      # RAW pre-sigmoid
        dyn_sh0 = self.splats["sh0"]
        dyn_shN = self.splats["shN"]

        if self.static is not None:
            means = torch.cat([self.static["means"], dyn_means], dim=0)
            quats = torch.cat([self.static["quats"], dyn_quats], dim=0)
            log_scales = torch.cat([self.static["scales"], dyn_log_scales], dim=0)
            raw_opacities = torch.cat([self.static["opacities"], dyn_raw_opac], dim=0)
            sh0 = torch.cat([self.static["sh0"], dyn_sh0], dim=0)
            shN = torch.cat([self.static["shN"], dyn_shN], dim=0)
        else:
            means = dyn_means
            quats = dyn_quats
            log_scales = dyn_log_scales
            raw_opacities = dyn_raw_opac
            sh0 = dyn_sh0
            shN = dyn_shN

        # diff_rasterize backward returns grad opacities shaped (N, 1), so the
        # input must come in as (N, 1) — otherwise autograd rejects the grad.
        # (rasterize() inference path accepts either shape, but we use (N, 1)
        # everywhere for consistency.)
        if raw_opacities.dim() == 1:
            raw_opacities = raw_opacities.unsqueeze(-1)
        settings = self._make_settings(cam_idx)

        # Fresh densification_info buffer per call. Passing the same tensor
        # across multiple diff_rasterize calls (even with zero_() between) has
        # triggered CUDA illegal-memory-access in the backward pass when the
        # trainer mixes no_grad TB-logger forwards with grad-enabled forwards.
        N = means.shape[0]
        dinfo = torch.zeros(N, device=self.device)

        image = diff_rasterize(
            means.contiguous(),
            log_scales.contiguous(),       # RAW, kernel does exp(2*raw)
            quats.contiguous(),
            raw_opacities.contiguous(),    # RAW, kernel does sigmoid
            sh0.contiguous(),
            shN.contiguous(),
            dinfo,
            settings,
        )  # (3, H, W)

        render_colors = image.permute(1, 2, 0).unsqueeze(0)        # (1, H, W, 3)
        H, W = image.shape[1], image.shape[2]
        render_alphas = torch.ones(1, H, W, 1, device=self.device)  # dummy
        return render_colors, render_alphas, {}

    # -----------------------------------------------------------------------
    # Loss functions
    # -----------------------------------------------------------------------
    def compute_tracking_loss(
        self, ct_idx: int, frame_idx: int, means_dynamic: Tensor
    ) -> Tensor:
        cam_idx = self.ct_to_cam[ct_idx]
        viewmat = self.viewmats[cam_idx]
        K = self.Ks[cam_idx]

        proj_2d, _ = project_points(means_dynamic, viewmat, K)
        bound_idx = self.binding_map[ct_idx]
        pred_2d = proj_2d[bound_idx]

        gt_2d = self.gt_tracks[ct_idx, frame_idx]
        vis = self.gt_vis[ct_idx, frame_idx] & self.binding_valid[ct_idx]
        if vis.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        return F.mse_loss(pred_2d[vis], gt_2d[vis])

    def compute_temporal_loss(self) -> Tensor:
        motion = self.splats["motion_offsets"]
        diff = motion[:, 1:, :] - motion[:, :-1, :]
        return diff.pow(2).mean()

    def compute_rigidity_loss(self, frame_idx: int) -> Tensor:
        motion_t = self.splats["motion_offsets"][:, frame_idx, :]
        neighbor_motion = motion_t[self.knn_indices]
        diff = motion_t.unsqueeze(1) - neighbor_motion
        return diff.pow(2).mean()

    def compute_photometric_loss(
        self, rendered: Tensor, gt_image: Tensor
    ) -> Tuple[Tensor, float, float, float]:
        pixels = gt_image.unsqueeze(0)
        h = min(rendered.shape[1], pixels.shape[1])
        w = min(rendered.shape[2], pixels.shape[2])
        rendered = rendered[:, :h, :w, :]
        pixels = pixels[:, :h, :w, :]
        l1 = F.l1_loss(rendered, pixels)
        ssim_metric = fused_ssim(
            rendered.permute(0, 3, 1, 2),
            pixels.permute(0, 3, 1, 2),
            padding="valid",
        )
        ssim_loss = 1.0 - ssim_metric
        loss = l1 * (1.0 - self.cfg.ssim_lambda) + ssim_loss * self.cfg.ssim_lambda
        mse = F.mse_loss(rendered.clamp(0, 1), pixels).item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        return loss, l1.item(), ssim_metric.item(), psnr

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        device = self.device

        cfg_dict = {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in vars(cfg).items()}
        with open(os.path.join(cfg.result_dir, "cfg.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        with open(os.path.join(cfg.result_dir, "cfg.yaml"), "w") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)

        num_ct_cams = len(self.ct_to_cam)
        num_frames = cfg.num_cotracker_frames

        self._motion_scheduler_gamma = 0.1 ** (1.0 / cfg.max_steps)
        motion_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.motion_optimizer, gamma=self._motion_scheduler_gamma,
        )
        # Store ref so _replace_dyn_splats can rebuild after structural changes
        self._motion_scheduler_ref = motion_scheduler

        # Boundary of stage 1 (density+mcmc) and stage 2 (refinement)
        refinement_start = cfg.max_steps - cfg.refinement_steps
        print(f"[Stages] density: 0..{refinement_start}  refinement: {refinement_start}..{cfg.max_steps}")

        global_tic = time.time()
        pbar = tqdm.tqdm(range(cfg.max_steps))

        def _sync_tick():
            torch.cuda.synchronize()
            return time.perf_counter()

        for step in pbar:
            step_t0 = _sync_tick()
            frame_idx = random.randint(0, num_frames - 1)
            means_dynamic = self.splats["means"] + self.splats["motion_offsets"][:, frame_idx, :]

            # --- tracking loss ---
            t0 = _sync_tick()
            loss_track = torch.tensor(0.0, device=device)
            for ct_idx in range(num_ct_cams):
                loss_track = loss_track + self.compute_tracking_loss(
                    ct_idx, frame_idx, means_dynamic
                )
            loss_track = loss_track / num_ct_cams
            t_track = (_sync_tick() - t0) * 1000.0

            # --- temporal + rigid ---
            t0 = _sync_tick()
            loss_temp = self.compute_temporal_loss()
            loss_rigid = self.compute_rigidity_loss(frame_idx)
            t_reg = (_sync_tick() - t0) * 1000.0

            # --- photometric (rasterize + loss) ---
            loss_photo = torch.tensor(0.0, device=device)
            l1_val = ssim_val = psnr_val = 0.0
            t_imgload = t_raster = t_photo = 0.0
            if step >= cfg.freeze_appearance_steps:
                ct_idx_photo = random.randint(0, num_ct_cams - 1)
                cam_idx_photo = self.ct_to_cam[ct_idx_photo]
                t0 = _sync_tick()
                gt_image = self.load_image(cam_idx_photo, frame_idx)
                t_imgload = (_sync_tick() - t0) * 1000.0

                t0 = _sync_tick()
                rendered, _, _ = self.rasterize_splats(cam_idx_photo, frame_idx)
                t_raster = (_sync_tick() - t0) * 1000.0

                t0 = _sync_tick()
                loss_photo, l1_val, ssim_val, psnr_val = self.compute_photometric_loss(
                    rendered, gt_image
                )
                t_photo = (_sync_tick() - t0) * 1000.0

            loss = (
                cfg.track_loss_weight * loss_track
                + cfg.temporal_smooth_weight * loss_temp
                + cfg.spatial_rigid_weight * loss_rigid
            )
            if step >= cfg.freeze_appearance_steps:
                loss = loss + cfg.photo_loss_weight * loss_photo

            # --- backward ---
            t0 = _sync_tick()
            loss.backward()
            t_bwd = (_sync_tick() - t0) * 1000.0

            # --- optimizer step ---
            t0 = _sync_tick()
            self.motion_optimizer.step()
            self.motion_optimizer.zero_grad(set_to_none=True)
            for opt in self.appearance_optimizers.values():
                if step >= cfg.freeze_appearance_steps:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            # Use the scheduler ref (may have been rebuilt by densify/prune)
            self._motion_scheduler_ref.step()
            t_opt = (_sync_tick() - t0) * 1000.0

            # --- MCMC noise (stage 1 only; skipped in refinement) ---
            in_density_stage = step < refinement_start
            if (cfg.use_mcmc_noise and in_density_stage
                    and step >= cfg.mcmc_noise_start):
                self._mcmc_noise_step(cfg.mcmc_noise_lr)

            # --- Periodic densify+prune (stage 1 only) ---
            if (cfg.use_densify_prune and in_density_stage
                    and cfg.densify_start <= step < cfg.densify_end
                    and step > 0 and step % cfg.densify_interval == 0):
                t0 = _sync_tick()
                self._densify_and_prune(step)
                motion_scheduler = self._motion_scheduler_ref  # keep local name in sync
                t_struct = (_sync_tick() - t0) * 1000.0
                self.writer.add_scalar("time/densify_prune_ms", t_struct, step)

            # --- Unsolvable-Gaussian regularization (paper §5.2) ---
            if (cfg.use_unsolvable_reg and in_density_stage
                    and cfg.unsolvable_start <= step < cfg.unsolvable_end
                    and step > 0 and step % cfg.unsolvable_interval == 0):
                t0 = _sync_tick()
                unsolv_mask = self._detect_unsolvable_dyn(cfg.unsolvable_score_views)
                self._repair_unsolvable_dyn(unsolv_mask, step)
                t_unsolv = (_sync_tick() - t0) * 1000.0
                self.writer.add_scalar("time/unsolvable_reg_ms", t_unsolv, step)

            t_step_total = (_sync_tick() - step_t0) * 1000.0
            it_per_sec = 1000.0 / max(t_step_total, 1e-6)
            avg_it_per_sec = (step + 1) / max(time.time() - global_tic, 1e-6)
            self.writer.add_scalar("time/step_total_ms", t_step_total, step)
            self.writer.add_scalar("time/it_per_sec",   it_per_sec,   step)
            self.writer.add_scalar("time/it_per_sec_avg", avg_it_per_sec, step)
            # ETA in seconds assuming constant avg rate
            eta_sec = max(cfg.max_steps - step - 1, 0) / max(avg_it_per_sec, 1e-6)
            self.writer.add_scalar("time/eta_sec", eta_sec, step)
            self.writer.add_scalar("time/tracking_ms",   t_track,      step)
            self.writer.add_scalar("time/reg_ms",        t_reg,        step)
            self.writer.add_scalar("time/backward_ms",   t_bwd,        step)
            self.writer.add_scalar("time/opt_step_ms",   t_opt,        step)
            if step >= cfg.freeze_appearance_steps:
                self.writer.add_scalar("time/image_load_ms", t_imgload, step)
                self.writer.add_scalar("time/raster_fwd_ms", t_raster,  step)
                self.writer.add_scalar("time/photo_loss_ms", t_photo,   step)

            desc = (
                f"loss={loss.item():.4f} "
                f"trk={loss_track.item():.4f} "
                f"tmp={loss_temp.item():.6f} "
                f"rig={loss_rigid.item():.6f}"
            )
            if step >= cfg.freeze_appearance_steps:
                desc += f" pho={loss_photo.item():.4f} psnr={psnr_val:.2f}"
            pbar.set_description(desc)

            self.writer.add_scalar("train/loss_total", loss.item(), step)
            self.writer.add_scalar("train/loss_track", loss_track.item(), step)
            self.writer.add_scalar("train/loss_temp", loss_temp.item(), step)
            self.writer.add_scalar("train/loss_rigid", loss_rigid.item(), step)
            self.writer.add_scalar(
                "train/lr_motion",
                self.motion_optimizer.param_groups[0]["lr"],
                step,
            )
            self.writer.add_scalar(
                "train/num_gaussians", self.splats["means"].shape[0], step
            )
            if step >= cfg.freeze_appearance_steps:
                self.writer.add_scalar("train/loss_photo", loss_photo.item(), step)
                self.writer.add_scalar("train/l1", l1_val, step)
                self.writer.add_scalar("train/ssim", ssim_val, step)
                self.writer.add_scalar("train/psnr", psnr_val, step)

            if step % cfg.tb_image_every == 0:
                self._log_tb_images(step)
                self._log_tb_eval_sample(step)

            if step + 1 in cfg.save_steps or step == cfg.max_steps - 1:
                ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{step}.pt")
                torch.save({"step": step, "splats": self.splats.state_dict()}, ckpt_path)
                print(f"\n[Checkpoint] Saved to {ckpt_path}")

            if step + 1 in cfg.eval_steps:
                self.eval_tracking(step)

            if step + 1 in cfg.ply_save_steps:
                self.export_per_frame_ply(step)

        elapsed = time.time() - global_tic
        print(f"\n[Done] Training finished in {elapsed:.1f}s")

        for cam_render_idx in cfg.render_video_cams:
            if cam_render_idx < self.num_cameras:
                self.render_video(cam_render_idx)

        self.writer.flush()
        self.writer.close()

    # -----------------------------------------------------------------------
    # TB images
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _log_tb_images(self, step: int):
        if len(self.ct_to_cam) == 0:
            return
        cam_idx = self.ct_to_cam[0]
        frame_idx = self.cfg.num_cotracker_frames // 2
        rendered, _, _ = self.rasterize_splats(cam_idx, frame_idx)
        gt = self.load_image(cam_idx, frame_idx).unsqueeze(0)
        h = min(rendered.shape[1], gt.shape[1])
        w = min(rendered.shape[2], gt.shape[2])
        rendered = rendered[:, :h, :w, :].clamp(0, 1)
        gt = gt[:, :h, :w, :]

        concat = torch.cat([gt[0], rendered[0]], dim=1)
        self.writer.add_image("images/gt_vs_rendered", concat.permute(2, 0, 1), step)

        mse = F.mse_loss(rendered, gt).item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        ssim = fused_ssim(
            rendered.permute(0, 3, 1, 2),
            gt.permute(0, 3, 1, 2),
            padding="valid",
        ).item()
        l1 = F.l1_loss(rendered, gt).item()
        self.writer.add_scalar("image_view/psnr", psnr, step)
        self.writer.add_scalar("image_view/ssim", ssim, step)
        self.writer.add_scalar("image_view/l1", l1, step)

    # -----------------------------------------------------------------------
    # Lightweight eval: 3 cams x 3 frames, pushed to TB every tb_image_every
    # steps. Complements the heavy full eval at eval_steps.
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _log_tb_eval_sample(self, step: int):
        cfg = self.cfg
        n_cams = min(cfg.tb_eval_sample_cams, len(self.ct_to_cam))
        n_frames = min(cfg.tb_eval_sample_frames, cfg.num_cotracker_frames)
        if n_cams == 0 or n_frames == 0:
            return
        cam_picks = list(range(0, len(self.ct_to_cam),
                               max(1, len(self.ct_to_cam) // n_cams)))[:n_cams]
        frame_picks = list(range(0, cfg.num_cotracker_frames,
                                 max(1, cfg.num_cotracker_frames // n_frames)))[:n_frames]

        total_mse = 0.0; total_l1 = 0.0; total_ssim = 0.0
        total_err = 0.0; total_pts = 0; n_views = 0

        for ct_idx in cam_picks:
            cam_idx = self.ct_to_cam[ct_idx]
            bound_idx = self.binding_map[ct_idx]
            for t in frame_picks:
                means_t = self.splats["means"] + self.splats["motion_offsets"][:, t, :]
                proj_2d, _ = project_points(
                    means_t, self.viewmats[cam_idx], self.Ks[cam_idx]
                )
                pred_2d = proj_2d[bound_idx]
                gt_2d = self.gt_tracks[ct_idx, t]
                vis = self.gt_vis[ct_idx, t] & self.binding_valid[ct_idx]
                if vis.sum() > 0:
                    err = (pred_2d[vis] - gt_2d[vis]).norm(dim=-1)
                    total_err += err.sum().item()
                    total_pts += int(vis.sum().item())

                rendered, _, _ = self.rasterize_splats(cam_idx, t)
                gt = self.load_image(cam_idx, t).unsqueeze(0)
                h = min(rendered.shape[1], gt.shape[1])
                w = min(rendered.shape[2], gt.shape[2])
                rendered = rendered[:, :h, :w, :].clamp(0, 1)
                gt = gt[:, :h, :w, :]
                total_mse += F.mse_loss(rendered, gt).item()
                total_l1 += F.l1_loss(rendered, gt).item()
                total_ssim += fused_ssim(
                    rendered.permute(0, 3, 1, 2),
                    gt.permute(0, 3, 1, 2),
                    padding="valid",
                ).item()
                n_views += 1

        mean_mse = total_mse / max(n_views, 1)
        psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))
        self.writer.add_scalar("eval/psnr", psnr, step)
        self.writer.add_scalar("eval/ssim", total_ssim / max(n_views, 1), step)
        self.writer.add_scalar("eval/l1",   total_l1  / max(n_views, 1), step)
        self.writer.add_scalar("eval/tracking_error_px",
                               total_err / max(total_pts, 1), step)

    # -----------------------------------------------------------------------
    # Eval / export / video — unchanged from the gsplat version
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def eval_tracking(self, step: int):
        cfg = self.cfg
        total_err = 0.0
        total_pts = 0
        total_mse = 0.0
        total_l1 = 0.0
        total_ssim = 0.0
        n_views = 0

        for ct_idx in tqdm.tqdm(range(len(self.ct_to_cam)),
                                desc=f"eval@{step}", leave=False):
            cam_idx = self.ct_to_cam[ct_idx]
            bound_idx = self.binding_map[ct_idx]
            for t in range(cfg.num_cotracker_frames):
                means_t = self.splats["means"] + self.splats["motion_offsets"][:, t, :]
                proj_2d, _ = project_points(
                    means_t, self.viewmats[cam_idx], self.Ks[cam_idx]
                )
                pred_2d = proj_2d[bound_idx]
                gt_2d = self.gt_tracks[ct_idx, t]
                vis = self.gt_vis[ct_idx, t] & self.binding_valid[ct_idx]
                if vis.sum() > 0:
                    err = (pred_2d[vis] - gt_2d[vis]).norm(dim=-1)
                    total_err += err.sum().item()
                    total_pts += vis.sum().item()

                rendered, _, _ = self.rasterize_splats(cam_idx, t)
                gt = self.load_image(cam_idx, t).unsqueeze(0)
                h = min(rendered.shape[1], gt.shape[1])
                w = min(rendered.shape[2], gt.shape[2])
                rendered = rendered[:, :h, :w, :].clamp(0, 1)
                gt = gt[:, :h, :w, :]
                total_mse += F.mse_loss(rendered, gt).item()
                total_l1 += F.l1_loss(rendered, gt).item()
                total_ssim += fused_ssim(
                    rendered.permute(0, 3, 1, 2),
                    gt.permute(0, 3, 1, 2),
                    padding="valid",
                ).item()
                n_views += 1

        mean_err = total_err / max(total_pts, 1)
        mean_mse = total_mse / max(n_views, 1)
        psnr = 10.0 * math.log10(1.0 / max(mean_mse, 1e-12))
        mean_l1 = total_l1 / max(n_views, 1)
        mean_ssim = total_ssim / max(n_views, 1)

        print(
            f"\n[Eval @ step {step}] trk={mean_err:.2f}px  "
            f"PSNR={psnr:.3f}  SSIM={mean_ssim:.4f}  L1={mean_l1:.4f}  "
            f"({total_pts} pts, {n_views} views)"
        )
        self.writer.add_scalar("eval/tracking_error_px", mean_err, step)
        self.writer.add_scalar("eval/psnr", psnr, step)
        self.writer.add_scalar("eval/ssim", mean_ssim, step)
        self.writer.add_scalar("eval/l1", mean_l1, step)

        metrics = {
            "step": step,
            "mean_tracking_error_px": mean_err,
            "psnr": psnr,
            "ssim": mean_ssim,
            "l1": mean_l1,
            "num_views": n_views,
            "num_visible_pts": total_pts,
        }
        with open(os.path.join(cfg.result_dir, f"eval_{step}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    @torch.no_grad()
    def export_per_frame_ply(self, step: int):
        cfg = self.cfg
        ply_dir = os.path.join(cfg.result_dir, "ply", f"step_{step}")
        os.makedirs(ply_dir, exist_ok=True)
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]
        scales = self.splats["scales"]
        quats = self.splats["quats"]
        opacities = self.splats["opacities"]
        for t in range(cfg.num_cotracker_frames):
            means_t = self.splats["means"] + self.splats["motion_offsets"][:, t, :]
            save_path = os.path.join(ply_dir, f"frame_{t:03d}.ply")
            export_splats(
                means=means_t, scales=scales, quats=quats,
                opacities=opacities, sh0=sh0, shN=shN,
                format="ply", save_to=save_path,
            )
        print(f"\n[PLY] Exported {cfg.num_cotracker_frames} frames to {ply_dir}/")

    @torch.no_grad()
    def render_video(self, cam_idx: int):
        cfg = self.cfg
        cam_name = self.cam_names[cam_idx]
        print(f"[Video] Rendering {cfg.num_cotracker_frames} frames from {cam_name}...")
        frames = []
        for t in tqdm.tqdm(range(cfg.num_cotracker_frames), desc=f"Rendering {cam_name}"):
            rendered, _, _ = self.rasterize_splats(cam_idx, t)
            frame = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
        video_path = os.path.join(self.video_dir, f"{cam_name}_motion.mp4")
        imageio.mimwrite(video_path, frames, fps=10, quality=8)
        print(f"[Video] Saved to {video_path}")

    def load_checkpoint(self, ckpt_path: str):
        data = torch.load(ckpt_path, map_location=self.device)
        self.splats.load_state_dict(data["splats"])
        print(f"[Checkpoint] Loaded from {ckpt_path} (step {data['step']})")

    # =======================================================================
    # MCMC densification / pruning / refinement (paper §5.2)
    # =======================================================================
    @torch.no_grad()
    def _mcmc_noise_step(self, current_lr: float):
        """Apply FasterGS add_noise to the NON-ANCHOR dynamic splats only.
        Anchors are the first `n_anchors` splats in the tensor and are kept
        untouched so their binding to cotracker tracks stays exact.
        """
        K = self.n_anchors
        N = self.splats["means"].shape[0]
        if K >= N:
            return
        # Contiguous tail-slice of a dim-0 contiguous tensor is itself
        # contiguous; in-place writes on the view propagate back.
        na_scales = self.splats["scales"].data[K:]
        na_quats  = self.splats["quats"].data[K:]
        na_opac   = self.splats["opacities"].data[K:]
        if na_opac.dim() == 1:
            na_opac = na_opac.unsqueeze(-1)
        na_means  = self.splats["means"].data[K:]    # mutated in place
        add_noise(na_scales, na_quats, na_opac, na_means, current_lr)

    @torch.no_grad()
    def _compose_splats_all(self, frame_idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build a (static + dynamic) composite in the same order rasterize_splats
        uses. Returns activation-free tensors ready for FasterGS kernels."""
        dyn_means = self.splats["means"] + self.splats["motion_offsets"][:, frame_idx, :]
        if self.static is None:
            means = dyn_means
            quats = self.splats["quats"]
            log_scales = self.splats["scales"]
            raw_opac = self.splats["opacities"]
            sh0 = self.splats["sh0"]
            shN = self.splats["shN"]
        else:
            means = torch.cat([self.static["means"], dyn_means], 0)
            quats = torch.cat([self.static["quats"], self.splats["quats"]], 0)
            log_scales = torch.cat([self.static["scales"], self.splats["scales"]], 0)
            raw_opac = torch.cat([self.static["opacities"], self.splats["opacities"]], 0)
            sh0 = torch.cat([self.static["sh0"], self.splats["sh0"]], 0)
            shN = torch.cat([self.static["shN"], self.splats["shN"]], 0)
        if raw_opac.dim() == 1:
            raw_opac = raw_opac.unsqueeze(-1)
        return (means.contiguous(), log_scales.contiguous(), quats.contiguous(),
                raw_opac.contiguous(), sh0.contiguous(), shN.contiguous())

    @torch.no_grad()
    def _accumulate_pruning_scores(self, n_views: int) -> Tensor:
        """Accumulate FasterGS per-primitive importance over random views.
        Returns (N_dyn,) — only the dynamic-range slice is returned, since
        static primitives are frozen and never pruned/grown."""
        cfg = self.cfg
        n_stat = 0 if self.static is None else self.static["means"].shape[0]
        N_total = n_stat + self.splats["means"].shape[0]
        scores = torch.zeros(N_total, device=self.device)

        num_ct = len(self.ct_to_cam)
        num_f  = cfg.num_cotracker_frames
        for _ in range(n_views):
            ct_idx = random.randint(0, num_ct - 1)
            cam_idx = self.ct_to_cam[ct_idx]
            t = random.randint(0, num_f - 1)
            means, log_s, q, op, sh0, shN = self._compose_splats_all(t)
            update_pruning_scores(scores, means, log_s, q, op, sh0, shN,
                                  self._make_settings(cam_idx))
        return scores[n_stat:]            # (N_dyn,)

    def _rebuild_optimizers_with_new_params(self):
        """Rebuild motion + appearance optimizers after splat Parameters have
        been replaced in-place. Adam momentum is lost, which is acceptable for
        infrequent prune/grow events."""
        self._setup_optimizers()
        # Re-attach the LR scheduler to the new motion optimizer.
        if hasattr(self, "_motion_scheduler_ref"):
            self._motion_scheduler_ref = torch.optim.lr_scheduler.ExponentialLR(
                self.motion_optimizer, gamma=self._motion_scheduler_gamma,
            )

    @torch.no_grad()
    def _replace_dyn_splats(self, new_tensors: Dict[str, Tensor],
                             new_motion: Tensor, new_anchor_mask: Tensor):
        """Replace the ParameterDict contents with new tensors of possibly
        different N. Keeps the anchors-first invariant: we assume the caller
        built new_tensors such that the first `n_anchors` rows are anchors.
        Rebuilds optimizers and KNN."""
        for name, tensor in new_tensors.items():
            self.splats[name] = torch.nn.Parameter(tensor.contiguous())
        self.splats["motion_offsets"] = torch.nn.Parameter(new_motion.contiguous())
        self.anchor_mask = new_anchor_mask
        self.n_anchors = int(new_anchor_mask.sum())    # invariant: == old n_anchors
        # Rebuild KNN for rigidity loss
        self.knn_indices = compute_knn(self.splats["means"].detach(), self.cfg.knn_k)
        self._rebuild_optimizers_with_new_params()

    @torch.no_grad()
    def _prune_non_anchors(self, scores: Tensor, step: int) -> int:
        """Remove the bottom `prune_score_quantile` fraction of non-anchor
        splats by score. Anchors are never touched; we also respect
        `min_dyn_gaussians` as a hard floor. Uses bottom-N-by-score rather
        than threshold-on-quantile because most pruning scores are exactly
        zero — a quantile threshold of 0 drops nothing."""
        cfg = self.cfg
        K = self.n_anchors
        N = self.splats["means"].shape[0]
        n_na = N - K
        if n_na <= 0:
            return 0
        n_drop = int(n_na * cfg.prune_score_quantile)
        # Hard floor on total N_dyn
        n_drop = min(n_drop, N - cfg.min_dyn_gaussians)
        if n_drop <= 0:
            return 0

        na_scores = scores[K:]
        # Take indices of the n_drop lowest-scoring non-anchor splats.
        drop_rel = torch.topk(na_scores, n_drop, largest=False).indices  # in [0, n_na)
        keep_rel = torch.ones(n_na, dtype=torch.bool, device=self.device)
        keep_rel[drop_rel] = False
        # Full-N keep mask: anchors always kept + non-anchor keeps
        keep = torch.cat([
            torch.ones(K, dtype=torch.bool, device=self.device),
            keep_rel,
        ], dim=0)

        new_tensors = {name: self.splats[name].data[keep]
                       for name in ("means", "scales", "quats", "opacities", "sh0", "shN")}
        new_motion = self.splats["motion_offsets"].data[keep]
        new_anchor = self.anchor_mask[keep]
        # n_anchors is invariant (we never drop anchors)
        self._replace_dyn_splats(new_tensors, new_motion, new_anchor)
        # n_anchors stays the same, assert for safety
        assert int(new_anchor[:K].all()) == 1
        self.writer.add_scalar("struct/prune_count", n_drop, step)
        return n_drop

    @torch.no_grad()
    def _grow_non_anchors(self, scores: Tensor, step: int) -> int:
        """Duplicate top-score non-anchor splats with small position jitter and
        apply FasterGS relocation_adjustment to shrink opacity/scale so the
        parent+child pair reconstructs roughly the original photometric
        contribution. Returns number of new splats added. Respects
        `max_dyn_gaussians` cap."""
        cfg = self.cfg
        N = self.splats["means"].shape[0]
        non_anchor = ~self.anchor_mask
        if non_anchor.sum() == 0:
            return 0
        # budget
        budget = cfg.max_dyn_gaussians - N
        if budget <= 0:
            return 0
        na_count = int(non_anchor.sum())
        n_clone = min(budget, int(cfg.grow_max_fraction * na_count))
        if n_clone <= 0:
            return 0
        # rank non-anchor splats by score, take top n_clone
        na_indices = torch.where(non_anchor)[0]
        na_scores = scores[na_indices]
        top_k = torch.topk(na_scores, n_clone, largest=True).indices
        src_idx = na_indices[top_k]

        # MCMC relocation adjustment: for each cloned splat, we now represent
        # the same blob with 2 primitives (parent + child), so each gets a
        # shrunk opacity/scale. relocation_adjustment expects activated values.
        src_opac_act = torch.sigmoid(self.splats["opacities"].data[src_idx]).unsqueeze(-1)  # (K, 1)
        src_scale_act = self.splats["scales"].data[src_idx].exp()                            # (K, 3)
        n_samples = torch.full((n_clone,), 2, dtype=torch.int64, device=self.device)
        new_opac_act, new_scale_act = relocation_adjustment(
            src_opac_act, src_scale_act, n_samples,
        )
        # back to raw
        new_raw_opac = torch.logit(new_opac_act.clamp(1e-6, 1 - 1e-6))    # (K, 1)
        new_raw_scale = new_scale_act.clamp_min(1e-8).log()                # (K, 3)

        # Update parent in-place to the adjusted values
        self.splats["opacities"].data[src_idx] = new_raw_opac.squeeze(-1)
        self.splats["scales"].data[src_idx] = new_raw_scale

        # Build child tensors (duplicate + jitter means)
        jitter = torch.randn(n_clone, 3, device=self.device) * cfg.jitter_scale_world
        child = {
            "means": self.splats["means"].data[src_idx] + jitter,
            "scales": new_raw_scale.clone(),
            "quats": self.splats["quats"].data[src_idx].clone(),
            "opacities": new_raw_opac.squeeze(-1).clone(),
            "sh0": self.splats["sh0"].data[src_idx].clone(),
            "shN": self.splats["shN"].data[src_idx].clone(),
        }
        child_motion = self.splats["motion_offsets"].data[src_idx].clone()

        # Concatenate old + new
        new_tensors = {}
        for name in ("means", "scales", "quats", "opacities", "sh0", "shN"):
            new_tensors[name] = torch.cat([self.splats[name].data, child[name]], dim=0)
        new_motion = torch.cat([self.splats["motion_offsets"].data, child_motion], dim=0)
        new_anchor = torch.cat([self.anchor_mask,
                                torch.zeros(n_clone, dtype=torch.bool, device=self.device)],
                               dim=0)
        self._replace_dyn_splats(new_tensors, new_motion, new_anchor)
        self.writer.add_scalar("struct/grow_count", n_clone, step)
        return n_clone

    # --------------- Unsolvable-Gaussian regularization (§5.2) ----------
    @torch.no_grad()
    def _detect_unsolvable_dyn(self, n_views: int) -> Tensor:
        """Return bool mask over DYNAMIC splats marking those that are
        unobservable in random views (paper §5.2): visible in fewer than
        `min_views_visible` views OR pruning-score below threshold."""
        cfg = self.cfg
        n_stat = 0 if self.static is None else self.static["means"].shape[0]
        N_total = n_stat + self.splats["means"].shape[0]

        view_count = torch.zeros(N_total, dtype=torch.int32, device=self.device)
        score_total = torch.zeros(N_total, device=self.device)

        num_ct = len(self.ct_to_cam)
        num_f  = cfg.num_cotracker_frames
        for _ in range(n_views):
            ct_idx = random.randint(0, num_ct - 1)
            cam_idx = self.ct_to_cam[ct_idx]
            t = random.randint(0, num_f - 1)
            means, log_s, q, op, sh0, shN = self._compose_splats_all(t)

            # fresh per-view score buffer; a splat with >0 contribution is
            # visible in this view (proxy for alpha accumulation + pixel coverage).
            per_view = torch.zeros(N_total, device=self.device)
            update_pruning_scores(per_view, means, log_s, q, op, sh0, shN,
                                  self._make_settings(cam_idx))
            view_count += (per_view > 0).to(torch.int32)
            score_total += per_view

        dyn_views = view_count[n_stat:]
        dyn_scores = score_total[n_stat:]
        unsolvable = (
            (dyn_views < cfg.min_views_visible) |
            (dyn_scores < cfg.min_prune_score_unsolvable)
        )
        return unsolvable

    @torch.no_grad()
    def _repair_unsolvable_dyn(self, unsolvable_mask: Tensor, step: int) -> int:
        """For each unsolvable non-anchor splat, copy the mean of its 8 nearest
        *solvable* (and non-unsolvable) neighbours into every param + the
        motion_offsets. Anchors are never touched, since their positions are
        bound to cotracker tracks."""
        cfg = self.cfg
        K = self.n_anchors
        N = self.splats["means"].shape[0]

        # Restrict repair to non-anchors
        tail = torch.zeros(N, dtype=torch.bool, device=self.device)
        tail[K:] = True
        unsolv = unsolvable_mask & tail
        n_unsolv = int(unsolv.sum())
        if n_unsolv == 0:
            return 0

        # Donors: solvable splats (NOT unsolvable). Anchors are always donors.
        solvable = ~unsolvable_mask
        if int(solvable.sum()) < cfg.unsolvable_k_nn:
            return 0
        solv_idx  = torch.where(solvable)[0]           # (M,)
        solv_means = self.splats["means"].data[solv_idx]  # (M, 3)
        unsolv_idx = torch.where(unsolv)[0]             # (U,)
        unsolv_means = self.splats["means"].data[unsolv_idx]

        # Chunked cdist to find k-NN among solvable donors for every unsolvable
        k = cfg.unsolvable_k_nn
        chunk = 2048
        nn_rel_all = []
        for i in range(0, unsolv_means.shape[0], chunk):
            d = torch.cdist(unsolv_means[i:i+chunk], solv_means)
            _, rel = d.topk(k, dim=-1, largest=False)
            nn_rel_all.append(rel)
        nn_rel = torch.cat(nn_rel_all, dim=0)           # (U, k)
        nn_abs = solv_idx[nn_rel]                        # (U, k) absolute dynamic indices

        # Replace each unsolvable splat with the mean of its k donors.
        names = ("means", "scales", "quats", "opacities", "sh0", "shN", "motion_offsets")
        for name in names:
            param = self.splats[name].data          # (N, ...)
            gathered = param[nn_abs]                # (U, k, ...)
            avg = gathered.mean(dim=1)              # (U, ...)
            param[unsolv_idx] = avg

        self.writer.add_scalar("struct/unsolvable_count", n_unsolv, step)
        print(f"\n[Unsolvable@{step}] {n_unsolv} non-anchor splats repaired via {k}-NN "
              f"(solvable donors: {int(solvable.sum())})")
        return n_unsolv

    @torch.no_grad()
    def _densify_and_prune(self, step: int):
        """Run one densify+prune event: accumulate pruning scores over a few
        random views, prune bottom-q non-anchors, grow top-q non-anchors."""
        cfg = self.cfg
        scores = self._accumulate_pruning_scores(cfg.densify_score_views)
        n_pruned = self._prune_non_anchors(scores, step)
        # Scores are stale w.r.t. the new N after pruning, but for growing we
        # re-accumulate to keep the decision consistent.
        if n_pruned > 0 or True:
            scores = self._accumulate_pruning_scores(cfg.densify_score_views)
        n_grown = self._grow_non_anchors(scores, step)
        self.writer.add_scalar("struct/n_dyn_gaussians",
                               self.splats["means"].shape[0], step)
        self.writer.add_scalar("struct/n_anchors", int(self.anchor_mask.sum()), step)
        print(f"\n[Densify@{step}] -{n_pruned} pruned   +{n_grown} grown   "
              f"N_dyn={self.splats['means'].shape[0]}  "
              f"N_anchors={int(self.anchor_mask.sum())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = tyro.cli(Config)
    runner = TrackerSplatRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
