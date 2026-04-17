"""
TrackerSplat 4D Gaussian Tracking Trainer

Learns per-Gaussian 3D motion offsets guided by CoTracker 2D trajectories.
Uses gsplat rasterization and a pre-trained static PLY as canonical frame.

Usage:
    conda activate gsplat
    python examples/trackersplat_trainer.py \
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
from gsplat.rendering import rasterization


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Paths
    data_dir: str = ""
    ply_path: str = ""  # dynamic / foreground PLY (trainable)
    static_ply_path: str = ""  # optional static / background PLY (frozen, no motion)
    cotracker_dir: str = ""
    result_dir: str = "/data/shared/elaheh/4D_demo/new_data/trackersplat_results"

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
    packed: bool = True
    data_factor: int = 15
    normalize_world_space: bool = True  # must match the PLY's training-time setting

    # KNN for rigidity
    knn_k: int = 5

    # Misc
    seed: int = 42
    render_video_cams: List[int] = field(default_factory=lambda: [0, 22, 44])

    # TensorBoard
    tb_image_every: int = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def project_points(means_3d: Tensor, viewmat: Tensor, K: Tensor) -> Tuple[Tensor, Tensor]:
    """Project 3D points to 2D pixel coordinates (differentiable).

    Args:
        means_3d: (N, 3)
        viewmat: (4, 4) world-to-camera
        K: (3, 3) intrinsics

    Returns:
        pixels: (N, 2) pixel coordinates
        depths: (N,) depths in camera frame
    """
    R = viewmat[:3, :3]  # (3, 3)
    t = viewmat[:3, 3]   # (3,)
    cam_pts = means_3d @ R.T + t[None, :]  # (N, 3)
    z = cam_pts[:, 2:3].clamp(min=0.01)    # (N, 1)
    uv = cam_pts[:, :2] / z               # (N, 2)
    px = K[0, 0] * uv[:, 0] + K[0, 2]
    py = K[1, 1] * uv[:, 1] + K[1, 2]
    return torch.stack([px, py], dim=-1), z.squeeze(-1)


def compute_knn(points: Tensor, k: int) -> Tensor:
    """Find k-nearest neighbors using batched cdist (GPU-friendly).

    Args:
        points: (N, 3)
        k: number of neighbors

    Returns:
        indices: (N, k) neighbor indices
    """
    N = points.shape[0]
    chunk = 4096
    all_indices = []
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        dists = torch.cdist(points[i:end], points)  # (chunk, N)
        # Exclude self by setting diagonal to inf
        for j in range(end - i):
            dists[j, i + j] = float("inf")
        _, idx = dists.topk(k, dim=-1, largest=False)  # (chunk, k)
        all_indices.append(idx)
    return torch.cat(all_indices, dim=0)  # (N, k)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class TrackerSplatRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda")
        set_random_seed(cfg.seed)

        # Create output directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.render_dir = os.path.join(cfg.result_dir, "renders")
        self.ckpt_dir = os.path.join(cfg.result_dir, "ckpts")
        self.video_dir = os.path.join(cfg.result_dir, "videos")
        self.tb_dir = os.path.join(cfg.result_dir, "tb")
        os.makedirs(self.render_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # 1. Load camera poses via COLMAP Parser
        print("[Init] Loading camera poses...")
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=9999,
            frame_num=1,
            skip_points3d=True,
        )
        self.num_cameras = len(self.parser.image_names)
        print(f"[Init] {self.num_cameras} cameras loaded")

        # Build camera arrays on GPU
        self.camtoworlds = torch.from_numpy(self.parser.camtoworlds).float().to(self.device)  # (C, 4, 4)
        self.viewmats = torch.linalg.inv(self.camtoworlds)  # (C, 4, 4)

        # Build per-camera intrinsics and image sizes
        self.Ks = []
        self.img_sizes = []  # (width, height) per camera
        self.cam_names = []  # camera directory names for image loading
        for i, name in enumerate(self.parser.image_names):
            cam_dir = os.path.dirname(name)
            self.cam_names.append(cam_dir)
            cam_id = self.parser.camera_ids[i]
            K = torch.from_numpy(self.parser.Ks_dict[cam_id]).float().to(self.device)
            self.Ks.append(K)
            self.img_sizes.append(self.parser.imsize_dict[cam_id])
        self.Ks_tensor = torch.stack(self.Ks, dim=0)  # (C, 3, 3)

        # 2. Load CoTracker data
        print("[Init] Loading CoTracker tracks...")
        self._load_cotracker_data()

        # 3. Load canonical Gaussians from PLY
        print(f"[Init] Loading PLY from {cfg.ply_path}...")
        self._init_splats()

        # 4. Build KNN for spatial rigidity
        print("[Init] Building KNN graph...")
        with torch.no_grad():
            self.knn_indices = compute_knn(
                self.splats["means"].detach(), cfg.knn_k
            )  # (N, k)
        print(f"[Init] KNN done: {self.knn_indices.shape}")

        # 5. Build track-to-Gaussian bindings
        print("[Init] Building track-to-Gaussian bindings...")
        self._build_bindings()

        # 6. Setup optimizers
        self._setup_optimizers()

        print(f"[Init] Ready. {self.splats['means'].shape[0]} Gaussians, "
              f"{self.num_cameras} cameras, {cfg.num_cotracker_frames} frames")

    # -----------------------------------------------------------------------
    # Data loading
    # -----------------------------------------------------------------------
    def _load_cotracker_data(self):
        """Load all CoTracker .npz files into GPU tensors."""
        cfg = self.cfg
        npz_files = sorted(Path(cfg.cotracker_dir).glob("*.npz"))
        assert len(npz_files) > 0, f"No .npz files in {cfg.cotracker_dir}"

        # Build mapping from camera name to npz file
        npz_by_cam = {}
        for f in npz_files:
            cam_name = f.stem  # e.g. "take_18_cam_01"
            npz_by_cam[cam_name] = f

        all_tracks = []
        all_vis = []
        self.cotracker_cam_indices = []  # which cameras have cotracker data

        for cam_idx, cam_name in enumerate(self.cam_names):
            if cam_name in npz_by_cam:
                data = np.load(npz_by_cam[cam_name])
                tracks = data["tracks"].squeeze(0)      # (50, 625, 2)
                vis = data["visibility"].squeeze(0)       # (50, 625)
                scale = float(data["downsample_factor"])
                # Scale to render resolution (full-res / data_factor)
                tracks = tracks * scale / cfg.data_factor
                all_tracks.append(torch.from_numpy(tracks).float())
                all_vis.append(torch.from_numpy(vis).bool())
                self.cotracker_cam_indices.append(cam_idx)

        self.gt_tracks = torch.stack(all_tracks, dim=0).to(self.device)  # (C', 50, 625, 2)
        self.gt_vis = torch.stack(all_vis, dim=0).to(self.device)        # (C', 50, 625)

        # Mapping from cotracker index to camera index
        self.ct_to_cam = self.cotracker_cam_indices  # list of cam_idx
        # Reverse: cam_idx -> cotracker index (or -1)
        self.cam_to_ct = [-1] * self.num_cameras
        for ct_idx, cam_idx in enumerate(self.ct_to_cam):
            self.cam_to_ct[cam_idx] = ct_idx

        print(f"[CoTracker] Loaded {self.gt_tracks.shape[0]} cameras, "
              f"tracks shape: {self.gt_tracks.shape}, "
              f"downsample_factor: {scale:.1f}")

    def _init_splats(self):
        """Initialize dynamic Gaussians (trainable) and optional static ones (frozen)."""
        cfg = self.cfg
        device = self.device

        # ---- Dynamic / foreground (trainable, with motion offsets) ----
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

        print(f"[Splats/dynamic] {N} Gaussians, motion_offsets: "
              f"{motion_offsets.shape}, VRAM: {motion_offsets.numel() * 4 / 1e6:.1f}MB")

        # ---- Static / background (frozen, no motion, never updated) ----
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
        """For each camera, bind CoTracker grid points to nearest DYNAMIC Gaussians.

        Tracks whose frame-0 nearest gaussian (across static+dynamic) lies in the
        static set are flagged invalid via binding_valid[ct_idx] and contribute
        nothing to the tracking loss — this excludes background motion.
        """
        dyn_means = self.splats["means"].detach()  # (N_dyn, 3)
        stat_means = self.static["means"] if self.static is not None else None

        self.binding_map = {}    # ct_idx -> (625,) LongTensor of DYNAMIC indices
        self.binding_valid = {}  # ct_idx -> (625,) BoolTensor

        total_tracks = 0
        total_valid = 0
        for ct_idx, cam_idx in enumerate(self.ct_to_cam):
            viewmat = self.viewmats[cam_idx]
            K = self.Ks[cam_idx]

            proj_dyn, _ = project_points(dyn_means, viewmat, K)  # (N_dyn, 2)
            gt_t0 = self.gt_tracks[ct_idx, 0]                    # (625, 2)

            dist_dyn = torch.cdist(gt_t0, proj_dyn)
            min_dyn, idx_dyn = dist_dyn.min(dim=-1)  # (625,), (625,)

            if stat_means is not None:
                proj_stat, _ = project_points(stat_means, viewmat, K)
                dist_stat = torch.cdist(gt_t0, proj_stat)
                min_stat, _ = dist_stat.min(dim=-1)
                valid = min_dyn < min_stat
            else:
                valid = torch.ones_like(min_dyn, dtype=torch.bool)

            self.binding_map[ct_idx] = idx_dyn
            self.binding_valid[ct_idx] = valid
            total_tracks += valid.numel()
            total_valid += int(valid.sum().item())

        if self.static is not None:
            pct = 100.0 * total_valid / max(total_tracks, 1)
            print(f"[Binding] {len(self.binding_map)} cameras, "
                  f"{total_valid}/{total_tracks} tracks kept ({pct:.1f}%), "
                  f"rest dropped as background")
        else:
            print(f"[Binding] Built bindings for {len(self.binding_map)} cameras")

    def _setup_optimizers(self):
        """Create optimizers: Adam for motion, separate for appearance params."""
        cfg = self.cfg

        # Motion optimizer (always active)
        self.motion_optimizer = torch.optim.Adam(
            [{"params": self.splats["motion_offsets"], "lr": cfg.motion_lr}],
            eps=1e-15,
        )

        # Appearance + geometry optimizers (frozen in phase 1)
        self.appearance_params = []
        param_lr_pairs = [
            ("means", cfg.means_lr),
            ("scales", cfg.scales_lr),
            ("quats", cfg.quats_lr),
            ("opacities", cfg.opacities_lr),
            ("sh0", cfg.sh0_lr),
            ("shN", cfg.shN_lr),
        ]
        self.appearance_optimizers = {}
        for name, lr in param_lr_pairs:
            opt = torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr}],
                eps=1e-15,
            )
            self.appearance_optimizers[name] = opt

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------
    def load_image(self, cam_idx: int, frame_idx: int) -> Tensor:
        """Load a GT image for a given camera and CoTracker frame index.

        Args:
            cam_idx: camera index (0..44)
            frame_idx: CoTracker frame index (0..49)

        Returns:
            image: (H, W, 3) float tensor on GPU, values in [0, 1]
        """
        # CoTracker frame_idx -> 1-indexed image number
        image_num = frame_idx * self.cfg.frame_step + 1
        cam_name = self.cam_names[cam_idx]
        img_path = os.path.join(
            self.cfg.data_dir, "images", cam_name, f"{image_num:06d}.jpg"
        )
        img = imageio.imread(img_path)
        img = torch.from_numpy(img).float().to(self.device) / 255.0

        # Apply data_factor downsampling if needed
        if self.cfg.data_factor > 1:
            f = self.cfg.data_factor
            img = img[::f, ::f]

        return img

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    def rasterize_splats(
        self,
        cam_idx: int,
        frame_idx: int,
    ) -> Tuple[Tensor, Tensor, dict]:
        """Render composite (static frozen + dynamic with motion) from a camera at time t."""
        # Dynamic (trainable)
        dyn_means = self.splats["means"] + self.splats["motion_offsets"][:, frame_idx, :]
        dyn_quats = self.splats["quats"]
        dyn_log_scales = self.splats["scales"]
        dyn_raw_opac = self.splats["opacities"]
        dyn_sh0 = self.splats["sh0"]
        dyn_shN = self.splats["shN"]

        if self.static is not None:
            means = torch.cat([self.static["means"], dyn_means], dim=0)
            quats = torch.cat([self.static["quats"], dyn_quats], dim=0)
            scales = torch.exp(torch.cat([self.static["scales"], dyn_log_scales], dim=0))
            opacities = torch.sigmoid(
                torch.cat([self.static["opacities"], dyn_raw_opac], dim=0)
            )
            sh0 = torch.cat([self.static["sh0"], dyn_sh0], dim=0)
            shN = torch.cat([self.static["shN"], dyn_shN], dim=0)
            colors = torch.cat([sh0, shN], dim=1)
        else:
            means = dyn_means
            quats = dyn_quats
            scales = torch.exp(dyn_log_scales)
            opacities = torch.sigmoid(dyn_raw_opac)
            colors = torch.cat([dyn_sh0, dyn_shN], dim=1)

        camtoworld = self.camtoworlds[cam_idx:cam_idx+1]  # (1, 4, 4)
        K = self.Ks_tensor[cam_idx:cam_idx+1]              # (1, 3, 3)
        width, height = self.img_sizes[cam_idx]

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=K,
            width=width,
            height=height,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            sh_degree=self.cfg.sh_degree,
            packed=self.cfg.packed,
            render_mode="RGB",
        )
        return render_colors, render_alphas, info

    # -----------------------------------------------------------------------
    # Loss functions
    # -----------------------------------------------------------------------
    def compute_tracking_loss(
        self, ct_idx: int, frame_idx: int, means_dynamic: Tensor
    ) -> Tensor:
        """Tracking loss: MSE between projected bound Gaussians and CoTracker 2D.

        Args:
            ct_idx: CoTracker camera index
            frame_idx: time step (0..49)
            means_dynamic: (N, 3) current 3D positions
        """
        cam_idx = self.ct_to_cam[ct_idx]
        viewmat = self.viewmats[cam_idx]
        K = self.Ks[cam_idx]

        # Project dynamic Gaussians to 2D
        proj_2d, _ = project_points(means_dynamic, viewmat, K)  # (N, 2)

        # Get bound Gaussian projections
        bound_idx = self.binding_map[ct_idx]  # (625,)
        pred_2d = proj_2d[bound_idx]          # (625, 2)

        # GT CoTracker points
        gt_2d = self.gt_tracks[ct_idx, frame_idx]  # (625, 2)
        vis = self.gt_vis[ct_idx, frame_idx]        # (625,)
        # Drop tracks whose binding is on the static background
        vis = vis & self.binding_valid[ct_idx]

        if vis.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        return F.mse_loss(pred_2d[vis], gt_2d[vis])

    def compute_temporal_loss(self) -> Tensor:
        """Temporal smoothness: penalize large frame-to-frame jumps."""
        motion = self.splats["motion_offsets"]  # (N, T, 3)
        diff = motion[:, 1:, :] - motion[:, :-1, :]  # (N, T-1, 3)
        return diff.pow(2).mean()

    def compute_rigidity_loss(self, frame_idx: int) -> Tensor:
        """Spatial rigidity: nearby Gaussians should have similar motion at frame t."""
        motion_t = self.splats["motion_offsets"][:, frame_idx, :]  # (N, 3)
        neighbor_motion = motion_t[self.knn_indices]                # (N, k, 3)
        diff = motion_t.unsqueeze(1) - neighbor_motion              # (N, k, 3)
        return diff.pow(2).mean()

    def compute_photometric_loss(
        self, rendered: Tensor, gt_image: Tensor
    ) -> Tuple[Tensor, float, float, float]:
        """L1 + SSIM photometric loss. Returns (loss, l1, ssim_metric, psnr)."""
        pixels = gt_image.unsqueeze(0)  # (1, H, W, 3)
        # Crop to match sizes (cameras may differ slightly after integer division)
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

        # Save config (json + yaml)
        cfg_dict = {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in vars(cfg).items()}
        with open(os.path.join(cfg.result_dir, "cfg.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2, default=str)
        with open(os.path.join(cfg.result_dir, "cfg.yaml"), "w") as f:
            yaml.safe_dump(cfg_dict, f, sort_keys=False)

        num_ct_cams = len(self.ct_to_cam)
        num_frames = cfg.num_cotracker_frames

        # LR scheduler for motion
        motion_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.motion_optimizer, gamma=0.1 ** (1.0 / cfg.max_steps)
        )

        global_tic = time.time()
        pbar = tqdm.tqdm(range(cfg.max_steps))

        for step in pbar:
            # Sample random frame
            frame_idx = random.randint(0, num_frames - 1)

            # Dynamic means for this frame
            means_dynamic = self.splats["means"] + self.splats["motion_offsets"][:, frame_idx, :]

            # --- Tracking loss: ALL cameras for this frame (multi-view consistency) ---
            loss_track = torch.tensor(0.0, device=device)
            for ct_idx in range(num_ct_cams):
                loss_track = loss_track + self.compute_tracking_loss(
                    ct_idx, frame_idx, means_dynamic
                )
            loss_track = loss_track / num_ct_cams  # average over cameras

            # --- Regularization ---
            loss_temp = self.compute_temporal_loss()
            loss_rigid = self.compute_rigidity_loss(frame_idx)

            # --- Photometric loss (phase 2 only, 1 random camera) ---
            loss_photo = torch.tensor(0.0, device=device)
            l1_val = 0.0
            ssim_val = 0.0
            psnr_val = 0.0
            if step >= cfg.freeze_appearance_steps:
                ct_idx_photo = random.randint(0, num_ct_cams - 1)
                cam_idx_photo = self.ct_to_cam[ct_idx_photo]
                gt_image = self.load_image(cam_idx_photo, frame_idx)
                rendered, _, _ = self.rasterize_splats(cam_idx_photo, frame_idx)
                loss_photo, l1_val, ssim_val, psnr_val = self.compute_photometric_loss(
                    rendered, gt_image
                )

            # --- Total loss ---
            loss = (
                cfg.track_loss_weight * loss_track
                + cfg.temporal_smooth_weight * loss_temp
                + cfg.spatial_rigid_weight * loss_rigid
            )
            if step >= cfg.freeze_appearance_steps:
                loss = loss + cfg.photo_loss_weight * loss_photo

            # --- Backward ---
            loss.backward()

            # --- Optimizer step ---
            self.motion_optimizer.step()
            self.motion_optimizer.zero_grad(set_to_none=True)

            if step >= cfg.freeze_appearance_steps:
                for opt in self.appearance_optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
            else:
                # Zero grads for frozen params
                for opt in self.appearance_optimizers.values():
                    opt.zero_grad(set_to_none=True)

            motion_scheduler.step()

            # --- Logging ---
            desc = (
                f"loss={loss.item():.4f} "
                f"trk={loss_track.item():.4f} "
                f"tmp={loss_temp.item():.6f} "
                f"rig={loss_rigid.item():.6f}"
            )
            if step >= cfg.freeze_appearance_steps:
                desc += f" pho={loss_photo.item():.4f} psnr={psnr_val:.2f}"
            pbar.set_description(desc)

            # --- TensorBoard scalars (every step) ---
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

            # --- TensorBoard images (GT vs rendered) ---
            if step % cfg.tb_image_every == 0:
                self._log_tb_images(step)

            # --- Checkpointing ---
            if step + 1 in cfg.save_steps or step == cfg.max_steps - 1:
                ckpt_path = os.path.join(self.ckpt_dir, f"ckpt_{step}.pt")
                torch.save({
                    "step": step,
                    "splats": self.splats.state_dict(),
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved to {ckpt_path}")

            # --- Evaluation ---
            if step + 1 in cfg.eval_steps:
                self.eval_tracking(step)

            # --- Per-frame PLY export ---
            if step + 1 in cfg.ply_save_steps:
                self.export_per_frame_ply(step)

        elapsed = time.time() - global_tic
        print(f"\n[Done] Training finished in {elapsed:.1f}s")

        # Render videos
        for cam_render_idx in cfg.render_video_cams:
            if cam_render_idx < self.num_cameras:
                self.render_video(cam_render_idx)

        self.writer.flush()
        self.writer.close()

    # -----------------------------------------------------------------------
    # TensorBoard image logging
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def _log_tb_images(self, step: int):
        """Log a (GT | rendered) side-by-side image to TensorBoard."""
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

        concat = torch.cat([gt[0], rendered[0]], dim=1)  # (H, 2W, 3)
        self.writer.add_image(
            "images/gt_vs_rendered", concat.permute(2, 0, 1), step
        )

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
    # Evaluation
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def eval_tracking(self, step: int):
        """Evaluate tracking error AND PSNR/SSIM/L1 over all cameras × frames."""
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
                # Tracking error
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

                # Photometric metrics
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

        print(f"\n[Eval @ step {step}] trk={mean_err:.2f}px  "
              f"PSNR={psnr:.3f}  SSIM={mean_ssim:.4f}  L1={mean_l1:.4f}  "
              f"({total_pts} pts, {n_views} views)")

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
        """Export a PLY file for each of the 50 CoTracker frames.

        Each PLY contains the Gaussians at their dynamic positions for that frame.
        Saved to: <result_dir>/ply/step_<step>/frame_<t>.ply
        """
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
                means=means_t,
                scales=scales,
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                format="ply",
                save_to=save_path,
            )

        print(f"\n[PLY] Exported {cfg.num_cotracker_frames} frames to {ply_dir}/")

    @torch.no_grad()
    def render_video(self, cam_idx: int):
        """Render all CoTracker frames from a camera and save as video."""
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

    @torch.no_grad()
    def render_tracking_overlay(self, cam_idx: int, frame_idx: int) -> np.ndarray:
        """Render a frame with CoTracker GT (green) and predicted (red) points overlaid."""
        rendered, _, _ = self.rasterize_splats(cam_idx, frame_idx)
        img = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()

        ct_idx = self.cam_to_ct[cam_idx]
        if ct_idx < 0:
            return img

        means_t = self.splats["means"] + self.splats["motion_offsets"][:, frame_idx, :]
        proj_2d, _ = project_points(means_t, self.viewmats[cam_idx], self.Ks[cam_idx])
        bound_idx = self.binding_map[ct_idx]
        pred_2d = proj_2d[bound_idx].cpu().numpy()
        gt_2d = self.gt_tracks[ct_idx, frame_idx].cpu().numpy()
        vis = (self.gt_vis[ct_idx, frame_idx] & self.binding_valid[ct_idx]).cpu().numpy()

        H, W = img.shape[:2]
        r = max(3, min(H, W) // 200)

        for i in range(len(gt_2d)):
            if not vis[i]:
                continue
            # GT point (green)
            gx, gy = int(gt_2d[i, 0]), int(gt_2d[i, 1])
            if 0 <= gx < W and 0 <= gy < H:
                img[max(0,gy-r):gy+r, max(0,gx-r):gx+r] = [0, 255, 0]
            # Predicted point (red)
            px, py = int(pred_2d[i, 0]), int(pred_2d[i, 1])
            if 0 <= px < W and 0 <= py < H:
                img[max(0,py-r):py+r, max(0,px-r):px+r] = [255, 0, 0]

        return img

    @torch.no_grad()
    def render_trajectory_video(self, cam_idx: int, tail_length: int = 10):
        """Render video with trajectory tails for each anchor point.

        Each frame shows:
        - The rendered Gaussians
        - CoTracker GT trajectory tails (green lines + dots)
        - Predicted anchor trajectory tails (red lines + dots)
        - Lines connecting GT to predicted at current frame (yellow, shows error)

        Args:
            cam_idx: camera index to render from
            tail_length: how many past frames to draw as trajectory tail
        """
        import cv2

        cfg = self.cfg
        ct_idx = self.cam_to_ct[cam_idx]
        if ct_idx < 0:
            print(f"[Traj] Camera {cam_idx} has no CoTracker data, skipping")
            return

        cam_name = self.cam_names[cam_idx]
        viewmat = self.viewmats[cam_idx]
        K = self.Ks[cam_idx]
        bound_idx = self.binding_map[ct_idx]
        num_frames = cfg.num_cotracker_frames

        # Pre-compute all predicted 2D trajectories: (50, 625, 2)
        pred_trajs = []
        for t in range(num_frames):
            means_t = self.splats["means"] + self.splats["motion_offsets"][:, t, :]
            proj_2d, _ = project_points(means_t, viewmat, K)
            pred_trajs.append(proj_2d[bound_idx].cpu().numpy())
        pred_trajs = np.stack(pred_trajs, axis=0)  # (50, 625, 2)

        gt_trajs = self.gt_tracks[ct_idx].cpu().numpy()   # (50, 625, 2)
        valid = self.binding_valid[ct_idx].cpu().numpy()   # (625,)
        vis_all = (self.gt_vis[ct_idx] & self.binding_valid[ct_idx][None, :]).cpu().numpy()

        # Generate a unique color per anchor point (HSV rainbow)
        num_pts = gt_trajs.shape[1]
        colors_hsv = np.zeros((num_pts, 1, 3), dtype=np.uint8)
        colors_hsv[:, 0, 0] = np.linspace(0, 170, num_pts, dtype=np.uint8)  # hue
        colors_hsv[:, 0, 1] = 255  # saturation
        colors_hsv[:, 0, 2] = 255  # value
        colors_bgr = cv2.cvtColor(colors_hsv, cv2.COLOR_HSV2BGR)
        anchor_colors = colors_bgr.squeeze(1).astype(int)  # (625, 3) BGR

        frames = []
        print(f"[Traj] Rendering trajectory video for {cam_name}...")

        for t in tqdm.tqdm(range(num_frames), desc=f"Traj {cam_name}"):
            # Render base frame
            rendered, _, _ = self.rasterize_splats(cam_idx, t)
            img = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            H, W = img.shape[:2]

            # Draw trajectory tails
            t_start = max(0, t - tail_length)
            for i in range(num_pts):
                if not vis_all[t, i]:
                    continue
                color = tuple(int(c) for c in anchor_colors[i])

                # GT trajectory tail (brighter, thicker)
                for t2 in range(t_start, t):
                    if not vis_all[t2, i] or not vis_all[t2 + 1, i]:
                        continue
                    p1 = (int(gt_trajs[t2, i, 0]), int(gt_trajs[t2, i, 1]))
                    p2 = (int(gt_trajs[t2 + 1, i, 0]), int(gt_trajs[t2 + 1, i, 1]))
                    if all(0 <= p[0] < W and 0 <= p[1] < H for p in [p1, p2]):
                        alpha = (t2 - t_start + 1) / (t - t_start + 1)
                        thickness = max(1, int(2 * alpha))
                        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)

                # Predicted trajectory tail (dashed, thinner)
                for t2 in range(t_start, t):
                    p1 = (int(pred_trajs[t2, i, 0]), int(pred_trajs[t2, i, 1]))
                    p2 = (int(pred_trajs[t2 + 1, i, 0]), int(pred_trajs[t2 + 1, i, 1]))
                    if all(0 <= p[0] < W and 0 <= p[1] < H for p in [p1, p2]):
                        cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

                # Current frame: GT dot (filled circle) + predicted dot (circle outline)
                gt_pt = (int(gt_trajs[t, i, 0]), int(gt_trajs[t, i, 1]))
                pr_pt = (int(pred_trajs[t, i, 0]), int(pred_trajs[t, i, 1]))
                r = max(4, min(H, W) // 300)

                if 0 <= gt_pt[0] < W and 0 <= gt_pt[1] < H:
                    cv2.circle(img, gt_pt, r, color, -1, cv2.LINE_AA)  # filled
                if 0 <= pr_pt[0] < W and 0 <= pr_pt[1] < H:
                    cv2.circle(img, pr_pt, r, (255, 255, 255), 2, cv2.LINE_AA)  # outline

                # Error line: GT → predicted (yellow)
                if (0 <= gt_pt[0] < W and 0 <= gt_pt[1] < H and
                    0 <= pr_pt[0] < W and 0 <= pr_pt[1] < H):
                    cv2.line(img, gt_pt, pr_pt, (0, 255, 255), 1, cv2.LINE_AA)

            # Add frame info text
            cv2.putText(img, f"Frame {t}/{num_frames-1}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img_rgb)

        video_path = os.path.join(self.video_dir, f"{cam_name}_trajectories.mp4")
        imageio.mimwrite(video_path, frames, fps=10, quality=8)
        print(f"[Traj] Saved to {video_path}")

    @torch.no_grad()
    def render_trajectory_image(self, cam_idx: int, save_path: Optional[str] = None):
        """Render a single image showing ALL trajectories across all 50 frames.

        Draws full trajectory paths on top of the frame-0 render:
        - Colored lines: GT CoTracker trajectories (one color per anchor)
        - White dots: predicted positions at each frame
        """
        import cv2

        cfg = self.cfg
        ct_idx = self.cam_to_ct[cam_idx]
        if ct_idx < 0:
            return

        cam_name = self.cam_names[cam_idx]
        viewmat = self.viewmats[cam_idx]
        K = self.Ks[cam_idx]
        bound_idx = self.binding_map[ct_idx]
        num_frames = cfg.num_cotracker_frames

        # Render frame 0 as background
        rendered, _, _ = self.rasterize_splats(cam_idx, 0)
        img = (rendered[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        # Darken background so trajectories pop
        img = (img * 0.4).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]

        # Pre-compute trajectories
        pred_trajs = []
        for t in range(num_frames):
            means_t = self.splats["means"] + self.splats["motion_offsets"][:, t, :]
            proj_2d, _ = project_points(means_t, viewmat, K)
            pred_trajs.append(proj_2d[bound_idx].cpu().numpy())
        pred_trajs = np.stack(pred_trajs, axis=0)  # (50, 625, 2)

        gt_trajs = self.gt_tracks[ct_idx].cpu().numpy()  # (50, 625, 2)
        vis_all = (self.gt_vis[ct_idx] & self.binding_valid[ct_idx][None, :]).cpu().numpy()

        # Colors per anchor
        num_pts = gt_trajs.shape[1]
        colors_hsv = np.zeros((num_pts, 1, 3), dtype=np.uint8)
        colors_hsv[:, 0, 0] = np.linspace(0, 170, num_pts, dtype=np.uint8)
        colors_hsv[:, 0, 1] = 255
        colors_hsv[:, 0, 2] = 255
        colors_bgr = cv2.cvtColor(colors_hsv, cv2.COLOR_HSV2BGR)
        anchor_colors = colors_bgr.squeeze(1).astype(int)

        # Draw each anchor's full trajectory
        for i in range(num_pts):
            color = tuple(int(c) for c in anchor_colors[i])

            # GT trajectory (colored line)
            for t in range(num_frames - 1):
                if not vis_all[t, i] or not vis_all[t + 1, i]:
                    continue
                p1 = (int(gt_trajs[t, i, 0]), int(gt_trajs[t, i, 1]))
                p2 = (int(gt_trajs[t + 1, i, 0]), int(gt_trajs[t + 1, i, 1]))
                if all(0 <= p[0] < W and 0 <= p[1] < H for p in [p1, p2]):
                    cv2.line(img, p1, p2, color, 2, cv2.LINE_AA)

            # Predicted trajectory (white line)
            for t in range(num_frames - 1):
                p1 = (int(pred_trajs[t, i, 0]), int(pred_trajs[t, i, 1]))
                p2 = (int(pred_trajs[t + 1, i, 0]), int(pred_trajs[t + 1, i, 1]))
                if all(0 <= p[0] < W and 0 <= p[1] < H for p in [p1, p2]):
                    cv2.line(img, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

            # Start dot (green) and end dot (red) for GT
            if vis_all[0, i]:
                p = (int(gt_trajs[0, i, 0]), int(gt_trajs[0, i, 1]))
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.circle(img, p, 5, (0, 255, 0), -1, cv2.LINE_AA)
            last_vis = np.where(vis_all[:, i])[0]
            if len(last_vis) > 0:
                tl = last_vis[-1]
                p = (int(gt_trajs[tl, i, 0]), int(gt_trajs[tl, i, 1]))
                if 0 <= p[0] < W and 0 <= p[1] < H:
                    cv2.circle(img, p, 5, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(img, f"{cam_name} - Full Trajectories (colored=GT, white=pred)",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if save_path is None:
            save_path = os.path.join(self.render_dir, f"{cam_name}_all_trajectories.png")
        imageio.imwrite(save_path, img_rgb)
        print(f"[Traj] Saved trajectory image to {save_path}")

    @torch.no_grad()
    def visualize_3d_trajectories(self, save_path: Optional[str] = None,
                                   subsample: int = 1,
                                   density_keep_pct: float = 0.5,
                                   max_anchors: int = 300,
                                   scene_max_points: int = 20000,
                                   marker_size: int = 6,
                                   scene_marker_size: int = 2,
                                   clip_to_camera_rig: bool = True,
                                   rig_margin: float = 0.05,
                                   trajectory_color: str = "red",
                                   trajectory_width: float = 1.5):
        """Interactive Plotly HTML: canonical scene at t=0 + tracked anchors in motion.

        Layout:
          - Background "scene" trace: all dynamic Gaussians at t=0 as a tiny
            colored point cloud (uses sh0→RGB), shows the object.
          - Tracked anchor trace: a sparse, uniformly-sampled subset of the
            dense region — larger dots, same RGB coloring — animated over time.
          - Faint trajectory lines for the tracked anchors.
          - Cameras.

        Args:
            save_path: output HTML
            subsample: every Nth anchor before density/voxel filtering
            density_keep_pct: keep this quantile of anchors by kNN density
            max_anchors: cap for the animated tracked set (voxel-sampled)
            scene_max_points: cap for the static canonical scene point cloud
            marker_size: animated anchor dot size
            scene_marker_size: scene dot size
        """
        import plotly.graph_objects as go

        cfg = self.cfg
        num_frames = cfg.num_cotracker_frames

        # ---- 0. Camera-rig AABB (clip everything outside)
        cam_pos = self.camtoworlds[:, :3, 3].cpu().numpy()  # (C, 3)
        cam_min = cam_pos.min(0)
        cam_max = cam_pos.max(0)
        margin = (cam_max - cam_min) * rig_margin
        bmin = cam_min - margin
        bmax = cam_max + margin

        def _inside_rig(xyz: np.ndarray) -> np.ndarray:
            return np.all((xyz >= bmin) & (xyz <= bmax), axis=-1)

        # ---- 1. Canonical "scene" point cloud (all dynamic gaussians, t=0),
        #         filtered to camera rig.
        all_xyz = self.splats["means"].detach().cpu().numpy()
        if clip_to_camera_rig:
            rig_mask = _inside_rig(all_xyz)
            inside_ids = np.where(rig_mask)[0]
        else:
            inside_ids = np.arange(all_xyz.shape[0])
        if len(inside_ids) > scene_max_points:
            rng = np.random.default_rng(0)
            scene_idx = rng.choice(inside_ids, size=scene_max_points, replace=False)
        else:
            scene_idx = inside_ids
        scene_idx_t = torch.tensor(scene_idx, device=self.device)
        scene_xyz = self.splats["means"][scene_idx_t].detach().cpu().numpy()
        scene_rgb = sh2rgb(
            self.splats["sh0"][scene_idx_t, 0].detach()
        ).clamp(0, 1).cpu().numpy()
        scene_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                        for r, g, b in scene_rgb]

        # ---- 2. Candidate anchor IDs (track points bound to dynamic Gaussians)
        all_gauss_ids = set()
        for ct_idx in range(len(self.ct_to_cam)):
            ids = self.binding_map[ct_idx].cpu().numpy()
            if ct_idx in getattr(self, "binding_valid", {}):
                valid = self.binding_valid[ct_idx].cpu().numpy()
                ids = ids[valid]
            all_gauss_ids.update(ids[::subsample].tolist())
        gauss_ids = np.array(sorted(all_gauss_ids), dtype=np.int64)
        if len(gauss_ids) == 0:
            raise RuntimeError("No valid anchor Gaussians found")

        canonical = self.splats["means"][gauss_ids].detach()  # (N, 3) GPU
        rgb = sh2rgb(self.splats["sh0"][gauss_ids, 0].detach()).clamp(0, 1).cpu().numpy()

        # ---- 2b. Drop anchors outside the camera rig
        if clip_to_camera_rig:
            rig_mask_a = _inside_rig(canonical.cpu().numpy())
            if rig_mask_a.sum() == 0:
                raise RuntimeError("All anchors fall outside the camera rig AABB")
            gauss_ids = gauss_ids[rig_mask_a]
            canonical = canonical[torch.from_numpy(rig_mask_a).to(canonical.device)]
            rgb = rgb[rig_mask_a]

        # ---- 3. Density filter: drop scattered outliers
        if density_keep_pct < 1.0 and canonical.shape[0] > 12:
            K = 10
            chunk = 4096
            all_means = []
            for i in range(0, canonical.shape[0], chunk):
                d = torch.cdist(canonical[i:i+chunk], canonical)
                topk, _ = d.topk(K + 1, dim=-1, largest=False)
                all_means.append(topk[:, 1:].mean(dim=-1))
            mean_nbr = torch.cat(all_means).cpu().numpy()
            thresh = float(np.quantile(mean_nbr, density_keep_pct))
            keep = mean_nbr <= thresh
            gauss_ids = gauss_ids[keep]
            canonical = canonical[torch.from_numpy(keep).to(canonical.device)]
            rgb = rgb[keep]

        canonical_np = canonical.cpu().numpy()

        # ---- 4. Uniform voxel-sample to max_anchors
        if canonical_np.shape[0] > max_anchors:
            bbox = canonical_np.max(0) - canonical_np.min(0)
            vox = (float(np.prod(bbox)) / max(max_anchors, 1)) ** (1.0 / 3.0)
            idx = np.floor((canonical_np - canonical_np.min(0)) / max(vox, 1e-9)).astype(np.int64)
            keys = idx[:, 0] * 1_000_003 * 1_000_003 + idx[:, 1] * 1_000_003 + idx[:, 2]
            _, first = np.unique(keys, return_index=True)
            if len(first) > max_anchors:
                rng = np.random.default_rng(0)
                first = rng.choice(first, size=max_anchors, replace=False)
            gauss_ids = gauss_ids[first]
            rgb = rgb[first]

        gauss_ids_t = torch.tensor(gauss_ids, device=self.device)

        # ---- 5. Tracked anchor positions over time
        positions = []
        for t in range(num_frames):
            means_t = self.splats["means"] + self.splats["motion_offsets"][:, t, :]
            positions.append(means_t[gauss_ids_t].cpu().numpy())
        positions = np.stack(positions, axis=0)  # (T, N, 3)

        num_anchors = positions.shape[1]
        print(f"[Plotly] scene={len(scene_idx)} pts, tracked={num_anchors} anchors "
              f"(density_keep_pct={density_keep_pct}, max_anchors={max_anchors})")

        marker_colors = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                         for r, g, b in rgb]

        fig_data = []

        # (0) Canonical scene cloud — static context, all at t=0
        fig_data.append(go.Scatter3d(
            x=scene_xyz[:, 0], y=scene_xyz[:, 1], z=scene_xyz[:, 2],
            mode="markers",
            marker=dict(size=scene_marker_size, color=scene_colors, opacity=0.55),
            name="scene (t=0)",
            hoverinfo="skip",
        ))

        # (1) Faint trajectory lines for tracked anchors
        xs, ys, zs = [], [], []
        for i in range(num_anchors):
            xs.extend(positions[:, i, 0].tolist() + [None])
            ys.extend(positions[:, i, 1].tolist() + [None])
            zs.extend(positions[:, i, 2].tolist() + [None])
        fig_data.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines",
            line=dict(width=trajectory_width, color=trajectory_color),
            showlegend=False, hoverinfo="skip", name="trajectories",
        ))

        # (2) Tracked anchors — animated
        fig_data.append(go.Scatter3d(
            x=positions[0, :, 0], y=positions[0, :, 1], z=positions[0, :, 2],
            mode="markers",
            marker=dict(size=marker_size, color=marker_colors,
                        line=dict(width=1, color="black")),
            name="tracked anchors",
            text=[f"id={gauss_ids[i]} t=0" for i in range(num_anchors)],
            hoverinfo="text",
        ))

        # (3) Cameras (cam_pos already computed above for the rig AABB)
        fig_data.append(go.Scatter3d(
            x=cam_pos[:, 0], y=cam_pos[:, 1], z=cam_pos[:, 2],
            mode="markers+text",
            marker=dict(size=5, color="cyan", symbol="diamond"),
            text=[self.cam_names[i].replace("take_18_", "") for i in range(len(cam_pos))],
            textposition="top center",
            textfont=dict(size=8),
            name="cameras",
        ))

        # Animation frames only redraw trace 2 (tracked anchors)
        frames = []
        for t in range(num_frames):
            frames.append(go.Frame(
                data=[go.Scatter3d(
                    x=positions[t, :, 0], y=positions[t, :, 1], z=positions[t, :, 2],
                    mode="markers",
                    marker=dict(size=marker_size, color=marker_colors,
                                line=dict(width=1, color="black")),
                    text=[f"id={gauss_ids[i]} t={t}" for i in range(num_anchors)],
                    hoverinfo="text",
                )],
                traces=[2],
                name=str(t),
            ))

        fig = go.Figure(data=fig_data, frames=frames)

        scene_layout = dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        )
        if clip_to_camera_rig:
            scene_layout["xaxis"] = dict(range=[float(bmin[0]), float(bmax[0])])
            scene_layout["yaxis"] = dict(range=[float(bmin[1]), float(bmax[1])])
            scene_layout["zaxis"] = dict(range=[float(bmin[2]), float(bmax[2])])

        fig.update_layout(
            title=f"TrackerSplat 3D Trajectories - {num_anchors} anchors x {num_frames} frames",
            scene=scene_layout,
            updatemenus=[dict(
                type="buttons", showactive=False, y=0, x=0.5, xanchor="center",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=200, redraw=True),
                                          fromcurrent=True)]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0,
                steps=[dict(
                    args=[[str(t)], dict(frame=dict(duration=0, redraw=True),
                                         mode="immediate")],
                    label=str(t), method="animate",
                ) for t in range(num_frames)],
                x=0.05, len=0.9, xanchor="left", y=-0.05,
                currentvalue=dict(prefix="Frame: ", visible=True),
            )],
            width=1400, height=900,
        )

        if save_path is None:
            save_path = os.path.join(cfg.result_dir, "trajectories_3d.html")
        fig.write_html(save_path)
        print(f"[Plotly] Saved interactive 3D trajectories to {save_path}")

    def load_checkpoint(self, ckpt_path: str):
        """Load a checkpoint to resume or visualize."""
        data = torch.load(ckpt_path, map_location=self.device)
        self.splats.load_state_dict(data["splats"])
        print(f"[Checkpoint] Loaded from {ckpt_path} (step {data['step']})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = tyro.cli(Config)
    runner = TrackerSplatRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
