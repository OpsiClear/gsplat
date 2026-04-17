"""
TrackerSplat pipeline orchestrator — per-frame PWI-LS → ISVD → Motion.

Mirrors the reference's `motionestimation.py` + `motionestimator/point_tracker/fuser.py`
but uses our Apache-2.0 motion_fusion (pure PyTorch) instead of the Inria fork.

Day-4 scope: TRANSLATION-ONLY motion compensation.
  * per-Gaussian 2D affine [A|b] solved via PWI-LS (motion_fusion + linear solve)
  * 3D mean triangulation via ISVD_Mean3D across all provided views
  * Returns Motion with translation_vector only; rotation/scale stay identity.

Rotation/scale updates (ILS_RotationScale) + propagation + refinement are
addressed in follow-up days once translation works end-to-end.

Apache-2.0.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor

try:
    from .motion import Motion
    from .motion_fusion_taichi import motion_fusion, MotionFusionOutput
    from .utils import ISVD_Mean3D
    from .utils.math_utils import compute_mean2D_equations
except ImportError:
    from trackersplat_paper.motion import Motion
    from trackersplat_paper.motion_fusion_taichi import motion_fusion, MotionFusionOutput
    from trackersplat_paper.utils import ISVD_Mean3D
    from trackersplat_paper.utils.math_utils import compute_mean2D_equations


# ---------------------------------------------------------------------------
# Sparse → dense motion_map builder
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_sparse_motion_map(
    tracks_0: Tensor, tracks_t: Tensor, vis_t: Tensor,
    H: int, W: int, device: Optional[torch.device] = None,
) -> Tensor:
    """(N_pts, 2) + (N_pts, 2) + (N_pts,) → (H, W, 2) motion_map.
    NaN at pixels without a visible track."""
    if device is None:
        device = tracks_0.device
    p0x = tracks_0[:, 0].round().long().clamp(0, W - 1)
    p0y = tracks_0[:, 1].round().long().clamp(0, H - 1)
    valid = vis_t.bool() & torch.isfinite(tracks_t).all(dim=-1)
    motion_map = torch.full((H, W, 2), float("nan"), device=device, dtype=torch.float32)
    motion_map[p0y[valid], p0x[valid]] = tracks_t[valid].to(torch.float32)
    return motion_map


def _proj_matrix_row_major(cam) -> Tensor:
    """Build the reference's row-major proj matrix M such that
    p_hom = [p, 1] @ M, p_hom[3] is the homogeneous w coord.
    Used by compute_mean2D_equations (ISVD triangulation equations)."""
    W, H = int(cam.image_width), int(cam.image_height)
    fx, fy = cam.focal_x, cam.focal_y
    cx, cy = cam.center_x, cam.center_y
    K4 = torch.zeros(4, 4, device=cam.device, dtype=torch.float32)
    K4[0, 0] = 2 * fx / W
    K4[1, 1] = 2 * fy / H
    K4[0, 2] = 2 * cx / W - 1
    K4[1, 2] = 2 * cy / H - 1
    K4[2, 2] = 1.0
    K4[3, 2] = 1.0
    return (K4 @ cam.w2c).T.contiguous()


# ---------------------------------------------------------------------------
# Translation-only solver
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_translation_motion(
    gaussians, cameras: Sequence,
    tracks_per_cam: Sequence[Tensor],
    vis_per_cam: Sequence[Tensor],
    target_frame_idx: int, source_frame_idx: int = 0,
    alpha_threshold: float = 1e-3, min_pixhit: int = 3,
    verbose: bool = False,
) -> Motion:
    """Run motion_fusion across all views, fit per-Gaussian 2D affine via
    PWI-LS, triangulate 3D mean via ISVD. Returns Motion with translation_vector
    only (rotation/scale left to follow-up days)."""
    device = gaussians._xyz.device
    N = gaussians._xyz.shape[0]
    V = len(cameras)

    isvd = ISVD_Mean3D(batch_size=N, device=device)
    views_seen = torch.zeros(N, dtype=torch.int32, device=device)

    means3d = gaussians._xyz.detach().to(torch.float32)

    for vi, cam in enumerate(cameras):
        tracks = tracks_per_cam[vi]
        vis = vis_per_cam[vi]
        H, W = int(cam.image_height), int(cam.image_width)
        motion_map = build_sparse_motion_map(
            tracks_0=tracks[source_frame_idx],
            tracks_t=tracks[target_frame_idx],
            vis_t=vis[target_frame_idx],
            H=H, W=W, device=device,
        )
        out: MotionFusionOutput = motion_fusion(
            gaussians, cam, motion_map, alpha_threshold=alpha_threshold,
        )
        usable = out.pixhit >= min_pixhit
        n_use = int(usable.sum())
        if verbose:
            print(f"    view {vi}  pixhit≥{min_pixhit}: {n_use}/{N}  "
                  f"total pixels: {int(out.pixhit.sum())}")
        if n_use == 0:
            continue

        V1 = out.V1[usable].double()
        V2 = out.V2[usable].double()
        reg = torch.eye(3, device=device, dtype=torch.float64).expand_as(V1) * 1e-4
        M = torch.linalg.solve(V1 + reg, V2)          # (K, 3, 2)
        affine = M.transpose(-2, -1).to(torch.float32) # (K, 2, 3)
        A2D, b2D = affine[..., :2], affine[..., 2]

        proj = _proj_matrix_row_major(cam)
        # Fitted target 2D mean = A2D · p_0 + b2D, where p_0 is the Gaussian's
        # frame-0 projected centre. We read p_0 straight from motion_fusion's
        # projection path: it's the mean2d computed by _project_gaussians_2d.
        # For simplicity re-project here (same math as inside motion_fusion).
        focal_x = W / (2.0 * math.tan(cam.FoVx * 0.5))
        focal_y = H / (2.0 * math.tan(cam.FoVy * 0.5))
        cam_pts = means3d[usable] @ cam.w2c.T[:3, :3] + cam.w2c.T[:3, 3][None]
        z = cam_pts[:, 2:3].clamp_min(1e-6)
        u = focal_x * cam_pts[:, 0:1] / z + 0.5 * W
        v = focal_y * cam_pts[:, 1:2] / z + 0.5 * H
        p_0 = torch.cat([u, v], dim=-1)               # (K, 2)
        p_target = (A2D @ p_0.unsqueeze(-1)).squeeze(-1) + b2D   # (K, 2)

        A_eq = compute_mean2D_equations(proj, W, H, p_target)     # (K, 2, 4)

        weights = out.motion_alpha[usable].to(torch.float64).clamp_min(1e-6)
        isvd.update(A_eq.to(torch.float64), usable, weights)
        views_seen[usable] += 1

    valid_mask = views_seen >= 2
    n_valid = int(valid_mask.sum())
    if verbose:
        print(f"    [compute_translation] {n_valid}/{N} Gaussians seen in ≥2 views")
    if n_valid == 0:
        return Motion()

    mean3D_hat, solved_mask = isvd.solve(valid_mask.clone())
    # solved_mask now corresponds to Gaussians that ISVD actually solved
    # (ISVD may drop points with rank-deficient systems).
    base_means = means3d[solved_mask]
    translation = mean3D_hat.to(torch.float32) - base_means
    if verbose:
        trans_norm = translation.norm(dim=-1)
        print(f"    [compute_translation] |Δ_xyz| per Gaussian — "
              f"mean={trans_norm.mean():.4f}  max={trans_norm.max():.4f}")

    return Motion(
        motion_mask_mean=solved_mask,
        translation_vector=translation,
    )
