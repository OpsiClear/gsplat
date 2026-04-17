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
# Project 3D Gaussians to a camera's 2D pixels — same math as motion_fusion
# ---------------------------------------------------------------------------
@torch.no_grad()
def _project_means_to_pixels(means3d: Tensor, cam) -> Tensor:
    """(N, 3) world means → (N, 2) pixel coords in this camera."""
    R = cam.w2c[:3, :3]
    T = cam.w2c[:3, 3]
    cam_pts = means3d @ R.T + T[None]
    z = cam_pts[:, 2:3].clamp_min(1e-6)
    u = cam.focal_x * cam_pts[:, 0:1] / z + cam.center_x
    v = cam.focal_y * cam_pts[:, 1:2] / z + cam.center_y
    return torch.cat([u, v], dim=-1)


# ---------------------------------------------------------------------------
# Direct-per-track triangulation — sparse-track-friendly fallback for PWI-LS.
#
# For each track point at source frame, find the nearest dynamic Gaussian in
# 2D pixel space. That Gaussian inherits the track's frame-t target pixel as
# its single 2D observation for this view. After all views are processed,
# each Gaussian with ≥2 views gets its 3D position triangulated via SVD on
# the stacked (camera projection ↔ target pixel) constraints.
#
# Works with sparse tracks (720 per view) where PWI-LS is under-constrained.
# Matches the "bind track to nearest Gaussian" heuristic the existing SGD
# trainer uses in examples/trackersplat_trainer_fastgs.py.
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_translation_motion_per_track(
    gaussians, cameras: Sequence,
    tracks_per_cam: Sequence[Tensor],
    vis_per_cam: Sequence[Tensor],
    target_frame_idx: int, source_frame_idx: int = 0,
    max_track_pixel_distance: float = 30.0,
    verbose: bool = False,
) -> Motion:
    """Returns a Motion with translation_vector only.

    Pipeline:
      1. For each camera, project all dynamic Gaussians to 2D at source frame.
      2. For each visible track, find its nearest projected Gaussian in pixel
         space. If the distance exceeds max_track_pixel_distance, drop it.
      3. Accumulate (A, pixel_target) pairs into ISVD_Mean3D.
      4. Solve; write translation_vector = mean3D_solved - base_means.
    """
    device = gaussians._xyz.device
    N = gaussians._xyz.shape[0]
    V = len(cameras)

    isvd = ISVD_Mean3D(batch_size=N, device=device)
    views_seen = torch.zeros(N, dtype=torch.int32, device=device)
    total_tracks_assigned = 0

    means3d = gaussians._xyz.detach().to(torch.float32)

    # STEP 1 — bind each track to one Gaussian, once, per camera using frame-0
    # proximity. Track indices ARE consistent across cameras (alltrackerxx
    # uses a fixed point grid), so per-camera bindings give a stable
    # track_idx → gaussian_idx mapping that we reuse for all target frames.
    per_cam_bindings = []            # (K_kept, gauss_idx_per_kept_track)
    per_cam_track_idx = []           # (K_kept,) original track indices in [0, N_pts)
    for vi, cam in enumerate(cameras):
        W, H = int(cam.image_width), int(cam.image_height)
        tracks = tracks_per_cam[vi]
        vis = vis_per_cam[vi].bool()
        pts0 = tracks[source_frame_idx]
        vis0 = vis[source_frame_idx]
        ok0 = vis0 & torch.isfinite(pts0).all(-1)
        track_idx_all = torch.arange(pts0.shape[0], device=device)
        track_idx_ok = track_idx_all[ok0]
        pts0_ok = pts0[ok0]
        proj2d = _project_means_to_pixels(means3d, cam)
        # chunked nearest-neighbour
        nearest_gi = torch.empty(pts0_ok.shape[0], dtype=torch.long, device=device)
        nearest_dist = torch.empty(pts0_ok.shape[0], device=device)
        chunk = 512
        for i in range(0, pts0_ok.shape[0], chunk):
            d = torch.cdist(pts0_ok[i:i+chunk], proj2d)
            vmin, imin = d.min(dim=-1)
            nearest_gi[i:i+chunk] = imin
            nearest_dist[i:i+chunk] = vmin
        within = nearest_dist <= max_track_pixel_distance
        # deduplicate: one track per Gaussian (keep the closest)
        sort_order = torch.argsort(nearest_dist[within])
        gi_sorted = nearest_gi[within][sort_order]
        ti_sorted = track_idx_ok[within][sort_order]
        seen = torch.zeros(N, dtype=torch.bool, device=device)
        keep_mask = torch.zeros_like(gi_sorted, dtype=torch.bool)
        for i, gi in enumerate(gi_sorted.cpu().tolist()):
            if not seen[gi]:
                seen[gi] = True
                keep_mask[i] = True
        per_cam_bindings.append(gi_sorted[keep_mask])
        per_cam_track_idx.append(ti_sorted[keep_mask])
        if verbose:
            print(f"    view {vi}: {int(within.sum())} in-radius, "
                  f"{int(keep_mask.sum())} after dedup → "
                  f"{int(keep_mask.sum())} track↔Gaussian bindings")

    # STEP 2 — feed each bound track's target pixel into ISVD, across all views.
    for vi, cam in enumerate(cameras):
        W, H = int(cam.image_width), int(cam.image_height)
        tracks = tracks_per_cam[vi]
        vis = vis_per_cam[vi].bool()
        track_gi = per_cam_bindings[vi]
        track_ti = per_cam_track_idx[vi]
        if track_gi.numel() == 0:
            continue
        ptsT = tracks[target_frame_idx][track_ti]        # (K, 2)
        visT = vis[target_frame_idx][track_ti]
        ok = visT & torch.isfinite(ptsT).all(-1)
        track_gi, ptsT = track_gi[ok], ptsT[ok]
        if track_gi.numel() == 0:
            continue

        K4 = torch.zeros(4, 4, device=device, dtype=torch.float32)
        K4[0, 0] = 2 * cam.focal_x / W
        K4[1, 1] = 2 * cam.focal_y / H
        K4[0, 2] = 2 * cam.center_x / W - 1
        K4[1, 2] = 2 * cam.center_y / H - 1
        K4[2, 2] = 1.0
        K4[3, 2] = 1.0
        proj = (K4 @ cam.w2c).T.contiguous()

        A_eq = compute_mean2D_equations(proj, W, H, ptsT)
        update_mask = torch.zeros(N, dtype=torch.bool, device=device)
        update_mask[track_gi] = True
        weights = torch.ones(track_gi.shape[0], device=device, dtype=torch.float64)
        isvd.update(A_eq.to(torch.float64), update_mask, weights)
        views_seen[track_gi] += 1
        total_tracks_assigned += track_gi.shape[0]

    valid_mask = views_seen >= 2
    n_valid = int(valid_mask.sum())
    if verbose:
        print(f"    [per-track] {total_tracks_assigned} tracks bound in total  "
              f"{n_valid}/{N} Gaussians seen in ≥2 views")
    if n_valid == 0:
        return Motion()

    mean3D_hat, solved_mask = isvd.solve(valid_mask.clone())
    base_means = means3d[solved_mask]
    translation = mean3D_hat.to(torch.float32) - base_means

    if verbose:
        norms = translation.norm(dim=-1)
        print(f"    [per-track] translation |Δ_xyz|   "
              f"mean={norms.mean():.4f}  p50={norms.median():.4f}  "
              f"max={norms.max():.4f}")

    return Motion(
        motion_mask_mean=solved_mask,
        translation_vector=translation,
    )


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
