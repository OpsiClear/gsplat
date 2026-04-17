"""
motion_fusion — per-Gaussian linear-system accumulator, pure-PyTorch.

Apache-2.0 replacement for the reference's Inria-derived CUDA pass at
`trackersplat/utils/motionfusion/__init__.py::motion_fusion`. Same contract
and same PWI-LS math as the paper.

Originally planned as a Taichi kernel for near-CUDA speed, but Taichi 1.7.3
fails to decorate new `@ti.kernel` after the vendored `propagation.py` and
`medianfilter.py` have initialised it (a quirk of the installed version's
NdarrayType annotation check). Falling back to a vectorised PyTorch path that
is ~5-10× slower than the paper's CUDA path but mathematically identical.

File name keeps the `_taichi` suffix for backwards-compat; the implementation
is pure PyTorch.

Paper reference: Yin et al. 2025, §4.3 "Parallel Weighted Incremental
Least Squares".

## Contract

Inputs (per view):
  gaussians    — duck-typed container with _xyz, _rotation, _scaling, _opacity
                 (_scaling is raw log-scale; _opacity is raw pre-sigmoid).
  camera       — examples.trackersplat_dataset.Camera
  motion_map   — (H, W, 2) float, per-pixel tracked target positions in pixel
                 coords (px', py'). Use NaN to mark pixels with no track.

Outputs (per Gaussian):
  V1           — (N, 3, 3): Σ_p  w(p,g) · [px, py, 1]·[px, py, 1]ᵀ
  V2           — (N, 3, 2): Σ_p  w(p,g) · [px, py, 1]·[px', py']ᵀ
  motion_alpha — (N,)     : Σ_p  w(p,g)
  motion_det   — (N,)     : det(2D cov_g)
  pixhit       — (N,)     : count of pixels where Gaussian g contributed

Current weight: w(p, g) = α(p, g). Transmittance T not yet modelled (add in a
follow-up if PSNR gap justifies the extra pass).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    from .utils.math_utils import compute_Jacobian, compute_T, compute_cov2D
except ImportError:
    from trackersplat_paper.utils.math_utils import compute_Jacobian, compute_T, compute_cov2D


@dataclass
class MotionFusionOutput:
    """Packed per-Gaussian linear-system accumulators produced by one view."""
    V1: Tensor            # (N, 3, 3)
    V2: Tensor            # (N, 3, 2)
    motion_alpha: Tensor  # (N,)
    motion_det: Tensor    # (N,)
    pixhit: Tensor        # (N,)  int32


# ---------------------------------------------------------------------------
# Projection: 3D Gaussians → (mean2d, conic2d, det2d, bbox) in pixel space
# ---------------------------------------------------------------------------
def _project_gaussians_2d(
    means3d: Tensor,
    rotations: Tensor,
    scales: Tensor,
    opacities: Tensor,
    fovx: float, fovy: float,
    width: int, height: int,
    w2c: Tensor,
    cam_position: Tensor,
    k_sigma: float = 3.0,
):
    N = means3d.shape[0]
    device = means3d.device

    # 3×3 rotation matrix from (w, x, y, z)
    r, x, y, z = rotations.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
        2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
        2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(N, 3, 3)
    S = torch.diag_embed(scales)
    RS = R @ S
    cov3D = RS @ RS.transpose(-2, -1)

    J = compute_Jacobian(means3d, fovx, fovy, width, height, w2c)
    T = compute_T(J, w2c)
    cov2D = compute_cov2D(T, cov3D)

    # Standard row-major w2c: rotation in w2c[:3, :3], translation in w2c[:3, 3].
    # Batch (row-vector) form: cam_pt_row = world_pt_row @ R.T + T_row.
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    cam_pts = means3d @ R.T + T[None]
    z_cam = cam_pts[:, 2:3].clamp_min(1e-6)
    focal_x = width / (2.0 * math.tan(fovx * 0.5))
    focal_y = height / (2.0 * math.tan(fovy * 0.5))
    u = focal_x * cam_pts[:, 0:1] / z_cam + 0.5 * width
    v = focal_y * cam_pts[:, 1:2] / z_cam + 0.5 * height
    mean2d = torch.cat([u, v], dim=-1)

    det2d = cov2D[..., 0, 0] * cov2D[..., 1, 1] - cov2D[..., 0, 1] ** 2
    det_safe = det2d.clamp_min(1e-6)
    inv = torch.empty_like(cov2D)
    inv[..., 0, 0] = cov2D[..., 1, 1] / det_safe
    inv[..., 0, 1] = -cov2D[..., 0, 1] / det_safe
    inv[..., 1, 0] = -cov2D[..., 0, 1] / det_safe
    inv[..., 1, 1] = cov2D[..., 0, 0] / det_safe
    conic = torch.stack([inv[..., 0, 0], inv[..., 0, 1], inv[..., 1, 1]], dim=-1)

    trace = cov2D[..., 0, 0] + cov2D[..., 1, 1]
    disc = torch.sqrt((trace * 0.5) ** 2 - det_safe).clamp_min(0.0)
    lam_max = trace * 0.5 + disc
    radius = k_sigma * torch.sqrt(lam_max.clamp_min(1e-6))
    xmin = (mean2d[:, 0] - radius).floor().long().clamp(0, width - 1)
    xmax = (mean2d[:, 0] + radius).ceil().long().clamp(0, width - 1)
    ymin = (mean2d[:, 1] - radius).floor().long().clamp(0, height - 1)
    ymax = (mean2d[:, 1] + radius).ceil().long().clamp(0, height - 1)
    bbox = torch.stack([xmin, ymin, xmax, ymax], dim=-1).to(torch.int32)

    valid = (z_cam.squeeze(-1) > 0) & (det_safe > 1e-6) & (radius > 0)
    # Force-invalid splats: empty bbox
    bbox[~valid, 0] = 1; bbox[~valid, 2] = 0
    bbox[~valid, 1] = 1; bbox[~valid, 3] = 0

    if opacities.dim() > 1:
        opacities = opacities.squeeze(-1)
    opacities = opacities.to(torch.float32)

    return (mean2d.to(torch.float32), conic.to(torch.float32),
            det2d.to(torch.float32), bbox.to(torch.int32), opacities, valid)


# ---------------------------------------------------------------------------
# Vectorised pure-PyTorch motion_fusion — one Gaussian at a time.
# Per Gaussian we compute α at every pixel in its 3σ bbox, then vectorise
# the V1/V2 outer-product accumulation across those pixels in one shot.
# ---------------------------------------------------------------------------
@torch.no_grad()
def motion_fusion(
    gaussians,
    camera,
    motion_map: Tensor,
    alpha_threshold: float = 1e-3,
    k_sigma: float = 3.0,
) -> MotionFusionOutput:
    device = camera.device
    means3d = gaussians._xyz.detach().to(device)
    rotations = torch.nn.functional.normalize(gaussians._rotation.detach(), dim=-1).to(device)
    scales = gaussians._scaling.detach().to(device)
    if scales.shape[-1] == 3:
        scales = scales.exp()
    opacities = gaussians._opacity.detach().to(device)
    if opacities.dim() == 2 and opacities.shape[1] == 1:
        opacities = opacities.squeeze(-1)
    opacities = torch.sigmoid(opacities)

    mean2d, conic, det2d, bbox, opac_act, _ = _project_gaussians_2d(
        means3d, rotations, scales, opacities,
        camera.FoVx, camera.FoVy,
        int(camera.image_width), int(camera.image_height),
        camera.w2c.to(device).to(means3d.dtype),
        camera.cam_position.to(device).to(means3d.dtype),
        k_sigma=k_sigma,
    )

    N = means3d.shape[0]
    H = int(camera.image_height); W = int(camera.image_width)
    V1 = torch.zeros(N, 3, 3, device=device)
    V2 = torch.zeros(N, 3, 2, device=device)
    motion_alpha = torch.zeros(N, device=device)
    pixhit = torch.zeros(N, dtype=torch.int32, device=device)

    if motion_map.shape != (H, W, 2):
        raise ValueError(f"motion_map must be (H={H}, W={W}, 2), got {tuple(motion_map.shape)}")
    motion_map = motion_map.to(torch.float32).to(device)

    bbox_cpu = bbox.cpu().tolist()
    for g in range(N):
        xmin, ymin, xmax, ymax = bbox_cpu[g]
        if xmax < xmin or ymax < ymin:
            continue
        ys = torch.arange(ymin, ymax + 1, device=device, dtype=torch.float32)
        xs = torch.arange(xmin, xmax + 1, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        fpx = xx.flatten() + 0.5
        fpy = yy.flatten() + 0.5
        dx = fpx - mean2d[g, 0]
        dy = fpy - mean2d[g, 1]
        a, b, c = conic[g, 0], conic[g, 1], conic[g, 2]
        exponent = (-0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy).clamp_max(0.0)
        alpha = opac_act[g] * torch.exp(exponent)
        keep = alpha >= alpha_threshold
        fpx, fpy, alpha = fpx[keep], fpy[keep], alpha[keep]
        if fpx.numel() == 0:
            continue
        # lookup target pixels, mask NaNs
        yi = yy.long().flatten()[keep]
        xi = xx.long().flatten()[keep]
        tx = motion_map[yi, xi, 0]
        ty = motion_map[yi, xi, 1]
        nan_keep = torch.isfinite(tx) & torch.isfinite(ty)
        if int(nan_keep.sum()) == 0:
            continue
        fpx, fpy, alpha = fpx[nan_keep], fpy[nan_keep], alpha[nan_keep]
        tx, ty = tx[nan_keep], ty[nan_keep]
        x_h = torch.stack([fpx, fpy, torch.ones_like(fpx)], dim=-1)   # (M, 3)
        targets = torch.stack([tx, ty], dim=-1)                        # (M, 2)
        w = alpha                                                       # (M,)
        V1[g] = (w[:, None, None] * x_h[:, :, None] * x_h[:, None, :]).sum(0)
        V2[g] = (w[:, None, None] * x_h[:, :, None] * targets[:, None, :]).sum(0)
        motion_alpha[g] = w.sum()
        pixhit[g] = fpx.numel()

    return MotionFusionOutput(
        V1=V1.to(torch.float32), V2=V2.to(torch.float32),
        motion_alpha=motion_alpha.to(torch.float32),
        motion_det=det2d.to(torch.float32),
        pixhit=pixhit,
    )


# Legacy alias for tests that imported motion_fusion_reference previously.
motion_fusion_reference = motion_fusion
