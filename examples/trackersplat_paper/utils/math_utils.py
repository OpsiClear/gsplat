"""
Standard 3DGS projection + 2D/3D covariance helpers.

Independently implemented in pure PyTorch from the formulas in
Kerbl et al. 2023, "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
(SIGGRAPH 2023) and the EWA splatting derivations in Zwicker et al. 2002.

Apache-2.0. No Inria-derived code.

These functions replace the non-vendored helpers in the reference repo's
`utils/motionfusion/diff_gaussian_rasterization/motion_utils.py`. They are
drop-in compatible with `incremental_ls.py` and the PWI-LS solver.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


# -----------------------------------------------------------------------------
# Symmetric matrix pack/unpack — 6-vector ↔ 3×3 symmetric
# Layout:
#   A[..., 0] = m_00        A[..., 1] = m_01        A[..., 2] = m_02
#                           A[..., 3] = m_11        A[..., 4] = m_12
#                                                   A[..., 5] = m_22
# -----------------------------------------------------------------------------
def unflatten_symmetry_2x2(A: Tensor) -> Tensor:
    """(...,3) → (...,2,2) symmetric with layout [a00, a01, a11]."""
    m = torch.zeros((A.shape[0], 2, 2), dtype=A.dtype, layout=A.layout, device=A.device)
    m[..., 0, 0] = A[..., 0]
    m[..., 0, 1] = A[..., 1]
    m[..., 1, 0] = A[..., 1]
    m[..., 1, 1] = A[..., 2]
    return m


def unflatten_symmetry_3x3(A: Tensor) -> Tensor:
    """(...,6) → (...,3,3) symmetric with layout [a00, a01, a02, a11, a12, a22]."""
    m = torch.zeros((A.shape[0], 3, 3), dtype=A.dtype, layout=A.layout, device=A.device)
    m[..., 0, 0] = A[..., 0]
    m[..., 0, 1] = A[..., 1]
    m[..., 0, 2] = A[..., 2]
    m[..., 1, 0] = A[..., 1]
    m[..., 1, 1] = A[..., 3]
    m[..., 1, 2] = A[..., 4]
    m[..., 2, 0] = A[..., 2]
    m[..., 2, 1] = A[..., 4]
    m[..., 2, 2] = A[..., 5]
    return m


# -----------------------------------------------------------------------------
# Projection Jacobian (EWA splatting, local-affine approximation)
# For a 3D point (x, y, z) in camera space projecting to screen with focal f:
#   u = fx x / z + cx
#   v = fy y / z + cy
# The Jacobian J = ∂(u, v) / ∂(x, y, z) is:
#   J = [[fx/z,       0,      -fx x / z²],
#        [   0,    fy/z,      -fy y / z²]]
# The 3DGS paper additionally clamps tx/tz and ty/tz to limits ±1.3·tan(fov/2)
# to prevent extreme distortion near view-frustum edges.
# -----------------------------------------------------------------------------
def compute_Jacobian(
    mean: Tensor, fovx: float, fovy: float,
    width: int, height: int, view_matrix: Tensor,
) -> Tensor:
    """
    Args:
        mean: (N, 3) 3D positions in world space.
        fovx, fovy: field-of-view in radians.
        width, height: image plane resolution in pixels.
        view_matrix: (4, 4) world→camera (row-major, following reference code's
                    convention `t = view_matrix.T[:3, :3] @ mean.T + ...`).
    Returns:
        (N, 2, 3) projection Jacobian per point.
    """
    # world → camera
    t = view_matrix.T[:3, :3] @ mean.T + view_matrix.T[:3, 3, None]     # (3, N)
    tan_fovx = math.tan(fovx * 0.5)
    tan_fovy = math.tan(fovy * 0.5)
    focal_x = width / (2.0 * tan_fovx)
    focal_y = height / (2.0 * tan_fovy)
    # clamp to frustum-edge limits (3DGS paper §3 / Zwicker §4)
    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = txtz.clamp(-limx, limx) * t[2]
    t[1] = tytz.clamp(-limy, limy) * t[2]
    J = torch.zeros((mean.shape[0], 2, 3), device=mean.device)
    J[:, 0, 0] = focal_x / t[2]
    J[:, 0, 2] = -focal_x * t[0] / (t[2] ** 2)
    J[:, 1, 1] = focal_y / t[2]
    J[:, 1, 2] = -focal_y * t[1] / (t[2] ** 2)
    return J


def compute_T(J: Tensor, view_matrix: Tensor) -> Tensor:
    """T = J · W, where W = view_matrix's upper-left 3×3 block (world-to-camera
    rotation, with the reference's row-major T convention)."""
    return J @ view_matrix.T[:3, :3]


# -----------------------------------------------------------------------------
# 3D ↔ 2D covariance via EWA splatting:
#   cov2D = T · cov3D · Tᵀ    (2×2 = (2,3)·(3,3)·(3,2))
# Affine transform of a 2D covariance:
#   cov2D' = A · cov2D · Aᵀ
# -----------------------------------------------------------------------------
def compute_cov2D(T: Tensor, cov3D: Tensor) -> Tensor:
    return T.bmm(cov3D).bmm(T.transpose(1, 2))


def transform_cov2D(A: Tensor, cov2D: Tensor) -> Tensor:
    return A.bmm(cov2D).bmm(A.transpose(1, 2))


def compute_cov3D_equations(T: Tensor, cov2D: Tensor):
    """Construct the linear system that recovers the 6-parameter 3D covariance
    from a 2D covariance: given cov2D = T·Σ3D·Tᵀ, this returns (X, Y) such that
    X · vec(Σ3D) = Y per observation, where vec(Σ3D) is the 6-vector
    [σ₀₀, σ₀₁, σ₀₂, σ₁₁, σ₁₂, σ₂₂]. The 3 equations here encode:
        cov2D[0,0] = Σᵢⱼ T₀ᵢ T₀ⱼ σᵢⱼ       (for x)
        cov2D[0,1] = Σᵢⱼ T₁ᵢ T₀ⱼ σᵢⱼ       (for y)
        cov2D[1,1] = Σᵢⱼ T₁ᵢ T₁ⱼ σᵢⱼ       (for z)
    """
    X = torch.zeros((T.shape[0], 3, 6), device=T.device)
    # row 0 — cov2D[0,0]
    X[..., 0, 0] = T[..., 0, 0] ** 2
    X[..., 0, 1] = 2 * T[..., 0, 1] * T[..., 0, 0]
    X[..., 0, 2] = 2 * T[..., 0, 2] * T[..., 0, 0]
    X[..., 0, 3] = T[..., 0, 1] ** 2
    X[..., 0, 4] = 2 * T[..., 0, 1] * T[..., 0, 2]
    X[..., 0, 5] = T[..., 0, 2] ** 2
    # row 1 — cov2D[0,1]
    X[..., 1, 0] = T[..., 1, 0] * T[..., 0, 0]
    X[..., 1, 1] = T[..., 1, 1] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 1]
    X[..., 1, 2] = T[..., 1, 2] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 2]
    X[..., 1, 3] = T[..., 1, 1] * T[..., 0, 1]
    X[..., 1, 4] = T[..., 1, 1] * T[..., 0, 2] + T[..., 1, 2] * T[..., 0, 1]
    X[..., 1, 5] = T[..., 1, 2] * T[..., 0, 2]
    # row 2 — cov2D[1,1]
    X[..., 2, 0] = T[..., 1, 0] ** 2
    X[..., 2, 1] = 2 * T[..., 1, 1] * T[..., 1, 0]
    X[..., 2, 2] = 2 * T[..., 1, 2] * T[..., 1, 0]
    X[..., 2, 3] = T[..., 1, 1] ** 2
    X[..., 2, 4] = 2 * T[..., 1, 1] * T[..., 1, 2]
    X[..., 2, 5] = T[..., 1, 2] ** 2
    Y = torch.zeros((T.shape[0], 3, 1), device=T.device)
    Y[..., 0, 0] = cov2D[..., 0, 0]
    Y[..., 1, 0] = cov2D[..., 0, 1]
    Y[..., 2, 0] = cov2D[..., 1, 1]
    return X, Y


# -----------------------------------------------------------------------------
# Projection to pixel-space and the linear 3D-from-2D triangulation equations
# NDC convention:   p_proj_ndc = (2·p_pixel + 1) / (W, H) - 1      (range −1..1)
# Homogeneous 3D   ↦ image-space pixel via projmatrix:
#   p_hom = [p_orig, 1] @ projmatrix
#   p_proj = p_hom[:3] / p_hom[3]
#   p_pixel = ((p_proj[:2] + 1) * (W, H) - 1) / 2
# -----------------------------------------------------------------------------
def compute_mean2D(projmatrix: Tensor, W: int, H: int, p_orig: Tensor) -> Tensor:
    """(N, 3) → (N, 2) pixel-space mean. `projmatrix` is the full proj transform
    used in the reference code (row-major, right-multiplied)."""
    p_hom = torch.cat([p_orig, torch.ones((p_orig.shape[0], 1), device=p_orig.device)], dim=1) @ projmatrix
    p_w = 1.0 / (p_hom[:, -1:] + 1e-7)
    p_proj = p_hom[:, :-1] * p_w
    wh = torch.tensor([[W, H]], device=p_proj.device, dtype=p_proj.dtype)
    return ((p_proj[:, :2] + 1.0) * wh - 1.0) * 0.5


def compute_mean2D_equations(projmatrix: Tensor, W: int, H: int, point_image: Tensor) -> Tensor:
    """Build the 2-equation linear system that expresses a target pixel location
    as 2 constraints on the homogeneous 3D point. Stacked across multiple
    views this yields a linear system solved for 3D mean by min-singular-value
    SVD (see ISVD_Mean3D in incremental_svd.py)."""
    wh = torch.tensor([[W, H]], device=point_image.device, dtype=projmatrix.dtype)
    p_proj = (2.0 * point_image + 1.0) / wh - 1.0
    eq1 = projmatrix[:, 0] - projmatrix[:, 3] * p_proj[:, 0:1]
    eq2 = projmatrix[:, 1] - projmatrix[:, 3] * p_proj[:, 1:2]
    return torch.stack([eq1, eq2], dim=1)


# -----------------------------------------------------------------------------
# Compose the per-Gaussian linear-system contributions for the PWI-LS solver.
# Given per-view:
#   mean (N, 3), cov3D (N, 6 flat), camera intrinsics + projmatrix,
#   transform2d (N, 2, 3) — the fitted 2D affine [A2D | b2D] from motion_fusion
# returns (X, Y, A) that incremental_ls/svd consume.
# -----------------------------------------------------------------------------
def solve_transform(
    mean: Tensor, cov3D: Tensor,
    fovx: float, fovy: float, width: int, height: int,
    view_matrix: Tensor, full_proj_transform: Tensor,
    transform2d: Tensor,
):
    J = compute_Jacobian(mean, fovx, fovy, width, height, view_matrix)
    T = compute_T(J, view_matrix)
    A2D, b2D = transform2d[..., :-1], transform2d[..., -1]
    cov2D = compute_cov2D(T, unflatten_symmetry_3x3(cov3D))
    X, Y = compute_cov3D_equations(T, transform_cov2D(A2D, cov2D))
    point_image = compute_mean2D(full_proj_transform, width, height, mean)
    A = compute_mean2D_equations(
        full_proj_transform, width, height,
        (A2D @ point_image.unsqueeze(-1)).squeeze(-1) + b2D,
    )
    return X, Y, A
