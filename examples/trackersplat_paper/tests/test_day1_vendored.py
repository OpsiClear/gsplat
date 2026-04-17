"""
Day-1 validation: every vendored / adapted module actually works.

Run from repo root:
    conda activate gsplat_fastergs
    python examples/trackersplat_paper/tests/test_day1_vendored.py

Each test:
  1. math_utils.unflatten_symmetry_3x3     — pack/unpack round-trip
  2. math_utils.compute_Jacobian           — matches analytic formula for a
                                             known point + view
  3. math_utils.solve_transform            — end-to-end X/Y/A shapes + finite
  4. ILS_RotationScale                     — synthetic 3D cov recovery
  5. ISVD_Mean3D                           — 3D point triangulation from
                                             stacked 2-view projections
  6. Taichi motion_median_filter           — filter removes a planted outlier
  7. Taichi propagate                      — 8-NN propagation expands a
                                             seeded mask
  8. Motion.validate                       — rejects malformed tuples
  9. compensate + compare                  — round-trip on a duck-typed
                                             Gaussian container
 10. motion_fusion stub                    — raises NotImplementedError

Prints [PASS] / [FAIL] per test. Exits non-zero on any failure.
"""
from __future__ import annotations
import math
import sys
import traceback

import torch

# Make the examples/ dir importable as a package.
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trackersplat_paper import Motion, compensate, compare, motion_fusion  # noqa: E402
from trackersplat_paper.utils import (                                      # noqa: E402
    ILS_RotationScale,
    ISVD_Mean3D,
    motion_median_filter,
    propagate,
    unflatten_symmetry_3x3,
    compute_Jacobian,
    compute_T,
    solve_transform,
)


GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; RESET = "\033[0m"


def _check(cond: bool, msg: str = ""):
    if not cond:
        raise AssertionError(msg or "check failed")


# ---------------------------------------------------------------------------
# 1. math_utils — symmetric pack/unpack
# ---------------------------------------------------------------------------
def test_unflatten_symmetry_3x3():
    dev = "cuda"
    # build a known symmetric matrix, flatten to 6-vector in the layout
    # [a00, a01, a02, a11, a12, a22], unflatten, compare.
    M = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 4.0, 5.0],
                      [3.0, 5.0, 6.0]], device=dev)
    flat = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], device=dev)
    out = unflatten_symmetry_3x3(flat)
    _check(out.shape == (1, 3, 3), f"shape {tuple(out.shape)}")
    _check(torch.allclose(out[0], M), f"mismatch:\n{out[0]}\nvs\n{M}")


# ---------------------------------------------------------------------------
# 2. math_utils — Jacobian on a known point
# For a point (0, 0, z) on the optical axis with focal (f, f):
#   J = [[f/z,    0,     0],
#        [   0,  f/z,    0]]
# ---------------------------------------------------------------------------
def test_compute_Jacobian_on_axis():
    dev = "cuda"
    W, H = 512, 512
    fovx = fovy = 2.0 * math.atan(W / (2.0 * 400.0))    # focal = 400 px
    view = torch.eye(4, device=dev)                      # identity w2c
    mean = torch.tensor([[0.0, 0.0, 3.0]], device=dev)   # 3 m in front
    J = compute_Jacobian(mean, fovx, fovy, W, H, view)
    _check(J.shape == (1, 2, 3), f"J shape {tuple(J.shape)}")
    # Note: compute_Jacobian modifies `t` in-place during clamping, so the
    # off-axis columns (J[0,2] and J[1,2]) can be non-zero if the point is
    # clamped away from the optical axis. For a point exactly on-axis (0,0,z)
    # the clamp does nothing and the J[*,2] entries MUST be zero.
    # focal recomputed: f = W / (2 tan(fovx/2))
    f = W / (2.0 * math.tan(fovx * 0.5))
    expect_00 = f / 3.0
    expect_11 = f / 3.0
    _check(abs(J[0, 0, 0].item() - expect_00) < 1e-4, f"J[0,0]={J[0,0,0]}, expect {expect_00}")
    _check(abs(J[0, 1, 1].item() - expect_11) < 1e-4, f"J[1,1]={J[0,1,1]}, expect {expect_11}")
    _check(abs(J[0, 0, 2].item()) < 1e-4)              # off-axis term = 0
    _check(abs(J[0, 1, 2].item()) < 1e-4)


# ---------------------------------------------------------------------------
# 3. math_utils.solve_transform — runs end-to-end, outputs finite + right shapes
# ---------------------------------------------------------------------------
def test_solve_transform_shapes():
    dev = "cuda"
    N = 50
    mean = torch.randn(N, 3, device=dev); mean[:, 2] = mean[:, 2].abs() + 1.0
    cov3D_flat = torch.zeros(N, 6, device=dev)
    cov3D_flat[:, 0] = 0.02; cov3D_flat[:, 3] = 0.02; cov3D_flat[:, 5] = 0.02   # diag
    fovx = fovy = 1.0
    W = H = 256
    view = torch.eye(4, device=dev)
    proj = torch.eye(4, device=dev)
    # fitted 2D affine per Gaussian — identity motion
    transform2d = torch.zeros(N, 2, 3, device=dev)
    transform2d[:, 0, 0] = 1.0
    transform2d[:, 1, 1] = 1.0
    X, Y, A = solve_transform(mean, cov3D_flat, fovx, fovy, W, H, view, proj, transform2d)
    _check(X.shape == (N, 3, 6), f"X shape {tuple(X.shape)}")
    _check(Y.shape == (N, 3, 1), f"Y shape {tuple(Y.shape)}")
    # A is per-view mean-projection constraints: 2 rows (from u,v) by 4 columns
    # (homogeneous 3D coords). Reference: compute_mean2D_equations.
    _check(A.shape == (N, 2, 4), f"A shape {tuple(A.shape)}")
    _check(torch.isfinite(X).all() and torch.isfinite(Y).all() and torch.isfinite(A).all())


# ---------------------------------------------------------------------------
# 4. ILS_RotationScale — recover a known 3D cov (pure PyTorch, no Taichi)
# Stuff 3 linearly-independent measurements of cov2D (= T·Σ3D·Tᵀ with 3 different
# Ts), then solve and check we get back Σ3D.
# ---------------------------------------------------------------------------
def test_ILS_RotationScale_round_trip():
    dev = "cuda"
    B = 4       # batch size
    # Ground-truth 3D covariance — diagonal for simplicity
    sigma_gt = torch.tensor([0.5, 0.3, 0.1], device=dev)   # scales²
    cov3D_gt = torch.diag_embed(sigma_gt.expand(B, 3))      # (B, 3, 3)

    # Build ILS_RotationScale. It expects batch of B Gaussians, keep k=4 rows for error.
    # Note: ILS_Cov3D's `n` param represents the matrix order (3 for a 3×3 symmetric),
    # it will flatten to n*(n+1)//2 = 6 internally.
    ils = ILS_RotationScale(batch_size=B, k=4)

    # Feed 4 (k=4) arbitrary but linearly independent T matrices to each Gaussian.
    torch.manual_seed(0)
    for step in range(4):
        T = torch.randn(B, 2, 3, device=dev).type(torch.float64)
        cov3D_b = cov3D_gt.type(torch.float64)
        cov2D = T.bmm(cov3D_b).bmm(T.transpose(1, 2))
        # Build X, Y for cov3D equations (matches compute_cov3D_equations layout)
        X = torch.zeros(B, 3, 6, device=dev, dtype=torch.float64)
        X[..., 0, 0] = T[..., 0, 0] ** 2
        X[..., 0, 1] = 2 * T[..., 0, 1] * T[..., 0, 0]
        X[..., 0, 2] = 2 * T[..., 0, 2] * T[..., 0, 0]
        X[..., 0, 3] = T[..., 0, 1] ** 2
        X[..., 0, 4] = 2 * T[..., 0, 1] * T[..., 0, 2]
        X[..., 0, 5] = T[..., 0, 2] ** 2
        X[..., 1, 0] = T[..., 1, 0] * T[..., 0, 0]
        X[..., 1, 1] = T[..., 1, 1] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 1]
        X[..., 1, 2] = T[..., 1, 2] * T[..., 0, 0] + T[..., 1, 0] * T[..., 0, 2]
        X[..., 1, 3] = T[..., 1, 1] * T[..., 0, 1]
        X[..., 1, 4] = T[..., 1, 1] * T[..., 0, 2] + T[..., 1, 2] * T[..., 0, 1]
        X[..., 1, 5] = T[..., 1, 2] * T[..., 0, 2]
        X[..., 2, 0] = T[..., 1, 0] ** 2
        X[..., 2, 1] = 2 * T[..., 1, 1] * T[..., 1, 0]
        X[..., 2, 2] = 2 * T[..., 1, 2] * T[..., 1, 0]
        X[..., 2, 3] = T[..., 1, 1] ** 2
        X[..., 2, 4] = 2 * T[..., 1, 1] * T[..., 1, 2]
        X[..., 2, 5] = T[..., 1, 2] ** 2
        Y = torch.zeros(B, 3, 1, device=dev, dtype=torch.float64)
        Y[..., 0, 0] = cov2D[..., 0, 0]
        Y[..., 1, 0] = cov2D[..., 0, 1]
        Y[..., 2, 0] = cov2D[..., 1, 1]
        weight = torch.ones(B, device=dev, dtype=torch.float64)
        valid_mask = torch.ones(B, dtype=torch.bool, device=dev)
        ils.update(X, Y, valid_mask, weight)

    R, S, error, mask = ils.solve(torch.ones(B, dtype=torch.bool, device=dev))
    # S holds sqrt(eigenvalues) — i.e. per-axis scales. Since our GT cov3D is
    # diagonal with values sigma_gt, the eigenvalues ARE sigma_gt and S should
    # be sqrt(sigma_gt). Compare after sorting (eigendecomp order is arbitrary).
    expected_S = torch.sqrt(sigma_gt).sort().values
    got_S = S.sort(-1).values   # (B, 3)
    err = (got_S - expected_S[None]).abs().max().item()
    _check(err < 1e-3, f"ILS scale recovery err={err}  "
                       f"got={got_S[0].cpu().tolist()}  expected={expected_S.cpu().tolist()}")


# ---------------------------------------------------------------------------
# 5. ISVD_Mean3D — triangulate a known 3D point from 2 views
# For a point P in world, the projection constraint from each view is a 2×4
# linear system on homogeneous P. Stacking 2 views gives a rank-3 4×4 system;
# the min-σ right-singular vector is P_hom.
# ---------------------------------------------------------------------------
def test_ISVD_Mean3D_two_view():
    dev = "cuda"
    B = 3
    torch.manual_seed(1)
    P_gt = torch.randn(B, 3, device=dev).type(torch.float64)

    isvd = ISVD_Mean3D(B, dev)
    for view in range(4):
        # Random projection matrix per view, non-degenerate
        proj = torch.randn(4, 4, device=dev, dtype=torch.float64) * 0.5
        proj[3, 3] += 1.0
        projB = proj.unsqueeze(0).expand(B, -1, -1).contiguous()     # (B, 4, 4)
        # Project P_gt with this view: projected_hom = P_hom @ proj
        P_hom = torch.cat([P_gt, torch.ones(B, 1, device=dev, dtype=torch.float64)], dim=1)
        projected = (P_hom.unsqueeze(1) @ projB).squeeze(1)          # (B, 4)
        uv = projected[:, :2] / projected[:, 3:]                     # (B, 2)
        # Per-Gaussian A (B, 2, 4): projection constraints p - proj*uv.
        # Use proj[:, k] (k-th column) as the (4,) vector for each equation.
        eq1 = projB[:, :, 0] - projB[:, :, 3] * uv[:, 0:1]            # (B, 4)
        eq2 = projB[:, :, 1] - projB[:, :, 3] * uv[:, 1:2]            # (B, 4)
        A = torch.stack([eq1, eq2], dim=1)                            # (B, 2, 4)
        mask = torch.ones(B, dtype=torch.bool, device=dev)
        weights = torch.ones(B, device=dev, dtype=torch.float64)
        isvd.update(A, mask, weights)

    mean3D, valid = isvd.solve(torch.ones(B, dtype=torch.bool, device=dev))
    err = (mean3D - P_gt).abs().max().item()
    _check(valid.all().item())
    _check(err < 1e-3, f"triangulation err={err}")


# ---------------------------------------------------------------------------
# 6. Taichi motion_median_filter — planted outlier gets removed
# ---------------------------------------------------------------------------
def test_median_filter_removes_outlier():
    dev = "cuda"
    N = 50; K = 8
    # All motions are [1, 2, 3] except one outlier at [100, 100, 100]
    mask = torch.ones(N, dtype=torch.bool, device=dev)
    motion = torch.tensor([1.0, 2.0, 3.0], device=dev).expand(N, 3).contiguous()
    motion[25] = torch.tensor([100.0, 100.0, 100.0], device=dev)
    # Each point's K nearest neighbours are itself and K-1 random others
    torch.manual_seed(2)
    neighbor_indices = torch.randint(0, N, (N, K), device=dev)
    neighbor_weights = torch.ones((N, K), device=dev)
    filtered = motion_median_filter(mask, motion.clone(), neighbor_indices, neighbor_weights)
    # Median over neighbours (mostly [1,2,3]) should pull the outlier back.
    deviation_before = (motion[25] - torch.tensor([1.0, 2.0, 3.0], device=dev)).abs().max()
    deviation_after = (filtered[25] - torch.tensor([1.0, 2.0, 3.0], device=dev)).abs().max()
    _check(deviation_after < deviation_before * 0.5,
           f"filter didn't reduce outlier: before={deviation_before}, after={deviation_after}")


# ---------------------------------------------------------------------------
# 7. Taichi propagate — mask grows from seed
# ---------------------------------------------------------------------------
def test_propagate_grows_mask():
    dev = "cuda"
    N = 20; K = 4
    torch.manual_seed(3)
    init_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    init_mask[:3] = True         # seed 3 points
    init_value_at_mask = torch.randn(3, 2, device=dev)
    init_weight_at_mask = torch.ones(3, device=dev)
    # Build a "ring" of neighbours so propagation actually spreads
    neighbor_indices = torch.arange(N, device=dev).unsqueeze(1) + torch.arange(1, K + 1, device=dev).unsqueeze(0)
    neighbor_indices = neighbor_indices % N
    neighbor_weights = torch.ones((N, K), device=dev) / K
    values, weights = propagate(
        init_mask, init_value_at_mask, init_weight_at_mask,
        neighbor_indices.long(), neighbor_weights, n_iter=10,
    )
    # At least SOME new points should have non-zero weights after propagation.
    new_active = (weights > 0) & (~init_mask)
    _check(new_active.sum() > 0,
           f"propagation didn't cover any new points (new_active.sum()={new_active.sum()})")


# ---------------------------------------------------------------------------
# 8. Motion.validate — rejects malformed
# ---------------------------------------------------------------------------
def test_motion_validate():
    Motion().validate()   # empty — should be fine
    fixed_mask = torch.tensor([True, False, True])
    # confidence_fix is a SEPARATE mask, size == fixed_mask.sum() (2 here) — OK
    Motion(fixed_mask=fixed_mask, confidence_fix=torch.tensor([0.9, 0.8])).validate()
    # confidence_fix with wrong size — must fail
    try:
        Motion(fixed_mask=fixed_mask, confidence_fix=torch.tensor([0.9])).validate()
    except AssertionError:
        return
    raise AssertionError("Motion.validate() should have rejected mismatched confidence_fix")


# ---------------------------------------------------------------------------
# 9. compensate + compare round-trip on a duck-typed container
# ---------------------------------------------------------------------------
class _DuckGaussians:
    def __init__(self, n: int, device="cuda"):
        self._xyz = torch.nn.Parameter(torch.randn(n, 3, device=device))
        self._rotation = torch.nn.functional.normalize(torch.randn(n, 4, device=device), dim=-1)
        self._rotation = torch.nn.Parameter(self._rotation)
        self._scaling = torch.nn.Parameter(torch.randn(n, 3, device=device) * 0.2)


def test_compensate_round_trip():
    dev = "cuda"
    base = _DuckGaussians(10, dev)
    # Apply a known per-Gaussian translation
    dt = torch.randn(10, 3, device=dev) * 0.1
    motion = Motion(translation_vector=dt)
    new = compensate(base, motion)
    err = (new._xyz.detach() - base._xyz.detach() - dt).abs().max().item()
    _check(err < 1e-6, f"translation round-trip err={err}")


# ---------------------------------------------------------------------------
# 10. motion_fusion stub raises
# ---------------------------------------------------------------------------
def test_motion_fusion_is_callable():
    # Day-1 only asserted this was a stub. After Day 2 it's implemented;
    # we only check here that the symbol is a callable function so the
    # package surface is intact.
    _check(callable(motion_fusion), "motion_fusion not callable")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
TESTS = [
    ("math_utils.unflatten_symmetry_3x3",  test_unflatten_symmetry_3x3),
    ("math_utils.compute_Jacobian",        test_compute_Jacobian_on_axis),
    ("math_utils.solve_transform shapes",  test_solve_transform_shapes),
    ("ILS_RotationScale round-trip",       test_ILS_RotationScale_round_trip),
    ("ISVD_Mean3D triangulation",          test_ISVD_Mean3D_two_view),
    ("Taichi motion_median_filter",        test_median_filter_removes_outlier),
    ("Taichi propagate",                   test_propagate_grows_mask),
    ("Motion.validate",                    test_motion_validate),
    ("compensate round-trip",              test_compensate_round_trip),
    ("motion_fusion exists and is callable", test_motion_fusion_is_callable),
]


def main() -> int:
    print(f"{YEL}=== TrackerSplat Day-1 validation ==={RESET}")
    print(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    print(f"torch:  {torch.__version__}  cuda: {torch.version.cuda}")
    print()
    failures = []
    for name, fn in TESTS:
        try:
            fn()
            print(f"  {GREEN}[PASS]{RESET}  {name}")
        except Exception:
            print(f"  {RED}[FAIL]{RESET}  {name}")
            traceback.print_exc()
            failures.append(name)
    print()
    if failures:
        print(f"{RED}{len(failures)}/{len(TESTS)} failed{RESET}: {failures}")
        return 1
    print(f"{GREEN}all {len(TESTS)} passed{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
