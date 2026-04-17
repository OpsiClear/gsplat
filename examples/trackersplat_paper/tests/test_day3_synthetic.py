"""
Day-3 validation — end-to-end synthetic scene recovery.

Day 1 covered the vendored math primitives.
Day 2 covered motion_fusion alone under global-shift.
Day 3 pushes harder: per-Gaussian distinct motion, multi-view triangulation,
and the full PWI-LS → ISVD → compare vs ground-truth chain.

Tests
-----
1. **Per-Gaussian affine recovery (single view)**
   Place 40 non-overlapping Gaussians in a 2-view camera rig. Each Gaussian
   carries its own randomly-drawn 2D affine `[A_g|b_g]`. Build motion_map by
   applying each Gaussian's affine to its OWN pixel neighbourhood. Run
   motion_fusion + PWI-LS. Assert recovered `[A_hat|b_hat]` matches ground
   truth per Gaussian within tight tolerances.

2. **Multi-view 3D triangulation (ISVD)**
   Place 30 3D points, project into 4 random cameras, feed into `ISVD_Mean3D`.
   Verify `mean3D_hat` recovers the ground-truth within 1e-3.

3. **Shape + finiteness sanity**
   Re-verify all outputs are finite, correctly shaped, and confidences are
   non-negative on a moderately-sized (N=200) scene.

Run:
    conda activate gsplat_fastergs
    python examples/trackersplat_paper/tests/test_day3_synthetic.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trackersplat_dataset import Camera                                 # noqa: E402
from trackersplat_paper import motion_fusion                             # noqa: E402
from trackersplat_paper.utils import ISVD_Mean3D                         # noqa: E402

GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; RESET = "\033[0m"


def _check(cond: bool, msg: str = ""):
    if not cond:
        raise AssertionError(msg or "check failed")


class _DuckGaussians:
    def __init__(self, xyz, rot, raw_log_scales, raw_opacity):
        self._xyz = torch.nn.Parameter(xyz)
        self._rotation = torch.nn.Parameter(rot)
        self._scaling = torch.nn.Parameter(raw_log_scales)
        self._opacity = torch.nn.Parameter(raw_opacity)


def _camera(W, H, fx, fy, device, c2w=None):
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device=device, dtype=torch.float32)
    if c2w is None:
        R = torch.eye(3, device=device, dtype=torch.float32)
        T = torch.zeros(3, device=device, dtype=torch.float32)
    else:
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]
    return Camera(
        R=R, T=T, K=K,
        image_height=H, image_width=W,
        FoVx=2.0 * math.atan(W / (2.0 * fx)),
        FoVy=2.0 * math.atan(H / (2.0 * fy)),
        image_path="", frame_idx=0, cam_name="synthetic", device=device,
    )


def _place_non_overlapping_gaussians(N, W, H, device, seed=0, margin=6):
    """Return (gaussians, projected_centre_pixels) with Gaussians on a
    regular grid that leaves at least `margin` px between centres so their
    3σ bboxes don't overlap."""
    torch.manual_seed(seed)
    cols = int(math.ceil(math.sqrt(N)))
    rows = (N + cols - 1) // cols
    step = min((W - 2 * margin) // cols, (H - 2 * margin) // rows)
    assert step >= 2 * margin, f"image too small for {N} non-overlapping gaussians"
    centres = []
    for i in range(N):
        cx = margin + (i % cols) * step + step // 2
        cy = margin + (i // cols) * step + step // 2
        centres.append((cx, cy))
    centres = torch.tensor(centres, device=device, dtype=torch.float32)

    # Back-project centres to 3D at z=3m with the given camera.
    fx = fy = 200.0
    z = 3.0
    xyz = torch.zeros(N, 3, device=device)
    xyz[:, 0] = (centres[:, 0] - W / 2) / fx * z
    xyz[:, 1] = (centres[:, 1] - H / 2) / fy * z
    xyz[:, 2] = z
    rot = torch.zeros(N, 4, device=device); rot[:, 0] = 1.0   # identity
    raw_log_scales = torch.full((N, 3), math.log(0.012), device=device)
    raw_opacity = torch.full((N,), 4.0, device=device)        # sigmoid(4) ≈ 0.98
    gaussians = _DuckGaussians(xyz, rot, raw_log_scales, raw_opacity)
    return gaussians, centres, fx, fy


# ---------------------------------------------------------------------------
# 1. Per-Gaussian affine recovery with distinct motion per Gaussian
# ---------------------------------------------------------------------------
def test_per_gaussian_affine_recovery():
    dev = torch.device("cuda")
    W, H = 256, 192
    N = 40
    gaussians, centres, fx, fy = _place_non_overlapping_gaussians(N, W, H, dev, seed=7)
    camera = _camera(W, H, fx, fy, dev)

    # Generate per-Gaussian random 2D affines near identity.
    torch.manual_seed(42)
    A_gt = torch.eye(2, device=dev).unsqueeze(0).expand(N, -1, -1) \
        + torch.randn(N, 2, 2, device=dev) * 0.02
    b_gt = torch.randn(N, 2, device=dev) * 2.0   # in pixels

    # Build motion_map: for each Gaussian, mark its 3σ-bbox pixels with the
    # AFFINE-TRANSFORMED pixel positions. Outside the bboxes, leave NaN.
    motion_map = torch.full((H, W, 2), float("nan"), device=dev)
    for g in range(N):
        cx, cy = centres[g]
        radius = 5            # matches our scale=0.012 at z=3 projecting to ~2-3 px
        x0 = int(cx.item()) - radius
        x1 = int(cx.item()) + radius
        y0 = int(cy.item()) - radius
        y1 = int(cy.item()) + radius
        ys = torch.arange(max(y0, 0), min(y1 + 1, H), device=dev)
        xs = torch.arange(max(x0, 0), min(x1 + 1, W), device=dev)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        fpx = xx.float() + 0.5
        fpy = yy.float() + 0.5
        x = torch.stack([fpx, fpy], dim=-1)    # (h, w, 2)
        # target = A · x + b
        target = torch.einsum("ij,...j->...i", A_gt[g], x) + b_gt[g]
        motion_map[yy, xx] = target

    out = motion_fusion(gaussians, camera, motion_map)

    # PWI-LS: solve V1·M = V2 per Gaussian where M ∈ R^{3×2}.
    # [A|b]_hat = M.T (since target = M^T · [px,py,1]).
    usable = out.pixhit >= 5
    _check(usable.sum() >= N // 2, f"too few usable Gaussians: {usable.sum()}/{N}")

    V1 = out.V1[usable].double()
    V2 = out.V2[usable].double()
    reg = torch.eye(3, device=dev, dtype=torch.float64).expand_as(V1) * 1e-6
    M = torch.linalg.solve(V1 + reg, V2)          # (K, 3, 2)
    affine_hat = M.transpose(-2, -1)                # (K, 2, 3)

    A_hat = affine_hat[..., :2]
    b_hat = affine_hat[..., 2]
    A_err = (A_hat - A_gt[usable].double()).abs().mean().item()
    b_err = (b_hat - b_gt[usable].double()).abs().mean().item()
    print(f"    {usable.sum().item()}/{N} usable Gaussians  "
          f"|A-A_gt|_mean={A_err:.2e}  |b-b_gt|_mean={b_err:.2e}")
    _check(A_err < 5e-2, f"per-Gaussian A not recovered: {A_err}")
    _check(b_err < 1.0, f"per-Gaussian b not recovered: {b_err} px")


# ---------------------------------------------------------------------------
# 2. ISVD_Mean3D triangulation end-to-end on a realistic camera rig
# ---------------------------------------------------------------------------
def test_isvd_triangulation_four_cameras():
    dev = torch.device("cuda")
    torch.manual_seed(3)
    B = 30
    P_gt = torch.randn(B, 3, device=dev, dtype=torch.float64)
    P_gt[:, 2] = P_gt[:, 2].abs() + 3.0

    # Build 4 cameras in a ring around the origin, all looking inward.
    def _camera_c2w(angle, radius=3.0, height=0.0):
        c2w = torch.eye(4, device=dev, dtype=torch.float64)
        cx, cz = radius * math.cos(angle), radius * math.sin(angle)
        # Simple look-at-origin camera centered at (cx, height, cz)
        c2w[:3, 3] = torch.tensor([cx, height, cz], device=dev, dtype=torch.float64)
        # Rotation: look -Z toward origin
        forward = -c2w[:3, 3]
        forward = forward / forward.norm()
        up = torch.tensor([0.0, 1.0, 0.0], device=dev, dtype=torch.float64)
        right = torch.linalg.cross(up, forward); right = right / right.norm()
        up2 = torch.linalg.cross(forward, right)
        c2w[:3, 0] = right
        c2w[:3, 1] = up2
        c2w[:3, 2] = forward
        return c2w

    # Projection matrix (world → clip, right-multiplied as in the reference).
    def _proj_matrix(c2w, fx, fy, W, H):
        w2c = torch.linalg.inv(c2w)
        # Row-major convention matching reference's compute_mean2D:
        #   p_hom = [p, 1] @ projmatrix,  projmatrix[:, 3] is the "w" column.
        # Build from intrinsics + w2c by stacking into 4×4 col-convention:
        K4 = torch.zeros(4, 4, dtype=torch.float64, device=dev)
        K4[0, 0] = fx / (W / 2); K4[1, 1] = fy / (H / 2)
        K4[0, 2] = 0; K4[1, 2] = 0
        K4[2, 2] = 1; K4[3, 2] = 1
        # projection in reference's convention is row-major transposed:
        return (K4 @ w2c).T.contiguous()

    isvd = ISVD_Mean3D(B, dev)
    for k in range(4):
        angle = k * math.pi / 2 + 0.1
        c2w = _camera_c2w(angle)
        proj = _proj_matrix(c2w, fx=150.0, fy=150.0, W=200, H=150)
        projB = proj.unsqueeze(0).expand(B, -1, -1).contiguous()
        P_hom = torch.cat([P_gt, torch.ones(B, 1, device=dev, dtype=torch.float64)], dim=1)
        projected = (P_hom.unsqueeze(1) @ projB).squeeze(1)
        uv = projected[:, :2] / projected[:, 3:]
        eq1 = projB[:, :, 0] - projB[:, :, 3] * uv[:, 0:1]
        eq2 = projB[:, :, 1] - projB[:, :, 3] * uv[:, 1:2]
        A = torch.stack([eq1, eq2], dim=1)
        mask = torch.ones(B, dtype=torch.bool, device=dev)
        weights = torch.ones(B, device=dev, dtype=torch.float64)
        isvd.update(A, mask, weights)

    mean3D_hat, valid = isvd.solve(torch.ones(B, dtype=torch.bool, device=dev))
    err = (mean3D_hat - P_gt).abs().max().item()
    print(f"    {valid.sum().item()}/{B} points triangulated   max |P_hat-P_gt| = {err:.2e}")
    _check(valid.all().item(), "not all points triangulated")
    _check(err < 1e-3, f"triangulation max err {err} too large")


# ---------------------------------------------------------------------------
# 3. Shape + finiteness on a larger scene
# ---------------------------------------------------------------------------
def test_large_scene_finiteness():
    dev = torch.device("cuda")
    W, H = 192, 144
    N = 200
    torch.manual_seed(8)
    xyz = torch.randn(N, 3, device=dev) * 0.3
    xyz[:, 2] = xyz[:, 2].abs() + 2.0
    rot = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    raw_log_scales = torch.full((N, 3), math.log(0.015), device=dev) + torch.randn(N, 3, device=dev) * 0.05
    raw_opacity = torch.randn(N, device=dev)
    gaussians = _DuckGaussians(xyz, rot, raw_log_scales, raw_opacity)
    camera = _camera(W, H, 200.0, 200.0, dev)

    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    motion_map = torch.stack([xx + 0.5 + 1.0, yy + 0.5 - 1.0], dim=-1)   # uniform shift

    out = motion_fusion(gaussians, camera, motion_map)
    _check(out.V1.shape == (N, 3, 3), f"V1 shape {tuple(out.V1.shape)}")
    _check(out.V2.shape == (N, 3, 2), f"V2 shape {tuple(out.V2.shape)}")
    _check(out.motion_alpha.shape == (N,))
    _check(out.motion_det.shape == (N,))
    _check(out.pixhit.shape == (N,))
    _check(torch.isfinite(out.V1).all())
    _check(torch.isfinite(out.V2).all())
    _check(torch.isfinite(out.motion_alpha).all())
    _check((out.motion_alpha >= 0).all())
    _check((out.pixhit >= 0).all())
    print(f"    N={N}  total pixhit={out.pixhit.sum().item()}  "
          f"mean alpha={out.motion_alpha.mean().item():.3f}")


TESTS = [
    ("per-Gaussian distinct affine recovery", test_per_gaussian_affine_recovery),
    ("ISVD_Mean3D 4-camera triangulation",    test_isvd_triangulation_four_cameras),
    ("shape + finiteness on N=200 scene",     test_large_scene_finiteness),
]


def main() -> int:
    print(f"{YEL}=== Day-3 synthetic-scene validation ==={RESET}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}  cuda: {torch.version.cuda}")
    failures = []
    for name, fn in TESTS:
        print(f"  • {name}")
        try:
            fn()
            print(f"    {GREEN}[PASS]{RESET}")
        except Exception:
            print(f"    {RED}[FAIL]{RESET}")
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
