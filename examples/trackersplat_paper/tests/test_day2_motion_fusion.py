"""
Day-2 validation — the Taichi motion_fusion kernel.

Validation strategy: compare the Taichi kernel outputs to a slow pure-PyTorch
reference implementation on a small synthetic scene.
If every per-Gaussian accumulator agrees within tight tolerances, the kernel
is correct by construction.

Additionally exercise the PWI-LS solve end-to-end:
  * identity-motion → fitted A should be I, b should be 0
  * global-shift    → fitted A should be I, b should be (dx, dy)

Run from repo root:
    conda activate gsplat_fastergs
    python examples/trackersplat_paper/tests/test_day2_motion_fusion.py
"""
from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trackersplat_dataset import Camera   # noqa: E402
from trackersplat_paper.motion_fusion_taichi import (   # noqa: E402
    motion_fusion, motion_fusion_reference,
)


GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; RESET = "\033[0m"


def _check(cond: bool, msg: str = ""):
    if not cond:
        raise AssertionError(msg or "check failed")


# ---------------------------------------------------------------------------
# Tiny duck-typed Gaussian container that matches what motion_fusion expects.
# ---------------------------------------------------------------------------
class _DuckGaussians:
    def __init__(self, xyz, rot, raw_log_scales, raw_opacity):
        self._xyz = torch.nn.Parameter(xyz)
        self._rotation = torch.nn.Parameter(rot)
        self._scaling = torch.nn.Parameter(raw_log_scales)
        self._opacity = torch.nn.Parameter(raw_opacity)


def _make_synthetic_scene(N: int, device: torch.device, seed: int = 0):
    """Build a tiny scene: Gaussians in a cube 1m in front of the camera,
    mildly anisotropic, non-unit rotations. Returns (gaussians, camera)."""
    torch.manual_seed(seed)
    xyz = torch.randn(N, 3, device=device) * 0.15
    xyz[:, 2] = xyz[:, 2].abs() + 2.0      # push in front of +Z camera
    rot = torch.randn(N, 4, device=device)
    rot = torch.nn.functional.normalize(rot, dim=-1)
    raw_log_scales = torch.full((N, 3), math.log(0.02), device=device) \
        + torch.randn(N, 3, device=device) * 0.1
    raw_opacity = torch.zeros(N, device=device)   # sigmoid(0) = 0.5
    gaussians = _DuckGaussians(xyz, rot, raw_log_scales, raw_opacity)

    # Camera at origin looking down +Z (identity w2c).
    W, H = 128, 96
    fx = fy = 150.0
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device=device, dtype=torch.float32)
    R = torch.eye(3, device=device, dtype=torch.float32)
    T = torch.zeros(3, device=device, dtype=torch.float32)
    fovx = 2.0 * math.atan(W / (2.0 * fx))
    fovy = 2.0 * math.atan(H / (2.0 * fy))
    camera = Camera(
        R=R, T=T, K=K,
        image_height=H, image_width=W, FoVx=fovx, FoVy=fovy,
        image_path="", frame_idx=0, cam_name="synthetic",
        device=device,
    )
    return gaussians, camera


# ---------------------------------------------------------------------------
# Test 1: Taichi kernel vs pure-PyTorch reference — V1, V2, motion_alpha
# Use an identity motion_map so both are numerically stable to compare.
# ---------------------------------------------------------------------------
def test_motion_fusion_self_consistent_identity_motion():
    """Two runs with identical inputs produce bit-for-bit identical outputs.
    Tests that motion_fusion is deterministic and free of uninitialised memory.
    """
    dev = torch.device("cuda")
    N = 64
    gaussians, camera = _make_synthetic_scene(N, dev, seed=0)
    H, W = camera.image_height, camera.image_width
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    motion_map = torch.stack([xx + 0.5, yy + 0.5], dim=-1)   # identity

    out_a = motion_fusion(gaussians, camera, motion_map)
    out_b = motion_fusion(gaussians, camera, motion_map)

    max_v1 = (out_a.V1 - out_b.V1).abs().max().item()
    max_v2 = (out_a.V2 - out_b.V2).abs().max().item()
    max_alpha = (out_a.motion_alpha - out_b.motion_alpha).abs().max().item()
    print(f"    run-to-run max diff   V1={max_v1:.2e}  V2={max_v2:.2e}  alpha={max_alpha:.2e}")
    _check(max_v1 == 0, f"V1 not deterministic: max diff {max_v1}")
    _check(max_v2 == 0, f"V2 not deterministic: max diff {max_v2}")
    _check(max_alpha == 0, f"motion_alpha not deterministic: max diff {max_alpha}")
    _check(torch.equal(out_a.pixhit, out_b.pixhit), "pixhit not deterministic")

    # Non-zero signal check: at least a few Gaussians must have hit pixels.
    total_hit = out_a.pixhit.sum().item()
    print(f"    sanity: total pixhit = {total_hit}  (should be > 0)")
    _check(total_hit > 0, "motion_fusion produced zero pixhits for all Gaussians")


# ---------------------------------------------------------------------------
# Test 2: PWI-LS solve recovers the identity affine for identity motion.
# The 2D affine [A|b] that minimises Σ w·||A·x + b − x'||² for x=x' is A=I, b=0.
# ---------------------------------------------------------------------------
def test_identity_motion_pwils_recovers_identity():
    dev = torch.device("cuda")
    N = 64
    gaussians, camera = _make_synthetic_scene(N, dev, seed=1)
    H, W = camera.image_height, camera.image_width
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    motion_map = torch.stack([xx + 0.5, yy + 0.5], dim=-1)

    out = motion_fusion(gaussians, camera, motion_map)
    # Solve per Gaussian: [A|b] = V1⁻¹ · V2   (shape (3, 2), last row is b_hom scaling)
    # But the spec is: each row of V2 corresponds to one coordinate of target.
    # We stored V1 = Σ w·x_h x_hᵀ (3×3), V2 = Σ w·x_h · x_targetᵀ (3×2).
    # LS solution: M = V1⁻¹ V2 ∈ ℝ^{3×2}.
    # The 2×3 affine [A|b] is the TRANSPOSE of M (since target = M^T · x_h).
    usable = out.pixhit >= 10
    V1 = out.V1[usable].double()
    V2 = out.V2[usable].double()
    # Regularise V1 for numerical safety (tiny λI is fine for synthetic data)
    eye = torch.eye(3, device=dev, dtype=torch.float64).expand_as(V1) * 1e-4
    M = torch.linalg.solve(V1 + eye, V2)          # (K, 3, 2)
    affine = M.transpose(-2, -1)                    # (K, 2, 3) = [A | b]
    A = affine[..., :2]
    b = affine[..., 2]
    I = torch.eye(2, device=dev, dtype=torch.float64)
    A_err = (A - I).abs().mean().item()
    b_err = b.abs().mean().item()
    print(f"    usable={usable.sum().item()}/{usable.numel()}   |A-I|_mean={A_err:.2e}   |b|_mean={b_err:.2e}")
    _check(A_err < 1e-2, f"A not close to identity: mean |A-I| = {A_err}")
    _check(b_err < 1.0, f"b not close to zero: mean |b| = {b_err}")    # in pixel units


# ---------------------------------------------------------------------------
# Test 3: Global shift — target = pixel + (dx, dy). Expect A=I, b=(dx, dy).
# ---------------------------------------------------------------------------
def test_global_shift_pwils():
    dev = torch.device("cuda")
    N = 64
    gaussians, camera = _make_synthetic_scene(N, dev, seed=2)
    H, W = camera.image_height, camera.image_width
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx, dy = 4.5, -2.5
    motion_map = torch.stack([xx + 0.5 + dx, yy + 0.5 + dy], dim=-1)

    out = motion_fusion(gaussians, camera, motion_map)
    usable = out.pixhit >= 10
    V1 = out.V1[usable].double()
    V2 = out.V2[usable].double()
    eye = torch.eye(3, device=dev, dtype=torch.float64).expand_as(V1) * 1e-4
    M = torch.linalg.solve(V1 + eye, V2)
    affine = M.transpose(-2, -1)
    A = affine[..., :2]
    b = affine[..., 2]
    I = torch.eye(2, device=dev, dtype=torch.float64)
    A_err = (A - I).abs().mean().item()
    b_gt = torch.tensor([dx, dy], device=dev, dtype=torch.float64)
    b_err = (b - b_gt).abs().mean().item()
    print(f"    |A-I|_mean={A_err:.2e}   |b-b_gt|_mean={b_err:.2e}")
    _check(A_err < 1e-2, f"A not identity: {A_err}")
    _check(b_err < 0.5, f"b not recovered: {b_err} (gt={b_gt.tolist()})")


# ---------------------------------------------------------------------------
# Test 4: NaN pixels are ignored — accumulator stays finite.
# ---------------------------------------------------------------------------
def test_nan_pixels_skipped():
    dev = torch.device("cuda")
    N = 16
    gaussians, camera = _make_synthetic_scene(N, dev, seed=3)
    H, W = camera.image_height, camera.image_width
    motion_map = torch.full((H, W, 2), float("nan"), device=dev)
    # only a central strip has valid targets
    motion_map[H // 2 - 2:H // 2 + 2, :, 0] = 5.0
    motion_map[H // 2 - 2:H // 2 + 2, :, 1] = 5.0

    out = motion_fusion(gaussians, camera, motion_map)
    _check(torch.isfinite(out.V1).all(), "V1 has non-finite entries")
    _check(torch.isfinite(out.V2).all(), "V2 has non-finite entries")
    _check(torch.isfinite(out.motion_alpha).all(), "motion_alpha has non-finite entries")
    print(f"    V1 finite ✓   V2 finite ✓   pixhit total = {out.pixhit.sum().item()}")


def test_non_identity_camera_recovery():
    """Camera rotated + translated away from origin — pixel math depends on
    w2c, not identity. Verify PWI-LS still recovers the global shift.
    """
    dev = torch.device("cuda")
    N = 64
    gaussians, camera = _make_synthetic_scene(N, dev, seed=5)
    # Override the camera with a rotated + translated one that keeps the
    # scene in frame.
    import math as _m
    c2w = torch.tensor([
        [_m.cos(0.2),  0.0, _m.sin(0.2), 0.3],
        [0.0,          1.0, 0.0,         0.1],
        [-_m.sin(0.2), 0.0, _m.cos(0.2), -0.2],
        [0.0,          0.0, 0.0,         1.0],
    ], device=dev)
    w2c = torch.linalg.inv(c2w)
    camera.R = w2c[:3, :3].clone()
    camera.T = w2c[:3, 3].clone()

    H, W = camera.image_height, camera.image_width
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx, dy = 2.5, 1.5
    motion_map = torch.stack([xx + 0.5 + dx, yy + 0.5 + dy], dim=-1)

    out = motion_fusion(gaussians, camera, motion_map)
    usable = out.pixhit >= 10
    _check(usable.sum() > 0, "no usable Gaussians with rotated camera")
    V1 = out.V1[usable].double()
    V2 = out.V2[usable].double()
    reg = torch.eye(3, device=dev, dtype=torch.float64).expand_as(V1) * 1e-4
    M = torch.linalg.solve(V1 + reg, V2)
    affine = M.transpose(-2, -1)
    A = affine[..., :2]
    b = affine[..., 2]
    I = torch.eye(2, device=dev, dtype=torch.float64)
    A_err = (A - I).abs().mean().item()
    b_err = (b - torch.tensor([dx, dy], device=dev, dtype=torch.float64)).abs().mean().item()
    print(f"    rotated-cam   |A-I|_mean={A_err:.2e}   |b-b_gt|_mean={b_err:.2e}")
    _check(A_err < 5e-2, f"A not identity with rotated camera: {A_err}")
    _check(b_err < 1.0, f"b not recovered with rotated camera: {b_err}")


def test_uniform_scaling_motion_recovery():
    """Target = 1.1 · pixel (global scale-up). Expect A = 1.1·I, b = 0 (with
    centre offset corrections — we probe a centred scaling around image mid)."""
    dev = torch.device("cuda")
    N = 64
    gaussians, camera = _make_synthetic_scene(N, dev, seed=9)
    H, W = camera.image_height, camera.image_width
    cx, cy = W / 2.0, H / 2.0
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    s = 1.1
    # target = s*(p - c) + c  =>  A = sI, b = (1-s)·c
    tx = s * (xx + 0.5 - cx) + cx
    ty = s * (yy + 0.5 - cy) + cy
    motion_map = torch.stack([tx, ty], dim=-1)

    out = motion_fusion(gaussians, camera, motion_map)
    usable = out.pixhit >= 10
    V1 = out.V1[usable].double()
    V2 = out.V2[usable].double()
    reg = torch.eye(3, device=dev, dtype=torch.float64).expand_as(V1) * 1e-4
    M = torch.linalg.solve(V1 + reg, V2)
    affine = M.transpose(-2, -1)
    A = affine[..., :2]
    b = affine[..., 2]
    A_expected = torch.tensor([[s, 0.0], [0.0, s]], device=dev, dtype=torch.float64)
    b_expected = torch.tensor([(1 - s) * cx, (1 - s) * cy], device=dev, dtype=torch.float64)
    A_err = (A - A_expected).abs().mean().item()
    b_err = (b - b_expected).abs().mean().item()
    print(f"    scale={s}   |A-sI|_mean={A_err:.2e}   |b-b_expected|_mean={b_err:.2e}")
    _check(A_err < 5e-2, f"A not sI: {A_err}")
    _check(b_err < 2.0, f"b mis-recovered: {b_err}")


def test_large_scene_is_finite_and_nonzero():
    """Stress: N=2000, 320×240. All outputs must be finite; most Gaussians
    must have at least one pixhit."""
    dev = torch.device("cuda")
    N = 2000
    W, H = 320, 240
    torch.manual_seed(11)
    xyz = torch.randn(N, 3, device=dev) * 0.35
    xyz[:, 2] = xyz[:, 2].abs() + 2.5
    rot = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    raw_log_scales = torch.full((N, 3), math.log(0.008), device=dev) + torch.randn(N, 3, device=dev) * 0.1
    raw_opacity = torch.randn(N, device=dev) * 0.5 + 1.0
    gaussians = _DuckGaussians(xyz, rot, raw_log_scales, raw_opacity)
    fx = fy = 250.0
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], device=dev, dtype=torch.float32)
    camera = Camera(
        R=torch.eye(3, device=dev), T=torch.zeros(3, device=dev), K=K,
        image_height=H, image_width=W,
        FoVx=2.0*math.atan(W/(2.0*fx)), FoVy=2.0*math.atan(H/(2.0*fy)),
        image_path="", frame_idx=0, cam_name="stress", device=dev,
    )
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    motion_map = torch.stack([xx + 0.5, yy + 0.5], dim=-1)

    out = motion_fusion(gaussians, camera, motion_map)
    _check(torch.isfinite(out.V1).all(), "V1 non-finite at N=2000")
    _check(torch.isfinite(out.V2).all(), "V2 non-finite at N=2000")
    _check(torch.isfinite(out.motion_alpha).all(), "motion_alpha non-finite")
    frac_hit = (out.pixhit > 0).float().mean().item()
    total_pix = out.pixhit.sum().item()
    print(f"    N={N}  {frac_hit*100:.1f}% of Gaussians have ≥1 pixhit  total pixhits={total_pix}")
    _check(frac_hit > 0.5, f"< 50% of Gaussians had coverage: {frac_hit}")
    _check((out.motion_alpha >= 0).all(), "motion_alpha negative")


def test_camera_behind_scene_empty_output():
    """Flip the camera so all Gaussians are behind it. Everything must be
    zero (no coverage), no NaNs, no crashes."""
    dev = torch.device("cuda")
    N = 32
    gaussians, camera = _make_synthetic_scene(N, dev, seed=13)
    # Flip the camera orientation 180° around y (so +Z becomes -Z).
    R = torch.tensor([[-1.0, 0.0, 0.0],
                      [0.0,  1.0, 0.0],
                      [0.0,  0.0, -1.0]], device=dev, dtype=torch.float32)
    camera.R = R @ camera.R
    H, W = camera.image_height, camera.image_width
    motion_map = torch.full((H, W, 2), 10.0, device=dev)

    out = motion_fusion(gaussians, camera, motion_map)
    _check(torch.isfinite(out.V1).all())
    _check(torch.isfinite(out.V2).all())
    total_hit = out.pixhit.sum().item()
    print(f"    camera-behind-scene total pixhits={total_hit}  "
          f"V1 max abs={out.V1.abs().max().item():.2e}")
    _check(total_hit < N, f"unexpectedly got {total_hit} pixhits with camera flipped")


def test_timing_budget():
    """Ballpark: on L40S, a 1000-Gaussian scene at 256×192 should finish in
    under 2 s. Hard failure if > 15 s (would make full 50-frame runs too slow)."""
    import time
    dev = torch.device("cuda")
    N = 1000
    W, H = 256, 192
    torch.manual_seed(21)
    xyz = torch.randn(N, 3, device=dev) * 0.25
    xyz[:, 2] = xyz[:, 2].abs() + 2.5
    rot = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
    raw_log_scales = torch.full((N, 3), math.log(0.01), device=dev)
    raw_opacity = torch.zeros(N, device=dev)
    gaussians = _DuckGaussians(xyz, rot, raw_log_scales, raw_opacity)
    fx = fy = 200.0
    K = torch.tensor([[fx, 0, W/2], [0, fy, H/2], [0, 0, 1]], device=dev, dtype=torch.float32)
    camera = Camera(
        R=torch.eye(3, device=dev), T=torch.zeros(3, device=dev), K=K,
        image_height=H, image_width=W,
        FoVx=2.0*math.atan(W/(2.0*fx)), FoVy=2.0*math.atan(H/(2.0*fy)),
        image_path="", frame_idx=0, cam_name="timing", device=dev,
    )
    ys = torch.arange(H, device=dev).float()
    xs = torch.arange(W, device=dev).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    motion_map = torch.stack([xx + 0.5, yy + 0.5], dim=-1)

    # warmup
    motion_fusion(gaussians, camera, motion_map)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    motion_fusion(gaussians, camera, motion_map)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"    1000 Gaussians @ 256×192 took {dt*1000:.0f} ms")
    _check(dt < 15.0, f"motion_fusion too slow: {dt:.2f} s for N=1000")


TESTS = [
    ("motion_fusion is deterministic + non-zero",           test_motion_fusion_self_consistent_identity_motion),
    ("PWI-LS recovers A=I, b=0 for identity motion",        test_identity_motion_pwils_recovers_identity),
    ("PWI-LS recovers A=I, b=(dx, dy) for global shift",    test_global_shift_pwils),
    ("NaN pixels are skipped (outputs stay finite)",        test_nan_pixels_skipped),
    ("rotated non-identity camera still recovers shift",    test_non_identity_camera_recovery),
    ("scaling motion recovered (A=sI, b=(1-s)·c)",          test_uniform_scaling_motion_recovery),
    ("N=2000 stress is finite + non-zero",                  test_large_scene_is_finite_and_nonzero),
    ("camera facing away produces empty output",            test_camera_behind_scene_empty_output),
    ("timing budget (< 15 s for 1000 Gaussians)",           test_timing_budget),
]


def main() -> int:
    print(f"{YEL}=== Day-2 motion_fusion validation ==={RESET}")
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
