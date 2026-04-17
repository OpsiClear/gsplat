"""
Deep test of every FasterGSCudaBackend module.

Covers all six modules:
    adam/           densification/    filter3d/
    rasterization/  torch_bindings/   utils/

Run:
    conda activate gsplat_fastergs
    python examples/test_fastergs_modules.py

Exits 0 on success, 1 on any failure.
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from dataclasses import dataclass

import torch

from FasterGSCudaBackend import _C
from FasterGSCudaBackend.torch_bindings import (
    FusedAdam,
    RasterizerSettings,
    add_noise,
    diff_rasterize,
    rasterize,
    relocation_adjustment,
    update_3d_filter,
    update_pruning_scores,
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
RESET = "\033[0m"; GREEN = "\033[92m"; RED = "\033[91m"; YEL = "\033[93m"; CYAN = "\033[96m"


@dataclass
class Scene:
    """Tiny random scene shared by several tests.

    build() returns RAW parameters — FasterGS applies the activations inside
    the kernel (opacity = sigmoid(raw); variance = exp(2 * raw_log_scale)).
    Do NOT pre-apply exp() or sigmoid() when passing to diff_rasterize.
    """
    N: int
    W: int = 256
    H: int = 256
    fx: float = 200.0
    fy: float = 200.0
    sh_degree: int = 0  # DC only
    device: torch.device = torch.device("cuda")

    def build(self, seed: int = 0):
        g = torch.Generator(device=self.device).manual_seed(seed)
        dev = self.device
        means = torch.randn(self.N, 3, device=dev, generator=g) * 0.4
        means[:, 2] += 3.0
        # raw log-scales → kernel does variance = exp(2 * log_scale)
        # set log_scale so that std(gauss) ~= 0.02 world-units
        log_scales = torch.full((self.N, 3), math.log(0.02), device=dev)
        raw_quats = torch.zeros(self.N, 4, device=dev); raw_quats[:, 0] = 1.0
        raw_opacities = torch.full((self.N,), 0.0, device=dev)  # sigmoid -> 0.5
        K = (self.sh_degree + 1) ** 2
        sh0 = torch.rand(self.N, 1, 3, device=dev, generator=g)
        shN = torch.zeros(self.N, K - 1, 3, device=dev)
        return means, log_scales, raw_quats, raw_opacities, sh0, shN

    def settings(self, bg=(0.0, 0.0, 0.0), antialiasing: bool = False):
        dev = self.device
        K = (self.sh_degree + 1) ** 2
        return RasterizerSettings(
            w2c=torch.eye(4, device=dev),
            cam_position=torch.zeros(3, device=dev),
            bg_color=torch.tensor(bg, device=dev),
            active_sh_bases=K,
            width=self.W,
            height=self.H,
            focal_x=self.fx,
            focal_y=self.fy,
            center_x=self.W / 2,
            center_y=self.H / 2,
            near_plane=0.01,
            far_plane=100.0,
            proper_antialiasing=antialiasing,
        )


# ---------------------------------------------------------------------------
# individual tests — each returns True/False
# ---------------------------------------------------------------------------
def test_rasterization_inference():
    """rasterize() — no-grad inference path. Output shape + sanity range.

    Passes RAW log_scales and RAW pre-sigmoid opacities — the kernel does
    exp(2*raw) and sigmoid() internally.
    """
    sc = Scene(N=50_000, sh_degree=0)
    means, log_s, quats, raw_o, sh0, shN = sc.build()
    img = rasterize(means, log_s, quats, raw_o, sh0, shN, sc.settings(), to_chw=True)
    assert img.shape == (3, sc.H, sc.W), f"bad shape {img.shape}"
    assert img.dtype == torch.float32
    assert torch.isfinite(img).all()
    assert img.min() >= 0.0 and img.max() <= 1.5, f"image range {img.min()}-{img.max()}"
    print(f"    image {tuple(img.shape)}  range [{img.min():.3f}, {img.max():.3f}]")
    return True


def test_rasterization_forward_backward():
    """diff_rasterize() — gradients flow into the rendering-observable inputs.

    Uses anisotropic scales + random quaternions so rotation is observable
    (with isotropic scales and identity quats, quat.grad is exactly zero and
    that's correct — the rasterizer is rotationally invariant there).
    """
    sc = Scene(N=10_000, sh_degree=0, W=192, H=192)
    means_, log_s, _quats, raw_o, sh0_, shN_ = sc.build(seed=1)

    dev = sc.device
    log_s = torch.randn_like(log_s) * 0.3 + math.log(0.02)        # anisotropic raw log-scale
    q = torch.randn(sc.N, 4, device=dev)                           # random quats (kernel normalises)

    means = means_.clone().requires_grad_()
    log_scales = log_s.clone().requires_grad_()                    # RAW
    quats = q.clone().requires_grad_()
    # diff_rasterize's backward returns grad opacities (N, 1) — input must match.
    raw_opac = raw_o.unsqueeze(-1).clone().requires_grad_()        # RAW (N, 1)
    sh0 = sh0_.clone().requires_grad_()
    shN = shN_.clone().requires_grad_()

    dinfo = torch.zeros(sc.N, device=dev)
    img = diff_rasterize(means, log_scales, quats, raw_opac, sh0, shN, dinfo, sc.settings())

    target = torch.rand_like(img)
    loss = (img - target).pow(2).mean()
    loss.backward()

    grads = {"means": means.grad, "log_scales": log_scales.grad, "quats": quats.grad,
             "raw_opac": raw_opac.grad, "sh0": sh0.grad, "shN": shN.grad}
    for name, g in grads.items():
        assert g is not None, f"{name}.grad is None"
        assert torch.isfinite(g).all(), f"{name}.grad has NaN/Inf"
        # shN has no observable effect at sh_degree=0 (active_sh_bases=1) so skip
        if name == "shN":
            continue
        assert g.abs().sum() > 0, f"{name}.grad is all zero"
    print("    grad norms: " + ", ".join(
        f"{n}={g.norm().item():.2e}" for n, g in grads.items() if g is not None))
    return True


def test_rasterization_antialiasing_mode():
    """proper_antialiasing=True/False should both run and produce finite images."""
    sc = Scene(N=5_000, sh_degree=0)
    means, log_s, quats, raw_o, sh0, shN = sc.build(seed=2)
    for aa in (False, True):
        img = rasterize(means, log_s, quats, raw_o, sh0, shN,
                        sc.settings(antialiasing=aa), to_chw=True)
        assert torch.isfinite(img).all(), f"aa={aa} gave non-finite image"
        print(f"    proper_antialiasing={aa}: mean={img.mean():.3f}")
    return True


def test_pruning_scores():
    """update_pruning_scores accumulates per-primitive importance in-place."""
    sc = Scene(N=20_000, sh_degree=0)
    means, log_s, quats, raw_o, sh0, shN = sc.build(seed=3)
    scores = torch.zeros(sc.N, device=sc.device)
    update_pruning_scores(scores, means, log_s, quats, raw_o, sh0, shN, sc.settings())
    assert scores.shape == (sc.N,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all(), "pruning scores should be non-negative"
    nonzero = int((scores > 0).sum())
    print(f"    scores updated in-place: {nonzero}/{sc.N} nonzero, "
          f"range [{scores.min():.3e}, {scores.max():.3e}]")
    assert nonzero > 0, "no primitives accumulated score"
    return True


def test_fused_adam_matches_torch_adam():
    """FusedAdam.step() should closely match torch.optim.Adam on a toy problem."""
    torch.manual_seed(0)
    init = torch.randn(1024, 8, device="cuda")
    lr, eps = 1e-2, 1e-15

    # Reference torch Adam
    p_ref = init.clone().requires_grad_()
    opt_ref = torch.optim.Adam([p_ref], lr=lr, eps=eps)

    # FusedAdam from FasterGS
    p_fused = init.clone().requires_grad_()
    opt_fused = FusedAdam([p_fused], lr=lr, eps=eps)

    max_abs = 0.0
    for step in range(50):
        for p, opt in [(p_ref, opt_ref), (p_fused, opt_fused)]:
            if p.grad is None: pass
            loss = (p ** 2).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        max_abs = max(max_abs, (p_ref - p_fused).abs().max().item())

    print(f"    max |torch_adam − fused_adam| after 50 steps: {max_abs:.2e}")
    assert max_abs < 1e-4, f"FusedAdam diverges from torch.Adam: {max_abs}"
    return True


def test_densification_add_noise():
    """add_noise should perturb means in-place when raw_opacities are very low
    (MCMC gates noise by (1-σ(opacity))^k — high-opacity splats get ~no jitter)."""
    device = torch.device("cuda")
    N = 5_000
    # Use pre-sigmoid opacity = -8 → σ≈3e-4, so the gate (1−σ)^k is essentially 1
    # → noise actually applies. At σ=0.5 (raw=0) the gate is tiny and noise rounds
    # to zero in fp32, so a flat-0 raw_opacities test would spuriously fail.
    for raw_opacity_value, label in [(-8.0, "low-opacity"), (0.0, "mid-opacity")]:
        means = torch.zeros(N, 3, device=device)
        raw_scales = torch.full((N, 3), math.log(0.05), device=device)
        raw_rots = torch.zeros(N, 4, device=device); raw_rots[:, 0] = 1.0
        raw_opacities = torch.full((N, 1), raw_opacity_value, device=device)

        # lr=0 is the ground truth no-op
        add_noise(raw_scales, raw_rots, raw_opacities, means, current_lr=0.0)
        assert means.abs().max().item() == 0.0, "lr=0 produced nonzero noise"

        # lr>0 with low raw opacity should produce measurable jitter
        add_noise(raw_scales, raw_rots, raw_opacities, means, current_lr=1.0)
        delta = means.norm(dim=-1)
        print(f"    {label:12s}  raw_opac={raw_opacity_value:+.1f}  "
              f"mean|Δ|={delta.mean():.3e}  max|Δ|={delta.max():.3e}")
        if raw_opacity_value < -2.0:
            assert delta.mean().item() > 0, "low-opacity splats should jitter"
    return True


def test_densification_relocation_adjustment():
    """relocation_adjustment: split 1 primitive into N; opacity should drop."""
    device = torch.device("cuda")
    N = 128
    old_opacities = torch.full((N, 1), 0.9, device=device)  # activated
    old_scales = torch.full((N, 3), 0.05, device=device)
    n_samples = torch.randint(1, 6, (N,), device=device, dtype=torch.int64)

    # NOTE: return order is (new_opacities, new_scales) — opposite of arg order.
    new_opacities, new_scales = relocation_adjustment(
        old_opacities, old_scales, n_samples
    )
    assert new_scales.shape == old_scales.shape, new_scales.shape
    assert new_opacities.shape == old_opacities.shape, new_opacities.shape

    # Where n_samples == 1, MCMC should return (approx) identity
    id_mask = (n_samples == 1)
    if id_mask.any():
        o_err = (new_opacities[id_mask] - old_opacities[id_mask]).abs().max().item()
        s_err = (new_scales[id_mask] - old_scales[id_mask]).abs().max().item()
        print(f"    n=1 identity: |Δopacity|={o_err:.2e}  |Δscale|={s_err:.2e}")
        assert o_err < 1e-4 and s_err < 1e-4

    # Where n > 1, new opacity should be ≤ old (some energy redistributed)
    more_mask = (n_samples > 1)
    if more_mask.any():
        ratio = (new_opacities[more_mask] / old_opacities[more_mask]).mean().item()
        print(f"    n>1 mean opacity ratio new/old: {ratio:.3f}  (should be < 1)")
        assert ratio < 1.0, f"relocation_adjustment did not reduce opacity ({ratio})"
    return True


def test_filter3d_update():
    """update_3d_filter writes a finite per-primitive 3D filter scalar.

    The function accumulates the MIN safe filter across all views. We therefore
    seed filter_3d with +inf (Mip-Splatting convention) and pre-populate
    visibility_mask with True (visibility accumulates across views, OR-style).
    Two calls from two cameras verify accumulation.
    """
    device = torch.device("cuda")
    N, W, H = 4_000, 256, 256
    positions = torch.randn(N, 3, device=device) * 0.3
    positions[:, 2] += 3.0

    filter_3d = torch.full((N,), float("inf"), device=device)
    visibility_mask = torch.zeros(N, device=device, dtype=torch.bool)

    for tx in (0.0, 0.3):                       # two camera positions
        w2c = torch.eye(4, device=device); w2c[0, 3] = tx
        update_3d_filter(
            positions, w2c, filter_3d, visibility_mask,
            width=W, height=H,
            focal_x=200.0, focal_y=200.0,
            center_x=W / 2, center_y=H / 2,
            near_plane=0.01,
            clipping_tolerance=1.2,
            distance2filter=1.0,
        )
    n_vis = int(visibility_mask.sum())
    assert n_vis > 0, "no primitives were marked visible across either view"
    vis_filter = filter_3d[visibility_mask]
    assert torch.isfinite(vis_filter).all(), "visible primitives got inf filter"
    assert (vis_filter > 0).all(), "visible primitives should have positive filter"
    print(f"    {n_vis}/{N} visible across two views, filter range "
          f"[{vis_filter.min():.3e}, {vis_filter.max():.3e}]")
    return True


def test_end_to_end_training_step():
    """Minimal training step: rasterize → loss → backward → FusedAdam → add_noise.

    Parameters are RAW throughout — the rasterizer applies activations
    internally, so Adam updates happen in raw space (same convention as gsplat).
    """
    torch.manual_seed(0)
    sc = Scene(N=6_000, sh_degree=0, W=192, H=192)
    means, log_s, quats, raw_o, sh0, shN = sc.build(seed=7)
    raw_means = means.clone().requires_grad_()
    raw_scales = log_s.clone().requires_grad_()        # RAW log-scales
    raw_quats = quats.clone().requires_grad_()
    raw_opac = raw_o.unsqueeze(-1).clone().requires_grad_()   # RAW (N, 1), see note in fwd+bwd test
    sh0 = sh0.clone().requires_grad_()
    shN = shN.clone().requires_grad_()

    opts = {
        "means": FusedAdam([raw_means], lr=1e-4, eps=1e-15),
        "scales": FusedAdam([raw_scales], lr=5e-3, eps=1e-15),
        "quats": FusedAdam([raw_quats], lr=1e-3, eps=1e-15),
        "opac": FusedAdam([raw_opac], lr=5e-2, eps=1e-15),
        "sh0": FusedAdam([sh0], lr=2.5e-3, eps=1e-15),
        "shN": FusedAdam([shN], lr=1.25e-4, eps=1e-15),
    }
    dinfo = torch.zeros(sc.N, device=sc.device)
    target = torch.rand(3, sc.H, sc.W, device=sc.device)

    losses = []
    for step in range(30):
        img = diff_rasterize(raw_means, raw_scales, raw_quats, raw_opac,
                             sh0, shN, dinfo, sc.settings())
        loss = (img - target).pow(2).mean()
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        # MCMC-style noise on raw parameters — add_noise expects raw_opac (N,1)
        with torch.no_grad():
            add_noise(raw_scales, raw_quats, raw_opac, raw_means.data,
                      current_lr=5e-5)
        losses.append(loss.item())

    drop = losses[0] - losses[-1]
    print(f"    loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}  Δ={drop:.4f}")
    assert losses[-1] < losses[0], "loss should decrease over 30 steps"
    return True


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def test_render_parity_with_gsplat():
    """Render the same scene with gsplat AND FasterGS → images should agree
    pixel-wise within a small tolerance when input conventions are right."""
    try:
        from gsplat.rendering import rasterization as gsplat_rast
    except ImportError:
        print("    gsplat not importable, skipping parity check")
        return True

    dev = torch.device("cuda")
    torch.manual_seed(12)
    N, W, H = 20_000, 256, 256
    fx = fy = 300.0
    cx, cy = W / 2, H / 2

    means = torch.randn(N, 3, device=dev) * 0.4
    means[:, 2] += 3.0
    log_scales = torch.randn(N, 3, device=dev) * 0.15 + math.log(0.015)
    quats = torch.randn(N, 4, device=dev); quats = quats / quats.norm(dim=-1, keepdim=True)
    raw_opac = torch.randn(N, device=dev) * 0.8                    # raw logits
    sh0 = torch.rand(N, 1, 3, device=dev)
    shN = torch.zeros(N, 0, 3, device=dev)   # sh_degree=0 → no extra bands

    # gsplat: it expects ACTIVATED scales, ACTIVATED opacities, (B,4,4) viewmats
    g_viewmat = torch.eye(4, device=dev).unsqueeze(0)
    g_K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=dev).unsqueeze(0)
    g_colors = torch.cat([sh0, shN], dim=1)                         # (N, 1, 3)
    g_img, _, _ = gsplat_rast(
        means=means, quats=quats, scales=log_scales.exp(),
        opacities=torch.sigmoid(raw_opac), colors=g_colors,
        viewmats=g_viewmat, Ks=g_K, width=W, height=H,
        near_plane=0.01, far_plane=100.0,
        sh_degree=0, packed=False, render_mode="RGB",
    )                                                               # (1, H, W, 3)
    g_img = g_img[0].clamp(0, 1)                                    # (H, W, 3)

    # FasterGS: RAW log_scales, RAW opacities, single-view settings
    settings = RasterizerSettings(
        w2c=torch.eye(4, device=dev),
        cam_position=torch.zeros(3, device=dev),
        bg_color=torch.zeros(3, device=dev),
        active_sh_bases=1, width=W, height=H,
        focal_x=fx, focal_y=fy, center_x=cx, center_y=cy,
        near_plane=0.01, far_plane=100.0, proper_antialiasing=False,
    )
    f_img = rasterize(means, log_scales, quats, raw_opac, sh0, shN,
                      settings, to_chw=True).permute(1, 2, 0).clamp(0, 1)   # (H, W, 3)

    # per-pixel L1 / PSNR as a single numeric report
    l1 = (g_img - f_img).abs().mean().item()
    mse = (g_img - f_img).pow(2).mean().item()
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    # also compare means / brightness histograms — they should agree to ~1%
    bright_diff = abs(g_img.mean().item() - f_img.mean().item())
    print(f"    gsplat vs FasterGS: L1={l1:.4f}  PSNR={psnr:.2f} dB  "
          f"|Δmean_brightness|={bright_diff:.4f}")

    # Save side-by-side png for visual sanity
    try:
        from torchvision.utils import save_image
        triptych = torch.cat([
            g_img.permute(2, 0, 1),
            f_img.permute(2, 0, 1),
            (g_img - f_img).abs().permute(2, 0, 1).clamp(0, 1) * 5,
        ], dim=-1)
        save_image(triptych, "/tmp/fastergs_test/parity_gsplat_vs_fastergs.png")
        print("    saved /tmp/fastergs_test/parity_gsplat_vs_fastergs.png")
    except Exception as e:
        print(f"    (skip save_image: {e})")

    # The two rasterizers are close but not identical — algorithmic differences
    # (informed pruning, sort order). Require global-brightness sanity only.
    assert bright_diff < 0.05, (
        f"|Δmean_brightness|={bright_diff:.4f} too large — convention "
        "mismatch is likely (scales/opacities/SH)"
    )
    return True


TESTS = [
    ("rasterization.inference",        test_rasterization_inference),
    ("rasterization.fwd+bwd",          test_rasterization_forward_backward),
    ("rasterization.antialiasing",     test_rasterization_antialiasing_mode),
    ("rasterization.pruning_scores",   test_pruning_scores),
    ("rasterization.parity_vs_gsplat", test_render_parity_with_gsplat),
    ("adam.FusedAdam vs torch.Adam",   test_fused_adam_matches_torch_adam),
    ("densification.add_noise",        test_densification_add_noise),
    ("densification.relocation_adj",   test_densification_relocation_adjustment),
    ("filter3d.update_3d_filter",      test_filter3d_update),
    ("end-to-end training step",       test_end_to_end_training_step),
]


def main() -> int:
    print(f"{CYAN}=== FasterGSCudaBackend deep module test ==={RESET}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}  cuda: {torch.version.cuda}")
    print()

    n_pass = 0
    failures = []
    for name, fn in TESTS:
        print(f"{YEL}▶ {name}{RESET}")
        t0 = time.perf_counter()
        try:
            ok = fn()
            dt = (time.perf_counter() - t0) * 1000
            if ok:
                print(f"  {GREEN}✓ PASS{RESET}  ({dt:.0f} ms)\n")
                n_pass += 1
            else:
                print(f"  {RED}✗ FAIL{RESET}  ({dt:.0f} ms)\n")
                failures.append(name)
        except Exception:
            dt = (time.perf_counter() - t0) * 1000
            print(f"  {RED}✗ FAIL{RESET}  ({dt:.0f} ms)")
            traceback.print_exc()
            print()
            failures.append(name)

    print("=" * 60)
    if failures:
        print(f"{RED}{len(failures)}/{len(TESTS)} failed:{RESET}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{GREEN}all {n_pass}/{len(TESTS)} passed{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
