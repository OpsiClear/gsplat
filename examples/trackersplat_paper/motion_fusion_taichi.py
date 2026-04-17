"""
motion_fusion — per-Gaussian linear-system accumulator.

Replaces the reference's Inria-derived CUDA pass at
`trackersplat/utils/motionfusion/__init__.py::motion_fusion`. Same contract,
same math, implemented in Apache-2.0 Python + Taichi.

## Contract (identical to the reference outputs)

Inputs (per view):
  gaussians    — duck-typed container with _xyz, _rotation, _scaling, _opacity
  camera       — examples.trackersplat_dataset.Camera
  motion_map   — (H, W, 2) float per-pixel tracked target positions, in pixel
                 coordinates (px', py') at the render resolution. If a pixel
                 has no track, use (NaN, NaN) — masked out inside the kernel.

Outputs (per Gaussian):
  V1           — (N, 3, 3) symmetric:  Σ_p  w(p,g) · [px, py, 1]·[px, py, 1]ᵀ
  V2           — (N, 3, 2)           :  Σ_p  w(p,g) · [px, py, 1]·[px', py']ᵀ
  motion_alpha — (N,)    :  Σ_p  w(p,g)       (accumulated alpha·transmittance)
  motion_det   — (N,)    :  det(2D cov_g)     (per-Gaussian stability gate)
  pixhit       — (N,)    :  count of pixels where Gaussian g was the front
                 contributor (before early-exit)

Where w(p, g) = α(p, g) · T(p, g) is the alpha-blend weight that the rasterizer
produces internally. For paper parity we include transmittance T; the blend
walks Gaussians depth-sorted front-to-back at each pixel.

## Status: STUB — Day 1 deliverable

This file establishes the public API so `pipeline.py` imports cleanly. The
Taichi kernel implementation lands in Day 2 (per the plan in
`examples/trackersplat_paper_plan.md`).

## Debug / test plan

See `tests/test_motion_fusion.py` in this folder for validation targets:
  1. identity-motion test         — motion_map = identity → A2D=I, b2D=0
  2. global-translation test      — motion_map = (+dx, +dy) → A2D=I, b2D=(dx, dy)
  3. single-Gaussian against
     pure-PyTorch reference       — V1, V2 within 1e-3 relative error
  4. rotationally symmetric scene — 8-NN median/propagation leaves untouched
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class MotionFusionOutput:
    """Packed per-Gaussian linear-system accumulators produced by one view."""
    V1: Tensor            # (N, 3, 3)
    V2: Tensor            # (N, 3, 2)
    motion_alpha: Tensor  # (N,)
    motion_det: Tensor    # (N,)
    pixhit: Tensor        # (N,)  int


def motion_fusion(
    gaussians,
    camera,
    motion_map: Tensor,
    alpha_threshold: float = 1e-3,
    k_sigma: float = 3.0,
) -> MotionFusionOutput:
    """Accumulate per-Gaussian V1/V2/α_acc/det/pixhit from one view.

    Paper: §4.3 Parallel Weighted Incremental Least Squares.

    STUB (Day 1). Implemented Day 2 as a Taichi kernel; until then this
    raises NotImplementedError to force explicit wiring verification.
    """
    raise NotImplementedError(
        "motion_fusion Taichi kernel lands in Day 2. "
        "See trackersplat_paper/motion_fusion_taichi.py for the contract."
    )
