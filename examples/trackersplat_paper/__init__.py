"""
TrackerSplat (paper-faithful) — Apache-2.0 port of Yin et al. 2025.

Runs on top of our FasterGS-based trainer for the final refinement stage and
Taichi for the GPU kernels that replace the reference's Inria-derived CUDA
paths. No Inria code, no diff-gaussian-rasterization fork.

Reference:      github.com/yindaheng98/TrackerSplat (Apache-2.0)
Our plan doc:   examples/trackersplat_paper_plan.md
License:        examples/trackersplat_paper/LICENSE.apache-2.0

Public API
----------
    Motion                        NamedTuple holding per-Gaussian Δμ/ΔR/ΔS + masks/confidences
    compensate(baseframe, motion) Apply a Motion to a Gaussian container
    compare(baseframe, curframe)  Diff two Gaussian containers → Motion
    motion_fusion(...)            Per-view per-Gaussian LS-system accumulator (Day 2)

    utils.ILS_RotationScale       PWI-LS solver for 3D cov (rotation + scale)
    utils.ISVD_Mean3D             Multi-view triangulation for 3D mean
    utils.motion_median_filter    K-NN median smoothing (Taichi)
    utils.propagate               8-NN motion propagation (Taichi)
    utils.solve_transform         Build (X, Y, A) from fitted 2D affine
"""

# utils imports trigger `ti.init()` (via medianfilter / propagation), which
# must happen BEFORE motion_fusion_taichi's @ti.kernel decorator evaluates.
from . import utils  # noqa: F401
from .motion import Motion, compensate, compare, quaternion_raw_multiply
from .motion_fusion_taichi import motion_fusion, MotionFusionOutput

__all__ = [
    "Motion", "compensate", "compare", "quaternion_raw_multiply",
    "motion_fusion", "MotionFusionOutput",
    "utils",
]
