"""Utilities vendored from github.com/yindaheng98/TrackerSplat (Apache-2.0)
plus our own math_utils replacement for the non-vendored Inria-derived helpers.
"""
from .pose import quaternion_invert, axis_angle_to_quaternion, quaternion_to_axis_angle
from .incremental_svd import SVD, IncrementalSVD, ISVD42, ISVD_Mean3D
from .incremental_ls import ILS, ILS_Cov3D, ILS_RotationScale
from .propagation import propagate
from .medianfilter import motion_median_filter
from .math_utils import (
    unflatten_symmetry_3x3, unflatten_symmetry_2x2,
    compute_Jacobian, compute_T, compute_cov3D_equations,
    compute_cov2D, transform_cov2D,
    compute_mean2D, compute_mean2D_equations,
    solve_transform,
)

__all__ = [
    "quaternion_invert", "axis_angle_to_quaternion", "quaternion_to_axis_angle",
    "SVD", "IncrementalSVD", "ISVD42", "ISVD_Mean3D",
    "ILS", "ILS_Cov3D", "ILS_RotationScale",
    "propagate", "motion_median_filter",
    "unflatten_symmetry_3x3", "unflatten_symmetry_2x2",
    "compute_Jacobian", "compute_T", "compute_cov3D_equations",
    "compute_cov2D", "transform_cov2D",
    "compute_mean2D", "compute_mean2D_equations",
    "solve_transform",
]
