"""
Regularization losses for 4DGS HexPlane deformation field.

Three losses to enforce spatial and temporal smoothness:
  1. PlaneTV       — total variation on spatial planes (XY, XZ, YZ)
  2. TimeSmoothness — second-order temporal derivative on temporal planes (XT, YT, ZT)
  3. L1TimePlanes  — L1 penalty to keep temporal plane values near zero

Reference: https://arxiv.org/abs/2310.08528, regulation.py in hustvl/4DGaussians
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from .hexplane import HexPlane


def plane_tv_loss(hexplane: HexPlane) -> Tensor:
    """
    Total variation loss on all spatial planes (XY, XZ, YZ).

    Penalizes abrupt spatial discontinuities in the feature grid.
    Operates on each plane independently, summing TV across all scales.

    Args:
        hexplane: HexPlane module.

    Returns:
        Scalar TV loss.
    """
    loss = torch.tensor(0.0, device=next(hexplane.parameters()).device)
    for plane in hexplane.get_spatial_planes():
        # plane: [1, C, H, W]
        # TV = sum of squared differences between adjacent pixels
        diff_h = plane[:, :, 1:, :] - plane[:, :, :-1, :]   # [1, C, H-1, W]
        diff_w = plane[:, :, :, 1:] - plane[:, :, :, :-1]   # [1, C, H, W-1]
        loss = loss + diff_h.pow(2).mean() + diff_w.pow(2).mean()
    return loss


def time_smoothness_loss(hexplane: HexPlane) -> Tensor:
    """
    Temporal smoothness loss on temporal planes (XT, YT, ZT).

    Penalizes second-order temporal derivatives (acceleration) of the feature grid,
    encouraging smooth deformation trajectories over time.

    For a temporal plane of shape [1, C, spatial, time], applies a second-order
    finite difference filter along the time dimension.

    Args:
        hexplane: HexPlane module.

    Returns:
        Scalar temporal smoothness loss.
    """
    loss = torch.tensor(0.0, device=next(hexplane.parameters()).device)
    for plane in hexplane.get_temporal_planes():
        # plane: [1, C, H, W]
        # For temporal planes, the time axis corresponds to W (the second axis
        # of the plane pair, since temporal pairs are (x,t), (y,t), (z,t)).
        # We apply second-order finite difference along W (time).
        if plane.shape[-1] < 3:
            continue
        # First-order differences along time (W axis)
        diff1 = plane[:, :, :, 1:] - plane[:, :, :, :-1]   # [1, C, H, W-1]
        # Second-order differences
        diff2 = diff1[:, :, :, 1:] - diff1[:, :, :, :-1]   # [1, C, H, W-2]
        loss = loss + diff2.pow(2).mean()
    return loss


def l1_time_planes_loss(hexplane: HexPlane) -> Tensor:
    """
    L1 penalty on temporal plane values to encourage sparsity near zero.

    This prevents the deformation field from drifting far from the identity
    in temporal directions when no deformation is needed.

    Args:
        hexplane: HexPlane module.

    Returns:
        Scalar L1 loss.
    """
    loss = torch.tensor(0.0, device=next(hexplane.parameters()).device)
    for plane in hexplane.get_temporal_planes():
        loss = loss + plane.abs().mean()
    return loss
