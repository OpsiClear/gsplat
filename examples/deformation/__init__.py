"""Deformation modules for 4D Gaussian Splatting."""

from .hexplane import HexPlane
from .deform_network import DeformationField, DeformOutput, apply_deformation
from .regulation import plane_tv_loss, time_smoothness_loss, l1_time_planes_loss

__all__ = [
    "HexPlane",
    "DeformationField",
    "DeformOutput",
    "apply_deformation",
    "plane_tv_loss",
    "time_smoothness_loss",
    "l1_time_planes_loss",
]
