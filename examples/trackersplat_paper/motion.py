"""
Motion NamedTuple and compensate() — adapted from
github.com/yindaheng98/TrackerSplat/blob/main/trackersplat/motion.py
(Apache-2.0).

Change from reference: we swap the `gaussian_splatting.GaussianModel`
dependency for a duck-typed protocol. Anything with the `._xyz`, `._rotation`,
`._scaling`, `._opacity`, `._features_dc`, `._features_rest` attributes can be
compensated — this includes our FasterGS-backed `GaussiansWrapper`.

Apache-2.0.
"""
from __future__ import annotations

import copy
from typing import Any, NamedTuple, Optional

import torch
import torch.nn as nn

from .utils.pose import quaternion_invert


# Hamilton quaternion multiplication, real part first. Copied from
# gaussian_splatting.utils (itself Apache-2.0) as a standalone helper so we
# don't depend on that package.
def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


class Motion(NamedTuple):
    """All masks are bool (N,). `confidence_*` is (K,) where K == mask.sum()
    and aligns with the non-None value tensors.

    Tensor fields all live at the "masked" count (K) rather than N, except
    `fixed_mask` / `motion_mask_*` which are full (N,). This matches the
    reference layout exactly.
    """
    fixed_mask: Optional[torch.Tensor] = None
    motion_mask_cov: Optional[torch.Tensor] = None
    motion_mask_mean: Optional[torch.Tensor] = None
    rotation_quaternion: Optional[torch.Tensor] = None
    scaling_modifier_log: Optional[torch.Tensor] = None
    translation_vector: Optional[torch.Tensor] = None
    confidence_fix: Optional[torch.Tensor] = None
    confidence_cov: Optional[torch.Tensor] = None
    confidence_mean: Optional[torch.Tensor] = None
    update_baseframe: bool = False

    opacity_modifier_log: Optional[torch.Tensor] = None
    features_dc_modifier: Optional[torch.Tensor] = None
    features_rest_modifier: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> "Motion":
        return self._replace(**{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self._asdict().items()
        })

    def validate(self):
        if self.fixed_mask is not None:
            assert self.fixed_mask.dtype == torch.bool and self.fixed_mask.dim() == 1
            if self.confidence_fix is not None:
                assert self.confidence_fix.dim() == 1 \
                       and self.confidence_fix.size(0) == int(self.fixed_mask.sum())
        else:
            assert self.confidence_fix is None

        if self.motion_mask_cov is not None:
            assert self.motion_mask_cov.dtype == torch.bool and self.motion_mask_cov.dim() == 1
            if self.confidence_cov is not None:
                assert self.confidence_cov.dim() == 1 \
                       and self.confidence_cov.size(0) == int(self.motion_mask_cov.sum())
            if self.rotation_quaternion is not None:
                assert self.rotation_quaternion.dim() == 2 \
                       and self.rotation_quaternion.size(0) == int(self.motion_mask_cov.sum()) \
                       and self.rotation_quaternion.size(1) == 4
            elif self.scaling_modifier_log is not None:
                assert self.scaling_modifier_log.dim() == 2 \
                       and self.scaling_modifier_log.size(0) == int(self.motion_mask_cov.sum()) \
                       and self.scaling_modifier_log.size(1) == 3
        else:
            assert self.confidence_cov is None

        if self.motion_mask_mean is not None:
            assert self.motion_mask_mean.dtype == torch.bool and self.motion_mask_mean.dim() == 1
            if self.confidence_mean is not None:
                assert self.confidence_mean.dim() == 1 \
                       and self.confidence_mean.size(0) == int(self.motion_mask_mean.sum())
            if self.translation_vector is not None:
                assert self.translation_vector.dim() == 2 \
                       and self.translation_vector.size(0) == int(self.motion_mask_mean.sum()) \
                       and self.translation_vector.size(1) == 3
        else:
            assert self.confidence_mean is None


# -----------------------------------------------------------------------------
# compensate() — duck-typed: takes anything with ._xyz / ._rotation / _scaling /
# _opacity / _features_dc / _features_rest (or a subset of those).
# -----------------------------------------------------------------------------
def transform_xyz(baseframe: Any, translation_vector: torch.Tensor,
                  motion_mask_mean: Optional[torch.Tensor] = None) -> torch.Tensor:
    with torch.no_grad():
        if motion_mask_mean is None:
            return baseframe._xyz + translation_vector
        xyz = baseframe._xyz.clone()
        xyz[motion_mask_mean] += translation_vector
        return xyz


def transform_rotation(baseframe: Any, rotation_quaternion: torch.Tensor,
                       motion_mask_cov: Optional[torch.Tensor] = None) -> torch.Tensor:
    with torch.no_grad():
        if motion_mask_cov is None:
            return quaternion_raw_multiply(rotation_quaternion, baseframe._rotation)
        rot = baseframe._rotation.clone()
        rot[motion_mask_cov] = quaternion_raw_multiply(
            rotation_quaternion, baseframe._rotation[motion_mask_cov]
        )
        return rot


def transform_scaling(baseframe: Any, scaling_modifier_log: torch.Tensor,
                      motion_mask_cov: Optional[torch.Tensor] = None) -> torch.Tensor:
    with torch.no_grad():
        if motion_mask_cov is None:
            return scaling_modifier_log + baseframe._scaling
        scaling = baseframe._scaling.clone()
        scaling[motion_mask_cov] = scaling_modifier_log + baseframe._scaling[motion_mask_cov]
        return scaling


def compensate(baseframe: Any, motion: Motion) -> Any:
    """Apply a Motion to a baseframe Gaussian container (anything duck-typed
    with the `_xyz`, `_rotation`, `_scaling` fields). Returns a deep-copy with
    the compensated parameters."""
    currframe = copy.deepcopy(baseframe)
    if motion.translation_vector is not None:
        currframe._xyz = nn.Parameter(transform_xyz(
            baseframe, motion.translation_vector, motion.motion_mask_mean))
    if motion.rotation_quaternion is not None:
        currframe._rotation = nn.Parameter(transform_rotation(
            baseframe, motion.rotation_quaternion, motion.motion_mask_cov))
    if motion.scaling_modifier_log is not None:
        currframe._scaling = nn.Parameter(transform_scaling(
            baseframe, motion.scaling_modifier_log, motion.motion_mask_cov))
    if motion.opacity_modifier_log is not None and hasattr(baseframe, "_opacity"):
        with torch.no_grad():
            currframe._opacity = nn.Parameter(motion.opacity_modifier_log + baseframe._opacity)
    if motion.features_dc_modifier is not None and hasattr(baseframe, "_features_dc"):
        with torch.no_grad():
            currframe._features_dc = nn.Parameter(motion.features_dc_modifier + baseframe._features_dc)
    if motion.features_rest_modifier is not None and hasattr(baseframe, "_features_rest"):
        with torch.no_grad():
            currframe._features_rest = nn.Parameter(motion.features_rest_modifier + baseframe._features_rest)
    return currframe


def compare(baseframe: Any, curframe: Any) -> Motion:
    """Convenience: diff two Gaussian containers into a Motion."""
    with torch.no_grad():
        kwargs = dict(
            translation_vector=curframe._xyz - baseframe._xyz,
            rotation_quaternion=torch.nn.functional.normalize(
                quaternion_raw_multiply(curframe._rotation, quaternion_invert(baseframe._rotation))
            ),
            scaling_modifier_log=curframe._scaling - baseframe._scaling,
        )
        if hasattr(baseframe, "_opacity") and hasattr(curframe, "_opacity"):
            kwargs["opacity_modifier_log"] = curframe._opacity - baseframe._opacity
        if hasattr(baseframe, "_features_dc") and hasattr(curframe, "_features_dc"):
            kwargs["features_dc_modifier"] = curframe._features_dc - baseframe._features_dc
        if hasattr(baseframe, "_features_rest") and hasattr(curframe, "_features_rest"):
            kwargs["features_rest_modifier"] = curframe._features_rest - baseframe._features_rest
        return Motion(**kwargs)
