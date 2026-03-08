"""
Deformation Field for 4D Gaussian Splatting.

Combines a HexPlane spatiotemporal feature grid with a small MLP decoder
to predict per-Gaussian deformation deltas at a given timestep.

Architecture:
  HexPlane(xyz, t) → features [N, F]
  → shared backbone Linear(F, W) + ReLU
  → parallel output heads for each deformation type

Reference: https://arxiv.org/abs/2310.08528
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .hexplane import HexPlane


@dataclass
class DeformOutput:
    """
    Output of the deformation field for one timestep.
    All tensors have shape [N, dim] and represent *additive* deltas
    applied to the corresponding Gaussian parameters.
    """
    delta_xyz: Tensor           # [N, 3] — position delta (world space)
    delta_rot: Tensor           # [N, 4] — quaternion delta (wxyz)
    delta_scale: Tensor         # [N, 3] — log-space scale delta
    delta_opacity: Optional[Tensor] = None   # [N, 1] — logit-space opacity delta
    delta_sh: Optional[Tensor] = None        # [N, 48] — SH coefficient delta


def _make_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP head with ReLU activations."""
    return nn.Sequential(
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DeformationField(nn.Module):
    """
    4DGS Deformation Field: HexPlane + MLP → per-Gaussian deltas.

    Args:
        grid_resolution: Spatial grid resolution for HexPlane.
        time_resolution: Temporal resolution for HexPlane.
        feature_dim: Feature channels per HexPlane plane.
        multires: Multi-resolution scale multipliers.
        net_width: Hidden dimension of shared MLP backbone.
        defor_depth: Number of hidden layers in backbone (usually 1).
        aabb: Scene bounding box [2, 3]. Used to normalize xyz coords.
        enable_opacity_deform: Add output head for opacity delta.
        enable_sh_deform: Add output head for SH color delta.
        sh_degree: SH degree — needed to compute delta_sh output size.
    """

    def __init__(
        self,
        grid_resolution: int = 64,
        time_resolution: int = 25,
        feature_dim: int = 16,
        multires: Optional[List[int]] = None,
        net_width: int = 64,
        defor_depth: int = 1,
        aabb: Optional[Tensor] = None,
        enable_opacity_deform: bool = False,
        enable_sh_deform: bool = False,
        sh_degree: int = 3,
    ):
        super().__init__()
        if multires is None:
            multires = [1, 2, 4, 8]

        self.enable_opacity_deform = enable_opacity_deform
        self.enable_sh_deform = enable_sh_deform

        # HexPlane feature grid
        self.hexplane = HexPlane(
            grid_resolution=grid_resolution,
            time_resolution=time_resolution,
            feature_dim=feature_dim,
            multires=multires,
            aabb=aabb,
        )
        in_dim = self.hexplane.out_dim  # feature_dim * 6 planes

        # Shared backbone: Linear(in → W) + optional extra hidden layers
        backbone_layers: list[nn.Module] = [nn.Linear(in_dim, net_width), nn.ReLU()]
        for _ in range(defor_depth - 1):
            backbone_layers += [nn.Linear(net_width, net_width), nn.ReLU()]
        self.backbone = nn.Sequential(*backbone_layers)

        # Output heads (always-on)
        self.head_xyz = _make_head(net_width, net_width, 3)
        self.head_rot = _make_head(net_width, net_width, 4)
        self.head_scale = _make_head(net_width, net_width, 3)

        # Optional heads
        self.head_opacity = None
        if enable_opacity_deform:
            self.head_opacity = _make_head(net_width, net_width, 1)

        self.head_sh = None
        if enable_sh_deform:
            # SH coefficients: (sh_degree+1)^2 × 3 per Gaussian
            sh_coeffs = (sh_degree + 1) ** 2 * 3
            self.head_sh = _make_head(net_width, net_width, sh_coeffs)

        # Initialize all output layers to near-zero so initial deformation ≈ 0
        self._zero_init_heads()


    def _zero_init_heads(self):
        """Initialize all output head last layers to near-zero weights/biases."""
        for head in [self.head_xyz, self.head_rot, self.head_scale,
                     self.head_opacity, self.head_sh]:
            if head is None:
                continue
            # The last Linear is at index -1 in the Sequential
            last_linear = head[-1]
            nn.init.uniform_(last_linear.weight, -1e-4, 1e-4)
            nn.init.zeros_(last_linear.bias)

    def forward(self, xyz: Tensor, t) -> DeformOutput:
        """
        Compute per-Gaussian deformation deltas at timestep t.

        Args:
            xyz: [N, 3] — canonical Gaussian centers (raw, not activated).
                 Passed as .detach() from training so gradient flows only
                 through delta_xyz back to the deformation network.
            t:   scalar float or Tensor — normalized time in [0, 1].

        Returns:
            DeformOutput with all delta tensors.
        """
        # Handle time as scalar or tensor
        if not isinstance(t, Tensor):
            t = torch.tensor(float(t), device=xyz.device, dtype=xyz.dtype)

        # Query HexPlane features: [N, F]
        features = self.hexplane(xyz, t)

        # Shared backbone: [N, W]
        hidden = self.backbone(features)

        # Position delta
        delta_xyz = self.head_xyz(hidden)         # [N, 3]

        # Rotation delta — raw quaternion offset, added to canonical quat then normalized.
        # Matches original 4DGS: r' = r + Δr (Eq. 8), ~0 at init → no rotation change.
        delta_rot = self.head_rot(hidden)          # [N, 4]

        # Scale delta (added in log-space)
        delta_scale = self.head_scale(hidden)     # [N, 3]

        # Optional heads
        delta_opacity = None
        if self.head_opacity is not None:
            delta_opacity = self.head_opacity(hidden)  # [N, 1]

        delta_sh = None
        if self.head_sh is not None:
            delta_sh = self.head_sh(hidden)            # [N, sh_coeffs]

        return DeformOutput(
            delta_xyz=delta_xyz,
            delta_rot=delta_rot,
            delta_scale=delta_scale,
            delta_opacity=delta_opacity,
            delta_sh=delta_sh,
        )



def apply_deformation(
    splats: dict,
    deform_out: DeformOutput,
) -> tuple:
    """
    Apply deformation deltas to raw Gaussian parameters.

    This function applies deltas in the *pre-activation* (log/logit) space
    where appropriate, consistent with gsplat's parameter storage convention.

    Args:
        splats: nn.ParameterDict with keys means, scales, quats, opacities, sh0, shN.
                All are stored in pre-activation form (log scales, logit opacities).
        deform_out: DeformOutput from DeformationField.forward().

    Returns:
        Tuple of (means_d, quats_d, scales_d, opacities_d, colors_d)
        with activations applied, ready for gsplat.rasterization().
    """
    # Means: add delta directly to raw means
    means_d = splats["means"] + deform_out.delta_xyz

    # Scales: add delta in log-space, then exp()
    scales_d = torch.exp(splats["scales"] + deform_out.delta_scale)

    # Quaternions: additive delta then normalize (Eq. 8: r' = r + Δr)
    # Matches original 4DGS. delta_rot ≈ 0 at init → no rotation change.
    quats_d = F.normalize(splats["quats"] + deform_out.delta_rot, p=2, dim=-1)

    # Opacities: add optional delta in logit-space, then sigmoid
    if deform_out.delta_opacity is not None:
        opacs_d = torch.sigmoid(
            splats["opacities"] + deform_out.delta_opacity.squeeze(-1)
        )
    else:
        opacs_d = torch.sigmoid(splats["opacities"])

    # Colors (SH): add optional delta to SH coefficients
    sh0 = splats["sh0"]
    shN = splats["shN"]
    if deform_out.delta_sh is not None:
        # delta_sh: [N, (sh_degree+1)^2 * 3] — reshape and split
        N = sh0.shape[0]
        K = sh0.shape[1] + shN.shape[1]  # total SH bands
        delta_reshaped = deform_out.delta_sh.view(N, K, 3)
        sh0_d = sh0 + delta_reshaped[:, :sh0.shape[1], :]
        shN_d = shN + delta_reshaped[:, sh0.shape[1]:, :]
        colors_d = torch.cat([sh0_d, shN_d], dim=1)  # [N, K, 3]
    else:
        colors_d = torch.cat([sh0, shN], dim=1)  # [N, K, 3]

    return means_d, quats_d, scales_d, opacs_d, colors_d
