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

import math
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


class Sine(nn.Module):
    """Pure sine activation: f(x) = sin(x)."""
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class SineReLU(nn.Module):
    """SineReLU: ReLU on positive half, eps*sin(x) on negative half.

    Prevents dead neurons while keeping oscillation amplitude bounded by eps.
    eps=0.01 recommended: negative branch ≈ leaky ReLU(0.01) near origin,
    oscillation amplitude ≤ eps per neuron (negligible relative to O(1) outputs).
    """
    def __init__(self, eps: float = 0.01):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x > 0, x, self.eps * torch.sin(x))


def get_activation(name: str) -> nn.Module:
    """Factory for head activations. Supported: relu, elu, sine, sinerelu, leakyrelu, siren."""
    name = name.lower()
    if name in ("relu",):
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "sine":
        return Sine()
    elif name == "sinerelu":
        return SineReLU(eps=0.01)
    elif name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif name == "siren":
        return Sine()   # same activation as sine; init is handled in _make_head
    else:
        raise ValueError(f"Unknown head activation '{name}'. Choose: relu, elu, sine, sinerelu, leakyrelu, siren")


def _siren_init_(linear: nn.Linear, w0: float = 1.0, c: float = 6.0):
    """SIREN uniform init for a hidden Linear layer.

    Weights drawn from Uniform(-sqrt(c/fan_in)/w0, sqrt(c/fan_in)/w0).
    Ensures sine pre-activations are approximately Uniform(-1,1) so no
    neurons are wasted in the linear (near 0) or flat (saturated) regime.
    w0=1 is appropriate when input features come from a learned encoder
    (HexPlane), not raw coordinates (which would need w0=30).
    """
    fan_in = linear.weight.shape[1]
    bound = math.sqrt(c / fan_in) / w0
    nn.init.uniform_(linear.weight, -bound, bound)
    if linear.bias is not None:
        nn.init.uniform_(linear.bias, -bound, bound)


def _make_head(in_dim: int, hidden_dim: int, out_dim: int, activation: str = "relu") -> nn.Sequential:
    """Two-layer MLP head with configurable activation.

    For activation='siren': uses Sine activation with SIREN initialization
    on the hidden linear layer. The output linear keeps near-zero init
    (critical for stable 4DGS training — initial deformation ≈ 0).
    All other activations use default PyTorch initialization.
    """
    use_siren_init = activation.lower() == "siren"
    act1 = get_activation(activation)
    act2 = get_activation(activation)
    hidden_linear = nn.Linear(hidden_dim, hidden_dim)
    if use_siren_init:
        _siren_init_(hidden_linear, w0=1.0)
    head = nn.Sequential(
        act1,
        hidden_linear,
        act2,
        nn.Linear(hidden_dim, out_dim),
    )
    return head


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
        time_pe_bands: int = 0,
        act_xyz: str = "relu",
        act_rot: str = "relu",
        act_scale: str = "relu",
        act_opacity: str = "relu",
        act_sh: str = "relu",
    ):
        super().__init__()
        if multires is None:
            multires = [1, 2, 4, 8]

        self.enable_opacity_deform = enable_opacity_deform
        self.enable_sh_deform = enable_sh_deform
        self.time_pe_bands = time_pe_bands

        # HexPlane feature grid
        self.hexplane = HexPlane(
            grid_resolution=grid_resolution,
            time_resolution=time_resolution,
            feature_dim=feature_dim,
            multires=multires,
            aabb=aabb,
        )
        in_dim = self.hexplane.out_dim + 2 * time_pe_bands  # +PE dims if any

        # Shared backbone: Linear(in → W) + defor_depth extra hidden layers.
        # defor_depth=0: just [Linear(in→W), ReLU]
        # defor_depth=1: [Linear(in→W), ReLU, Linear(W→W), ReLU]
        backbone_layers: list[nn.Module] = [nn.Linear(in_dim, net_width), nn.ReLU()]
        for _ in range(defor_depth):
            backbone_layers += [nn.Linear(net_width, net_width), nn.ReLU()]
        self.backbone = nn.Sequential(*backbone_layers)

        # Output heads (always-on)
        self.head_xyz = _make_head(net_width, net_width, 3, activation=act_xyz)
        self.head_rot = _make_head(net_width, net_width, 4, activation=act_rot)
        self.head_scale = _make_head(net_width, net_width, 3, activation=act_scale)

        # Optional heads
        self.head_opacity = None
        if enable_opacity_deform:
            self.head_opacity = _make_head(net_width, net_width, 1, activation=act_opacity)

        self.head_sh = None
        if enable_sh_deform:
            # SH coefficients: (sh_degree+1)^2 × 3 per Gaussian
            sh_coeffs = (sh_degree + 1) ** 2 * 3
            self.head_sh = _make_head(net_width, net_width, sh_coeffs, activation=act_sh)

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
            t:   scalar float or Tensor — normalized time in [-0.5, 0.5].

        Returns:
            DeformOutput with all delta tensors.
        """
        # Handle time as scalar or tensor
        if not isinstance(t, Tensor):
            t = torch.tensor(float(t), device=xyz.device, dtype=xyz.dtype)

        # Query HexPlane features: [N, F]
        features = self.hexplane(xyz, t)

        # Optional time positional encoding: [sin(2^k π t), cos(2^k π t)] for k=0..B-1
        if self.time_pe_bands > 0:
            t_f = t.float() if isinstance(t, Tensor) else torch.tensor(float(t), device=xyz.device, dtype=xyz.dtype)
            bands = 2.0 ** torch.arange(self.time_pe_bands, device=xyz.device, dtype=xyz.dtype)  # [B]
            ang = math.pi * t_f * bands  # [B]
            time_pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [2B]
            time_pe = time_pe.unsqueeze(0).expand(features.shape[0], -1)   # [N, 2B]
            features = torch.cat([features, time_pe], dim=-1)              # [N, F+2B]

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
    w: Optional[Tensor] = None,
    aabb: Optional[Tensor] = None,
) -> tuple:
    """
    Apply deformation deltas to raw Gaussian parameters.

    This function applies deltas in the *pre-activation* (log/logit) space
    where appropriate, consistent with gsplat's parameter storage convention.

    Args:
        splats: nn.ParameterDict with keys means, scales, quats, opacities, sh0, shN.
                All are stored in pre-activation form (log scales, logit opacities).
        deform_out: DeformOutput from DeformationField.forward().
        w: Optional [N] weight tensor. If provided, all deltas are scaled by w
           before being applied (enables blending / masking of deformation).
        aabb: Optional [2, 3] scene bounding box. When provided, delta_xyz is
              interpreted in normalized [-1, 1] space and scaled to world coords
              by the AABB half-extent. This decouples the MLP output scale from
              the physical scene size, so weight_constraint and regularization
              weights have consistent meaning regardless of scene scale.

    Returns:
        Tuple of (means_d, quats_d, scales_d, opacities_d, colors_d)
        with activations applied, ready for gsplat.rasterization().
    """
    if w is not None:
        ww = w.unsqueeze(-1)  # [N,1] for broadcasting over spatial dims
        delta_xyz    = deform_out.delta_xyz    * ww
        delta_rot    = deform_out.delta_rot    * ww
        delta_scale  = deform_out.delta_scale  * ww
        delta_opacity = deform_out.delta_opacity * ww if deform_out.delta_opacity is not None else None
        delta_sh      = deform_out.delta_sh * ww      if deform_out.delta_sh      is not None else None
    else:
        delta_xyz    = deform_out.delta_xyz
        delta_rot    = deform_out.delta_rot
        delta_scale  = deform_out.delta_scale
        delta_opacity = deform_out.delta_opacity
        delta_sh      = deform_out.delta_sh

    # Means: scale delta_xyz from normalized space to world coords if aabb provided.
    # The MLP outputs delta_xyz in [-1, 1] normalized units; multiplying by the
    # AABB half-extent converts to meters. Without this, a scene_scale=3m would
    # require the MLP to output raw deltas of ~1.0, while the weight_constraint
    # (in squared world coords) would be 10x stronger than the reconstruction loss.
    if aabb is not None:
        aabb_half = (aabb[1] - aabb[0]) / 2.0  # [3] per-axis half-extent
        delta_xyz = delta_xyz * aabb_half

    means_d = splats["means"] + delta_xyz

    # Scales: add delta in log-space, then exp()
    scales_d = torch.exp(splats["scales"] + delta_scale)

    # Quaternions: additive delta then normalize (Eq. 8: r' = r + Δr)
    # Matches original 4DGS. delta_rot ≈ 0 at init → no rotation change.
    quats_d = F.normalize(splats["quats"] + delta_rot, p=2, dim=-1)

    # Opacities: add optional delta in logit-space, then sigmoid
    if delta_opacity is not None:
        opacs_d = torch.sigmoid(
            splats["opacities"] + delta_opacity.squeeze(-1)
        )
    else:
        opacs_d = torch.sigmoid(splats["opacities"])

    # Colors (SH): add optional delta to SH coefficients
    sh0 = splats["sh0"]
    shN = splats["shN"]
    if delta_sh is not None:
        # delta_sh: [N, deform_K * 3] where deform_K = (deform_sh_degree+1)^2.
        # deform_K may be <= total canonical bands (partial coverage allowed).
        N = sh0.shape[0]
        total_K = sh0.shape[1] + shN.shape[1]  # canonical total bands
        deform_K = delta_sh.shape[1] // 3       # bands the head covers
        delta_reshaped = delta_sh.view(N, deform_K, 3)
        if deform_K < total_K:
            # Pad zeros for higher-order bands not covered by the deform head
            pad = torch.zeros(N, total_K - deform_K, 3, device=delta_sh.device, dtype=delta_sh.dtype)
            delta_reshaped = torch.cat([delta_reshaped, pad], dim=1)  # [N, total_K, 3]
        sh0_d = sh0 + delta_reshaped[:, :sh0.shape[1], :]
        shN_d = shN + delta_reshaped[:, sh0.shape[1]:, :]
        colors_d = torch.cat([sh0_d, shN_d], dim=1)  # [N, total_K, 3]
    else:
        colors_d = torch.cat([sh0, shN], dim=1)  # [N, K, 3]

    return means_d, quats_d, scales_d, opacs_d, colors_d
