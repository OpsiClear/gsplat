"""
HexPlane feature grid for 4D Gaussian Splatting.

Encodes spatiotemporal features using 6 axis-aligned planes:
  XY, XZ, XT, YZ, YT, ZT
at multiple resolutions, similar to the HexPlane paper and used in 4DGaussians.

Reference: https://arxiv.org/abs/2310.08528
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# All 6 pairs of the 4 axes (x=0, y=1, z=2, t=3)
PLANE_PAIRS = [
    (0, 1),  # XY
    (0, 2),  # XZ
    (0, 3),  # XT
    (1, 2),  # YZ
    (1, 3),  # YT
    (2, 3),  # ZT
]


class HexPlane(nn.Module):
    """
    HexPlane spatiotemporal feature grid.

    Stores 6 axis-aligned 2D feature planes at multiple resolutions.
    Given a 4D coordinate (x, y, z, t), looks up features from all 6 planes
    via bilinear interpolation and sums them to produce a single feature vector.

    Args:
        grid_resolution: Spatial grid resolution (H=W=grid_resolution).
        time_resolution: Temporal grid resolution for XT, YT, ZT planes.
        feature_dim: Number of feature channels per plane.
        multires: List of resolution multipliers (multi-resolution).
        aabb: Scene bounding box [2, 3] (min xyz, max xyz). Used to normalize
              spatial coordinates to [-1, 1]. If None, assumes coords in [-1, 1].
    """

    def __init__(
        self,
        grid_resolution: int = 64,
        time_resolution: int = 25,
        feature_dim: int = 16,
        multires: List[int] = None,
        aabb: Optional[Tensor] = None,
    ):
        super().__init__()
        if multires is None:
            multires = [1, 2, 4, 8]

        self.grid_resolution = grid_resolution
        self.time_resolution = time_resolution
        self.feature_dim = feature_dim
        self.multires = multires

        # Register aabb as buffer so it moves to correct device
        if aabb is not None:
            self.register_buffer("aabb", aabb.float())
        else:
            self.aabb = None

        # Build all plane grids: 6 pairs × len(multires) resolutions
        # planes[i][j] = plane for pair i at scale j
        # Shape: [1, feature_dim, H, W]
        self.planes = nn.ParameterList()
        for axis_i, axis_j in PLANE_PAIRS:
            is_temporal = (axis_i == 3 or axis_j == 3)
            for scale in multires:
                H = self._plane_res(axis_i, scale)
                W = self._plane_res(axis_j, scale)
                plane = nn.Parameter(
                    torch.zeros(1, feature_dim, H, W)
                )
                if is_temporal:
                    # Temporal planes (XT, YT, ZT): init to 1 (identity for product).
                    # Initial deformation is zero since product is dominated by spatial planes.
                    nn.init.ones_(plane)
                else:
                    # Spatial planes (XY, XZ, YZ): small random values.
                    nn.init.uniform_(plane, 0.1, 0.5)
                self.planes.append(plane)

        self.num_pairs = len(PLANE_PAIRS)
        self.num_scales = len(multires)
        # Total output dim = feature_dim × num_scales (product across 6 pairs per scale,
        # then concatenate across scales). Matches Eq. 7 of the 4DGS paper.
        self.out_dim = feature_dim * self.num_scales

    def _plane_res(self, axis: int, scale: int) -> int:
        """Return resolution for a given axis and scale multiplier.

        Spatial axes scale linearly with multires. Temporal axis stays FIXED
        across all scales, matching the reference 4DGS implementation.
        """
        if axis == 3:  # time axis — fixed resolution (reference behavior)
            return self.time_resolution
        else:
            return self.grid_resolution * scale

    def _normalize_coord(self, coord: Tensor, axis: int) -> Tensor:
        """Normalize coord for a given axis to [-1, 1] for grid_sample."""
        if axis == 3:
            # Time is expected in [-0.5, 0.5]; map to [-1, 1]
            return coord * 2.0
        else:
            # Spatial: normalize via aabb if available
            if self.aabb is not None:
                aabb_min = self.aabb[0, axis]
                aabb_max = self.aabb[1, axis]
                coord = (coord - aabb_min) / (aabb_max - aabb_min + 1e-8)
                coord = coord * 2.0 - 1.0
            # else assume already in [-1, 1]
            return coord

    def forward(self, xyz: Tensor, t: Tensor) -> Tensor:
        """
        Look up spatiotemporal features for a batch of Gaussians.

        Args:
            xyz: [N, 3] — canonical Gaussian centers (world coords)
            t:   [N] or scalar — normalized time in [-0.5, 0.5]

        Returns:
            features: [N, out_dim] — concatenated plane features
        """
        N = xyz.shape[0]
        device = xyz.device

        # Broadcast t to [N]
        if isinstance(t, (int, float)):
            t = torch.full((N,), float(t), device=device, dtype=xyz.dtype)
        elif t.dim() == 0:
            t = t.expand(N)
        elif t.dim() == 1 and t.shape[0] != N:
            # Batch size may differ from N (e.g., batch_size cameras)
            # Use the first element
            t = t[0:1].expand(N)

        # Build the 4D coord tensor: [N, 4] (x, y, z, t)
        xyzT = torch.cat([xyz, t.unsqueeze(-1)], dim=-1)  # [N, 4]

        # Pre-compute normalized UV coordinates for each plane pair.
        # F.grid_sample convention: grid[..., 0] → W (axis_j), grid[..., 1] → H (axis_i).
        # Plane shape: [1, C, H=res(axis_i), W=res(axis_j)].
        # So the grid must be [axis_j_norm, axis_i_norm] — axis_j first (→ W), axis_i second (→ H).
        # This ensures temporal planes (XT/YT/ZT) sample time at W=time_resolution cells,
        # not at H=grid_resolution cells (which would give 4-5x lower temporal resolution).
        pair_uvs = []
        for axis_i, axis_j in PLANE_PAIRS:
            u = self._normalize_coord(xyzT[:, axis_i], axis_i)  # [N] → H coord
            v = self._normalize_coord(xyzT[:, axis_j], axis_j)  # [N] → W coord
            uv = torch.stack([v, u], dim=-1).unsqueeze(0).unsqueeze(2)  # [1, N, 1, 2]: (W-coord, H-coord)
            pair_uvs.append(uv)

        # For each resolution level: element-wise PRODUCT of all 6 plane features,
        # then concatenate across resolutions (Eq. 7 of 4DGS paper).
        # Planes stored as: planes[pair_idx * num_scales + scale_idx]
        scale_features = []
        for scale_idx in range(self.num_scales):
            prod_feat = None
            for pair_idx in range(self.num_pairs):
                plane_idx = pair_idx * self.num_scales + scale_idx
                plane = self.planes[plane_idx]  # [1, C, H, W]
                feat = F.grid_sample(
                    plane,
                    pair_uvs[pair_idx],
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )  # [1, C, N, 1]
                feat = feat.squeeze(0).squeeze(-1).T  # [N, C]
                if prod_feat is None:
                    prod_feat = feat
                else:
                    prod_feat = prod_feat * feat
            scale_features.append(prod_feat)  # [N, C]

        # Concatenate across resolution levels: [N, C * num_scales]
        features = torch.cat(scale_features, dim=-1)
        return features

    def get_all_planes(self) -> List[Tensor]:
        """Return all plane tensors (for regularization losses)."""
        return list(self.planes)

    def get_spatial_planes(self) -> List[Tensor]:
        """Return only spatial planes (XY, XZ, YZ) for PlaneTV loss."""
        spatial_idx = []
        plane_idx = 0
        for pair_idx, (axis_i, axis_j) in enumerate(PLANE_PAIRS):
            # Spatial plane: neither axis is time (axis=3)
            is_spatial = axis_i != 3 and axis_j != 3
            for scale_idx in range(self.num_scales):
                if is_spatial:
                    spatial_idx.append(plane_idx)
                plane_idx += 1
        return [self.planes[i] for i in spatial_idx]

    def get_temporal_planes(self) -> List[Tensor]:
        """Return only temporal planes (XT, YT, ZT) for time smoothness loss."""
        temporal_idx = []
        plane_idx = 0
        for pair_idx, (axis_i, axis_j) in enumerate(PLANE_PAIRS):
            # Temporal plane: one axis is time
            is_temporal = axis_i == 3 or axis_j == 3
            for scale_idx in range(self.num_scales):
                if is_temporal:
                    temporal_idx.append(plane_idx)
                plane_idx += 1
        return [self.planes[i] for i in temporal_idx]
