"""Generate a 3D bounding box as tiny Gaussians for visualization in SuperSplat."""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsplat.exporter import export_splats


def generate_bbox_gaussians(
    bbox_min, bbox_max, points_per_edge=200, scale=0.005, color=(1.0, 0.0, 0.0)
):
    """Sample points along 12 edges of an axis-aligned bounding box.

    Returns means, scales, quats, opacities, sh0, shN as tensors ready for export_splats.
    """
    mn = np.array(bbox_min, dtype=np.float32)
    mx = np.array(bbox_max, dtype=np.float32)

    # 8 corners
    corners = np.array([
        [mn[0], mn[1], mn[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mx[1], mn[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mx[2]],
        [mn[0], mx[1], mx[2]],
    ])

    # 12 edges (pairs of corner indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]

    all_points = []
    for i, j in edges:
        t = np.linspace(0, 1, points_per_edge, dtype=np.float32)
        pts = corners[i][None, :] * (1 - t[:, None]) + corners[j][None, :] * t[:, None]
        all_points.append(pts)

    means = torch.from_numpy(np.concatenate(all_points, axis=0))  # (N, 3)
    N = len(means)

    # Tiny uniform scale (log-space for export_splats)
    scales = torch.full((N, 3), np.log(scale), dtype=torch.float32)

    # Identity quaternion
    quats = torch.zeros((N, 4), dtype=torch.float32)
    quats[:, 0] = 1.0

    # Full opacity (logit space)
    opacities = torch.full((N,), 5.0, dtype=torch.float32)  # sigmoid(5) ≈ 0.993

    # Color as DC SH coefficient: rgb_to_sh = (rgb - 0.5) / 0.2820948
    C0 = 0.2820948
    rgb = np.array(color, dtype=np.float32)
    sh_dc = (rgb - 0.5) / C0
    sh0 = torch.from_numpy(sh_dc).unsqueeze(0).unsqueeze(0).expand(N, 1, 3).clone()
    shN = torch.zeros((N, 15, 3), dtype=torch.float32)

    return means, scales, quats, opacities, sh0, shN


def main():
    sep_dir = "/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/static_dynamic_output"
    roi_bounds = np.load(os.path.join(sep_dir, "roi_bounds.npy"))
    bbox_min = roi_bounds[0]
    bbox_max = roi_bounds[1]

    print(f"BBox min: {bbox_min}")
    print(f"BBox max: {bbox_max}")
    print(f"BBox size: {bbox_max - bbox_min}")

    means, scales, quats, opacities, sh0, shN = generate_bbox_gaussians(
        bbox_min, bbox_max, points_per_edge=300, scale=0.005, color=(1.0, 0.0, 0.0)
    )
    print(f"Generated {len(means)} Gaussians along 12 edges")

    out_path = os.path.join(sep_dir, "roi_bbox_gaussians.ply")
    export_splats(
        means=means, scales=scales, quats=quats,
        opacities=opacities, sh0=sh0, shN=shN,
        format="ply", save_to=out_path,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
