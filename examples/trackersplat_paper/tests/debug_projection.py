"""
Diagnostic: do `_project_gaussians_2d` and `compute_mean2D` agree on pixel
position for a real thenewface camera? If not, the triangulation ISVD is
garbage because source and target coordinates use different conventions.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from trackersplat_dataset import build_thenewface_video_dataset  # noqa
from trackersplat_paper.motion_fusion_taichi import _project_gaussians_2d  # noqa
from trackersplat_paper.utils.math_utils import compute_mean2D  # noqa


def _proj_matrix_row_major(cam):
    """Reference-style row-major full projection matrix."""
    W, H = int(cam.image_width), int(cam.image_height)
    fx, fy = cam.focal_x, cam.focal_y
    cx, cy = cam.center_x, cam.center_y
    K4 = torch.zeros(4, 4, device=cam.device, dtype=torch.float32)
    K4[0, 0] = 2 * fx / W
    K4[1, 1] = 2 * fy / H
    K4[0, 2] = 2 * cx / W - 1
    K4[1, 2] = 2 * cy / H - 1
    K4[2, 2] = 1.0
    K4[3, 2] = 1.0
    return (K4 @ cam.w2c).T.contiguous()


def _project_pinhole_direct(cam, means3d):
    """Plain pinhole projection with the standard row-major w2c.
    Returns (N, 2) pixel coords. This is the ground-truth."""
    W, H = int(cam.image_width), int(cam.image_height)
    R = cam.w2c[:3, :3]
    T = cam.w2c[:3, 3]
    cam_pts = means3d @ R.T + T[None]             # (N, 3)  camera space
    z = cam_pts[:, 2:3].clamp_min(1e-6)
    u = cam.focal_x * cam_pts[:, 0:1] / z + cam.center_x
    v = cam.focal_y * cam_pts[:, 1:2] / z + cam.center_y
    return torch.cat([u, v], dim=-1)


def main():
    dev = torch.device("cuda")
    video = build_thenewface_video_dataset(
        data_dir="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/",
        n_frames=1, data_factor=4, device="cuda",
    )
    # Pick camera 0 and a 3D point roughly at the scene center
    cam = video[0, 0]
    print(f"camera: {cam.cam_name}  W={cam.image_width} H={cam.image_height}")
    print(f"fx={cam.focal_x:.1f} fy={cam.focal_y:.1f}  cx={cam.center_x:.1f} cy={cam.center_y:.1f}")
    print(f"cam_position (world): {cam.cam_position.cpu().tolist()}")
    print(f"w2c:\n{cam.w2c.cpu().numpy()}")

    # A point at world (0, 0, 0) — i.e. the world origin. Also (0.1, 0, 0).
    means = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.0, 0.1, 0.0],
        [0.0, 0.0, 0.1],
    ], device=dev, dtype=torch.float32)

    # 1) direct pinhole (truth)
    pix_direct = _project_pinhole_direct(cam, means)

    # 2) compute_mean2D (used by ISVD triangulation)
    proj = _proj_matrix_row_major(cam)
    pix_cm2d = compute_mean2D(proj, int(cam.image_width), int(cam.image_height), means)

    # 3) _project_gaussians_2d (used by motion_fusion)
    rotations = torch.zeros(len(means), 4, device=dev)
    rotations[:, 0] = 1.0
    scales = torch.full((len(means), 3), math.log(0.01), device=dev)
    opacities = torch.zeros(len(means), device=dev)
    mean2d, _conic, _det, _bbox, _opac, _valid = _project_gaussians_2d(
        means, rotations, scales.exp(), opacities,
        cam.FoVx, cam.FoVy,
        int(cam.image_width), int(cam.image_height),
        cam.w2c, cam.cam_position, k_sigma=3.0,
    )

    print("\npoint     |  pinhole (truth)  |  compute_mean2D   |  motion_fusion")
    for i in range(len(means)):
        p = means[i].cpu().tolist()
        a = pix_direct[i].cpu().tolist()
        b = pix_cm2d[i].cpu().tolist()
        c = mean2d[i].cpu().tolist()
        print(f"{p}  |  ({a[0]:7.2f}, {a[1]:7.2f})  |  ({b[0]:7.2f}, {b[1]:7.2f})  |  ({c[0]:7.2f}, {c[1]:7.2f})")

    err_cm2d = (pix_cm2d - pix_direct).abs().max().item()
    err_mf = (mean2d - pix_direct).abs().max().item()
    print(f"\nmax |compute_mean2D - pinhole| = {err_cm2d:.3f} px")
    print(f"max |motion_fusion  - pinhole| = {err_mf:.3f} px")
    if err_cm2d > 1.0:
        print("  [FAIL] compute_mean2D differs from pinhole > 1 px")
    if err_mf > 1.0:
        print("  [FAIL] motion_fusion differs from pinhole > 1 px")
    if err_cm2d < 1.0 and err_mf < 1.0:
        print("  [PASS] both agree with pinhole — projections are consistent")


if __name__ == "__main__":
    main()
