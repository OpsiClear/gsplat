import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add the project root to sys.path to import gsplat and examples
project_root = "/home/elaheh/projects/gsplat"
if project_root not in sys.path:
    sys.path.append(project_root)

from examples.datasets.read_write_model import read_model

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

colmap_dir = "/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/sparse/0"
cams, imgs, pts = read_model(Path(colmap_dir))
img = list(imgs.values())[0]
cam = cams[img.camera_id]
print(f"Image: {img.name}")
print(f"Camera ID: {img.camera_id}, Model: {cam.model}, Params: {cam.params}")

R = torch.from_numpy(qvec2rotmat(img.qvec)).float()
T = torch.from_numpy(img.tvec).float()
K = torch.tensor([
    [cam.params[0], 0, cam.params[2]],
    [0, cam.params[1], cam.params[3]],
    [0, 0, 1]
]).float()

from gsplat.io_ply import import_splats
d_means, _, _, _, _, _ = import_splats('/data/shared/elaheh/elly_static_v2/dynamic.ply', device='cpu')
p = d_means[:1]
print(f"World Point: {p}")
p_cam = (R @ p.T + T.view(3, 1)).T
print(f"Cam Point: {p_cam}")
p_proj = (K @ p_cam.T).T
p_2d = p_proj[:, :2] / p_proj[:, 2:3]
print(f"2D Point: {p_2d}")
