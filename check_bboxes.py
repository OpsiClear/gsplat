import torch
import numpy as np
from examples.datasets.read_write_model import read_model
from pathlib import Path
from gsplat.io_ply import import_splats

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

d_means, _, _, _, _, _ = import_splats('/data/shared/elaheh/elly_static_v2/dynamic.ply', device='cpu')
min_p = torch.min(d_means, dim=0).values
max_p = torch.max(d_means, dim=0).values
padding_3d = 0.05
center = (min_p + max_p) / 2
size = (max_p - min_p) * (1 + padding_3d)
min_p = center - size/2
max_p = center + size/2

corners_3d = torch.stack([torch.tensor([x,y,z]) for x in [min_p[0],max_p[0]] for y in [min_p[1],max_p[1]] for z in [min_p[2],max_p[2]]])

colmap_dir = "/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/sparse/0"
cams, imgs, pts = read_model(Path(colmap_dir))

for img in list(imgs.values())[:5]:
    cam = cams[img.camera_id]
    R = torch.from_numpy(qvec2rotmat(img.qvec)).float()
    T = torch.from_numpy(img.tvec).float()
    K = torch.tensor([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]]).float()
    p_cam = (R @ corners_3d.T + T.view(3, 1)).T
    valid = p_cam[:, 2] > 0
    p_proj = (K @ p_cam.T).T
    p_2d = p_proj[:, :2] / p_proj[:, 2:3]
    if valid.any():
        print(f"Image: {img.name}, BBox: {p_2d[valid, 0].min().item():.1f}, {p_2d[valid, 1].min().item():.1f} to {p_2d[valid, 0].max().item():.1f}, {p_2d[valid, 1].max().item():.1f}")
    else:
        print(f"Image: {img.name}, BBox: WRONG")
