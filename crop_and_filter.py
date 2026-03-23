import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import sys

# Add the project root to sys.path to import gsplat and examples
project_root = "/home/elaheh/projects/gsplat"
if project_root not in sys.path:
    sys.path.append(project_root)

from gsplat.io_ply import import_splats, export_splats
from examples.datasets.read_write_model import read_model, write_model, Camera, Image, Point3D

def project_points(points, K, R, T):
    """Project 3D points to 2D image coordinates."""
    points_cam = (R @ points.T + T.view(3, 1)).T
    points_proj = (K @ points_cam.T).T
    points_2d = points_proj[:, :2] / (points_proj[:, 2:3] + 1e-6)
    return points_2d, points_cam[:, 2]

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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    static_ply_path = "/data/shared/elaheh/elly_static_v2/static.ply"
    dynamic_ply_path = "/data/shared/elaheh/elly_static_v2/dynamic.ply"
    data_dir = "/data/shared/elaheh/4D_demo/outdoor/elly/undistorted"
    colmap_dir = os.path.join(data_dir, "sparse/0")
    image_dir = os.path.join(data_dir, "images")
    output_image_dir = os.path.join(data_dir, "cropped_images")
    # Using recropped_sparse as user asked for "rewrtite it as recoped soparse" (probably recropped)
    output_sparse_dir = os.path.join(data_dir, "recropped_sparse", "0")
    output_static_ply = "/data/shared/elaheh/elly_static_v2/cropped_static.ply"
    
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_sparse_dir, exist_ok=True)
    
    start_frame = 0
    end_frame = 150
    
    print("Loading splats...")
    dynamic_splats = import_splats(dynamic_ply_path, device=device)
    static_splats = import_splats(static_ply_path, device=device)
    d_means = dynamic_splats[0]
    s_means = static_splats[0]
    num_static = s_means.shape[0]
    
    # Compute 3D bbox from dynamic Gaussians once, add padding per axis
    print("Computing 3D bounding box from dynamic Gaussians...")
    bbox_min = d_means.min(dim=0).values  # (3,)
    bbox_max = d_means.max(dim=0).values  # (3,)
    extents = bbox_max - bbox_min
    bbox_min = bbox_min.cpu().numpy()
    bbox_max = bbox_max.cpu().numpy()
    print(f"  3D bbox: {bbox_min.round(3)} -> {bbox_max.round(3)}")

    # 8 corners of the padded 3D bbox
    corners = torch.tensor([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ], dtype=torch.float32, device=device)  # (8, 3)

    print("Loading COLMAP model...")
    cameras, images, points3D = read_model(Path(colmap_dir))
    cam_bboxes = {}

    print("Projecting 3D bbox corners to each camera view...")
    for img_id, img_data in tqdm(images.items(), desc="BBox Computation"):
        cam_data = cameras[img_data.camera_id]
        R = torch.from_numpy(qvec2rotmat(img_data.qvec)).float().to(device)
        T = torch.from_numpy(img_data.tvec).float().to(device)
        p = cam_data.params
        if cam_data.model == "PINHOLE": fx, fy, cx, cy = p
        elif cam_data.model == "SIMPLE_PINHOLE": fx = fy = p[0]; cx, cy = p[1], p[2]
        else: fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device).float()

        # Project only 8 corners (not all dynamic Gaussians)
        p_2d, depths = project_points(corners, K, R, T)
        valid = depths > 0.1

        if valid.any():
            pts_v = p_2d[valid]
            m_x = torch.min(pts_v[:, 0]).item()
            M_x = torch.max(pts_v[:, 0]).item()
            m_y = torch.min(pts_v[:, 1]).item()
            M_y = torch.max(pts_v[:, 1]).item()

            # Fixed 10 pixel padding
            m_x -= 10;  M_x += 10
            m_y -= 10;  M_y += 10

            m_x = max(0, int(m_x));  M_x = min(cam_data.width,  int(M_x))
            m_y = max(0, int(m_y));  M_y = min(cam_data.height, int(M_y))
            cam_bboxes[img_id] = [m_x, m_y, M_x, M_y]
        else:
            cam_bboxes[img_id] = [0, 0, 0, 0]

    print("Cropping and updating COLMAP...")
    new_images = {}
    new_cameras = {}
    for img_id, bbox in tqdm(cam_bboxes.items(), desc="Processing"):
        img_data = images[img_id]
        cam_data = cameras[img_data.camera_id]
        m_x, m_y, M_x, M_y = bbox
        new_w, new_h = M_x - m_x, M_y - m_y
        if new_w <= 1 or new_h <= 1: continue

        new_cam_id = img_id
        op = cam_data.params
        if cam_data.model == "PINHOLE": np_p = np.array([op[0], op[1], op[2] - m_x, op[3] - m_y])
        elif cam_data.model == "SIMPLE_PINHOLE": np_p = np.array([op[0], op[1] - m_x, op[2] - m_y])
        else: np_p = op.copy(); np_p[2] -= m_x; np_p[3] -= m_y
        
        new_cameras[new_cam_id] = Camera(id=new_cam_id, model=cam_data.model, width=new_w, height=new_h, params=np_p)
        new_images[img_id] = Image(id=img_id, qvec=img_data.qvec, tvec=img_data.tvec, camera_id=new_cam_id, name=img_data.name, xys=img_data.xys - np.array([m_x, m_y]), point3D_ids=img_data.point3D_ids)
        
        cam_folder = os.path.dirname(img_data.name)
        os.makedirs(os.path.join(output_image_dir, cam_folder), exist_ok=True)
        for f in range(start_frame, end_frame + 1):
            fname = f"{f:06d}.jpg"; src = os.path.join(image_dir, cam_folder, fname); dst = os.path.join(output_image_dir, cam_folder, fname)
            if os.path.exists(src):
                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img[m_y:M_y, m_x:M_x])

    write_model(new_cameras, new_images, points3D, Path(output_sparse_dir))

    print("Filtering static splats...")
    view_data = []
    for img_id, bbox in cam_bboxes.items():
        img_data = images[img_id]
        if "000000" not in img_data.name: continue
        m_x, m_y, M_x, M_y = bbox
        if M_x <= m_x or M_y <= m_y: continue
        cam_data = cameras[img_data.camera_id]
        R = torch.from_numpy(qvec2rotmat(img_data.qvec)).float().to(device)
        T = torch.from_numpy(img_data.tvec).float().to(device)
        p = cam_data.params
        if cam_data.model == "PINHOLE": fx, fy, cx, cy = p
        elif cam_data.model == "SIMPLE_PINHOLE": fx = fy = p[0]; cx, cy = p[1], p[2]
        else: fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device).float()
        view_data.append({"K": K, "R": R, "T": T, "bbox": bbox, "W": cam_data.width, "H": cam_data.height})

    # Keep any static Gaussian visible inside the crop of at least one camera
    in_any_crop = torch.zeros(num_static, dtype=torch.bool, device=device)
    chunk_size = 200000
    for i in tqdm(range(0, num_static, chunk_size), desc="Filtering"):
        end = min(i+chunk_size, num_static); chunk = s_means[i:end]
        for v in view_data:
            p_2d, ds = project_points(chunk, v["K"], v["R"], v["T"])
            m_x, m_y, M_x, M_y = v["bbox"]
            in_crop = (ds > 0) & (p_2d[:,0]>=m_x) & (p_2d[:,0]<M_x) & (p_2d[:,1]>=m_y) & (p_2d[:,1]<M_y)
            in_any_crop[i:end] |= in_crop

    print(f"Visible in at least one crop: {torch.sum(in_any_crop).item()} / {num_static}")
    keep_mask = in_any_crop
    
    final_count = torch.sum(keep_mask).item()
    print(f"Final Count: {final_count}")
    
    if final_count > 0:
        means, scales, quats, opacities, sh0, shN = static_splats
        export_splats(means[keep_mask], scales[keep_mask], quats[keep_mask], opacities[keep_mask], sh0[keep_mask], shN[keep_mask], format="ply", save_to=output_static_ply)
        print(f"Saved {final_count} points to {output_static_ply}")
    else:
        print("No points kept.")
    print("Done!")

if __name__ == "__main__":
    main()
