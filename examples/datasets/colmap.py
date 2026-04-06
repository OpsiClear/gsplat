import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never


from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
from .read_write_model import read_model


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    image_exts = {".png", ".jpg", ".jpeg"}
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            if os.path.splitext(f)[1].lower() in image_exts:
                paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def resize_image(image: np.ndarray, factor: int) -> np.ndarray:
    """Resize an image using bicubic interpolation."""
    resized_size = (
        int(round(image.shape[1] / factor)),
        int(round(image.shape[0] / factor)),
    )
    resized_image = np.array(
        Image.fromarray(image).resize(resized_size, Image.BICUBIC)
    )
    return resized_image


def resize_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Resize a mask using nearest-neighbor interpolation."""
    resized_size = (
        int(round(mask.shape[1] / factor)),
        int(round(mask.shape[0] / factor)),
    )
    resized_mask = np.array(Image.fromarray(mask).resize(resized_size, Image.NEAREST))
    return resized_mask


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 64,
        undistort_input: bool = True,
        use_masks: bool = False,
        optimize_foreground: bool = False,
        foreground_margin: float = 0.1,
        load_images_in_memory: bool = False,
        frame_num: Optional[int] = None,
        skip_points3d: bool = False,
        mask_dir: Optional[str] = None,
        invert_masks: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.undistort_input = undistort_input
        self.use_masks = use_masks
        self.optimize_foreground = optimize_foreground
        self.foreground_margin = foreground_margin
        self.load_images_in_memory = load_images_in_memory
        self.use_alpha_as_mask = False
        self.invert_masks = invert_masks
        self._temp_image_cache = {}

        # Keep original data_dir for COLMAP data
        self.original_data_dir = data_dir
        
        # Determine COLMAP directory path - use sparse/0/ in data_dir
        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        self.extracted_images_dir = None
        self.frame_dir = None

        cameras, images, points3D = read_model(path=Path(colmap_dir))

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats_and_cam_ids = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for im in images.values():
            rot = im.qvec2rotmat()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats_and_cam_ids.append((im.name, w2c, im.camera_id))

        # Sort by image name.
        w2c_mats_and_cam_ids.sort(key=lambda x: x[0])

        # Per-frame ground truth loading for fixed camera rigs.
        # COLMAP gives us one pose per camera (from a reference frame).
        # Camera poses are identical across all frames, so we keep the
        # poses/intrinsics from COLMAP and swap the image filename to
        # point at the target frame.
        if frame_num is not None:
            image_dir = os.path.join(data_dir, "images")

            # Deduplicate by camera directory (one entry per camera).
            # COLMAP may have registered multiple frames per camera,
            # but the rig is fixed so all poses for the same camera
            # are identical.
            seen_cam_dirs = set()
            remapped = []
            frame_fmt = None  # auto-detect digit format on first camera

            for n, w, c in w2c_mats_and_cam_ids:
                cam_dir = os.path.dirname(n)   # e.g. "take_18_cam_01" or ""
                ext = os.path.splitext(n)[1]   # e.g. ".jpg" or ".png"

                # If COLMAP names have no subdir (e.g. "cam00.png"), use the
                # stem as camera directory (images are at cam00/00001.png)
                if not cam_dir:
                    cam_dir = os.path.splitext(os.path.basename(n))[0]  # "cam00"

                if cam_dir in seen_cam_dirs:
                    continue
                seen_cam_dirs.add(cam_dir)

                # Auto-detect frame filename format from the first camera dir
                if frame_fmt is None:
                    import re
                    cam_path = os.path.join(image_dir, cam_dir)
                    if os.path.isdir(cam_path):
                        fnames = sorted(f for f in os.listdir(cam_path) if re.match(r'\d+\.', f))
                        if fnames:
                            stem, detected_ext = os.path.splitext(fnames[0])
                            frame_fmt = f"0{len(stem)}d"
                            ext = detected_ext
                    if frame_fmt is None:
                        frame_fmt = "06d"  # fallback

                frame_str = f"{frame_num:{frame_fmt}}"
                new_name = os.path.join(cam_dir, frame_str + ext)
                full_path = os.path.join(image_dir, new_name)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(
                        f"[Parser] Ground truth image not found: {full_path}\n"
                        f"  frame_num={frame_num}, camera='{cam_dir}'\n"
                        f"  Check that frame {frame_num} exists in {os.path.join(image_dir, cam_dir)}/"
                    )
                remapped.append((new_name, w, c))

            w2c_mats_and_cam_ids = remapped
            print(
                f"[Parser] frame_num={frame_num}: loaded {len(remapped)} cameras, "
                f"frame_fmt={frame_fmt}, GT images = '<cam_dir>/{frame_str}{ext}'"
            )

        image_names, w2c_mats, camera_ids = zip(
            *((n, w, c) for n, w, c in w2c_mats_and_cam_ids)
        )
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Process camera intrinsics
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        undistort_mask_dict = dict()
        camtype_dict = dict()  # store camera type per camera_id
        for cam_id, cam in cameras.items():
            if cam.model == "SIMPLE_PINHOLE":
                fx = fy = cam.params[0]
                cx, cy = cam.params[1], cam.params[2]
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "PINHOLE":
                fx, fy, cx, cy = cam.params
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "SIMPLE_RADIAL":
                fx, cx, cy, k = cam.params
                fy = fx
                params = np.array([k, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "RADIAL":
                fx, cx, cy, k1, k2 = cam.params
                fy = fx
                params = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "OPENCV":
                fx, fy, cx, cy, k1, k2, p1, p2 = cam.params
                params = np.array([k1, k2, p1, p2], dtype=np.float32)
                camtype = "perspective"
            elif cam.model == "OPENCV_FISHEYE":
                fx, fy, cx, cy, k1, k2, k3, k4 = cam.params
                params = np.array([k1, k2, k3, k4], dtype=np.float32)
                camtype = "fisheye"
            else:
                raise ValueError(f"Unknown camera model {cam.model}")

            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[cam_id] = K
            params_dict[cam_id] = params
            camtype_dict[cam_id] = camtype
            imsize_dict[cam_id] = (cam.width // factor, cam.height // factor)
            undistort_mask_dict[cam_id] = None

        print(
            f"[Parser] {len(images)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(images) == 0:
            raise ValueError("No images found in COLMAP.")

        # Create a mapping from the potentially modified image names to the original names
        # that are used as keys in the point_indices dictionary.
        self.image_name_to_original_name = {}

        # Get original image names from COLMAP model to use for mask mapping
        original_image_names_for_masks = [im.name for im in images.values()]

        # The user's frame directories (frame_0000XX) contain an 'images' folder
        # where images are always named '000000.jpg' inside their respective camera folders.
        # We should NOT rename them here.
        for name in image_names:
            self.image_name_to_original_name[name] = name

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        # This is handled by the sort above.
        
        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        # For this workflow, we now always point directly to the 'images' directory.
        image_dir = os.path.join(data_dir, "images")
        if not os.path.exists(image_dir):
            raise ValueError(f"Image folder {image_dir} does not exist.")
        
        # With the dynamic path generation, the image paths are now the image names.
        image_paths = [os.path.join(image_dir, f) for f in image_names]

        # Load masks if requested.
        self.segmentation_mask_paths = None
        if use_masks:
            # Resolve mask directory: explicit mask_dir takes priority, then data_dir/masks
            resolved_mask_dir = mask_dir if mask_dir else os.path.join(data_dir, "masks")
            if os.path.exists(resolved_mask_dir):
                print(f"[Parser] Loading masks from {resolved_mask_dir}" + (" (inverted)" if invert_masks else ""))
                self.segmentation_mask_paths = []
                for f in image_names:
                    # Direct path matching: mask has same relative path as image
                    mask_path = os.path.join(resolved_mask_dir, f)
                    if os.path.exists(mask_path):
                        self.segmentation_mask_paths.append(mask_path)
                    else:
                        # Try alternative extensions
                        base = os.path.splitext(f)[0]
                        found = False
                        for ext in ['.png', '.jpg', '.jpeg']:
                            alt_path = os.path.join(resolved_mask_dir, base + ext)
                            if os.path.exists(alt_path):
                                self.segmentation_mask_paths.append(alt_path)
                                found = True
                                break
                        if not found:
                            # Fallback: look for motion_mask.png in the camera subdirectory
                            cam_dir = os.path.dirname(f)
                            for motion_name in ['motion_mask.png', 'motion_mask.jpg']:
                                motion_path = os.path.join(resolved_mask_dir, cam_dir, motion_name)
                                if os.path.exists(motion_path):
                                    self.segmentation_mask_paths.append(motion_path)
                                    found = True
                                    break
                        if not found:
                            self.segmentation_mask_paths.append(None)
                print(f"[Parser] Found {sum(1 for p in self.segmentation_mask_paths if p is not None)} masks out of {len(image_names)} images")
            else:
                print(f"[Parser] Warning: use_masks=True but mask directory {resolved_mask_dir} does not exist.")
                print("[Parser] Fallback: Will use alpha channel from images as masks.")
                self.use_alpha_as_mask = True
                self.segmentation_mask_paths = [None] * len(image_names)

        # Load dynamic masks for static/dynamic labeling (separate from segmentation masks)
        # These masks indicate which pixels are dynamic (1) vs static (0)
        self.dynamic_mask_paths = None
        self.has_dynamic_map = False  # Flag to indicate if dynamic masks are available
        dynamic_mask_dir = os.path.join(data_dir, "dynamic_map")
        if os.path.exists(dynamic_mask_dir):
            print(f"[Parser] Loading dynamic masks from {dynamic_mask_dir}")
            dynamic_mask_files = sorted(_get_rel_paths(dynamic_mask_dir))
            colmap_files = sorted(original_image_names_for_masks)
            colmap_to_dynamic_mask = dict(zip(colmap_files, dynamic_mask_files))
            self.dynamic_mask_paths = []
            missing_masks = []
            for f in image_names:
                if f in colmap_to_dynamic_mask:
                    mask_path = os.path.join(dynamic_mask_dir, colmap_to_dynamic_mask[f])
                    if os.path.exists(mask_path) and os.path.splitext(mask_path)[1].lower() in {".png", ".jpg", ".jpeg"}:
                        self.dynamic_mask_paths.append(mask_path)
                    else:
                        self.dynamic_mask_paths.append(None)
                        missing_masks.append(f)
                else:
                    self.dynamic_mask_paths.append(None)
                    missing_masks.append(f)

            num_found = sum(1 for p in self.dynamic_mask_paths if p is not None)
            print(f"[Parser] Found {num_found} dynamic masks out of {len(image_names)} images")

            if num_found > 0:
                self.has_dynamic_map = True

            if missing_masks:
                print(f"[Parser] WARNING: {len(missing_masks)} images are missing dynamic masks!")
                if len(missing_masks) <= 10:
                    for m in missing_masks:
                        print(f"  - {m}")
                else:
                    for m in missing_masks[:5]:
                        print(f"  - {m}")
                    print(f"  ... and {len(missing_masks) - 5} more")
        else:
            print(f"[Parser] No dynamic_map directory found at {dynamic_mask_dir}")

        # Optimize foreground
        self.foreground_bboxes = {}
        if self.optimize_foreground:
            if not self.use_masks:
                raise ValueError("optimize_foreground requires use_masks=True")

            new_Ks_dict = {}
            new_imsize_dict = {}
            new_params_dict = {}
            new_camtype_dict = {}
            new_undistort_mask_dict = {}
            new_camera_ids = []

            for i, image_name in enumerate(tqdm(image_names, desc="Optimizing foreground")):
                camera_id = i  # new unique camera id
                new_camera_ids.append(camera_id)
                original_camera_id = camera_ids[i]

                # copy params from original camera
                new_params_dict[camera_id] = params_dict[original_camera_id]
                new_camtype_dict[camera_id] = camtype_dict[original_camera_id]
                new_undistort_mask_dict[camera_id] = undistort_mask_dict[original_camera_id]
                K = Ks_dict[original_camera_id].copy()
                width, height = imsize_dict[original_camera_id]

                mask = None
                if self.segmentation_mask_paths is not None:
                    mask_path = self.segmentation_mask_paths[i]
                    if mask_path is not None and os.path.exists(mask_path):
                        mask = imageio.imread(mask_path)

                if mask is None and self.use_alpha_as_mask:
                    # Load original image to get alpha channel
                    original_image_path = os.path.join(self.data_dir, "overlays", image_names[i])
                    if not os.path.exists(original_image_path):
                        original_image_path = os.path.join(self.data_dir, "images", image_names[i])
                    
                    if os.path.splitext(original_image_path)[1].lower() in {".png", ".jpg", ".jpeg"} and os.path.exists(original_image_path):
                        if image_names[i] in self._temp_image_cache:
                            rgba_image = self._temp_image_cache[image_names[i]]
                        else:
                            rgba_image = imageio.imread(original_image_path)
                            self._temp_image_cache[image_names[i]] = rgba_image
                        
                        if rgba_image.shape[-1] == 4:
                            mask = rgba_image[..., 3]

                if mask is not None:
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]

                    # Downsample mask if factor > 1, before calculating bbox.
                    if self.factor > 1:
                        mask = resize_mask(mask, self.factor)

                    mask = mask.astype(np.float32) / 255.0
                    # get foreground bounding box
                    y_indices, x_indices = np.nonzero(mask > 0.1)
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        y_min, y_max = y_indices.min(), y_indices.max()
                        x_min, x_max = x_indices.min(), x_indices.max()

                        # add margin
                        h, w = y_max - y_min, x_max - x_min
                        margin_y = int(h * self.foreground_margin)
                        margin_x = int(w * self.foreground_margin)

                        y_min = max(0, y_min - margin_y)
                        y_max = min(height, y_max + margin_y)
                        x_min = max(0, x_min - margin_x)
                        x_max = min(width, x_max + margin_x)

                        self.foreground_bboxes[image_name] = (x_min, y_min, x_max - x_min, y_max - y_min)

                        # update camera intrinsics
                        K[0, 2] -= x_min
                        K[1, 2] -= y_min
                        width, height = x_max - x_min, y_max - y_min
                    else:
                         self.foreground_bboxes[image_name] = (0, 0, width, height)
                else:
                    self.foreground_bboxes[image_name] = (0, 0, width, height)

                new_Ks_dict[camera_id] = K
                new_imsize_dict[camera_id] = (width, height)

            # replace old dicts
            Ks_dict = new_Ks_dict
            imsize_dict = new_imsize_dict
            params_dict = new_params_dict
            camtype_dict = new_camtype_dict
            undistort_mask_dict = new_undistort_mask_dict
            camera_ids = new_camera_ids


        # 3D points and {image_name -> [point_idx]}
        if not skip_points3d:
            points_list = []
            points_err_list = []
            points_rgb_list = []
            point3D_id_to_idx = {}
            for i, (p_id, p) in enumerate(points3D.items()):
                points_list.append(p.xyz)
                points_err_list.append(p.error)
                points_rgb_list.append(p.rgb)
                point3D_id_to_idx[p_id] = i

            points = np.array(points_list).astype(np.float32)
            points_err = np.array(points_err_list).astype(np.float32)
            points_rgb = np.array(points_rgb_list).astype(np.uint8)

            point_indices = dict()
            image_id_to_name = {img.id: img.name for img in images.values()}
            for p_id, p in points3D.items():
                point_idx = point3D_id_to_idx[p_id]
                for image_id in p.image_ids:
                    if image_id in image_id_to_name:
                        image_name = image_id_to_name[image_id]
                        point_indices.setdefault(image_name, []).append(point_idx)

            point_indices = {
                k: np.array(v).astype(np.int32) for k, v in point_indices.items()
            }
        else:
            print("[Parser] Skipping point3D loading (skip_points3d=True)")
            points = np.zeros((0, 3), dtype=np.float32)
            points_err = np.zeros((0,), dtype=np.float32)
            points_rgb = np.zeros((0, 3), dtype=np.uint8)
            point_indices = {}

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            if len(points) > 0:
                points = transform_points(T1, points)

                # Recenter the scene based on the median of the point cloud.
                # This provides better centering than the camera-based method
                # for turntable-style captures.
                centroid = np.median(points, axis=0)
                T_recenter = np.eye(4)
                T_recenter[:3, 3] = -centroid

                points = transform_points(T_recenter, points)
                camtoworlds = transform_cameras(T_recenter, camtoworlds)

                transform = T_recenter @ T1
            else:
                # No points available, use camera-only normalization
                transform = T1

            # # Fix for up side down. We assume more points towards
            # # the bottom of the scene which is true when ground floor is
            # # present in the images.
            # if np.median(points[:, 2]) > np.mean(points[:, 2]):
            #     # rotate 180 degrees around x axis such that z is flipped
            #     T3 = np.array(
            #         [
            #             [1.0, 0.0, 0.0, 0.0],
            #             [0.0, -1.0, 0.0, 0.0],
            #             [0.0, 0.0, -1.0, 0.0],
            #             [0.0, 0.0, 0.0, 1.0],
            #         ]
            #     )
            #     camtoworlds = transform_cameras(T3, camtoworlds)
            #     points = transform_points(T3, points)
            #     transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.camtype_dict = camtype_dict  # Dict of camera_id -> camera type
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.undistort_mask_dict = undistort_mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # Camera ID to index mapping
        unique_camera_ids = sorted(set(self.camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.num_cameras = len(unique_camera_ids)
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in self.camera_ids]

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        # Compare at the target resolution (after factor) to avoid undoing the factor.
        if not self.optimize_foreground:
            actual_image = imageio.imread(self.image_paths[0])[..., :3]
            if self.factor > 1:
                actual_image = resize_image(actual_image, self.factor)
            actual_height, actual_width = actual_image.shape[:2]
            colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
            s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
            # Only apply correction if there is a real mismatch (not just rounding)
            if abs(s_width - 1.0) > 0.01 or abs(s_height - 1.0) > 0.01:
                for camera_id, K in self.Ks_dict.items():
                    K[0, :] *= s_width
                    K[1, :] *= s_height
                    self.Ks_dict[camera_id] = K
                    width, height = self.imsize_dict[camera_id]
                    self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion (conditional based on undistort_input flag)
        if self.undistort_input:
            self.mapx_dict = dict()
            self.mapy_dict = dict()
            self.roi_undist_dict = dict()
            for camera_id in self.params_dict.keys():
                params = self.params_dict[camera_id]
                if len(params) == 0:
                    continue  # no distortion
                assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
                assert (
                    camera_id in self.params_dict
                ), f"Missing params for camera {camera_id}"
                K = self.Ks_dict[camera_id]
                width, height = self.imsize_dict[camera_id]
                camtype = self.camtype_dict[camera_id]

                if camtype == "perspective":
                    K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                        K, params, (width, height), 0
                    )
                    mapx, mapy = cv2.initUndistortRectifyMap(
                        K, params, None, K_undist, (width, height), cv2.CV_32FC1
                    )
                    mask = None
                elif camtype == "fisheye":
                    fx = K[0, 0]
                    fy = K[1, 1]
                    cx = K[0, 2]
                    cy = K[1, 2]
                    grid_x, grid_y = np.meshgrid(
                        np.arange(width, dtype=np.float32),
                        np.arange(height, dtype=np.float32),
                        indexing="xy",
                    )
                    x1 = (grid_x - cx) / fx
                    y1 = (grid_y - cy) / fy
                    theta = np.sqrt(x1**2 + y1**2)
                    r = (
                        1.0
                        + params[0] * theta**2
                        + params[1] * theta**4
                        + params[2] * theta**6
                        + params[3] * theta**8
                    )
                    mapx = (fx * x1 * r + width // 2).astype(np.float32)
                    mapy = (fy * y1 * r + height // 2).astype(np.float32)

                    # Use mask to define ROI
                    mask = np.logical_and(
                        np.logical_and(mapx > 0, mapy > 0),
                        np.logical_and(mapx < width - 1, mapy < height - 1),
                    )
                    y_indices, x_indices = np.nonzero(mask)
                    y_min, y_max = y_indices.min(), y_indices.max() + 1
                    x_min, x_max = x_indices.min(), x_indices.max() + 1
                    mask = mask[y_min:y_max, x_min:x_max]
                    K_undist = K.copy()
                    K_undist[0, 2] -= x_min
                    K_undist[1, 2] -= y_min
                    roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    assert_never(camtype)

                self.mapx_dict[camera_id] = mapx
                self.mapy_dict[camera_id] = mapy
                self.Ks_dict[camera_id] = K_undist
                self.roi_undist_dict[camera_id] = roi_undist
                self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
                self.undistort_mask_dict[camera_id] = mask
        else:
            # When undistortion is disabled, initialize empty dictionaries
            self.mapx_dict = dict()
            self.mapy_dict = dict()
            self.roi_undist_dict = dict()

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Load images into memory
        self.images_dict = {}
        self.masks_dict = {}
        self.dynamic_map_dict = {}  # For static/dynamic labeling
        if self.load_images_in_memory:
            print(f"[Parser] Loading {len(self.image_paths)} images into memory...")
            for i, image_path in enumerate(tqdm(self.image_paths, desc="Loading images")):
                image_name = self.image_names[i]
                
                if image_name in self._temp_image_cache:
                    full_image = self._temp_image_cache[image_name]
                else:
                    full_image = imageio.imread(image_path)

                image = full_image[..., :3]
                camera_id = self.camera_ids[i]
                params = self.params_dict[camera_id]

                # Load segmentation mask from file if use_masks is enabled and mask path exists
                segmentation_mask = None
                if self.use_masks:
                    if self.segmentation_mask_paths is not None:
                        mask_path = self.segmentation_mask_paths[i]
                        if mask_path is not None and os.path.exists(mask_path):
                            segmentation_mask = imageio.imread(mask_path)
                            if len(segmentation_mask.shape) == 3:
                                segmentation_mask = segmentation_mask[..., 0]
                            if self.invert_masks:
                                segmentation_mask = 255 - segmentation_mask
                    if segmentation_mask is None and self.use_alpha_as_mask:
                        if full_image.shape[-1] == 4:
                            segmentation_mask = full_image[..., 3]

                # --- In-memory processing pipeline ---

                # 1. Downsample image and mask
                if self.factor > 1:
                    image = resize_image(image, self.factor)
                    if segmentation_mask is not None:
                        segmentation_mask = resize_mask(
                            segmentation_mask, self.factor
                        )

                # 2. Undistort
                if self.undistort_input and len(params) > 0 and camera_id in self.mapx_dict:
                    mapx, mapy = self.mapx_dict[camera_id], self.mapy_dict[camera_id]
                    image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                    if segmentation_mask is not None:
                        segmentation_mask = cv2.remap(
                            segmentation_mask, mapx, mapy, cv2.INTER_NEAREST
                        )
                    x, y, w, h = self.roi_undist_dict[camera_id]
                    image = image[y : y + h, x : x + w]
                    if segmentation_mask is not None:
                        segmentation_mask = segmentation_mask[y : y + h, x : x + w]

                # 3. Crop to foreground
                if self.optimize_foreground:
                    x, y, w, h = self.foreground_bboxes[image_name]
                    image = image[y : y + h, x : x + w]
                    if segmentation_mask is not None:
                        segmentation_mask = segmentation_mask[y : y + h, x : x + w]

                # Load dynamic mask for static/dynamic labeling (if available)
                dynamic_mask = None
                if self.dynamic_mask_paths is not None:
                    dynamic_mask_path = self.dynamic_mask_paths[i]
                    if dynamic_mask_path is not None and os.path.exists(dynamic_mask_path):
                        dynamic_mask = imageio.imread(dynamic_mask_path)
                        if len(dynamic_mask.shape) == 3:
                            dynamic_mask = dynamic_mask[..., 0]

                        # Apply same transformations as image
                        if self.factor > 1:
                            dynamic_mask = resize_mask(dynamic_mask, self.factor)

                        if self.undistort_input and len(params) > 0 and camera_id in self.mapx_dict:
                            mapx, mapy = self.mapx_dict[camera_id], self.mapy_dict[camera_id]
                            dynamic_mask = cv2.remap(dynamic_mask, mapx, mapy, cv2.INTER_NEAREST)
                            x, y, w, h = self.roi_undist_dict[camera_id]
                            dynamic_mask = dynamic_mask[y : y + h, x : x + w]

                        if self.optimize_foreground:
                            x, y, w, h = self.foreground_bboxes[image_name]
                            dynamic_mask = dynamic_mask[y : y + h, x : x + w]

                # 4. Final conversion
                image = image.astype(np.float32) / 255.0
                self.images_dict[image_name] = torch.from_numpy(image.copy()).float()
                if segmentation_mask is not None:
                    segmentation_mask = segmentation_mask.astype(np.float32) / 255.0
                    self.masks_dict[image_name] = torch.from_numpy(
                        segmentation_mask.copy()
                    ).float()
                if dynamic_mask is not None:
                    dynamic_mask = dynamic_mask.astype(np.float32) / 255.0
                    self.dynamic_map_dict[image_name] = torch.from_numpy(
                        dynamic_mask.copy()
                    ).float()
        
class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))

        if self.parser.test_every < 1:
            # All images used for training; val repeats a few from training set
            if split == "train":
                self.indices = indices  # all images
            else:
                # Pick up to 5 evenly-spaced images from the training set as val
                n_val = min(5, len(indices))
                val_idx = np.linspace(0, len(indices) - 1, n_val, dtype=int)
                self.indices = indices[val_idx]
        else:
            # Normal behavior: split based on test_every
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image_name = self.parser.image_names[index]
        camera_id = self.parser.camera_ids[index]

        if self.parser.load_images_in_memory:
            image = self.parser.images_dict[image_name]
            segmentation_mask = self.parser.masks_dict.get(image_name)
        else:
            full_image = imageio.imread(self.parser.image_paths[index])
            image = full_image[..., :3]
            params = self.parser.params_dict[camera_id]

            # Load segmentation mask from file if use_masks is enabled and mask path exists
            segmentation_mask = None
            if self.parser.use_masks:
                if self.parser.segmentation_mask_paths is not None:
                    mask_path = self.parser.segmentation_mask_paths[index]
                    if mask_path is not None and os.path.exists(mask_path):
                        segmentation_mask = imageio.imread(mask_path)
                        if len(segmentation_mask.shape) == 3:
                            segmentation_mask = segmentation_mask[..., 0]
                        if self.parser.invert_masks:
                            segmentation_mask = 255 - segmentation_mask
                if segmentation_mask is None and self.parser.use_alpha_as_mask:
                    if full_image.shape[-1] == 4:
                        segmentation_mask = full_image[..., 3]

            # Downsample to match K (mirrors the in-memory preload path)
            if self.parser.factor > 1:
                image = resize_image(image, self.parser.factor)
                if segmentation_mask is not None:
                    segmentation_mask = resize_mask(segmentation_mask, self.parser.factor)

            # Convert to float after resize
            image = image.astype(np.float32) / 255.0
            if segmentation_mask is not None:
                segmentation_mask = segmentation_mask.astype(np.float32) / 255.0

            if self.parser.undistort_input and len(params) > 0 and camera_id in self.parser.mapx_dict:
                # Images are distorted and undistortion is enabled. Undistort them.
                mapx, mapy = (
                    self.parser.mapx_dict[camera_id],
                    self.parser.mapy_dict[camera_id],
                )
                image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
                # Also undistort the segmentation mask if it exists
                if segmentation_mask is not None:
                    segmentation_mask = cv2.remap(segmentation_mask, mapx, mapy, cv2.INTER_NEAREST)
                x, y, w, h = self.parser.roi_undist_dict[camera_id]
                image = image[y : y + h, x : x + w]
                if segmentation_mask is not None:
                    segmentation_mask = segmentation_mask[y : y + h, x : x + w]

            if self.parser.optimize_foreground:
                x, y, w, h = self.parser.foreground_bboxes[image_name]
                image = image[y : y + h, x : x + w]
                if segmentation_mask is not None:
                    segmentation_mask = segmentation_mask[y : y + h, x : x + w]

        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K if undistort_input=True, original K if False
        camtoworlds = self.parser.camtoworlds[index]
        undistort_mask = self.parser.undistort_mask_dict[camera_id]  # mask from undistortion (fisheye cameras)
        params = self.parser.params_dict[camera_id]

        # Track patch crop coordinates for applying to all masks
        patch_crop_x, patch_crop_y = None, None
        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            patch_crop_x = np.random.randint(0, max(w - self.patch_size, 1))
            patch_crop_y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[patch_crop_y : patch_crop_y + self.patch_size, patch_crop_x : patch_crop_x + self.patch_size]
            if segmentation_mask is not None:
                segmentation_mask = segmentation_mask[patch_crop_y : patch_crop_y + self.patch_size, patch_crop_x : patch_crop_x + self.patch_size]
            K[0, 2] -= patch_crop_x
            K[1, 2] -= patch_crop_y

        if not self.parser.load_images_in_memory:
            image = torch.from_numpy(image).float()
            if segmentation_mask is not None:
                segmentation_mask = torch.from_numpy(segmentation_mask).float()

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": image,
            "image_id": item,  # the index of the image in the dataset
            "camera_id": camera_id,
            "camera_idx": self.parser.camera_indices[index],
        }

        # Add undistortion mask if it exists (for fisheye cameras)
        if undistort_mask is not None:
            data["undistort_mask"] = torch.from_numpy(undistort_mask).bool()

        # Add segmentation mask if it exists (from file)
        if segmentation_mask is not None:
            data["segmentation_mask"] = segmentation_mask

        # Add distortion parameters and camera type to the data
        data["distortion_params"] = torch.from_numpy(params).float()
        data["camera_type"] = self.parser.camtype_dict[camera_id]

        # Load dynamic mask for static/dynamic labeling (NOT used in loss, just for labeling)
        dynamic_mask = None
        if self.parser.load_images_in_memory:
            # Use pre-loaded dynamic mask from memory
            dynamic_mask = self.parser.dynamic_map_dict.get(image_name)
            if dynamic_mask is not None:
                # Apply patch crop if needed (mask is already a tensor)
                if self.patch_size is not None and patch_crop_x is not None and patch_crop_y is not None:
                    dynamic_mask = dynamic_mask[patch_crop_y : patch_crop_y + self.patch_size, patch_crop_x : patch_crop_x + self.patch_size]
        else:
            # Load dynamic mask from file
            if self.parser.dynamic_mask_paths is not None:
                dynamic_mask_path = self.parser.dynamic_mask_paths[index]
                if dynamic_mask_path is not None and os.path.exists(dynamic_mask_path):
                    dynamic_mask = imageio.imread(dynamic_mask_path).astype(np.float32) / 255.0
                    if len(dynamic_mask.shape) == 3:
                        dynamic_mask = dynamic_mask[..., 0]  # use first channel if RGB

                    # Apply same transformations as image
                    if self.parser.factor > 1:
                        dynamic_mask = resize_mask((dynamic_mask * 255).astype(np.uint8), self.parser.factor).astype(np.float32) / 255.0

                    if self.parser.undistort_input and len(params) > 0 and camera_id in self.parser.mapx_dict:
                        mapx, mapy = self.parser.mapx_dict[camera_id], self.parser.mapy_dict[camera_id]
                        dynamic_mask = cv2.remap(dynamic_mask, mapx, mapy, cv2.INTER_NEAREST)
                        roi_x, roi_y, roi_w, roi_h = self.parser.roi_undist_dict[camera_id]
                        dynamic_mask = dynamic_mask[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

                    if self.parser.optimize_foreground:
                        fg_x, fg_y, fg_w, fg_h = self.parser.foreground_bboxes[image_name]
                        dynamic_mask = dynamic_mask[fg_y : fg_y + fg_h, fg_x : fg_x + fg_w]

                    # Apply patch crop using the same coordinates used for the image
                    if self.patch_size is not None and patch_crop_x is not None and patch_crop_y is not None:
                        dynamic_mask = dynamic_mask[patch_crop_y : patch_crop_y + self.patch_size, patch_crop_x : patch_crop_x + self.patch_size]

                    dynamic_mask = torch.from_numpy(dynamic_mask).float()

        if dynamic_mask is not None:
            data["dynamic_mask"] = dynamic_mask

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            original_image_name = self.parser.image_name_to_original_name[image_name]
            point_indices = self.parser.point_indices[original_image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse
    import numpy as np
    # We don't need imageio or cv2 for this test
    # import imageio.v2 as imageio
    # import cv2 

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--frame_num", type=int, default=0, help="Frame number to load from camera sequences.")
    parser.add_argument(
        "--use_masks", action="store_true", help="Load masks from masks directory"
    )
    parser.add_argument(
        "--optimize_foreground",
        action="store_true",
        help="Optimize foreground by cropping",
    )
    parser.add_argument(
        "--load_images_in_memory",
        action="store_true",
        help="Load all images into memory",
    )
    parser.add_argument(
        "--exclude_prefixes",
        type=str,
        nargs="+",
        default=[],
        help="Prefixes of images to exclude.",
    )
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=True,
        test_every=8,
        use_masks=args.use_masks,
        optimize_foreground=args.optimize_foreground,
        load_images_in_memory=args.load_images_in_memory,
        exclude_prefixes=args.exclude_prefixes,
        frame_num=args.frame_num, 
    )
    # We set load_depths=False to speed up this test
    dataset = Dataset(parser, split="train", load_depths=True)
    
    print(f"Dataset: {len(dataset)} images.")
    print(f"--- Running Debug Test for frame_num: {args.frame_num} ---")

    # --- NEW DEBUGGING CODE ---
    
    # Get 5 random indices from the dataset
    num_samples_to_test = 5
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check your data_dir and splits.")
    else:
        if len(dataset) < num_samples_to_test:
            print(f"Warning: Dataset size ({len(dataset)}) is less than {num_samples_to_test}. Testing all samples.")
            sample_indices = np.arange(len(dataset))
        else:
            sample_indices = np.random.choice(len(dataset), num_samples_to_test, replace=False)

        print(f"Testing {len(sample_indices)} random samples...")

        for i, item_index in enumerate(sample_indices):
            print(f"\n--- Sample {i+1}/{len(sample_indices)} (Dataset Index: {item_index}) ---")
            
            # Get the data from the dataset
            data = dataset[item_index]
            
            # Find the original corresponding index in the parser
            parser_index = dataset.indices[item_index]

            # Get the original colmap name (e.g., "cam_01/000000.jpg")
            # We need to map the modified name back to the original for this
            modified_image_name = dataset.parser.image_names[parser_index]
            original_image_name = parser.image_name_to_original_name[modified_image_name]
            
            # Get the path of the image that *was actually loaded*
            loaded_image_path = dataset.parser.image_paths[parser_index]

            print(f"  Colmap Ref Name:   {original_image_name}")
            print(f"  Loaded Image Path: {loaded_image_path}")
            print(f"  Internal Camera ID:  {data['camera_id']}")
            
            print("  Intrinsics (K):")
            print(data['K'].numpy())
            
            print("  Extrinsics (camtoworld):")
            print(data['camtoworld'].numpy())

    print("\n--- Debug Test Complete ---")

# Alias for compatibility with simple_trainer_orig.py
GSCDataset = Dataset
