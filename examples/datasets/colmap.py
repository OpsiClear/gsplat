# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from exif import compute_exposure_from_exif
from .normalize import (
    align_principal_axes,
    orient_and_center,
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
                rel_path = os.path.relpath(os.path.join(dp, f), path_dir)
                # Normalize to forward slashes for cross-platform consistency
                paths.append(rel_path.replace(os.sep, '/'))
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


def _resolve_parallel_worker_count(num_items: int) -> int:
    if num_items <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(num_items, 8, cpu_count))


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        image = imageio.imread(image_path)[..., :3]
        resized_image = resize_image(image, factor)
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        undistort_input: bool = True,
        use_masks: bool = False,
        optimize_foreground: bool = False,
        foreground_margin: float = 0.1,
        load_images_in_memory: bool = False,
        load_images_to_gpu: bool = False,
        exclude_prefixes: Optional[List[str]] = None,
        load_exposure: bool = False,
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
        self.load_images_to_gpu = load_images_to_gpu
        self.use_alpha_as_mask = False
        self._temp_image_cache = {}
        self.exclude_prefixes = exclude_prefixes or []
        self.load_exposure = load_exposure

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        cameras, images, points3D = read_model(path=Path(colmap_dir))

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        undistort_mask_dict = dict()
        camtype_dict = dict()  # store camera type per camera_id
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for im in images.values():
            rot = im.qvec2rotmat()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]

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
            Ks_dict[camera_id] = K

            params_dict[camera_id] = params
            camtype_dict[camera_id] = camtype
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            undistort_mask_dict[camera_id] = None

        print(
            f"[Parser] {len(images)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(images) == 0:
            raise ValueError("No images found in COLMAP.")
        # if not (type_ == 0 or type_ == 1): # TODO: check this
        #     print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [im.name for im in images.values()]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        if self.exclude_prefixes:
            print(f"[Parser] Excluding images with prefixes: {self.exclude_prefixes}")
            keep_indices = []
            original_image_names = image_names
            image_names = []
            for i, name in enumerate(original_image_names):
                if not any(name.startswith(p) for p in self.exclude_prefixes):
                    keep_indices.append(i)
                    image_names.append(name)

            camtoworlds = camtoworlds[keep_indices]
            camera_ids = [camera_ids[i] for i in keep_indices]
            print(f"[Parser] {len(image_names)} images remaining after exclusion.")

            # Filter images and points3D
            kept_image_names = set(image_names)
            kept_image_ids = set()
            new_images = {}
            for img_id, img in images.items():
                if img.name in kept_image_names:
                    new_images[img_id] = img
                    kept_image_ids.add(img_id)
            images = new_images

            new_points3D = {}
            for p_id, p in points3D.items():
                new_image_ids = []
                new_point2D_idxs = []
                for i, img_id in enumerate(p.image_ids):
                    if img_id in kept_image_ids:
                        new_image_ids.append(img_id)
                        new_point2D_idxs.append(p.point2D_idxs[i])
                if len(new_image_ids) > 0:
                    new_points3D[p_id] = p._replace(
                        image_ids=np.array(new_image_ids),
                        point2D_idxs=np.array(new_point2D_idxs),
                    )
            points3D = new_points3D
            print(f"[Parser] {len(points3D)} 3D points remaining after exclusion.")

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
        colmap_image_dir = os.path.join(data_dir, "overlays")
        if not os.path.exists(colmap_image_dir):
            print(f"[Parser] Overlays folder {colmap_image_dir} does not exist. Using images folder instead.")
            colmap_image_dir = os.path.join(data_dir, "images")

        if not os.path.exists(colmap_image_dir):
            raise ValueError(f"Image folder {colmap_image_dir} does not exist.")

        if self.load_images_in_memory:
            # If loading into memory, we will do resizing on the fly.
            # Paths should point to original images.
            image_dir = colmap_image_dir
        else:
            # If not loading into memory, check for pre-downsampled images and
            # create them if they don't exist.
            if factor > 1 and not self.extconf["no_factor_suffix"]:
                image_dir_suffix = f"_{factor}"
            else:
                image_dir_suffix = ""
            image_dir = colmap_image_dir + image_dir_suffix
            if not os.path.exists(image_dir):
                _resize_image_folder(colmap_image_dir, image_dir, factor=factor)

            # Check for JPGs in downsampled folder and convert to PNGs if needed.
            # This is a legacy holdover that ensures compatibility with datasets
            # that were processed with a pipeline that converted JPGs to PNGs.
            image_files_for_check = sorted(_get_rel_paths(image_dir))
            if (
                factor > 1
                and image_files_for_check
                and os.path.splitext(image_files_for_check[0])[1].lower() == ".jpg"
            ):
                print("Found JPGs in downsampled folder, converting to PNGs.")
                image_dir = _resize_image_folder(
                    colmap_image_dir, image_dir + "_png", factor=factor
                )

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # Load masks if requested.
        self.segmentation_mask_paths = None
        if use_masks:
            # When loading into memory, mask paths should point to original masks.
            mask_dir_suffix = "" if self.load_images_in_memory else image_dir_suffix
            mask_dir = os.path.join(data_dir, "masks" + mask_dir_suffix)
            if not os.path.exists(mask_dir):
                mask_dir = os.path.join(data_dir, "masks")
            if os.path.exists(mask_dir):
                print(f"[Parser] Loading masks from {mask_dir}")
                mask_files = sorted(_get_rel_paths(mask_dir))
                # Create mapping from image files to mask files
                colmap_to_mask = dict(zip(colmap_files, mask_files))
                self.segmentation_mask_paths = []
                for f in image_names:
                    if f in colmap_to_mask:
                        mask_path = os.path.join(mask_dir, colmap_to_mask[f])
                        if os.path.exists(mask_path) and os.path.splitext(mask_path)[1].lower() in {".png", ".jpg", ".jpeg"}:
                            self.segmentation_mask_paths.append(mask_path)
                        else:
                            self.segmentation_mask_paths.append(None)
                    else:
                        self.segmentation_mask_paths.append(None)
                print(f"[Parser] Found {sum(1 for p in self.segmentation_mask_paths if p is not None)} masks out of {len(image_names)} images")
            else:
                print(f"[Parser] Warning: use_masks=True but mask directory {mask_dir} does not exist.")
                print("[Parser] Fallback: Will use alpha channel from images as masks.")
                self.use_alpha_as_mask = True
                self.segmentation_mask_paths = [None] * len(image_names)
        
        # 3D points and {image_name -> [point_idx]}
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

        # Normalize the world space.
        if normalize:
            transform = orient_and_center(camtoworlds, points)
            points = transform_points(transform, points)
            camtoworlds = transform_cameras(transform, camtoworlds)
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

        # Create 0-based contiguous camera indices from COLMAP camera_ids.
        # This is useful for camera-based embeddings/modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Load EXIF exposure data if requested.
        # Always read from original (non-downscaled) images since PNG doesn't support EXIF.
        if load_exposure:
            exposure_paths = [Path(colmap_image_dir) / image_name for image_name in image_names]
            exposure_worker_count = _resolve_parallel_worker_count(len(exposure_paths))
            if exposure_worker_count > 1:
                print(f"[Parser] Loading EXIF exposure with {exposure_worker_count} workers...")
                with ThreadPoolExecutor(max_workers=exposure_worker_count) as executor:
                    exposure_values = list(
                        tqdm(
                            executor.map(compute_exposure_from_exif, exposure_paths),
                            total=len(exposure_paths),
                            desc="Loading EXIF exposure",
                        )
                    )
            else:
                exposure_values = [
                    compute_exposure_from_exif(original_path)
                    for original_path in tqdm(exposure_paths, desc="Loading EXIF exposure")
                ]

            # Compute mean across all valid exposures and subtract
            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                print(
                    f"[Parser] Loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )
            else:
                self.exposure_values = [None] * len(exposure_values)
                print("[Parser] No valid EXIF exposure data found in any image.")
        else:
            self.exposure_values = [None] * len(image_paths)

        # --- Start of new, simplified pipeline ---

        # 1. Initial Sanity Check and Scaling
        # This must happen before any distortion logic.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        # Get dimensions from COLMAP for the first image's camera
        first_camera_id = self.camera_ids[0]
        colmap_width, colmap_height = self.imsize_dict[first_camera_id]
        
        # Calculate scaling factors
        s_height = actual_height / colmap_height
        s_width = actual_width / colmap_width

        # If there's a significant mismatch, scale all camera parameters
        if abs(s_width - 1.0) > 1e-6 or abs(s_height - 1.0) > 1e-6:
            print(f"[Parser] Scaling camera intrinsics by ({s_width}, {s_height}) to match image dimensions.")
            for camera_id in self.Ks_dict.keys():
                K = self.Ks_dict[camera_id]
                K[0, :] *= s_width
                K[1, :] *= s_height
                self.Ks_dict[camera_id] = K

                width, height = self.imsize_dict[camera_id]
                self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))
        
        # 2. Undistortion (destructive update, as in original gsplat)
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue
            
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            camtype = self.camtype_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(K, params, (width, height), 0)
                mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (width, height), cv2.CV_32FC1)
                mask = None
            elif camtype == "fisheye":
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (1.0 + params[0] * theta**2 + params[1] * theta**4 + 
                     params[2] * theta**6 + params[3] * theta**8)
                mapx = (fx * x1 * r + cx).astype(np.float32)
                mapy = (fy * y1 * r + cy).astype(np.float32)

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
            
            # Destructively update the dictionaries with undistorted parameters
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.undistort_mask_dict[camera_id] = mask

        # 3. Foreground Optimization (The new, final step)
        self.foreground_bboxes = {}
        if self.optimize_foreground:
            if not self.use_masks:
                raise ValueError("optimize_foreground requires use_masks=True")

            print("[Parser] Calculating foreground crops...")
            new_Ks_dict, new_imsize_dict, new_camera_ids = {}, {}, []
            new_params_dict, new_camtype_dict, new_undistort_mask_dict = {}, {}, {}
            
            for i, image_name in enumerate(tqdm(image_names, desc="Calculating BBoxes")):
                original_camera_id = self.camera_ids[i]

                mask = None
                if self.segmentation_mask_paths is not None:
                    mask_path = self.segmentation_mask_paths[i]
                    if mask_path is not None and os.path.exists(mask_path):
                        mask = imageio.imread(mask_path)
                        if len(mask.shape) == 3: mask = mask[..., 0]

                if mask is None and self.use_alpha_as_mask:
                    original_image_path = os.path.join(self.data_dir, "images", image_names[i])
                    if not os.path.exists(original_image_path):
                         original_image_path = os.path.join(self.data_dir, "overlays", image_names[i])
                    if os.path.exists(original_image_path):
                        rgba_image = imageio.imread(original_image_path)
                        if rgba_image.shape[-1] == 4: mask = rgba_image[..., 3]
                
                # The camera parameters have been updated by undistortion, so imsize_dict has the post-undistortion size
                width, height = self.imsize_dict[original_camera_id]
                bbox_x, bbox_y, bbox_w, bbox_h = 0, 0, width, height

                if mask is not None:
                    # The mask must be transformed in the same way the final image will be
                    # 1. Scale to match COLMAP's reported image size (pre-sanity check)
                    # NOTE: This requires careful state management; for now, we assume mask has same initial size as images
                    
                    # 2. Undistort
                    if self.undistort_input and original_camera_id in self.mapx_dict:
                        mapx, mapy = self.mapx_dict[original_camera_id], self.mapy_dict[original_camera_id]
                        mask = cv2.remap(mask, mapx, mapy, cv2.INTER_LINEAR)
                        x, y, w, h = self.roi_undist_dict[original_camera_id]
                        mask = mask[y:y+h, x:x+w]
                    
                    # 3. Calculate bbox from this final, transformed mask
                    mask_for_bbox = mask.astype(np.float32) / 255.0
                    y_indices, x_indices = np.nonzero(mask_for_bbox > 0.1)
                    
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        y_min, y_max_inclusive = y_indices.min(), y_indices.max()
                        x_min, x_max_inclusive = x_indices.min(), x_indices.max()
                        y_max_exclusive, x_max_exclusive = y_max_inclusive + 1, x_max_inclusive + 1
                        
                        h_box, w_box = y_max_exclusive - y_min, x_max_exclusive - x_min
                        margin_y, margin_x = int(h_box * self.foreground_margin), int(w_box * self.foreground_margin)
                        
                        bbox_x = max(0, x_min - margin_x)
                        bbox_y = max(0, y_min - margin_y)
                        bbox_w = min(width, x_max_exclusive + margin_x) - bbox_x
                        bbox_h = min(height, y_max_exclusive + margin_y) - bbox_y

                self.foreground_bboxes[image_name] = (bbox_x, bbox_y, bbox_w, bbox_h)

                # Create the new per-image camera profile
                new_camera_id = i
                new_camera_ids.append(new_camera_id)
                
                K = self.Ks_dict[original_camera_id].copy()
                K[0, 2] -= bbox_x
                K[1, 2] -= bbox_y
                
                new_Ks_dict[new_camera_id] = K
                new_imsize_dict[new_camera_id] = (bbox_w, bbox_h)
                
                # Copy over other params
                new_params_dict[new_camera_id] = self.params_dict[original_camera_id]
                new_camtype_dict[new_camera_id] = self.camtype_dict[original_camera_id]
                new_undistort_mask_dict[new_camera_id] = self.undistort_mask_dict.get(original_camera_id)

            # Atomically replace the dictionaries
            self.Ks_dict = new_Ks_dict
            self.imsize_dict = new_imsize_dict
            self.camera_ids = new_camera_ids
            self.params_dict = new_params_dict
            self.camtype_dict = new_camtype_dict
            self.undistort_mask_dict = new_undistort_mask_dict

        # Keep a mapping from per-image IDs to their original camera ID
        self.image_to_original_camera_id = camera_ids # The original list before it was replaced

        # --- End of Major Refactoring ---

        # At this point, all camera parameters (Ks_dict, imsize_dict) and
        # transformation data (maps, rois, bboxes) are finalized.

        # scene size
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)

        # Load images into memory
        self.images_cpu_list: Optional[List[np.ndarray]] = None
        self.masks_cpu_list: Optional[List[Optional[np.ndarray]]] = None

        if self.load_images_in_memory:
            print(f"[Parser] Loading {len(self.image_paths)} images into memory...")
            image_worker_count = _resolve_parallel_worker_count(len(self.image_paths))
            if image_worker_count > 1:
                print(f"[Parser] Preprocessing images with {image_worker_count} workers...")
                with ThreadPoolExecutor(max_workers=image_worker_count) as executor:
                    processed_images = list(
                        tqdm(
                            executor.map(self._load_and_process_image, range(len(self.image_paths))),
                            total=len(self.image_paths),
                            desc="Loading images",
                        )
                    )
            else:
                processed_images = [
                    self._load_and_process_image(i)
                    for i in tqdm(range(len(self.image_paths)), desc="Loading images")
                ]

            self.images_cpu_list = [image.astype(np.uint8, copy=False) for image, _, _ in processed_images]
            processed_masks = [
                segmentation_mask.astype(np.uint8, copy=False)
                if segmentation_mask is not None
                else None
                for _, segmentation_mask, _ in processed_images
            ]
            self.masks_cpu_list = processed_masks if any(mask is not None for mask in processed_masks) else None

            for camera_id, (_, _, undistort_mask) in zip(self.camera_ids, processed_images):
                self.undistort_mask_dict[camera_id] = undistort_mask

            # Convert arrays to monolithic tensors where possible.
            self.camtoworlds = torch.from_numpy(self.camtoworlds).float()
            # For Ks and params, they are now dictionaries with per-image IDs,
            # so we create tensors from them based on the image order.
            self.all_Ks_cpu = torch.from_numpy(np.array([self.Ks_dict[self.camera_ids[i]] for i in range(len(self.image_names))])).float()
            self.all_distortion_params_cpu = torch.from_numpy(np.array([self.params_dict.get(self.image_to_original_camera_id[i], self.params_dict.get(self.camera_ids[i])) for i in range(len(self.image_names))])).float()
        
        # Clean up cache
        del self._temp_image_cache

    def _load_and_process_image(
        self, index: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        image_name = self.image_names[index]
        if image_name in self._temp_image_cache:
            full_image = self._temp_image_cache[image_name]
        else:
            full_image = imageio.imread(self.image_paths[index])
        return self._process_image_and_mask(index, full_image)

    def _process_image_and_mask(
        self, index: int, full_image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Helper to apply the full processing pipeline to an image and its mask."""
        image_name = self.image_names[index]
        # Use the new per-image camera ID for most things
        camera_id = self.camera_ids[index]
        # But use the original camera ID to look up distortion recipes
        original_camera_id = self.image_to_original_camera_id[index]
        
        image = full_image[..., :3]

        # Load segmentation mask
        segmentation_mask = None
        if self.use_masks:
            if self.segmentation_mask_paths is not None:
                mask_path = self.segmentation_mask_paths[index]
                if mask_path is not None and os.path.exists(mask_path):
                    segmentation_mask = imageio.imread(mask_path)
                    if len(segmentation_mask.shape) == 3:
                        segmentation_mask = segmentation_mask[..., 0]
            if segmentation_mask is None and self.use_alpha_as_mask:
                if full_image.shape[-1] == 4:
                    segmentation_mask = full_image[..., 3]

        # Get undistortion mask (for fisheye)
        undistort_mask = self.undistort_mask_dict.get(camera_id)

        # 1. Downsample image and mask
        if self.factor > 1:
            image = resize_image(image, self.factor)
            if segmentation_mask is not None:
                segmentation_mask = resize_mask(
                    segmentation_mask, self.factor
                )

        # 2. Undistort
        original_camera_id = self.image_to_original_camera_id[index]
        if self.undistort_input and original_camera_id in self.mapx_dict:
            mapx, mapy = self.mapx_dict[original_camera_id], self.mapy_dict[original_camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            if segmentation_mask is not None:
                segmentation_mask = cv2.remap(
                    segmentation_mask, mapx, mapy, cv2.INTER_LINEAR
                )
            x, y, w, h = self.roi_undist_dict[original_camera_id]
            image = image[y : y + h, x : x + w]
            if segmentation_mask is not None:
                segmentation_mask = segmentation_mask[y : y + h, x : x + w]
            if undistort_mask is not None:
                undistort_mask = undistort_mask[y : y + h, x : x + w]

        # 3. Crop to foreground
        if self.optimize_foreground and image_name in self.foreground_bboxes:
            x, y, w, h = self.foreground_bboxes[image_name]
            image = image[y : y + h, x : x + w]
            if segmentation_mask is not None:
                segmentation_mask = segmentation_mask[y : y + h, x : x + w]
            if undistort_mask is not None:
                undistort_mask = undistort_mask[y : y + h, x : x + w]
        
        return image, segmentation_mask, undistort_mask

class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        device: Optional[str] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.on_gpu = False
        indices = np.arange(len(self.parser.image_names))

        if self.parser.test_every < 1:
            # If test_every < 1, put all images in trainset and none in val/test
            if split == "train":
                self.indices = indices  # all images
            else:
                self.indices = np.array([], dtype=np.int64)  # no images for val/test
        else:
            # Normal behavior: split based on test_every
            if split == "train":
                self.indices = indices[indices % self.parser.test_every != 0]
            else:
                self.indices = indices[indices % self.parser.test_every == 0]
        
        if self.parser.load_images_to_gpu:
            if not self.parser.load_images_in_memory:
                raise ValueError("load_images_to_gpu requires load_images_in_memory to be True")
            assert device is not None, "Device must be provided when loading images to GPU"
            
            print(f"[Dataset] Moving dataset for split '{split}' to device: {device}")
            
            # Move data to GPU, keeping them as lists of tensors if shapes vary
            self.images_gpu_list = [torch.from_numpy(self.parser.images_cpu_list[i]).to(device) for i in self.indices]
            if self.parser.masks_cpu_list is not None:
                self.masks_gpu_list = [
                    torch.from_numpy(mask).to(device) if mask is not None else None
                    for mask in (self.parser.masks_cpu_list[i] for i in self.indices)
                ]
            else:
                self.masks_gpu_list = None

            self.camtoworlds = self.parser.camtoworlds[self.indices].to(device)
            self.Ks = self.parser.all_Ks_cpu[self.indices].to(device)
            self.distortion_params = self.parser.all_distortion_params_cpu[self.indices].to(device)
            
            self.undistort_mask_dict = {k: torch.from_numpy(v).bool().to(device) for k, v in self.parser.undistort_mask_dict.items() if v is not None}
            self.camera_types = self.parser.camtype_dict
            self.image_ids = self.parser.camera_ids
            
            self.on_gpu = True

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        
        if self.on_gpu:
            # Fast path: data is already on the GPU
            original_index = self.indices[item]
            image_name = self.parser.image_names[original_index]
            camera_id = self.image_ids[original_index]

            image = self.images_gpu_list[item].float() / 255.0
            K = self.Ks[item]
            camtoworlds = self.camtoworlds[item]

            data = {
                "K": K,
                "camtoworld": camtoworlds,
                "image": image,
                # Keep image_id split-local so trainset-sized per-frame modules
                # can index into embeddings and optimizer tables safely.
                "image_id": item,
                "parser_index": original_index,
                "image_name": image_name,
                "distortion_params": self.distortion_params[item],
                "camera_type": self.camera_types[camera_id],
                "camera_idx": self.parser.camera_indices[original_index],
            }
            if self.masks_gpu_list is not None:
                segmentation_mask = self.masks_gpu_list[item]
                if segmentation_mask is not None:
                    data["segmentation_mask"] = segmentation_mask.float() / 255.0
            if camera_id in self.undistort_mask_dict:
                data["undistort_mask"] = self.undistort_mask_dict[camera_id]

            # Add exposure if available
            exposure = self.parser.exposure_values[original_index]
            if exposure is not None:
                data["exposure"] = torch.tensor(exposure, dtype=torch.float32, device=image.device)

            # Patch size is not supported with GPU loading for simplicity
            if self.patch_size is not None:
                 raise NotImplementedError("patch_size is not supported when load_images_to_gpu is True.")

            return data

        # Slower path for CPU-based data (disk or pre-loaded to RAM)
        index = self.indices[item]
        image_name = self.parser.image_names[index]
        camera_id = self.parser.camera_ids[index]

        if self.parser.load_images_in_memory:
            # Slice from pre-loaded lists of numpy arrays
            image = torch.from_numpy(self.parser.images_cpu_list[index]).float() / 255.0
            segmentation_mask = None
            if self.parser.masks_cpu_list is not None:
                mask_np = self.parser.masks_cpu_list[index]
                if mask_np is not None:
                    segmentation_mask = torch.from_numpy(mask_np).float() / 255.0
            K = self.parser.all_Ks_cpu[index].clone()
            camtoworlds = self.parser.camtoworlds[index].clone()
            undistort_mask = self.parser.undistort_mask_dict.get(camera_id)
        else:
            image_np, segmentation_mask_np, undistort_mask = self.parser._load_and_process_image(index)
            image = torch.from_numpy(image_np).float() / 255.0
            segmentation_mask = None
            if segmentation_mask_np is not None:
                segmentation_mask = torch.from_numpy(segmentation_mask_np).float() / 255.0
            K = self.parser.Ks_dict[camera_id].copy() # Use camera_id for dict
            camtoworlds = self.parser.camtoworlds[index] # Use index for numpy array

        # Use original camera ID to get original distortion params
        original_camera_id = self.parser.image_to_original_camera_id[index]
        if self.parser.load_images_in_memory:
            params = self.parser.all_distortion_params_cpu[index]
        else:
            params = self.parser.params_dict.get(original_camera_id, self.parser.params_dict.get(camera_id))

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if segmentation_mask is not None:
                segmentation_mask = segmentation_mask[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        if not self.parser.load_images_in_memory:
            K = torch.from_numpy(K).float()
            camtoworlds = torch.from_numpy(camtoworlds).float()

        data = {
            "K": K,
            "camtoworld": camtoworlds,
            "image": image,
            # Keep image_id split-local so trainset-sized per-frame modules
            # can index into embeddings and optimizer tables safely.
            "image_id": item,
            "parser_index": index,
            "image_name": image_name,
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }

        # Add undistortion mask if it exists (for fisheye cameras)
        if undistort_mask is not None:
            data["undistort_mask"] = torch.from_numpy(undistort_mask).bool()

        # Add segmentation mask if it exists (from file)
        if segmentation_mask is not None:
            data["segmentation_mask"] = segmentation_mask

        # Add distortion parameters and camera type to the data
        if self.parser.load_images_in_memory:
             data["distortion_params"] = params
        else:
             data["distortion_params"] = torch.from_numpy(params).float()
        data["camera_type"] = self.parser.camtype_dict[camera_id]

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        if self.load_depths:
            # projected points to image plane to get depths
            camtoworlds_np = camtoworlds.numpy() if isinstance(camtoworlds, torch.Tensor) else camtoworlds
            K_np = K.numpy() if isinstance(K, torch.Tensor) else K
            worldtocams = np.linalg.inv(camtoworlds_np)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K_np @ points_cam.T).T
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

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--use_masks", action="store_true", help="Load masks from masks directory")
    parser.add_argument("--optimize_foreground", action="store_true", help="Optimize foreground by cropping")
    parser.add_argument("--load_images_in_memory", action="store_true", help="Load all images into memory")
    parser.add_argument("--load_images_to_gpu", action="store_true", help="Load images to GPU")
    parser.add_argument("--exclude_prefixes", type=str, nargs="+", default=[], help="Prefixes of images to exclude.")
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
        load_images_to_gpu=args.load_images_to_gpu,
        exclude_prefixes=args.exclude_prefixes,
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
