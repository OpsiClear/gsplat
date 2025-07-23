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
        self._temp_image_cache = {}

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
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
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

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        if not self.optimize_foreground:
            actual_image = imageio.imread(self.image_paths[0])[..., :3]
            actual_height, actual_width = actual_image.shape[:2]
            colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
            s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
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

                # 4. Final conversion
                image = image.astype(np.float32) / 255.0
                self.images_dict[image_name] = torch.from_numpy(image.copy()).float()
                if segmentation_mask is not None:
                    segmentation_mask = segmentation_mask.astype(np.float32) / 255.0
                    self.masks_dict[image_name] = torch.from_numpy(
                        segmentation_mask.copy()
                    ).float()
        
        # Clean up cache
        del self._temp_image_cache

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
            image = full_image[..., :3].astype(np.float32) / 255.0
            params = self.parser.params_dict[camera_id]

            # Load segmentation mask from file if use_masks is enabled and mask path exists
            segmentation_mask = None
            if self.parser.use_masks:
                if self.parser.segmentation_mask_paths is not None:
                    mask_path = self.parser.segmentation_mask_paths[index]
                    if mask_path is not None and os.path.exists(mask_path):
                        segmentation_mask = imageio.imread(mask_path).astype(np.float32) / 255.0
                        if len(segmentation_mask.shape) == 3:
                            segmentation_mask = segmentation_mask[..., 0]  # use first channel if RGB
                if segmentation_mask is None and self.parser.use_alpha_as_mask:
                    if full_image.shape[-1] == 4:
                        segmentation_mask = (full_image[..., 3]).astype(np.float32) / 255.0


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
            image = torch.from_numpy(image).float()
            if segmentation_mask is not None:
                segmentation_mask = torch.from_numpy(segmentation_mask).float()

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": image,
            "image_id": item,  # the index of the image in the dataset
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

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
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

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    parser.add_argument("--use_masks", action="store_true", help="Load masks from masks directory")
    parser.add_argument("--optimize_foreground", action="store_true", help="Optimize foreground by cropping")
    parser.add_argument("--load_images_in_memory", action="store_true", help="Load all images into memory")
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
