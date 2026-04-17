"""
Per-frame multi-view dataset shaped like
github.com/yindaheng98/TrackerSplat/blob/main/trackersplat/dataset/dataset.py

The reference paper organises data as:
    VideoCameraDataset[frame_idx]            -> FrameCameraDataset
    VideoCameraDataset[frame_idx][cam_idx]   -> Camera
    VideoCameraDataset[start:end]            -> VideoCameraDataset (slice of frames)
    VideoCameraDataset[frame_idx, cam_idx]   -> Camera

Our recordings live as a *single* COLMAP workspace with images stored as
    <data_dir>/images/<cam_name>/<frame_idx:06d>.jpg
which is the inverse of the reference layout (frames-as-folders). We adapt
by parsing the COLMAP rig once for camera intrinsics + extrinsics, then
emitting one DatasetCameraMeta per (cam, frame) pair that varies only the
image_path and frame_idx — perfectly matching the "fixed-view" assumption
of the reference's FixedViewColmapVideoCameraDataset.

This module pulls no external dependencies beyond what trackersplat_trainer_fastgs
already imports (datasets.colmap.Parser, gsplat). FasterGS-compatible.

Usage:
    from trackersplat_dataset import build_thenewface_video_dataset
    video = build_thenewface_video_dataset(
        data_dir="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/",
        n_frames=50, frame_step=6, data_factor=4,
    )
    print(len(video), "frames", len(video[0]), "cameras")
    cam = video[0, 5]                 # frame 0, camera 5
    img = cam.load_image()            # (H, W, 3) float in [0, 1]
    K, R, T = cam.K, cam.R, cam.T     # intrinsics + extrinsics
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Camera object — minimal, FasterGS-compatible, no external 3DGS deps
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    """A single rendered/captured view at a specific time-step.

    Conventions mirror the reference impl's Camera:
      - R (3, 3) world-to-camera rotation (OpenCV convention)
      - T (3,) world-to-camera translation
      - K (3, 3) intrinsics in pixel units at the camera's image_height/width
      - FoVx / FoVy in radians (derived from K and image size)
    Stored as torch tensors on `device`.
    """
    R: Tensor                       # (3, 3) world->cam rotation
    T: Tensor                       # (3,)   world->cam translation
    K: Tensor                       # (3, 3) intrinsics
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    image_path: str
    image_mask_path: str = ""
    depth_path: str = ""
    depth_mask_path: str = ""
    frame_idx: int = 0
    cam_name: str = ""
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    # ---- derived getters (no extra storage) ----
    @property
    def w2c(self) -> Tensor:
        """4x4 world-to-camera matrix on self.device."""
        m = torch.eye(4, device=self.device, dtype=self.R.dtype)
        m[:3, :3] = self.R
        m[:3, 3] = self.T
        return m

    @property
    def c2w(self) -> Tensor:
        return torch.linalg.inv(self.w2c)

    @property
    def cam_position(self) -> Tensor:
        """Camera centre in world space."""
        return self.c2w[:3, 3]

    @property
    def focal_x(self) -> float: return float(self.K[0, 0])
    @property
    def focal_y(self) -> float: return float(self.K[1, 1])
    @property
    def center_x(self) -> float: return float(self.K[0, 2])
    @property
    def center_y(self) -> float: return float(self.K[1, 2])

    # ---- IO ----
    def load_image(self) -> Tensor:
        """Load and return (image_height, image_width, 3) float tensor in [0, 1]
        on self.device. The on-disk file may be at full resolution; we stride-
        decimate to match the camera's intrinsic resolution (which the parser
        has already scaled by data_factor)."""
        img = imageio.imread(self.image_path)
        H_full, W_full = img.shape[:2]
        # Stride decimation matches what trackersplat_trainer_fastgs does.
        sx = max(1, round(W_full / self.image_width))
        sy = max(1, round(H_full / self.image_height))
        if sx > 1 or sy > 1:
            img = img[::sy, ::sx]
        return torch.from_numpy(img).float().to(self.device) / 255.0

    def load_mask(self) -> Optional[Tensor]:
        if not self.image_mask_path or not os.path.exists(self.image_mask_path):
            return None
        m = imageio.imread(self.image_mask_path)
        return torch.from_numpy(m).bool().to(self.device)

    def to(self, device) -> "Camera":
        return Camera(
            R=self.R.to(device), T=self.T.to(device), K=self.K.to(device),
            image_height=self.image_height, image_width=self.image_width,
            FoVx=self.FoVx, FoVy=self.FoVy,
            image_path=self.image_path, image_mask_path=self.image_mask_path,
            depth_path=self.depth_path, depth_mask_path=self.depth_mask_path,
            frame_idx=self.frame_idx, cam_name=self.cam_name, device=device,
        )

    def __repr__(self):
        return (f"Camera(cam={self.cam_name!r} frame={self.frame_idx} "
                f"{self.image_width}x{self.image_height} "
                f"f=({self.focal_x:.1f},{self.focal_y:.1f}))")


# ---------------------------------------------------------------------------
# Reference-shaped meta tuple — drop-in equivalent
# ---------------------------------------------------------------------------
class DatasetCameraMeta(NamedTuple):
    """Same field set as
    github.com/yindaheng98/TrackerSplat/.../dataset.py::DatasetCameraMeta."""
    image_height: int
    image_width: int
    FoVx: float
    FoVy: float
    R: Tensor               # world->cam rotation, (3, 3)
    T: Tensor               # world->cam translation, (3,)
    image_path: str
    image_mask_path: str
    depth_path: str
    depth_mask_path: str
    K: Tensor               # extra: intrinsics (we keep them pre-baked)
    cam_name: str           # extra: directory name
    frame_idx: int

    def build_camera(self, device=torch.device("cuda")) -> Camera:
        return Camera(
            R=self.R.to(device), T=self.T.to(device), K=self.K.to(device),
            image_height=self.image_height, image_width=self.image_width,
            FoVx=self.FoVx, FoVy=self.FoVy,
            image_path=self.image_path, image_mask_path=self.image_mask_path,
            depth_path=self.depth_path, depth_mask_path=self.depth_mask_path,
            frame_idx=self.frame_idx, cam_name=self.cam_name, device=device,
        )


# ---------------------------------------------------------------------------
# FrameCameraDataset / VideoCameraDataset
# ---------------------------------------------------------------------------
class FrameCameraDataset(Dataset):
    """All cameras for a single time-step. Mirrors the reference class."""
    def __init__(self, metas: Sequence[DatasetCameraMeta],
                 device=torch.device("cuda")):
        super().__init__()
        self._metas = [DatasetCameraMeta(**m._asdict()) for m in metas]
        self.to(device)

    def to(self, device) -> "FrameCameraDataset":
        self.device = device
        self.cameras: List[Camera] = [m.build_camera(device=device) for m in self._metas]
        return self

    def __getitem__(self, idx) -> Camera:
        return self.cameras[idx]

    def __len__(self) -> int:
        return len(self.cameras)


MetaFrame = List[DatasetCameraMeta]


class VideoCameraDataset(Dataset):
    """A list of frames, each frame is a list of fixed-pose cameras.

    Indexing semantics match the reference VideoCameraDataset:
      video[t]              FrameCameraDataset (all cams at frame t)
      video[t, c]           Camera             (frame t, cam c)
      video[a:b]            VideoCameraDataset (slice of frames)
      video[t, a:b]         FrameCameraDataset (frame t, cams a:b)
    """
    def __init__(self, frames: List[MetaFrame],
                 device=torch.device("cuda")):
        super().__init__()
        self._framemetas: List[MetaFrame] = [
            [DatasetCameraMeta(**c._asdict()) for c in frame] for frame in frames
        ]
        self.to(device)

    def to(self, device) -> "VideoCameraDataset":
        self.device = device
        return self

    def __len__(self) -> int:
        return len(self._framemetas)

    def get_metas(self) -> List[MetaFrame]:
        return [[DatasetCameraMeta(**c._asdict()) for c in frame]
                for frame in self._framemetas]

    def __getitem__(self, idx) -> Union["VideoCameraDataset", FrameCameraDataset, Camera]:
        if isinstance(idx, tuple) and len(idx) == 1:
            idx = idx[0]
        if isinstance(idx, int):
            return FrameCameraDataset(self._framemetas[idx], device=self.device)
        if isinstance(idx, slice) or isinstance(idx, list):
            return VideoCameraDataset(self._framemetas[idx], device=self.device)
        if isinstance(idx, tuple) and len(idx) == 2 and isinstance(idx[0], int):
            frame = self._framemetas[idx[0]]
            if isinstance(idx[1], int):
                return frame[idx[1]].build_camera(device=self.device)
            if isinstance(idx[1], slice) or isinstance(idx[1], list):
                return FrameCameraDataset(frame[idx[1]], device=self.device)
        raise ValueError(f"VideoCameraDataset: invalid index {idx!r}")


# ---------------------------------------------------------------------------
# Builder for *our* data layout (cameras-as-folders, frames-as-filenames)
# ---------------------------------------------------------------------------
def _fov_from_K_and_size(K: np.ndarray, W: int, H: int) -> Tuple[float, float]:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    return 2.0 * math.atan(W / (2.0 * fx)), 2.0 * math.atan(H / (2.0 * fy))


def build_thenewface_video_dataset(
    data_dir: str,
    n_frames: int = 50,
    frame_step: int = 6,
    data_factor: int = 4,
    normalize: bool = False,
    skip_points3d: bool = True,
    image_subdir: str = "images",
    image_ext: str = "jpg",
    device: Union[str, torch.device] = "cuda",
) -> VideoCameraDataset:
    """Adapter for our project's data layout.

    Layout assumed:  <data_dir>/<image_subdir>/<cam_name>/<frame:06d>.<ext>
    plus a single COLMAP workspace under <data_dir>/sparse/0/ that defines
    the (fixed) intrinsics + extrinsics for every camera.

    Frame indexing matches the trackersplat_trainer_fastgs convention:
        image_num = frame_idx * frame_step + 1     (so frame_idx=0 → 000001.jpg)
    """
    # We rely on the existing gsplat colmap parser for intrinsics+extrinsics.
    # Import here so this module stays optional — only needed at build time.
    from datasets.colmap import Parser

    device = torch.device(device)
    parser = Parser(
        data_dir=data_dir, factor=data_factor,
        normalize=normalize, test_every=9999,
        frame_num=1, skip_points3d=skip_points3d,
    )
    n_cams = len(parser.image_names)

    # Pre-build per-camera intrinsics + extrinsics tensors (shared across frames).
    cam_records = []
    for ci in range(n_cams):
        cam_id = parser.camera_ids[ci]
        K_np   = np.asarray(parser.Ks_dict[cam_id], dtype=np.float32)
        W, H   = parser.imsize_dict[cam_id]
        c2w_np = np.asarray(parser.camtoworlds[ci], dtype=np.float32)
        w2c    = np.linalg.inv(c2w_np)
        R = torch.from_numpy(w2c[:3, :3].copy())
        T = torch.from_numpy(w2c[:3, 3].copy())
        K = torch.from_numpy(K_np)
        FoVx, FoVy = _fov_from_K_and_size(K_np, W, H)
        cam_name = os.path.dirname(parser.image_names[ci])
        cam_records.append(dict(
            R=R, T=T, K=K, W=int(W), H=int(H), FoVx=FoVx, FoVy=FoVy,
            cam_name=cam_name,
        ))

    # Enumerate frames; build per-frame metas with only image_path + frame_idx changing.
    frames: List[MetaFrame] = []
    for fi in range(n_frames):
        image_num = fi * frame_step + 1
        metas: MetaFrame = []
        for rec in cam_records:
            img_path = os.path.join(
                data_dir, image_subdir, rec["cam_name"],
                f"{image_num:06d}.{image_ext}",
            )
            metas.append(DatasetCameraMeta(
                image_height=rec["H"], image_width=rec["W"],
                FoVx=rec["FoVx"], FoVy=rec["FoVy"],
                R=rec["R"], T=rec["T"],
                image_path=img_path, image_mask_path="",
                depth_path="", depth_mask_path="",
                K=rec["K"], cam_name=rec["cam_name"],
                frame_idx=fi,
            ))
        frames.append(metas)

    return VideoCameraDataset(frames, device=device)


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------
def _selftest():
    """Run as a script: python examples/trackersplat_dataset.py"""
    DATA = "/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/"
    video = build_thenewface_video_dataset(
        data_dir=DATA, n_frames=5, frame_step=6, data_factor=4,
    )
    print(f"VideoCameraDataset: {len(video)} frames, "
          f"{len(video[0])} cameras per frame")
    print(f"  video[0]      → {type(video[0]).__name__}")
    print(f"  video[0, 0]   → {video[0, 0]}")
    print(f"  video[1:3]    → VideoCameraDataset(len={len(video[1:3])})")
    cam = video[0, 0]
    img = cam.load_image()
    print(f"  loaded image  → shape {tuple(img.shape)}  range "
          f"[{img.min():.3f}, {img.max():.3f}]")
    # Verify fixed-view invariant: same R/T across frames for the same cam idx.
    cam_b = video[3, 0]
    assert torch.allclose(cam.R, cam_b.R) and torch.allclose(cam.T, cam_b.T), \
        "expected R/T to match across frames for the same camera index"
    print("  fixed-view invariant ✓ (R/T identical across frames per cam)")


if __name__ == "__main__":
    _selftest()
