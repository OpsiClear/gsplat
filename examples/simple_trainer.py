import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
)

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.compression.sort import sort_splats
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.utils import log_transform
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap
import kornia

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "/data/shared/datasets/dtu_2dgs/scan83"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "/data/shared/aly/scan83/"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000,20_000, 30_000, 50_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000,20_000, 30_000, 50_000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = True
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000, 50_000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False
    # Whether to render all training views
    render_train_views: bool = False

    # Whether to pre-load all images into memory
    load_images_in_memory: bool = False
    # Whether to optimize by cropping to foreground bounding box
    optimize_foreground: bool = False
    # Margin for foreground optimization
    foreground_margin: float = 0.1

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    # Whether use fused-bilateral grid
    use_fused_bilagrid: bool = False

    # Whether to undistort COLMAP input (disable for fisheye with 3DGUT)
    undistort_colmap_input: bool = True

    # Whether to use masks
    use_masks: bool = False

    # Enable erank loss. (experimental)
    use_erank_loss: bool = False
    # Start step for erank loss
    erank_start_step: int = 7000

    # Whether to presort the splats at initialization
    use_sort: bool = False
    sort_kernel_size: int = 5
    sort_sigma: float = 3.0
    sort_lambda: float = 1.0
    
    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    use_sort: bool = False,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    if use_sort:
        print("Pre-sorting splats at initialization...")
        # 1. Group initial data into a dictionary
        N_total = points.shape[0]
        splat_tensors = {
            "means": points,
            "scales": scales,
            "opacities": torch.logit(torch.full((N_total,), init_opacity)),
            "quats": torch.rand((N_total, 4)),
            "sh0": rgb_to_sh(rgbs),
            "rgbs": rgbs,  # Also carry rgbs for feature_dim case
        }

        # 2. Crop to the nearest perfect square number
        n_gs = N_total
        n_sidelen = int(n_gs**0.5)
        n_square = n_sidelen**2
        if n_gs != n_square:
            print(
                f"Cropping splats from {n_gs} to {n_square} to make it a perfect square."
            )
            # We can just truncate, as initial opacities are identical
            for k, v in splat_tensors.items():
                splat_tensors[k] = v[:n_square]

        # 3. Prepare a temporary copy for sorting
        splats_for_sorting = {k: v.clone() for k, v in splat_tensors.items() if k != "rgbs"}
        splats_for_sorting["means"] = log_transform(splats_for_sorting["means"])
        splats_for_sorting["quats"] = F.normalize(
            splats_for_sorting["quats"], dim=-1
        )

        # 4. Get the sorting indices
        _, sort_indices = sort_splats(splats_for_sorting, verbose=False)

        # 5. Apply sorting to original tensors
        for k, v in splat_tensors.items():
            splat_tensors[k] = v[sort_indices]

        # 6. Unpack the sorted tensors to be used by the rest of the function
        points = splat_tensors["means"]
        scales = splat_tensors["scales"]
        opacities = splat_tensors["opacities"]  # Already in logit scale
        quats = splat_tensors["quats"]
        sh0s = splat_tensors["sh0"]
        rgbs = splat_tensors["rgbs"]
    else:
        # Original path
        opacities = torch.logit(torch.full((points.shape[0],), init_opacity))
        quats = torch.rand((points.shape[0], 4))
        sh0s = rgb_to_sh(rgbs)

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    scales = scales[world_rank::world_size]
    quats = quats[world_rank::world_size]
    opacities = opacities[world_rank::world_size]
    sh0s = sh0s[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]

    N = points.shape[0]
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = sh0s
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def erode_masks(masks, kernel_size=3, iterations=1):
    """Apply erosion to masks using max pooling with negative values."""
    import torch.nn.functional as F

    # Create padding to maintain size
    padding = kernel_size // 2
    eroded = masks

    for _ in range(iterations):
        # Invert mask (1->0, 0->1), apply max pooling, then invert back
        inverted = 1 - eroded
        pooled = F.max_pool2d(inverted.float(), kernel_size=kernel_size, stride=1, padding=padding)
        eroded = 1 - pooled

    return eroded

class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        if len(cfg.save_steps) > 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        if cfg.test_every > 0 and len(cfg.eval_steps) > 0:
            os.makedirs(self.render_dir, exist_ok=True)
        if len(cfg.ply_steps) > 0:
            self.ply_dir = f"{cfg.result_dir}/ply"
            os.makedirs(self.ply_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            undistort_input=cfg.undistort_colmap_input,
            use_masks=cfg.use_masks,
            load_images_in_memory=cfg.load_images_in_memory,
            optimize_foreground=cfg.optimize_foreground,
            foreground_margin=cfg.foreground_margin,
        )
        
        # Auto-detect fisheye cameras and adjust config accordingly
        if not cfg.undistort_colmap_input and hasattr(self.parser, 'camtype_dict'):
            # Check if any camera is fisheye
            is_fisheye = any(camtype == "fisheye" for camtype in self.parser.camtype_dict.values())
            if is_fisheye:
                print("Fisheye cameras detected. Setting camera_model to 'fisheye' and enabling 3DGUT flags.")
                cfg.camera_model = "fisheye"
                cfg.with_ut = True
                cfg.with_eval3d = True
        
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            use_sort=cfg.use_sort,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Unify sorting flag
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.cfg.strategy.sort = self.cfg.use_sort

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
        camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device)  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            undistort_masks = data["undistort_mask"].to(device) if "undistort_mask" in data else None  # [1, H, W]
            segmentation_masks = data["segmentation_mask"].to(device) if "segmentation_mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            # Extract distortion parameters if provided by the dataset
            distortion_params = data.get("distortion_params", None)
            if distortion_params is not None:
                distortion_params = distortion_params.to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Prepare distortion coefficients for rasterization
            radial_coeffs_to_pass = None
            tangential_coeffs_to_pass = None
            # thin_prism_coeffs_to_pass = None  # Not typically used in COLMAP

            if not cfg.undistort_colmap_input and distortion_params is not None:
                if cfg.camera_model == "fisheye":
                    # For OPENCV_FISHEYE, COLMAP params are [k1, k2, k3, k4]
                    # The rasterization function expects fisheye radial_coeffs as [batch_size, 4]
                    if distortion_params.shape[-1] == 4:
                        radial_coeffs_to_pass = distortion_params  # Should be [batch_size, 4]
                    else:
                        print(f"Warning: Fisheye model expects 4 distortion params, got {distortion_params.shape[-1]}")
                elif cfg.camera_model == "pinhole":
                    # For pinhole with distortion (OPENCV, RADIAL, SIMPLE_RADIAL)
                    # rasterization expects radial_coeffs [..., C, 6] and tangential_coeffs [..., C, 2]
                    num_params = distortion_params.shape[-1]
                    if num_params >= 1:
                        # Prepare radial coefficients (pad to 6 elements)
                        rad_params = torch.zeros(distortion_params.shape[0], 6, device=distortion_params.device)
                        
                        if num_params == 4:  # OPENCV: [k1, k2, p1, p2]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                            rad_params[:, 1] = distortion_params[:, 1]  # k2
                            # k3-k6 remain zero
                            tangential_coeffs_to_pass = distortion_params[:, [2, 3]].unsqueeze(0)  # [1, C, 2] p1, p2
                        elif num_params == 2:  # RADIAL: [k1, k2, 0, 0] -> extract [k1, k2]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                            rad_params[:, 1] = distortion_params[:, 1]  # k2
                        elif num_params == 1:  # SIMPLE_RADIAL: [k1, 0, 0, 0] -> extract [k1]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                        else:
                            print(f"Warning: Unexpected number of distortion parameters: {num_params}")
                            
                        radial_coeffs_to_pass = rad_params.unsqueeze(0)  # [1, C, 6]

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=undistort_masks,
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_masks and segmentation_masks is not None:
                colors[segmentation_masks<0.5] = 0.0
                pixels[segmentation_masks<0.5] = 0.0

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            if segmentation_masks is not None and cfg.use_masks:
                segmentation_loss = torch.sum(alphas * (1.0 - segmentation_masks.unsqueeze(-1))) / ((1.0 - segmentation_masks).sum())
                eroded_masks = erode_masks(segmentation_masks, kernel_size=3, iterations=1)
                foreground_loss = 0.1 * torch.sum((1.0 - alphas) * eroded_masks.unsqueeze(-1)) / eroded_masks.sum()
                loss += segmentation_loss + foreground_loss

            if self.cfg.sort_lambda > 0.0 and cfg.use_sort:
                sort_loss_val = self.sort_loss()
                loss += sort_loss_val

            # erank loss
            if cfg.use_erank_loss and step > cfg.erank_start_step:
                lambda_erank = 0.05
                original_scales = torch.exp(self.splats["scales"])
                s = original_scales * original_scales
                S = torch.sum(s, dim=-1)
                q = torch.div(s, S.unsqueeze(dim=-1))
                H = -torch.sum(q * torch.log(q + 1e-8), dim=-1)
                erank = torch.exp(H)
                erank_loss = torch.sum(
                    lambda_erank * torch.maximum(-torch.log(erank - 1 + 1e-5), torch.zeros_like(erank))
                    + torch.min(original_scales, dim=-1)[0]
                )
                loss += erank_loss

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.use_masks:
                    self.writer.add_scalar("train/segmentation_loss", segmentation_loss.item(), step)
                if cfg.use_erank_loss and step > cfg.erank_start_step:
                    self.writer.add_scalar("train/erank_loss", erank_loss.item(), step)
                if cfg.sort_lambda > 0.0 and cfg.use_sort:
                    self.writer.add_scalar("train/sort_loss", sort_loss_val.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:

                if self.cfg.app_opt:
                    # eval at origin to bake the appeareance into the colors
                    rgb = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb = rgb + self.splats["colors"]
                    rgb = torch.sigmoid(rgb).squeeze(0).unsqueeze(1)
                    sh0 = rgb_to_sh(rgb)
                    shN = torch.empty([sh0.shape[0], 0, 3], device=sh0.device)
                else:
                    sh0 = self.splats["sh0"]
                    shN = self.splats["shN"]

                means = self.splats["means"]
                scales = self.splats["scales"]
                quats = self.splats["quats"]
                opacities = self.splats["opacities"]
                export_splats(
                    means=means,
                    scales=scales,
                    quats=quats,
                    opacities=opacities,
                    sh0=sh0,
                    shN=shN,
                    format="ply",
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # self.render_traj(step)
                if cfg.render_train_views:
                    self.render_all_train_views(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    def sort_loss(self) -> torch.Tensor:
        """Compute sorting loss."""
        if not self.cfg.use_sort or self.cfg.sort_lambda == 0.0:
            return torch.tensor(0.0)

        n_gs = self.splats["means"].shape[0]
        n_sidelen = int(n_gs**0.5)
        if n_sidelen * n_sidelen != n_gs:
            return torch.tensor(0.0)

        keys_to_regularize = ["means", "scales", "quats", "opacities", "sh0"]
        total_loss = torch.tensor(0.0, device=self.device)

        for key in keys_to_regularize:
            tensor = self.splats[key].data

            if key == "means":
                preprocessed_tensor = log_transform(tensor)
            elif key == "quats":
                preprocessed_tensor = F.normalize(tensor, dim=-1)
            elif key == "sh0":
                preprocessed_tensor = tensor.squeeze(1)
            else:
                preprocessed_tensor = tensor
            
            # Ensure tensor is 2D for consistent processing before reshaping
            if preprocessed_tensor.ndim == 1:
                preprocessed_tensor = preprocessed_tensor.unsqueeze(-1)
            
            # Reshape to grid. The -1 will correctly infer the feature dimension.
            grid_img = preprocessed_tensor.reshape(n_sidelen, n_sidelen, -1)
            grid_img = grid_img.permute(2, 0, 1) # (D, H, W)
            grid_img_batched = grid_img.unsqueeze(0) # (1, D, H, W)

            # Blur along X dimension with 'circular' padding
            blurred_x = kornia.filters.gaussian_blur2d(
                grid_img_batched,
                kernel_size=(1, self.cfg.sort_kernel_size),
                sigma=(self.cfg.sort_sigma, self.cfg.sort_sigma),
                border_type="circular",
            )
            # Blur along Y dimension with 'reflect' padding
            blurred_img = kornia.filters.gaussian_blur2d(
                blurred_x,
                kernel_size=(self.cfg.sort_kernel_size, 1),
                sigma=(self.cfg.sort_sigma, self.cfg.sort_sigma),
                border_type="reflect",
            )
            
            loss = F.huber_loss(blurred_img, grid_img_batched)
            total_loss += loss

        return total_loss * self.cfg.sort_lambda / float(len(keys_to_regularize))

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size
        
        if cfg.test_every > 0 and len(self.valset) > 0:
            print(f"Evaluating on {len(self.valset)} validation images")
            valloader = torch.utils.data.DataLoader(
                self.valset, batch_size=1, shuffle=False, num_workers=1
            )
        else:
            print(f"Evaluating on {len(self.trainset)} training images")
            valloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=1, shuffle=False, num_workers=1
            )
        
        
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) 
            undistort_masks = data["undistort_mask"].to(device) if "undistort_mask" in data else None
            segmentation_masks = data["segmentation_mask"].to(device) if "segmentation_mask" in data else None
            height, width = pixels.shape[1:3]

            # Extract distortion parameters if provided by the dataset
            distortion_params = data.get("distortion_params", None)
            camera_type_from_data = data.get("camera_type", None)
            if distortion_params is not None:
                distortion_params = distortion_params.to(device)

            # Prepare distortion coefficients for rasterization
            radial_coeffs_to_pass = None
            tangential_coeffs_to_pass = None

            if not cfg.undistort_colmap_input and distortion_params is not None:
                if cfg.camera_model == "fisheye":
                    # For OPENCV_FISHEYE, COLMAP params are [k1, k2, k3, k4]
                    # The rasterization function expects fisheye radial_coeffs as [batch_size, 4]
                    if distortion_params.shape[-1] == 4:
                        radial_coeffs_to_pass = distortion_params  # Should be [batch_size, 4]
                    else:
                        print(f"Warning: Fisheye model expects 4 distortion params, got {distortion_params.shape[-1]}")
                elif cfg.camera_model == "pinhole":
                    # For pinhole with distortion (OPENCV, RADIAL, SIMPLE_RADIAL)
                    # rasterization expects radial_coeffs [..., C, 6] and tangential_coeffs [..., C, 2]
                    num_params = distortion_params.shape[-1]
                    if num_params >= 1:
                        # Prepare radial coefficients (pad to 6 elements)
                        rad_params = torch.zeros(distortion_params.shape[0], 6, device=distortion_params.device)
                        
                        if num_params == 4:  # OPENCV: [k1, k2, p1, p2]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                            rad_params[:, 1] = distortion_params[:, 1]  # k2
                            # k3-k6 remain zero
                            tangential_coeffs_to_pass = distortion_params[:, [2, 3]].unsqueeze(0)  # [1, C, 2] p1, p2
                        elif num_params == 2:  # RADIAL: [k1, k2, 0, 0] -> extract [k1, k2]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                            rad_params[:, 1] = distortion_params[:, 1]  # k2
                        elif num_params == 1:  # SIMPLE_RADIAL: [k1, 0, 0, 0] -> extract [k1]
                            rad_params[:, 0] = distortion_params[:, 0]  # k1
                        else:
                            print(f"Warning: Unexpected number of distortion parameters: {num_params}")
                            
                        radial_coeffs_to_pass = rad_params.unsqueeze(0)  # [1, C, 6]

            torch.cuda.synchronize()
            tic = time.time()
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=undistort_masks,
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)
            
            if segmentation_masks is not None:
                pixels[segmentation_masks<0.5] = 0.0

            canvas_list = [pixels, colors]

            if world_rank == 0:
                

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                psnr_val = self.psnr(colors_p, pixels_p)
                ssim_val = self.ssim(colors_p, pixels_p)
                lpips_val = self.lpips(colors_p, pixels_p)
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
                metrics["lpips"].append(lpips_val)
                if cfg.test_every > 0 and len(self.valset) > 0:
                    # write images
                    canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                    canvas = (canvas * 255).astype(np.uint8)
                    
                    # Add metrics text to image
                    from PIL import Image, ImageDraw, ImageFont
                    img = Image.fromarray(canvas)
                    draw = ImageDraw.Draw(img)
                    metrics_text = f"PSNR: {psnr_val:.2f}\nSSIM: {ssim_val:.3f}\nLPIPS: {lpips_val:.3f}"
                    # Use a larger font size (48 pixels)
                    font = ImageFont.load_default(size=48)
                    draw.text((10, 10), metrics_text, fill=(255,255,255), font=font)
                    
                    img.save(f"{self.render_dir}/{stage}_step{step}_{i:04d}.png")
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            if cfg.use_bilateral_grid:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"CC_PSNR: {stats['cc_psnr']:.3f}, CC_SSIM: {stats['cc_ssim']:.4f}, CC_LPIPS: {stats['cc_lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            else:
                print(
                    f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                    f"Time: {stats['ellipse_time']:.3f}s/image "
                    f"Number of GS: {stats['num_GS']}"
                )
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        if self.cfg.disable_video:
            return
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # Get distortion parameters for novel view synthesis (use first camera's params)
        radial_coeffs_to_pass = None
        tangential_coeffs_to_pass = None
        
        if not cfg.undistort_colmap_input and hasattr(self.parser, 'params_dict') and len(self.parser.params_dict) > 0:
            first_camera_id = list(self.parser.params_dict.keys())[0]
            distortion_params = torch.from_numpy(self.parser.params_dict[first_camera_id]).float().to(device)
            
            if cfg.camera_model == "fisheye":
                # For OPENCV_FISHEYE, COLMAP params are [k1, k2, k3, k4]
                if distortion_params.shape[-1] == 4:
                    radial_coeffs_to_pass = distortion_params.unsqueeze(0)  # [1, 4] for single camera
            elif cfg.camera_model == "pinhole":
                # For pinhole with distortion (OPENCV, RADIAL, SIMPLE_RADIAL)
                num_params = distortion_params.shape[-1]
                if num_params >= 1:
                    # Prepare radial coefficients (pad to 6 elements)
                    rad_params = torch.zeros(1, 6, device=distortion_params.device)
                    
                    if num_params == 4:  # OPENCV: [k1, k2, p1, p2]
                        rad_params[0] = distortion_params[0]  # k1
                        rad_params[1] = distortion_params[1]  # k2
                        # k3-k6 remain zero
                        tangential_coeffs_to_pass = distortion_params[[2, 3]].unsqueeze(0)  # [1, 2] p1, p2
                    elif num_params == 2:  # RADIAL: [k1, k2, 0, 0] -> extract [k1, k2]
                        rad_params[0] = distortion_params[0]  # k1
                        rad_params[1] = distortion_params[1]  # k2
                    elif num_params == 1:  # SIMPLE_RADIAL: [k1, 0, 0, 0] -> extract [k1]
                        rad_params[0] = distortion_params[0]  # k1
                        
                    radial_coeffs_to_pass = rad_params.unsqueeze(0)  # [1, 1, 6]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")

    @torch.no_grad()
    def render_all_train_views(self, step: int):
        """Renders all training views with depth images."""
        print("Rendering all training views...")
        cfg = self.cfg
        device = self.device

        train_render_set = Dataset(
            self.parser,
            split="train",
            patch_size=None,
            load_depths=False,
        )
        trainloader = torch.utils.data.DataLoader(
            train_render_set, batch_size=1, shuffle=False, num_workers=1
        )

        render_output_dir = f"{self.render_dir}/train_view_renders_{step}"
        os.makedirs(render_output_dir, exist_ok=True)

        for i, data in enumerate(tqdm.tqdm(trainloader, desc="Rendering training views")):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) 
            
            # Get dimensions from actual image data to ensure correctness
            height, width = pixels.shape[1:3]

            distortion_params = data.get("distortion_params", None)
            if distortion_params is not None:
                distortion_params = distortion_params.to(device)

            radial_coeffs_to_pass = None
            tangential_coeffs_to_pass = None

            if not cfg.undistort_colmap_input and distortion_params is not None:
                if cfg.camera_model == "fisheye":
                    if distortion_params.shape[-1] == 4:
                        radial_coeffs_to_pass = distortion_params
                    else:
                        print(f"Warning: Fisheye model expects 4 distortion params, got {distortion_params.shape[-1]}")
                elif cfg.camera_model == "pinhole":
                    num_params = distortion_params.shape[-1]
                    if num_params >= 1:
                        rad_params = torch.zeros(distortion_params.shape[0], 6, device=distortion_params.device)
                        if num_params == 4:
                            rad_params[:, 0] = distortion_params[:, 0]
                            rad_params[:, 1] = distortion_params[:, 1]
                            tangential_coeffs_to_pass = distortion_params[:, [2, 3]].unsqueeze(0)
                        elif num_params == 2:
                            rad_params[:, 0] = distortion_params[:, 0]
                            rad_params[:, 1] = distortion_params[:, 1]
                        elif num_params == 1:
                            rad_params[:, 0] = distortion_params[:, 0]
                        else:
                            print(f"Warning: Unexpected number of distortion parameters: {num_params}")
                        radial_coeffs_to_pass = rad_params.unsqueeze(0)

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                radial_coeffs=radial_coeffs_to_pass,
                tangential_coeffs=tangential_coeffs_to_pass,
            )

            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)
            depths = renders[..., 3:4]

            # Save original image for comparison
            gt_img = (pixels.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            imageio.imwrite(f"{render_output_dir}/gt_{i:04d}.png", gt_img)

            # Save RGB image
            color_img = colors.squeeze(0).cpu().numpy()
            color_img = (color_img * 255).astype(np.uint8)
            imageio.imwrite(f"{render_output_dir}/rgb_{i:04d}.png", color_img)

            # Save depth image
            depth_img = depths.squeeze(0).cpu().numpy()
            
            # Save colormapped depth
            depth_img_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-10)
            depth_colormap = apply_float_colormap(torch.from_numpy(depth_img_normalized), "viridis").numpy()
            imageio.imwrite(f"{render_output_dir}/depth_colored_{i:04d}.png", (depth_colormap * 255).astype(np.uint8))

        print(f"Training views rendered and saved to {render_output_dir}")


    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors, render_alphas, info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )  # [1, H, W, 3]
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode == "rgb":
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha = 1 - alpha
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        return renders


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.render_train_views:
            runner.render_all_train_views(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=9 python -m examples.simple_trainer default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # Import BilateralGrid and related functions based on configuration
    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            from fused_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )
        else:
            cfg.use_bilateral_grid = True
            from lib_bilagrid import (
                BilateralGrid,
                color_correct,
                slice,
                total_variation_loss,
            )

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)
