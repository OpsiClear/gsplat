import os
import argparse
import time
import yaml
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
import json

from gsplat import WebpCompression
from gsplat.exporter import export_splats, load_splats
from gsplat.rendering import rasterization
from gsplat.utils import inverse_log_transform

from datasets.colmap import Parser, Dataset


def render_view(
    splats: dict,
    view_data: dict,
    K: torch.Tensor,
    train_cfg: dict,
    device: str,
):
    """Renders a single view of the Gaussian splats."""
    camtoworld = view_data["camtoworld"].to(device)

    # Get image dimensions from the image tensor, like in the trainer.
    height, width = view_data["image"].shape[:2]

    # Prepare splat data for rendering
    means = splats["means"]
    scales = torch.exp(splats["scales"])
    quats = splats["quats"]
    opacities = torch.sigmoid(splats["opacities"])
    sh0 = splats["sh0"]
    shN = splats["shN"]
    colors = torch.cat([sh0, shN], 1)

    max_sh_degree = int(np.sqrt(1 + shN.shape[1]) - 1) if shN.shape[1] > 0 else 0

    render_colors, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworld[None]),
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=max_sh_degree,
        packed=False,
        rasterize_mode="classic",
        camera_model="pinhole",
    )
    return torch.clamp(render_colors.squeeze(0), 0.0, 1.0), alpha.squeeze(0)


def main():
    """Main function to run the compression evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate compression of a .ply file on quality, speed, and size."
    )
    parser.add_argument("--input_ply", help="Input PLY file path.")
    parser.add_argument(
        "--config", help="Path to the training config file (cfg.yml)."
    )
    parser.add_argument(
        "--output_dir",
        default="./compression_evaluation",
        help="Directory for compressed files and evaluation results.",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=10,
        help="Number of random views to evaluate.",
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="webp",
        choices=["webp", "png"],
        help="Compression method.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Quality for lossy parameters (1-100).",
    )
    parser.add_argument(
        "--lossy-params",
        nargs="+",
        default=[],
        help="List of parameters to compress lossily. Others will be lossless.",
    )
    parser.add_argument(
        "--render-device", default="cuda", help="Device to use for rendering (cuda/cpu)."
    )
    parser.add_argument(
        "--decompression-device",
        default="cpu",
        help="Device to use for decompression when decompressing to tensors (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--no-tensors",
        action="store_true",
        help="Decompress to NumPy arrays instead of PyTorch Tensors to measure raw decompression speed.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build quality settings dictionary from CLI args
    quality_settings = {}
    for param in args.lossy_params:
        quality_settings[param] = {"lossless": False, "quality": args.quality}

    # Dictionary to store timing and metrics
    stats = {}

    # Load training configuration
    print(f"Loading training config from {args.config}...")
    t0 = time.time()
    with open(args.config, "r") as f:
        config_text = f.read()

    # Remove Python-specific tags (e.g., !!python/tuple, !!python/object)
    # to allow safe loading. This makes the loader treat tuples as lists
    # and complex objects as simple dictionaries.
    config_text = re.sub(r"!!python/\S+", "", config_text)
    train_cfg = yaml.safe_load(config_text)

    # The 'strategy' object is complex and not needed for evaluation, so we remove it.
    if "strategy" in train_cfg:
        del train_cfg["strategy"]
    stats["config_load_time"] = time.time() - t0

    # Select and initialize compressor
    if args.compressor == "png":
        try:
            from gsplat import PngCompression
        except ImportError:
            raise ImportError(
                "To use PNG compression, you need to install torchpq and plas.\n"
                "torchpq: https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install\n"
                "plas: pip install git+https://github.com/fraunhoferhhi/PLAS.git"
            )
        compressor = PngCompression(use_sort=True, verbose=True)
    else:
        # Defaults to lossless, will be overridden by quality_settings
        compressor = WebpCompression(use_sort=True, verbose=True)

    # --- Step 1: Load Original Splats ---
    print(f"Loading PLY file from {args.input_ply}...")
    t0 = time.time()
    means, scales, quats, opacities, sh0, shN = load_splats(
        args.input_ply, device=args.render_device
    )
    stats["ply_load_time"] = time.time() - t0
    
    original_splats = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }
    print(f"Loaded {means.shape[0]} splats.")

    # Prepare dictionary for compression (pre-activation values)
    t0 = time.time()
    splats_dict_for_compression = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0.squeeze(1),
        "shN": (
            shN.permute(0, 2, 1).reshape(means.shape[0], -1)
            if shN.shape[1] > 0
            else torch.zeros(
                (means.shape[0], 0), device=means.device
            )
        ),
    }
    stats["data_prep_time"] = time.time() - t0

    compression_dir = os.path.join(args.output_dir, "compressed_data")
    os.makedirs(compression_dir, exist_ok=True)

    # --- Step 2: Compression ---
    print(f"\nCompressing splats to {compression_dir}...")
    t0 = time.time()
    compressor.compress(
        compression_dir,
        splats_dict_for_compression,
        quality_settings=quality_settings,
    )
    stats["compression_time"] = time.time() - t0
    print(f"  Compression took: {stats['compression_time']:.2f} seconds.")

    # --- Step 3: Decompression ---
    print(f"\nDecompressing splats from {compression_dir}...")
    t0 = time.time()
    decompressed_splats_dict = compressor.decompress(compression_dir, device=args.device)
    stats["decompression_time"] = time.time() - t0
    print(f"  Decompression took: {stats['decompression_time']:.2f} seconds.")

    # Move/convert decompressed data for rendering
    t0 = time.time()
    if args.no_tensors:
        # Convert numpy arrays to tensors and move to render device
        for key in decompressed_splats_dict:
            decompressed_splats_dict[key] = torch.from_numpy(
                decompressed_splats_dict[key]
            ).to(args.render_device)
        # Apply inverse log transform which was skipped during decompression
        decompressed_splats_dict["means"] = inverse_log_transform(
            decompressed_splats_dict["means"]
        )
    else:
        # Tensors are already on `decompression_device`, move to `render_device`
        for key in decompressed_splats_dict:
            decompressed_splats_dict[key] = decompressed_splats_dict[key].to(
                args.render_device
            )

    # Reshape tensors to match original splat format
    dec_means = decompressed_splats_dict["means"]
    dec_scales = decompressed_splats_dict["scales"]
    dec_quats = decompressed_splats_dict["quats"]
    dec_opacities = decompressed_splats_dict["opacities"]
    dec_sh0 = decompressed_splats_dict["sh0"].unsqueeze(1)
    if decompressed_splats_dict["shN"].shape[1] > 0:
        shN_flat = decompressed_splats_dict["shN"]
        K = shN_flat.shape[1] // 3
        dec_shN = shN_flat.reshape(dec_means.shape[0], 3, K).permute(0, 2, 1)
    else:
        dec_shN = torch.zeros(
            (dec_means.shape[0], 0, 3), device=dec_means.device
        )

    decompressed_splats = {
        "means": dec_means,
        "scales": dec_scales,
        "quats": dec_quats,
        "opacities": dec_opacities,
        "sh0": dec_sh0,
        "shN": dec_shN,
    }
    stats["tensor_processing_time"] = time.time() - t0

    # --- Step 4: Export and Report Size ---
    output_ply_path = os.path.join(args.output_dir, "decompressed.ply")
    t0 = time.time()
    export_splats(
        means=dec_means,
        scales=dec_scales,
        quats=dec_quats,
        opacities=dec_opacities,
        sh0=dec_sh0,
        shN=dec_shN,
        format="ply",
        save_to=output_ply_path,
    )
    stats["ply_export_time"] = time.time() - t0

    input_size = os.path.getsize(args.input_ply)
    compressed_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(compression_dir)
        for f in fn
    )
    stats["input_size_mb"] = input_size / 1e6
    stats["compressed_size_mb"] = compressed_size / 1e6
    stats["compression_ratio"] = input_size / compressed_size

    # --- Step 5: Quality Evaluation ---
    print("\nSetting up for quality evaluation...")
    t0 = time.time()
    parser = Parser(
        data_dir=train_cfg["data_dir"],
        factor=train_cfg.get("data_factor", 1),
        normalize=train_cfg.get("normalize_world_space", True),
        test_every=train_cfg.get("test_every", 0),  # Do not create a test set
        undistort_input=train_cfg.get("undistort_colmap_input", True),
        use_masks=train_cfg.get("use_masks", False),
    )
    trainset = Dataset(parser, split="train")
    stats["dataset_setup_time"] = time.time() - t0

    # Create an ideal pinhole camera matrix based on image dimensions
    # Get dimensions from the first image in the dataset
    first_image = trainset[0]["image"]
    height, width = first_image.shape[:2]
    focal_length = 1.2 * max(height, width)
    
    # Create ideal pinhole camera matrix
    K = torch.tensor(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
        dtype=torch.float32,
        device=args.render_device,
    )

    if args.num_views > len(trainset):
        print(f"Warning: Requested {args.num_views} views, but only {len(trainset)} are available. Using all views.")
        args.num_views = len(trainset)
    view_indices = np.random.choice(len(trainset), args.num_views, replace=False)

    total_l1_error = 0.0
    renders_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)

    print(f"\nEvaluating on {args.num_views} random views...")
    t0 = time.time()
    for i in tqdm(view_indices, desc="Rendering views"):
        data = trainset[i]

        original_render, original_alpha = render_view(
            original_splats, data, K, train_cfg, args.render_device
        )
        decompressed_render, _ = render_view(
            decompressed_splats, data, K, train_cfg, args.render_device
        )

        # Get indices of True values
        indices = torch.where(original_alpha > 0.1)
        y_indices, x_indices = indices[0], indices[1]

        # Get min/max coordinates
        x_min = x_indices.min().item()
        x_max = x_indices.max().item()
        y_min = y_indices.min().item()
        y_max = y_indices.max().item()

        # Crop the images (HWC format)
        original_crop = original_render[y_min:y_max, x_min:x_max, :]
        decompressed_crop = decompressed_render[y_min:y_max, x_min:x_max, :]

        # Compute L1 error
        total_l1_error += F.l1_loss(original_crop, decompressed_crop)
        
        # Create comparison image
        diff_image_vis = torch.abs(original_crop - decompressed_crop)
        canvas = torch.cat([original_crop, decompressed_crop, diff_image_vis], dim=1)
        
        # Save the comparison
        view_name = f"view_{i:04d}"
        canvas_np = (canvas.cpu().numpy() * 255).astype(np.uint8)
        imageio.imwrite(
            os.path.join(renders_dir, f"{view_name}_comparison.png"),
            canvas_np
        )

    stats["rendering_time"] = time.time() - t0
    stats["avg_l1_error"] = (total_l1_error / args.num_views).item()

    # Save all statistics to a file
    stats_path = os.path.join(args.output_dir, "evaluation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n--- Quality Metrics ---")
    print(f"  Average L1 error over {args.num_views} views: {stats['avg_l1_error']:.6f}")
    print(f"  Rendered images saved in: {renders_dir}")
    print(f"  Detailed statistics saved to: {stats_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

