import os
import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
import json
import copy
from typing import Dict
from torch import Tensor

from gsplat import WebpCompression
from gsplat.exporter import export_splats, load_splats
from gsplat.rendering import rasterization
from gsplat.utils import inverse_log_transform, log_transform
from gsplat.compression.webp_compression import _write_image, _precompress_webp, _precompress_webp_16bit, _crop_n_splats, sort_splats, _precompress_kmeans
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


def run_evaluation(
    config_path,
    output_dir,
    lossy_param,
    quality,
    num_views,
    render_device,
    decompression_device,
    use_tensors,
    original_splats: Dict[str, Tensor],
    splats_for_compression: Dict[str, Tensor],
    train_cfg: Dict,
    dataset: Dataset,
    K: torch.Tensor,
):
    """Runs a single compression and evaluation run for a given lossy parameter."""

    config_str = f"Lossy: {lossy_param or 'None'}, Quality: {quality}, Decomp Device: {decompression_device}, Output: {'Tensor' if use_tensors else 'NumPy'}"
    print(f"EVALUATING: {config_str}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store timing and metrics
    stats = {
        "lossy_param": lossy_param,
        "quality": quality if lossy_param else 100,
        "decompression_device": decompression_device,
        "use_tensors": use_tensors,
    }

    # Initialize compressor
    compressor = WebpCompression(
        use_sort=True, verbose=False
    )  # Verbose is off to keep logs clean
    quality_settings = {}
    if lossy_param:
        quality_settings[lossy_param] = {"lossless": False, "quality": quality}

    # --- Step 1: Data is already loaded and prepared ---

    compression_dir = os.path.join(output_dir, "compressed_data")
    os.makedirs(compression_dir, exist_ok=True)

    # --- Step 2: Compression is now done outside this function ---

    # --- Step 3: Decompression ---
    t0 = time.time()
    decompressed_splats_dict, timings = compressor.decompress(
        compression_dir, device=decompression_device, to_tensors=use_tensors
    )
    stats["decompression_time"] = time.time() - t0
    stats["decompression_timings"] = timings

    # Post-process for rendering
    # If we got numpy arrays, convert them to tensors for rendering
    if not use_tensors:
        for k, v in decompressed_splats_dict.items():
            decompressed_splats_dict[k] = torch.from_numpy(v)
        # also need to apply inverse_log_transform for means, which is skipped when to_tensors=False
        if "means" in decompressed_splats_dict:
            decompressed_splats_dict["means"] = inverse_log_transform(
                decompressed_splats_dict["means"]
            )

    # Now everything is a tensor, but might be on the wrong device for rendering.
    # Move to render_device
    for k, v in decompressed_splats_dict.items():
        decompressed_splats_dict[k] = v.to(render_device)

    # Combine original and decompressed splats
    # (assuming all params are present, as this script compresses all)
    dec_means = decompressed_splats_dict["means"]
    dec_scales = decompressed_splats_dict["scales"]
    dec_quats = decompressed_splats_dict["quats"]
    dec_opacities = decompressed_splats_dict["opacities"]
    dec_sh0 = decompressed_splats_dict["sh0"].unsqueeze(1)
    if decompressed_splats_dict["shN"].shape[1] > 0:
        shN_flat = decompressed_splats_dict["shN"]
        K_sh = shN_flat.shape[1] // 3
        dec_shN = shN_flat.reshape(dec_means.shape[0], 3, K_sh).permute(0, 2, 1)
    else:
        dec_shN = torch.zeros(
            (dec_means.shape[0], 0, 3), device=render_device
        )

    decompressed_splats = {
        "means": dec_means,
        "scales": dec_scales,
        "quats": dec_quats,
        "opacities": dec_opacities,
        "sh0": dec_sh0,
        "shN": dec_shN,
    }

    # --- Step 4: Export and Report Size ---
    output_ply_path = os.path.join(output_dir, "decompressed.ply")
    export_splats(
        means=dec_means,
        scales=dec_scales,
        quats=dec_quats,
        opacities=dec_opacities,
        sh0=dec_sh0,
        shN=dec_shN,
        save_to=output_ply_path,
    )

    # Get input size from the original splats dictionary
    input_size_bytes = sum(v.element_size() * v.nelement() for v in original_splats.values())
    compressed_size_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(compression_dir)
        for f in fn
    )
    stats["input_size_mb"] = input_size_bytes / 1e6
    stats["compressed_size_mb"] = compressed_size_bytes / 1e6
    stats["compression_ratio"] = input_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else float('inf')

    # --- Step 5: Quality Evaluation ---
    # Dataset is now pre-loaded
    view_indices = np.random.choice(len(dataset), num_views, replace=False)

    total_l1_error = 0.0
    for i in tqdm(
        view_indices, desc=f"Rendering ({lossy_param or 'lossless'})"
    ):
        data = dataset[i]
        original_render, original_alpha = render_view(
            original_splats, data, K, train_cfg, render_device
        )
        decompressed_render, _ = render_view(
            decompressed_splats, data, K, train_cfg, render_device
        )

        indices = torch.where(original_alpha > 0.1)
        original_crop = original_render[indices[0], indices[1]]
        decompressed_crop = decompressed_render[indices[0], indices[1]]

        total_l1_error += F.l1_loss(original_crop, decompressed_crop)

    stats["avg_l1_error"] = (total_l1_error / num_views).item()

    # Save stats
    stats_path = os.path.join(output_dir, "evaluation_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)

    print(f"  Avg L1 Error: {stats['avg_l1_error']:.6f}")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")

    total_io = sum(t.get("io_time", 0) for t in timings.values())
    total_proc = sum(t.get("processing_time", 0) for t in timings.values())
    print(f"  Decompression I/O: {total_io:.4f}s, Processing: {total_proc:.4f}s")

    print(f"  Results saved to {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the impact of making individual parameters lossy."
    )
    parser.add_argument("--input_ply", required=True, help="Input PLY file path.")
    parser.add_argument(
        "--config", required=True, help="Path to the training config file (cfg.yml)."
    )
    parser.add_argument(
        "--output_dir", required=True, help="Base directory for evaluation results."
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Quality for lossy compression (1-100).",
    )
    parser.add_argument(
        "--num_views", type=int, default=10, help="Number of random views to evaluate."
    )
    parser.add_argument(
        "--render_device", default="cuda", help="Device to use for rendering (cuda/cpu)."
    )
    args = parser.parse_args()

    # --- Load Data and Config Once ---
    print("=" * 80)
    print("Loading data and configuration once...")

    # Load training configuration
    with open(args.config, "r") as f:
        config_text = f.read()
    config_text = re.sub(r"!!python/\S+", "", config_text)
    train_cfg = yaml.safe_load(config_text)
    if "strategy" in train_cfg:
        del train_cfg["strategy"]

    # Load splats
    means, scales, quats, opacities, sh0, shN = load_splats(
        args.input_ply, device=args.render_device
    )
    original_splats = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }
    splats_for_compression = {
        "means": means.clone(),
        "scales": scales.clone(),
        "quats": quats.clone(),
        "opacities": opacities.clone(),
        "sh0": sh0.squeeze(1).clone(),
        "shN": (
            shN.permute(0, 2, 1).reshape(means.shape[0], -1).clone()
            if shN.shape[1] > 0
            else torch.zeros((means.shape[0], 0), device=args.render_device)
        ),
    }

    # Prepare dataset and camera intrinsics
    colmap_parser = Parser(
        data_dir=train_cfg["data_dir"],
        factor=train_cfg.get("data_factor", 1),
        normalize=train_cfg.get("normalize_world_space", True),
        test_every=0,
        undistort_input=train_cfg.get("undistort_colmap_input", True),
        use_masks=train_cfg.get("use_masks", False),
    )
    dataset = Dataset(colmap_parser, split="train")
    first_image = dataset[0]["image"]
    height, width = first_image.shape[:2]
    focal_length = 1.2 * max(height, width)
    K = torch.tensor(
        [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
        dtype=torch.float32,
        device=args.render_device,
    )
    print("Data loaded.")
    print("=" * 80)

    summary = []
    
    # --- Pre-computation ---
    print("Running pre-computation step...")
    compressor = WebpCompression(use_sort=True, verbose=True)
    
    # We need to run the expensive parts of compression once
    # The `compress` method does sorting and cropping, which we want
    # We'll create a temporary compressor instance to do this
    temp_compressor = WebpCompression(use_sort=True, verbose=False)
    temp_splats = copy.deepcopy(splats_for_compression)
    
    # Run sort and crop
    temp_splats["means"] = log_transform(temp_splats["means"])
    temp_splats["quats"] = F.normalize(temp_splats["quats"], dim=-1)
    n_gs = len(temp_splats["means"])
    n_sidelen = int(n_gs**0.5)
    n_crop = n_gs - n_sidelen**2
    if n_crop != 0:
        temp_splats = _crop_n_splats(temp_splats, n_crop)
    
    # This is the key part: sorting is expensive
    sorted_splats = sort_splats(temp_splats)

    # Now, generate the in-memory images for each parameter
    precomputed_images = {}
    precomputed_meta = {}

    for param_name in sorted_splats.keys():
        if param_name == "means":
            meta, images = _precompress_webp_16bit(sorted_splats[param_name], n_sidelen)
        elif param_name == "shN" and torch.numel(sorted_splats[param_name]) > 0:
            meta, images = _precompress_kmeans(sorted_splats[param_name], n_sidelen, verbose=False)
        else:
            meta, images = _precompress_webp(sorted_splats[param_name], n_sidelen)
        precomputed_images[param_name] = images
        precomputed_meta[param_name] = meta
    
    print("Pre-computation finished.")

    # --- Define Test Cases ---
    test_cases = [
        {"name": "baseline_lossless_cpu_tensor", "lossy_param": None, "decomp_device": "cpu", "use_tensors": True},
        {"name": "baseline_lossless_cuda_tensor", "lossy_param": None, "decomp_device": "cuda", "use_tensors": True},
        {"name": "baseline_lossless_cpu_numpy", "lossy_param": None, "decomp_device": "cpu", "use_tensors": False},
    ]
    params_to_test = ["means_l", "means_u", "scales", "quats", "opacities", "sh0", "shN_centroids", "shN_labels"]
    for param in params_to_test:
        test_cases.append({"name": f"lossy_{param}_q{args.quality}", "lossy_param": param, "decomp_device": "cpu", "use_tensors": True})

    # --- Run Evaluations ---
    for case in test_cases:
        if "cuda" not in args.render_device and "cuda" in case["decomp_device"]:
            print(f"\nSkipping {case['name']} (render_device is not cuda)")
            continue

        print("\n" + "#"*20 + f" Running Test: {case['name']} " + "#"*20)
        
        run_output_dir = os.path.join(args.output_dir, case["name"])
        compression_dir = os.path.join(run_output_dir, "compressed_data")
        os.makedirs(compression_dir, exist_ok=True)
        
        # --- Save pre-computed images with current test settings ---
        t0_compress = time.time()
        final_meta = copy.deepcopy(precomputed_meta)
        quality_settings = {}
        if case["lossy_param"]:
            quality_settings[case["lossy_param"]] = {"lossless": False, "quality": args.quality}

        for param_name, images in precomputed_images.items():
            param_files = []
            # Get base quality settings for the main parameter (e.g., 'means')
            base_param_quality = quality_settings.get(param_name, {})
            base_lossless = base_param_quality.get("lossless", True)
            base_quality = base_param_quality.get("quality", args.quality)

            for img_key, img_data in images.items():
                # For sub-images like 'means_l', check for specific settings
                sub_param_name = f"{param_name}_{img_key}"
                sub_param_quality = quality_settings.get(sub_param_name, base_param_quality)
                
                lossless = sub_param_quality.get("lossless", base_lossless)
                quality = sub_param_quality.get("quality", base_quality)

                # Write the image to disk
                filename = _write_image(
                    compression_dir, 
                    sub_param_name if img_key != "img" else param_name,
                    img_data,
                    lossless=lossless,
                    quality=quality if not lossless else 100,
                    verbose=False,
                )
                param_files.append(filename)
            final_meta[param_name]["files"] = param_files

        with open(os.path.join(compression_dir, "meta.json"), "w") as f:
            json.dump(final_meta, f, indent=2)

        compression_time = time.time() - t0_compress

        # --- Run Decompression and Rendering Evaluation ---
        stats = run_evaluation(
            config_path=args.config,
            output_dir=run_output_dir,
            lossy_param=case["lossy_param"],
            quality=args.quality,
            num_views=args.num_views,
            render_device=args.render_device,
            decompression_device=case["decomp_device"],
            use_tensors=case["use_tensors"],
            original_splats=original_splats,
            splats_for_compression=sorted_splats, # Pass sorted splats
            train_cfg=train_cfg,
            dataset=dataset,
            K=K,
        )
        stats["compression_time"] = compression_time
        summary.append(stats)

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY OF EVALUATION") 
    print("=" * 80)
    header = f"{'Lossy Parameter':<18} | {'Comp Time (s)':<15} | {'Decomp Time (s)':<15} | {'Size (MB)':<10} | {'Comp Ratio':<12} | {'Avg L1 Error':<15} | {'Decomp Device':<15} | {'Output':<8}"
    print(header)
    print("-" * len(header))

    # Sort results logically for comparison
    sorted_summary = sorted(summary, key=lambda x: (
        x["lossy_param"] is not None, # Group lossless and lossy tests
        x["lossy_param"] or "",       # Sort by lossy param name
        x["decompression_device"],    # Then by device
        not x["use_tensors"]          # Then by output type (Tensor first)
    ))

    # Open file to save summary table
    with open(os.path.join(args.output_dir, "evaluation_table.txt"), "w") as f:
        f.write("SUMMARY OF EVALUATION\n")
        f.write("=" * 80 + "\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for stats in sorted_summary:
            param_name = stats["lossy_param"] or "None"
            device_name = stats["decompression_device"]
            output_type = "Tensor" if stats["use_tensors"] else "NumPy"
            row = (
                f"{param_name:<18} | "
                f"{stats['compression_time']:<15.4f} | "
                f"{stats['decompression_time']:<15.4f} | "
                f"{stats['compressed_size_mb']:<10.2f} | "
                f"{stats['compression_ratio']:<12.2f}x | "
                f"{stats['avg_l1_error']:<15.6f} | "
                f"{device_name:<15} | "
                f"{output_type:<8}"
            )
            print(row)
            f.write(row + "\n")
    summary_path = os.path.join(args.output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main() 