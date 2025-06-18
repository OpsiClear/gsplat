import os
import argparse
import time
import yaml
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import re
import json

from gsplat import Compression
from gsplat.exporter import export_splats, load_splats, load_ply_gaussian
from gsplat.rendering import rasterization



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


def _encode_image(img, lossless: bool = True, quality: int = 100):
    """
    Compresses the image as webp.
    """
    from PIL import Image
    import io

    buffer = io.BytesIO()
    Image.fromarray(img).save(
        buffer,
        format="webp",
        lossless=lossless,
        quality=quality if not lossless else 100,
        method=6,
        exact=True,
    )
    return buffer.getvalue()


def main():
    """Main function to run the dynamic compression."""
    parser = argparse.ArgumentParser(
        description="Compress a sequence of .ply files for dynamic scenes."
    )
    parser.add_argument("--input_dir", help="Input directory containing .ply files.")
    parser.add_argument(
        "--output_dir",
        default="./dynamic_compression_output",
        help="Directory for compressed files.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=100,
        help="Quality for lossy parameters (1-100).",
    )
    parser.add_argument(
        "--lossy-params",
        nargs="+",
        default=[],
        help="List of parameters to compress lossily. Others will be lossless.",
    )
    parser.add_argument(
        "--resort_interval",
        type=int,
        default=0,
        help="Interval for re-sorting frames to adapt to changes. 0 to disable.",
    )
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build quality settings dictionary from CLI args
    quality_settings = {}
    for param in args.lossy_params:
        quality_settings[param] = {"lossless": False, "quality": args.quality}

    # Dictionary to store timing and metrics for the whole sequence
    total_stats = {
        "total_compression_time": 0.0,
        "total_input_size_mb": 0.0,
        "total_compressed_size_mb": 0.0,
    }

    # Centralized metadata for the entire sequence
    full_meta = {"video_properties": {}, "frames": {}}
    atlas_images_in_memory = []


    # Select and initialize compressor
    compressor = Compression(use_sort=True, verbose=False, seed=42)  # Verbose off for sequence

    ply_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".ply")])
    
    sort_indices = None  # Initialize sort indices
    shn_codebook = None # Initialize shn codebook

    for frame_idx, ply_file in tqdm(enumerate(ply_files), desc="Compressing frames", total=len(ply_files)):
        frame_name = os.path.splitext(ply_file)[0]
        input_ply_path = os.path.join(args.input_dir, ply_file)
        
        frame_stats = {}

        # --- Step 1: Load Original Splats ---
        t0 = time.time()
        means, scales, quats, opacities, sh0, shN = load_ply_gaussian(
            input_ply_path, device=args.device
        )
        frame_stats["ply_load_time"] = time.time() - t0

        # Prepare dictionary for compression (pre-activation values)
        t0 = time.time()
        splats_dict_for_compression = {
            "means": means,
            "scales": scales,
            "quats": quats,
            "opacities": opacities,
            "sh0": sh0,
            "shN": shN,
        }
        frame_stats["data_prep_time"] = time.time() - t0

        # --- Step 2: Compression ---
        t0 = time.time()
        
        # Determine if a re-sort is needed
        force_resort = False
        if args.resort_interval > 0 and frame_idx > 0 and frame_idx % args.resort_interval == 0:
            force_resort = True

        meta, compressed_arrays, new_indices, new_shn_codebook = compressor.compress(
            splats_dict_for_compression,
            quality_settings=quality_settings,
            sort_indices=sort_indices,
            force_resort=force_resort,
            shn_initial_centroids=None if force_resort else shn_codebook,
        )
        if new_indices is not None:
            sort_indices = new_indices
        if new_shn_codebook is not None:
            shn_codebook = new_shn_codebook

        frame_stats["compression_time"] = time.time() - t0
        total_stats["total_compression_time"] += frame_stats["compression_time"]

        # --- Step 3: Create Atlas and Report Size ---
        input_size = os.path.getsize(input_ply_path)

        atlas_layout = [
            ["means_l", "means_u", "opacities"],
            ["quats", "scales", "sh0"],
            ["shN_centroids", "shN_labels"],
        ]
        atlas_meta = {}
        prepared_images = {}

        # Convert all component arrays to RGBA uint8
        for name, array in compressed_arrays.items():
            if array.ndim == 2:  # Grayscale
                rgba_array = np.stack([array, array, array, np.full(array.shape, 255, dtype=np.uint8)], axis=-1)
            elif array.ndim == 3 and array.shape[2] == 3:  # RGB
                rgba_array = np.concatenate([array, np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)
            elif array.ndim == 3 and array.shape[2] == 4:  # Already RGBA
                rgba_array = array
            else:
                raise ValueError(f"Unsupported array shape for atlas: {name} {array.shape}")
            
            prepared_images[name] = rgba_array.astype(np.uint8)


        # Stitch images into an atlas
        row_images = []
        current_y = 0
        max_w = 0
        for row_layout in atlas_layout:
            images_in_row = []
            max_h = 0
            existing_params = [p for p in row_layout if p in prepared_images]
            if not existing_params:
                continue

            for name in existing_params:
                img = prepared_images[name]
                images_in_row.append(img)
                if img.shape[0] > max_h:
                    max_h = img.shape[0]

            padded_images = []
            current_x = 0
            for i, name in enumerate(existing_params):
                img = images_in_row[i]
                h, w, _ = img.shape
                pad_h = max_h - h
                padded_img = np.pad(
                    img, ((0, pad_h), (0, 0), (0, 0)), "constant"
                )
                padded_images.append(padded_img)

                atlas_meta[name] = {"x": current_x, "y": current_y, "w": w, "h": h}
                current_x += w

            stitched_row = np.hstack(padded_images)
            row_images.append(stitched_row)
            if stitched_row.shape[1] > max_w:
                max_w = stitched_row.shape[1]
            current_y += max_h

        # Pad rows to the same width and stack
        padded_rows = [
            np.pad(
                r, ((0, 0), (0, max_w - r.shape[1]), (0, 0)), "constant"
            )
            for r in row_images
        ]
        atlas_image = np.vstack(padded_rows)
        atlas_images_in_memory.append(atlas_image)
        
        meta["atlas_layout"] = atlas_meta
        meta["original_ply"] = ply_file
        full_meta['frames'][str(frame_idx)] = meta


        total_stats["total_input_size_mb"] += input_size / 1e6

    # --- Final Step: Write Video and Centralized Metadata ---
    if atlas_images_in_memory:
        video_path = os.path.join(args.output_dir, "dynamic_scene.webp")
        print(f"\nWriting video to {video_path}...")
        imageio.mimwrite(video_path, atlas_images_in_memory, codec="libwebp", lossless=True)
        
        # Get video properties for metadata
        first_atlas = atlas_images_in_memory[0]
        h, w, _ = first_atlas.shape
        full_meta["video_properties"] = {
            "codec": "libwebp",
            "pixel_format": "rgba",
            "frame_count": len(atlas_images_in_memory),
            "atlas_width": w,
            "atlas_height": h,
        }
        
        total_stats["total_compressed_size_mb"] += os.path.getsize(video_path) / 1e6

    # Save the centralized metadata file
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(full_meta, f, indent=4)
    total_stats["total_compressed_size_mb"] += os.path.getsize(meta_path) / 1e6


    total_stats["avg_compression_time_per_frame"] = total_stats["total_compression_time"] / len(ply_files)
    total_stats["overall_compression_ratio"] = total_stats["total_input_size_mb"] / total_stats["total_compressed_size_mb"] if total_stats["total_compressed_size_mb"] > 0 else float('inf')

    # Save all statistics to a file
    stats_path = os.path.join(args.output_dir, "compression_stats.json")
    with open(stats_path, "w") as f:
        json.dump(total_stats, f, indent=4)

    print("\n--- Compression Summary ---")
    print(f"  Processed {len(ply_files)} PLY files.")
    print(f"  Total input size: {total_stats['total_input_size_mb']:.2f} MB")
    print(f"  Total compressed size: {total_stats['total_compressed_size_mb']:.2f} MB")
    print(f"  Overall compression ratio: {total_stats['overall_compression_ratio']:.2f}x")
    print(f"  Average compression time: {total_stats['avg_compression_time_per_frame']:.2f} s/frame")
    print(f"  Detailed statistics saved to: {stats_path}")
    print("\nDynamic compression complete!")

    # --- Decompression Step ---
    print("\n--- Decompressing files ---")
    decompressed_dir = os.path.join(args.output_dir, "decompressed_plys")
    os.makedirs(decompressed_dir, exist_ok=True)

    # Load the centralized metadata
    meta_path = os.path.join(args.output_dir, "meta.json")
    with open(meta_path, "r") as f:
        full_meta = json.load(f)

    # Read all frames from the video into memory
    video_path = os.path.join(args.output_dir, "dynamic_scene.webp")
    print(f"Reading video frames from {video_path}...")
    try:
        video_frames = imageio.mimread(video_path)
    except FileNotFoundError:
        print(f"Error: Video file not found at {video_path}")
        return

    total_decompression_time = 0.0

    frame_count = full_meta.get("video_properties", {}).get("frame_count", 0)
    for frame_idx in tqdm(range(frame_count), desc="Decompressing frames"):
        
        # Get the metadata for the current frame
        meta = full_meta["frames"][str(frame_idx)]
        original_ply_name = meta.pop("original_ply", f"frame_{frame_idx:04d}.ply")

        # Get the atlas image for the current frame
        atlas_image = video_frames[frame_idx]

        # Load compressed data from the atlas
        atlas_layout = meta.pop("atlas_layout", {})
        compressed_arrays_loaded = {}

        # This dictionary defines how many channels to extract for each parameter.
        # It's based on the expected input of the decompression functions.
        channel_map = {
            "means_l": 3, "means_u": 3, "opacities": 1,
            "quats": 4, "scales": 3, "sh0": 3,
            "shN_centroids": 3, "shN_labels": 3
        }

        for name, coords in atlas_layout.items():
            x, y, w, h = coords["x"], coords["y"], coords["w"], coords["h"]
            # Slice the image, retaining original dimensions
            sub_image = atlas_image[y : y + h, x : x + w]

            # Extract the correct channels for the parameter
            num_channels = channel_map.get(name)
            if num_channels is None:
                print(f"Warning: No channel mapping for {name}. Skipping.")
                continue

            if sub_image.ndim == 2 and num_channels > 1:
                 # This can happen if a channel was all black and got saved as grayscale
                print(f"Warning: Grayscale sub-image for {name}, but expected {num_channels} channels. Tiling.")
                sub_image = np.stack([sub_image] * num_channels, axis=-1)

            if num_channels == 1:
                # For grayscale, take one channel and drop the channel dimension
                if sub_image.ndim > 2:
                    sub_image = sub_image[..., 0]
            else:
                # For multi-channel, slice to the expected number of channels
                if sub_image.ndim == 3 and sub_image.shape[2] > num_channels:
                    sub_image = sub_image[..., :num_channels]
            
            compressed_arrays_loaded[name] = sub_image

        # Decompress
        t0 = time.time()
        decompressed_splats = compressor.decompress(meta, compressed_arrays_loaded)
        total_decompression_time += time.time() - t0

        num_splats = (
            decompressed_splats["means"].shape[0]
            if "means" in decompressed_splats and decompressed_splats["means"] is not None
            else 0
        )
        device = (
            decompressed_splats["means"].device
            if "means" in decompressed_splats and decompressed_splats["means"] is not None
            else args.device
        )

        # Ensure all keys have a tensor, even if empty, for consistent export
        if decompressed_splats.get("means") is None:
            decompressed_splats["means"] = torch.empty((0, 3), device=device)
        if decompressed_splats.get("scales") is None:
            decompressed_splats["scales"] = torch.empty((num_splats, 3), device=device)
        if decompressed_splats.get("quats") is None:
            decompressed_splats["quats"] = torch.empty((num_splats, 4), device=device)
        if decompressed_splats.get("opacities") is None:
            decompressed_splats["opacities"] = torch.empty(
                (num_splats, 1), device=device
            )
        if decompressed_splats.get("sh0") is None:
            decompressed_splats["sh0"] = torch.empty((num_splats, 1, 3), device=device)
        if decompressed_splats.get("shN") is None:
            decompressed_splats["shN"] = torch.empty((num_splats, 0, 3), device=device)
        
        # Export the decompressed splats to a .ply file
        output_ply_path = os.path.join(decompressed_dir, original_ply_name)
        export_splats(
            save_to=output_ply_path,
            means=decompressed_splats["means"],
            scales=decompressed_splats["scales"],
            quats=decompressed_splats["quats"],
            opacities=decompressed_splats["opacities"],
            sh0=decompressed_splats["sh0"],
            shN=decompressed_splats["shN"],
        )
    
    avg_decompression_time = total_decompression_time / frame_count if frame_count > 0 else 0
    print("\n--- Decompression Summary ---")
    print(f"  Decompressed {frame_count} frames.")
    print(f"  Decompressed files saved to: {decompressed_dir}")
    print(f"  Average decompression time: {avg_decompression_time:.2f} s/frame")
    print("\nDecompression complete!")


if __name__ == "__main__":
    main() 