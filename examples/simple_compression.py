import os
import argparse
import torch
from gsplat import PngCompression, WebpCompression
from gsplat.exporter import export_splats, load_splats


def main():
    parser = argparse.ArgumentParser(description="Load PLY -> Compress -> Decompress -> Export PLY")
    parser.add_argument("input_ply", help="Input PLY file path")
    parser.add_argument("--output_ply", default="output_compressed.ply", help="Output PLY file path")
    parser.add_argument("--compression_dir", default="./compression_temp", help="Directory for compressed files")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input_ply):
        print(f"Error: Input file {args.input_ply} does not exist")
        return

    print(f"Loading PLY file from {args.input_ply}...")
    
    # Step 1: Load the PLY file
    means, scales, quats, opacities, sh0, shN = load_splats(args.input_ply, device=args.device)
    
    print(f"Loaded {means.shape[0]} splats:")
    print(f"  means: {means.shape}")
    print(f"  scales: {scales.shape}")
    print(f"  quats: {quats.shape}")
    print(f"  opacities: {opacities.shape}")
    print(f"  sh0: {sh0.shape}")
    print(f"  shN: {shN.shape}")
    
    # Step 2: Convert to dictionary format for compression
    # Note: PngCompression expects pre-activation values
    splats_dict = {
        "means": means,
        "scales": scales,  # Keep as log scale (pre-activation)
        "quats": quats,
        "opacities": opacities,  # Keep as logit opacity (pre-activation)
        "sh0": sh0.squeeze(1),  # Remove the middle dimension for compression
        "shN": shN.permute(0, 2, 1).reshape(means.shape[0], -1) if shN.shape[1] > 0 else torch.zeros((means.shape[0], 0), device=means.device),  # Reshape SH
    }
    
    print(f"\nPrepared splats dictionary for compression:")
    for key, value in splats_dict.items():
        print(f"  {key}: {value.shape}")
    
    # Step 3: Compress using PngCompression
    print(f"\nCompressing splats to {args.compression_dir}...")
    os.makedirs(args.compression_dir, exist_ok=True)
    
    compressor = WebpCompression(use_sort=True, verbose=True)
    compressor.compress(args.compression_dir, splats_dict)
    
    print("Compression completed!")
    
    # Step 4: Decompress
    print(f"\nDecompressing splats from {args.compression_dir}...")
    decompressed_splats = compressor.decompress(args.compression_dir)
    
    print(f"Decompressed splats:")
    for key, value in decompressed_splats.items():
        print(f"  {key}: {value.shape}")
    
    # Step 5: Convert back to individual tensors for export
    dec_means = decompressed_splats["means"]
    dec_scales = decompressed_splats["scales"]
    dec_quats = decompressed_splats["quats"]
    dec_opacities = decompressed_splats["opacities"]
    dec_sh0 = decompressed_splats["sh0"].unsqueeze(1)  # Add back the middle dimension
    
    # Reshape shN back to proper format
    if decompressed_splats["shN"].shape[1] > 0:
        shN_flat = decompressed_splats["shN"]
        K = shN_flat.shape[1] // 3
        dec_shN = shN_flat.reshape(dec_means.shape[0], 3, K).permute(0, 2, 1)
    else:
        dec_shN = torch.zeros((dec_means.shape[0], 0, 3), device=dec_means.device)
    
    print(f"\nPrepared tensors for export:")
    print(f"  means: {dec_means.shape}")
    print(f"  scales: {dec_scales.shape}")
    print(f"  quats: {dec_quats.shape}")
    print(f"  opacities: {dec_opacities.shape}")
    print(f"  sh0: {dec_sh0.shape}")
    print(f"  shN: {dec_shN.shape}")
    
    # Step 6: Export to PLY
    print(f"\nExporting to {args.output_ply}...")
    export_splats(
        means=dec_means,
        scales=dec_scales,
        quats=dec_quats,
        opacities=dec_opacities,
        sh0=dec_sh0,
        shN=dec_shN,
        format="ply",
        save_to=args.output_ply
    )
    
    print("Export completed!")
    
    # Step 7: Compare sizes and report compression ratio
    input_size = os.path.getsize(args.input_ply)
    output_size = os.path.getsize(args.output_ply)
    
    # Calculate size of compressed files
    compressed_size = 0
    for root, dirs, files in os.walk(args.compression_dir):
        for file in files:
            compressed_size += os.path.getsize(os.path.join(root, file))
    
    print(f"\nFile sizes:")
    print(f"  Original PLY: {input_size / 1024 / 1024:.2f} MB")
    print(f"  Compressed files: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"  Decompressed PLY: {output_size / 1024 / 1024:.2f} MB")
    print(f"  Compression ratio: {input_size / compressed_size:.2f}x")

    
    print(f"\nCompression pipeline completed successfully!")
    print(f"Compressed files saved in: {args.compression_dir}")
    print(f"Output PLY saved as: {args.output_ply}")


if __name__ == "__main__":
    main()

