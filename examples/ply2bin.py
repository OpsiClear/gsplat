import numpy as np
import torch
import struct
from pathlib import Path
from collections import namedtuple
from typing import Dict

# --- Import your actual IO functions ---
from gsplat.io_ply import import_splats, sh2rgb

# --- COLMAP Binary Writing Utilities ---

# A simple structure to hold point data, mirroring COLMAP's internal representation.
Point3D = namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

def write_next_bytes(fid, data, format_char):
    """Writes data to a binary file using the specified format."""
    if isinstance(data, (list, tuple)):
        # When data is a list (like for xyz or rgb), unpack it for struct.pack
        fid.write(struct.pack(format_char, *data))
    else:
        # When data is a single item (like an id or count)
        fid.write(struct.pack(format_char, data))

def write_points3D_binary(points3D: Dict[int, Point3D], path_to_model_file: Path):
    """
    Writes a dictionary of Point3D objects to a binary file in COLMAP format.
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")
    print(f"✅ Saved {len(points3D)} points to {path_to_model_file}")

# --- Main Conversion Function ---

def convert_ply_to_colmap_bin(ply_path: str, output_dir: str):
    """
    Reads a Gaussian splatting PLY file, extracts XYZ and color,
    and saves them into a COLMAP-compatible points3D.bin file.
    """
    output_dir = Path(output_dir)
    # The output path should be a file, so we ensure its parent directory exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Step 1: Importing splat data...")
    means, _, _, _, sh0, _ = import_splats(ply_path, device=device)

    print("Step 2: Calculating RGB colors from spherical harmonics...")
    colors = sh2rgb(sh0).squeeze(1)
    print(colors.shape)

    final_points_xyz = means.detach().cpu().numpy()
    final_colors_rgb = (colors * 255).detach().cpu().to(torch.uint8).numpy()

    print("Step 3: Structuring data for COLMAP...")
    points3D = {}
    for i, (xyz, rgb) in enumerate(zip(final_points_xyz, final_colors_rgb)):
        points3D[i] = Point3D(
            id=i,
            xyz=xyz,
            rgb=rgb,
            error=0.0,
            image_ids=np.array([]),
            point2D_idxs=np.array([])
        )

    print("Step 4: Writing points3D.bin file...")
    # The output_dir variable is now the full path to the .bin file
    write_points3D_binary(points3D, output_dir)

# --- Example Usage ---
if __name__ == "__main__":
    # Define the path to your input PLY file
    input_ply_file = "/data/shared/elaheh/4D/4D_scenes/elly/gsplat-frame001-distorted/ply/point_cloud_29999.ply"
    
    # Define the FULL path for the output file
    output_file_path = "/data/shared/elaheh/4D/4D_scenes/elly/gsplat-frame001-distorted/bin/points3D.bin"


    print(f"Running conversion for '{input_ply_file}'...")
    # Pass the full output file path to the function
    convert_ply_to_colmap_bin(ply_path=input_ply_file, output_dir=output_file_path)
    print("\nConversion complete! ✨")