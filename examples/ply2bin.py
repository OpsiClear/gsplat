import numpy as np
import torch
import struct
from pathlib import Path
from collections import namedtuple
from typing import Dict
from plyfile import PlyData

# --- Your existing COLMAP Binary Writing Utilities ---
# (These functions are unchanged)
Point3D = namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

def write_next_bytes(fid, data, format_char):
    if isinstance(data, (list, tuple)):
        fid.write(struct.pack(format_char, *data))
    else:
        fid.write(struct.pack(format_char, data))

def write_points3D_binary(points3D: Dict[int, Point3D], path_to_model_file: Path):
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0] if pt.image_ids.ndim > 0 else 0
            write_next_bytes(fid, track_length, "Q")
            if track_length > 0:
                for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                    write_next_bytes(fid, [image_id, point2D_id], "ii")
    print(f"✅ Saved {len(points3D)} points to {path_to_model_file}")

# --- Utility to convert Spherical Harmonics to RGB ---
def sh2rgb(sh0):
    C0 = 0.28209479177387814
    return (sh0 * C0 + 0.5).clamp(0.0, 1.0)

# --- NEW Main Conversion and Sampling Function ---

def convert_and_sample_ply_to_colmap(
    ply_path: str, output_path: str, target_point_count: int
):
    """
    Reads a Gaussian splatting PLY file, samples it based on opacity if it's too large,
    and saves the result to a COLMAP-compatible points3D.bin file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Step 1: Read PLY header to get total point count ---
    print("Step 1: Reading PLY file header...")
    plydata = PlyData.read(ply_path)
    num_points = len(plydata['vertex'].data)
    print(f"🔍 Found {num_points:,} points in the PLY file.")

    mask = None
    # --- MODIFIED: Only sample if the point count exceeds the target ---
    if num_points > target_point_count:
        print(f"❗️ Point count exceeds target of {target_point_count:,}. Starting sampling...")
        
        # --- First Pass: Read opacities for filtering (memory-efficient) ---
        print("   (Pass 1/2) Reading opacities to determine most significant points...")
        opacities = torch.from_numpy(plydata['vertex']['opacity'][:]).to(device)

        # Find the K-th largest opacity value. This is our cutoff threshold.
        k = target_point_count
        # We look for the (num_points - k + 1)-th smallest value, which is the k-th largest
        threshold, _ = torch.kthvalue(opacities, num_points - k + 1)
        
        # Create a boolean mask of points to keep
        mask = opacities >= threshold
        
        # --- NEW: Ensure the EXACT number of points is selected ---
        # This handles cases where multiple points have the same opacity as the threshold
        current_count = mask.sum().item()
        if current_count > target_point_count:
            print(f"   Threshold resulted in {current_count:,} points. Sub-sampling to exact target...")
            # Get indices of all points that passed the threshold
            valid_indices = torch.where(mask)[0]
            # Randomly select from this subset to get the exact count
            perm = torch.randperm(valid_indices.shape[0], device=device)
            final_indices = valid_indices[perm[:target_point_count]]
            # Recreate the mask from scratch to be safe
            mask = torch.zeros_like(opacities, dtype=torch.bool)
            mask[final_indices] = True
        
        print(f"   Keeping exactly {mask.sum():,} points.")

    else:
        print(f"✅ Point count ({num_points:,}) is within the target ({target_point_count:,}). No sampling needed.")

    # --- Second Pass: Read the required data using the mask ---
    print("Step 2: Loading point data...")
    vertex_data = plydata['vertex'].data
    
    means = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    sh0 = np.vstack([vertex_data['f_dc_0'], vertex_data['f_dc_1'], vertex_data['f_dc_2']]).T
    
    means = torch.from_numpy(means).float().to(device)
    sh0 = torch.from_numpy(sh0).float().to(device)

    # Apply the mask if sampling was performed
    if mask is not None:
        means = means[mask]
        sh0 = sh0[mask]

    print(f"Step 3: Calculating RGB colors for {means.shape[0]:,} points...")
    colors = sh2rgb(sh0)

    final_points_xyz = means.cpu().numpy()
    final_colors_rgb = (colors * 255).cpu().to(torch.uint8).numpy()

    print("Step 4: Structuring data for COLMAP...")
    points3D = {}
    for i, (xyz, rgb) in enumerate(zip(final_points_xyz, final_colors_rgb)):
        points3D[i+1] = Point3D( # COLMAP points are 1-indexed
            id=i+1,
            xyz=xyz,
            rgb=rgb,
            error=0.0,
            image_ids=np.array([], dtype=np.int32),
            point2D_idxs=np.array([], dtype=np.int32)
        )

    print("Step 5: Writing points3D.bin file...")
    write_points3D_binary(points3D, output_path)

# --- Example Usage ---
if __name__ == "__main__":
    input_ply_file = "/data/shared/elaheh/4D/4D_scenes/twins_mike/gsplat-frame001-50-undistorted_radial_to_pinhole_mcmc/ply/point_cloud_29999.ply"
    output_file_path = "/data/shared/elaheh/4D/4D_scenes/twins_mike/gsplat-frame001-50-undistorted_radial_to_pinhole_mcmc/bin/points3D.bin"
    
    # Set your desired number of points
    TARGET_POINTS = 1100_000

    print(f"Running conversion for '{input_ply_file}'...")
    convert_and_sample_ply_to_colmap(
        ply_path=input_ply_file,
        output_path=output_file_path,
        target_point_count=TARGET_POINTS
    )
    print("\nConversion complete! ✨")