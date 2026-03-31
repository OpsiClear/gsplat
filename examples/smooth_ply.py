import torch
import numpy as np
import glob
import os
import concurrent.futures
from tqdm import tqdm
from scipy.spatial import KDTree
from gsplat.exporter import import_splats, export_splats # Assuming you have this

# --- Configuration ---

# 1. Define your input and output directories
PLY_DIR = "/data/shared/elaheh/4D_demo/case_4/res_gsplat_perframe_from_gifsteam/point_cloud_29999/"       # Directory containing your N .ply files
OUTPUT_DIR = "/data/shared/elaheh/4D_demo/case_4/res_gsplat_perframe_from_gifsteam/smoothed_point_cloud_29999_k3_m7/"   # Directory to save the new .ply files

# 2. Define smoothing parameters
K_TEMPORAL = 3   # Temporal window size (k=2 means we look at t-2, t-1, t+1, t+2)
M_SPATIAL = 7    # Number of spatial neighbors to average

# 3. Parallelization
N_WORKERS = 12 # Set to None to use all available cores (os.cpu_count())

# 4. Define device
DEVICE = "cpu"   # "cpu" is recommended for multiprocessing

# ---------------------
# Helper functions
# ---------------------

def filter_splats_by_bbox(splat_data, min_b, max_b):
    """Filters all splat properties based on a bounding box."""
    means, scales, quats, opacities, sh0, shN = splat_data
    
    mask = (means >= min_b).all(dim=1) & (means <= max_b).all(dim=1)
    
    if mask.shape[0] == 0:
        return splat_data
        
    return (
        means[mask],
        scales[mask],
        quats[mask],
        opacities[mask],
        sh0[mask],
        shN[mask]
    )

def quaternion_mean(quats):
    """Calculates the average of a set of quaternions."""
    avg_quats = quats.mean(dim=1)
    norm = torch.norm(avg_quats, p=2, dim=1, keepdim=True)
    avg_quats = avg_quats / norm.clamp(min=1e-6)
    return avg_quats

# ---------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    ply_files = sorted(glob.glob(os.path.join(PLY_DIR, "*.ply")))
    if not ply_files:
        print(f"Error: No .ply files found in {PLY_DIR}")
        return
        
    num_frames = len(ply_files)
    print(f"Found {num_frames} PLY files.")

    # --- Step 1: Load all splats and filter by BBox (Sequential) ---
    print("Loading all splat data...")
    all_splat_data = []
    
    try:
        means_0, _, _, _, _, _ = import_splats(ply_files[0], device=DEVICE)
        if means_0.shape[0] == 0:
            print("Error: First frame is empty, cannot determine bounding box.")
            return
        min_bound = means_0.min(dim=0).values
        max_bound = means_0.max(dim=0).values
        print(f"Using bounding box from frame 0:")
        print(f"  Min: {min_bound.cpu().numpy()}")
        print(f"  Max: {max_bound.cpu().numpy()}")
    except Exception as e:
        print(f"Error loading first frame {ply_files[0]}: {e}")
        return

    for ply_path in ply_files:
        try:
            splat_data = import_splats(ply_path, device=DEVICE)
            filtered_data = filter_splats_by_bbox(splat_data, min_bound, max_bound)
            all_splat_data.append(filtered_data)
        except Exception as e:
            print(f"Warning: Could not load or filter {ply_path}: {e}")
            all_splat_data.append(( # Add empty tensors
                torch.empty(0, 3, device=DEVICE), torch.empty(0, 3, device=DEVICE),
                torch.empty(0, 4, device=DEVICE), torch.empty(0, device=DEVICE),
                torch.empty(0, 1, 3, device=DEVICE), torch.empty(0, 0, 3, device=DEVICE)
            ))
            
    # --- Step 2: Define the Worker Function ---
    
    def process_frame(t):
        """
        Worker function to process and save a single frame 't'.
        """
        try:
            current_data = all_splat_data[t]
            current_means = current_data[0]
            output_path = os.path.join(OUTPUT_DIR, os.path.basename(ply_files[t]))

            if current_means.shape[0] == 0:
                return f"Frame {t}: Empty, skipped"

            # --- Build neighbor dataset ---
            t_start = max(0, t - K_TEMPORAL)
            t_end = min(num_frames, t + K_TEMPORAL + 1)
            
            neighbor_frames_data = []
            for j in range(t_start, t_end):
                if j == t: continue
                if all_splat_data[j][0].shape[0] > 0:
                    neighbor_frames_data.append(all_splat_data[j])
            
            # --- Decide: Smooth or Save As-Is ---
            if not neighbor_frames_data:
                # No neighbors, just save the original (filtered) frame
                (means, scales, quats, opacities, sh0, shN) = current_data
            else:
                # --- We have neighbors, perform smoothing ---
                try:
                    neighbor_means = torch.cat([d[0] for d in neighbor_frames_data], dim=0)
                    neighbor_scales = torch.cat([d[1] for d in neighbor_frames_data], dim=0)
                    neighbor_quats = torch.cat([d[2] for d in neighbor_frames_data], dim=0)
                    neighbor_opacities = torch.cat([d[3] for d in neighbor_frames_data], dim=0)
                    neighbor_sh0 = torch.cat([d[4] for d in neighbor_frames_data], dim=0)
                    neighbor_shN = torch.cat([d[5] for d in neighbor_frames_data], dim=0)
                except Exception as e:
                    (means, scales, quats, opacities, sh0, shN) = current_data
                else:
                    # --- KNN Search ---
                    neighbor_kdtree = KDTree(neighbor_means.cpu().numpy())
                    distances, indices = neighbor_kdtree.query(
                        current_means.cpu().numpy(), k=M_SPATIAL, workers=-1
                    )
                    indices_tensor = torch.tensor(indices, device=DEVICE)

                    # --- Gather and average properties ---
                    avg_scales = neighbor_scales[indices_tensor].mean(dim=1)
                    avg_opacities = neighbor_opacities[indices_tensor].mean(dim=1)
                    avg_sh0 = neighbor_sh0[indices_tensor].mean(dim=1)
                    avg_shN = neighbor_shN[indices_tensor].mean(dim=1)
                    avg_quats = quaternion_mean(neighbor_quats[indices_tensor])

                    # Final smoothed data
                    (means, scales, quats, opacities, sh0, shN) = (
                        current_means, avg_scales, avg_quats, avg_opacities, avg_sh0, avg_shN
                    )

            # --- Step 3: Save the file ---
            
            # ################################################
            # ## THIS IS THE CORRECTED SECTION ##
            # ################################################
            #
            # Passing all averaged tensors directly, as you specified.
            
            export_splats(
                means=means,
                scales=scales,       # <-- CORRECTED: No torch.log()
                quats=quats,
                opacities=opacities,
                sh0=sh0,
                shN=shN,
                save_to=output_path,
            )
            return f"Frame {t}: Processed"
            
        except Exception as e:
            return f"Frame {t}: FAILED ({e})"

    # --- Step 4: Run the Parallel Pool ---
    frame_indices = list(range(num_frames))
    
    print(f"Starting parallel smoothing with {N_WORKERS or os.cpu_count()} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(tqdm(
            executor.map(process_frame, frame_indices), 
            total=num_frames,
            desc="Smoothing Frames"
        ))

    print("--- Processing Complete ---")
    for res in results:
        if "FAILED" in res or "Warning" in res:
            print(res)
    print(f"Done. Smoothed files are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()