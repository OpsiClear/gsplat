import os
import shutil
import glob

def aggregate_single_ply(base_dir, target_ply_name, frame_folders):
    """
    Aggregates one specific .ply file from all frame folders
    into its own dedicated directory.
    """
    print(f"\n--- Processing Target: {target_ply_name} ---")

    if not target_ply_name.endswith('.ply'):
        print(f"  [ERROR] Invalid name: {target_ply_name}. Skipping.")
        return

    # --- 1. Create Output Directory (e.g., "point_cloud_9999") ---
    output_folder_name = os.path.splitext(target_ply_name)[0]
    output_dir = os.path.join(base_dir, output_folder_name)
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✅ Output directory: {output_dir}")
    except OSError as e:
        print(f"  [ERROR] Could not create directory {output_dir}. {e}. Skipping.")
        return

    # --- 2. Iterate over all frames, Copy, and Rename ---
    copied_count = 0
    skipped_count = 0
    
    for frame_path in frame_folders:
        if not os.path.isdir(frame_path):
            continue
            
        try:
            # a. Extract frame number (e.g., "frame_12" -> 12)
            frame_name = os.path.basename(frame_path)
            frame_num_str = frame_name.split('_')[-1]
            frame_num = int(frame_num_str)
            
            # b. Define source file path
            source_file = os.path.join(frame_path, 'ply', target_ply_name)
            
            # c. Define destination file path (e.g., "0012.ply")
            dest_name = f"{frame_num:04d}.ply"
            dest_file = os.path.join(output_dir, dest_name)
            
            # d. Check if source exists and copy
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                copied_count += 1
            else:
                # File not found in this specific frame folder
                skipped_count += 1
                
        except ValueError:
            # Could not parse frame number from folder name
            skipped_count += 1
        except Exception as e:
            print(f"    [ERROR] Failed to process {frame_name}: {e}")
            skipped_count += 1

    # --- 3. Final Summary for this file ---
    print(f"  ➡️  Summary: {copied_count} files copied, {skipped_count} skipped/not found.")


def main():
    """
    Main function to find all frame folders and loop through all
    target .ply files to aggregate them.
    """
    
    # --- 1. Configuration ---
    
    base_dir = "/data/shared/elaheh/4D_demo/case_4/res_gsplat_perframe_mrged_ply_gif_default/"
    
    # === EDIT THIS LIST ===
    # Add all the .ply filenames you want to process here
    # target_ply_list = [
        
    #     'point_cloud_4999.ply',
    #     'point_cloud_9999.ply',
    #     'point_cloud_14999.ply',
    #     'point_cloud_19999.ply',
    #     'point_cloud_24999.ply',
    #     'point_cloud_29999.ply',
    #     # 'point_cloud_6999.ply', # <-- Example: uncomment to add more
    # ]
    target_ply_list = [
        'point_cloud_29999.ply',
    ]
    # =======================
    
    print(f"Starting batch aggregation for {len(target_ply_list)} target file(s)...")
    print(f"Base directory: {base_dir}")
    
    # --- 2. Find Frame Folders (Do this only *once*) ---
    search_pattern = os.path.join(base_dir, "frame_*")
    frame_folders = sorted(glob.glob(search_pattern))
    
    if not frame_folders:
        print(f"❌ CRITICAL ERROR: No 'frame_*' directories found in {base_dir}")
        print("Aborting.")
        return
        
    print(f"Found {len(frame_folders)} frame directories to scan.")

    # --- 3. Loop over each target file and process it ---
    for target_name in target_ply_list:
        aggregate_single_ply(base_dir, target_name, frame_folders)
        
    print("\n--- All Batch Aggregation Complete ---")

if __name__ == "__main__":
    main()