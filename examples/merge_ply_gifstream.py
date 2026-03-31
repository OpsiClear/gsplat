import os
import shutil
from pathlib import Path

# --- Configuration ---
# !! 1. Set this to your main results directory !!
# (This is the folder that contains GOP_0, GOP_1, etc.)
BASE_RESULTS_DIR = Path("/data/shared/elaheh/4Ddemo_gifstream_results/elly/")

# !! 2. Set this to your desired output folder !!
# (A new folder will be created here)
MERGED_OUTPUT_DIR = BASE_RESULTS_DIR / "merged_ply_sequence_FINAL"

# --- Advanced Configuration (These match your structure) ---
# (You probably don't need to change these based on your 'ls' output)
RUN_SUBFOLDER = "r0"
PLY_FOLDER_NAME = "ply_sequence_29999"
GOP_PREFIX = "GOP_"
# --- End Configuration ---


def merge_gop_ply_files(base_dir, output_dir, run_folder, ply_folder, gop_prefix):
    """
    Finds all GOP folders, sorts them, and merges the .ply files
    from their 'ply_sequence' subfolder into a single output directory
    with continuous sequential numbering.
    """
    
    # 1. Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Created output directory: {output_dir}")

    # 2. Find all GOP directories
    try:
        # Get all directories starting with the prefix
        gop_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(gop_prefix)]
        
        if not gop_dirs:
            print(f"❌ Error: No directories found with prefix '{gop_prefix}' in {base_dir}")
            print("Please check your 'BASE_RESULTS_DIR' and 'GOP_PREFIX' variables.")
            return

        # **Crucial:** Sort GOPs numerically based on the number after '_'
        # This handles GOP_10 correctly (sorts after GOP_9)
        gop_dirs.sort(key=lambda d: int(d.name.split('_')[1]))
        
        print(f"Found {len(gop_dirs)} GOP directories. Processing in this order:")
        for d in gop_dirs:
            print(f"  -> {d.name}")

    except Exception as e:
        print(f"❌ Error finding or sorting GOP directories: {e}")
        print("Please ensure BASE_RESULTS_DIR is correct and GOP folders are named like 'GOP_0', 'GOP_1', etc.")
        return

    # This is the global counter that will not reset
    global_frame_counter = 0
    
    # 3. Loop through each sorted GOP directory
    for gop_dir in gop_dirs:
        # Construct the full path to the .ply files for this GOP
        ply_sequence_path = gop_dir / run_folder / ply_folder
        
        if not ply_sequence_path.exists():
            print(f"\n⚠️ Warning: Path not found, skipping: {ply_sequence_path}")
            continue

        # 4. Find and sort all .ply files within this GOP
        # A simple sort works because they are zero-padded (0000.ply, 0001.ply)
        ply_files = sorted(list(ply_sequence_path.glob("*.ply")))
        
        if not ply_files:
            print(f"\n⚠️ Warning: No .ply files found in: {ply_sequence_path}")
            continue

        print(f"\nProcessing {len(ply_files)} files from {gop_dir.name}...")
        
        # 5. Copy and rename each file
        for src_file_path in ply_files:
            # Format the new filename (e.g., 0000.ply, 0001.ply, ..., 0060.ply, ...)
            # :04d means "pad with zeros to 4 digits"
            target_filename = f"{global_frame_counter:04d}.ply"
            dest_file_path = output_dir / target_filename
            
            # Copy the file
            try:
                shutil.copy(src_file_path, dest_file_path)
            except Exception as e:
                print(f"  ❌ Error copying {src_file_path} to {dest_file_path}: {e}")

            # Increment the global counter for the next file
            global_frame_counter += 1

    print("\n---")
    print(f"🎉 Success! Merged a total of {global_frame_counter} .ply files into:")
    print(f"{output_dir}")

# --- This makes the script runnable from the command line ---
if __name__ == "__main__":
    # Basic check before starting
    if not BASE_RESULTS_DIR.exists():
        print(f"❌ Error: Base directory not found: {BASE_RESULTS_DIR}")
        print("Please update the 'BASE_RESULTS_DIR' variable in the script.")
    else:
        merge_gop_ply_files(
            base_dir=BASE_RESULTS_DIR,
            output_dir=MERGED_OUTPUT_DIR,
            run_folder=RUN_SUBFOLDER,
            ply_folder=PLY_FOLDER_NAME,
            gop_prefix=GOP_PREFIX
        )