# --- Configuration ---
PROJECT_DIR="~/projects/gsplat"
CONDA_ENV_NAME="rade"

# Base directory containing the 'scans' folder and where results will be organized.
BASE_DIR="/mnt/OpsiClearNas1/softbox_scan/P5_3DGS/20251028_Yosef_mix_of_toys/"
SCANS_DIR="${BASE_DIR}"

# Output and tracking files will be located at the root of BASE_DIR.
AGGREGATE_RENDER_DIR="${BASE_DIR}/all_renders"
AGGREGATE_PLY_DIR="${BASE_DIR}/all_splats"
AGGREGATE_MESH_DIR="${BASE_DIR}/all_meshes"
TRACKER_FILE="${BASE_DIR}/completed_scans.txt"
FAILED_LOG_FILE="${BASE_DIR}/failed_scans.txt"

# Python script (relative to PROJECT_DIR) and its static arguments
PYTHON_SCRIPT="examples/simple_trainer.py"
STATIC_ARGS="default \
--load_images_in_memory \
--load_images_to_gpu \
--optimize_foreground \
--use_masks \
--disable_viewer \
--disable_video \
--save_steps 30000 \
--ply_steps 30000 \
--eval_steps 30000 \
--test_every 0 \
--data_factor 1 \
--random_bkgd \
--strategy.no-verbose \
--use_rade"
# --use_bilateral_grid \
# --use_fused_bilagrid"
# --pose_opt
# --exclude_prefixes cam_8"

# Define the specific GPU IDs to be used for parallel jobs.
# This allows for precise control over which GPUs are utilized.
# Example for using GPUs 0, 1, 4, and 7: GPU_IDS=(0 1 4 7)
GPU_IDS=(7 6 5 4 3 2)

# --- Setup ---
# Expand the tilde (~) to the full home directory path
eval PROJECT_DIR="$PROJECT_DIR"

# Navigate to the project directory
cd "$PROJECT_DIR" || { echo "Error: Could not navigate to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

# Initialize Conda for shell scripting
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the conda environment
conda activate "$CONDA_ENV_NAME" || { echo "Error: Could not activate conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Activated Conda environment: $CONDA_ENV_NAME"
echo "---"

# Ensure the base directory and output directories/files exist
mkdir -p "$SCANS_DIR"
mkdir -p "$AGGREGATE_RENDER_DIR"
mkdir -p "$AGGREGATE_PLY_DIR"
mkdir -p "$AGGREGATE_MESH_DIR"
touch "$TRACKER_FILE"
touch "$FAILED_LOG_FILE"

# Load already completed scans into an associative array for fast lookups
declare -A completed_scans_map
readarray -t completed_scans_list < "$TRACKER_FILE"
for scan in "${completed_scans_list[@]}"; do
    completed_scans_map["$scan"]=1
done

echo "Loaded ${#completed_scans_map[@]} completed scans to be skipped."
echo "---"

# --- Script Logic ---
# Find all scan directories, then filter out the completed ones
echo "Finding all potential scans..."
all_scans_found=()
while IFS= read -r d; do
    all_scans_found+=("$d")
done < <(find "$SCANS_DIR" -maxdepth 1 -type d -name "scan_*")

SCANS_QUEUE=()
for data_dir in "${all_scans_found[@]}"; do
    scan_name=$(basename "$data_dir")
    if [[ -v completed_scans_map["$scan_name"] ]]; then
        echo "-> Skipping '$scan_name' (already completed)."
    else
        SCANS_QUEUE+=("$data_dir")
    fi
done

# Update TOTAL_SCANS to reflect only the ones we will run now
TOTAL_SCANS=${#SCANS_QUEUE[@]}

if [ $TOTAL_SCANS -eq 0 ]; then
  echo "No new scans to process. All tasks are complete."
  exit 0
fi

echo "Found $TOTAL_SCANS new scans to process. Starting job queue..."
echo "---"

# Associative arrays to track job details by their Process ID (PID)
declare -A pids_to_gpu
declare -A pids_to_scan_name
declare -A pids_to_start_time
declare -A pids_to_result_dir

# Array to manage available GPUs
free_gpus=("${GPU_IDS[@]}")

# --- Cleanup on Exit ---
# Trap signals to ensure that if the script is terminated, all background Python jobs are also killed.
function cleanup() {
    echo ""
    echo "---"
    echo "Caught exit signal. Cleaning up background processes..."
    # The keys of pids_to_gpu are the PIDs of the running jobs
    pids=("${!pids_to_gpu[@]}")
    if [ ${#pids[@]} -gt 0 ]; then
        echo "Terminating the following PIDs: ${pids[*]}"
        # Send SIGTERM to all child processes
        kill "${pids[@]}" 2>/dev/null
        # Wait for them to terminate
        wait
    else
        echo "No background processes to terminate."
    fi

    echo "Deactivating Conda environment..."
    conda deactivate
    echo "Cleanup complete."
}
trap cleanup EXIT SIGINT SIGTERM

# --- Main Loop ---
scans_processed_count=0
total_duration=0

while [ $scans_processed_count -lt $TOTAL_SCANS ]; do
  # Launch new jobs if there are free GPUs and scans in the queue
  while [ ${#free_gpus[@]} -gt 0 ] && [ ${#SCANS_QUEUE[@]} -gt 0 ]; do
    # Get a free GPU and a scan from the queues
    gpu_id=${free_gpus[0]}
    free_gpus=("${free_gpus[@]:1}") # Dequeue GPU
    
    data_dir=${SCANS_QUEUE[0]}
    SCANS_QUEUE=("${SCANS_QUEUE[@]:1}") # Dequeue scan

    scan_name=$(basename "$data_dir")
    # Results will be saved inside the scan folder in a '3DGS' subdirectory.
    result_dir="${data_dir}/3DGS"
    rm -rf "$result_dir"
    mkdir -p "$result_dir"

    echo "🚀 Launching job for scan '$scan_name' on GPU $gpu_id..."
    LOG_FILE="${data_dir}/gsplat.log"

    # Record start time and run the command in the background
    start_time=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu_id python $PYTHON_SCRIPT $STATIC_ARGS --data_dir "$data_dir" --result_dir "$result_dir" > "$LOG_FILE" 2>&1 &
    
    # Store the new job's PID and its associated data
    pid=$!
    pids_to_gpu[$pid]=$gpu_id
    pids_to_scan_name[$pid]=$scan_name
    pids_to_start_time[$pid]=$start_time
    pids_to_result_dir[$pid]=$result_dir
  done

  # Wait for any background job to finish and capture its exit code
  wait -n
  exit_code=$?
  
  # Now, find which job PID has finished since `wait -n` doesn't tell us
  finished_pid=""
  for pid in "${!pids_to_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
          finished_pid=$pid
          break
      fi
  done

  if [ -z "$finished_pid" ]; then
      # This can happen if all jobs finish at once at the end
      if [ ${#pids_to_gpu[@]} -eq 0 ]; then break; fi
      sleep 1; continue
  fi

  # --- Process the finished job ---
  # Retrieve job details
  gpu_id=${pids_to_gpu[$finished_pid]}
  scan_name=${pids_to_scan_name[$finished_pid]}
  start_time=${pids_to_start_time[$finished_pid]}
  result_dir=${pids_to_result_dir[$finished_pid]}
  
  # Calculate duration
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  total_duration=$((total_duration + duration))
  scans_processed_count=$((scans_processed_count + 1))

  # Check the exit code to determine success or failure
  if [ $exit_code -eq 0 ]; then
      # Success
      printf "✅ Job for scan '%s' (GPU %d) finished SUCCESSFULLY in %d min %d sec. (%d/%d complete)\n" \
          "$scan_name" "$gpu_id" "$((duration / 60))" "$((duration % 60))" "$scans_processed_count" "$TOTAL_SCANS"
      
      # Add the scan name to the tracker file on success
      echo "$scan_name" >> "$TRACKER_FILE"

      # Copy the last render image to the aggregate directory
      echo "📸 Copying representative render for '$scan_name'..."
      source_render_dir="${result_dir}/renders"
      last_image=$(find "$source_render_dir" -type f \( -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" \) -print0 | sort -z | tail -zn1 | xargs -0)

      if [ -n "$last_image" ]; then
          # Get the file extension and copy the file
          extension="${last_image##*.}"
          destination_path="${AGGREGATE_RENDER_DIR}/${scan_name}.${extension}"
          cp "$last_image" "$destination_path"
          echo "-> Copied to $destination_path"
      else
          echo "⚠️  Warning: No render image found for scan '$scan_name' in '$source_render_dir'."
      fi

      # Copy the last ply file to the aggregate directory
      echo "📸 Copying PLY file for '$scan_name'..."
      source_ply_dir="${result_dir}/ply"
      last_ply_file=$(find "$source_ply_dir" -type f -name "*.ply" -print0 | sort -z | tail -zn1 | xargs -0)
      destination_path="${AGGREGATE_PLY_DIR}/${scan_name}.ply"
      if [ -n "$last_ply_file" ]; then
          cp "$last_ply_file" "$destination_path"
          echo "-> Copied to $destination_path"
      else
          echo "⚠️  Warning: No PLY file found for scan '$scan_name' in '$source_ply_dir'."
      fi

      # Copy the mesh file if it exists
      echo "📦 Copying mesh file for '$scan_name'..."
      source_mesh_file="${result_dir}/mesh/recon_29999.ply"
      if [ -f "$source_mesh_file" ]; then
          destination_path="${AGGREGATE_MESH_DIR}/${scan_name}.ply"
          cp "$source_mesh_file" "$destination_path"
          echo "-> Copied to $destination_path"
      else
          echo "⚠️  Warning: No mesh file 'recon_29999.ply' found for scan '$scan_name' in '${result_dir}/mesh'."
      fi

      
  else
      # Failure
      printf "❌ Job for scan '%s' (GPU %d) FAILED with exit code %d after %d min %d sec. (%d/%d complete)\n" \
          "$scan_name" "$gpu_id" "$exit_code" "$((duration / 60))" "$((duration % 60))" "$scans_processed_count" "$TOTAL_SCANS"
      
      # Optional: Log failures to a separate file for later inspection
      echo "$scan_name (exit code: $exit_code)" >> "$FAILED_LOG_FILE"
  fi

  # Free up the GPU and remove the PID from tracking
  free_gpus+=($gpu_id)
  unset "pids_to_gpu[$finished_pid]"
  unset "pids_to_scan_name[$finished_pid]"
  unset "pids_to_start_time[$finished_pid]"
  unset "pids_to_result_dir[$finished_pid]"
done

# --- Final Report ---
echo "---"
echo "All $TOTAL_SCANS scans have been processed."
average_duration=$((total_duration / TOTAL_SCANS))
printf "📊 Average job time: %d minutes and %d seconds.\n" "$((average_duration / 60))" "$((average_duration % 60))"
printf "Total time: %d minutes and %d seconds.\n" "$((total_duration / 60))" "$((total_duration % 60))"
echo "All tasks are complete."