# --- Configuration ---
PROJECT_DIR="~/projects/gsplat"
CONDA_ENV_NAME="gsplat"

# Base directory containing the 'scans' folder and where results will be organized.
BASE_DIR="/mnt/OpsiClearNas1/softbox_scan/4_features_and_trianglization/orange_mold_timelapse"
SCANS_DIR="${BASE_DIR}"

# Output and tracking files will be located at the root of BASE_DIR.
AGGREGATE_RENDER_DIR="${BASE_DIR}/all_sequential_renders"
AGGREGATE_PLY_DIR="${BASE_DIR}/all_sequential_plys"
TRACKER_FILE="${BASE_DIR}/sequential_completed_scans.txt"
FAILED_LOG_FILE="${BASE_DIR}/sequential_failed_scans.txt"
LOCK_FILE="${BASE_DIR}/.script.lock"

# --- Script Behavior ---
# RESET_RATE: Determines how often to start a new chain from scratch. 
# A value of 40 means scans 0, 40, 80, etc., will be fresh runs. 
# A value of 0 means only the very first scan is run from scratch, and all others are chained.
RESET_RATE=40

# RESUME_STEP: The step number to start from when resuming training from a chained checkpoint.
RESUME_STEP=20000

# Python script (relative to PROJECT_DIR) and its static arguments
PYTHON_SCRIPT="examples/simple_trainer_sequential.py"
STATIC_ARGS="mcmc \
--load_images_in_memory \
--optimize_foreground \
--use_masks \
--disable_viewer \
--save_steps 30000 \
--ply_steps 30000 \
--eval_steps 30000 \
--test_every 0 \
--data_factor 1 \
--random_bkgd \
--strategy.no-verbose \
--no-normalize-world-space \
--exclude_prefixes cam_8"

# Define the specific GPU IDs to be used for parallel jobs.
# This allows for precise control over which GPUs are utilized.
# Example for using GPUs 0, 1, 4, and 7: GPU_IDS=(0 1 4 7)
GPU_IDS=(7 5 4 3 2)

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
# Find all scan directories, sort them to ensure sequential order, and then filter out completed ones.
echo "Finding all potential scans..."
all_scans_found=()
while IFS= read -r d; do
    all_scans_found+=("$d")
# The `sort` command ensures we process scans in their natural alphanumeric order.
done < <(find "$SCANS_DIR" -maxdepth 1 -type d -name "scan_*" | sort)

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

# --- Main Execution ---

# This function processes a single chain of scans sequentially on a given GPU.
# It's designed to be run as a background job for parallel execution.
process_chain() {
    local gpu_id=$1
    local chain_idx=$2
    shift 2
    local chain_scans=("$@")
    
    echo "--- Starting Chain $chain_idx on GPU $gpu_id ---"
    
    for (( i=0; i<${#chain_scans[@]}; i++ )); do
        local data_dir=${chain_scans[$i]}
        local scan_name=$(basename "$data_dir")
        local result_dir="${data_dir}/3DGS_sequential"
        
        local resume_args=""
        # The first scan in any chain always starts from scratch
        if [ $i -eq 0 ]; then
            echo "🔥 (Chain $chain_idx) Starting '$scan_name' from scratch on GPU $gpu_id."
            rm -rf "$result_dir"
        else
            # Find the checkpoint from the previous scan in this chain
            local previous_data_dir=${chain_scans[$((i-1))]}
            local previous_result_dir="${previous_data_dir}/3DGS_sequential"
            local ckpt_dir="${previous_result_dir}/ckpts"
            
            if [ -d "$ckpt_dir" ]; then
                local latest_ckpt=$(find "$ckpt_dir" -type f -name "ckpt_*.pt" | sort -V | tail -n 1)
                if [ -n "$latest_ckpt" ]; then
                    echo "⛓️ (Chain $chain_idx) Chaining '$scan_name' from '$latest_ckpt' on GPU $gpu_id."
                    resume_args="--resume_from_ckpt $latest_ckpt --resume_from_step $RESUME_STEP"
                else
                    echo "❌ ERROR (Chain $chain_idx): Could not find checkpoint in '$ckpt_dir'. Chain broken. Skipping '$scan_name'." >&2
                    continue
                fi
            else
                echo "❌ ERROR (Chain $chain_idx): Checkpoint dir '$ckpt_dir' not found. Chain broken. Skipping '$scan_name'." >&2
                continue
            fi
        fi
        
        mkdir -p "$result_dir"
        local LOG_FILE="${data_dir}/gsplat_parallel.log"
        
        echo "🚀 (Chain $chain_idx) Launching job for '$scan_name' on GPU $gpu_id..."
        local start_time=$(date +%s)
        
        CUDA_VISIBLE_DEVICES=$gpu_id python $PYTHON_SCRIPT $STATIC_ARGS $resume_args --data_dir "$data_dir" --result_dir "$result_dir" > "$LOG_FILE" 2>&1
        local exit_code=$?
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Use flock on a dedicated lock file for safe concurrent writing
        flock "$LOCK_FILE" -c "
            if [ $exit_code -eq 0 ]; then
                printf '✅ (Chain %d) Job for scan '\''%s'\'' on GPU %d finished SUCCESSFULLY in %d min %d sec.\n' \
                    '$chain_idx' '$scan_name' '$gpu_id' '$((duration / 60))' '$((duration % 60))'
                echo '$scan_name' >> '$TRACKER_FILE'

                # --- Aggregate results ---
                echo '📸 Copying representative render and PLY for '\''$scan_name'\''...'
                
                # Copy render
                source_render_dir=\"${result_dir}/renders\"
                last_image=\$(find \"\$source_render_dir\" -type f \\( -name \"*.png\" -o -name \"*.jpg\" -o -name \"*.jpeg\" \\) -print0 | sort -z | tail -zn1 | xargs -0)
                if [ -n \"\$last_image\" ]; then
                    extension=\"\${last_image##*.}\"
                    destination_path=\"${AGGREGATE_RENDER_DIR}/${scan_name}.\${extension}\"
                    cp \"\$last_image\" \"\$destination_path\"
                fi

                # Copy PLY
                source_ply_dir=\"${result_dir}/ply\"
                last_ply_file=\$(find \"\$source_ply_dir\" -type f -name \"*.ply\" -print0 | sort -z | tail -zn1 | xargs -0)
                if [ -n \"\$last_ply_file\" ]; then
                    destination_path=\"${AGGREGATE_PLY_DIR}/${scan_name}.ply\"
                    cp \"\$last_ply_file\" \"\$destination_path\"
                fi

            else
                printf '❌ (Chain %d) Job for scan '\''%s'\'' on GPU %d FAILED with exit code %d after %d min %d sec.\n' \
                    '$chain_idx' '$scan_name' '$gpu_id' '$exit_code' '$((duration / 60))' '$((duration % 60))'
                echo '$scan_name (exit code: $exit_code)' >> '$FAILED_LOG_FILE'
            fi
        "
        
        # If a job in the chain fails, stop processing the rest of the chain.
        if [ $exit_code -ne 0 ]; then
            echo "--- Chain $chain_idx on GPU $gpu_id broken due to failure. Stopping this chain. ---" >&2
            # The function will naturally exit, but we could also 'return 1' here.
            break 
        fi
    done
    echo "--- Chain $chain_idx on GPU $gpu_id Finished ---"
}

# 1. Group scans into chains based on RESET_RATE
chains=()
if [ ${#SCANS_QUEUE[@]} -gt 0 ]; then
    current_chain=()
    # Special case: RESET_RATE=0 means one single chain
    if [ $RESET_RATE -eq 0 ]; then
        chains+=("${SCANS_QUEUE[*]}")
    else
        for (( i=0; i<${#SCANS_QUEUE[@]}; i++ )); do
            # If it's a reset scan (and not the very first scan), end the current chain and start a new one.
            if [ $i -gt 0 ] && [ $((i % RESET_RATE)) -eq 0 ]; then
                chains+=("${current_chain[*]}")
                current_chain=()
            fi
            current_chain+=("${SCANS_QUEUE[$i]}")
        done
        # Add the last chain
        chains+=("${current_chain[*]}")
    fi
fi

num_chains=${#chains[@]}
num_gpus=${#GPU_IDS[@]}
echo "Discovered $num_chains chains to process on $num_gpus GPUs."

# 2. Smartly dispatch chains
if [ $num_chains -le 1 ] || [ $num_gpus -le 1 ]; then
    # --- Sequential Execution ---
    echo "🏃‍♂️ Running in SEQUENTIAL mode (single chain or single GPU)."
    gpu_id=${GPU_IDS[0]}
    # The first (and only) chain is all scans if chains array has one element
    read -r -a chain_to_run <<< "${chains[0]}"
    process_chain "$gpu_id" "0" "${chain_to_run[@]}"
else
    # --- Parallel Execution ---
    echo "🚀 Running in PARALLEL mode."
    
    # Associative array to track chain PIDs and their assigned GPU
    declare -A pids_to_gpu
    # Array to manage the queue of chains to be processed
    chain_queue=("${chains[@]}")
    # Array to manage available GPUs
    free_gpus=("${GPU_IDS[@]}")
    
    pids=() # To store all spawned PIDs for the final wait
    chain_idx_counter=0

    while [ ${#chain_queue[@]} -gt 0 ]; do
      # Launch new jobs if there are free GPUs and chains in the queue
      while [ ${#free_gpus[@]} -gt 0 ] && [ ${#chain_queue[@]} -gt 0 ]; do
        gpu_id=${free_gpus[0]}
        free_gpus=("${free_gpus[@]:1}") # Dequeue GPU
        
        current_chain_str=${chain_queue[0]}
        chain_queue=("${chain_queue[@]:1}") # Dequeue chain
        read -r -a current_chain_arr <<< "$current_chain_str"

        echo "Dispatching Chain $chain_idx_counter to GPU $gpu_id..."
        process_chain "$gpu_id" "$chain_idx_counter" "${current_chain_arr[@]}" &
        
        pid=$!
        pids+=($pid)
        pids_to_gpu[$pid]=$gpu_id
        chain_idx_counter=$((chain_idx_counter + 1))
      done

      # Wait for any background job to finish
      if [ ${#pids[@]} -gt 0 ]; then
        wait -n
        
        # Find which job (PID) finished and release its GPU
        for pid_to_check in "${!pids_to_gpu[@]}"; do
            if ! kill -0 "$pid_to_check" 2>/dev/null; then
                finished_pid=$pid_to_check
                gpu_id=${pids_to_gpu[$finished_pid]}
                echo "GPU $gpu_id is now free."
                free_gpus+=($gpu_id)
                unset "pids_to_gpu[$finished_pid]"
                break
            fi
        done
      fi
    done
fi

# --- Final Report ---
echo "---"
echo "All $TOTAL_SCANS scans have been processed."
average_duration=$((total_duration / TOTAL_SCANS))
printf "📊 Average job time: %d minutes and %d seconds.\n" "$((average_duration / 60))" "$((average_duration % 60))"
printf "Total time: %d minutes and %d seconds.\n" "$((total_duration / 60))" "$((total_duration % 60))"
# Deactivate conda environment
conda deactivate
echo "All tasks are complete."