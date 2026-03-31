#!/bin/bash
# --- Configuration ---
PROJECT_DIR="~/projects/gsplat"
CONDA_ENV_NAME="gsplat"

# Base directory containing the COLMAP data and 'images' folder.
BASE_DIR="/data/shared/elaheh/4D_demo/twin/undistorted/"

# Define the frame range to be processed.
START_FRAME=155
END_FRAME=400 # Adjust as needed

# Root directory for all outputs related to this run.
RESULTS_ROOT_DIR="$(dirname "$BASE_DIR")/res_gsplat_perframe_mrged_ply_gif_default"
AGGREGATE_RENDER_DIR="${RESULTS_ROOT_DIR}/all_renders"
AGGREGATE_PLY_DIR="${RESULTS_ROOT_DIR}/all_plys"
TRACKER_FILE="${RESULTS_ROOT_DIR}/completed_frames.txt"
FAILED_LOG_FILE="${RESULTS_ROOT_DIR}/failed_frames.txt"

# Python script (relative to PROJECT_DIR) and its static arguments
PYTHON_SCRIPT="examples/simple_trainer.py"
STATIC_ARGS=(
    "default"
    "--load_images_in_memory"
    "--disable_viewer"
    "--save_steps" "7000"   "20000"  "30000" "50000"
    "--ply_steps"  "7000"   "20000"  "30000" "50000"
    "--eval_steps" "7000"  "20000"  "30000" "50000"
    "--test_every" "0"
    "--data_factor" "1"
    "--random_bkgd"
    "--strategy.no-verbose"
    "--strategy.refine_stop_iter" "0"
    "--sh_degree" "0"
)

# Define the specific GPU IDs to be used for parallel jobs.
GPU_IDS=(  1 6 7)

# --- Setup ---
eval PROJECT_DIR="$PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "Error: Could not navigate to $PROJECT_DIR"; exit 1; }
echo "Changed directory to $(pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME" || { echo "Error: Could not activate conda environment '$CONDA_ENV_NAME'"; exit 1; }
echo "Activated Conda environment: $CONDA_ENV_NAME"
echo "---"

# Ensure the base directory and output directories/files exist
mkdir -p "$BASE_DIR"
mkdir -p "$RESULTS_ROOT_DIR"
mkdir -p "$AGGREGATE_RENDER_DIR"
mkdir -p "$AGGREGATE_PLY_DIR"
touch "$TRACKER_FILE"
touch "$FAILED_LOG_FILE"

# Load already completed frames into an associative array for fast lookups
declare -A completed_frames_map
if [ -f "$TRACKER_FILE" ]; then
    readarray -t completed_frames_list < "$TRACKER_FILE"
    for frame in "${completed_frames_list[@]}"; do
        if [[ -n "$frame" ]]; then # Ensure we don't process empty lines
            completed_frames_map["$frame"]=1
        fi
    done
fi

echo "Loaded ${#completed_frames_map[@]} completed frames to be skipped."
echo "---"

# --- Script Logic ---
# Create a queue of frames to process
echo "Creating job queue for frames $START_FRAME to $END_FRAME..."
FRAMES_QUEUE=()
for frame_num in $(seq $START_FRAME $END_FRAME); do
    if grep -q -x "$frame_num" "$TRACKER_FILE"; then
        echo "-> Skipping frame '$frame_num' (already completed)."
    else
        FRAMES_QUEUE+=("$frame_num")
    fi
done

TOTAL_FRAMES=${#FRAMES_QUEUE[@]}
if [ $TOTAL_FRAMES -eq 0 ]; then
    echo "No new frames to process. All tasks are complete."
    exit 0
fi

echo "Found $TOTAL_FRAMES new frames to process. Starting job queue..."
echo "---"

# Associative arrays to track job details by their Process ID (PID)
declare -A pids_to_gpu pids_to_frame_num pids_to_start_time pids_to_result_dir
free_gpus=("${GPU_IDS[@]}")
frames_processed_count=0
total_duration=0

# --- Main Loop ---
while [ $frames_processed_count -lt $TOTAL_FRAMES ]; do
    while [ ${#free_gpus[@]} -gt 0 ] && [ ${#FRAMES_QUEUE[@]} -gt 0 ]; do
        gpu_id=${free_gpus[0]}; free_gpus=("${free_gpus[@]:1}")
        frame_num=${FRAMES_QUEUE[0]}; FRAMES_QUEUE=("${FRAMES_QUEUE[@]:1}")

        result_dir="${RESULTS_ROOT_DIR}/frame_${frame_num}"
        mkdir -p "$result_dir"
        LOG_FILE="${result_dir}/gsplat.log"

        echo "🚀 Launching job for frame '$frame_num' on GPU $gpu_id..."

        start_time=$(date +%s)
        CUDA_VISIBLE_DEVICES=$gpu_id python $PYTHON_SCRIPT "${STATIC_ARGS[@]}" \
            --data_dir "$BASE_DIR" \
            --result_dir "$result_dir" \
            --frame_num "$frame_num" \
            --init_type "sfm" > "$LOG_FILE" 2>&1 &
        
        pid=$!
        pids_to_gpu[$pid]=$gpu_id
        pids_to_frame_num[$pid]=$frame_num
        pids_to_start_time[$pid]=$start_time
        pids_to_result_dir[$pid]=$result_dir
    done

    wait -n
    exit_code=$?
    
    finished_pid=""
    for pid in "${!pids_to_gpu[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            finished_pid=$pid; break
        fi
    done

    if [ -z "$finished_pid" ]; then
        if [ ${#pids_to_gpu[@]} -eq 0 ]; then break; fi
        sleep 1; continue
    fi

    gpu_id=${pids_to_gpu[$finished_pid]}
    frame_num=${pids_to_frame_num[$finished_pid]}
    start_time=${pids_to_start_time[$finished_pid]}
    result_dir=${pids_to_result_dir[$finished_pid]}
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    total_duration=$((total_duration + duration))
    frames_processed_count=$((frames_processed_count + 1))

    if [ $exit_code -eq 0 ]; then
        printf "✅ Job for frame '%s' (GPU %d) finished in %d min %d sec. (%d/%d)\n" \
            "$frame_num" "$gpu_id" "$((duration / 60))" "$((duration % 60))" "$frames_processed_count" "$TOTAL_FRAMES"
        echo "$frame_num" >> "$TRACKER_FILE"

        source_render_dir="${result_dir}/renders"
        last_image=$(find "$source_render_dir" -type f \( -name "*.png" -o -name "*.jpg" \) -print0 | sort -z | tail -zn1 | xargs -0)
        if [ -n "$last_image" ]; then
            cp "$last_image" "${AGGREGATE_RENDER_DIR}/frame_${frame_num}.${last_image##*.}"
        else
            echo "⚠️ No render found for frame '$frame_num'."
        fi

        source_ply_dir="${result_dir}/ply"
        last_step_ply_file="${source_ply_dir}/point_cloud_29999.ply"
        if [ -f "$last_step_ply_file" ]; then
            cp "$last_step_ply_file" "${AGGREGATE_PLY_DIR}/frame_${frame_num}.ply"
        else
            echo "⚠️ PLY file not found for frame '$frame_num'."
        fi
    else
        printf "❌ Job for frame '%s' (GPU %d) FAILED with code %d. (%d/%d)\n" \
            "$frame_num" "$gpu_id" "$exit_code" "$frames_processed_count" "$TOTAL_FRAMES"
        echo "$frame_num (exit code: $exit_code)" >> "$FAILED_LOG_FILE"
    fi

    free_gpus+=($gpu_id)
    unset "pids_to_gpu[$finished_pid]" "pids_to_frame_num[$finished_pid]" \
          "pids_to_start_time[$finished_pid]" "pids_to_result_dir[$finished_pid]"
done

# --- Final Report ---
echo "---"
echo "All $TOTAL_FRAMES frames have been processed."
if [ $TOTAL_FRAMES -gt 0 ]; then
    average_duration=$((total_duration / TOTAL_FRAMES))
    printf "📊 Avg Time: %d min %d sec.\n" "$((average_duration / 60))" "$((average_duration % 60))"
fi
printf "Total Time: %d min %d sec.\n" "$((total_duration / 60))" "$((total_duration % 60))"
# Deactivate conda environment
conda deactivate
echo "All tasks are complete."