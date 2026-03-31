#!/bin/bash
# Batch run simple_trainer.py on the first frame of each indoor dataset.
# Uses 2 GPUs per job (from GPUs 1,2,3,4), runs 2 jobs in parallel.

set -eo pipefail

# Activate conda environment (no -u, conda scripts use unbound vars)
eval "$(conda shell.bash hook 2>/dev/null)" || source /home/elaheh/miniforge3/etc/profile.d/conda.sh
conda activate gsplat

set -u

BASE_DIR="/data/shared/elaheh/4D_demo/completed_indoor"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAINER="$SCRIPT_DIR/simple_trainer.py"

# GPU pairs for 2 concurrent jobs
GPU_PAIR_A="3"
GPU_PAIR_B="4"

# All dataset directories
DATASETS=(
    clarie
    elaheh_tech
    Katie_Grace
    mike_tech
    raghav_rao
    team
    tri_cleaners
    vaibhav_kumar
    winterad_gib
    yehe_tech
)

LOG_DIR="$BASE_DIR/gsplat_logs"
mkdir -p "$LOG_DIR"

# Detect first frame number for a dataset by inspecting the first camera folder
get_first_frame() {
    local data_dir="$1"
    local img_dir="$data_dir/undistorted/images"
    local first_cam
    first_cam=$(ls "$img_dir" | sort | head -1)
    local first_img
    first_img=$(ls "$img_dir/$first_cam/" | sort | head -1)
    # Extract frame number from filename like 000000.jpg -> 0
    echo "${first_img%.*}" | sed 's/^0*//' | sed 's/^$/0/'
}

run_one() {
    local dataset="$1"
    local gpus="$2"

    local data_dir="$BASE_DIR/$dataset/undistorted"
    local frame_num
    frame_num=$(get_first_frame "$BASE_DIR/$dataset")

    # Format result folder name: gsplat_frame000000 or gsplat_frame000001
    local frame_tag
    frame_tag=$(printf "gsplat_frame%06d" "$frame_num")
    local result_dir="$data_dir/$frame_tag"

    # Skip if already completed
    if [ -d "$result_dir/ply" ] && ls "$result_dir/ply/"*.ply &>/dev/null; then
        echo "[SKIP] $dataset — already has PLY in $result_dir/ply"
        return 0
    fi

    echo "[START] $dataset  frame=$frame_num  gpus=$gpus  result=$result_dir"

    CUDA_VISIBLE_DEVICES="$gpus" python "$TRAINER" default \
        --data_dir "$data_dir" \
        --result_dir "$result_dir" \
        --frame_num "$frame_num" \
        --max_steps 30000 \
        --save_ply \
        --ply_steps 29999 \
        --save_steps 29999 \
        --eval_steps 29999 \
        --disable_viewer \
        --test_every 100 \
        --steps_scaler 1.0 \
        > "$LOG_DIR/${dataset}.log" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[DONE]  $dataset"
    else
        echo "[FAIL]  $dataset  (exit code $rc, see $LOG_DIR/${dataset}.log)"
    fi
    return $rc
}

# Process datasets in pairs of 2 (one per GPU pair)
idx=0
total=${#DATASETS[@]}

while [ $idx -lt $total ]; do
    pids=()

    # Launch job A
    if [ $idx -lt $total ]; then
        run_one "${DATASETS[$idx]}" "$GPU_PAIR_A" &
        pids+=($!)
        idx=$((idx + 1))
    fi

    # Launch job B
    if [ $idx -lt $total ]; then
        run_one "${DATASETS[$idx]}" "$GPU_PAIR_B" &
        pids+=($!)
        idx=$((idx + 1))
    fi

    # Wait for both to finish before starting the next pair
    for pid in "${pids[@]}"; do
        wait "$pid" || true
    done
done

echo ""
echo "=== All done ==="
echo "Logs: $LOG_DIR/"
