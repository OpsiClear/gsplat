#!/bin/bash
# Multi-GPU per-frame fine-tuning queue.
#
# Distributes frames across multiple GPUs, running one frame per GPU at a time.
# When a GPU finishes, it picks the next available frame.
#
# Usage:
#   bash examples/run_finetune_multi_gpu.sh
#
# Or override defaults:
#   bash examples/run_finetune_multi_gpu.sh --gpus "3 4 5" --start_frame 0 --end_frame 999

set -e

PYTHON="/home/elaheh/miniforge3/envs/gsplat/bin/python"

# ---- Defaults ----
GPUS="3 4 5"
START_FRAME=0
END_FRAME=999
PLY_DIR="/data/shared/elaheh/final_4d_results/merge_ply_all_scenes/thenewface/ply_sequence_merged_40000_merged"
DATA_DIR="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted"
RESULT_BASE="/data/shared/elaheh/newface_finetune"
MAX_STEPS=10000
IMAGE_FRAME_OFFSET=1
DATA_FACTOR=1
TEST_EVERY=-1
BBOX_FILTER=true
OPA_CLAMP=""

# ---- Parse CLI arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)          GPUS="$2";              shift 2;;
        --start_frame)   START_FRAME="$2";       shift 2;;
        --end_frame)     END_FRAME="$2";         shift 2;;
        --ply_dir)       PLY_DIR="$2";           shift 2;;
        --data_dir)      DATA_DIR="$2";          shift 2;;
        --result_base)   RESULT_BASE="$2";       shift 2;;
        --max_steps)     MAX_STEPS="$2";         shift 2;;
        --offset)        IMAGE_FRAME_OFFSET="$2"; shift 2;;
        --data_factor)   DATA_FACTOR="$2";       shift 2;;
        --test_every)    TEST_EVERY="$2";        shift 2;;
        --no-bbox-filter) BBOX_FILTER=false;    shift 1;;
        --opa_clamp)     OPA_CLAMP="$2";       shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Convert GPU string to array
read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

echo "============================================"
echo "  Multi-GPU Per-frame Fine-tuning with PPISP"
echo "============================================"
echo "GPUs:         ${GPUS} (${NUM_GPUS} GPUs)"
echo "Frames:       ${START_FRAME} -> ${END_FRAME}"
echo "PLY dir:      ${PLY_DIR}"
echo "Data dir:     ${DATA_DIR}"
echo "Results:      ${RESULT_BASE}"
echo "Max steps:    ${MAX_STEPS}"
echo "Frame offset: ${IMAGE_FRAME_OFFSET}"
echo "Bbox filter:  ${BBOX_FILTER}"
echo "============================================"

# Lockfile for thread-safe frame counter
LOCK_DIR="/tmp/gsplat_ftune_lock"
FRAME_FILE="/tmp/gsplat_ftune_next_frame"
LOG_DIR="${RESULT_BASE}/logs"
mkdir -p "${LOG_DIR}"

# Initialize frame counter
echo "${START_FRAME}" > "${FRAME_FILE}"

# Cleanup lock on exit
cleanup() {
    echo "Cleanup: skipping file removal for safety"
    # Kill all background jobs
    jobs -p | xargs -r kill 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT

# Function to atomically get and increment next frame
get_next_frame() {
    while ! mkdir "${LOCK_DIR}" 2>/dev/null; do
        sleep 0.1
    done
    local frame=$(cat "${FRAME_FILE}")
    if [ "${frame}" -le "${END_FRAME}" ]; then
        echo $((frame + 1)) > "${FRAME_FILE}"
        rmdir "${LOCK_DIR}"
        echo "${frame}"
    else
        rmdir "${LOCK_DIR}"
        echo "-1"
    fi
}

# Function to run frames on a single GPU
run_gpu_worker() {
    local GPU_ID=$1

    while true; do
        local FRAME=$(get_next_frame)
        if [ "${FRAME}" -eq "-1" ]; then
            echo "[GPU ${GPU_ID}] No more frames. Worker done."
            break
        fi

        local PLY_FILE=$(printf "%s/%04d.ply" "${PLY_DIR}" "${FRAME}")
        local IMAGE_FRAME=$((FRAME + IMAGE_FRAME_OFFSET))
        local RESULT_DIR="${RESULT_BASE}/frame_$(printf '%04d' ${FRAME})"
        local LOG_FILE="${LOG_DIR}/frame_$(printf '%04d' ${FRAME}).log"

        if [ ! -f "${PLY_FILE}" ]; then
            echo "[GPU ${GPU_ID}] SKIP frame ${FRAME}: PLY not found at ${PLY_FILE}"
            continue
        fi

        # Skip if already completed (result dir has a ckpt)
        if [ -f "${RESULT_DIR}/ckpts/ckpt_${MAX_STEPS}_rank0.pt" ]; then
            echo "[GPU ${GPU_ID}] SKIP frame ${FRAME}: already completed"
            continue
        fi

        echo "[GPU ${GPU_ID}] START frame ${FRAME} (image=${IMAGE_FRAME}) -> ${RESULT_DIR}"

        local BBOX_ARG=""
        if [ "${BBOX_FILTER}" = "false" ]; then
            BBOX_ARG="--no-bbox-filter"
        fi

        local OPA_ARG=""
        if [ -n "${OPA_CLAMP}" ]; then
            OPA_ARG="--init-opa-clamp ${OPA_CLAMP}"
        fi

        CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} examples/simple_trainer_ftune.py ftune \
            --data-dir "${DATA_DIR}" \
            --data-factor ${DATA_FACTOR} \
            --result-dir "${RESULT_DIR}" \
            --ply-path "${PLY_FILE}" \
            --frame-num ${IMAGE_FRAME} \
            --max-steps ${MAX_STEPS} \
            --test-every ${TEST_EVERY} \
            --disable-viewer \
            ${BBOX_ARG} \
            ${OPA_ARG} \
            > "${LOG_FILE}" 2>&1

        local EXIT_CODE=$?
        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "[GPU ${GPU_ID}] DONE  frame ${FRAME} (exit 0)"
        else
            echo "[GPU ${GPU_ID}] FAIL  frame ${FRAME} (exit ${EXIT_CODE}) - see ${LOG_FILE}"
        fi
    done
}

# Launch one worker per GPU
echo ""
echo "Launching ${NUM_GPUS} GPU workers..."
PIDS=()
for GPU_ID in "${GPU_ARRAY[@]}"; do
    run_gpu_worker "${GPU_ID}" &
    PIDS+=($!)
    echo "  Worker PID $! on GPU ${GPU_ID}"
done

echo ""
echo "All workers launched. Waiting for completion..."
echo "(Logs in ${LOG_DIR})"
echo ""

# Wait for all workers to finish
FAILED=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "WARNING: Worker for GPU ${GPU_ARRAY[$i]} exited with code ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "  All workers complete!"
if [ ${FAILED} -gt 0 ]; then
    echo "  WARNING: ${FAILED} worker(s) had errors"
fi
# Count completed frames
COMPLETED=$(find "${RESULT_BASE}" -name "ckpt_${MAX_STEPS}_rank0.pt" 2>/dev/null | wc -l)
echo "  Completed frames: ${COMPLETED}"
echo "============================================"

# Collect all finetuned PLYs into a single folder
PLY_OUT="${RESULT_BASE}/ply_finetuned"
mkdir -p "${PLY_OUT}"
COLLECTED=0
for FRAME in $(seq ${START_FRAME} ${END_FRAME}); do
    FRAME_PAD=$(printf '%04d' ${FRAME})
    # The trainer saves PLY as point_cloud_<step-1>.ply (0-indexed step)
    SRC="${RESULT_BASE}/frame_${FRAME_PAD}/ply/point_cloud_$((MAX_STEPS - 1)).ply"
    DST="${PLY_OUT}/${FRAME_PAD}.ply"
    if [ -f "${SRC}" ]; then
        cp "${SRC}" "${DST}"
        COLLECTED=$((COLLECTED + 1))
    fi
done
echo ""
echo "Collected ${COLLECTED} finetuned PLYs -> ${PLY_OUT}/"
