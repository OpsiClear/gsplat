#!/bin/bash
# Per-frame fine-tuning of pre-trained Gaussian PLY files with PPISP post-processing.
#
# Usage:
#   bash examples/run_finetune.sh --gpu 0 --start_frame 0 --end_frame 5
#
# The script loops over frames [start_frame, end_frame] (inclusive), loads the
# corresponding PLY and filters COLMAP images to that frame, then fine-tunes
# for 12k iterations (10k training + 2k PPISP controller distillation).

set -e

# ---- Defaults (override via CLI flags) ----
GPU_ID=0
START_FRAME=0
END_FRAME=5
PLY_DIR="/data/shared/elaheh/final_4d_results/merge_ply_all_scenes/thenewface/ply_sequence_merged_40000_merged"
DATA_DIR="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted"
RESULT_BASE="/data/shared/elaheh/newface_finetune"
MAX_STEPS=10000
# Image frame offset: PLY frame 0 -> image filename 000001 (1-indexed)
# Set to 0 if PLY and image frames match directly.
IMAGE_FRAME_OFFSET=1
DATA_FACTOR=1
TEST_EVERY=-1   # -1 means use all images for training

# ---- Parse CLI arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)          GPU_ID="$2";          shift 2;;
        --start_frame)  START_FRAME="$2";     shift 2;;
        --end_frame)    END_FRAME="$2";       shift 2;;
        --ply_dir)      PLY_DIR="$2";         shift 2;;
        --data_dir)     DATA_DIR="$2";        shift 2;;
        --result_base)  RESULT_BASE="$2";     shift 2;;
        --max_steps)    MAX_STEPS="$2";       shift 2;;
        --offset)       IMAGE_FRAME_OFFSET="$2"; shift 2;;
        --data_factor)  DATA_FACTOR="$2";     shift 2;;
        --test_every)   TEST_EVERY="$2";      shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

echo "============================================"
echo "  Per-frame Gaussian Fine-tuning with PPISP"
echo "============================================"
echo "GPU:          ${GPU_ID}"
echo "Frames:       ${START_FRAME} -> ${END_FRAME}"
echo "PLY dir:      ${PLY_DIR}"
echo "Data dir:     ${DATA_DIR}"
echo "Results:      ${RESULT_BASE}"
echo "Max steps:    ${MAX_STEPS}"
echo "Frame offset: ${IMAGE_FRAME_OFFSET}"
echo "============================================"

for FRAME in $(seq ${START_FRAME} ${END_FRAME}); do
    # PLY file: 4-digit zero-padded (0000.ply, 0001.ply, ...)
    PLY_FILE=$(printf "%s/%04d.ply" "${PLY_DIR}" "${FRAME}")

    # Image frame number (COLMAP images may be 1-indexed)
    IMAGE_FRAME=$((FRAME + IMAGE_FRAME_OFFSET))

    RESULT_DIR="${RESULT_BASE}/frame_$(printf '%04d' ${FRAME})"

    if [ ! -f "${PLY_FILE}" ]; then
        echo "[SKIP] Frame ${FRAME}: PLY not found at ${PLY_FILE}"
        continue
    fi

    echo ""
    echo "--------------------------------------------"
    echo "  Frame ${FRAME}  |  PLY: ${PLY_FILE}"
    echo "  Image frame: ${IMAGE_FRAME}  |  Result: ${RESULT_DIR}"
    echo "--------------------------------------------"

    CUDA_VISIBLE_DEVICES=${GPU_ID} /home/elaheh/miniforge3/envs/gsplat/bin/python examples/simple_trainer_ftune.py ftune \
        --data-dir "${DATA_DIR}" \
        --data-factor ${DATA_FACTOR} \
        --result-dir "${RESULT_DIR}" \
        --ply-path "${PLY_FILE}" \
        --frame-num ${IMAGE_FRAME} \
        --max-steps ${MAX_STEPS} \
        --test-every ${TEST_EVERY} \
        --disable-viewer

    echo "[DONE] Frame ${FRAME} -> ${RESULT_DIR}"
done

echo ""
echo "============================================"
echo "  All frames complete!"
echo "============================================"
