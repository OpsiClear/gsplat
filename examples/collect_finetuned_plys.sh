#!/bin/bash
# Collect fine-tuned PLY files from per-frame results into a single directory.
#
# For each frame, picks the PLY at a given iteration and copies it as XXXX.ply
# (4-digit zero-padded frame number) into the output directory.
#
# Usage:
#   bash examples/collect_finetuned_plys.sh
#   bash examples/collect_finetuned_plys.sh --step 12999 --start_frame 0 --end_frame 999
#   bash examples/collect_finetuned_plys.sh --step 4999   # collect 5k iteration PLYs

set -e

# ---- Defaults ----
RESULT_BASE="/data/shared/elaheh/newface_finetune"
OUTPUT_DIR=""  # auto-generated if empty
START_FRAME=0
END_FRAME=999
# PLY step to collect (note: ply_steps saves at step-1, so 13000 -> 12999)
STEP=4999

# ---- Parse CLI arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --result_base)  RESULT_BASE="$2";  shift 2;;
        --output_dir)   OUTPUT_DIR="$2";   shift 2;;
        --start_frame)  START_FRAME="$2";  shift 2;;
        --end_frame)    END_FRAME="$2";    shift 2;;
        --step)         STEP="$2";         shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# Auto-generate output dir if not specified
if [ -z "${OUTPUT_DIR}" ]; then
    OUTPUT_DIR="${RESULT_BASE}/ply_sequence_step${STEP}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  Collect Fine-tuned PLY Files"
echo "============================================"
echo "Result base:  ${RESULT_BASE}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Frames:       ${START_FRAME} -> ${END_FRAME}"
echo "Step:         ${STEP}"
echo "============================================"

COLLECTED=0
MISSING=0

for FRAME in $(seq ${START_FRAME} ${END_FRAME}); do
    FRAME_DIR=$(printf "%s/frame_%04d" "${RESULT_BASE}" "${FRAME}")
    SRC_PLY="${FRAME_DIR}/ply/point_cloud_${STEP}.ply"
    DST_PLY=$(printf "%s/%04d.ply" "${OUTPUT_DIR}" "${FRAME}")

    if [ -f "${SRC_PLY}" ]; then
        cp "${SRC_PLY}" "${DST_PLY}"
        COLLECTED=$((COLLECTED + 1))
    else
        echo "[MISSING] Frame ${FRAME}: ${SRC_PLY}"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "============================================"
echo "  Done!"
echo "  Collected: ${COLLECTED} PLY files"
echo "  Missing:   ${MISSING} frames"
echo "  Output:    ${OUTPUT_DIR}"
echo "============================================"
