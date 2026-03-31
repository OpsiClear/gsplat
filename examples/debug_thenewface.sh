#!/usr/bin/env bash
# Debugging script: Runs the diagnostic Python script (debug_4dgs.py)
# prints deformation stats, AABB coverage, and identity constraint checks.

set -e

# Specify your preferred GPU (defaults to 4)
GPU_ID=${1:-4}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Result directory from training
RESULT_DIR="/data/shared/elaheh/4D_demo/thenewface_4dgs_v3_fixed"
CHECKPOINT_PATH=$(ls -t "$RESULT_DIR"/ckpts/*.pt | head -n 1)

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "No checkpoint found in $RESULT_DIR/ckpts/"
    exit 1
fi

echo "Running 4DGS Diagnostic on: $CHECKPOINT_PATH"

# Run our diagnostic script to check deformation/AABB/scales
python debug_4dgs.py \
    --ckpt "$CHECKPOINT_PATH" \
    --device "cuda:$GPU_ID" \
    --out "$RESULT_DIR/debug_stats"
