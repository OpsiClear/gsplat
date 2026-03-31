#!/usr/bin/env bash
# Evaluation and Viewer Script for thenewface
# Loads the latest checkpoint from the new results directory.

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

echo "Loading latest checkpoint: $CHECKPOINT_PATH"

# Run simple_trainer_static_dynamic in eval-only mode with viewer enabled
python simple_trainer_static_dynamic.py \
    --ckpt "$CHECKPOINT_PATH" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 300 \
    --data_factor 4 \
    --use_deformation \
    --disable_viewer false \
    "${@:2}"
