#!/usr/bin/env bash
# Sequential per-frame 3DGS on elly — first 30 frames
# Uses optical flow peaks to allocate more training to high-motion frames
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
MASK_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/masks
FLOW_PATH=/data/shared/elaheh/4D_demo/outdoor/elly/flow_results/077-002/flow_magnitudes.npy
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_sequential_30f
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_sequential.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --mask_dir "$MASK_DIR" \
    --flow_path "$FLOW_PATH" \
    --data_factor 1 \
    --start_frame 0 \
    --num_frames 30 \
    --num_peaks 50 \
    --min_peak_separation 10 \
    --first_frame_steps 20000 \
    --normal_frame_steps 5000 \
    --peak_frame_steps 10000 \
    --sh_degree 3 \
    --ssim_lambda 0.2 \
    --scale_reg 0.01 \
    --test_every 8 \
    --eval_every_n_frames 5 \
    --tb_every 100 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
