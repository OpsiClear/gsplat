#!/usr/bin/env bash
# Run 4DGS trainer on elly dataset (90 frames, fixed 64-camera rig)
# Frames: 0, 10, 20, ..., 890 (every 10th frame, 90 total)

set -e

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f30
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 30 \
    --frame_stride 5 \
    --frame_start 0 \
    --max_steps 30000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 25 \
    --deform_feature_dim 16 \
    --deform_net_width 128 \
    --deform_net_depth 6 \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 2000 5000 10000 20000 30000 \
    --save_steps 10000 20000 30000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
