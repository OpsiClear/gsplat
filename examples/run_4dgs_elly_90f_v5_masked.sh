#!/usr/bin/env bash
# Run 4DGS trainer on elly dataset — 90 frames, v5 + masks: tanh-bounded deformation + grad clipping
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f90_v5_masked
MASK_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/masks
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --mask_dir "$MASK_DIR" \
    --dataset_mode rig \
    --num_frames 90 \
    --frame_stride 5 \
    --frame_start 0 \
    --max_steps 50000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 300 \
    --deform_feature_dim 32 \
    --deform_net_width 128 \
    --deform_net_depth 6 \
    --deform_time_pe_bands 8 \
    --enable_opacity_deform \
    --enable_sh_deform \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --ssim_lambda 0.2 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.001 \
    --time_smooth_weight_final 0.0001 \
    --time_smooth_order 1 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 20000 30000 50000 \
    --save_steps 20000 30000 50000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
