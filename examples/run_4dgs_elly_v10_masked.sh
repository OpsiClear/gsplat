#!/usr/bin/env bash
# Run 4DGS on elly — 300 frames, v10 config + masks
# v10: higher temporal capacity for complex motion (walking, sitting, hand gestures)
#   - time_resolution 900 (3 cells/frame), feature_dim 64
#   - MLP depth 2, time_pe_bands 8, SH deform on
#   - relaxed temporal smoothing (order 1, annealed)
#   - 3M Gaussian cap, 100k steps
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
MASK_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/masks
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f150_v10_masked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --mask_dir "$MASK_DIR" \
    --dataset_mode rig \
    --num_frames 150 \
    --frame_stride 1 \
    --frame_start 1 \
    --max_steps 100000 \
    --coarse_iters 7000 \
    --init_type sfm \
    --data_factor 4 \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 600 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 \
    --deform_net_width 128 \
    --deform_net_depth 2 \
    --deform_time_pe_bands 8 \
    --enable_opacity_deform \
    --enable_sh_deform \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 1500000 \
    --ssim_lambda 0.2 \
    --scale_reg 0.01 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.001 \
    --time_smooth_weight_final 0.0001 \
    --time_smooth_order 1 \
    --l1_time_planes_weight 0.0001 \
    --weight_constraint_init 1.0 \
    --weight_constraint_after 0.2 \
    --weight_constraint_decay_iters 5000 \
    --strategy.refine-stop-iter 30000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 30000 40000 50000 60000 70000 80000 90000 100000 \
    --save_steps 20000 40000 60000 80000 100000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
