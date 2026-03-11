#!/usr/bin/env bash
# Run 4DGS on thenewface — 300 frames, v12 config + masks
# v12: centered time [-0.5, 0.5] + progressive frame sampling
#   - no opacity deform, opacity_reg 0.001
#   - ssim_lambda 0.08
#   - progressive warmup 10k iters from ±10%
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
MASK_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/masks
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_4dgs_f300_v12_masked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --mask_dir "$MASK_DIR" \
    --dataset_mode rig \
    --num_frames 300 \
    --frame_stride 1 \
    --frame_start 1 \
    --max_steps 80000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --data_factor 2 \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 600 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 \
    --deform_net_width 128 \
    --deform_net_depth 0 \
    --deform_time_pe_bands 4 \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 2000000 \
    --ssim_lambda 0.08 \
    --opacity_reg 0.001 \
    --scale_reg 0.01 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --time_smooth_weight_final -1.0 \
    --time_smooth_order 2 \
    --l1_time_planes_weight 0.0001 \
    --weight_constraint_init 1.0 \
    --weight_constraint_after 0.2 \
    --weight_constraint_decay_iters 5000 \
    --progressive_time_warmup 10000 \
    --progressive_time_initial 0.1 \
    --strategy.refine-stop-iter 30000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 \
    --save_steps 20000 40000 60000 80000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
