#!/usr/bin/env bash
# Run 4DGS on mike_tech — 150 frames, v15 config
#
# Changes from v14:
#   time_smooth_weight: 0.01   -> 0.0007  (less temporal smoothing = more motion allowed)
#   weight_constraint_after: 0.2 -> 0.5   (allow larger deformation magnitudes)
#   deform_net_depth: 0        -> 1       (add one hidden layer for hand motion capacity)
#
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/mike_tech/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/mike_tech_4dgs_f150_v15
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 150 \
    --frame_stride 1 \
    --frame_start 1 \
    --max_steps 80000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --data_factor 4 \
    --sh_degree 2 \
    --use_deformation \
    --deform_grid_resolution 128 \
    --deform_time_resolution 800 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 16 \
    --deform_net_width 128 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 4 \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 1500000 \
    --ssim_lambda 0.2 \
    --opacity_reg 0.001 \
    --scale_reg 0.01 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.0007 \
    --time_smooth_weight_final -1.0 \
    --time_smooth_order 2 \
    --l1_time_planes_weight 0.0001 \
    --weight_constraint_init 1.0 \
    --weight_constraint_after 0.5 \
    --weight_constraint_decay_iters 5000 \
    --progressive_time_warmup 25000 \
    --progressive_time_initial 0.05 \
    --strategy.refine-stop-iter 64000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 40000 50000 60000 70000 80000 \
    --save_steps 20000 40000 60000 80000 \

    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
