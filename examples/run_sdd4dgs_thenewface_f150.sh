#!/usr/bin/env bash
# SDD-4DGS on thenewface — 150 frames, GPU 3
# Based on run_4dgs_thenewface_v13.sh config + SDD-4DGS additions
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_sdd4dgs_f150_v1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export CUDA_VISIBLE_DEVICES=3
PYTHON=/home/elaheh/miniforge3/envs/gsplat/bin/python
cd "$SCRIPT_DIR"

echo "=== SDD-4DGS: thenewface 150 frames, GPU 3 ==="
echo "Data:   $DATA_DIR"
echo "Result: $RESULT_DIR"

$PYTHON simple_trainer_sdd4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 150 \
    --frame_stride 2 \
    --frame_start 1 \
    --max_steps 50000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --data_factor 4 \
    --sh_degree 2 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 300 \
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
    --ssim_lambda 0.15 \
    --opacity_reg 0.005 \
    --scale_reg 0.05 \
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
    --strategy.refine-stop-iter 40000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 40000 50000 \
    --save_steps 20000 40000 50000 \
    --export_per_frame_ply \
    --sdd_mode \
    --w_lr 1e-3 \
    --binary_entropy_alpha 1e-4 \
    --binary_entropy_weight 0.01 \
    --lasg_weight 1.0 \
    --train_threshold 0.5 \
    "$@"

echo "=== Done. Results in $RESULT_DIR ==="
echo "TensorBoard: tensorboard --logdir $RESULT_DIR/tb"
