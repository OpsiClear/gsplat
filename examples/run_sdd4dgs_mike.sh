#!/usr/bin/env bash
# Run SDD-4DGS on mike_tech — 30 frames, 1 chunk, GPU 1
# SDD-4DGS: Static-Dynamic Decoupled 4D Gaussian Splatting
# Adds per-Gaussian dynamic perception coefficient w on top of 4DGS deformation.
# Based on run_4dgs_mike_chunked.sh config + SDD-specific args.
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/mike_tech/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/mike_sdd4dgs_f30_v1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRAME_START=0

export CUDA_VISIBLE_DEVICES=1
cd "$SCRIPT_DIR"

echo "=== SDD-4DGS: mike_tech 30 frames, GPU 1 ==="
echo "Data:   $DATA_DIR"
echo "Result: $RESULT_DIR"

python simple_trainer_sdd4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 30 \
    --frame_stride 1 \
    --frame_start $FRAME_START \
    --max_steps 30000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --data_factor 4 \
    --sh_degree 2 \
    --use_deformation \
    --deform_grid_resolution 128 \
    --deform_time_resolution 200 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 16 \
    --deform_net_width 128 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 4 \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --aabb_margin 1.3 \
    --max_num_gaussians 1000000 \
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
    --progressive_time_warmup 10000 \
    --progressive_time_initial 0.05 \
    --strategy.refine-stop-iter 24000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 \
    --save_steps 15000 30000 \
    --export_per_frame_ply \
    --sdd_mode \
    --w_lr 1e-3 \
    --binary_entropy_alpha 1e-4 \
    --binary_entropy_weight 0.01 \
    --lasg_weight 1.0 \
    --train_threshold 0.5 \
    "$@"

echo "=== Done. Results in $RESULT_DIR ==="
echo "Launch TensorBoard: tensorboard --logdir $RESULT_DIR/tb"
echo ""
echo "Verification checks:"
echo "  1. sigmoid(w).mean() at step 0 should be ~0.5 (printed at start)"
echo "  2. TensorBoard: train/lambda_bi ramps from 0 to 0.01 over training"
echo "  3. TensorBoard: train/w_sigmoid histogram at 30k should be bimodal"
echo "  4. PLY output: frame_*_wcolor.ply — moving=red, static=blue"
