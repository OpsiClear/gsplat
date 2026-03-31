#!/usr/bin/env bash
# Training script for thenewface dataset (300 frames)
# Using tight AABB and separate Static/Dynamic PLY initializations.

set -e

# Specify your preferred GPU (defaults to 4)
GPU_ID=${1:-4}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Corrected paths for thenewface
DATA_DIR="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted"
STATIC_PLY="/data/shared/elaheh/thenewface_static_v2/cropped_static.ply"
DYNAMIC_PLY="/data/shared/elaheh/thenewface_static_v2/dynamic.ply"
RESULT_DIR="/data/shared/elaheh/4D_demo/thenewface_4dgs_v3_fixed"

# Activate environment if needed (already set in shell)
# conda activate gsplat

python simple_trainer_static_dynamic.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --static_ply_path "$STATIC_PLY" \
    --dynamic_ply_path "$DYNAMIC_PLY" \
    --dataset_mode rig \
    --num_frames 300 \
    --frame_stride 1 \
    --frame_start 1 \
    --max_steps 50000 \
    --coarse_iters 0 \
    --init_type ply \
    --data_factor 4 \
    --sh_degree 2 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 300 \
    --deform_feature_dim 32 \
    --deform_net_width 128 \
    --deform_act_sh sinerelu \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --max_num_gaussians 1500000 \
    --ssim_lambda 0.15 \
    --opacity_reg 0.001 \
    --scale_reg 0.01 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --progressive_time_warmup 10000 \
    --progressive_time_initial 0.2 \
    --progressive_time_forward true \
    --use_masks false \
    --tb_every 100 \
    --eval_steps 10000 20000 30000 40000 50000 \
    --save_steps 20000 40000 50000 \
    "${@:2}"
