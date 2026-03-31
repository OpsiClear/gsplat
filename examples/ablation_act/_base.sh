#!/usr/bin/env bash
# ============================================================
# Base config for activation ablation experiments.
# Source this file and set EXP_NAME + activation vars before calling run_experiment.
#
# All other hyperparams are identical to run_thenewface_static_dynamic.sh
# so only the activation choice is the variable.
# ============================================================

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

STATIC_PLY=/data/shared/elaheh/thenewface_static_v2/static.ply
DYNAMIC_PLY=/data/shared/elaheh/thenewface_static_v2/dynamic.ply
DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
ABLATION_BASE=/data/shared/elaheh/4D_demo/ablation_act

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

run_experiment() {
    local name="$1"
    local act_xyz="$2"
    local act_rot="$3"
    local act_scale="$4"
    local act_sh="$5"

    RESULT_DIR="${ABLATION_BASE}/${name}"
    echo ""
    echo "================================================================"
    echo "  EXP: $name"
    echo "  act_xyz=$act_xyz  act_rot=$act_rot  act_scale=$act_scale  act_sh=$act_sh"
    echo "  result_dir: $RESULT_DIR"
    echo "================================================================"

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
        --max_num_gaussians 1500000 \
        --ssim_lambda 0.15 \
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
        --tb_image_every 500 \
        --eval_steps 10000 25000 50000 \
        --save_steps 50000 \
        --deform_act_xyz "$act_xyz" \
        --deform_act_rot "$act_rot" \
        --deform_act_scale "$act_scale" \
        --deform_act_sh "$act_sh"

    echo "Done: $name → $RESULT_DIR"
    echo "TensorBoard: tensorboard --logdir ${ABLATION_BASE}"
}
