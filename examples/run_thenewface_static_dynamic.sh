#!/usr/bin/env bash
# Run 4DGS on thenewface — 300 frames, static+dynamic PLY init, no masks
# v2: same improved hyperparameters as run_elly_static_dynamic_v2.sh
#
# Key changes from v1:
#   opacity_reg  0.001  → 0.0001  (face Gaussians must be opaque; 0.001 caused 160K→80K pruning)
#   scale_reg    0.01   → 0.005   (less aggressive scale shrinkage)
#   time_smooth_weight_final -1.0 → 0.001  (anneal from 0.01→0.001 for sharper late-training motion)
#   weight_constraint_after  0.2  → 0.0   (let constraint fully decay; residual 0.2 was fighting position learning)
#   weight_constraint_decay_iters 5000 → 15000  (slower decay, constraint stays helpful longer)
#   deform_net_depth 0 → 1  (now actually adds 1 extra backbone layer; 0 and 1 were identical before fix)
#   strategy.reset-every 3000  *** RESTORED (step-0 fire fixed in gsplat/strategy/default.py) ***
#     Root fix: added `step > 0` guard in default.py so reset never fires at step 0.
#     reset_every=3000 is now safe and useful: cleans up floaters every 3K steps.
#     PLY-initialized face Gaussians keep their high opacity (reset skipped at step 0).
#
# Dataset-specific differences from elly:
#   num_frames 300 (vs 150), frame_start 1 (vs 0)
#   deform_time_resolution 600 (scaled 2x for 2x frames)
#   sh_degree 3 (vs 2)
#   CUDA_VISIBLE_DEVICES=3
set -e

export CUDA_VISIBLE_DEVICES=4

STATIC_PLY=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/outside_05.ply
DYNAMIC_PLY=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/inside_05.ply
DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
ROI_BOUNDS_PATH=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/roi_bounds.npy
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_4dgs_f300_static_dynamic_05ply_df20_roi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

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
    --data_factor 20 \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 600 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 \
    --deform_net_width 128 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 4 \
    --deform_act_xyz relu \
    --deform_act_rot relu \
    --deform_act_scale relu \
    --deform_act_sh sinerelu \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 1500000 \
    --ssim_lambda 0.15 \
    --opacity_reg 0.0001 \
    --scale_reg 0.005 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --time_smooth_weight_final 0.001 \
    --time_smooth_order 2 \
    --l1_time_planes_weight 0.0001 \
    --weight_constraint_init 1.0 \
    --weight_constraint_after 0.0 \
    --weight_constraint_decay_iters 15000 \
    --progressive_time_warmup 10000 \
    --progressive_time_initial 0.1 \
    --progressive_time_forward  \
    --promotion_every 5000 \
    --promotion_start 25500 \
    --promotion_num_time_samples 100 \
    --promotion_xyz_threshold 0.0001 \
    --promotion_grad_threshold 0.0001 \
    --promotion_percentile 2.0 \
    --roi_bounds_path "$ROI_BOUNDS_PATH" \
    --roi_padding 0.05 \
    --strategy.refine-stop-iter 35000 \
    --strategy.reset-every 3000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 40000 50000 \
    --save_steps 30000 40000 50000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
