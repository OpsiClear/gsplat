#!/usr/bin/env bash
# Run 4DGS on elly — cropped images + cropped static PLY
set -e

export CUDA_VISIBLE_DEVICES=2

UNDISTORTED=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
STATIC_PLY=/data/shared/elaheh/elly_static_v2/cropped_static.ply
DYNAMIC_PLY=/data/shared/elaheh/elly_static_v2/dynamic.ply
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f150_cropped_static_dynamic
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build a lightweight data_dir with symlinks so the trainer finds
# images/ and sparse/0/ in the expected places
DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted_cropped
mkdir -p "$DATA_DIR"
ln -sfn "$UNDISTORTED/cropped_images"    "$DATA_DIR/images"
ln -sfn "$UNDISTORTED/recropped_sparse"  "$DATA_DIR/sparse"

cd "$SCRIPT_DIR"

# Progressive sampling: --progressive_time_forward starts from frame 0 and expands forward
# (window [0, 10%] → [0, 20%] → ... → [0, 100%]).
# Remove --progressive_time_forward to revert to old symmetric expansion from center
# (window ±10% → ±50% around frame 75).
python simple_trainer_static_dynamic.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --static_ply_path "$STATIC_PLY" \
    --dynamic_ply_path "$DYNAMIC_PLY" \
    --dataset_mode rig \
    --num_frames 150 \
    --frame_stride 1 \
    --frame_start 0 \
    --max_steps 50000 \
    --coarse_iters 0 \
    --init_type ply \
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
    --progressive_time_forward \
    --strategy.refine-stop-iter 30000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 40000 50000 \
    --save_steps 20000 40000 50000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
