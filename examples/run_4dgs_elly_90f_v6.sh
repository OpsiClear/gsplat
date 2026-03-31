#!/usr/bin/env bash
# Run 4DGS trainer on elly dataset — 90 frames, v6: match reference 4DGS
#
# Key changes from v5:
#   - Removed tanh bounding (reference uses unbounded outputs + zero-init)
#   - Added LR warmup (lr_delay_mult=0.01, 1000-step cosine warmup)
#   - Reduced depth 6→1 (reference DyNeRF uses 0-1)
#   - Reduced multires [1,2,4,8]→[1,2] (reference DyNeRF config)
#   - Fixed temporal res scaling (now fixed, not sqrt-scaled)
#   - Enabled l1_time_planes=0.0001 (keeps temporal planes near identity)
#   - Disabled time_pe_bands (reference doesn't use them)
#   - Disabled opacity/SH deform (reference DyNeRF default: off)
#   - Reduced Gaussian cap to 360K (reference cap)
#   - Fixed gradient clipping order (clip BEFORE optimizer.step)
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f90_v6
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 90 \
    --frame_stride 5 \
    --frame_start 0 \
    --max_steps 30000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 300 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 \
    --deform_net_width 128 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 0 \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 360000 \
    --ssim_lambda 0.0 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --time_smooth_order 2 \
    --l1_time_planes_weight 0.0001 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 20000 30000 \
    --save_steps 10000 20000 30000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
