#!/usr/bin/env bash
# Run 4DGS trainer on elly dataset — 90 frames, v7: capacity from grid, not depth
#
# v6: depth=1, multires=[1,2], width=128 → stable PSNR 23.3 but blurry
# v7 strategy: keep depth=1 (stable), add capacity via:
#   - wider MLP (256 vs 128)
#   - more multires [1,2,4] (grid capacity, not MLP depth)
#   - higher grid resolution (128 vs 64)
#   - time PE 4 bands
#   - opacity deform
#   - 1st order smoothness (allow fast motion)
#   - SSIM for sharpness
#   - 1M Gaussians, 60K steps
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f90_v7
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_4dgs.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --dataset_mode rig \
    --num_frames 90 \
    --frame_stride 5 \
    --frame_start 0 \
    --max_steps 60000 \
    --coarse_iters 3000 \
    --init_type sfm \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 128 \
    --deform_time_resolution 300 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 \
    --deform_net_width 256 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 4 \
    --enable_opacity_deform \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 1000000 \
    --ssim_lambda 0.2 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.005 \
    --time_smooth_weight_final 0.0005 \
    --time_smooth_order 1 \
    --l1_time_planes_weight 0.0001 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 \
    --save_steps 20000 40000 60000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
