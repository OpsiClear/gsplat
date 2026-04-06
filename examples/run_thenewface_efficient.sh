#!/usr/bin/env bash
# Efficient training: thenewface frame 1, df=15, cap=5000
# Full-batch (all views per step), pre-cached resized images
# 25k steps total: Gaussians-only 0-20k, PPISP enabled 20k-25k
# ~253x133 px images, 5k Gaussians, ~42 training views
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_efficient
CACHE_DIR="${RESULT_DIR}/image_cache"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU=${1:-1}

cd "$SCRIPT_DIR"

echo "========================================================"
echo "  Efficient MinSplat — thenewface frame 1"
echo "  df=15 (~253x133 px) | cap=5000 | full-batch"
echo "  GPU: ${GPU}"
echo "========================================================"

CUDA_VISIBLE_DEVICES=${GPU} python simple_trainer_ftune.py mcmc \
    --data_dir "$DATA_DIR" \
    --result_dir "${RESULT_DIR}" \
    --data_factor 15 \
    --init_num_pts 2500 \
    --batch_size -1 \
    --max_steps 25000 \
    --test_every 100000 \
    --sh_degree 3 \
    --ssim_lambda 0.05 \
    --antialiased \
    --post_processing ppisp \
    --ppisp_start_step 20000 \
    --no-ppisp_use_controller \
    --no-ppisp_controller_distillation \
    --strategy.cap-max 5000 \
    --strategy.refine-every 300 \
    --strategy.refine-stop-iter 15000 \
    --strategy.refine-start-iter 500 \
    --strategy.noise-lr 1e4 \
    --strategy.min-opacity 0.001 \
    --strategy.verbose \
    --opacity_reg 0.001 \
    --scale_reg 0.0001 \
    --save_ply \
    --eval_steps 25000 \
    --save_steps 25000 \
    --ply_steps 25000 \
    --tb_every 100 \
    --disable_viewer \
    --disable_video \
    --load_images_in_memory \
    --cache_dir "${CACHE_DIR}" \
    --frame_num 1 \
    2>&1 | tee "${RESULT_DIR}/train.log"

echo "Done! Results in ${RESULT_DIR}/"
