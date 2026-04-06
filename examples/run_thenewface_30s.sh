#!/usr/bin/env bash
# ~35s training: frozen static background + SFM-seeded dynamic Gaussians
# Static: 63k frozen (pre-rendered once, cached per view)
# Dynamic: 5k SFM seeds → MCMC densifies to 20k cap
# Full-batch (all 44 views per step), cached images, SSIM+L1
# Result: PSNR ~26, combined PLY ~83k Gaussians
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
STATIC_PLY="${DATA_DIR}/static_dynamic_output/outside_05.ply"
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_30s
CACHE_DIR="${RESULT_DIR}/image_cache"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU=${1:-0}
FRAME=${2:-1}

cd "$SCRIPT_DIR"

echo "========================================================"
echo "  35s MinSplat — thenewface frame ${FRAME}"
echo "  Static: 63k frozen | Dynamic: 5k SFM→20k MCMC | full-batch"
echo "  GPU: ${GPU}"
echo "========================================================"

CUDA_VISIBLE_DEVICES=${GPU} python simple_trainer_ftune.py mcmc \
    --data_dir "$DATA_DIR" \
    --result_dir "${RESULT_DIR}" \
    --data_factor 15 \
    --static_ply_path "${STATIC_PLY}" \
    --init_type sfm \
    --init_num_pts 5000 \
    --batch_size -1 \
    --max_steps 5000 \
    --test_every 100000 \
    --sh_degree 3 \
    --ssim_lambda 0.2 \
    --antialiased \
    --strategy.cap-max 20000 \
    --strategy.refine-every 100 \
    --strategy.refine-stop-iter 4000 \
    --strategy.refine-start-iter 100 \
    --strategy.noise-lr 1e4 \
    --strategy.min-opacity 0.005 \
    --strategy.verbose \
    --opacity_reg 0.001 \
    --scale_reg 0.0001 \
    --save_ply \
    --eval_steps 5000 \
    --save_steps 5000 \
    --ply_steps 5000 \
    --tb_every 0 \
    --disable_viewer \
    --disable_video \
    --load_images_in_memory \
    --cache_dir "${CACHE_DIR}" \
    --frame_num "${FRAME}" \
    --port 0 \
    --no-bbox_filter \
    2>&1 | tee "${RESULT_DIR}/train.log"

echo "Done! Results in ${RESULT_DIR}/"
