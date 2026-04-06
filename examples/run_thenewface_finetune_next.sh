#!/usr/bin/env bash
# ~10s per-frame fine-tuning: load previous frame's dynamic PLY + frozen static cache
# Usage: ./run_thenewface_finetune_next.sh [GPU] [FRAME] [PREV_DYNAMIC_PLY]
# Example: ./run_thenewface_finetune_next.sh 0 2 /path/to/frame1/ply/point_cloud_4999.ply
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
STATIC_PLY="${DATA_DIR}/static_dynamic_output/outside_05.ply"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU=${1:-0}
FRAME=${2:-2}
PREV_PLY=${3:-/data/shared/elaheh/4D_demo/thenewface_ablation_speed/baseline/ply/point_cloud_4999.ply}
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_ftune_frame${FRAME}
CACHE_DIR="${RESULT_DIR}/image_cache"

cd "$SCRIPT_DIR"

echo "========================================================"
echo "  10s Fine-tune — thenewface frame ${FRAME}"
echo "  Static: 63k frozen | Dynamic: prev PLY fine-tuned"
echo "  Init PLY: ${PREV_PLY}"
echo "  GPU: ${GPU}"
echo "========================================================"

CUDA_VISIBLE_DEVICES=${GPU} python simple_trainer_ftune.py mcmc \
    --data_dir "$DATA_DIR" \
    --result_dir "${RESULT_DIR}" \
    --data_factor 15 \
    --static_ply_path "${STATIC_PLY}" \
    --init_type ply \
    --ply_path "${PREV_PLY}" \
    --batch_size -1 \
    --max_steps 1500 \
    --test_every 100000 \
    --sh_degree 3 \
    --ssim_lambda 0.2 \
    --antialiased \
    --strategy.cap-max 20000 \
    --strategy.refine-every 100 \
    --strategy.refine-stop-iter 1000 \
    --strategy.refine-start-iter 50 \
    --strategy.noise-lr 1e4 \
    --strategy.min-opacity 0.005 \
    --strategy.verbose \
    --opacity_reg 0.001 \
    --scale_reg 0.0001 \
    --save_ply \
    --eval_steps 1500 \
    --save_steps 1500 \
    --ply_steps 1500 \
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
