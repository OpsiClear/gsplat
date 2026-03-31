#!/bin/bash
# Train static 3DGS on elly frame 0 (64 cameras, SFM init, no masks).
# COLMAP sparse/0/ already contains only frame 0 images: CAM/000000.jpg

set -e

DATA_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/undistorted"
RESULT_DIR="/data/shared/elaheh/elly_static_v2"

# Pick GPU 1 if free, else fail
GPU=3
FREE=$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
if [ "$FREE" -ne 0 ]; then
    echo "GPU $GPU is busy. Exiting."
    exit 1
fi

echo "=== elly static training (frame 0, SFM init) ==="
echo "  data_dir : $DATA_DIR"
echo "  result   : $RESULT_DIR"
echo "  GPU      : $GPU"
echo ""

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=$GPU /home/elaheh/miniforge3/envs/gsplat/bin/python simple_trainer.py default \
    --data_dir    "$DATA_DIR"   \
    --result_dir  "$RESULT_DIR" \
    --data_factor 2             \
    --test_every  0             \
    --port        6072          \
    --max_steps   30000         \
    --eval_steps  7000 30000    \
    --save_steps  7000 30000    \
    --ply_steps   7000 30000    \
    --sh_degree   0

echo ""
echo "=== Done! Results saved to $RESULT_DIR ==="
