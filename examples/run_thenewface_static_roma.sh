#!/bin/bash
# Train static 3DGS on thenewface (frame 000001) initialised from ROMA dense point cloud.
# Camera poses come from COLMAP; point positions + colors from ROMA numpy files.
# Inverted motion masks: black=keep (train), white=exclude.

set -e

DATA_DIR="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted"
ROMA_DIR="${DATA_DIR}/keyframe_roma"
MASK_DIR="${DATA_DIR}/motion_masks"
RESULT_DIR="/data/shared/elaheh/thenewface_static_roma_v1"

GPU=3

echo "=== thenewface static training (ROMA init) ==="
echo "  data_dir   : $DATA_DIR"
echo "  roma pts   : $ROMA_DIR/points3d_frame000001.npy"
echo "  roma colors: $ROMA_DIR/colors_frame000001.npy"
echo "  mask_dir   : $MASK_DIR  (inverted: black=keep)"
echo "  result     : $RESULT_DIR"
echo "  GPU        : $GPU"
echo ""

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=$GPU /home/elaheh/miniforge3/envs/gsplat/bin/python simple_trainer.py default \
    --data_dir        "$DATA_DIR"  \
    --result_dir      "$RESULT_DIR" \
    --mask_dir        "$MASK_DIR"  \
    --use_masks                    \
    --invert_masks                 \
    --init_type       npy          \
    --npy_pts_path    "${ROMA_DIR}/points3d_frame000001.npy" \
    --npy_colors_path "${ROMA_DIR}/colors_frame000001.npy"   \
    --port            6070         \
    --max_steps       30000        \
    --eval_steps      7000 15000 30000 \
    --save_steps      7000 15000 30000 \
    --ply_steps       7000 30000   \
    --test_every      0

echo ""
echo "=== Done! Results saved to $RESULT_DIR ==="
