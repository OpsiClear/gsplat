#!/bin/bash
# 4C4D: Train on thenewface dataset at low resolution (data_factor=15)
#
# Architecture: frozen static (63K) + trainable dynamic (17.6K) = 81K Gaussians
# Data: 45 cameras × 300 frames, all preloaded to GPU (~1.4 GB)
# Batching: 45 cameras × 5 frames/block = 225 renders/step
# One sweep: 60 blocks. 30k steps = 500 sweeps. ~5 GB VRAM.
#
# Usage:
#   bash run_4c4d_face_lowres.sh [GPU_ID]

set -e

GPU=${1:-0}
DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/results_4c4d

echo "============================================"
echo "4C4D Training - thenewface (low-res)"
echo "  GPU:         $GPU"
echo "  Data:        $DATA_DIR"
echo "  Results:     $RESULT_DIR"
echo "  Static PLY:  outside_05.ply (63K frozen)"
echo "  Dynamic PLY: inside_05.ply (17.6K trainable)"
echo "  Factor:      15"
echo "  Frames:      300 (5 per block × 60 blocks)"
echo "  Batch:       45 cameras × 5 frames = 225/step"
echo "============================================"

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=$GPU python simple_trainer_4c4d.py default \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --data_factor 15 \
    --num_frames 300 \
    --frames_per_step 5 \
    --frame_start 1 \
    --frame_step 1 \
    --max_steps 30000 \
    --eval_steps 7000 15000 30000 \
    --save_steps 7000 30000 \
    --batch_size 45 \
    --sh_degree 0 \
    --normalize_world_space \
    --static_ply "$DATA_DIR/static_dynamic_output/outside_05.ply" \
    --dynamic_ply "$DATA_DIR/static_dynamic_output/inside_05.ply" \
    --test_every 0 \
    --val_num_cameras 5 \
    --val_num_frames 5 \
    --decay_warmup 500 \
    --decay_mlp_lr 1e-3 \
    --invisible_decay_beta 0.999 \
    --temporal_lr 1e-3 \
    --ssim_lambda 0.2 \
    --tb_every 100 \
    --tb_image_every 200 \
    --tb_image_num_views 3 \
    --disable_viewer

echo "Training complete. Results at: $RESULT_DIR"
