#!/usr/bin/env bash
# TrackerSplat on thenewface at data_factor=4 (higher-res than the df=15 run).
# Reuses the same PLY and CoTracker npz as the df=15 run — tracks are
# auto-rescaled by trackersplat_trainer via downsample_factor / data_factor.

set -euo pipefail

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/
PLY=/data/shared/elaheh/4D_demo/thenewface_multiframe_fast/frame_001/ply/point_cloud_combined_2999.ply
COTRACKER_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/cotracker_out/
OUT=/data/shared/elaheh/4D_demo/new_data/trackersplat_results/run_df4_50k

mkdir -p "${OUT}"

python examples/trackersplat_trainer.py \
  --data_dir "${DATA_DIR}" \
  --ply_path "${PLY}" \
  --cotracker_dir "${COTRACKER_DIR}" \
  --result_dir "${OUT}" \
  --data_factor 4 \
  --num_cotracker_frames 50 \
  --frame_step 6 \
  --sh_degree 3 \
  --max_steps 50000 \
  --freeze_appearance_steps 1000 \
  --eval_steps 1000 3000 5000 10000 25000 50000 \
  --save_steps 5000 25000 50000 \
  --ply_save_steps 7000 15000 25000 50000 \
  --tb_image_every 200 \
  --render_video_cams 0 22 44
