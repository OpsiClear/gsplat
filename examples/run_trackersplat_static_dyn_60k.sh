#!/usr/bin/env bash
# TrackerSplat with frozen static + trainable dynamic, 60k steps, df=4.
# Static PLY is loaded as detached tensors (not nn.Parameters), so gradients
# flow only to the dynamic Gaussians + motion_offsets. Photo loss is full-frame
# composite vs full GT; cotracker tracks bound to background are excluded.

set -euo pipefail

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/
DYN_PLY=/data/shared/elaheh/thenewface_static_v2/dynamic.ply
STATIC_PLY=/data/shared/elaheh/thenewface_static_v2/static.ply
COTRACKER_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/cotracker_out/
OUT=/data/shared/elaheh/4D_demo/new_data/trackersplat_results/run_static_dyn_60k_df4

mkdir -p "${OUT}"

python examples/trackersplat_trainer.py \
  --data_dir "${DATA_DIR}" \
  --ply_path "${DYN_PLY}" \
  --static_ply_path "${STATIC_PLY}" \
  --cotracker_dir "${COTRACKER_DIR}" \
  --result_dir "${OUT}" \
  --data_factor 4 \
  --sh_degree 0 \
  --no-normalize_world_space \
  --num_cotracker_frames 50 \
  --frame_step 6 \
  --max_steps 60000 \
  --freeze_appearance_steps 1000 \
  --eval_steps 10000 20000 30000 40000 50000 60000 \
  --save_steps 30000 60000 \
  --ply_save_steps 10000 30000 60000 \
  --tb_image_every 200 \
  --render_video_cams 0 22 44
