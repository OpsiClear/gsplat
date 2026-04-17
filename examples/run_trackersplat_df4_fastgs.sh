#!/usr/bin/env bash
# TrackerSplat with frozen static + trainable dynamic, 60k steps, df=4.
# FasterGS CUDA backend variant — mirrors run_trackersplat_static_dyn_60k.sh
# but uses trackersplat_trainer_fastgs.py and the gsplat_fastergs conda env.
# Static PLY is loaded as detached tensors (not nn.Parameters); gradients
# flow only to the 10k dynamic Gaussians + motion_offsets.

set -eo pipefail

# --- env ---------------------------------------------------------------------
source /home/elaheh/miniforge3/etc/profile.d/conda.sh
conda activate gsplat_fastergs
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${CUDA_HOME}/bin:${PATH}"

# --- paths -------------------------------------------------------------------
DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/
# inside.ply = dynamic (ROI-interior splats), outside.ply = static (background)
DYN_PLY=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/inside.ply
STATIC_PLY=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/outside.ply
COTRACKER_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/alltrackerxx_out/
OUT=/data/shared/elaheh/4D_demo/new_data/trackersplat_results/run_static_dyn_60k_df4_fastgs_inside_outside_alltrackerxx_unsolv

mkdir -p "${OUT}"

cd /home/elaheh/projects/gsplat

echo "[run_trackersplat_df4_fastgs] env=$(conda info --envs | awk '/\*/{print $1}')"
echo "[run_trackersplat_df4_fastgs] dyn=${DYN_PLY}"
echo "[run_trackersplat_df4_fastgs] static=${STATIC_PLY}"
echo "[run_trackersplat_df4_fastgs] output=${OUT}"
echo "[run_trackersplat_df4_fastgs] start=$(date -Is)"

python examples/trackersplat_trainer_fastgs.py \
  --data_dir "${DATA_DIR}" \
  --ply_path "${DYN_PLY}" \
  --static_ply_path "${STATIC_PLY}" \
  --cotracker_dir "${COTRACKER_DIR}" \
  --result_dir "${OUT}" \
  --data_factor 4 \
  --sh_degree 0 \
  --no-normalize_world_space \
  --skip_points3d \
  --num_cotracker_frames 50 \
  --frame_step 6 \
  --max_steps 60000 \
  --freeze_appearance_steps 1000 \
  --eval_steps 10000 20000 30000 40000 50000 60000 \
  --save_steps 30000 60000 \
  --ply_save_steps 10000 30000 60000 \
  --tb_image_every 200 \
  --render_video_cams 0 22 44 \
  --use_unsolvable_reg \
  --unsolvable_start 2000 \
  --unsolvable_interval 1000 \
  --unsolvable_end 50000 \
  --unsolvable_score_views 8 \
  --unsolvable_k_nn 8

echo "[run_trackersplat_df4_fastgs] done=$(date -Is)"
