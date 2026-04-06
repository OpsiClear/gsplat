#!/usr/bin/env bash
# Elly chunked training — Chunk 2 (frames 50-119, overlap with chunk 1 at 50-69)
# Same v2 hyperparameters as the full run.
set -e

export CUDA_VISIBLE_DEVICES=4

CHUNK1_DIR=/data/shared/elaheh/4D_demo/elly_chunked/chunk1
CHUNK1_CKPT=${CHUNK1_DIR}/ckpts/ckpt_29999_rank0.pt
CHUNK1_DYNAMIC=${CHUNK1_DIR}/ply/next_dynamic.ply
CHUNK1_STATIC=${CHUNK1_DIR}/ply/next_static.ply

echo "Waiting for chunk 1 to finish (looking for ${CHUNK1_DYNAMIC})..."
while [ ! -f "$CHUNK1_DYNAMIC" ]; do
    sleep 30
done
echo "Chunk 1 exports found! Starting chunk 2..."

DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_chunked/chunk2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$RESULT_DIR"
cd "$SCRIPT_DIR"

/home/elaheh/miniforge3/envs/gsplat/bin/python simple_trainer_static_dynamic.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --static_ply_path "$CHUNK1_STATIC" \
    --dynamic_ply_path "$CHUNK1_DYNAMIC" \
    --resume_aabb_from "$CHUNK1_CKPT" \
    --canonical_frame_rank 9 \
    --dataset_mode rig \
    --num_frames 70 \
    --frame_stride 1 \
    --frame_start 50 \
    --max_steps 30000 \
    --coarse_iters 0 \
    --init_type ply \
    --data_factor 8 \
    --sh_degree 3 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 140 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 \
    --deform_net_width 128 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 4 \
    --deform_act_xyz relu \
    --deform_act_rot relu \
    --deform_act_scale relu \
    --deform_act_sh sinerelu \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 1500000 \
    --ssim_lambda 0.15 \
    --opacity_reg 0.0001 \
    --scale_reg 0.005 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --time_smooth_weight_final 0.001 \
    --time_smooth_order 2 \
    --l1_time_planes_weight 0.0001 \
    --weight_constraint_init 1.0 \
    --weight_constraint_after 0.0 \
    --weight_constraint_decay_iters 15000 \
    --progressive_time_warmup 10000 \
    --progressive_time_initial 0.1 \
    --progressive_time_forward \
    --chunk_overlap 10 \
    --export_at_frame_rank 59 \
    --strategy.refine-stop-iter 22000 \
    --strategy.reset-every 3000 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 10000 20000 30000 \
    --save_steps 30000 \
    --export_last_frame_ply \
    "$@"

echo "Chunk 2 complete. Results in $RESULT_DIR"
