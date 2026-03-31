#!/usr/bin/env bash
# Chunked 4DGS training for mike_tech — 5 chunks of 30 frames each
#
# Based on v15 config with adjustments for 30-frame chunks:
#   num_frames:  150 -> 30
#   max_steps:   80000 -> 30000
#   deform_time_resolution: 800 -> 200
#   progressive_time_warmup: 25000 -> 10000
#   strategy.refine-stop-iter: 64000 -> 24000
#   max_num_gaussians: 1500000 -> 1000000
#
# Each chunk exports per-frame PLYs with deformation baked in.
#
# Usage:
#   bash run_4dgs_mike_chunked.sh                    # fresh training
#   bash run_4dgs_mike_chunked.sh --resume_deform    # each chunk warm-starts
#       deformation network from previous chunk's final checkpoint
#
set -e

# Parse our own flags (pass the rest through to the trainer)
RESUME_DEFORM=false
PASSTHROUGH_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--resume_deform" ]]; then
        RESUME_DEFORM=true
    else
        PASSTHROUGH_ARGS+=("$arg")
    fi
done

DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/mike_tech/undistorted
RESULT_BASE=/data/shared/elaheh/4D_demo/mike_tech_4dgs_chunked_aabb3x
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

CHUNKS=1
FRAMES_PER_CHUNK=30

for CHUNK in $(seq 0 $((CHUNKS - 1))); do
    FRAME_START=$((CHUNK * FRAMES_PER_CHUNK + 1))
    RESULT_DIR="${RESULT_BASE}/mike_tech_4dgs_chunk${CHUNK}_f30"

    # Build optional resume flag from previous chunk's checkpoint
    DEFORM_CKPT_ARG=()
    if [[ "$RESUME_DEFORM" == "true" && $CHUNK -gt 0 ]]; then
        PREV_CHUNK=$((CHUNK - 1))
        PREV_CKPT="${RESULT_BASE}/mike_tech_4dgs_chunk${PREV_CHUNK}_f30/ckpts/ckpt_29999_rank0.pt"
        if [[ -f "$PREV_CKPT" ]]; then
            DEFORM_CKPT_ARG=(--resume_deform_ckpt "$PREV_CKPT")
            echo "Resuming deformation from: ${PREV_CKPT}"
        else
            echo "WARNING: previous checkpoint not found: ${PREV_CKPT}, starting fresh"
        fi
    fi

    echo "========================================"
    echo "Chunk ${CHUNK}: frames ${FRAME_START}–$((FRAME_START + FRAMES_PER_CHUNK - 1))"
    echo "Result dir: ${RESULT_DIR}"
    echo "========================================"

    python simple_trainer_4dgs.py \
        --data_dir "$DATA_DIR" \
        --result_dir "$RESULT_DIR" \
        --dataset_mode rig \
        --num_frames $FRAMES_PER_CHUNK \
        --frame_stride 1 \
        --frame_start $FRAME_START \
        --max_steps 30000 \
        --coarse_iters 3000 \
        --init_type sfm \
        --data_factor 4 \
        --sh_degree 2 \
        --use_deformation \
        --deform_grid_resolution 128 \
        --deform_time_resolution 200 \
        --deform_feature_dim 32 \
        --deform_multires 1 2 4 8 16 \
        --deform_net_width 128 \
        --deform_net_depth 1 \
        --deform_time_pe_bands 4 \
        --deform_lr 1.6e-4 \
        --grid_lr 1.6e-3 \
        --deform_lr_delay_mult 0.01 \
        --deform_lr_warmup_steps 1000 \
        --aabb_margin 0.3 \
        --max_num_gaussians 1000000 \
        --ssim_lambda 0.2 \
        --opacity_reg 0.001 \
        --scale_reg 0.01 \
        --plane_tv_weight 0.0001 \
        --time_smooth_weight 0.001 \
        --time_smooth_weight_final -1.0 \
        --time_smooth_order 2 \
        --l1_time_planes_weight 0.001 \
        --weight_constraint_init 0.1 \
        --weight_constraint_after 0.05 \
        --weight_constraint_decay_iters 5000 \
        --progressive_time_warmup 10000 \
        --progressive_time_initial 0.05 \
        --strategy.refine-stop-iter 3000 \
        --tb_every 100 \
        --tb_image_every 200 \
        --eval_steps 5000 10000 15000 20000 25000 30000 \
        --save_steps 15000 30000 \
        --export_per_frame_ply \
        "${DEFORM_CKPT_ARG[@]}" \
        "${PASSTHROUGH_ARGS[@]}"

    echo "Chunk ${CHUNK} done."
    echo ""
done

echo "All ${CHUNKS} chunks complete. Results in ${RESULT_BASE}/"
