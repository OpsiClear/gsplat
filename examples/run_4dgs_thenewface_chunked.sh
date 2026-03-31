#!/usr/bin/env bash
# Chunked 4DGS training for thenewface — 10 chunks of 30 frames each, NO masks
#
# Based on v12 config adapted for 30-frame chunks:
#   num_frames:  300 -> 30
#   max_steps:   50000 -> 30000
#   deform_time_resolution: 600 -> 200
#   strategy.refine-stop-iter: 30000 -> 24000
#   max_num_gaussians: 1500000 -> 1000000
#
# Chunk 0: SFM init, full coarse phase + warmups
# Chunk 1+: Load ALL gaussian properties from previous chunk's last-frame PLY,
#           resume deformation from previous checkpoint,
#           skip coarse phase, no LR warmup.
#
# Usage:
#   bash run_4dgs_thenewface_chunked.sh
#
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
RESULT_BASE=/data/shared/elaheh/4D_demo/thenewface_4dgs_chunked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

CHUNKS=10
FRAMES_PER_CHUNK=30
FRAME_STRIDE=1
FIRST_FRAME=1    # thenewface starts at frame 1

for CHUNK in $(seq 0 $((CHUNKS - 1))); do
    FRAME_START=$((FIRST_FRAME + CHUNK * FRAMES_PER_CHUNK * FRAME_STRIDE))
    RESULT_DIR="${RESULT_BASE}/thenewface_4dgs_chunk${CHUNK}_f30"

    # Build init args: chunk 0 uses SFM, chunk 1+ loads PLY from previous chunk
    INIT_ARGS=()
    DEFORM_ARGS=()

    if [[ $CHUNK -eq 0 ]]; then
        INIT_ARGS=(--init_type sfm --coarse_iters 3000)
        DEFORM_ARGS=(
            --deform_lr_delay_mult 0.01
            --deform_lr_warmup_steps 1000
            --progressive_time_warmup 10000
            --progressive_time_initial 0.1
        )
    else
        # Find previous chunk's last-frame PLY (all gaussian properties baked in)
        PREV_CHUNK=$((CHUNK - 1))
        PREV_FRAME_START=$((FIRST_FRAME + PREV_CHUNK * FRAMES_PER_CHUNK * FRAME_STRIDE))
        PREV_LAST_FRAME_IDX=$((PREV_FRAME_START + (FRAMES_PER_CHUNK - 1) * FRAME_STRIDE))
        PREV_PLY="${RESULT_BASE}/thenewface_4dgs_chunk${PREV_CHUNK}_f30/ply_per_frame/frame_$(printf '%06d' $PREV_LAST_FRAME_IDX).ply"
        PREV_CKPT="${RESULT_BASE}/thenewface_4dgs_chunk${PREV_CHUNK}_f30/ckpts/ckpt_29999_rank0.pt"

        if [[ ! -f "$PREV_PLY" ]]; then
            echo "ERROR: previous chunk PLY not found: ${PREV_PLY}"
            exit 1
        fi

        echo "Loading gaussians (pos/color/scale/opacity/quat) from: ${PREV_PLY}"
        INIT_ARGS=(
            --init_type ply
            --ply_path "$PREV_PLY"
            --coarse_iters 0
        )

        # No warmup — gaussians are pre-trained, start deformation immediately
        DEFORM_ARGS=(
            --deform_lr_delay_mult 1.0
            --deform_lr_warmup_steps 0
            --progressive_time_warmup 0
        )

        # Resume deformation network from previous chunk's checkpoint
        if [[ -f "$PREV_CKPT" ]]; then
            DEFORM_ARGS+=(--resume_deform_ckpt "$PREV_CKPT")
            echo "Resuming deformation from: ${PREV_CKPT}"
        else
            echo "WARNING: previous checkpoint not found: ${PREV_CKPT}, starting deformation fresh"
        fi
    fi

    echo "========================================"
    echo "Chunk ${CHUNK}: frames ${FRAME_START}–$((FRAME_START + (FRAMES_PER_CHUNK - 1) * FRAME_STRIDE))"
    echo "Result dir: ${RESULT_DIR}"
    echo "========================================"

    python simple_trainer_4dgs.py \
        --data_dir "$DATA_DIR" \
        --result_dir "$RESULT_DIR" \
        --dataset_mode rig \
        --num_frames $FRAMES_PER_CHUNK \
        --frame_stride $FRAME_STRIDE \
        --frame_start $FRAME_START \
        --max_steps 30000 \
        "${INIT_ARGS[@]}" \
        --data_factor 4 \
        --sh_degree 3 \
        --use_deformation \
        --deform_grid_resolution 64 \
        --deform_time_resolution 200 \
        --deform_feature_dim 32 \
        --deform_multires 1 2 4 8 \
        --deform_net_width 128 \
        --deform_net_depth 0 \
        --deform_time_pe_bands 4 \
        --deform_lr 1.6e-4 \
        --grid_lr 1.6e-3 \
        "${DEFORM_ARGS[@]}" \
        --max_num_gaussians 1000000 \
        --ssim_lambda 0.15 \
        --opacity_reg 0.001 \
        --scale_reg 0.01 \
        --plane_tv_weight 0.0001 \
        --time_smooth_weight 0.01 \
        --time_smooth_weight_final -1.0 \
        --time_smooth_order 2 \
        --l1_time_planes_weight 0.0001 \
        --weight_constraint_init 1.0 \
        --weight_constraint_after 0.2 \
        --weight_constraint_decay_iters 5000 \
        --strategy.refine-stop-iter 24000 \
        --tb_every 100 \
        --tb_image_every 200 \
        --eval_steps 5000 10000 15000 20000 25000 30000 \
        --save_steps 15000 30000 \
        --export_per_frame_ply \
        "$@"

    echo "Chunk ${CHUNK} done."
    echo ""
done

echo "All ${CHUNKS} chunks complete. Results in ${RESULT_BASE}/"
