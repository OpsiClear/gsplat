#!/usr/bin/env bash
# Chunked 4DGS training for team dataset — 10 chunks of 30 frames (last=29), no masks
#
# Based on mike_chunked config (same indoor rig setup):
#   num_frames:  30 per chunk (29 for last)
#   frame_stride: 1
#   max_steps:   30000 per chunk
#
# Chunk 0: SFM init, full coarse phase + warmups
# Chunk 1+: Load ALL gaussian properties from previous chunk's last-frame PLY,
#           resume deformation from previous checkpoint,
#           skip coarse phase, no LR warmup.
#
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/team/undistorted
RESULT_BASE=/data/shared/elaheh/4D_demo/team_4dgs_chunked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

TOTAL_FRAMES=299
CHUNKS=10
FRAMES_PER_CHUNK=30
FRAME_STRIDE=1
FIRST_FRAME=0

for CHUNK in $(seq 0 $((CHUNKS - 1))); do
    FRAME_START=$((FIRST_FRAME + CHUNK * FRAMES_PER_CHUNK * FRAME_STRIDE))

    # Last chunk may have fewer frames
    REMAINING=$((TOTAL_FRAMES - CHUNK * FRAMES_PER_CHUNK))
    if [[ $REMAINING -lt $FRAMES_PER_CHUNK ]]; then
        NUM_FRAMES=$REMAINING
    else
        NUM_FRAMES=$FRAMES_PER_CHUNK
    fi

    RESULT_DIR="${RESULT_BASE}/team_4dgs_chunk${CHUNK}_f${NUM_FRAMES}"

    # Build init args: chunk 0 uses SFM, chunk 1+ loads PLY from previous chunk
    INIT_ARGS=()
    DEFORM_ARGS=()

    if [[ $CHUNK -eq 0 ]]; then
        INIT_ARGS=(--init_type sfm --coarse_iters 3000)
        DEFORM_ARGS=(
            --deform_lr_delay_mult 0.01
            --deform_lr_warmup_steps 1000
            --progressive_time_warmup 10000
            --progressive_time_initial 0.05
        )
    else
        # Find previous chunk's last-frame PLY
        PREV_CHUNK=$((CHUNK - 1))
        PREV_FRAME_START=$((FIRST_FRAME + PREV_CHUNK * FRAMES_PER_CHUNK * FRAME_STRIDE))
        PREV_REMAINING=$((TOTAL_FRAMES - PREV_CHUNK * FRAMES_PER_CHUNK))
        if [[ $PREV_REMAINING -lt $FRAMES_PER_CHUNK ]]; then
            PREV_NUM_FRAMES=$PREV_REMAINING
        else
            PREV_NUM_FRAMES=$FRAMES_PER_CHUNK
        fi
        PREV_LAST_FRAME_IDX=$((PREV_FRAME_START + (PREV_NUM_FRAMES - 1) * FRAME_STRIDE))
        PREV_RESULT_DIR="${RESULT_BASE}/team_4dgs_chunk${PREV_CHUNK}_f${PREV_NUM_FRAMES}"
        PREV_PLY="${PREV_RESULT_DIR}/ply_per_frame/frame_$(printf '%06d' $PREV_LAST_FRAME_IDX).ply"
        PREV_CKPT="${PREV_RESULT_DIR}/ckpts/ckpt_29999_rank0.pt"

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

        # No warmup — start deformation immediately
        DEFORM_ARGS=(
            --deform_lr_delay_mult 1.0
            --deform_lr_warmup_steps 0
            --progressive_time_warmup 0
        )

        if [[ -f "$PREV_CKPT" ]]; then
            DEFORM_ARGS+=(--resume_deform_ckpt "$PREV_CKPT")
            echo "Resuming deformation from: ${PREV_CKPT}"
        else
            echo "WARNING: previous checkpoint not found: ${PREV_CKPT}, starting deformation fresh"
        fi
    fi

    echo "========================================"
    echo "Chunk ${CHUNK}: frames ${FRAME_START}–$((FRAME_START + (NUM_FRAMES - 1) * FRAME_STRIDE)) (${NUM_FRAMES} frames)"
    echo "Result dir: ${RESULT_DIR}"
    echo "========================================"

    python simple_trainer_4dgs.py \
        --data_dir "$DATA_DIR" \
        --result_dir "$RESULT_DIR" \
        --dataset_mode rig \
        --num_frames $NUM_FRAMES \
        --frame_stride $FRAME_STRIDE \
        --frame_start $FRAME_START \
        --max_steps 30000 \
        "${INIT_ARGS[@]}" \
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
        "${DEFORM_ARGS[@]}" \
        --max_num_gaussians 1000000 \
        --ssim_lambda 0.2 \
        --opacity_reg 0.001 \
        --scale_reg 0.01 \
        --plane_tv_weight 0.0001 \
        --time_smooth_weight 0.0007 \
        --time_smooth_weight_final -1.0 \
        --time_smooth_order 2 \
        --l1_time_planes_weight 0.0001 \
        --weight_constraint_init 1.0 \
        --weight_constraint_after 0.5 \
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
