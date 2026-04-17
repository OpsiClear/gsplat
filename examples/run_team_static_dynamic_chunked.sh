#!/usr/bin/env bash
# Chunked 4DGS training on team dataset with frozen static background.
#
# - Static PLY (frozen, no gradients): outside_simplified_0.3_voxclean.ply
# - Dynamic PLY (trainable, deformable): inside.ply
# - ROI bounds gate HexPlane deformation (only splats inside ROI deform)
# - 3 chunks × ~100 frames over 299 total frames
# - Chunk handoff: next_static.ply + next_dynamic.ply + deform ckpt + AABB
set -e

export CUDA_VISIBLE_DEVICES=0

DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/team/undistorted
BASE_SD=/data/shared/elaheh/4D_demo/completed_indoor/team/undistorted/static_dynamic_output
STATIC_PLY_CHUNK0="${BASE_SD}/outside_simplified_0.3_voxclean.ply"
DYNAMIC_PLY_CHUNK0="${BASE_SD}/inside.ply"
ROI_BOUNDS_PATH="${BASE_SD}/roi_bounds.npy"

RESULT_BASE=/data/shared/elaheh/4D_demo/team_4dgs_static_dynamic_chunked
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TOTAL_FRAMES=299
FRAMES_PER_CHUNK=100
FRAME_STRIDE=1
FIRST_FRAME=0
CHUNKS=3   # 100, 100, 99

MAX_STEPS=30000

for CHUNK in $(seq 0 $((CHUNKS - 1))); do
    FRAME_START=$((FIRST_FRAME + CHUNK * FRAMES_PER_CHUNK * FRAME_STRIDE))
    REMAINING=$((TOTAL_FRAMES - CHUNK * FRAMES_PER_CHUNK))
    if [[ $REMAINING -lt $FRAMES_PER_CHUNK ]]; then
        NUM_FRAMES=$REMAINING
    else
        NUM_FRAMES=$FRAMES_PER_CHUNK
    fi

    RESULT_DIR="${RESULT_BASE}/chunk${CHUNK}_f${NUM_FRAMES}"

    INIT_ARGS=()
    CHUNK_ARGS=()
    DEFORM_WARMUP_ARGS=()

    if [[ $CHUNK -eq 0 ]]; then
        STATIC_PLY="$STATIC_PLY_CHUNK0"
        DYNAMIC_PLY="$DYNAMIC_PLY_CHUNK0"
        DEFORM_WARMUP_ARGS=(
            --deform_lr_delay_mult 0.01
            --deform_lr_warmup_steps 1000
            --progressive_time_warmup 10000
            --progressive_time_initial 0.05
        )
    else
        PREV_CHUNK=$((CHUNK - 1))
        PREV_REMAINING=$((TOTAL_FRAMES - PREV_CHUNK * FRAMES_PER_CHUNK))
        if [[ $PREV_REMAINING -lt $FRAMES_PER_CHUNK ]]; then
            PREV_NUM_FRAMES=$PREV_REMAINING
        else
            PREV_NUM_FRAMES=$FRAMES_PER_CHUNK
        fi
        PREV_RESULT_DIR="${RESULT_BASE}/chunk${PREV_CHUNK}_f${PREV_NUM_FRAMES}"
        STATIC_PLY="${PREV_RESULT_DIR}/ply/next_static.ply"
        DYNAMIC_PLY="${PREV_RESULT_DIR}/ply/next_dynamic.ply"
        PREV_CKPT="${PREV_RESULT_DIR}/ckpts/ckpt_$((MAX_STEPS - 1))_rank0.pt"

        if [[ ! -f "$STATIC_PLY" || ! -f "$DYNAMIC_PLY" ]]; then
            echo "ERROR: handoff PLYs missing: $STATIC_PLY or $DYNAMIC_PLY"
            exit 1
        fi
        CHUNK_ARGS+=(--resume_aabb_from "$PREV_CKPT")
        if [[ -f "$PREV_CKPT" ]]; then
            CHUNK_ARGS+=(--resume_deform_ckpt "$PREV_CKPT")
            echo "Resuming deformation from: ${PREV_CKPT}"
        fi
        DEFORM_WARMUP_ARGS=(
            --deform_lr_delay_mult 1.0
            --deform_lr_warmup_steps 0
            --progressive_time_warmup 0
        )
    fi

    echo "========================================"
    echo "Chunk ${CHUNK}: frames ${FRAME_START}..$((FRAME_START + NUM_FRAMES - 1)) (${NUM_FRAMES} frames)"
    echo "  static : ${STATIC_PLY}"
    echo "  dynamic: ${DYNAMIC_PLY}"
    echo "  result : ${RESULT_DIR}"
    echo "========================================"

    python simple_trainer_static_dynamic.py \
        --data_dir "$DATA_DIR" \
        --result_dir "$RESULT_DIR" \
        --static_ply_path "$STATIC_PLY" \
        --dynamic_ply_path "$DYNAMIC_PLY" \
        --dataset_mode rig \
        --num_frames $NUM_FRAMES \
        --frame_stride $FRAME_STRIDE \
        --frame_start $FRAME_START \
        --max_steps $MAX_STEPS \
        --coarse_iters 0 \
        --init_type ply \
        --data_factor 2 \
        --sh_degree 3 \
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
        "${DEFORM_WARMUP_ARGS[@]}" \
        --roi_bounds_path "$ROI_BOUNDS_PATH" \
        --roi_padding 0.1 \
        --max_num_gaussians 1500000 \
        --ssim_lambda 0.2 \
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
        --strategy.refine-stop-iter $((MAX_STEPS * 4 / 5)) \
        --strategy.reset-every 3000 \
        --tb_every 100 \
        --tb_image_every 200 \
        --eval_steps 5000 10000 15000 20000 25000 30000 \
        --save_steps 15000 30000 \
        --export_per_frame_ply \
        "${CHUNK_ARGS[@]}" \
        "$@"

    echo "Chunk ${CHUNK} done."
    echo ""
done

echo "All ${CHUNKS} chunks complete. Results in ${RESULT_BASE}/"
