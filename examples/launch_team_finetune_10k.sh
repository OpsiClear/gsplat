#!/usr/bin/env bash
# 4-GPU stride-4 per-frame ROI finetune for team chunk0 (100 frames).
# 10k steps, densify_stop=6500, 0.9x LR, dynamic-only output.
# Final PLYs: <BASE>/finetune_10k/XXXX.ply  (XXXX = 4-digit frame idx)
set -u

BASE=/data/shared/elaheh/4D_demo/team_4dgs_static_dynamic_chunked/chunk0_f100
DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/team/undistorted
ROI=${DATA_DIR}/static_dynamic_output/roi_bounds_padded.npy
STATIC_PLY=${DATA_DIR}/static_dynamic_output/outside_simplified_0.3_voxclean.ply
DYN_DIR=${BASE}/ply_per_frame
OUT_ROOT=${BASE}/finetune_10k
LOG_DIR=${OUT_ROOT}/_logs
mkdir -p "$LOG_DIR"

cd /home/elaheh/projects/gsplat

run_gpu () {
    local GPU=$1; shift
    local FRAMES=("$@")
    local LOG=${LOG_DIR}/gpu${GPU}.log
    : > "$LOG"
    for F in "${FRAMES[@]}"; do
        local F4=$(printf '%04d' $F)
        local F6=$(printf '%06d' $F)
        local WORK_DIR=${OUT_ROOT}/work/f${F4}
        local FLAT_PLY=${OUT_ROOT}/${F4}.ply
        if [[ -f "$FLAT_PLY" ]]; then
            echo "[GPU${GPU}] f${F4}: SKIP (exists)" | tee -a "$LOG"
            continue
        fi
        echo "[GPU${GPU}] f${F4}: START" | tee -a "$LOG"
        CUDA_VISIBLE_DEVICES=${GPU} python examples/per_frame_finetune_roi.py \
            --data_dir "$DATA_DIR" \
            --dynamic_ply_path ${DYN_DIR}/frame_${F6}.ply \
            --static_ply_path "$STATIC_PLY" \
            --roi_bounds_path "$ROI" \
            --result_dir "$WORK_DIR" \
            --frame_idx $F --max_steps 10000 --data_factor 2 \
            --lr_scales 9e-4 --lr_opacities 4.5e-2 --lr_sh0 2.25e-3 --lr_shN 1.125e-4 \
            --densify_stop 6500 --save_dynamic_only \
            >> "$LOG" 2>&1
        if [[ -f "$WORK_DIR/frame_${F4}_dynamic.ply" ]]; then
            mv "$WORK_DIR/frame_${F4}_dynamic.ply" "$FLAT_PLY"
            echo "[GPU${GPU}] f${F4}: DONE -> ${FLAT_PLY}" | tee -a "$LOG"
        else
            echo "[GPU${GPU}] f${F4}: FAILED (no PLY produced)" | tee -a "$LOG"
        fi
    done
}

F1=(); F2=(); F3=(); F4=(); F5=(); F6=()
for F in $(seq 0 99); do
    case $((F % 6)) in
        0) F1+=($F) ;;
        1) F2+=($F) ;;
        2) F3+=($F) ;;
        3) F4+=($F) ;;
        4) F5+=($F) ;;
        5) F6+=($F) ;;
    esac
done

echo "GPU1: ${F1[*]}"
echo "GPU2: ${F2[*]}"
echo "GPU3: ${F3[*]}"
echo "GPU4: ${F4[*]}"
echo "GPU5: ${F5[*]}"
echo "GPU6: ${F6[*]}"

run_gpu 1 "${F1[@]}" &
run_gpu 2 "${F2[@]}" &
run_gpu 3 "${F3[@]}" &
run_gpu 4 "${F4[@]}" &
run_gpu 5 "${F5[@]}" &
run_gpu 6 "${F6[@]}" &
wait
echo "ALL DONE"
