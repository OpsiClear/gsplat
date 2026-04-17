#!/usr/bin/env bash
# Batch per-frame ROI finetune for yehe_tech (frames 0..299), dynamic-only output.
#
# Recipe (matches the tested settings):
#   - max_steps 5000, data_factor 2, pad_sh_to 3
#   - densify_start 800, every 400, stop 3500
#   - scale_reg 0.01, max_scale_clamp 0.3, prune_opa 0.01
#
# Output: all_ply_ftune/{0000..0299}.ply  (dynamic-only subset)
set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}

DATA_DIR=/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/undistorted
IN_DIR=/data/shared/elaheh/final_4d_results/merge_ply_all_scenes/yehe_tech/ply_sequence_merged_35000_merged
ROI=/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/undistorted/static_dynamic_output/roi_bounds.npy
OUT_DIR=/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/roi_finetune/all_ply_ftune
TMP_BASE=/data/shared/elaheh/4D_demo/completed_indoor/yehe_tech/roi_finetune/_tmp_batch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p "$OUT_DIR" "$TMP_BASE"

FRAME_START=${FRAME_START:-0}
FRAME_END=${FRAME_END:-299}

for i in $(seq $FRAME_START $FRAME_END); do
    FI=$(printf '%04d' $i)
    OUT_PLY="$OUT_DIR/$FI.ply"
    IN_PLY="$IN_DIR/$FI.ply"

    if [[ ! -f "$IN_PLY" ]]; then
        echo "[$FI] SKIP — input missing: $IN_PLY"
        continue
    fi
    if [[ -f "$OUT_PLY" ]]; then
        echo "[$FI] SKIP — already done"
        continue
    fi

    TMP_DIR="$TMP_BASE/f${FI}"
    rm -rf "$TMP_DIR"

    echo "=== [$FI] training ==="
    python -u per_frame_finetune_roi.py \
        --data_dir "$DATA_DIR" \
        --per_frame_ply "$IN_PLY" \
        --roi_bounds_path "$ROI" \
        --result_dir "$TMP_DIR" \
        --frame_idx $i --max_steps 5000 --data_factor 2 \
        --densify_start 800 --densify_every 400 --densify_stop 3500 \
        --pad_sh_to 3 --bright_penalty 0 --save_dynamic_only \
        --lr_sh0 2.5e-4 --lr_shN 1.25e-5 --lr_scales 5e-5 --lr_opacities 5e-3 \
        --scale_reg 0.02 --max_scale_clamp 0.2

    if [[ -f "$TMP_DIR/frame_${FI}_dynamic.ply" ]]; then
        mv "$TMP_DIR/frame_${FI}_dynamic.ply" "$OUT_PLY"
        rm -rf "$TMP_DIR"
        echo "[$FI] -> $OUT_PLY"
    else
        echo "[$FI] ERROR: dynamic PLY not produced; keeping $TMP_DIR for inspection"
    fi
done

echo "All done. Output: $OUT_DIR/"
