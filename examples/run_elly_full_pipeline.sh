#!/bin/bash
# Full pipeline: Train frame 1 → Separate static/dynamic → Simplify → Multiframe
#
# Step 1: Train frame 1 fully with DefaultStrategy (39k steps, refine_every=500)
# Step 2: Separate static/dynamic using WAFT's run_static_dynamic.py
# Step 3: Simplify static PLY and dynamic PLY (dynamic → 5k Gaussians)
# Step 4: Run multiframe fine-tuning (two experiments: no_mcmc + default_strategy)

set -e

# ---- Config ----
PYTHON="/home/elaheh/miniforge3/envs/gsplat/bin/python"
DATA_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/undistorted"
COLMAP_SPARSE="${DATA_DIR}/sparse/0"
IMAGES_DIR="${DATA_DIR}/images"
DATA_FACTOR=16
FRAME_NUM=1
GPU_TRAIN=1
GPU_EXP_A=1
GPU_EXP_B=2
FRAME_START=1
FRAME_END=150

# Step 1 output
TRAIN_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/frame1_full_train"
TRAIN_STEPS=39000

# Step 2 output
SEP_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/frame1_static_dynamic"

# Step 3 output (simplification targets)
DYN_TARGET=5000

# Step 4 output
RESULT_A="/data/shared/elaheh/4D_demo/outdoor/elly/results_no_mcmc"
RESULT_B="/data/shared/elaheh/4D_demo/outdoor/elly/results_default_strategy"

FTUNE_STEPS=750
FRAME1_STEPS=1500

echo "============================================"
echo "  Full Pipeline: Elly 150-frame"
echo "============================================"

# =========================================================
# Step 1: Train frame 1 fully
# =========================================================
echo ""
echo "[Step 1] Training frame ${FRAME_NUM} for ${TRAIN_STEPS} steps on GPU ${GPU_TRAIN}..."

if [ -f "${TRAIN_DIR}/ply/point_cloud_$((TRAIN_STEPS-1)).ply" ]; then
    echo "[Step 1] SKIP — PLY already exists at ${TRAIN_DIR}/ply/point_cloud_$((TRAIN_STEPS-1)).ply"
else
    CUDA_VISIBLE_DEVICES=${GPU_TRAIN} ${PYTHON} examples/simple_trainer.py default \
        --data-dir "${DATA_DIR}" \
        --data-factor ${DATA_FACTOR} \
        --frame-num ${FRAME_NUM} \
        --max-steps ${TRAIN_STEPS} \
        --result-dir "${TRAIN_DIR}" \
        --disable-viewer \
        --test-every -1 \
        --strategy.refine-every 500
    echo "[Step 1] DONE — PLY at ${TRAIN_DIR}/ply/point_cloud_$((TRAIN_STEPS-1)).ply"
fi

PLY_PATH="${TRAIN_DIR}/ply/point_cloud_$((TRAIN_STEPS-1)).ply"

# =========================================================
# Step 2: Separate static/dynamic
# =========================================================
echo ""
echo "[Step 2] Separating static/dynamic..."

if [ -f "${SEP_DIR}/inside.ply" ] && [ -f "${SEP_DIR}/outside.ply" ]; then
    echo "[Step 2] SKIP — inside.ply and outside.ply already exist in ${SEP_DIR}"
else
    ${PYTHON} /home/elaheh/projects/WAFT/static_dynamic_split/run_static_dynamic.py \
        --colmap-sparse "${COLMAP_SPARSE}" \
        --images-dir "${IMAGES_DIR}" \
        --splat-ply "${PLY_PATH}" \
        --n-cameras 12 \
        --frame-stride 10 \
        --dilation-radius 3 \
        --output-dir "${SEP_DIR}"
    echo "[Step 2] DONE — inside.ply and outside.ply in ${SEP_DIR}"
fi

# =========================================================
# Step 3: Simplify PLYs
# =========================================================
echo ""
echo "[Step 3] Simplifying PLYs..."

# Count dynamic Gaussians
DYN_COUNT=$(${PYTHON} -c "
from gsplat.exporter import load_ply_gaussian
m,_,_,_,_,_ = load_ply_gaussian('${SEP_DIR}/inside.ply', device='cpu')
print(len(m))
")
echo "  Dynamic Gaussians: ${DYN_COUNT}"

# Compute ratio for dynamic (target 5k)
DYN_RATIO=$(${PYTHON} -c "print(min(0.99, ${DYN_TARGET} / ${DYN_COUNT}))")
echo "  Dynamic simplification ratio: ${DYN_RATIO} (target ${DYN_TARGET})"

# Simplify dynamic
DYN_SIMPLIFIED="${SEP_DIR}/inside_simplified_${DYN_TARGET}.ply"
if [ -f "${DYN_SIMPLIFIED}" ]; then
    echo "  [Dynamic] SKIP — already exists"
else
    echo "  [Dynamic] Simplifying ${DYN_COUNT} → ~${DYN_TARGET}..."
    ${PYTHON} examples/simplify_gaussians.py \
        --static_ply "${SEP_DIR}/inside.ply" \
        -o "${DYN_SIMPLIFIED}" \
        -r ${DYN_RATIO}
fi

# Simplify static (keep 20% — static is large but mostly background)
STATIC_COUNT=$(${PYTHON} -c "
from gsplat.exporter import load_ply_gaussian
m,_,_,_,_,_ = load_ply_gaussian('${SEP_DIR}/outside.ply', device='cpu')
print(len(m))
")
echo "  Static Gaussians: ${STATIC_COUNT}"

STATIC_SIMPLIFIED="${SEP_DIR}/outside_simplified_0.2.ply"
if [ -f "${STATIC_SIMPLIFIED}" ]; then
    echo "  [Static] SKIP — already exists"
else
    echo "  [Static] Simplifying to 20%..."
    ${PYTHON} examples/simplify_gaussians.py \
        --static_ply "${SEP_DIR}/outside.ply" \
        -o "${STATIC_SIMPLIFIED}" \
        -r 0.2
fi

echo "[Step 3] DONE"

# =========================================================
# Step 4: Run multiframe (two experiments in parallel)
# =========================================================
echo ""
echo "[Step 4] Launching multiframe experiments..."

# Experiment A: No MCMC (pure gradient fine-tuning)
echo "  [Exp A] No MCMC → GPU ${GPU_EXP_A}, result: ${RESULT_A}"
CUDA_VISIBLE_DEVICES=${GPU_EXP_A} ${PYTHON} examples/run_multiframe_fast.py \
    --data_dir "${DATA_DIR}" \
    --result_dir "${RESULT_A}" \
    --static_ply_path "${STATIC_SIMPLIFIED}" \
    --separation_dir "${SEP_DIR}" \
    --init_ply "${DYN_SIMPLIFIED}" \
    --strategy mcmc \
    --frame_start ${FRAME_START} \
    --frame_end ${FRAME_END} \
    --frame1_steps ${FRAME1_STEPS} \
    --ftune_steps ${FTUNE_STEPS} \
    --frame1_cap ${DYN_TARGET} \
    --ftune_cap ${DYN_TARGET} \
    --noise_lr 1e4 \
    --scale_reg 0.0001 \
    --opacity_reg 0.001 \
    --gpu 0 &
PID_A=$!

# Experiment B: DefaultStrategy (split/duplicate/prune)
echo "  [Exp B] DefaultStrategy → GPU ${GPU_EXP_B}, result: ${RESULT_B}"
CUDA_VISIBLE_DEVICES=${GPU_EXP_B} ${PYTHON} examples/run_multiframe_fast.py \
    --data_dir "${DATA_DIR}" \
    --result_dir "${RESULT_B}" \
    --static_ply_path "${STATIC_SIMPLIFIED}" \
    --separation_dir "${SEP_DIR}" \
    --init_ply "${DYN_SIMPLIFIED}" \
    --strategy default \
    --frame_start ${FRAME_START} \
    --frame_end ${FRAME_END} \
    --frame1_steps ${FRAME1_STEPS} \
    --ftune_steps ${FTUNE_STEPS} \
    --frame1_cap ${DYN_TARGET} \
    --ftune_cap ${DYN_TARGET} \
    --noise_lr 1e4 \
    --scale_reg 0.0001 \
    --opacity_reg 0.001 \
    --gpu 0 &
PID_B=$!

echo "  Waiting for both experiments (PIDs: ${PID_A}, ${PID_B})..."
wait ${PID_A}
echo "  [Exp A] DONE"
wait ${PID_B}
echo "  [Exp B] DONE"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Exp A results: ${RESULT_A}/all_ply/"
echo "  Exp B results: ${RESULT_B}/all_ply/"
echo "============================================"
