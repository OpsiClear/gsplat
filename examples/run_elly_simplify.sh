#!/bin/bash
# Simplify static (outside) Gaussians using NanoGS pairwise merging.
# Dynamic (inside) PLY is kept unchanged.

set -e

DATA_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/undistorted"
STATIC_PLY="${DATA_DIR}/static_dynamic_output/outside.ply"
DYNAMIC_PLY="${DATA_DIR}/static_dynamic_output/inside.ply"

RATIO=0.5          # keep 50% of static splats
K=16               # KNN neighbourhood
MERGE_CAP=0.5      # max merges per pass
OPACITY_THR=0.1    # prune low-opacity before merging
LAM_GEO=1.0
LAM_SH=1.0

OUTPUT="${DATA_DIR}/static_dynamic_output/outside_simplified_${RATIO}.ply"

echo "=== NanoGS simplification of static Gaussians ==="
echo "  static  : ${STATIC_PLY}"
echo "  dynamic : ${DYNAMIC_PLY}  (unchanged)"
echo "  ratio   : ${RATIO}"
echo "  output  : ${OUTPUT}"
echo ""

cd "$(dirname "$0")"

python simplify_gaussians.py \
    --static_ply "$STATIC_PLY" \
    -o "$OUTPUT" \
    -r "$RATIO" \
    --k "$K" \
    --merge_cap "$MERGE_CAP" \
    --opacity_threshold "$OPACITY_THR" \
    --lam_geo "$LAM_GEO" \
    --lam_sh "$LAM_SH"

echo ""
echo "=== Done! Simplified static PLY: ${OUTPUT} ==="
echo "=== Dynamic PLY unchanged:       ${DYNAMIC_PLY} ==="
