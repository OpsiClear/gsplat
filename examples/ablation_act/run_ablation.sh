#!/usr/bin/env bash
# ============================================================
# Activation Ablation: launch all 8 experiments sequentially
#
# Usage:
#   bash run_ablation.sh                  # run all 8
#   bash run_ablation.sh 1 3 7            # run only configs 1, 3, 7
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
    CONFIGS=(1 2 3 4 5 6 7 8)
fi

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Running ablation config $cfg"
    echo "=========================================="
    bash "$SCRIPT_DIR/exp${cfg}_baseline.sh"         2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_all_sinerelu.sh"  2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_all_sine.sh"      2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_all_elu.sh"       2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_sine_pos.sh"      2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_sine_color.sh"    2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_mixed_elu.sh"     2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}_mixed_relu.sh"    2>/dev/null \
    || bash "$SCRIPT_DIR/exp${cfg}"*.sh
done

echo ""
echo "All selected experiments complete."
echo "Compare with: tensorboard --logdir /data/shared/elaheh/4D_demo/ablation_act/"
