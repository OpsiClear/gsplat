#!/usr/bin/env bash
# ============================================================
# Run activation ablation experiments (exps 2-8) sequentially.
# Baseline = existing run_thenewface_static_dynamic.sh result.
#
# Usage:
#   bash run_all.sh              # all 7 experiments
#   CUDA_VISIBLE_DEVICES=2 bash run_all.sh
#   bash run_all.sh 3 7 8        # run only experiments 3, 7, 8
#
# Results land in:
#   /data/shared/elaheh/4D_demo/ablation_act/exp{N}_*/
#
# Compare everything (including baseline) in TensorBoard:
#   tensorboard --logdir_spec \
#     baseline:/data/shared/elaheh/4D_demo/thenewface_4dgs_f300_static_dynamic/tb,\
#     all_sinerelu:/data/shared/elaheh/4D_demo/ablation_act/exp2_all_sinerelu/tb,\
#     all_sine:/data/shared/elaheh/4D_demo/ablation_act/exp3_all_sine/tb,\
#     all_elu:/data/shared/elaheh/4D_demo/ablation_act/exp4_all_elu/tb,\
#     sine_xyz:/data/shared/elaheh/4D_demo/ablation_act/exp5_sine_xyz_only/tb,\
#     sine_sh:/data/shared/elaheh/4D_demo/ablation_act/exp6_sine_sh_only/tb,\
#     mixed_elu:/data/shared/elaheh/4D_demo/ablation_act/exp7_sine_xyz_sh__elu_rot_scale/tb,\
#     mixed_relu:/data/shared/elaheh/4D_demo/ablation_act/exp8_sine_xyz_sh__relu_rot_scale/tb
# ============================================================
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

ALL=(2 3 4 5 6 7 8)
SELECTED=("${@:-${ALL[@]}}")

for n in "${SELECTED[@]}"; do
    case $n in
        2) bash "$DIR/exp2_all_sinerelu.sh" ;;
        3) bash "$DIR/exp3_all_sine.sh" ;;
        4) bash "$DIR/exp4_all_elu.sh" ;;
        5) bash "$DIR/exp5_sine_pos_only.sh" ;;
        6) bash "$DIR/exp6_sine_color_only.sh" ;;
        7) bash "$DIR/exp7_mixed_elu.sh" ;;
        8) bash "$DIR/exp8_mixed_relu.sh" ;;
        *) echo "Unknown exp $n (valid: 2-8)"; exit 1 ;;
    esac
done

echo ""
echo "============================================================"
echo "  All done. Compare in TensorBoard (includes baseline):"
echo ""
echo "  tensorboard --logdir_spec \\"
echo "    baseline:/data/shared/elaheh/4D_demo/thenewface_4dgs_f300_static_dynamic/tb,\\"
echo "    all_sinerelu:/data/shared/elaheh/4D_demo/ablation_act/exp2_all_sinerelu/tb,\\"
echo "    all_sine:/data/shared/elaheh/4D_demo/ablation_act/exp3_all_sine/tb,\\"
echo "    all_elu:/data/shared/elaheh/4D_demo/ablation_act/exp4_all_elu/tb,\\"
echo "    sine_xyz:/data/shared/elaheh/4D_demo/ablation_act/exp5_sine_xyz_only/tb,\\"
echo "    sine_sh:/data/shared/elaheh/4D_demo/ablation_act/exp6_sine_sh_only/tb,\\"
echo "    mixed_elu:/data/shared/elaheh/4D_demo/ablation_act/exp7_sine_xyz_sh__elu_rot_scale/tb,\\"
echo "    mixed_relu:/data/shared/elaheh/4D_demo/ablation_act/exp8_sine_xyz_sh__relu_rot_scale/tb"
echo "============================================================"
