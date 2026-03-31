#!/usr/bin/env bash
# Exp 8 — Mixed: Sine for xyz+sh, ReLU for rot+scale
# Hypothesis: simpler version of exp7 — does ELU on rot+scale actually help
#             over plain ReLU, or does Sine on xyz+sh dominate the gain?
# Key comparison: exp7 (ELU for rot+scale) vs this (ReLU for rot+scale)
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp8_sine_xyz_sh__relu_rot_scale" sine relu relu sine
