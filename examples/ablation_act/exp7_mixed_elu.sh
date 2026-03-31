#!/usr/bin/env bash
# Exp 7 — Mixed: Sine for xyz+sh, ELU for rot+scale  (recommended config)
# Hypothesis: captures high-freq position+color with Sine while keeping
#             rotation/scale monotone (no oscillation artifacts in quats)
# Key comparison: exp8 (same but ReLU for rot+scale instead of ELU)
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp7_sine_xyz_sh__elu_rot_scale" sine elu elu sine
