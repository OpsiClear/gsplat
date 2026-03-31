#!/usr/bin/env bash
# Exp 5 — Sine for position only (head_xyz), rest ReLU
# Hypothesis: isolates whether position head is the primary beneficiary of Sine
# Key comparison: exp1 (baseline) and exp8 (sine xyz+sh)
# If exp5 ≈ exp8, then sh doesn't matter much; if exp5 < exp8, color helps too
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp5_sine_xyz_only" sine relu relu relu
