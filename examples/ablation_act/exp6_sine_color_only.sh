#!/usr/bin/env bash
# Exp 6 — Sine for color only (head_sh), rest ReLU
# Hypothesis: isolates whether SH head is the primary beneficiary of Sine
# Key comparison: exp1 (baseline) and exp5 (sine xyz only)
# If exp6 > exp5, color matters more than position for quality improvement
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp6_sine_sh_only" relu relu relu sine
