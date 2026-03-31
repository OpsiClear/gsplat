#!/usr/bin/env bash
# Exp 3 — All pure Sine (no SIREN init, keep near-zero last-layer init)
# Hypothesis: user reports Sine gives better position+color; this tests if
#             Sine everywhere helps or if rot/scale become unstable
# Watch for: quaternion normalization artifacts, scale explosion
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp3_all_sine" sine sine sine sine
