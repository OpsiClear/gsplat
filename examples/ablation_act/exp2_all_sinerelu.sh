#!/usr/bin/env bash
# Exp 2 — All SineReLU (eps=0.01)
# Hypothesis: fixes dead neurons in all heads; oscillation bounded by eps
# Watch for: jitter in rotation deltas at later training steps
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp2_all_sinerelu" sinerelu sinerelu sinerelu sinerelu
