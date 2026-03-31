#!/usr/bin/env bash
# Exp 9 — All SIREN (Sine activation + SIREN init on hidden linear, w0=1)
# Mirrors exp2 (all SineReLU) but with principled SIREN initialization.
# Hypothesis: SIREN init spreads hidden weights so sine is used in its
#             well-utilized regime (not stuck near 0), improving convergence.
# Key comparison: exp2 (all SineReLU) — does SIREN init beat SineReLU?
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp9_all_siren" siren siren siren siren
