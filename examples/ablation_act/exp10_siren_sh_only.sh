#!/usr/bin/env bash
# Exp 10 — SIREN for SH head only, ReLU for xyz/rot/scale
# Mirrors exp6 (sine SH only) but with SIREN init on the SH head hidden linear.
# Hypothesis: SH head benefits most from sinusoidal activation (exp6 confirmed);
#             SIREN init may further improve color representation quality.
# Key comparison: exp6 (sine SH only, default init) — best result so far.
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp10_siren_sh_only" relu relu relu siren
