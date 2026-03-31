#!/usr/bin/env bash
# Exp 4 — All ELU
# Hypothesis: smooth, monotone, no dead neurons — theoretically best for
#             deformation fields; does smoothness translate to better PSNR?
# Watch for: slower convergence vs ReLU due to weaker sparsity
set -e
source "$(dirname "$0")/_base.sh"
run_experiment "exp4_all_elu" elu elu elu elu
