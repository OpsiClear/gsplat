#!/bin/bash
# Experiment B: DefaultStrategy fine-tuning (split/duplicate/prune every 300 steps).
# Frame 1: MCMC to grow from inside.ply seeds to 10k Gaussians.
# Frames 2-150: DefaultStrategy with gradient-based densification, ROI prune.

set -e

SEP_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/static_dynamic_output"

/home/elaheh/miniforge3/envs/gsplat/bin/python examples/run_multiframe_fast.py \
    --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \
    --result_dir /data/shared/elaheh/4D_demo/outdoor/elly/results_default_strategy \
    --static_ply_path "${SEP_DIR}/outside.ply" \
    --separation_dir "${SEP_DIR}" \
    --init_ply "${SEP_DIR}/inside.ply" \
    --strategy default \
    --frame_start 1 \
    --frame_end 150 \
    --frame1_steps 1500 \
    --ftune_steps 750 \
    --frame1_cap 10000 \
    --noise_lr 1e4 \
    --scale_reg 0.0001 \
    --opacity_reg 0.01 \
    --needle_reg 0.005 \
    --small_scale_reg 0.01 \
    --min_scale 0.002 \
    --prune_opa 0.02 \
    --prune_small_scale 0.001 \
    --prune_contribution 0.005 \
    --opacity_lr 1e-2 \
    --reset_opacity_at 0.5 \
    --gpu 3
