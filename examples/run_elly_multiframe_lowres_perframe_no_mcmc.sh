#!/bin/bash
# Pure gradient fine-tuning (no MCMC, no strategy) for all frames.
# Frame 1: 5000 steps from inside.ply. Frames 2-150: 750 steps fine-tune.

set -e

SEP_DIR="/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/static_dynamic_output"

/home/elaheh/miniforge3/envs/gsplat/bin/python examples/run_multiframe_fast.py \
    --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \
    --result_dir /data/shared/elaheh/4D_demo/outdoor/elly/results_no_mcmc \
    --static_ply_path "${SEP_DIR}/outside_simplified_0.05.ply" \
    --separation_dir "${SEP_DIR}" \
    --init_ply "${SEP_DIR}/inside_clean_5000.ply" \
    --strategy mcmc \
    --frame_start 1 \
    --frame_end 150 \
    --frame1_steps 5000 \
    --ftune_steps 750 \
    --frame1_cap 10000 \
    --noise_lr 1e4 \
    --opacity_reg 0.0 \
    --scale_reg 0.0 \
    --prune_small_scale 0.0 \
    --prune_contribution 0.0 \
    --prune_every 5 \
    --densify_min_ratio 0.8 \
    --densify_burn_in 100 \
    --gpu 1
