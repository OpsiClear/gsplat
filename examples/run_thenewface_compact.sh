#!/usr/bin/env bash
# MinSplat: Compact MCMC training on thenewface with PPISP
# Ultra-low res (data_factor=20), 10k Gaussian cap, 7k initial points
# Uses simple_trainer_ftune.py for PPISP post-processing support
set -e

export CUDA_VISIBLE_DEVICES=4

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/thenewface_compact_mcmc
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_ftune.py mcmc \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --data_factor 20 \
    --init_num_pts 7000 \
    --max_steps 30000 \
    --test_every 100000 \
    --sh_degree 3 \
    --ssim_lambda 0.05 \
    --antialiased \
    \
    --strategy.cap-max 10000 \
    --strategy.refine-every 300 \
    --strategy.refine-stop-iter 25000 \
    --strategy.refine-start-iter 500 \
    --strategy.noise-lr 1e4 \
    --strategy.min-opacity 0.001 \
    --strategy.verbose \
    --opacity_reg 0.001 \
    --scale_reg 0.0001 \
    --eval_steps 10000 20000 30000 \
    --save_steps 30000 \
    --ply_steps 30000 \
    --tb_every 100 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
