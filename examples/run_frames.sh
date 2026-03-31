#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gsplat

# Base configuration
DATA_ROOT="/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/roma"
RESULT_ROOT="/data/shared/elaheh/outdoor_elly_roma_results"
START_FRAME=0
END_FRAME=30

mkdir -p "$RESULT_ROOT"

for i in $(seq -f "%06g" $START_FRAME $END_FRAME); do
    FRAME_DIR="$DATA_ROOT/frame_$i"
    RESULT_DIR="$RESULT_ROOT/frame_$i"
    
    echo "========================================================="
    echo "Processing Frame $i"
    echo "Data Path: $FRAME_DIR"
    echo "Result Path: $RESULT_DIR"
    echo "========================================================="
    
    # Check if data directory exists
    if [ ! -d "$FRAME_DIR" ]; then
        echo "Warning: Directory $FRAME_DIR does not exist. Skipping."
        continue
    fi

    # Run gsplat training
    # Note: We use the 'default' subcommand as required by simple_trainer.py
    CUDA_VISIBLE_DEVICES=1 python examples/simple_trainer.py default \
        --data-dir "$FRAME_DIR" \
        --result-dir "$RESULT_DIR" \
        --random-bkgd \
        --test-every 10000 \
        --max-steps 30000 \
        --disable-viewer \
        --save-ply \
        --ply-steps 7000 10000 12000 15000 20000 25000 30000

    echo "Completed Frame $i"
done
