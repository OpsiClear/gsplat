#!/bin/bash
# Train static 3DGS on thenewface (frame 000001) with inverted motion masks.
# motion_mask.png convention: black=keep (train), white=exclude → invert_masks=True
# test_every=0 → all 45 cameras are used for training; val uses ~5 evenly-spaced subset.

set -e

DATA_DIR="/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted"
RESULT_DIR="/data/shared/elaheh/thenewface_static_v2"

# Pick first free GPU among 1,2,3,4
GPU=""
for g in 1 2 3 4; do
    FREE=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    if [ "$FREE" -eq 0 ]; then
        GPU=$g
        break
    fi
done
if [ -z "$GPU" ]; then
    echo "No free GPU found among 1,2,3,4. Exiting."
    exit 1
fi

echo "=== thenewface static training ==="
echo "  data_dir : $DATA_DIR"
echo "  result   : $RESULT_DIR"
echo "  GPU      : $GPU"
echo ""

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=$GPU /home/elaheh/miniforge3/envs/gsplat/bin/python simple_trainer.py default \
    --data_dir     "$DATA_DIR"  \
    --result_dir   "$RESULT_DIR" \
    --port         6071         \
    --max_steps    30000        \
    --eval_steps   7000 15000 30000 \
    --save_steps   7000 15000 30000 \
    --ply_steps    7000 30000       \
    --test_every   0            \
    --sh_degree    0

echo ""
echo "=== Done! Results saved to $RESULT_DIR ==="
