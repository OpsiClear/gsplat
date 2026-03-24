#!/usr/bin/env bash
# Run 4DGS on elly — 150 frames, static+dynamic PLY init, no masks
# v2: improved hyperparameters after audit
#
# Key changes from v1:
#   opacity_reg  0.001  → 0.0001  (face Gaussians must be opaque; 0.001 caused 160K→80K pruning)
#   scale_reg    0.01   → 0.005   (less aggressive scale shrinkage)
#   time_smooth_weight_final -1.0 → 0.001  (anneal from 0.01→0.001 for sharper late-training motion)
#   weight_constraint_after  0.2  → 0.0   (let constraint fully decay; residual 0.2 was fighting position learning)
#   weight_constraint_decay_iters 5000 → 15000  (slower decay, constraint stays helpful longer)
#   deform_net_depth 0 → 1  (now actually adds 1 extra backbone layer; 0 and 1 were identical before fix)
#   strategy.reset-every 99999  *** MAJOR FIX ***
#     DefaultStrategy resets ALL opacities to sigmoid(-4.6)≈0.01 at step 0, 3K, 6K, ..., 27K.
#     PLY-initialized face Gaussians have sigmoid(opacity)≈0.8-0.98 — immediately destroyed at step 0.
#     10 resets × ~2K recovery steps each = ~20K steps wasted rebuilding opacity.
#     With PLY init + opacity_reg to handle floaters, resets are unnecessary and very harmful.
#
# Code fixes applied (trainer + deform_network):
#   - identity constraint uses canonical_means.detach() → constraint only teaches MLP, not positions
#   - main deform forward uses self.splats["means"].detach() → prevents HexPlane from creating
#     uneven gradient paths that cause ghost artifacts (some Gaussians deform, others stay)
#   - defor_depth=0 → 0 extra layers, defor_depth=1 → 1 extra layer (was off-by-one before)
set -e

export CUDA_VISIBLE_DEVICES=4

STATIC_PLY=/data/shared/elaheh/elly_static_v2/static.ply
DYNAMIC_PLY=/data/shared/elaheh/elly_static_v2/dynamic.ply
DATA_DIR=/data/shared/elaheh/4D_demo/outdoor/elly/undistorted
RESULT_DIR=/data/shared/elaheh/4D_demo/elly_4dgs_f150_static_dynamic_v2
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

python simple_trainer_static_dynamic.py \
    --data_dir "$DATA_DIR" \
    --result_dir "$RESULT_DIR" \
    --static_ply_path "$STATIC_PLY" \
    --dynamic_ply_path "$DYNAMIC_PLY" \
    --dataset_mode rig \
    --num_frames 150 \
    --frame_stride 1 \
    --frame_start 0 \
    --max_steps 50000 \
    --coarse_iters 0 \
    --init_type ply \
    --data_factor 4 \
    --sh_degree 2 \
    --use_deformation \
    --deform_grid_resolution 64 \
    --deform_time_resolution 300 \
    --deform_feature_dim 32 \
    --deform_multires 1 2 4 8 \
    --deform_net_width 128 \
    --deform_net_depth 1 \
    --deform_time_pe_bands 4 \
    --deform_act_xyz relu \
    --deform_act_rot relu \
    --deform_act_scale relu \
    --deform_act_sh sinerelu \
    --deform_lr 1.6e-4 \
    --grid_lr 1.6e-3 \
    --deform_lr_delay_mult 0.01 \
    --deform_lr_warmup_steps 1000 \
    --max_num_gaussians 1500000 \
    --ssim_lambda 0.15 \
    --opacity_reg 0.0001 \
    --scale_reg 0.005 \
    --plane_tv_weight 0.0001 \
    --time_smooth_weight 0.01 \
    --time_smooth_weight_final 0.001 \
    --time_smooth_order 2 \
    --l1_time_planes_weight 0.0001 \
    --weight_constraint_init 1.0 \
    --weight_constraint_after 0.0 \
    --weight_constraint_decay_iters 15000 \
    --progressive_time_warmup 10000 \
    --progressive_time_initial 0.1 \
    --progressive_time_forward  \
    --strategy.refine-stop-iter 30000 \
    --strategy.reset-every 99999 \
    --tb_every 100 \
    --tb_image_every 200 \
    --eval_steps 5000 10000 15000 20000 25000 30000 40000 50000 \
    --save_steps 20000 40000 50000 \
    "$@"

echo "Training complete. Results in $RESULT_DIR"
echo "Launch TensorBoard with: tensorboard --logdir $RESULT_DIR/tb"
