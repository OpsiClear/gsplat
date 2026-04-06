#!/usr/bin/env bash
# Ablation study: data_factor × cap_max on thenewface frame 1
# 4 data factors × 6 cap sizes = 24 experiments
# Runs 4 at a time across GPUs 1,2,3,4
# 40k steps total: Gaussians-only 0-30k, PPISP enabled 30k-40k
# Eval + PLY saved at step 30k (before PPISP) and 40k (after PPISP)
set -e

DATA_DIR=/data/shared/elaheh/4D_demo/new_data/thenewface/undistorted
BASE_RESULT=/data/shared/elaheh/4D_demo/thenewface_ablation
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$SCRIPT_DIR"

DATA_FACTORS=(4 8 10 15)
CAP_SIZES=(5000 10000 15000 20000 30000 50000)
GPUS=(1 2 3 4)

run_experiment() {
    local gpu=$1
    local df=$2
    local cap=$3
    local init_pts=$((cap / 2))
    local result_dir="${BASE_RESULT}/mcmc_df${df}_cap${cap}_init${init_pts}_refine300_noise1e4_ssim005_ppisp30k"

    echo "[GPU ${gpu}] Starting: df=${df} cap=${cap} init=${init_pts} → ${result_dir}"

    mkdir -p "${result_dir}"

    # Save experiment config
    cat > "${result_dir}/experiment_config.txt" <<CFGEOF
========================================
Experiment: MinSplat Ablation
========================================
Dataset:        thenewface (undistorted)
Frame:          1
Data factor:    ${df}
Image size:     ~$((3800/df))x$((2000/df)) px

Strategy:       MCMC
Cap max:        ${cap}
Init points:    ${init_pts} (subsampled from SFM)
Refine every:   300
Refine stop:    25000
Noise LR:       1e4
Min opacity:    0.001

Training:
  Max steps:    40000
  SH degree:    3
  SSIM lambda:  0.05
  Antialiased:  yes
  Opacity reg:  0.001
  Scale reg:    0.0001

PPISP:
  Enabled at:   step 30000
  Controller:   off
  Distillation: off

Eval/PLY at:    30000, 40000
GPU:            ${gpu}
========================================
CFGEOF

    CUDA_VISIBLE_DEVICES=${gpu} python simple_trainer_ftune.py mcmc \
        --data_dir "$DATA_DIR" \
        --result_dir "${result_dir}" \
        --data_factor ${df} \
        --init_num_pts ${init_pts} \
        --max_steps 40000 \
        --test_every 100000 \
        --sh_degree 3 \
        --ssim_lambda 0.05 \
        --antialiased \
        --post_processing ppisp \
        --ppisp_start_step 30000 \
        --no-ppisp_use_controller \
        --no-ppisp_controller_distillation \
        --strategy.cap-max ${cap} \
        --strategy.refine-every 300 \
        --strategy.refine-stop-iter 25000 \
        --strategy.refine-start-iter 500 \
        --strategy.noise-lr 1e4 \
        --strategy.min-opacity 0.001 \
        --strategy.verbose \
        --opacity_reg 0.001 \
        --scale_reg 0.0001 \
        --save_ply \
        --eval_steps 30000 40000 \
        --save_steps 30000 40000 \
        --ply_steps 30000 40000 \
        --tb_every 100 \
        --disable_viewer \
        --disable_video \
        --load_images_in_memory \
        --frame_num 1 \
        2>&1 | tee "${result_dir}/train.log"

    echo "[GPU ${gpu}] Done: df=${df} cap=${cap}"
}

# Build job queue
JOBS=()
for df in "${DATA_FACTORS[@]}"; do
    for cap in "${CAP_SIZES[@]}"; do
        JOBS+=("${df}:${cap}")
    done
done

echo "========================================================"
echo "  MinSplat Ablation Study — thenewface frame 1"
echo "  ${#JOBS[@]} experiments | GPUs: ${GPUS[*]}"
echo "  Data factors: ${DATA_FACTORS[*]}"
echo "  Cap sizes:    ${CAP_SIZES[*]}"
echo "  Results:      ${BASE_RESULT}/"
echo "========================================================"

# Run jobs in batches of 4 (one per GPU)
idx=0
while [ $idx -lt ${#JOBS[@]} ]; do
    PIDS=()
    batch_desc=""
    for gpu_i in "${!GPUS[@]}"; do
        job_idx=$((idx + gpu_i))
        if [ $job_idx -ge ${#JOBS[@]} ]; then
            break
        fi
        job="${JOBS[$job_idx]}"
        df="${job%%:*}"
        cap="${job##*:}"
        gpu="${GPUS[$gpu_i]}"
        batch_desc="${batch_desc} [GPU${gpu}:df${df}_cap${cap}]"

        run_experiment "$gpu" "$df" "$cap" &
        PIDS+=($!)
    done

    echo "--- Batch $((idx / ${#GPUS[@]} + 1)): ${batch_desc} ---"

    # Wait for this batch to finish
    for pid in "${PIDS[@]}"; do
        wait "$pid" || true
    done

    idx=$((idx + ${#GPUS[@]}))
    echo "========= Batch complete (${idx}/${#JOBS[@]}) ========="
done

echo ""
echo "========================================================"
echo "  All ${#JOBS[@]} experiments complete!"
echo "========================================================"
echo ""
echo "Results summary:"
printf "%-6s %-8s %-12s %-12s %-10s\n" "DF" "CAP" "PSNR@30k" "PSNR@40k" "#GS"
printf "%-6s %-8s %-12s %-12s %-10s\n" "----" "------" "----------" "----------" "--------"
for df in "${DATA_FACTORS[@]}"; do
    for cap in "${CAP_SIZES[@]}"; do
        init_pts=$((cap / 2))
        dir="${BASE_RESULT}/mcmc_df${df}_cap${cap}_init${init_pts}_refine300_noise1e4_ssim005_ppisp30k"
        psnr_30k="N/A"
        psnr_40k="N/A"
        num_gs="N/A"
        if [ -f "${dir}/stats/val_step29999.json" ]; then
            psnr_30k=$(python -c "import json; d=json.load(open('${dir}/stats/val_step29999.json')); print(f'{d[\"psnr\"]:.2f}')" 2>/dev/null || echo "N/A")
            num_gs=$(python -c "import json; d=json.load(open('${dir}/stats/val_step29999.json')); print(d.get('num_GS','?'))" 2>/dev/null || echo "N/A")
        fi
        if [ -f "${dir}/stats/val_step39999.json" ]; then
            psnr_40k=$(python -c "import json; d=json.load(open('${dir}/stats/val_step39999.json')); print(f'{d[\"psnr\"]:.2f}')" 2>/dev/null || echo "N/A")
        fi
        printf "%-6s %-8s %-12s %-12s %-10s\n" "${df}" "${cap}" "${psnr_30k}" "${psnr_40k}" "${num_gs}"
    done
done
