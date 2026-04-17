# Multi-Frame 4D Gaussian Splatting Pipeline

## Overview

Per-frame dynamic Gaussian splat reconstruction with a frozen static background. The pipeline: train one frame fully, separate static/dynamic, clean outliers, simplify to ~5k Gaussians, then fine-tune frame-to-frame with pure gradient descent.

## Pipeline

### Step 1: Train Frame 1 Fully

Train frame 1 from scratch using gsplat's `simple_trainer.py` with DefaultStrategy (split/duplicate/prune) for full convergence.

- **Script**: `examples/simple_trainer.py`
- **Steps**: ~39,000
- **Output**: Single PLY with all Gaussians (static + dynamic combined)
- **Example**: `frame1_full_train/ply/point_cloud_38999.ply`

### Step 2: Separate Static and Dynamic

Split the trained PLY into static background and dynamic foreground using WAFT's temporal deviation analysis.

**Script**: `WAFT/static_dynamic_split/run_static_dynamic.py`

```bash
python run_static_dynamic.py \
    --colmap-sparse <sparse/0> \
    --images-dir <images> \
    --splat-ply <frame1.ply> \
    --n-cameras 12 \
    --frame-stride 10 \
    --dilation-radius 3 \
    --output-dir <sep_dir>
```

**How it works**:
1. Compute cumulative L1 pixel deviation heatmaps per camera across frames
2. Otsu's method auto-thresholds each heatmap
3. Project 3D Gaussian means into heatmaps, score by mean-of-visible-cameras
4. Voxel grid labeling + dilation fills motion gaps
5. Connected component cleanup removes noise clusters

**Outputs**:
- `inside.ply` — dynamic Gaussians (moving subject)
- `outside.ply` — static Gaussians (background)
- `voxel_labels.npy` — (R,R,R) voxel classification
- `grid_bounds.npy` — ROI bounding box [min, max]

### Step 3: Clean Dynamic PLY

Three cleaning steps before simplification:

#### 3a. ROI Bounding Box Filter
Remove any Gaussian whose extent (mean +/- exp(scale)) falls outside the ROI bounding box. Even if part of the Gaussian sticks out, it gets removed.

```python
s = torch.exp(scales)
inside = ((means - s) >= grid_min).all(dim=1) & ((means + s) <= grid_max).all(dim=1)
```

#### 3b. Connected Component Cleanup
Remove small disconnected clusters (floaters/noise) using KNN graph + union-find.

```bash
python examples/clean_gaussians.py \
    --input inside_roi.ply \
    --output inside_roi_clean.ply \
    --min_cluster_frac 0.01 \
    --cc_k 20 \
    --cc_radius_mult 5.0
```

Clusters smaller than 1% of total points are removed. Uses `cc_radius_mult * median_NN_distance` as the connection radius.

#### 3c. Result
Typical reduction: 98k -> 78k (ROI) -> 64k (cleanup)

### Step 4: Simplify to ~5k Gaussians

Use NanoGS pairwise merging (KL-divergence based) to reduce Gaussian count.

```bash
# Dynamic: target 5k
RATIO=$(python -c "print(5000 / N_CLEANED)")
python examples/simplify_gaussians.py \
    --static_ply inside_roi_clean.ply \
    -o inside_final_5000.ply \
    -r $RATIO

# Static: same ratio for consistency
python examples/simplify_gaussians.py \
    --static_ply outside.ply \
    -o outside_simplified_0.05.ply \
    -r 0.05
```

### Step 5: Multi-Frame Fine-Tuning

**Script**: `examples/run_multiframe_fast.py`

#### Architecture
- **Static**: Loaded once, pre-rendered into per-view cache. Frozen throughout.
- **Dynamic**: Initialized from simplified PLY. Fine-tuned per frame via Adam. Passed frame-to-frame in memory.
- **Compositing**: `final = dynamic_rgb + static_cache * (1 - dynamic_alpha)`

#### Strategy
**No MCMC, no DefaultStrategy** — pure gradient descent on all frames. No densification, no relocation, no noise. Gaussian count stays approximately constant. This prevents floater accumulation from densification.

#### Training Schedule

| | Frame 1 | Frames 2-150 |
|---|---------|-------------|
| Steps | 5,000 | 750 |
| Strategy | None (pure Adam) | None (pure Adam) |

#### Optimizer (per-parameter learning rates)

| Parameter | LR |
|-----------|----|
| means | 1.6e-4 * scene_scale |
| scales | 5e-3 |
| quats | 1e-3 |
| opacities | 3e-2 |
| sh0 | 2.5e-3 |
| shN | 1.25e-4 |

Adam with eps=1e-15, betas scaled by batch size.

#### Loss

```
loss = L1(rendered, target)
     + needle_reg * mean(clamp(aspect_ratio - 10, min=0))
     + opacity_reg * mean(clamp(sigmoid(opacity) - 0.9, min=0))
```

- **L1**: Primary reconstruction loss
- **Needle reg** (weight=0.008): Hinge penalty on aspect ratio. **Zero penalty below 10**. Only penalizes extreme needles linearly above 10. This prevents needle formation without creating dot artifacts.
- **Bright opacity reg** (weight=0.01): Hinge penalty on opacity. **Zero penalty below 0.9**. Only penalizes super bright Gaussians (opacity > 0.9) that create bright spot artifacts. Moderate opacities are untouched.
- **No scale_reg**: Scale is free

**Why hinge regularization works**: Previous attempts used penalties on all Gaussians equally (e.g., `log1p(aspect)` for needles, `sigmoid(opa).mean()` for opacity). These push everything toward spheres/transparency, creating dot artifacts or killing useful Gaussians. The hinge approach (`clamp(x - threshold, 0)`) leaves moderate values completely untouched and only fights the extremes.

#### Post-Training Cleanup (at save time, each frame)

After training and drift correction, before saving:

1. **Bbox**: Remove dynamic Gaussians whose mean +/- scale extends outside ROI
2. **Low opacity**: Remove Gaussians with sigmoid(opacity) < 0.005
3. **Needles** (frame 2+): Frame 2 computes `threshold = max(top_1%_aspect_ratio, 100)`. This fixed threshold is reused for all subsequent frames.

#### Drift Correction
If a frame's loss exceeds 1.5x the baseline (frame 1 loss), run 1500 extra optimization steps to recover before saving.

### Run Command

```bash
SEP_DIR="<path_to_static_dynamic_output>"

python examples/run_multiframe_fast.py \
    --data_dir <undistorted_data> \
    --result_dir <output_results> \
    --static_ply_path "${SEP_DIR}/outside_simplified_0.05.ply" \
    --separation_dir "${SEP_DIR}" \
    --init_ply "${SEP_DIR}/inside_final_5000.ply" \
    --strategy mcmc \
    --frame_start 1 --frame_end 150 \
    --frame1_steps 5000 \
    --ftune_steps 750 \
    --opacity_reg 0.01 \
    --scale_reg 0.0 \
    --needle_reg 0.008 \
    --gpu 1
```

Note: `--strategy mcmc` is passed but all MCMC ops are disabled (refine_start_iter=999999, noise_lr=0). It's effectively no-strategy.

### Output

```
<result_dir>/
    frame_001/ply/point_cloud_4999.ply
    frame_002/ply/point_cloud_749.ply
    ...
    frame_150/ply/point_cloud_749.ply
    all_ply/    # Collected dynamic PLYs for easy access
```

Each PLY contains only dynamic Gaussians. Combine with the frozen static PLY for full scene rendering.

## Key Design Decisions

1. **No densification**: MCMC/DefaultStrategy add/remove Gaussians causing floater accumulation over frames. Pure fine-tuning keeps the set clean.

2. **Pre-cleaned initialization**: Clean once upfront (ROI + connected components + simplify) instead of pruning during training, which compounds frame-over-frame.

3. **Hinge needle reg**: `clamp(aspect - 20, 0)` not `log1p(aspect)`. Leaves moderate Gaussians alone, only fights extreme needles. No dot artifacts.

4. **Frozen static background**: Pre-rendered and cached. Only dynamic Gaussians train, reducing compute and preventing background drift.

5. **Low-res training**: Auto-factor reduces images to ~100px short side for fast iteration (~2-3s per frame after frame 1).

6. **Fixed needle threshold**: Computed once from frame 2 distribution, reused for all frames. Prevents compounding removal across frames.
