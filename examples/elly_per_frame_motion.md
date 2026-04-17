# Per-Frame Masked Gaussian Splatting — Implementation Log

## Overview

Per-frame training of 3D Gaussians using SAM2 binary masks to separate dynamic (object) from static (background). Each frame is fine-tuned from the previous frame's Gaussians, with masks guiding where to train.

**Dataset**: elly (64 cameras, 900 frames, 600 SAM2 masks)
**Script**: `examples/simple_trainer_perframe_masked.py`
**Parallel runner**: `examples/run_parallel_perframe.py`

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `simple_trainer_perframe_masked.py` | Per-frame trainer with mask loss, anti-halo, freeze, motion guidance |
| `run_parallel_perframe.py` | Coordinator: stride-3 keyframes, 3 GPUs parallel |
| `filter_dynamic_ply.py` | Post-process: re-filter PLYs at any mask threshold |
| `crop_sphere_ply.py` | Post-process: crop PLYs to sphere around scene center |
| `viewer_ply_sequence.py` | Viewer: scrub through PLY sequence with optional static overlay |

---

## Pipeline

### Frame 0: Train from SFM
```bash
python simple_trainer_perframe_masked.py \
    --ply_path '' --mode mask_only \
    --first_frame_steps 30000 --data_factor 4 \
    --densify_every 300 --densify_start 500 --densify_stop 15000 \
    --alpha_outside_weight 0.5 --sh_degree 3
```
- SFM init from `sparse/0/points3D.bin`
- Full 30K training with mask-only loss + anti-halo
- Produces: `frame0_sfm_maskonly/ply_frames/dynamic/000000.ply` (245K Gaussians)

### Frame 1+: Parallel fine-tuning
```bash
python run_parallel_perframe.py \
    --init_ply .../000000.ply \
    --gpus 1,2,3 --num_frames 600 \
    --steps_per_frame 7000 --data_factor 4 \
    --alpha_outside_weight 0.5 \
    --densify_every 300 --densify_start 500 --densify_stop 5000
```
- 3 GPUs parallel, stride 3 (keyframe every 3rd frame)
- Each frame: load keyframe PLY → fine-tune 7K steps → save
- Motion-guided loss + distance-weighted anti-halo

---

## Features Implemented

### 1. Mask-Only Loss
Loss computed ONLY on white (object) pixels in the SAM2 mask.
```python
obj_mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
l1 = (|rendered - gt| * weighted_mask).sum() / mask_sum / 3
```

### 2. Motion-Guided Loss
Compares current frame mask vs previous frame mask. Areas where the person moved get full training weight; static areas get reduced weight (less jitter).
```python
motion = |curr_mask - prev_mask|  # where person moved
weight_map = 0.1 + 0.9 * motion   # motion=1.0, static=0.1
loss = |rendered - gt| * weight_map * mask
```
In parallel mode: `--prev_frame_idx` passed via CLI, loads previous mask from disk.

### 3. Distance-Weighted Anti-Halo
Prevents Gaussians from rendering outside the mask. Gentle near mask edge (allows cloth to extend), harsh far away (kills head halo).
```python
# Within 10px of mask edge: NO penalty (cloth extends)
# Beyond 10px: FULL penalty (alpha + color → zero)
dilated = max_pool2d(mask, kernel=21, padding=10)
near_edge = dilated & ~mask   # 10px border zone
far_from_edge = outside_mask - near_edge

loss += weight * (0.0 * near_penalty + 1.0 * far_penalty)
```

### 4. Gaussian-Level Freeze (attempted, disabled)
Render all cameras, compute per-Gaussian error, freeze correct ones as plain tensors. Strategy only sees trainable Gaussians.

**Issues encountered:**
- `scatter_reduce_` CUDA assertion error with packed mode `gaussian_ids`
- Fixed by projecting Gaussian centers to 2D instead (no packed mode)
- But: strategy `_update_state` still crashes when merged (frozen+trainable) render produces IDs beyond trainable count
- Fixed by filtering `info["gaussian_ids"]` to only include trainable IDs before passing to strategy
- But: `frozen_data` not returned from `train_frame` correctly → NameError
- **Status**: Disabled. Using motion-guided loss instead.

### 5. Parallel Training (stride-3 keyframes)
```
Group 1: frame 0 (keyframe) → GPU1:frame1, GPU2:frame2, GPU3:frame3
Group 2: frame 3 (keyframe) → GPU1:frame4, GPU2:frame5, GPU3:frame6
...
```
- Last frame of each group becomes next keyframe
- Each subprocess is independent (no shared state)
- Skip-completed: existing PLYs are skipped

### 6. Image Preloading
All 64 camera images + masks preloaded to GPU at start of each frame. Eliminates per-step disk I/O.
```python
preloaded_images = [load_image(...) for ci in range(num_cameras)]
preloaded_masks = [load_mask(...) for ci in range(num_cameras)]
```

### 7. PLY Export
- Direct row filter from original PLY (preserves exact format/colors)
- Frame 0: saved as-is from SFM training
- Frame 1+: saved from `save_ply()` with `ref_ply_path` for format matching

---

## Bugs Found & Fixed

### Bug 1: Subprocess treats every frame as "first frame"
**Problem**: Parallel runner calls `--num_frames 1 --frame_start N`. Inside trainer, `fi=0` always, so `is_first_frame=True`. Freeze check, border discard, and step calculation all wrong.
**Fix**: Changed conditions from `fi == 0` to `frame_idx == 0` (absolute frame number).

### Bug 2: Border discard removes Gaussians at every frame
**Problem**: The `touches_mask` + `is_dynamic` classification ran on every frame, discarding 35K+ Gaussians each time. Cascading shrink: 245K → 80K.
**Fix**: Only run border discard on frame 0 (initial split). Skip for frame > 0.

### Bug 3: `post_split_steps=0` doesn't skip training
**Problem**: `post_split_steps=0` fell through to `steps_per_frame=7000`. Every subprocess ran 7K steps on frame 0 instead of skipping.
**Fix**: Added explicit `if frame_steps > 0` check before calling `train_frame()`.

### Bug 4: `frozen_data` not defined at save
**Problem**: `frozen_data` variable created inside `train_frame` but accessed in `main()`.
**Fix**: Return `frozen_data` from `train_frame` along with `is_dynamic` and `splats`.

### Bug 5: Strategy crashes with frozen+trainable merged render
**Problem**: Rendering merges frozen (IDs N_trainable to N_total) + trainable (IDs 0 to N_trainable). Strategy's `_update_state` uses `gaussian_ids` which include frozen IDs → out-of-bounds scatter.
**Fix**: Filter `info["gaussian_ids"]` to only include IDs < N_trainable before passing to strategy.

### Bug 6: `scatter_reduce_` CUDA assertion in freeze check
**Problem**: The render-based freeze check used packed mode `gaussian_ids` with `scatter_reduce_("amax")`. Some IDs exceeded tensor bounds.
**Fix**: Replaced with direct Gaussian center projection to 2D + error map sampling (no packed mode).

### Bug 7: `save_ply` changes color format
**Problem**: `save_ply()` reconstructs PLY from ParameterDict with different SH ordering → colors look wrong.
**Fix**: Use direct PLY row filter for untrained frames (preserves exact byte format). For trained frames, use `ref_ply_path` to match original dtype.

### Bug 8: Viewer SH ordering wrong
**Problem**: `viewer_ply_sequence.py` loaded SH rest as `[N, n_rest//3, 3]` but PLY stores SH transposed.
**Fix**: `sh_rest.reshape(-1, 3, n_rest//3).transpose(1, 2)` to match gsplat convention.

### Bug 9: Viewer OOM loading all frames
**Problem**: Loading 291 PLYs (up to 1.3M Gaussians each) to GPU at once → 39GB OOM.
**Fix**: Lazy loading — only current frame on GPU, load from disk on frame switch.

### Bug 10: Motion loss not working in parallel mode
**Problem**: `train_frame._prev_masks` stored per-process but parallel subprocesses are independent.
**Fix**: Pass `--prev_frame_idx` via CLI, load previous frame's masks from disk directly.

---

## Anti-Halo Weight Experiments

| Weight | Result |
|--------|--------|
| 0.1 | Halo visible around head and body |
| 0.5 | Good balance — some cloth visible at edge |
| 0.8 | Slightly harsh on cloth |
| 1.0 | Cloth starts losing color |
| 2.0 | Black patches on cloth, too aggressive |

**Final**: 0.5 with distance-weighted (0 penalty within 10px of edge, full penalty beyond).

---

## Gaussian Count Tracking

### Without pruning protection (prune_opa=0.005 default)
```
Frame 0:  245K
Frame 10: 129K  (-47%)
Frame 30:  91K  (-63%)
Frame 60:  80K  (-67%)
```
**Cause**: Strategy prunes low-opacity Gaussians. Anti-halo pushes edge opacity down → strategy kills them.

### With prune_opa=0.0 (pruning disabled)
```
Frame 0:  245K
Frame 15: 245K  (stable)
```
**Result**: Count stays constant. Only densification (splitting) can change count.

---

## Data Paths

### Elly dataset
```
/data/shared/elaheh/4D_demo/outdoor/elly/undistorted/
  images/<cam>/000000.jpg ... 000899.jpg       (64 cameras × 900 frames)
  tracking_experiment/<cam>/sam2/000000.png ... (64 cameras × 600 masks)
  sparse/0/                                     (COLMAP)
```

### Results
```
/data/shared/elaheh/4D_demo/outdoor/elly/frame0_sfm_maskonly/  (frame 0 training)
/data/shared/elaheh/4D_demo/outdoor/elly/exp_parallel/         (parallel per-frame)
  ply_frames/dynamic/000000.ply ... 000599.ply
  train.log
  gpu*_frame*.log
```

### Viewer
```bash
python examples/viewer_ply_sequence.py \
    --ply-dir .../exp_parallel/ply_frames/dynamic/ \
    --start-frame 0 --end-frame 50 --port 8080
```

---

## Current Best Config

```bash
python examples/run_parallel_perframe.py \
    --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \
    --init_ply .../frame0_sfm_maskonly/ply_frames/dynamic/000000.ply \
    --result_dir .../exp_parallel \
    --gpus 1,2,3 \
    --num_frames 600 --frame_start 0 \
    --steps_per_frame 7000 \
    --data_factor 4 --sh_degree 3 \
    --alpha_outside_weight 0.5 \
    --batch_size 8 \
    --densify_every 300 --densify_start 500 --densify_stop 5000
```

**Features active:**
- Motion-guided loss (weight 1.0 where mask changed, 0.1 where static)
- Distance-weighted anti-halo (0 within 10px of edge, 0.5 beyond)
- Strategy ON (densify every 300, stop 5K, prune_opa=0.0)
- 3 GPUs parallel, stride 3 keyframes
- Per-frame PLY export
