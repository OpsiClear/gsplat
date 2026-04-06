# Inside/Outside Promote-to-Static: Deformation Zeroing for 4DGS

## Overview

In the static+dynamic 4DGS pipeline, dynamic Gaussians (from `inside.ply`) are trained with a HexPlane deformation field to learn motion across frames. However, some "dynamic" Gaussians don't actually move — they waste HexPlane queries and accumulate deformation noise.

**Solution:** Periodically evaluate deformation magnitude across all channels (position, rotation, scale, opacity, color). Gaussians with negligible deformation are marked **time-independent** — they remain trainable (like standard 3DGS) but skip the HexPlane entirely. Zero deltas mean gradient flows directly to canonical parameters.

This is **not freezing**. Time-independent Gaussians keep full optimizer state, receive gradients from every frame's reconstruction loss, and participate in densification. They just don't go through the deformation network.

## Architecture

```
Static (outside.ply)     — frozen, never trained, always rendered
Time-independent         — trainable, no deformation (promoted from dynamic)
Deformable (inside.ply)  — trainable, HexPlane deformation per frame
```

At render time:
- Deformable Gaussians: query HexPlane -> get deltas -> apply to canonical params
- Time-independent Gaussians: zero deltas -> canonical params used directly
- Static Gaussians: pre-activated tensors concatenated after dynamic set

## Files Modified

### `examples/simple_trainer_static_dynamic.py`

**Config fields** added to `Config`:
```python
promotion_steps: List[int]    # explicit steps to run promotion (e.g. [25000])
promotion_every: int = 0      # run every N steps (alternative to promotion_steps)
promotion_start: int = 25500  # don't run before this step
promotion_num_time_samples: int = 20  # uniform time samples over [-0.5, 0.5]
promotion_xyz_threshold: float = 0.0  # > 0: fixed threshold; <= 0: adaptive percentile
promotion_percentile: float = 2.0     # bottom N% marked time-independent (adaptive mode)
```

**Runner.__init__** additions:
- `self.deform_mask: Optional[torch.Tensor]` — bool mask `[N_dynamic]`, `True` = deformable
- `self._locked_promotion_threshold: Optional[float]` — threshold locked after first adaptive run

**Forward pass** (`rasterize_splats`):
- When `deform_mask` exists and has `False` entries, only queries HexPlane for deformable Gaussians
- Scatters zero deltas for time-independent Gaussians via `DeformOutput`
- Falls back to original path when all Gaussians are deformable

**New methods:**
- `_evaluate_deformation_scores()` — evaluates ALL deformation channels (xyz, rot, scale, opacity, SH) across uniform time samples. Z-score normalizes each channel, returns `max(z_scores)` per Gaussian.
- `_zero_out_static_deformations(step)` — marks near-static Gaussians. On first adaptive call, computes percentile threshold and locks it. Subsequent calls reuse the locked threshold.

**Training loop integration:**
- Promotion triggers at `promotion_start`, then every `promotion_every` steps, plus always at last step
- `deform_mask` stored in `strategy_state` so `ops.py` keeps it in sync during densification (remove/duplicate/split)
- Gaussian count cap also filters `deform_mask`

**Checkpoint save/load:**
- Saves `deform_mask` and `locked_promotion_threshold` in checkpoint
- Restores both on load, syncs `deform_mask` into `strategy_state`

### `examples/viewer_4dgs.py`

- `load_checkpoint()` now returns `deform_mask` from checkpoint
- `_deform_at_time(t)` helper applies masked deformation (zero deltas for time-independent)
- `_precompute_all_frames()`, `_export_per_frame_plys()`, `_get_deformed()` all use the masked path
- Stats display shows deformable vs time-independent counts

### `examples/run_thenewface_static_dynamic.sh`

- PLY paths updated to `outside.ply` / `inside.ply` from `static_dynamic_output/`
- Added promotion flags: `--promotion_every 5000 --promotion_start 25500 --promotion_percentile 2.0`

## Threshold Logic

### Adaptive Mode (default, `promotion_xyz_threshold <= 0`)

1. **First promotion step** (e.g. 25500): evaluate combined deformation score for all Gaussians, compute the threshold at the bottom `promotion_percentile`% (default 2%), **lock the threshold**.
2. **Subsequent steps** (30500, 35500, ...): re-evaluate scores using the **same locked threshold**. As training converges, more Gaussians may fall below the threshold and get promoted — but the bar doesn't keep moving down.
3. **Last step**: always runs promotion so the final checkpoint has up-to-date `deform_mask`.

### Fixed Mode (`promotion_xyz_threshold > 0`)

Uses the given value as a fixed threshold on the combined z-score every time.

## Combined Deformation Score

Each channel is evaluated independently across `promotion_num_time_samples` uniform time samples in `[-0.5, 0.5]`:

| Channel | What's measured | Space |
|---------|----------------|-------|
| xyz | `max_t \|\|delta_xyz * aabb_half\|\|` | world-space meters |
| rot | `max_t \|\|delta_rot\|\|` | quaternion delta norm |
| scale | `max_t \|\|delta_scale\|\|` | log-space norm |
| opacity | `max_t \|delta_opacity\|` | logit-space abs |
| sh | `max_t \|\|delta_sh\|\|` | SH coefficient norm |

Each channel is z-score normalized: `z = (x - mean) / std`. Combined score = `max(z_xyz, z_rot, z_scale, z_opacity, z_sh)`. A Gaussian is time-independent only if **all** channels show low deformation.

## Densification Sync

`deform_mask` is stored in `self.strategy_state["deform_mask"]`. The ops.py functions (`remove()`, `duplicate()`, `split()`) iterate all tensors in `state` and apply the same indexing:

- **Prune** (`remove`): filters by keep mask — pruned Gaussians removed from `deform_mask`
- **Clone** (`duplicate`): appends copy of source entries — cloned Gaussian inherits deform status
- **Split** (`split`): repeats source entries — split Gaussians inherit deform status
- **Gaussian cap**: manually filters `deform_mask` by `keep_mask`

After each densification step, `self.deform_mask` is synced back from `strategy_state`.

## Running Experiments

### thenewface (300 frames)

```bash
# tmux session: thenewface_promo_v2, GPU 2
CUDA_VISIBLE_DEVICES=2 python simple_trainer_static_dynamic.py \
    --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted \
    --result_dir /data/shared/elaheh/4D_demo/thenewface_4dgs_f300_static_dynamic_promotion_v2 \
    --static_ply_path .../static_dynamic_output/outside.ply \
    --dynamic_ply_path .../static_dynamic_output/inside.ply \
    --dataset_mode rig --num_frames 300 --frame_start 1 \
    --max_steps 50000 --coarse_iters 0 --init_type ply \
    --deform_time_resolution 600 \
    --promotion_every 5000 --promotion_start 25500 --promotion_percentile 2.0 \
    ...
```

### elly (150 frames)

```bash
# tmux session: elly_promo_v2, GPU 1
CUDA_VISIBLE_DEVICES=1 python simple_trainer_static_dynamic.py \
    --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \
    --result_dir /data/shared/elaheh/4D_demo/elly_4dgs_f150_static_dynamic_promotion_v2 \
    --static_ply_path .../static_dynamic_output/outside.ply \
    --dynamic_ply_path .../static_dynamic_output/inside.ply \
    --dataset_mode rig --num_frames 150 --frame_start 0 \
    --max_steps 50000 --coarse_iters 0 --init_type ply \
    --deform_time_resolution 300 \
    --promotion_every 5000 --promotion_start 25500 --promotion_percentile 2.0 \
    ...
```

### Viewing Results

```bash
python viewer_4dgs.py \
    --ckpt /path/to/ckpts/ckpt_49999_rank0.pt \
    --gpu 0 --port 8080
```

The viewer auto-loads `static_ply_path` from checkpoint config and respects `deform_mask`.

## Verification Checklist

1. Console prints "N Gaussians marked time-independent" at promotion steps
2. `deform_mask` in checkpoint: `ckpt["deform_mask"].sum()` < total dynamic count
3. TensorBoard: `promotion/combined_score` histogram, per-channel histograms
4. Locked threshold reused on subsequent promotion steps (check log)
5. Viewer stats panel shows "deformable" vs "time-indep" counts
6. PSNR comparable or better vs baseline (no promotion)

## First Experiment Results (fixed threshold 0.005, step 30k only)

- thenewface: 968 / 154,766 marked time-independent (0.6%) — threshold too strict
- Motivated switch to adaptive percentile-based approach
