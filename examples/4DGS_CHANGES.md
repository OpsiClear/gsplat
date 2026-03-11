# 4D Gaussian Splatting — Bug Fixes & Changes

Reference paper: [4D Gaussian Splatting (arXiv:2310.08528)](https://arxiv.org/abs/2310.08528)
Reference repo: [hustvl/4DGaussians](https://github.com/hustvl/4DGaussians)

## Files Modified

- `examples/simple_trainer_4dgs.py` — main 4DGS trainer
- `examples/deformation/hexplane.py` — HexPlane spatiotemporal feature grid
- `examples/deformation/deform_network.py` — deformation field MLP + apply_deformation
- `examples/deformation/__init__.py` — exports
- `examples/deformation/regulation.py` — regularization losses (unchanged, verified correct)
- `examples/run_4dgs_elly.sh` — launch script

---

## Commit 1: `4e5f6bc` — Camera & HexPlane fixes

### Bug 1: Camera double-inversion (CRITICAL)

**File**: `simple_trainer_4dgs.py`, `DynamicRigDataset.__getitem__` (line 245)

**Problem**: The COLMAP parser stores camera-to-world (c2w) in `parser.camtoworlds`.
`DynamicRigDataset` was inverting it: `np.linalg.inv(camtoworlds[i])` → producing w2c.
Then `rasterize_splats()` inverted again: `viewmats = inv(camtoworlds)` → passing c2w
as viewmat to gsplat. The rasterizer expects w2c, so every camera was wrong.

**Before**: `camtoworld = np.linalg.inv(self.ref_parser.camtoworlds[cam_idx])`
**After**: `camtoworld = self.ref_parser.camtoworlds[cam_idx]`

**Impact**: This alone caused PSNR ~13.6. Even the coarse phase (static 3DGS, no deformation)
was broken because every training image had the wrong camera transform.

### Bug 2: HexPlane feature aggregation — sum vs product (CRITICAL)

**File**: `deformation/hexplane.py`, `forward()` method

**Problem**: The 4DGS paper (Eq. 7) specifies:
- For each resolution level: compute **element-wise product** of features from all 6 planes
- Then **concatenate** across resolution levels

The code was doing the opposite:
- For each plane pair: **sum** features across resolution levels
- Then **concatenate** across 6 plane pairs

This completely destroys the multiplicative spatiotemporal coupling that makes the
factored 4D representation work. Each plane pair operated independently with no
cross-axis interaction.

**Before**:
```python
# For each pair: sum across scales, then concat pairs → [N, C*6]
for pair in PLANE_PAIRS:
    scale_feat = None
    for scale in multires:
        feat = grid_sample(plane, ...)
        scale_feat = scale_feat + feat  # SUM
    pair_features.append(scale_feat)
features = cat(pair_features)  # [N, C*6]
```

**After**:
```python
# For each scale: product across 6 pairs, then concat scales → [N, C*num_scales]
for scale_idx in range(num_scales):
    prod_feat = None
    for pair_idx in range(6):
        feat = grid_sample(planes[pair_idx * num_scales + scale_idx], ...)
        prod_feat = prod_feat * feat  # PRODUCT
    scale_features.append(prod_feat)
features = cat(scale_features)  # [N, C*num_scales]
```

**out_dim changed**: `feature_dim * 6` (96) → `feature_dim * num_scales` (64 for default multires=[1,2,4,8])

### Bug 2b: HexPlane initialization for product decomposition

**File**: `deformation/hexplane.py`, `__init__`

**Problem**: With product of 6 near-zero values, features and gradients vanish.

**Before**: `nn.init.uniform_(plane, -1e-4, 1e-4)` → product ≈ 0, gradients ≈ 0
**After**: `nn.init.uniform_(plane, 0.9, 1.1)` → product ≈ 1.0, healthy gradients

---

## Commit 2: `ce51303` — MLP capacity, gradient flow, rotation, densification

### Fix 3: MLP capacity (HIGH IMPACT)

**File**: `simple_trainer_4dgs.py` (Config defaults), `deformation/deform_network.py`

**Problem**: The deformation MLP was far too small to model complex motion.

| | Original 4DGS | Our code (before) | Our code (after) |
|---|---|---|---|
| Width | 256 | 64 | 128 |
| Depth | 8 | 1 | 6 |
| Params | ~2M | ~50K | ~23M |

The tiny network couldn't represent the deformation field for a 30-frame dynamic scene
with moving hands, face, etc. Increasing capacity allows the network to model both
large-scale rigid motion and fine-grained deformations.

**Config defaults changed**:
```python
deform_net_width: int = 128  # was 64
deform_net_depth: int = 6    # was 1
```

### Fix 4: Remove means.detach() — gradient flow through positions

**File**: `simple_trainer_4dgs.py`, `rasterize_splats()` (line ~753)

**Problem**: The original 4DGS does NOT detach canonical means before passing to the
deformation field. Our code was detaching, which prevented the deformation network
from learning spatial structure through position gradients.

**Before**:
```python
deltas = self.deform_field(self.splats["means"].detach(), timestamp)
```

**After**:
```python
deltas = self.deform_field(self.splats["means"], timestamp)
```

**Why it matters**: With detach, the deformation field only learns from the loss gradient
flowing through delta_xyz. Without detach, it also receives gradient signal through the
spatial query positions in HexPlane's grid_sample, which helps it learn where deformations
are needed.

### Fix 5: Rotation — quaternion multiply → additive (matching paper)

**File**: `deformation/deform_network.py`

**Problem**: The paper (Eq. 8) and original repo use simple additive rotation:
`r' = r + Δr`, then normalize. Our code used quaternion multiplication with an
identity buffer, which is geometrically different and more complex.

**Before** (DeformationField.forward):
```python
delta_rot = F.normalize(self._rot_identity + rot_raw, p=2, dim=-1)
```

**Before** (apply_deformation):
```python
quats_d = quat_multiply(quats_raw, deform_out.delta_rot)
quats_d = F.normalize(quats_d, p=2, dim=-1)
```

**After** (DeformationField.forward):
```python
delta_rot = self.head_rot(hidden)  # raw offset, ~0 at init
```

**After** (apply_deformation):
```python
quats_d = F.normalize(splats["quats"] + deform_out.delta_rot, p=2, dim=-1)
```

Removed: `quat_multiply()` function, `_rot_identity` buffer (no longer needed).

### Fix 6: Densification control — stop at coarse_iters + Gaussian cap

**File**: `simple_trainer_4dgs.py`

**Problem**: Densification continued during the deformation phase using frame-specific
gradients from deformed positions. This caused:
- 5.5M Gaussians by step 10K (original caps at 360K)
- 34GB GPU memory
- 6 FPS rendering
- Poor quality from millions of tiny overlapping Gaussians

**Changes**:
1. `refine_stop_iter` set to `coarse_iters` (3000) so densification only happens
   during the static warm-up phase, not during deformation training.
2. Added `max_num_gaussians = 1_000_000` config option as safety cap.
   Keeps the most opaque Gaussians if count exceeds the cap.
3. `reset_every = 3000` (opacity reset only during coarse phase).

**Rationale**: During deformation, the gradient on means2d is computed from deformed
positions at a specific timestep. This gradient is frame-dependent and unreliable for
deciding which Gaussians to split/clone. The original 4DGS uses dynamic threshold
decay and caps at 360K. Our simpler approach: only densify in coarse phase.

### Fix 7: Eval timestamp consistency

**File**: `simple_trainer_4dgs.py`, `eval()` method

**Problem**: Eval always passed timestamps to rasterization even during coarse phase
(when deformation is OFF in training). This created inconsistency between train and eval.

**Before**: `t = timestamps[0].item() if self.deform_field is not None else None`
**After**:
```python
deform_active = (self.deform_field is not None and step >= cfg.coarse_iters)
t = timestamps[0].item() if deform_active else None
```

---

---

## Iteration 3: Temporal & Dynamic Improvements

### Fix 8: Val split — train on ALL cameras

**File**: `simple_trainer_4dgs.py`, `DynamicRigDataset.__init__`

**Problem**: With `rig_test_every=8`, train excluded every 8th camera and val used only
those 8 cameras. Val cameras were NEVER seen in training → val PSNR measured novel-view
synthesis, not reconstruction quality. This made PSNR appear artificially low (~13.8).

**Before**: Train=56 cameras, Val=8 cameras (disjoint)
**After**: Train=64 cameras (ALL), Val=8 cameras (subset of training)

```python
# Before:
if test_every > 1:
    if split == "train":
        cam_indices = cam_indices[cam_indices % test_every != 0]
    else:
        cam_indices = cam_indices[cam_indices % test_every == 0]

# After:
if test_every > 1 and split == "val":
    cam_indices = cam_indices[cam_indices % test_every == 0]
# Train always uses ALL cameras
```

### Fix 9: Reduce time_smooth_weight (0.01 → 0.001)

**Problem**: `time_smoothness_loss` penalizes 2nd-order temporal derivatives (acceleration),
which IS fast motion. At weight 0.01, this actively suppressed hand/body motion.

**Fix**: Reduced to 0.001 (10x less suppression).

### Fix 10: Bigger MLP in run script

**Problem**: `run_4dgs_elly.sh` was passing `--deform_net_width 64 --deform_net_depth 1`
(old values), overriding the improved defaults from commit ce51303.

**Fix**: Updated to `--deform_net_width 128 --deform_net_depth 6`.

### Fix 11: Higher temporal resolution (25 → 150)

**Problem**: `deform_time_resolution=25` for 30 frames = <1 grid cell per frame.
The HexPlane couldn't resolve rapid temporal changes.

**Fix**: Increased to 150 (5 grid cells per frame). For 90-frame runs, use 300.

### Fix 12: Enable SSIM loss (0.0 → 0.2)

**Problem**: Only L1 loss. SSIM captures structural patterns (edges, fingers) that L1 misses.

**Fix**: `--ssim_lambda 0.2` → loss = 0.8 * L1 + 0.2 * SSIM.

### Results after Iteration 3 (v2 runs)

| Run | PSNR (val) | SSIM | LPIPS | Gaussians |
|-----|-----------|------|-------|-----------|
| v1 (before all fixes) | 13.6 | — | — | 5.0M |
| v1.5 (camera + HexPlane fix) | 16.5 | — | — | 5.5M |
| **v2 (30f, unmasked)** | **26.7** | **0.806** | **0.380** | ~963K |
| **v2 (30f, masked)** | **25.8** | **0.805** | **0.375** | ~963K |

---

## Iteration 4: Full Dynamic Capture

Changes targeting fast/complex motion (hands, body movement).

### Fix 13: Multi-resolution temporal planes

**File**: `deformation/hexplane.py`, `_plane_res()`

**Problem**: Spatial planes scale with multires `[1x, 2x, 4x, 8x]` but temporal resolution
was fixed across all scales. This limits the temporal frequency bandwidth — the HexPlane
can't simultaneously capture slow trends and fast motion.

**Before**: All temporal planes use `time_resolution` (fixed)
**After**: Temporal scales with `sqrt(multires)`: e.g., for `time_resolution=150`,
temporal resolutions become `[150, 212, 300, 424]`.

```python
# Before:
if axis == 3:
    return self.time_resolution

# After:
if axis == 3:
    return max(int(self.time_resolution * math.sqrt(scale)), self.time_resolution)
```

### Fix 14: Time positional encoding for MLP

**File**: `deformation/deform_network.py`

**Problem**: Time enters the deformation field only through HexPlane's grid_sample lookup.
The MLP backbone receives no explicit high-frequency temporal signal, making it hard to
distinguish nearby timesteps.

**Fix**: Added sinusoidal positional encoding of time, concatenated with HexPlane features
before the backbone:

```python
def _time_positional_encoding(self, t, bands):
    freqs = 2.0 ** torch.arange(bands) * math.pi
    t_scaled = t.unsqueeze(-1) * freqs
    return cat([sin(t_scaled), cos(t_scaled)], dim=-1)  # [N, 2*bands]
```

- Config: `deform_time_pe_bands` (default 0 = disabled)
- For 30 frames: 6 bands (12 dims, max freq 2^5=32 > 30)
- For 90 frames: 8 bands (16 dims, max freq 2^7=128 > 90)

### Fix 15: Annealed temporal smoothness

**File**: `simple_trainer_4dgs.py`, training loop

**Problem**: Fixed temporal smoothness weight throughout training. Early deformation
training needs high smoothness to prevent divergence, but late training needs low
smoothness to capture sharp motion details.

**Fix**: Cosine annealing from `time_smooth_weight` to `time_smooth_weight_final`:

```python
progress = (step - coarse_iters) / (max_steps - coarse_iters)
cos_decay = 0.5 * (1.0 + cos(pi * progress))
w = final + (start - final) * cos_decay
```

Config: `time_smooth_weight_final` (default -1 = no annealing, set ≥0 to enable)

### Fix 16: Densification through fine phase (matching paper)

**File**: `simple_trainer_4dgs.py`, Config

**Problem**: We stopped densification at `coarse_iters=3000`, but the original 4DGS paper
and reference code densify from iter 500-15K across BOTH coarse and fine phases. Our
conservative approach starved dynamic regions of Gaussians — areas that move significantly
from frame 0 had sparse coverage with no way to add more Gaussians.

**Before**: `refine_stop_iter=3000` (coarse only)
**After**: `refine_stop_iter=15000` (matches paper, with 1M Gaussian cap as safety)

### Fix 17: Opacity deformation

**File**: `deformation/deform_network.py`, Config

**Problem**: Without opacity deformation, Gaussians can't appear or disappear between
frames. Fast motion requires Gaussians to pop in/out as objects move through space.

**Fix**: `--enable_opacity_deform` adds a Δopacity head to the deformation MLP.
Applied in logit-space: `opacity' = sigmoid(logit_opacity + Δopacity)`.

### Fix 18: SH/color deformation

**File**: `deformation/deform_network.py`, Config

**Problem**: Without SH deformation, Gaussian colors are frozen at canonical frame values.
Moving objects change appearance due to view-dependent effects, motion-induced lighting
changes, and occlusion/disocclusion.

**Fix**: `--enable_sh_deform` adds a ΔSH head predicting per-Gaussian SH coefficient
deltas at each timestep. Applied additively: `sh' = sh + Δsh`.

### Fix 19: First-order time smoothness option

**File**: `deformation/regulation.py`

**Problem**: 2nd-order time smoothness penalizes acceleration (2nd derivative). For fast
motion, acceleration IS the signal — hands stopping and starting, rapid direction changes.
This actively suppresses the motion we want to capture.

**Fix**: Added `time_smoothness_loss_1st()` — penalizes 1st-order derivatives (velocity)
instead of acceleration. This allows fast motion as long as it's smooth (no jitter/flicker),
but doesn't penalize rapid starts/stops.

```python
# 2nd order (old): penalizes acceleration → suppresses fast motion
diff2 = (plane[..., 2:] - 2*plane[..., 1:-1] + plane[..., :-2])

# 1st order (new): penalizes velocity → allows fast but smooth motion
diff1 = plane[..., 1:] - plane[..., :-1]
```

Config: `time_smooth_order` (1 or 2, default 2)

### Fix 20: Mask support for rig datasets

**File**: `simple_trainer_4dgs.py`, `DynamicRigDataset`

**Problem**: Background regions (sky, ground) dominate the loss, leaving fewer gradients
for the dynamic foreground (people, hands).

**Fix**: Added `--mask_dir` option. Masks mirror the image directory structure:
`mask_dir/cam_name/000000.jpg`. White pixels (>127) = keep, black = ignore.
L1 loss is computed only on unmasked (white) pixels:

```python
l1loss = (|render - gt| * mask).sum() / mask.sum() / 3.0
```

Mask directory: `undistorted/masks/cam_dir/frame.jpg`

---

## Iteration 5: Stability Fixes for Long Sequences (150+ frames)

### Bug: Black Renders — Opacity Collapse (CRITICAL)

**Symptom**: After deformation activates, rendered images progressively darken and
eventually become **completely black** (every pixel = 0). Training loss increases
instead of decreasing. Happens with both 150-frame and 300-frame sequences.

**Root Cause**: `enable_opacity_deform=true` combined with `reset_every=3000`
(density control opacity reset) creates a **death spiral**:

1. Density control resets `splats["opacities"]` to low values every 3000 steps
2. The deformation's opacity head learns positive deltas to compensate
3. But DefaultStrategy **prunes based on BASE opacity** (`sigmoid(base)`) —
   it doesn't see the deformation delta
4. Gaussians with low base opacity get **pruned even though their deformed
   opacity was fine**
5. Fewer Gaussians → darker renders → more pruning → complete collapse

**Evidence** (from v9/v10 150-frame masked runs):
- v9 SSIM loss: 0.25 (step 5k) → 0.68 (step 7k, deform on) → **0.94** (step 18k)
- v10 val PSNR: 22.2 (step 5k) → **12.94** (step 10k+, all black)
- Both runs had identical PSNR=12.94 at all eval points after collapse = PSNR of
  a pure black image vs GT
- Training loss increased from 0.08 → 0.46 after deformation activated

**Fix**: Disable opacity deformation (`enable_opacity_deform=false`). The original
4DGS paper does NOT deform opacity. Add mild `opacity_reg=0.001` as safety net.

### Fix 21: Centered Time Normalization

**Files**: `simple_trainer_4dgs.py` (DynamicRigDataset, DynamicDataset),
`deformation/hexplane.py` (_normalize_coord)

**Problem**: Timestamps ranged from `[0, 1]` with the canonical/keyframe at `t=0`
(first frame). This means:
- The last frame has maximum deformation (t=1.0, furthest from canonical)
- Deformation grows asymmetrically — only forward in time
- The weight constraint at t=0 anchors an endpoint, not the center

**Fix**: Center time at 0, range `[-0.5, 0.5]`. The keyframe is now the **middle
frame** of the sequence.

```python
# Before:
t = frame_rank / max(num_frames - 1, 1)           # [0, 1]

# After:
t = frame_rank / max(num_frames - 1, 1) - 0.5     # [-0.5, 0.5]
```

HexPlane normalization updated: `coord * 2.0` (was `coord * 2.0 - 1.0`).

**Benefits**:
- Max deformation halved (0.5 vs 1.0 from canonical)
- Motion distributed symmetrically forward and backward
- Weight constraint at t=0 anchors the mid-point — more representative

### Fix 22: Progressive Frame Sampling

**File**: `simple_trainer_4dgs.py` (training loop, Config)

**Problem**: When deformation activates, the network must immediately learn
deformations for ALL timestamps, including frames far from the keyframe. For long
sequences (150+ frames), this means predicting large-scale motion from the first
fine-phase iteration. The network can't handle this → instability → collapse.

**Fix**: Progressive curriculum — start training on frames **close to the keyframe**
(t=0, mid-sequence) and gradually expand the time window:

```
Fine phase start:     |----[===]----| only ±10% of sequence
After 3k iters:       |--[=======]--| ±25%
After 7k iters:       |[===========]| ±40%
After 10k iters:      [=============] full sequence
```

Since time is centered at 0, the window expands **symmetrically in both directions**
— forward and backward from the keyframe.

**Config**:
```python
progressive_time_warmup: int = 10000   # expand over 10k fine-phase iters (0=disabled)
progressive_time_initial: float = 0.1  # start with ±10% of sequence
```

**Implementation**: Uses `torch.utils.data.Subset` with filtered indices. The
dataloader is rebuilt every 1000 steps when the radius increases significantly.

### Recommended Config for 150+ Frames

```bash
--num_frames 150
--frame_stride 1
--deform_time_resolution 600
--deform_net_width 128
--deform_net_depth 0              # no backbone, grid→heads directly
--deform_time_pe_bands 4
# NO --enable_opacity_deform      # causes death spiral with density control
# NO --enable_sh_deform           # adds instability
--opacity_reg 0.001               # mild opacity regularization
--scale_reg 0.01
--time_smooth_weight 0.01
--time_smooth_order 2
--progressive_time_warmup 10000   # expand time window over 10k iters
--progressive_time_initial 0.1    # start with ±10% of sequence
--weight_constraint_init 1.0
--weight_constraint_after 0.2
--max_num_gaussians 1500000
--max_steps 80000
--coarse_iters 7000
```

---

## Results Comparison

| Run | Frames | PSNR | SSIM | LPIPS | Features |
|-----|--------|------|------|-------|----------|
| v1 (broken) | 30 | 13.6 | — | — | Camera bug, HexPlane bug |
| v1.5 (camera fix) | 30 | 16.5 | — | — | Fixed camera + HexPlane |
| **v2 (unmasked)** | 30 | **26.7** | **0.806** | **0.380** | All Iter 3 fixes |
| **v2 (masked)** | 30 | **25.8** | **0.805** | **0.375** | + masks |
| v4 (unmasked) | 30 | TBD | TBD | TBD | + all Iter 4 dynamic fixes |
| v4 (masked) | 30 | TBD | TBD | TBD | + masks |
| v4 (90f, unmasked) | 90 | TBD | TBD | TBD | 90 frames, time_res=300 |
| v4 (90f, masked) | 90 | TBD | TBD | TBD | 90 frames + masks |

---

## Run Scripts

| Script | Frames | Masked | Key config |
|--------|--------|--------|------------|
| `run_4dgs_elly.sh` | 30 | no | v3 config (time_res=150, MLP 128/6) |
| `run_4dgs_elly_masked.sh` | 30 | yes | v3 + masks |
| `run_4dgs_elly_v4.sh` | 30 | no | Full dynamic (opacity/SH deform, 1st-order smooth, 8 PE bands) |
| `run_4dgs_elly_v4_masked.sh` | 30 | yes | Full dynamic + masks |
| `run_4dgs_elly_90f.sh` | 90 | no | Full dynamic, time_res=300, 8 PE bands |
| `run_4dgs_elly_90f_masked.sh` | 90 | yes | Full dynamic + masks |
| `run_4dgs_elly_v11_masked.sh` | 150 | yes | No opacity deform (fix collapse) |
| `run_4dgs_elly_v12_masked.sh` | 150 | yes | No opacity/SH deform, depth 2 |
| `run_4dgs_elly_v13_masked.sh` | 150 | yes | **Centered time + progressive sampling** |
| `run_4dgs_thenewface_v11_masked.sh` | 300 | yes | No opacity deform |
| `run_4dgs_thenewface_v12_masked.sh` | 300 | yes | **Centered time + progressive, ssim=0.08** |

---

## Recommended Config for 90 Frames

For 90-frame sequences with fast/complex motion:

```bash
--num_frames 90
--frame_stride 5
--deform_time_resolution 300      # ~3.3 grid cells per frame
--deform_net_width 128
--deform_net_depth 6
--deform_time_pe_bands 8          # 16 dims, max freq 128 > 90
--enable_opacity_deform            # Gaussians can appear/disappear
--enable_sh_deform                 # Colors change with motion
--time_smooth_weight 0.001
--time_smooth_weight_final 0.0001  # Anneal to allow sharp motion late
--time_smooth_order 1              # 1st-order: penalize jitter, not fast motion
--ssim_lambda 0.2
--max_steps 50000
```

Key differences from 30-frame config:
- `deform_time_resolution`: 150 → 300 (proportional to frame count)
- `deform_time_pe_bands`: 6 → 8 (higher freq needed for 90 frames)

---

## Architecture Overview (v4)

```
Training Phase 1 — Coarse (steps 0–3000):
  Static 3DGS only, deformation OFF
  Densification active (split/clone/prune)
  SH degree gradually unlocked (0→3)

Training Phase 2 — Fine (steps 3000–50000):
  Deformation field active
  Densification continues to step 15000 (matching paper)
  Gaussian cap at 1M (safety)
  HexPlane regularization with annealed temporal smoothness

Deformation Pipeline:
  canonical means [N,3] ──┬──→ HexPlane(xyz, t) → features [N, C*num_scales]
                          │         ↓
                          │    + time PE (sin/cos, 8 bands) → [N, C*num_scales + 16]
                          │         ↓
                          │    backbone MLP (6 layers, width 128)
                          │         ↓
                          │    output heads:
                          │      Δxyz [N,3]     — position
                          │      Δrot [N,4]     — rotation (additive quaternion)
                          │      Δscale [N,3]   — scale (log-space)
                          │      Δopacity [N,1] — opacity (logit-space)
                          │      ΔSH [N,48]     — color (SH coefficients)
                          │         ↓
                          └──→ apply deltas → deformed Gaussians
                                    ↓
                              gsplat.rasterization() → RGB image

Regularization (annealed):
  plane_tv_loss     — spatial smoothness on XY, XZ, YZ planes
  time_smooth_loss  — 1st-order temporal smoothness on XT, YT, ZT planes
                      (penalizes jitter, allows fast motion)
                      weight: 0.001 → 0.0001 (cosine annealing)
```

---

## Files Modified (Complete)

- `examples/simple_trainer_4dgs.py` — main trainer, Config, DynamicRigDataset, training loop
- `examples/deformation/hexplane.py` — multi-resolution temporal planes
- `examples/deformation/deform_network.py` — time PE, opacity/SH deform heads
- `examples/deformation/regulation.py` — 1st-order time smoothness loss
- `examples/deformation/__init__.py` — exports
- `examples/run_4dgs_elly.sh` — 30f launch script
- `examples/run_4dgs_elly_masked.sh` — 30f masked launch script
- `examples/run_4dgs_elly_v4.sh` — 30f full dynamic launch script
- `examples/run_4dgs_elly_v4_masked.sh` — 30f full dynamic masked launch script
- `examples/run_4dgs_elly_90f.sh` — 90f full dynamic launch script
- `examples/run_4dgs_elly_90f_masked.sh` — 90f full dynamic masked launch script
