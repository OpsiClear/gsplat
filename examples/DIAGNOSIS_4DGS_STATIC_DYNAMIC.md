# 4DGS Static+Dynamic Trainer — Diagnosis & Fixes

Debugging session for `simple_trainer_static_dynamic.py` on the elly 150-frame outdoor sequence.
Starting PSNR: 19.44 dB (5K steps) → 21.46 dB (50K steps). Expected: 25–30 dB.

---

## Bug 1 — Opacity reset at step 0 destroys PLY initialization *(critical)*

**File**: `DefaultStrategy` config in `simple_trainer_static_dynamic.py`
**Fix**: `--strategy.reset-every 99999` in run script (v2)

### What happens
`DefaultStrategy.step_post_backward` fires `reset_opa` whenever `step % reset_every == 0`.
Since `reset_every=3000` and `0 % 3000 == 0`, this triggers **at step 0** before any training.

`reset_opa` clamps all opacities to `logit(prune_opa * 2) = logit(0.01) ≈ −4.6` in log-space:
```python
opacities = torch.clamp(p, max=torch.logit(torch.tensor(0.01)))  # ≈ −4.6
```

PLY-initialized face Gaussians have opacity ≈ 0.8–0.98 (logit ≈ 1.4–3.9).
After the step-0 reset: opacity = sigmoid(−4.6) ≈ **0.01** — nearly invisible.

### Cascade failure
With coarse_iters=0, the deformation field has learned nothing at step 0.
Canonical Gaussians render at the frame-0 position even when training on non-zero timestamps.
The misplaced, now-transparent Gaussians receive "don't exist here" gradient from reconstruction.
Opacity is pushed further below `prune_opa=0.005` threshold at the next densification step.

**Result**: 160,911 → 118,869 Gaussians in the first 1K steps (−42K). Repeats every 3K steps
(10 resets total up to step 27K), wasting roughly 20K of 50K training steps rebuilding opacity.

### Why reset_every exists
In standard 3DGS, Gaussians are initialized randomly with opacity ≈ 0.1 — the reset is harmless.
For PLY-initialized models with good opacity, it is **catastrophically harmful**.
With `opacity_reg` active, floaters are handled by regularization + prune threshold — no reset needed.

---

## Bug 2 — Identity constraint corrupts canonical Gaussian positions *(critical)*

**File**: `simple_trainer_static_dynamic.py`, line ~1478
**Fix**: Added `.detach()` to `canonical_means[idx]`

### What happens
The identity constraint (deformation should be zero at t=−0.5) evaluates:
```python
# BEFORE (bug):
deform_t0 = self.deform_field(canonical_means[idx], -0.5)
constraint_loss = deform_t0.delta_xyz.pow(2).mean() + ...
```

`canonical_means[idx]` is a slice of `self.splats["means"]` — a live `nn.Parameter`.
The loss `||delta_xyz||²` backpropagates through the HexPlane bilinear interpolation
all the way into the Gaussian positions.

This gradient tells the canonical positions: *"move to where the HexPlane gives near-zero
features at t=−0.5"* — which conflicts with the reconstruction loss that says *"stay where
you look correct"*. Adam receives two opposing signals on `splats["means"]` every step.

### Fix
```python
# AFTER (fixed):
deform_t0 = self.deform_field(canonical_means[idx].detach(), -0.5)
```
The constraint now only teaches the deformation MLP to output zero at t=−0.5.
Canonical positions are unaffected.

---

## Bug 3 — HexPlane lookup creates uneven gradient paths → ghost artifacts *(critical)*

**File**: `simple_trainer_static_dynamic.py`, `rasterize_splats()`, line ~1047
**Fix**: Added `.detach()` to `self.splats["means"]` in the main deformation forward call

### What happens
```python
# BEFORE (bug):
deltas = self.deform_field(self.splats["means"], timestamp)
```

Without `.detach()`, the gradient from reconstruction loss flows through two paths:
1. **Direct path**: `d(loss)/d(means_d) → d(means)` via `means_d = means + delta_xyz`
2. **HexPlane path**: `d(loss)/d(delta_xyz) → d(HexPlane_features) → d(query_position) → d(means)`

Path 2 magnitude varies hugely across Gaussians depending on where they fall in the HexPlane grid.
Gaussians in well-covered cells (face center) learn strong deformation fast.
Gaussians in sparse cells (hair edges, thin/boundary regions) get near-zero gradient through path 2 →
their `delta_xyz ≈ 0` at all timesteps → they stay at the canonical (frame 0) position.

**Visual symptom**: Most of the head moves correctly, but some Gaussians (hair edges, thin areas)
appear frozen at the original position — a **ghost artifact** of the canonical frame.

### Fix
```python
# AFTER (fixed):
deltas = self.deform_field(self.splats["means"].detach(), timestamp)
```
Only path 1 (direct, uniform) remains. All Gaussians receive the same-quality gradient signal.
Canonical positions still get gradient from reconstruction through `means_d = means + delta_xyz`
(where `means` here in `apply_deformation` is `splats["means"]`, NOT detached).

---

## Bug 4 — `defor_depth` off-by-one: depth=0 and depth=1 are identical *(minor)*

**File**: `examples/deformation/deform_network.py`, line ~176
**Fix**: Changed `range(defor_depth - 1)` → `range(defor_depth)`

### What happens
```python
# BEFORE:
for _ in range(defor_depth - 1):   # range(-1) == range(0) == []
    backbone_layers += [nn.Linear(net_width, net_width), nn.ReLU()]
```
`defor_depth=0` → `range(-1)` = empty → 0 extra layers
`defor_depth=1` → `range(0)` = empty → 0 extra layers  ← same!

Both produced an identical 1-layer backbone: `[Linear(in→W), ReLU()]`.

### Fix
```python
# AFTER:
for _ in range(defor_depth):       # range(0)==[], range(1)==[0]
    backbone_layers += [nn.Linear(net_width, net_width), nn.ReLU()]
```
`defor_depth=0` → 0 extra layers (just projection + ReLU)
`defor_depth=1` → 1 extra hidden layer
`defor_depth=2` → 2 extra hidden layers

v2 run script uses `--deform_net_depth 1` to take advantage of the extra capacity.

---

## Hyperparameter issues (v1 run script)

| Parameter | v1 | v2 | Reason |
|---|---|---|---|
| `opacity_reg` | 0.001 | 0.0001 | 0.001 pushes sigmoid(opacity)→0 for all Gaussians; face Gaussians need high opacity. Combined with step-0 reset caused 160K→80K pruning over 50K steps |
| `scale_reg` | 0.01 | 0.005 | Less aggressive scale shrinkage preserves face coverage |
| `time_smooth_weight_final` | −1.0 (never anneals) | 0.001 | Constant 0.01 weight suppresses sharp temporal transitions; annealing to 0.001 allows the HexPlane to capture rapid face motion in later training |
| `weight_constraint_after` | 0.2 (permanent) | 0.0 | The identity constraint at t=−0.5 should be a warm-start regularizer, not permanent. Residual weight=0.2 forever creates an ongoing conflict with reconstruction gradients on the canonical positions |
| `weight_constraint_decay_iters` | 5000 | 15000 | Slower decay keeps the constraint effective longer before fully releasing |
| `deform_net_depth` | 0 (= depth 1 before fix) | 1 (= actual depth 2 after fix) | Slightly deeper backbone |
| `strategy.reset-every` | 3000 (10 resets) | 99999 (disabled) | See Bug 1 |
| `progressive_time_forward` | True (forward) | True (explicit) | Forward progressive is correct for frame-0 canonical PLY: train near-zero deformation frames first, expand forward. Symmetric (starting at t≈0) requires large deformation from the very first step — too hard for an uninitialised deformation field |

---

## Progressive sampling note

With `coarse_iters=0` and canonical PLY = frame 0 (t=−0.5):

- **Forward progressive** (`progressive_time_forward=True`): starts with window [−0.5, −0.4],
  only frames requiring small deformation. Expands to full range over `progressive_time_warmup` steps.
  ✅ Correct for frame-0 canonical.

- **Symmetric progressive** (`progressive_time_forward=False`): starts with frames near t=0
  (mid-sequence), where the face may have moved significantly from canonical.
  Requires large deformations immediately → instability → aggressive pruning → poor early PSNR.

The v2 run that was observed used symmetric (the default wasn't being parsed correctly).
`--progressive_time_forward True` is now explicit in v2.sh.

---

## HexPlane UV axis (previous fix, commit 9b9c4b5)

**File**: `examples/deformation/hexplane.py`

Temporal planes (XT, YT, ZT) have shape `[1, C, H=spatial_res, W=time_res]`.
`F.grid_sample` convention: `grid[..., 0]` → W axis, `grid[..., 1]` → H axis.

**Before fix** (in backup `deformation copy/hexplane.py`):
```python
uv = torch.stack([u, v], dim=-1)  # u=axis_i coord → grid[...,0] → W=time
```
For temporal planes: time coordinate was sampled at H=64 spatial cells instead of W=300 time cells.
This gave 4–5× lower temporal resolution, severely limiting the HexPlane's ability to distinguish
different timesteps.

**After fix**:
```python
uv = torch.stack([v, u], dim=-1)  # v=axis_j coord → grid[...,0] → W (correct)
```
Time coordinate now samples at W=time_resolution=300 cells as intended.

---

## Files changed

| File | Change |
|---|---|
| `examples/simple_trainer_static_dynamic.py` | Bug 2 fix (detach in identity constraint); Bug 3 fix (detach in main deform forward) |
| `examples/deformation/deform_network.py` | Bug 4 fix (`range(defor_depth)`) |
| `examples/run_elly_static_dynamic_v2.sh` | New run script with all hyperparameter fixes |
| `examples/deformation/hexplane.py` | UV axis fix (committed separately as 9b9c4b5) |
