# 4C4D: 4 Camera 4D Gaussian Splatting — Implementation Guide

Implementation of **4C4D** (CVPR 2026, Zhou et al.) on top of the gsplat framework.

**Paper:** "4C4D: 4 Camera 4D Gaussian Splatting" — enables high-fidelity 4D dynamic scene reconstruction from as few as 4 cameras.

## Files

| File | Description |
|------|-------------|
| `examples/simple_trainer_4c4d.py` | Main training script (~1400 lines) |
| `examples/run_4c4d_face_lowres.sh` | Launch script for thenewface dataset |
| `examples/4c4d_implement.md` | This documentation |

---

## Paper Equations → Code Mapping

### Eq 1: Time-dependent position (Section 3.1)

```
sigma(t) = sigma + Sigma_{1:3,4} * Sigma_{4,4}^{-1} * (t - mu_t)
```

**Code** (`compute_4d_gaussians`): We learn `sigma_cross = Sigma_{1:3,4} * Sigma_{4,4}^{-1}` as a direct [N, 3] parameter per Gaussian. Position at time t:

```python
sigma_cross = self.splats["sigma_cross"]  # [N, 3]
dt = timestamp - mu_t
means_t = means + sigma_cross * dt.unsqueeze(-1)
```

This allows each Gaussian to move linearly in 3D over time — critical for modeling facial motion, body movement, etc.

### Eq 2-3: Temporal opacity factor (Section 3.1)

```
o(t) = omega(t) * o
omega(t) = exp(-0.5 * (t - mu_t)^2 / Sigma_{4,4})
```

**Code** (`compute_4d_gaussians`):

```python
mu_t = self.splats["means_t"]                    # temporal center
scale_t = torch.exp(self.splats["scales_t"])      # temporal duration (log-space)
temporal_factor = torch.exp(-0.5 * dt**2 / (scale_t**2 + 1e-8))
opacities_t = opacities * temporal_factor
```

Each Gaussian has a temporal center (`mu_t`) and duration (`scale_t`). Gaussians far from the current timestamp become transparent.

### Eq 4: 4D Spherical Harmonics (Section 3.1)

```
c(t, theta, phi) = cos(2*pi*n/T * t) * Y_lm(theta, phi)
```

**Code** (`compute_4d_gaussians`): Higher-order SH coefficients (shN) are modulated by a temporal Fourier basis:

```python
freq_idx = (torch.arange(K) % num_fourier_freqs) + 1
modulation = torch.cos(2 * pi * freq_idx * timestamp)
shN_modulated = shN * modulation.unsqueeze(-1)
colors = torch.cat([sh0, shN_modulated], 1)
```

DC term (sh0) stays constant — baseline color. Higher-order SH varies with time, enabling time-varying appearance (expression changes, lighting shifts).

### Eq 5-6: Neural Decaying Function (Section 3.2)

```
tau = f_theta(opacity, xyzt, scales_xyzt)
o(t) = tau * omega(t) * o
```

**Code** (`NeuralDecayCoefficient` class + `compute_4d_gaussians`):

```python
# Matches author's Coefficient class:
# Input: (opacity[1], 4D_positions[4], 4D_scales[4]) = 9 dims, all normalized
# Network: Linear(9,32) → ReLU → Dropout(0.1) → Linear(32,1) → Sigmoid
positions_4d = torch.cat([means, mu_t.unsqueeze(-1)], dim=1)     # [N, 4]
scales_4d = torch.cat([scales, scale_t.unsqueeze(-1)], dim=1)    # [N, 4]
coef = self.decay_mlp(opacities.unsqueeze(-1), positions_4d, scales_4d)  # [N, 1]
tau = cfg.decay_f_min + (cfg.decay_f_max - cfg.decay_f_min) * coef.squeeze(-1)
opacities_t = opacities * temporal_factor * tau
```

The coefficient output is mapped to `[f_min, f_max] = [0.996, 0.998]` — a gentle decay that forces gradients to focus on geometric learning.

**Warmup**: NDF activates after 500 iterations (`decay_warmup=500`).

**Persistent decay**: After each optimizer step, visible Gaussians' stored opacity is decayed by the coefficient (once per step, matching the author's per-render decay with batch_size=1):

```python
# After optimizer.step():
new_opa = torch.where(vis_mask, opa * tau, opa)  # visible: neural decay; invisible: unchanged
self.splats["opacities"].data = torch.logit(new_opa.clamp(1e-7, 1-1e-7))
```

---

## Architecture: All-Trainable Gaussians

When using PLY init, both static and dynamic PLYs are **merged into one trainable set** (no frozen Gaussians). When using SFM init, COLMAP points are used with standard densification.

| Init Mode | Source | Count | Trainable | Densification |
|-----------|--------|-------|-----------|---------------|
| PLY (merged) | `outside_05.ply` + `inside_05.ply` | 81,043 | **All** | Off |
| SFM | COLMAP `points3D.bin` | ~658K | All | On (DefaultStrategy) |

### Per-Gaussian parameters

| Parameter | Shape | Space | Description |
|-----------|-------|-------|-------------|
| `means` | [N, 3] | world | Canonical position |
| `scales` | [N, 3] | log | Gaussian scale |
| `quats` | [N, 4] | raw | Rotation quaternion |
| `opacities` | [N] | logit | Base opacity |
| `sh0` | [N, 1, 3] | raw | DC color (SH degree 0) |
| `shN` | [N, K, 3] | raw | Higher-order SH (view+time dependent) |
| `means_t` | [N] | raw | Temporal center mu_t |
| `scales_t` | [N] | log | Temporal duration (log-space, exp for sigma_t) |
| `sigma_cross` | [N, 3] | raw | Position shift coupling (Eq 1) |

---

## Data Pipeline

### Image preloading (zero I/O training)

All images preloaded to GPU as uint8 tensors at startup:

- **First run**: Threaded JPEG reading (16 workers) + resize → saves `.pt` cache to disk
- **Subsequent runs**: Loads `.pt` cache in ~2 seconds
- Per-camera resolution preserved (no padding — cameras have different sizes due to undistortion)
- Cache location: `<data_dir>/cache_f<factor>_<num_frames>f.pt`

Memory at factor 15: ~1.4 GB VRAM for 45 cams x 300 frames.

### Batching: Two sampling modes

**Random sampling** (`--sampling random`, default, matches author):
Each step picks `batch_size` random (camera, frame) pairs from the full dataset. Grouped by timestamp for efficient `compute_4d_gaussians` (computed once per unique timestamp).

**Progressive sweep** (`--sampling progressive`):
Sweeps through time blocks sequentially, all cameras per block.

```
# Random: each step samples batch_size random pairs
Step 1: cam12/frame23, cam07/frame41, cam30/frame02, ...
Step 2: cam44/frame19, cam01/frame38, cam22/frame07, ...

# Progressive: consecutive blocks, all cameras
Step 1: Block 0 (frames 1-5)  → 45 cameras
Step 2: Block 1 (frames 6-10) → 45 cameras
...
```

### Per-camera rendering

Each camera rendered individually (not batched) because:
1. Cameras have different resolutions after undistortion
2. Bounds peak rasterization VRAM to 1 camera
3. Loss accumulated across all renders, single backward pass

---

## Training Loop (per step)

```
1. Sample (camera, frame) pairs:
   - Random mode: pick batch_size random pairs, group by timestamp
   - Progressive mode: get next time block, use all cameras
2. For each unique timestamp:
   a. compute_4d_gaussians() ONCE:
      - Eq 1: shift positions by sigma_cross * dt
      - Eq 2-3: temporal opacity modulation
      - Eq 4: Fourier-modulate SH coefficients
      - Eq 5-6: NDF coefficient (after warmup)
   b. For each camera at this timestamp:
      - Slice GT image from GPU cache (zero I/O)
      - rasterization() at camera's native resolution
      - L1 + SSIM loss
      - Track visible Gaussian IDs
3. Average loss over all renders
4. Single loss.backward()
5. Step optimizers (Gaussian params + NDF MLP)
6. Persistent opacity decay (once per step, visible Gaussians only)
7. Log to TensorBoard
```

---

## Config Reference

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | — | Path to COLMAP dataset (with `images/` and `sparse/0/`) |
| `result_dir` | — | Output directory |
| `data_factor` | 15 | Image downscale factor |
| `static_ply` | None | Path to frozen background PLY |
| `dynamic_ply` | None | Path to trainable foreground PLY |

### Temporal / 4D

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_frames` | 300 | Total frames to train on |
| `frames_per_step` | 5 | Consecutive frames per time block |
| `frame_start` | 1 | First frame index |
| `frame_step` | 1 | Frame stride |
| `num_fourier_freqs` | 1 | Fourier frequencies for 4D SH (Eq 4) |
| `temporal_lr` | 1e-3 | LR for mu_t, scale_t, sigma_cross |

### Neural Decaying Function (matches author's Coefficient)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decay_warmup` | 500 | Steps before NDF activates (paper: 500) |
| `decay_mlp_lr` | 1e-3 | NDF MLP learning rate |
| `decay_mlp_hidden` | 32 | MLP hidden dimension (author's default) |
| `decay_dropout` | 0.1 | Dropout rate in NDF MLP |
| `decay_f_min` | 0.996 | Min decay factor |
| `decay_f_max` | 0.998 | Max decay factor |
| `sampling` | random | `random` (author's) or `progressive` (block sweep) |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 30000 | Total training steps |
| `batch_size` | 45 | Cameras per step (45 = all cameras) |
| `sh_degree` | 1 | SH degree (0=DC only, 1=first-order, 3=full) |
| `ssim_lambda` | 0.2 | Weight for SSIM loss |
| `eval_steps` | [7000, 15000, 30000] | Evaluation checkpoints |
| `tb_image_every` | 200 | GT vs rendered images to TensorBoard |

---

## Running

### Quick test (50 frames, ~2 min)

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n gsplat python simple_trainer_4c4d.py default \
    --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted \
    --result_dir /data/shared/elaheh/4D_demo/new_data/thenewface/results_4c4d_test \
    --data_factor 15 --num_frames 50 --frames_per_step 5 --batch_size 45 \
    --sh_degree 1 --num_fourier_freqs 1 --max_steps 1000 --normalize_world_space \
    --static_ply /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/outside_05.ply \
    --dynamic_ply /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/inside_05.ply \
    --disable_viewer
```

### Full training (300 frames, 30K steps, ~8 hours)

Via tmux (recommended):

```bash
tmux new-session -d -s 4c4d "PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2 \
    /home/elaheh/miniforge3/envs/gsplat/bin/python -u \
    /home/elaheh/projects/gsplat/examples/simple_trainer_4c4d.py default \
    --data_dir /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted \
    --result_dir /data/shared/elaheh/4D_demo/new_data/thenewface/results_4c4d \
    --data_factor 15 --num_frames 300 --frames_per_step 5 --frame_start 1 \
    --max_steps 30000 --eval_steps 7000 15000 30000 --save_steps 7000 30000 \
    --batch_size 45 --sh_degree 1 --num_fourier_freqs 1 --normalize_world_space \
    --static_ply /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/outside_05.ply \
    --dynamic_ply /data/shared/elaheh/4D_demo/new_data/thenewface/undistorted/static_dynamic_output/inside_05.ply \
    --test_every 0 --val_num_cameras 5 --val_num_frames 5 \
    --decay_warmup 500 --decay_mlp_lr 1e-3 --invisible_decay_beta 0.999 \
    --temporal_lr 1e-3 --ssim_lambda 0.2 \
    --tb_every 100 --tb_image_every 200 --tb_image_num_views 3 \
    --disable_viewer 2>&1 | tee /data/shared/elaheh/4D_demo/new_data/thenewface/results_4c4d/train.log"
```

Or via the run script:

```bash
bash run_4c4d_face_lowres.sh 2   # GPU 2
```

### Monitoring

```bash
# TensorBoard (scalars: loss/PSNR/mem, images: GT vs rendered)
tensorboard --logdir /data/shared/elaheh/4D_demo/new_data/thenewface/results_4c4d/tb --port 6006 --bind_all

# Live training log
tail -f /data/shared/elaheh/4D_demo/new_data/thenewface/results_4c4d/train.log

# Attach to tmux session
tmux attach -t 4c4d   # Ctrl+B then D to detach
```

---

## Output Structure

```
results_4c4d/
  cfg.yml       - Config snapshot
  train.log     - Full training log (stdout+stderr)
  tb/           - TensorBoard events
                  - Scalars: train/loss, train/num_GS, train/mem, train/block_t_start/end
                  - Images: train_cmp/cam{N}_frame{F} (GT | Rendered, 4x upscaled)
  ckpts/        - Checkpoints: ckpt_{step}_rank0.pt
                  Contains: splats state_dict + decay_mlp state_dict
  renders/      - Eval images (GT | Rendered side-by-side)
  videos/       - Trajectory render MP4s
  stats/        - JSON metrics per eval step (PSNR, SSIM, LPIPS, timing)
```

---

---

## Bugs Found & Fixed (2026-04-10)

### Bug 1: Double Camera Intrinsics Scaling (CRITICAL)

**Problem:** The `Parser` class in `datasets/colmap.py` already divides camera intrinsics by `data_factor`:

```python
# Inside Parser.__init__ (datasets/colmap.py line 220):
K[:2, :] /= factor
imsize_dict[cam_id] = (cam.width // factor, cam.height // factor)
```

But the trainer divided K again in two places:

```python
# WRONG — training loop (was line 1032):
if cfg.data_factor > 1:
    all_Ks[:, :2, :] /= cfg.data_factor  # DOUBLE SCALING!

# WRONG — dataset __getitem__ (was line 382):
if self.factor > 1:
    K[:2] /= self.factor  # DOUBLE SCALING!
```

With `data_factor=15`, the focal length was `15 x 15 = 225x` too small. The camera behaved like an extreme fisheye — all Gaussians projected to wrong positions. Renders looked nothing like ground truth.

**Fix:** Removed both redundant divisions. K is already correct from the Parser.

---

### Bug 2: Optimizer Batch Scaling Destroying Gaussians (CRITICAL)

**Problem:** gsplat's standard trainer caps batch scaling at 10:

```python
# gsplat simple_trainer.py:
BS_scale = min(BS, 10)  # capped!
```

Our code used `BS = batch_size * world_size = 45` **uncapped**:

```python
# WRONG — no cap on BS:
BS = cfg.batch_size * world_size  # 45!
beta1 = max(1e-8, 1 - BS * (1 - 0.9))   # = max(1e-8, -3.5) = 1e-8  → NO MOMENTUM
beta2 = max(1e-8, 1 - BS * (1 - 0.999)) # = 0.955
lr = lr_base * math.sqrt(BS)              # = lr * 6.7x
```

With `beta1 ≈ 0`, Adam becomes a sign-based optimizer — each step changes parameters by `±lr` regardless of gradient magnitude. Scales changed by ±0.034 per step in log-space. After 600 steps: `exp(20) ≈ 500 million x`. Gaussians exploded into noise within the first few steps.

**Fix:** Removed all batch scaling. `batch_size=45` means "number of cameras to render per step" — the loss is already averaged over all renders, so standard Adam hyperparameters (lr as-is, beta1=0.9, beta2=0.999) are correct:

```python
# FIXED:
self.optimizers = {
    name: torch.optim.Adam(
        [{"params": self.splats[name], "lr": lr, "name": name}],
        eps=1e-15,
        betas=(0.9, 0.999),  # standard Adam, no batch scaling
    )
    for name, _, lr in params
}
```

---

### Bug 3: PLY Normalization Mismatch (CRITICAL)

**Problem:** When `--normalize_world_space` is used, the Parser applies a 4x4 similarity transform to camera positions and SFM points:

```python
# Inside Parser.__init__ (datasets/colmap.py):
T1 = similarity_from_cameras(camtoworlds)
camtoworlds = transform_cameras(T1, camtoworlds)
points = transform_points(T1, points)  # SFM points get transformed
self.transform = T_recenter @ T1       # saved for later use
```

But PLY files loaded via `load_ply_splats()` were NOT transformed. The PLY Gaussians remained in the original COLMAP world coordinate frame while cameras were in the normalized frame. Result: Gaussians and cameras in completely different coordinate systems — everything looked blurry/wrong.

**What the PLY contains** (verified by inspecting the actual file):

```
=== OPACITY (from PLY) ===    range: [-1.970, 13.802]  → logit-space (raw 3DGS)
=== SCALES (from PLY) ===     range: [-13.285, -1.117]  → log-space (raw 3DGS)
=== POSITIONS (from PLY) ===  x=[-0.62, 2.18], y=[-3.23, 1.08], z=[-2.86, 1.37]  → COLMAP frame
```

**Fix:** Apply the Parser's normalization transform to PLY positions and adjust scales:

```python
# FIXED — apply same transform that Parser applied to cameras:
if cfg.normalize_world_space and hasattr(self.parser, 'transform'):
    T = self.parser.transform  # [4, 4] numpy
    # Transform positions: p' = p @ R^T + t
    means_np = merged["means"].cpu().numpy()
    means_transformed = means_np @ T[:3, :3].T + T[:3, 3]
    merged["means"] = torch.tensor(means_transformed, dtype=torch.float32, device=self.device)
    # Scale Gaussian sizes by the transform's scale factor
    scale_factor = np.cbrt(np.linalg.det(T[:3, :3]))  # extract uniform scale
    if abs(scale_factor - 1.0) > 1e-6:
        merged["scales"] = merged["scales"] + math.log(scale_factor)  # add in log-space
```

For the thenewface dataset, `scale_factor = 0.2675` — the PLY was in a frame ~3.7x larger than normalized space.

---

### Bug 4: Frozen Static Gaussians (MAJOR)

**Problem:** The 63K background Gaussians from `outside_05.ply` were frozen (no gradients, no optimization):

```python
# WRONG — static Gaussians were a plain dict, no grad:
self.static_splats = static_data  # plain dict of tensors, no grad
```

Any imperfection in the background rendering was permanent. Combined with Bug 3 (wrong coordinate frame), the frozen background was rendered in the wrong place with no way to fix it. Meanwhile SFM init had everything trainable and performed much better.

**Fix:** Merge both PLYs into one all-trainable set:

```python
# FIXED — merge static + dynamic, all trainable:
merged = {}
for key in ["means", "scales", "quats", "opacities"]:
    merged[key] = torch.cat([static_data[key], dyn_data[key]], 0)
# ... all 81K Gaussians are nn.Parameters with gradients
self.static_splats = None  # no frozen Gaussians
```

---

### Bug 5: Wrong NeuralDecayingFunction Architecture (MAJOR)

**Problem:** Our implementation didn't match the author's `Coefficient` class:

| | Our code (wrong) | Author's code |
|---|---|---|
| Input | `(xyz_3d, opacity, quats)` = 8 dims | `(opacity, xyzt_4d, scales_4d)` = 9 dims |
| Normalization | None | opacity→[-1,1], positions mean/std, scales log/mean/std |
| Network | 3 layers, hidden=64, no dropout | 2 layers, hidden=32, dropout=0.1 |
| Formula | `opacity * tau` directly | `opacity * (f_min + (f_max - f_min) * coef)` |
| f_min, f_max | N/A | 0.996, 0.998 |

```python
# WRONG — old NeuralDecayingFunction:
class NeuralDecayingFunction(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64):
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, means, opacities, quats):
        x = torch.cat([means, opacities.unsqueeze(-1), quats], dim=-1)  # [N, 8]
        return self.network(x).squeeze(-1)
```

**Fix:** Replaced with `NeuralDecayCoefficient` matching the author's architecture:

```python
# FIXED — matches author's Coefficient class:
class NeuralDecayCoefficient(nn.Module):
    def __init__(self, hidden_dim=32, dropout_rate=0.1):
        input_dim = 9  # opacity(1) + xyzt(4) + scales_xyzt(4)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1), nn.Sigmoid())
    def forward(self, opacity, positions, scales):
        # Normalize inputs (matching author):
        opa = opacity * 2 - 1                      # [-1, 1]
        pos = (positions - positions.mean(0)) / (positions.std(0) + 1e-6)
        sca = torch.log(scales + 1e-6)
        sca = (sca - sca.mean(0)) / (sca.std(0) + 1e-6)
        x = torch.cat([opa, pos, sca], dim=1)      # [N, 9]
        return self.net(x)

# Applied as:
tau = cfg.decay_f_min + (cfg.decay_f_max - cfg.decay_f_min) * coef  # ~0.997
opacities_t = opacities * temporal_factor * tau
```

---

### Bug 6: Wrong Temporal Initialization (MODERATE)

**Problem:** PLY Gaussians were trained at frame 0, but temporal center was initialized to 0.5 (mid-sequence). And temporal scale covered the entire range instead of ~1/5 like the author:

```python
# WRONG:
means_t = torch.full((N,), 0.5)   # should be 0.0 for PLY (trained at frame 0)
scales_t = torch.zeros((N,))      # exp(0) = 1.0, covers entire range
```

The author initializes temporal centers randomly and scales to cover ~1/5 of the time range:

```python
# Author's create_from_pcd:
fused_times = random(N, 1) * time_duration   # random centers
scales_t = log(sqrt(time_duration / 5))       # each covers ~1/5
```

**Fix:**

```python
# FIXED — PLY init (trained at frame 0):
means_t = torch.full((N,), 0.0)
scales_t = torch.full((N,), math.log(math.sqrt(0.2)))  # covers ~1/5 of [0,1]

# FIXED — SFM init (random like author):
means_t = torch.rand((N,))
scales_t = torch.full((N,), math.log(math.sqrt(0.2)))
```

---

### Bug 7: Inefficient Batching — 45x Redundant Computation (PERFORMANCE)

**Problem:** `compute_4d_gaussians(timestamp)` was called inside the per-camera loop — 225 times per step (45 cameras x 5 frames). But for the same timestamp, the 4D Gaussians are identical across all cameras.

```python
# WRONG — computed per camera:
for t_idx in range(N_frames):
    for ci in cam_perm:
        renders = self.rasterize_splats(timestamp=timestamp, ...)
        # rasterize_splats calls compute_4d_gaussians() inside → 225 calls!
```

**Fix:** Compute once per timestamp, rasterize per camera:

```python
# FIXED — compute once per timestamp:
for t_idx, frame_rank in enumerate(unique_frame_ranks):
    timestamp = all_timestamps[t_idx].item()
    # Compute 4D Gaussians ONCE (same for all cameras)
    dyn_means, dyn_quats, dyn_scales, dyn_opacities, dyn_colors = \
        self.compute_4d_gaussians(timestamp)
    # Merge with static once
    all_means = torch.cat([dyn_means, s_means], 0)
    ...
    for ci in cams_for_this_frame:
        # Only change camera params — reuse precomputed Gaussians
        rasterization(means=all_means, ..., viewmats=viewmat_i, Ks=K_i, ...)
```

Result: 5 `compute_4d_gaussians()` calls instead of 225. **~2x overall speedup.**

---

### Bug 8: Progressive Sweep vs Author's Random Sampling (ALIGNMENT)

**Problem:** Our training swept progressively through time blocks (all cameras per block). The author randomly samples individual (camera, frame) pairs via a standard DataLoader with shuffle.

**Fix:** Added `--sampling random` config option:

```python
# Random mode (matching author): each step picks batch_size random (camera, frame) pairs
if cfg.sampling == "random":
    cam_perm = torch.randperm(ds.num_cameras)[:B].tolist()
    rand_frame_ranks = torch.randint(0, ds.total_frames, (B,)).tolist()
    # Group by unique timestamp for efficient compute_4d_gaussians
    samples_by_frame = {}
    for ci, fr in zip(cam_perm, rand_frame_ranks):
        samples_by_frame.setdefault(fr, []).append(ci)
```

---

### Bug 9: Video Writer Crash (MINOR)

**Problem:** `imageio.get_writer("file.mp4", fps=30)` defaulted to TIFF writer instead of FFMPEG, crashing at eval.

**Fix:** Explicit FFMPEG format:

```python
writer = imageio.get_writer(path, fps=30, format="FFMPEG", codec="libx264")
```

---

### Summary: Impact of Each Fix

| Bug | Symptom | Impact |
|-----|---------|--------|
| Double K scaling | Renders look nothing like GT | **Completely broken** |
| Optimizer batch scaling | Gaussians explode into noise in ~10 steps | **Completely broken** |
| PLY normalization | PLY init looks blurry, SFM works fine | **PLY init broken** |
| Frozen static | Background can't adapt | **Lower quality** |
| Wrong NDF architecture | Suboptimal opacity regularization | **Moderate** |
| Wrong temporal init | Gaussians start at wrong time | **Moderate** |
| Redundant computation | 2x slower than needed | **Performance** |
| Progressive sampling | Different training dynamics than author | **Alignment** |
| Video writer | Crash at eval | **Minor** |

---

## PLY Files

| File | Points | Role | VRAM |
|------|--------|------|------|
| `outside_05.ply` | 63,422 | Static background (frozen) | ~50 MB |
| `inside_05.ply` | 17,621 | Dynamic foreground (trainable) | ~15 MB |
| `outside.ply` | 1,268,425 | Full static (not used — too large) | |
| `inside.ply` | 352,406 | Full dynamic (not used — use for higher quality) | |

---

## Performance

| Metric | Value |
|--------|-------|
| Preload (first run, 300f) | ~3.5 min (threaded) |
| Preload (cached) | ~2s |
| Training speed | ~1 it/s (225 renders/step) |
| VRAM | ~5.9 GB |
| ETA 30K steps | ~8 hours |
| Image cache | ~1.4 GB (uint8 on GPU) |

---

## Data Requirements

The dataset must have:
```
<data_dir>/
  images/           - Camera subdirectories with numbered frames
    take_18_cam_01/
      000001.jpg
      000002.jpg
      ...
    take_18_cam_02/
      ...
  sparse/0/         - COLMAP model (cameras.bin, images.bin, points3D.bin)
```

Camera poses come from COLMAP (one pose per camera from a reference frame). The rig is assumed fixed — all cameras share the same pose across all frames.
