# TrackerSplat (paper-faithful) + FasterGS — implementation plan

## Why we can't just use existing FasterGS functions

You keep asking whether we can build motion_fusion out of existing FasterGS APIs. We can't, and here's precisely why.

### What motion_fusion must produce

For each Gaussian `g` after looking at one camera view:

```
V1[g]     = Σ over pixels p that Gaussian g covers:  w(p,g) · [px,py,1]·[px,py,1]ᵀ     (3×3)
V2[g]     = Σ over pixels p that Gaussian g covers:  w(p,g) · [px,py,1]·[px',py']ᵀ    (3×2)
α_acc[g]  = Σ over pixels p that Gaussian g covers:  w(p,g)                             (1)
pixhit[g] = count of pixels p that Gaussian g covers                                     (1)
```

where `w(p,g) = α(p,g) · T(p)` is the alpha-blend weight that the rasterizer already computes internally per (pixel, Gaussian) pair, and `[px',py'] = motion_map[py,px]` is the DOT tracker's predicted position for pixel (px,py).

**The math requires per-(pixel, Gaussian) pair access.** Not per-pixel aggregates, not per-Gaussian aggregates — individual pair visits.

### What FasterGS actually exposes

| FasterGS API | Per-pair access? | What it gives us | Why it's not enough |
|---|---|---|---|
| `diff_rasterize` | No | Final image only | Pairs are consumed in atomic pixel blends; only the aggregate `Σ_g w(p,g)·color_g` escapes |
| `rasterize` | No | Final image only | Same |
| `update_pruning_scores` | **Yes, internally** | Writes **1 float** `dL_dgaussian²` per Gaussian | Per-pair loop is already inside the CUDA kernel, but output buffer is fixed-width; we need 13 floats per Gaussian, not 1 |
| `update_3d_filter` | No | Per-Gaussian filter + visibility | Based on projection geometry only, no alpha-blend weights |
| `add_noise` | No | Modifies params in-place | Not an accumulator |
| `relocation_adjustment` | No | Pure opacity/scale math | Not an accumulator |

### The "just reuse the rasterizer with fake colors" trick doesn't work either

If we set each Gaussian's color to a one-hot vector (`color_g = e_g`), the rasterized image at pixel `p` becomes:
```
image[p] = Σ_g α(p,g)·T(p) · e_g  =  [α(p,1)·T(p), α(p,2)·T(p), ..., α(p,N)·T(p)]
```

That would tell us `w(p,g)` for every pair — **but FasterGS images are 3 channels (RGB), not N channels**. For N = 1.6M Gaussians we'd need 533k separate rasterizer passes with different color assignments, then reconstruct per-pair weights from the image outputs. Cost is orders of magnitude worse than just doing the math in PyTorch.

### So our two real options

1. **Modify FasterGS** — clone `pruning_scores.cu` into a new `motion_fusion.cu` kernel that writes 13 atomic floats per Gaussian instead of 1. ~750 lines of CUDA. This is the paper's exact path, Apache-licensed, ~5× faster than pure PyTorch, but needs CUDA debugging which we've ruled out.
2. **Pure-PyTorch motion_fusion** — skip the rasterizer entirely and implement the per-Gaussian, per-pixel accumulation ourselves in Python. Mathematically identical (minus the transmittance approximation). Works on GPU via PyTorch ops. ~5–10× slower per frame but requires **zero CUDA work**. This is what we're doing.

---

## Scope: pure-PyTorch paper-faithful TrackerSplat

Goal: reproduce the paper's pipeline with results within ~0.5 dB PSNR of the reference, under Apache 2.0, using only FasterGS (Apache) for the final refinement rasterization. No Inria code, no CUDA modifications.

### Pipeline (matches reference `motionestimation.py` order)

```
Frame 0: InstantSplat-style init (we already have this via inside.ply / outside.ply)
           │
           ▼
For each subsequent frame t:
  Point Tracking        DOT (or alltrackerxx) per view          ← already have npz
           │
           ▼
  Motion Fusion         per-Gaussian V1, V2, α_acc, det, pixhit ← pure PyTorch
           │
           ▼
  PWI-LS solve          per-Gaussian [A|b] 2D affine            ← vendored (incremental_ls.py)
           │
           ▼
  ISVD triangulation    multi-view 2D affine → 3D (μ, R, S)    ← vendored (incremental_svd.py)
           │
           ▼
  Regularization        K-NN median filter + 8-NN propagation  ← vendored (medianfilter.py + propagation.py)
           │
           ▼
  Compensate            apply Motion → updated Gaussians        ← vendored (motion.py::compensate)
           │
           ▼
  Refinement            1000 SGD iters, photometric only        ← existing trackersplat_trainer_fastgs.py
           │
           ▼
Frame t output PLY
```

### Files we'll add to `examples/trackersplat_paper/`

Vendored verbatim from reference (Apache 2.0, license notice preserved):
```
motion.py                # Motion NamedTuple + compensate() method
utils/incremental_ls.py  # ILS_Cov3D: per-Gaussian cov3D normal equations
utils/incremental_svd.py # ISVD_Mean3D: per-Gaussian 3D triangulation via stacked SVD
utils/medianfilter.py    # K-NN median filter on Δμ, ΔR, ΔS
utils/propagation.py     # 8-NN motion propagation for unsolvables
motionestimator/refiner/{filter.py, propogate.py, training.py}
motionestimation.py      # pipeline orchestrator
```

Written by us, pure PyTorch:
```
motion_fusion_pytorch.py  # THE KEY REPLACEMENT — does per-Gaussian, per-pixel
                          # accumulation of V1/V2/α_acc/det/pixhit. Vectorized
                          # via scatter_add_. Runs on L40S in ~2–10 s per view.
                          #
                          # Approximation: ignores transmittance T (treats
                          # T=1). Paper's LS is opacity-weighted and the
                          # transmittance contribution is dominated by the
                          # α(p,g) term anyway.
```

### Day-by-day

| Day | Deliverable |
|---|---|
| 1 | Vendor reference Apache Python. Swap their `gaussian_splatting.Camera` imports for our `examples/trackersplat_dataset.Camera`. All files import cleanly. `motion_fusion()` stubbed to raise `NotImplementedError`. |
| 2 | Write `motion_fusion_pytorch.py` against the spec in the Motion Fusion section. Chunked per-Gaussian accumulation to cap peak GPU memory. |
| 3 | Validate motion_fusion output on a 100-Gaussian synthetic scene with known ground-truth affine motion. V1/V2 match analytic values within 1e-4. |
| 4 | Wire the full pipeline on ONE frame of `thenewface`. Produce a Motion object; compensate; render before/after. |
| 5 | Run on all 50 frames sequentially. Compare PSNR/timing to SGD baseline. |
| 6–7 | Debug edge cases (degenerate V1, few-view triangulation, numerical instability). Tune propagation K / median filter K. |

Budget: **5–7 focused days**, pure PyTorch, no CUDA to debug.

### Known trade-offs vs paper

1. **No transmittance weighting.** We use `α(p,g)` alone instead of `α(p,g)·T(p)` for the LS weights. Matters for heavily-overlapping dynamic foreground Gaussians. For single-layer scenes (`thenewface`) the impact is small.
2. **~5–10× slower per view.** Paper CUDA hits ~30 ms/view; pure PyTorch on L40S will hit ~300 ms – a few seconds/view. Full 50-frame run: ~2–7 hours instead of ~30 minutes.
3. **No parallel-GPU sharding.** Single-process. Easy to add later if the sequential version works.

### Files we won't touch

- `examples/trackersplat_trainer_fastgs.py` stays as-is (used for the 1000-iter refinement phase at the end of each frame).
- FasterGSCudaBackend source is **completely untouched**.
- `gsplat` source is **completely untouched**.
- No new pip installs.

## Starting condition

When you say "go":
1. I create `examples/trackersplat_paper/` layout.
2. Fetch + vendor the reference Apache Python files (license headers preserved).
3. Swap Camera imports.
4. Stub `motion_fusion()` with the exact spec from this doc.
5. Commit as the Day-1 deliverable.

After Day 1 you can read the full architecture in our repo before I write any math.
