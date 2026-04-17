# TrackerSplat (Apache-only) port — final status

## What's delivered

Paper-faithful implementation of Yin et al. 2025 "TrackerSplat" running under
Apache-2.0 only, on top of FasterGSCudaBackend for final rendering, with zero
Inria-derived code. Everything lives under `examples/trackersplat_paper/`.

### Pipeline — every stage implemented

```
dataset (multi-view fixed-pose)        ✅ trackersplat_dataset.py
  │
  ▼
point tracking                          ✅ alltrackerxx adapter wired;
                                           DOT-ready via --cotracker_dir
  │
  ▼
motion_fusion (per-Gaussian V1, V2,
  α_acc, pixhit accumulation)           ✅ motion_fusion_taichi.py  (pure
                                           PyTorch — Taichi blocked by a
                                           Taichi-1.7.3 quirk, documented)
  │
  ▼
PWI-LS 2D affine solve                  ✅ linear solve inline
  │
  ▼
multi-view 3D triangulation (ISVD)      ✅ utils/incremental_svd.py
  │
  ▼
median filter (K-NN)                    ✅ vendored Taichi kernel
  │
  ▼
8-NN motion propagation                 ✅ vendored Taichi kernel
                                           (off by default — sparse-track risk)
  │
  ▼
compensate(Motion) → updated splats     ✅ motion.py
  │
  ▼
FasterGS render (static + dynamic)      ✅ via run_trackersplat_paper_day4.py
```

### Sparse-track fallback (for alltrackerxx)

When the tracker is sparse (720 points / view vs paper's dense DOT), PWI-LS's
2D affine fit is under-constrained for most Gaussians. Added
`compute_translation_motion_per_track` as a sparse-friendly alternative:

    for each track point → bind to nearest projected Gaussian (one per cam)
    across views → ISVD triangulation → 3D translation per Gaussian

Gives ~0.3s solve vs 51s for full PWI-LS. CLI: `--solver per_track`.

## Tests (26/26 pass at time of commit)

| file | covers | result |
|---|---|---|
| `tests/test_day1_vendored.py` | math_utils, ILS, ISVD, Taichi kernels | 10/10 ✅ |
| `tests/test_day2_motion_fusion.py` | motion_fusion on synthetic scenes | 9/9 ✅ |
| `tests/test_day3_synthetic.py` | per-Gaussian affine + triangulation | 3/3 ✅ |
| `tests/debug_projection.py` | pinhole round-trip on real camera | ✅ |

## Real-data result

`thenewface` frame 0 → 3, 10 cameras, 100k dynamic + 1.27M static:

| configuration | mean PSNR |
|---|---|
| Baseline (render PLY at frame 0 against frame-3 GT) | 24.46 dB |
| Motion compensation via per-track triangulation | 23.08 dB |
| + outlier clip (p=0.5) + median filter | 23.19 dB |
| + 8-NN propagation **(unsafe on sparse)** | 17.35 dB |

Outputs at `/data/shared/elaheh/4D_demo/new_data/trackersplat_paper_day4/`.

## Why PSNR doesn't improve today

Motion-compensation quality is bounded by **track density**. The paper's whole
premise is DOT-style dense tracks (~510,000 targets per 1M-pixel image). With
720 sparse tracks/view:

- Per-Gaussian triangulation has ~1-unit noise floor (tracks are pixel-rounded
  from ~3m-distant cameras)
- Only 142-400 of 100k Gaussians get a direct motion signal
- Propagating the noisy motion catastrophically blurs the face

## Path to matching the paper's numbers

1. **Run DOT on your videos** (~1 day install + inference). Dense tracks unlock
   PWI-LS's per-Gaussian 2D-affine fit with hundreds of constraints per
   Gaussian, and make propagation safe to enable.
2. **OR continue with the existing SGD trainer** (`trackersplat_trainer_fastgs.py`)
   which handles sparse tracks correctly via gradient-descent averaging and
   already hits 26 dB on this data.

This repo has both paths. The paper path is `examples/trackersplat_paper/` +
`run_trackersplat_paper_day4.py`; the SGD path is
`examples/trackersplat_trainer_fastgs.py`.

## File map
```
examples/
├── trackersplat_paper/
│   ├── README.md                       ← what's vendored vs ours
│   ├── LICENSE.apache-2.0
│   ├── __init__.py                     ← public API
│   ├── motion.py                       ← Motion NamedTuple + compensate
│   ├── motion_fusion_taichi.py         ← per-Gaussian V1/V2/α accumulator
│   ├── pipeline.py                     ← PWI-LS + per-track + reg
│   ├── utils/
│   │   ├── math_utils.py               ← 3DGS projection helpers (ours)
│   │   ├── pose.py                     ← quaternion helpers (vendored)
│   │   ├── incremental_ls.py           ← PWI-LS solver (vendored)
│   │   ├── incremental_svd.py          ← ISVD triangulation (vendored)
│   │   ├── medianfilter.py             ← Taichi median (vendored)
│   │   └── propagation.py              ← Taichi propagation (vendored)
│   ├── tests/
│   │   ├── test_day1_vendored.py
│   │   ├── test_day2_motion_fusion.py
│   │   ├── test_day3_synthetic.py
│   │   └── debug_projection.py
│   └── FINAL_STATUS.md                 ← you are here
├── trackersplat_dataset.py             ← VideoCameraDataset
├── trackersplat_trainer_fastgs.py      ← the SGD trainer (PSNR 26 dB)
├── run_trackersplat_df4_fastgs.sh      ← SGD trainer launch
└── run_trackersplat_paper_day4.py      ← paper pipeline runner
```
