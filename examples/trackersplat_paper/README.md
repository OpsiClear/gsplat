# TrackerSplat (paper-faithful) — vendored pipeline

Pure-Python + Taichi port of the paper **TrackerSplat** (Yin et al. 2025) under
Apache-2.0, depending only on FasterGS (Apache-2.0) for the final refinement
rasterization. No Inria-derived code.

Reference implementation: https://github.com/yindaheng98/TrackerSplat (Apache-2.0)
Reference license copied into `LICENSE.apache-2.0`.

## What's vendored (verbatim from reference, Apache-2.0)

| file | source | notes |
|---|---|---|
| `utils/pose.py`             | `trackersplat/utils/pose.py`             | quaternion helpers |
| `utils/incremental_ls.py`   | `trackersplat/utils/incremental_ls.py`   | PWI-LS solver (import patched to `math_utils`) |
| `utils/incremental_svd.py`  | `trackersplat/utils/incremental_svd.py`  | ISVD triangulation |
| `utils/medianfilter.py`     | `trackersplat/utils/medianfilter.py`     | Taichi K-NN median |
| `utils/propagation.py`      | `trackersplat/utils/propagation.py`      | Taichi 8-NN propagation |

## What's our own (Apache-2.0, written fresh to avoid Inria-licensed deps)

| file | role |
|---|---|
| `utils/math_utils.py`       | `unflatten_symmetry_3x3` + standard 3DGS projection Jacobian (Kerbl et al. 2023 formulas, independently implemented) |
| `motion_fusion_taichi.py`   | per-Gaussian V1/V2/α_acc/det/pixhit accumulation — Taichi kernel replacing the reference's Inria-derived CUDA pass |
| `motion.py`                 | `Motion` NamedTuple + `compensate()` — adapted from reference, with its `gaussian_splatting.GaussianModel` dependency removed |
| `gaussians_wrapper.py`      | tiny duck-typed Gaussian container matching the reference's expected `._xyz`, `._rotation`, … surface, backed by FasterGS parameters |
| `pipeline.py`               | end-to-end orchestrator mirroring reference `motionestimation.py` |

## Missing pieces (by design — not in paper or filled in from our existing code)

- **Point tracking**: we provide loaders for the reference's expected format and our existing `alltrackerxx_out` npz format. DOT can be added later as a drop-in pip dep.
- **Final refinement**: uses our existing `examples/trackersplat_trainer_fastgs.py` with `refinement_steps=1000` and densification disabled.

## Imports
```python
from examples.trackersplat_paper import Motion, compensate, run_frame, ...
```
