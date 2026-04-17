#!/usr/bin/env python
"""
Reference-guided gaussian cleaning — scale-aware, parameter-light.

Principle: a target gaussian T is "supported" by the reference iff its
k_sigma-footprint overlaps at least one reference gaussian's k_sigma-footprint.
If T has no touching ref neighbour, it is a floater.

Concretely, with R = nearest-reference-gaussian to T in world distance:
    touching = dist(T, R) < (scale(T) + scale(R)) * k_sigma

This naturally handles:
  - big target gaussians: need closer / bigger ref to be supported
  - small valid gaussians near the subject: always supported
  - floaters far from the subject (including top/above): dropped
  - per-frame scene variation: no hardcoded thresholds

Usage:
    python clean_gaussians_by_reference.py \
        --reference clean.ply --target dirty.ply --output out.ply \
        [--bbox_margin 0.05] [--k_sigma 2.0] \
        [--op_ref 0.3] [--op_target 0.1] [--k_neighbors 1]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsplat.io_ply import import_splats
from gsplat.exporter import export_splats


def _activated_opac(o_raw: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-o_raw))


def clean_one(ref_path: str, tgt_path: str, out_path: str, *,
              bbox_margin: float, k_sigma: float,
              op_ref: float, op_target: float,
              k_neighbors: int,
              big_dim_scale_mult: float = 2.0,
              bbox_margin_top: float | None = None) -> dict:
    # Load
    rm, rs, _rq, ro, _, _ = import_splats(ref_path, device='cpu')
    tm, ts, tq, to, tsh0, tshN = import_splats(tgt_path, device='cpu')
    rm_np = rm.numpy(); rs_np = rs.numpy(); ro_np = ro.numpy()
    tm_np = tm.numpy(); ts_np = ts.numpy(); to_np = to.numpy()
    N0 = tm_np.shape[0]

    ro_act = _activated_opac(ro_np)
    to_act = _activated_opac(to_np)
    rs_lin_max = np.exp(np.clip(rs_np, -30, 30)).max(axis=1)    # per-ref scale (world)
    ts_lin_max = np.exp(np.clip(ts_np, -30, 30)).max(axis=1)    # per-target scale

    # Evaluate support for:
    #   (a) visible targets (opacity > op_target)                — standard case
    #   (b) "big-but-dim" targets (dim, but scale >> median)     — hazy cloud floaters
    # Dim-and-small gaussians are ambient structure, left alone.
    target_vis = to_act > op_target
    scale_median = float(np.median(ts_lin_max)) if ts_lin_max.size else 0.0
    big_dim = (ts_lin_max > big_dim_scale_mult * max(scale_median, 1e-8)) & ~target_vis
    needs_support = target_vis | big_dim                         # [Nt]

    # --- Stage A: bbox clip -------------------------------------------------
    bmin = rm_np.min(axis=0) - bbox_margin
    bmax = rm_np.max(axis=0) + bbox_margin
    if bbox_margin_top is not None:
        bmax[2] = rm_np[:, 2].max() + bbox_margin_top
    in_bbox = ((tm_np >= bmin) & (tm_np <= bmax)).all(axis=1)

    # --- Stage B: scale-aware nearest-neighbour support --------------------
    ref_vis = ro_act > op_ref
    ref_pts = rm_np[ref_vis]
    ref_scale_max = rs_lin_max[ref_vis]

    tree = cKDTree(ref_pts)
    k = max(1, int(k_neighbors))
    dists, idxs = tree.query(tm_np, k=k, workers=-1)
    if k == 1:
        dists = dists[:, None]; idxs = idxs[:, None]
    # Per-neighbour touching criterion.
    neigh_ref_scale = ref_scale_max[idxs]                        # [Nt, k]
    touch_radius = (ts_lin_max[:, None] + neigh_ref_scale) * k_sigma
    touching = (dists < touch_radius).any(axis=1)                # at least one neighbour touches

    # Drop if (needs support) AND (outside bbox OR unsupported).
    drop_bbox = needs_support & ~in_bbox
    drop_support = needs_support & in_bbox & ~touching
    drop = drop_bbox | drop_support
    keep = ~drop

    keep_t = torch.from_numpy(keep)
    export_splats(
        means=tm[keep_t], scales=ts[keep_t], quats=tq[keep_t],
        opacities=to[keep_t], sh0=tsh0[keep_t], shN=tshN[keep_t],
        format='ply', save_to=out_path,
    )
    return {
        'in': N0, 'out': int(keep.sum()), 'dropped': int(drop.sum()),
        'bbox_drop': int(drop_bbox.sum()),
        'support_drop': int(drop_support.sum()),
        'ref_vis': int(ref_vis.sum()),
        'tgt_vis': int(target_vis.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--target', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--bbox_margin', type=float, default=0.05)
    ap.add_argument('--k_sigma', type=float, default=2.0,
                    help='k-sigma envelope for gaussian footprint (2.0 = 2 std-dev).')
    ap.add_argument('--op_ref', type=float, default=0.3)
    ap.add_argument('--op_target', type=float, default=0.1)
    ap.add_argument('--k_neighbors', type=int, default=1,
                    help='Number of nearest ref neighbours to check for touching.')
    ap.add_argument('--big_dim_scale_mult', type=float, default=2.0,
                    help='Dim targets with max_scale > this * target_scale_median also need ref support.')
    ap.add_argument('--bbox_margin_top', type=float, default=None,
                    help='Override +Z (top) margin only. Negative crops into clean region.')
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    stats = clean_one(
        a.reference, a.target, a.output,
        bbox_margin=a.bbox_margin, k_sigma=a.k_sigma,
        op_ref=a.op_ref, op_target=a.op_target,
        k_neighbors=a.k_neighbors,
        big_dim_scale_mult=a.big_dim_scale_mult,
        bbox_margin_top=a.bbox_margin_top,
    )
    print(f"{a.target} -> {a.output}")
    print(f"  {stats['in']:,} -> {stats['out']:,}  dropped={stats['dropped']:,} "
          f"(bbox={stats['bbox_drop']:,}  support={stats['support_drop']:,})  "
          f"ref_vis={stats['ref_vis']:,}  tgt_vis={stats['tgt_vis']:,}")


if __name__ == '__main__':
    main()
