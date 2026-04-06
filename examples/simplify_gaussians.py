#!/usr/bin/env python
"""
Simplify static Gaussian splats using NanoGS pairwise merging.

Based on: NanoGS: Training-Free Gaussian Splat Simplification
          (https://arxiv.org/abs/2603.16103)
          (https://github.com/saliteta/NanoGS)

Reads/writes PLY files using gsplat's import_splats / export_splats.

Usage:
    python simplify_gaussians.py --static_ply <input.ply> [-o <output.ply>] [-r 0.5]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsplat.io_ply import import_splats
from gsplat.exporter import export_splats


# ── NanoGS math primitives (from utils/splat_utils.py) ──────────────────


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(B, 4) wxyz → (B, 3, 3)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """(B, 3, 3) → (B, 4) wxyz"""
    M = R.shape[0]
    q = np.empty((M, 4), dtype=np.float32)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mask_tr = tr > 0
    mask_00 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & ~mask_tr
    mask_11 = (R[:, 1, 1] > R[:, 2, 2]) & ~mask_tr & ~mask_00
    mask_22 = ~mask_tr & ~mask_00 & ~mask_11
    if np.any(mask_tr):
        S = np.sqrt(tr[mask_tr] + 1.0) * 2.0
        q[mask_tr, 0] = 0.25 * S
        q[mask_tr, 1] = (R[mask_tr, 2, 1] - R[mask_tr, 1, 2]) / S
        q[mask_tr, 2] = (R[mask_tr, 0, 2] - R[mask_tr, 2, 0]) / S
        q[mask_tr, 3] = (R[mask_tr, 1, 0] - R[mask_tr, 0, 1]) / S
    if np.any(mask_00):
        S = np.sqrt(1.0 + R[mask_00, 0, 0] - R[mask_00, 1, 1] - R[mask_00, 2, 2]) * 2.0
        q[mask_00, 0] = (R[mask_00, 2, 1] - R[mask_00, 1, 2]) / S
        q[mask_00, 1] = 0.25 * S
        q[mask_00, 2] = (R[mask_00, 0, 1] + R[mask_00, 1, 0]) / S
        q[mask_00, 3] = (R[mask_00, 0, 2] + R[mask_00, 2, 0]) / S
    if np.any(mask_11):
        S = np.sqrt(1.0 + R[mask_11, 1, 1] - R[mask_11, 0, 0] - R[mask_11, 2, 2]) * 2.0
        q[mask_11, 0] = (R[mask_11, 0, 2] - R[mask_11, 2, 0]) / S
        q[mask_11, 1] = (R[mask_11, 0, 1] + R[mask_11, 1, 0]) / S
        q[mask_11, 2] = 0.25 * S
        q[mask_11, 3] = (R[mask_11, 1, 2] + R[mask_11, 2, 1]) / S
    if np.any(mask_22):
        S = np.sqrt(1.0 + R[mask_22, 2, 2] - R[mask_22, 0, 0] - R[mask_22, 1, 1]) * 2.0
        q[mask_22, 0] = (R[mask_22, 1, 0] - R[mask_22, 0, 1]) / S
        q[mask_22, 1] = (R[mask_22, 0, 2] + R[mask_22, 2, 0]) / S
        q[mask_22, 2] = (R[mask_22, 1, 2] + R[mask_22, 2, 1]) / S
        q[mask_22, 3] = 0.25 * S
    return _quat_normalize(q)


def _sigma_from_sq(scales: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Build covariance Sigma = R diag(s^2) R^T."""
    R = _quat_to_rotmat(q)
    s2 = (scales * scales).astype(np.float32)
    return np.matmul(R * s2[:, None, :], np.transpose(R, (0, 2, 1)))


def _gauss_logpdf(x, mu, R, invdiag, logdet):
    """log N(x | mu, Sigma) with rotated-diagonal form."""
    log2pi = np.float32(np.log(2.0 * np.pi))
    d = x - mu[:, None, :]
    y = np.matmul(d, R)
    quad = np.sum((y * y) * invdiag[:, None, :], axis=2)
    return -0.5 * (3.0 * log2pi + logdet[:, None] + quad)


# ── NanoGS simplification core ──────────────────────────────────────────


def _knn(means: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(means)
    _, idx = tree.query(means, k=k + 1, workers=-1)
    return idx[:, 1:]


def _undirected_edges(nbr: np.ndarray) -> np.ndarray:
    N, k = nbr.shape
    ii = np.repeat(np.arange(N, dtype=np.int32), k)
    jj = nbr.ravel().astype(np.int32)
    u = np.minimum(ii, jj)
    v = np.maximum(ii, jj)
    mask = u != v
    edges = np.unique(np.stack([u[mask], v[mask]], axis=1), axis=0)
    return edges.astype(np.int32)


def _edge_costs(
    edges: np.ndarray,
    mu: np.ndarray,
    sc: np.ndarray,
    q: np.ndarray,
    op: np.ndarray,
    sh: np.ndarray,
    lam_geo: float,
    lam_sh: float,
    block: int = 100_000,
) -> np.ndarray:
    """KL(mixture || moment-matched merge) + SH L2, per edge."""
    M = edges.shape[0]
    w = np.empty(M, dtype=np.float32)
    eps = np.float32(1e-8)
    log2pi = np.float32(np.log(2.0 * np.pi))
    rng = np.random.default_rng(0)
    Z = rng.standard_normal(size=(1, 3)).astype(np.float32)

    for e0 in tqdm(range(0, M, block), desc="Edge costs"):
        e1 = min(M, e0 + block)
        uv = edges[e0:e1]
        ui, vi = uv[:, 0], uv[:, 1]

        mu_u, sc_u, q_u, op_u = mu[ui], sc[ui], q[ui], op[ui]
        mu_v, sc_v, q_v, op_v = mu[vi], sc[vi], q[vi], op[vi]

        R_u = _quat_to_rotmat(q_u)
        R_v = _quat_to_rotmat(q_v)
        Rt_u = np.transpose(R_u, (0, 2, 1))
        Rt_v = np.transpose(R_v, (0, 2, 1))

        v_u = (sc_u * sc_u + eps).astype(np.float32)
        v_v = (sc_v * sc_v + eps).astype(np.float32)
        invd_u = (1.0 / np.maximum(v_u, 1e-30)).astype(np.float32)
        invd_v = (1.0 / np.maximum(v_v, 1e-30)).astype(np.float32)
        ld_u = np.sum(np.log(np.maximum(v_u, 1e-30)), axis=1).astype(np.float32)
        ld_v = np.sum(np.log(np.maximum(v_v, 1e-30)), axis=1).astype(np.float32)

        # mixture weights
        w_u = (2 * np.pi) ** 1.5 * op_u * np.prod(sc_u, axis=1) + 1e-12
        w_v = (2 * np.pi) ** 1.5 * op_v * np.prod(sc_v, axis=1) + 1e-12
        W = w_u + w_v
        W_s = np.where(W > 0, W, 1.0).astype(np.float32)
        pi = np.clip(w_u / W_s, 1e-12, 1 - 1e-12).astype(np.float32)
        lp_i = np.log(pi).astype(np.float32)
        lp_j = np.log(1.0 - pi).astype(np.float32)

        mu_u32 = mu_u.astype(np.float32)
        mu_v32 = mu_v.astype(np.float32)
        mu_m = pi[:, None] * mu_u32 + (1 - pi)[:, None] * mu_v32

        di = mu_u32 - mu_m
        dj = mu_v32 - mu_m
        Sig_u = np.matmul(R_u * v_u[:, None, :], Rt_u)
        Sig_v = np.matmul(R_v * v_v[:, None, :], Rt_v)
        Sig_m = pi[:, None, None] * (Sig_u + di[:, :, None] * di[:, None, :]) + (
            1 - pi
        )[:, None, None] * (Sig_v + dj[:, :, None] * dj[:, None, :])
        I3 = np.eye(3, dtype=np.float32)[None, :, :]
        Sig_m = 0.5 * (Sig_m + np.transpose(Sig_m, (0, 2, 1))) + eps * I3

        _, ld_m = np.linalg.slogdet(Sig_m)
        ld_m = ld_m.astype(np.float32)

        E_neg = 0.5 * (3.0 * log2pi + ld_m + 3.0)

        # MC samples (1 sample, deterministic)
        std_u = np.sqrt(np.maximum(v_u, 0)).astype(np.float32)
        std_v = np.sqrt(np.maximum(v_v, 0)).astype(np.float32)
        x_u = mu_u32[:, None, :] + np.matmul(
            Z[None, :, :] * std_u[:, None, :], Rt_u
        )
        x_v = mu_v32[:, None, :] + np.matmul(
            Z[None, :, :] * std_v[:, None, :], Rt_v
        )

        lNu_u = _gauss_logpdf(x_u, mu_u32, R_u, invd_u, ld_u)
        lNv_u = _gauss_logpdf(x_u, mu_v32, R_v, invd_v, ld_v)
        lp_u = np.logaddexp(lp_i[:, None] + lNu_u, lp_j[:, None] + lNv_u)
        Eu = np.mean(lp_u, axis=1).astype(np.float32)

        lNu_v = _gauss_logpdf(x_v, mu_u32, R_u, invd_u, ld_u)
        lNv_v = _gauss_logpdf(x_v, mu_v32, R_v, invd_v, ld_v)
        lp_v = np.logaddexp(lp_i[:, None] + lNu_v, lp_j[:, None] + lNv_v)
        Ev = np.mean(lp_v, axis=1).astype(np.float32)

        geo = (pi * Eu + (1 - pi) * Ev) + E_neg

        # SH L2
        if sh.shape[1] > 0:
            diff = sh[ui].astype(np.float32) - sh[vi].astype(np.float32)
            c_sh = np.sum(diff * diff, axis=1).astype(np.float32)
        else:
            c_sh = np.zeros_like(geo)

        w[e0:e1] = (lam_geo * geo + lam_sh * c_sh).astype(np.float32)

    return w


def _greedy_pairs(
    edges: np.ndarray, w: np.ndarray, N: int, P: int
) -> np.ndarray:
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)
    valid = np.isfinite(w)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)
    idx = np.nonzero(valid)[0]
    order = idx[np.argsort(w[idx], kind="mergesort")]
    used = np.zeros(N, dtype=bool)
    pairs = []
    for ei in order:
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        pairs.append((u, v))
        if P is not None and len(pairs) >= P:
            break
    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


def _merge_pairs(
    mu: np.ndarray,
    sc: np.ndarray,
    q: np.ndarray,
    op: np.ndarray,
    sh: np.ndarray,
    pairs: np.ndarray,
) -> tuple:
    """Moment-matched merge of selected pairs."""
    if pairs.shape[0] == 0:
        return mu, sc, q, op, sh

    i, j = pairs[:, 0], pairs[:, 1]

    Sig_i = _sigma_from_sq(sc[i], q[i])
    Sig_j = _sigma_from_sq(sc[j], q[j])

    w_i = (2 * np.pi) ** 1.5 * op[i] * np.prod(sc[i], axis=1) + 1e-12
    w_j = (2 * np.pi) ** 1.5 * op[j] * np.prod(sc[j], axis=1) + 1e-12
    W = np.maximum(w_i + w_j, 1e-12)

    mu_m = (w_i[:, None] * mu[i] + w_j[:, None] * mu[j]) / W[:, None]

    di = (mu[i] - mu_m).astype(np.float32)
    dj = (mu[j] - mu_m).astype(np.float32)
    Sig_m = (
        w_i[:, None, None] * (Sig_i + di[:, :, None] * di[:, None, :])
        + w_j[:, None, None] * (Sig_j + dj[:, :, None] * dj[:, None, :])
    ) / W[:, None, None]
    I3 = np.eye(3, dtype=np.float32)[None, :, :]
    Sig_m = 0.5 * (Sig_m + np.transpose(Sig_m, (0, 2, 1))) + 1e-8 * I3

    evals, evecs = np.linalg.eigh(Sig_m)
    evals = np.maximum(evals, 1e-18).astype(np.float32)
    # sort descending
    order = np.argsort(evals, axis=1)[:, ::-1]
    evals = np.take_along_axis(evals, order, axis=1)
    evecs = np.take_along_axis(evecs, order[:, None, :], axis=2)
    # enforce right-handed
    flip = np.linalg.det(evecs) < 0
    if np.any(flip):
        evecs[flip, :, 2] *= -1.0

    sc_m = np.sqrt(evals).astype(np.float32)
    q_m = _rotmat_to_quat(evecs.astype(np.float32))
    op_m = op[i] + op[j] - op[i] * op[j]

    if sh.shape[1] > 0:
        sh_m = ((w_i[:, None] * sh[i] + w_j[:, None] * sh[j]) / W[:, None]).astype(
            np.float32
        )
    else:
        sh_m = sh[i]

    # keep un-merged, append merged
    used = np.zeros(mu.shape[0], dtype=bool)
    used[i] = True
    used[j] = True
    keep = np.nonzero(~used)[0]

    return (
        np.concatenate([mu[keep], mu_m.astype(np.float32)]),
        np.concatenate([sc[keep], sc_m]),
        np.concatenate([q[keep], q_m]),
        np.concatenate([op[keep], op_m]),
        np.concatenate([sh[keep], sh_m]),
    )


def _prune_opacity(mu, sc, q, op, sh, threshold=0.1):
    threshold = min(threshold, float(np.median(op)))
    print(
        f"Pruning opacity < {threshold:.4f}  "
        f"(mean={np.mean(op):.4f}, median={np.median(op):.4f})"
    )
    keep = op >= threshold
    return mu[keep], sc[keep], q[keep], op[keep], sh[keep]


def simplify(
    mu: np.ndarray,
    sc: np.ndarray,
    q: np.ndarray,
    op: np.ndarray,
    sh: np.ndarray,
    ratio: float = 0.5,
    k: int = 16,
    merge_cap: float = 0.5,
    opacity_threshold: float = 0.1,
    lam_geo: float = 1.0,
    lam_sh: float = 1.0,
) -> tuple:
    N0 = mu.shape[0]
    target = max(int(math.ceil(N0 * ratio)), 1)
    print(f"Initial: {N0}, target: {target}")

    mu, sc, q, op, sh = _prune_opacity(mu, sc, q, op, sh, opacity_threshold)
    print(f"After pruning: {mu.shape[0]}")

    p_cap = max(1, int(merge_cap * N0))
    iteration = 0

    while mu.shape[0] > target:
        N = mu.shape[0]
        print(f"\nPass {iteration + 1}: {N} splats")

        k_eff = min(max(1, k), max(1, N - 1))
        nbr = _knn(mu, k=k_eff)
        edges = _undirected_edges(nbr)
        w = _edge_costs(edges, mu, sc, q, op, sh, lam_geo, lam_sh)

        merges_needed = N - target
        P = min(merges_needed, p_cap)
        pairs = _greedy_pairs(edges, w, N, P)
        print(f"  edges={edges.shape[0]}, pairs={pairs.shape[0]}, need={merges_needed}")

        if pairs.shape[0] == 0:
            print("  No valid pairs — stopping early.")
            break

        mu, sc, q, op, sh = _merge_pairs(mu, sc, q, op, sh, pairs)
        iteration += 1

    op = np.clip(op, 0.0, 1.0).astype(np.float32)
    print(f"\nFinal: {mu.shape[0]} splats")
    return mu, sc, q, op, sh


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Simplify static Gaussian splats via NanoGS pairwise merging."
    )
    ap.add_argument("--static_ply", required=True, help="Input static PLY (outside.ply)")
    ap.add_argument("-o", "--output", default=None, help="Output PLY path")
    ap.add_argument(
        "-r", "--ratio", type=float, default=0.5, help="Fraction of splats to keep (0,1)"
    )
    ap.add_argument("--k", type=int, default=16, help="KNN neighbourhood size")
    ap.add_argument(
        "--merge_cap", type=float, default=0.5,
        help="Max merges per pass as fraction of original count",
    )
    ap.add_argument(
        "--opacity_threshold", type=float, default=0.1,
        help="Prune splats below this opacity before merging",
    )
    ap.add_argument("--lam_geo", type=float, default=1.0, help="Geometric cost weight")
    ap.add_argument("--lam_sh", type=float, default=1.0, help="SH cost weight")
    args = ap.parse_args()

    assert 0.0 < args.ratio < 1.0, "--ratio must be in (0, 1)"
    merge_cap = max(0.01, min(0.5, args.merge_cap))

    if args.output is None:
        base, ext = os.path.splitext(args.static_ply)
        tag = f"{args.ratio}".rstrip("0").rstrip(".")
        args.output = f"{base}_simplified_{tag}.ply"

    # ── 1. Load PLY (raw space: log-scales, logit-opacities) ──
    print(f"Loading: {args.static_ply}")
    means, scales, quats, opacities, sh0, shN = import_splats(
        args.static_ply, device="cpu"
    )
    N = means.shape[0]
    print(f"Loaded {N} splats  (sh0={sh0.shape}, shN={shN.shape})")

    # ── 2. Activate to linear space (NanoGS convention) ──
    mu = means.numpy().astype(np.float32)
    sc = np.exp(np.clip(scales.numpy(), -30, 30)).astype(np.float32)
    q = _quat_normalize(quats.numpy().astype(np.float32))
    op = (1.0 / (1.0 + np.exp(-opacities.numpy()))).astype(np.float32)

    # Flatten SH: sh0 (N,1,3) + shN (N,K,3) → (N, 3+3K)
    sh0_np = sh0.numpy().reshape(N, 3).astype(np.float32)
    shN_np = shN.numpy().reshape(N, -1).astype(np.float32)  # (N, 3K) or (N, 0)
    n_shN = shN_np.shape[1]
    if n_shN > 0:
        sh_flat = np.concatenate([sh0_np, shN_np], axis=1)
    else:
        sh_flat = sh0_np

    # ── 3. Simplify ──
    mu, sc, q, op, sh_flat = simplify(
        mu,
        sc,
        q,
        op,
        sh_flat,
        ratio=args.ratio,
        k=args.k,
        merge_cap=merge_cap,
        opacity_threshold=args.opacity_threshold,
        lam_geo=args.lam_geo,
        lam_sh=args.lam_sh,
    )

    # ── 4. Deactivate back to raw space ──
    N_out = mu.shape[0]
    means_out = torch.from_numpy(mu)
    scales_out = torch.from_numpy(np.log(np.maximum(sc, 1e-12)).astype(np.float32))
    quats_out = torch.from_numpy(q)
    op_c = np.clip(op, 1e-6, 1 - 1e-6)
    opacities_out = torch.from_numpy(np.log(op_c / (1 - op_c)).astype(np.float32))

    # Un-flatten SH back to (N,1,3) + (N,K,3)
    sh0_out = torch.from_numpy(sh_flat[:, :3]).reshape(N_out, 1, 3)
    if n_shN > 0:
        K = n_shN // 3
        shN_out = torch.from_numpy(sh_flat[:, 3:]).reshape(N_out, K, 3)
    else:
        shN_out = torch.zeros(N_out, 0, 3)

    # ── 5. Save ──
    print(f"\nSaving simplified PLY: {args.output}")
    export_splats(
        means=means_out,
        scales=scales_out,
        quats=quats_out,
        opacities=opacities_out,
        sh0=sh0_out,
        shN=shN_out,
        format="ply",
        save_to=args.output,
    )
    print("Done!")


if __name__ == "__main__":
    main()
