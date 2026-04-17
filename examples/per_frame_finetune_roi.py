#!/usr/bin/env python
"""
Per-frame ROI-based Gaussian finetune.

- Load a merged per-frame PLY (single 3DGS scene for one frame).
- Split gaussians by ROI AABB:
    * inside  → trainable nn.ParameterDict
    * outside → frozen plain tensors (no grad, no strategy)
- Train with gentle densification (DefaultStrategy) on the trainable subset only.
- Save merged PLY (trainable + frozen) at the end.

Usage:
    python per_frame_finetune_roi.py \
        --data_dir  /data/.../yehe_tech/undistorted \
        --per_frame_ply /data/.../ply_sequence_merged_35000_merged/0149.ply \
        --roi_bounds_path /data/.../static_dynamic_output/roi_bounds.npy \
        --result_dir /tmp/yehe_roi_f149 \
        --frame_idx 149 --max_steps 5000 --data_factor 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from fused_ssim import fused_ssim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import imageio.v2 as iio
from plyfile import PlyData, PlyElement

from datasets.colmap import Parser
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy


# ---------------------------------------------------------------------------
# PLY I/O (matches simple_trainer_perframe_masked conventions)
# ---------------------------------------------------------------------------

def load_ply(path: str, device: str = "cuda") -> Dict[str, Tensor]:
    pd = PlyData.read(path)
    v = pd["vertex"].data
    N = len(v)
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T
    f_dc = sorted([n for n in v.dtype.names if n.startswith("f_dc_")])
    f_rest = sorted([n for n in v.dtype.names if n.startswith("f_rest_")])
    sc = sorted([n for n in v.dtype.names if n.startswith("scale_")])
    rt = sorted([n for n in v.dtype.names if n.startswith("rot_")])
    dc = np.vstack([v[n] for n in f_dc]).T if f_dc else np.zeros((N, 3))
    rest = np.vstack([v[n] for n in f_rest]).T if f_rest else None
    scales = np.vstack([v[n] for n in sc]).T
    quats = np.vstack([v[n] for n in rt]).T
    opacities = np.array(v["opacity"])
    sh0 = dc.reshape(-1, 3, 1).transpose(0, 2, 1)
    if rest is not None:
        rest_k = len(f_rest) // 3
        shN = rest.reshape(-1, 3, rest_k).transpose(0, 2, 1)
    else:
        shN = np.zeros((N, 0, 3))

    out = {
        "means": torch.tensor(xyz, dtype=torch.float32, device=device),
        "scales": torch.tensor(scales, dtype=torch.float32, device=device),
        "quats": torch.tensor(quats, dtype=torch.float32, device=device),
        "opacities": torch.tensor(opacities, dtype=torch.float32, device=device),
        "sh0": torch.tensor(sh0, dtype=torch.float32, device=device),
        "shN": torch.tensor(shN, dtype=torch.float32, device=device),
    }
    print(f"Loaded {N:,} Gaussians from {path}  (sh0={sh0.shape}, shN={shN.shape})")
    return out


def save_ply(tensors: Dict[str, Tensor], path: str, ref_ply_path: str):
    """Save merged PLY matching reference dtype."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ref = PlyData.read(ref_ply_path)
    ref_dtype = ref["vertex"].data.dtype
    N = tensors["means"].shape[0]
    data = np.empty(N, dtype=ref_dtype)
    means = tensors["means"].detach().cpu().numpy()
    data["x"], data["y"], data["z"] = means[:, 0], means[:, 1], means[:, 2]
    if "nx" in ref_dtype.names:
        data["nx"] = data["ny"] = data["nz"] = 0.0
    sh0 = tensors["sh0"].detach().cpu().numpy().reshape(N, -1)
    for i, name in enumerate(n for n in ref_dtype.names if n.startswith("f_dc_")):
        data[name] = sh0[:, i] if i < sh0.shape[1] else 0.0
    shN = tensors["shN"].detach().cpu().numpy().transpose(0, 2, 1).reshape(N, -1)
    for i, name in enumerate(n for n in ref_dtype.names if n.startswith("f_rest_")):
        data[name] = shN[:, i] if i < shN.shape[1] else 0.0
    data["opacity"] = tensors["opacities"].detach().cpu().numpy()
    scales = tensors["scales"].detach().cpu().numpy()
    for i, name in enumerate(sorted(n for n in ref_dtype.names if n.startswith("scale_"))):
        data[name] = scales[:, i]
    quats = tensors["quats"].detach().cpu().numpy()
    for i, name in enumerate(sorted(n for n in ref_dtype.names if n.startswith("rot_"))):
        data[name] = quats[:, i]
    PlyData([PlyElement.describe(data, "vertex")]).write(path)
    print(f"Saved {N:,} Gaussians to {path}")


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(image_dir, cam_name, frame_idx, factor, device):
    for ext in (".jpg", ".png"):
        p = os.path.join(image_dir, cam_name, f"{frame_idx:06d}{ext}")
        if os.path.exists(p):
            img = iio.imread(p)[..., :3].astype(np.float32) / 255.0
            if factor > 1:
                from PIL import Image as PILImage
                h, w = img.shape[:2]
                img = np.array(PILImage.fromarray((img * 255).astype(np.uint8)).resize(
                    (w // factor, h // factor), PILImage.BICUBIC)).astype(np.float32) / 255.0
            return torch.from_numpy(img).float().to(device)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--per_frame_ply", default=None,
                    help="Merged PLY for this frame. Mutually exclusive with --dynamic_ply_path.")
    ap.add_argument("--dynamic_ply_path", default=None,
                    help="Dynamic (foreground) PLY. Use with --static_ply_path to skip pre-merging.")
    ap.add_argument("--static_ply_path", default=None,
                    help="Static (background) PLY, reused across frames. Concatenated internally.")
    ap.add_argument("--roi_bounds_path", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--frame_idx", type=int, required=True)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--data_factor", type=int, default=2)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--densify_start", type=int, default=800)
    ap.add_argument("--densify_every", type=int, default=600)
    ap.add_argument("--densify_stop", type=int, default=3000)
    ap.add_argument("--prune_opa", type=float, default=0.01,
                    help="Prune trainable gaussians with opacity below this (frozen not affected)")
    ap.add_argument("--densify_grad_threshold", type=float, default=0.0002,
                    help="Higher = less densification (fewer floaters)")
    ap.add_argument("--ssim_lambda", type=float, default=0.2)
    ap.add_argument("--scale_reg", type=float, default=0.01,
                    help="Penalty on max gaussian scale (mean) to avoid runaway growth")
    ap.add_argument("--bright_penalty", type=float, default=0.5,
                    help="Penalty weight for bright × opaque trainable gaussians. "
                         "Drives either opacity→0 (→ pruned) or brightness→below threshold.")
    ap.add_argument("--bright_threshold", type=float, default=0.9,
                    help="RGB threshold (0..1). Gaussians with RGB above this get penalized.")
    ap.add_argument("--max_scale_clamp", type=float, default=0.3,
                    help="Hard clamp on linear max scale per gaussian (world units)")
    ap.add_argument("--pad_sh_to", type=int, default=0,
                    help="Pad PLY SH to this degree with zero-init shN. 0 = no padding "
                         "(avoid view-dependent overfitting that causes shiny white floaters)")
    ap.add_argument("--lr_means", type=float, default=1.6e-4)
    ap.add_argument("--lr_scales", type=float, default=1e-3)
    ap.add_argument("--lr_quats", type=float, default=1e-3)
    ap.add_argument("--lr_opacities", type=float, default=5e-2)
    ap.add_argument("--lr_sh0", type=float, default=2.5e-3)
    ap.add_argument("--lr_shN", type=float, default=1.25e-4)
    ap.add_argument("--near_plane", type=float, default=0.01)
    ap.add_argument("--far_plane", type=float, default=1e10)
    ap.add_argument("--save_dynamic_only", action="store_true",
                    help="Save only the trainable (in-ROI) subset, skip frozen.")
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)
    device = "cuda"
    writer = SummaryWriter(os.path.join(args.result_dir, "tb"))

    # --- Parser + poses ---
    parser = Parser(
        data_dir=args.data_dir, factor=args.data_factor,
        normalize=False, test_every=0, frame_num=args.frame_idx,
    )
    cam_names = []
    seen = set()
    for n in parser.image_names:
        c = os.path.dirname(n)
        if c not in seen:
            seen.add(c); cam_names.append(c)
    num_cams = len(cam_names)
    all_camtoworlds = torch.from_numpy(parser.camtoworlds).float().to(device)
    all_viewmats = torch.linalg.inv(all_camtoworlds)
    all_Ks = torch.stack([
        torch.from_numpy(parser.Ks_dict[parser.camera_ids[i]].copy()).float()
        for i in range(num_cams)
    ]).to(device)
    scene_scale = parser.scene_scale * 1.1
    print(f"[Frame {args.frame_idx}] {num_cams} cameras  scene_scale={scene_scale:.3f}")

    # --- Preload images ---
    image_dir = os.path.join(args.data_dir, "images")
    preloaded = []
    for c in cam_names:
        img = load_image(image_dir, c, args.frame_idx, args.data_factor, device)
        preloaded.append(img)
    n_imgs = sum(im is not None for im in preloaded)
    print(f"Preloaded {n_imgs}/{num_cams} GT images")

    # --- Load per-frame PLY, split by ROI ---
    use_split = args.dynamic_ply_path is not None
    assert (args.per_frame_ply is None) != (not use_split), (
        "Provide exactly one of --per_frame_ply OR (--dynamic_ply_path [+ --static_ply_path])."
    )
    if use_split:
        dyn = load_ply(args.dynamic_ply_path, device=device)
        if args.static_ply_path is not None:
            sta = load_ply(args.static_ply_path, device=device)
            # Pad shorter shN with zeros so we can concat
            kd, ks = dyn["shN"].shape[1], sta["shN"].shape[1]
            K = max(kd, ks)
            def _pad(x, K):
                if x.shape[1] == K: return x
                p = torch.zeros(x.shape[0], K - x.shape[1], 3, device=x.device, dtype=x.dtype)
                return torch.cat([x, p], dim=1)
            dyn["shN"] = _pad(dyn["shN"], K)
            sta["shN"] = _pad(sta["shN"], K)
            all_gs = {k: torch.cat([dyn[k], sta[k]], dim=0) for k in dyn}
            print(f"Concatenated dyn+static: {all_gs['means'].shape[0]:,} Gaussians")
        else:
            all_gs = dyn
        ref_ply_for_save = args.dynamic_ply_path
    else:
        all_gs = load_ply(args.per_frame_ply, device=device)
        ref_ply_for_save = args.per_frame_ply
    # Detect current SH degree.
    total_sh = all_gs["sh0"].shape[1] + all_gs["shN"].shape[1]
    detected_sh = int(round(total_sh ** 0.5)) - 1
    # Pad shN with zeros so rasterizer can learn higher-degree SH (view-dependent color).
    target_sh = max(detected_sh, args.pad_sh_to)
    target_K = (target_sh + 1) ** 2                          # total coefs
    target_shN_K = target_K - 1                              # coefs stored in shN
    cur_shN_K = all_gs["shN"].shape[1]
    if target_shN_K > cur_shN_K:
        N = all_gs["means"].shape[0]
        pad = torch.zeros(N, target_shN_K - cur_shN_K, 3, device=device)
        all_gs["shN"] = torch.cat([all_gs["shN"], pad], dim=1)
        print(f"[sh_degree] PLY had deg {detected_sh} ({total_sh} coefs); "
              f"padded to deg {target_sh} ({target_K} coefs, shN={target_shN_K}).")
    args.sh_degree = target_sh
    roi = np.load(args.roi_bounds_path)
    roi_min = torch.from_numpy(roi[0]).float().to(device)
    roi_max = torch.from_numpy(roi[1]).float().to(device)
    mu = all_gs["means"]
    in_roi = ((mu >= roi_min) & (mu <= roi_max)).all(dim=-1)
    n_in = int(in_roi.sum()); n_out = int((~in_roi).sum())
    print(f"ROI split: trainable={n_in:,}  frozen={n_out:,}  (ROI={roi_min.tolist()}→{roi_max.tolist()})")

    # Trainable = nn.Parameter; Frozen = plain detached tensors
    trainable = nn.ParameterDict({
        k: nn.Parameter(v[in_roi].clone()) for k, v in all_gs.items()
    }).to(device)
    frozen = {k: v[~in_roi].detach().clone() for k, v in all_gs.items()}

    # --- Optimizers ---
    param_lrs = [
        ("means", args.lr_means * scene_scale),
        ("scales", args.lr_scales), ("quats", args.lr_quats),
        ("opacities", args.lr_opacities),
        ("sh0", args.lr_sh0), ("shN", args.lr_shN),
    ]
    optimizers = {
        n: torch.optim.Adam([{"params": trainable[n], "lr": lr, "name": n}],
                            eps=1e-15, betas=(0.9, 0.999))
        for n, lr in param_lrs
    }

    # --- Strategy (gentle densify, no pruning, no opa reset) ---
    strategy = DefaultStrategy(
        verbose=False,
        refine_start_iter=args.densify_start,
        refine_stop_iter=args.densify_stop,
        refine_every=args.densify_every,
        reset_every=10_000_000,             # disable opacity reset
        prune_opa=args.prune_opa,           # prunes floaters in trainable only (frozen outside strategy)
        grow_grad2d=args.densify_grad_threshold,
    )
    strategy.check_sanity(trainable, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # Means LR schedule (exp decay 1→0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
    )

    # --- Training loop ---
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    pbar = tqdm.trange(args.max_steps, desc=f"Frame {args.frame_idx}")
    tic = time.time()

    for step in pbar:
        ci = torch.randint(0, num_cams, (1,)).item()
        gt = preloaded[ci]
        if gt is None:
            continue
        H, W = gt.shape[:2]
        gt_b = gt.unsqueeze(0)

        t_means = trainable["means"]
        t_quats = trainable["quats"]
        t_scales = torch.exp(trainable["scales"])
        t_opas = torch.sigmoid(trainable["opacities"])
        t_colors = torch.cat([trainable["sh0"], trainable["shN"]], 1)

        # Concat trainable + frozen (frozen has no grad path)
        means = torch.cat([t_means, frozen["means"]], 0)
        quats = torch.cat([t_quats, frozen["quats"]], 0)
        scales = torch.cat([t_scales, torch.exp(frozen["scales"])], 0)
        opacities = torch.cat([t_opas, torch.sigmoid(frozen["opacities"])], 0)
        colors = torch.cat([t_colors, torch.cat([frozen["sh0"], frozen["shN"]], 1)], 0)

        renders, alphas, info = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=all_viewmats[ci:ci + 1], Ks=all_Ks[ci:ci + 1],
            width=W, height=H, sh_degree=args.sh_degree,
            near_plane=args.near_plane, far_plane=args.far_plane,
            packed=True,
        )
        rendered = renders[..., :3]

        l1 = F.l1_loss(rendered, gt_b)
        ssim = 1.0 - fused_ssim(
            rendered.permute(0, 3, 1, 2), gt_b.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1 * (1 - args.ssim_lambda) + ssim * args.ssim_lambda
        # Scale regularization: penalize mean of max-scale on trainable gaussians only.
        if args.scale_reg > 0:
            loss = loss + args.scale_reg * t_scales.max(dim=-1).values.mean()
        # Bright-opaque penalty: discourage only saturated shiny-white gaussians.
        # RGB = sh0 * C0 + 0.5. Penalize when the MIN of all three channels is
        # above threshold (true white, not just bright wall color).
        # penalty = opacity * relu(rgb_min - threshold)
        if args.bright_penalty > 0:
            C0 = 0.28209479177387814
            rgb = trainable["sh0"].squeeze(1) * C0 + 0.5     # [N, 3]
            rgb_min = rgb.min(dim=-1).values                 # [N] — only true-white scores high
            bright = (rgb_min - args.bright_threshold).clamp(min=0.0)
            bright_loss = (t_opas.squeeze(-1) * bright).mean()
            loss = loss + args.bright_penalty * bright_loss

        # Strategy pre-backward
        strategy.step_pre_backward(
            params=trainable, optimizers=optimizers, state=state,
            step=step, info=info,
        )
        loss.backward()

        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        scheduler.step()

        # Hard clamp on max scale (log-space). Safety net: no single trainable
        # gaussian may grow past max_scale_clamp meters.
        if args.max_scale_clamp > 0:
            with torch.no_grad():
                trainable["scales"].clamp_(max=float(np.log(args.max_scale_clamp)))

        # Strategy post-backward — filter info to trainable subset.
        # Indexing means2d creates a new tensor that loses .grad / .absgrad;
        # manually re-attach the matching slice so the strategy can read it.
        if "gaussian_ids" in info:
            N_train = trainable["means"].shape[0]
            m = info["gaussian_ids"] < N_train
            filt = {k: v for k, v in info.items()}
            for k in ("gaussian_ids", "means2d", "radii", "depths", "conics", "opacities"):
                if k in filt and filt[k] is not None:
                    filt[k] = filt[k][m]
            orig_m2d = info["means2d"]
            if orig_m2d.grad is not None:
                filt["means2d"].grad = orig_m2d.grad[m].clone()
            if hasattr(orig_m2d, "absgrad") and orig_m2d.absgrad is not None:
                filt["means2d"].absgrad = orig_m2d.absgrad[m].clone()
        else:
            filt = info
        strategy.step_post_backward(
            params=trainable, optimizers=optimizers, state=state,
            step=step, info=filt, packed=True,
        )

        # Logging
        if step % 50 == 0:
            with torch.no_grad():
                psnr = psnr_metric(rendered.permute(0, 3, 1, 2), gt_b.permute(0, 3, 1, 2))
            N_tr = trainable["means"].shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", l1=f"{l1.item():.4f}",
                              psnr=f"{psnr.item():.2f}", Ntr=N_tr)
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/l1", l1.item(), step)
            writer.add_scalar("train/psnr", psnr.item(), step)
            writer.add_scalar("train/num_trainable", N_tr, step)

        if step % 500 == 0 or step == args.max_steps - 1:
            with torch.no_grad():
                canvas = torch.cat([gt_b[0], rendered[0].clamp(0, 1)], dim=1)
                writer.add_image("gt_vs_render", canvas.permute(2, 0, 1), step)

    writer.flush(); writer.close()
    elapsed = time.time() - tic
    N_tr = trainable["means"].shape[0]
    print(f"\nTraining done in {elapsed:.1f}s. Trainable: {N_tr:,}  Frozen: {frozen['means'].shape[0]:,}")

    # --- Save output PLY ---
    if args.save_dynamic_only:
        out = {k: trainable[k].detach() for k in
               ["means", "scales", "quats", "opacities", "sh0", "shN"]}
        out_ply = os.path.join(args.result_dir, f"frame_{args.frame_idx:04d}_dynamic.ply")
        save_ply(out, out_ply, ref_ply_path=ref_ply_for_save)
        print(f"Dynamic-only PLY -> {out_ply}")
    else:
        merged = {k: torch.cat([trainable[k].detach(), frozen[k]], 0) for k in
                  ["means", "scales", "quats", "opacities", "sh0", "shN"]}
        out_ply = os.path.join(args.result_dir, f"frame_{args.frame_idx:04d}_finetuned.ply")
        save_ply(merged, out_ply, ref_ply_path=ref_ply_for_save)
        print(f"Merged PLY -> {out_ply}")


if __name__ == "__main__":
    main()
