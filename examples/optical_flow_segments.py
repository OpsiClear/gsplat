"""
Optical Flow Peak Detection & Video Segmentation using RAFT.

For a single camera's frame sequence, computes optical flow magnitude
between consecutive frames, finds the top-K peaks (with minimum separation),
and partitions the video into segments where each peak marks the START
of a new segment.

Segments: [0, peak0-1], [peak0, peak1-1], ..., [last_peak, last_frame]

Usage:
    # Single camera
    python optical_flow_segments.py \
        --image-dir /path/to/images/002-002 \
        --num-peaks 30

    # Multiple cameras (random 5 from a parent directory)
    python optical_flow_segments.py \
        --parent-dir /path/to/images \
        --num-cameras 5 \
        --num-peaks 30
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # server — no display
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def load_frames(image_dir: str):
    """Load sorted frame paths from a directory."""
    exts = {".jpg", ".jpeg", ".png"}
    paths = sorted([
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in exts
    ])
    return paths


def compute_flow_magnitudes(frame_paths, model, transforms, resize,
                            device="cuda", batch_size=1):
    """Run RAFT on consecutive frame pairs, return per-pair mean flow magnitude."""
    magnitudes = []
    n = len(frame_paths)

    with torch.no_grad():
        for i in range(n - 1):
            img1 = F.to_tensor(Image.open(frame_paths[i]).convert("RGB")).unsqueeze(0)
            img2 = F.to_tensor(Image.open(frame_paths[i + 1]).convert("RGB")).unsqueeze(0)

            img1 = resize(img1).to(device)
            img2 = resize(img2).to(device)

            img1_t, img2_t = transforms(img1, img2)
            flow = model(img1_t, img2_t)[-1]  # [1, 2, H, W]

            mag = torch.sqrt(flow[0, 0] ** 2 + flow[0, 1] ** 2)  # [H, W]
            magnitudes.append(float(mag.mean().cpu()))

            if (i + 1) % 100 == 0 or i == n - 2:
                print(f"    {i + 1}/{n - 1} pairs")

    return np.array(magnitudes)


def find_peaks(magnitudes, num_peaks=30, min_separation=10):
    """Find top-K peaks in magnitude signal with minimum separation."""
    sorted_indices = np.argsort(magnitudes)[::-1]

    peaks = []
    for idx in sorted_indices:
        if len(peaks) >= num_peaks:
            break
        too_close = any(abs(int(idx) - p) < min_separation for p in peaks)
        if not too_close:
            peaks.append(int(idx))

    peaks.sort()
    return peaks


def partition_segments(peaks, num_frames):
    """Partition frames into segments where each peak is the START of a new segment.

    Segments: [0, peak0-1], [peak0, peak1-1], ..., [last_peak, num_frames-1]
    """
    segments = []

    # First segment: frame 0 to peak0-1
    if peaks[0] > 0:
        segments.append((0, peaks[0] - 1))

    # Middle segments: each peak to next_peak-1
    for i in range(len(peaks) - 1):
        segments.append((peaks[i], peaks[i + 1] - 1))

    # Last segment: last peak to end
    segments.append((peaks[-1], num_frames - 1))

    return segments


def process_camera(cam_name, image_dir, output_dir, model, transforms, resize,
                   num_peaks, min_separation, device, magnitudes=None):
    """Process a single camera: compute flow, find peaks, save results."""
    print(f"\n{'='*70}")
    print(f"Camera: {cam_name}")
    print(f"{'='*70}")

    frame_paths = load_frames(image_dir)
    num_frames = len(frame_paths)
    print(f"  Frames: {num_frames}")

    cam_output = os.path.join(output_dir, cam_name)
    os.makedirs(cam_output, exist_ok=True)

    # Compute or load magnitudes
    mag_path = os.path.join(cam_output, "flow_magnitudes.npy")
    if magnitudes is not None:
        np.save(mag_path, magnitudes)
        print(f"  Using pre-computed magnitudes")
    elif os.path.exists(mag_path):
        magnitudes = np.load(mag_path)
        print(f"  Loaded cached magnitudes from {mag_path}")
    else:
        print(f"  Computing optical flow...")
        magnitudes = compute_flow_magnitudes(
            frame_paths, model, transforms, resize, device=device
        )
        np.save(mag_path, magnitudes)
        print(f"  Saved magnitudes to {mag_path}")

    # Find peaks
    peaks = find_peaks(magnitudes, num_peaks=num_peaks,
                       min_separation=min_separation)

    # Partition into segments (peak = start of segment)
    segments = partition_segments(peaks, num_frames)

    # --- Print results ---
    print(f"\n  Peaks: {len(peaks)}, Segments: {len(segments)}")
    print(f"\n  {'Seg':>4} | {'Start Frame':>12} | {'End Frame':>12} | {'Start':>6} → {'End':>6} | {'Length':>6} | {'Peak Mag':>10}")
    print(f"  {'-'*76}")
    for i, (start, end) in enumerate(segments):
        length = end - start + 1
        start_name = frame_paths[start].name
        end_name = frame_paths[end].name
        # Peak mag is the magnitude at the start of the segment (except first segment)
        if start > 0 and start - 1 < len(magnitudes):
            peak_mag = f"{magnitudes[start]:.3f}"
        else:
            peak_mag = "—"
        print(f"  {i:4d} | {start_name:>12} | {end_name:>12} | {start:6d} → {end:6d} | {length:6d} | {peak_mag:>10}")

    # --- Save text report ---
    report_path = os.path.join(cam_output, "segments.txt")
    with open(report_path, "w") as f:
        f.write(f"Optical Flow Segmentation — Camera {cam_name}\n")
        f.write(f"Source: {image_dir}\n")
        f.write(f"Total frames: {num_frames}\n")
        f.write(f"Peaks: {len(peaks)} (min separation: {min_separation})\n")
        f.write(f"Segments: {len(segments)}\n")
        f.write(f"Peak indices: {peaks}\n\n")
        f.write(f"{'Seg':>4} | {'Start Frame':>12} | {'End Frame':>12} | {'Start':>6} → {'End':>6} | {'Length':>6} | {'Peak Mag':>10}\n")
        f.write("-" * 80 + "\n")
        for i, (start, end) in enumerate(segments):
            length = end - start + 1
            start_name = frame_paths[start].name
            end_name = frame_paths[end].name
            peak_mag = f"{magnitudes[start]:.3f}" if start > 0 and start - 1 < len(magnitudes) else "—"
            f.write(f"{i:4d} | {start_name:>12} | {end_name:>12} | {start:6d} → {end:6d} | {length:6d} | {peak_mag:>10}\n")

        f.write(f"\nKeyframes (peak starts): {', '.join(frame_paths[p].name for p in peaks)}\n")
    print(f"\n  Saved report → {report_path}")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(20, 6))
    frame_indices = np.arange(len(magnitudes))
    ax.plot(frame_indices, magnitudes, linewidth=0.5, color="steelblue",
            label="Flow magnitude")

    # Mark peaks
    peak_mags = magnitudes[peaks]
    ax.scatter(peaks, peak_mags, color="red", s=40, zorder=5,
               label=f"Top {len(peaks)} peaks")

    # Segment boundaries at peaks
    for p in peaks:
        ax.axvline(x=p, color="red", alpha=0.3, linewidth=0.8, linestyle="--")

    # Shade alternating segments
    for i, (start, end) in enumerate(segments):
        color = "lightyellow" if i % 2 == 0 else "lavender"
        ax.axvspan(start, end, alpha=0.3, color=color)

    ax.set_xlabel("Frame index")
    ax.set_ylabel("Mean optical flow magnitude")
    ax.set_title(f"Camera {cam_name} — {num_frames} frames, "
                 f"{len(peaks)} peaks, {len(segments)} segments")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(cam_output, "flow_magnitude_plot.jpg")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot → {plot_path}")

    return magnitudes, peaks, segments


def main():
    parser = argparse.ArgumentParser(description="Optical flow peak segmentation")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Single camera directory")
    parser.add_argument("--parent-dir", type=str, default=None,
                        help="Parent dir containing camera subdirectories")
    parser.add_argument("--cameras", type=str, nargs="*", default=None,
                        help="Specific camera names to process")
    parser.add_argument("--num-cameras", type=int, default=5,
                        help="Number of random cameras to pick (with --parent-dir)")
    parser.add_argument("--num-peaks", type=int, default=30,
                        help="Number of motion peaks to find")
    parser.add_argument("--min-separation", type=int, default=10,
                        help="Minimum frames between peaks")
    parser.add_argument("--output-dir", type=str, default="./flow_results",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        args.device = f"cuda:{args.gpu}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Build list of (cam_name, image_dir) pairs
    camera_dirs = []
    if args.image_dir:
        cam_name = Path(args.image_dir).name
        camera_dirs.append((cam_name, args.image_dir))
    elif args.parent_dir:
        all_cams = sorted([
            d.name for d in Path(args.parent_dir).iterdir() if d.is_dir()
        ])
        if args.cameras:
            selected = [c for c in args.cameras if c in all_cams]
        else:
            selected = random.sample(all_cams, min(args.num_cameras, len(all_cams)))
            selected.sort()
        for cam in selected:
            camera_dirs.append((cam, str(Path(args.parent_dir) / cam)))
        print(f"Selected cameras: {[c[0] for c in camera_dirs]}")
    else:
        parser.error("Provide --image-dir or --parent-dir")

    # Load RAFT model once
    print("Loading RAFT model...")
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=True).to(args.device).eval()

    # Determine resize from first camera's first image
    first_frame = load_frames(camera_dirs[0][1])[0]
    sample = Image.open(first_frame)
    orig_w, orig_h = sample.size
    max_long_edge = 520
    scale = max_long_edge / max(orig_w, orig_h)
    new_h = int(orig_h * scale) // 8 * 8
    new_w = int(orig_w * scale) // 8 * 8
    resize = torch.nn.Upsample(size=(new_h, new_w), mode="bilinear", align_corners=False)
    print(f"Resize: {orig_w}x{orig_h} → {new_w}x{new_h}")

    # Process each camera
    all_results = {}
    for cam_name, image_dir in camera_dirs:
        magnitudes, peaks, segments = process_camera(
            cam_name=cam_name,
            image_dir=image_dir,
            output_dir=args.output_dir,
            model=model,
            transforms=transforms,
            resize=resize,
            num_peaks=args.num_peaks,
            min_separation=args.min_separation,
            device=args.device,
        )
        all_results[cam_name] = {
            "peaks": peaks,
            "segments": segments,
            "magnitudes": magnitudes,
        }

    # --- Summary across all cameras ---
    print(f"\n\n{'='*70}")
    print(f"SUMMARY — {len(camera_dirs)} cameras, {args.num_peaks} peaks each")
    print(f"{'='*70}")
    for cam_name, res in all_results.items():
        segs = res["segments"]
        lengths = [e - s + 1 for s, e in segs]
        print(f"  {cam_name}: {len(segs)} segments, "
              f"lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}")

    print(f"\nAll results saved to {args.output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
