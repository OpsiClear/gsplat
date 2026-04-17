"""
Parallel per-frame training with stride-3 keyframes.

Uses 3 GPUs to train 3 frames simultaneously from the same keyframe.
Every 3rd frame becomes the next keyframe.

  Step 1: frame 0 (keyframe) → GPU1:frame1, GPU2:frame2, GPU3:frame3 (parallel)
  Step 2: frame 3 (keyframe) → GPU1:frame4, GPU2:frame5, GPU3:frame6 (parallel)
  Step 3: frame 6 (keyframe) → GPU1:frame7, GPU2:frame8, GPU3:frame9 (parallel)
  ...

Usage:
    python examples/run_parallel_perframe.py \
        --data_dir /data/shared/elaheh/4D_demo/outdoor/elly/undistorted \
        --init_ply /path/to/frame0.ply \
        --result_dir /path/to/output \
        --gpus 1,2,3 \
        --num_frames 600 \
        --steps_per_frame 7000
"""

import argparse
import os
import subprocess
import sys
import time
import shutil


def run_single_frame(gpu_id, data_dir, init_ply, result_dir, frame_idx,
                     steps, data_factor, sh_degree, alpha_weight, batch_size):
    """Train a single frame on a specific GPU. Blocks until done."""
    cmd = [
        sys.executable, "examples/simple_trainer_perframe_masked.py",
        "--data_dir", data_dir,
        "--ply_path", init_ply,
        "--result_dir", result_dir,
        "--mode", "mask_only",
        "--num_frames", "1",
        "--frame_start", str(frame_idx),
        "--first_frame_steps", str(steps),
        "--post_split_steps", "0",
        "--steps_per_frame", str(steps),
        "--densify_start", str(args.densify_start),
        "--densify_stop", str(args.densify_stop),
        "--densify_every", str(args.densify_every),
        "--batch_size", str(batch_size),
        "--sh_degree", str(sh_degree),
        "--data_factor", str(data_factor),
        "--no-normalize_world_space",
        "--alpha_outside_weight", str(alpha_weight),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # Stream output
    for line in proc.stdout:
        line = line.decode().rstrip()
        if line:
            print(f"  [GPU{gpu_id} F{frame_idx}] {line}")
    proc.wait()
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--init_ply", required=True, help="Frame 0 PLY")
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--gpus", default="1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--num_frames", type=int, default=600)
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--steps_per_frame", type=int, default=7000)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--alpha_outside_weight", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--densify_every", type=int, default=1500)
    parser.add_argument("--densify_start", type=int, default=500)
    parser.add_argument("--densify_stop", type=int, default=5000)
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    stride = len(gpus)
    ply_dir = os.path.join(args.result_dir, "ply_frames", "dynamic")
    os.makedirs(ply_dir, exist_ok=True)

    # Copy frame 0 PLY as the first keyframe
    frame0_ply = os.path.join(ply_dir, f"{args.frame_start:06d}.ply")
    if not os.path.exists(frame0_ply):
        shutil.copy2(args.init_ply, frame0_ply)
        print(f"Copied init PLY as frame {args.frame_start}")

    total_groups = (args.num_frames + stride - 1) // stride
    print(f"\n{'='*60}")
    print(f"Parallel Per-Frame Training")
    print(f"{'='*60}")
    print(f"  GPUs: {gpus} ({stride} parallel)")
    print(f"  Frames: {args.frame_start} to {args.frame_start + args.num_frames - 1}")
    print(f"  Groups: {total_groups} (stride {stride})")
    print(f"  Steps/frame: {args.steps_per_frame}")
    print(f"  Anti-halo: {args.alpha_outside_weight}")
    print(f"  Result: {args.result_dir}")
    print(f"{'='*60}\n")

    keyframe_ply = frame0_ply
    frame_idx = args.frame_start

    for group in range(total_groups):
        # Frames to train in this group
        frames_this_group = []
        for i in range(stride):
            f = frame_idx + 1 + i
            if f >= args.frame_start + args.num_frames:
                break
            frames_this_group.append(f)

        if not frames_this_group:
            break

        print(f"\n--- Group {group + 1}/{total_groups}: "
              f"keyframe={frame_idx}, training {frames_this_group} ---")

        # Launch all frames in parallel
        procs = []
        for i, f in enumerate(frames_this_group):
            gpu = gpus[i]
            out_ply = os.path.join(ply_dir, f"{f:06d}.ply")
            if os.path.exists(out_ply):
                print(f"  Frame {f} already done, skipping")
                continue

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            # Previous frame for motion map (keyframe or f-1)
            prev_f = max(f - 1, frame_idx)
            cmd = [
                sys.executable, "examples/simple_trainer_perframe_masked.py",
                "--data_dir", args.data_dir,
                "--ply_path", keyframe_ply,
                "--result_dir", args.result_dir,
                "--mode", "mask_only",
                "--num_frames", "1",
                "--frame_start", str(f),
                "--first_frame_steps", str(args.steps_per_frame),
                "--post_split_steps", "0",
                "--steps_per_frame", str(args.steps_per_frame),
                "--densify_start", str(args.densify_start),
                "--densify_stop", str(args.densify_stop),
                "--densify_every", str(args.densify_every),
                "--batch_size", str(args.batch_size),
                "--sh_degree", str(args.sh_degree),
                "--data_factor", str(args.data_factor),
                "--no-normalize_world_space",
                "--alpha_outside_weight", str(args.alpha_outside_weight),
                "--prev_frame_idx", str(prev_f),
            ]
            print(f"  GPU {gpu}: training frame {f} from keyframe {frame_idx}")
            log_file = open(os.path.join(args.result_dir, f"gpu{gpu}_frame{f}.log"), "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append((gpu, f, proc))

        # Wait for all to finish
        for gpu, f, proc in procs:
            proc.wait()
            out_ply = os.path.join(ply_dir, f"{f:06d}.ply")
            if os.path.exists(out_ply):
                print(f"  GPU {gpu}: frame {f} done ✓")
            else:
                print(f"  GPU {gpu}: frame {f} FAILED ✗")

        # Last frame of this group becomes next keyframe
        last_frame = frames_this_group[-1]
        next_keyframe_ply = os.path.join(ply_dir, f"{last_frame:06d}.ply")
        if os.path.exists(next_keyframe_ply):
            keyframe_ply = next_keyframe_ply
            frame_idx = last_frame
        else:
            print(f"  WARNING: keyframe {last_frame} missing, reusing {frame_idx}")

    print(f"\nDone! {len(os.listdir(ply_dir))} PLY files at {ply_dir}")


if __name__ == "__main__":
    main()
