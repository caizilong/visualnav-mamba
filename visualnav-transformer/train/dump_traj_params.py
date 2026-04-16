#!/usr/bin/env python3
"""
导出轨迹的 position 与 yaw 到文本或 CSV 文件。

用法示例：
  python3 dump_traj_params.py -d /workspace/datasets/carla_dataset -t trajectory_000000 -o /workspace/visualnav-transformer/train/trajectory_000000_pos_yaw.txt
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np


def to_scalar(x):
    a = np.asarray(x)
    if a.size == 0:
        return float('nan')
    return float(a.ravel()[0])


def load_traj_data(pkl_path: Path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected pickle format: {pkl_path}")
    pos = data.get('position')
    yaw = data.get('yaw')
    if pos is None or yaw is None:
        raise ValueError(f"Missing 'position' or 'yaw' in {pkl_path}")

    # Robust conversion to numpy arrays
    try:
        pos_arr = np.asarray(pos, dtype=float)
        if pos_arr.ndim == 1:
            pos_arr = pos_arr.reshape(-1, 2)
        elif pos_arr.ndim == 2 and pos_arr.shape[1] != 2:
            # try to stack objects
            pos_arr = np.vstack([np.asarray(p, dtype=float).reshape(1, -1) for p in pos])
    except Exception:
        pos_arr = np.vstack([np.asarray(p, dtype=float).reshape(1, -1) for p in pos])

    yaw_arr = np.asarray(yaw)
    yaw_arr = yaw_arr.ravel()

    return pos_arr, yaw_arr


def dump_traj(traj_path: Path, out_fh):
    pkl = traj_path / 'traj_data.pkl'
    if not pkl.exists():
        raise FileNotFoundError(f"traj_data.pkl not found in {traj_path}")

    pos_arr, yaw_arr = load_traj_data(pkl)

    # count images
    images = sorted([p for p in traj_path.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    n_images = len(images)

    out_fh.write(f"Trajectory: {traj_path.name}\n")
    out_fh.write(f"N_images: {n_images}\n")
    out_fh.write(f"position.shape: {pos_arr.shape}\n")
    out_fh.write(f"yaw.shape: {yaw_arr.shape}\n")
    out_fh.write("index,x,y,yaw\n")

    for i in range(min(len(pos_arr), len(yaw_arr))):
        x = float(pos_arr[i, 0])
        y = float(pos_arr[i, 1])
        yaw = to_scalar(yaw_arr[i])
        out_fh.write(f"{i},{x},{y},{yaw}\n")

    # if arrays lengths mismatch, still print remaining as best-effort
    if len(pos_arr) > len(yaw_arr):
        for i in range(len(yaw_arr), len(pos_arr)):
            x = float(pos_arr[i, 0])
            y = float(pos_arr[i, 1])
            out_fh.write(f"{i},{x},{y},nan\n")
    elif len(yaw_arr) > len(pos_arr):
        for i in range(len(pos_arr), len(yaw_arr)):
            yaw = to_scalar(yaw_arr[i])
            out_fh.write(f"{i},nan,nan,{yaw}\n")

    out_fh.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, help='dataset root path (e.g. datasets/carla_dataset)')
    parser.add_argument('-t', '--traj', default=None, help='trajectory name or index (0-based). If omitted and --all not set, default to 0')
    parser.add_argument('--all', action='store_true', help='export all trajectories into a single output file')
    parser.add_argument('-o', '--out', default=None, help='output file path')

    args = parser.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"Dataset not found: {root}")
        sys.exit(2)

    traj_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not traj_dirs:
        print(f"No trajectory directories found under {root}")
        sys.exit(3)

    if args.all:
        out_path = Path(args.out) if args.out else Path(f"{root.name}_all_trajs_pos_yaw.txt")
        with open(out_path, 'w') as fh:
            for d in traj_dirs:
                try:
                    dump_traj(d, fh)
                except Exception as e:
                    fh.write(f"# Error dumping {d.name}: {e}\n\n")
        print(f"Wrote {out_path}")
        return

    # single trajectory
    if args.traj is None:
        traj_path = traj_dirs[0]
    else:
        # try index
        try:
            idx = int(args.traj)
            traj_path = traj_dirs[idx]
        except Exception:
            candidate = root / args.traj
            if candidate.exists():
                traj_path = candidate
            else:
                print(f"Trajectory not found: {args.traj}")
                sys.exit(4)

    out_path = Path(args.out) if args.out else Path(f"{traj_path.name}_pos_yaw.txt")
    with open(out_path, 'w') as fh:
        dump_traj(traj_path, fh)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
