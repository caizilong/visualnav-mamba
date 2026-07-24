#!/usr/bin/env python3
"""Export a NoMaD topomap from a recorded CARLA trajectory."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
VINT_ROOT = Path(os.environ.get("VISUALNAV_ROOT", SCRIPT_DIR.parents[1])).expanduser().resolve()
CARLA_WORKSPACE = Path(os.environ.get("CARLA_WORKSPACE", "/home/czl/CARLA")).expanduser().resolve()
DEFAULT_DATASET_DIR = CARLA_WORKSPACE / "carla_fisheye_dataset"
DEFAULT_TOPO_ROOT = VINT_ROOT / "deployment" / "topomaps"


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _resolve_trajectory(dataset_dir: Path, trajectory: Optional[str]) -> Path:
    traj_dirs = sorted(
        path for path in dataset_dir.iterdir()
        if path.is_dir() and path.name.startswith("trajectory_")
    )
    if not traj_dirs:
        raise FileNotFoundError(f"No trajectory_* folders found under {dataset_dir}")

    if trajectory is None or trajectory == "latest":
        return traj_dirs[-1]

    candidate = dataset_dir / trajectory
    if candidate.exists():
        return candidate
    if trajectory.isdigit():
        candidate = dataset_dir / f"trajectory_{int(trajectory):06d}"
        if candidate.exists():
            return candidate
        idx = int(trajectory)
        if 0 <= idx < len(traj_dirs):
            return traj_dirs[idx]
        raise IndexError(f"trajectory index {idx} out of range [0, {len(traj_dirs) - 1}]")
    raise FileNotFoundError(f"Trajectory not found: {trajectory}")


def _frame_indices(num_frames: int, stride: int, include_last: bool) -> List[int]:
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    indices = list(range(0, num_frames, stride))
    if include_last and num_frames > 0 and indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)
    return indices


def _read_camera_meta(traj_dir: Path) -> Optional[Dict]:
    meta_path = traj_dir / "camera_meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def export_topomap(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    topomap_root = Path(args.topomap_root).expanduser().resolve()
    traj_dir = _resolve_trajectory(dataset_dir, args.trajectory)
    traj_data_path = traj_dir / "traj_data.pkl"
    if not traj_data_path.exists():
        raise FileNotFoundError(f"Missing traj_data.pkl: {traj_data_path}")

    traj_data = _load_pickle(traj_data_path)
    positions = np.asarray(traj_data["position"], dtype=np.float32)
    yaws = np.asarray(traj_data["yaw"], dtype=np.float32).reshape(-1)
    if len(positions) != len(yaws):
        raise ValueError(f"position/yaw length mismatch: {len(positions)} vs {len(yaws)}")

    indices = _frame_indices(len(positions), int(args.stride), bool(args.include_last))
    image_dir = topomap_root / "images" / args.name
    meta_dir = topomap_root / "meta"
    if image_dir.exists() and args.overwrite:
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    nodes = []
    for node_idx, frame_idx in enumerate(indices):
        src = traj_dir / f"{frame_idx}.jpg"
        if not src.exists():
            raise FileNotFoundError(f"Missing source image: {src}")
        dst_name = f"{node_idx}.png"
        dst = image_dir / dst_name
        with Image.open(src) as img:
            img.save(dst)
        nodes.append(
            {
                "node": int(node_idx),
                "source_frame": int(frame_idx),
                "image": dst_name,
                "source_image": str(src),
                "x": float(positions[frame_idx, 0]),
                "y": float(positions[frame_idx, 1]),
                "yaw": float(yaws[frame_idx]),
            }
        )

    metadata = {
        "name": args.name,
        "dataset_dir": str(dataset_dir),
        "trajectory": traj_dir.name,
        "stride": int(args.stride),
        "include_last": bool(args.include_last),
        "image_dir": str(image_dir),
        "nodes": nodes,
        "camera_meta": _read_camera_meta(traj_dir),
    }
    meta_path = meta_dir / f"{args.name}.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(nodes)} topomap nodes to {image_dir}")
    print(f"Wrote metadata to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a CARLA fisheye trajectory as a NoMaD topomap.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument(
        "--trajectory",
        default="latest",
        help="trajectory folder name, numeric index, or 'latest' (default).",
    )
    parser.add_argument("--name", default="fisheye_topomap")
    parser.add_argument("--topomap-root", default=str(DEFAULT_TOPO_ROOT))
    parser.add_argument("--stride", type=int, default=8, help="Frame stride; 8 means about 1 Hz for 8 FPS recordings.")
    parser.add_argument("--include-last", action="store_true", default=True)
    parser.add_argument("--no-include-last", dest="include_last", action="store_false")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    args = parser.parse_args()
    export_topomap(args)


if __name__ == "__main__":
    main()
