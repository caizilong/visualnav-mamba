#!/usr/bin/env python3
"""Summarize CARLA deployment motion logs produced by carla_navigate.py."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
VINT_ROOT = Path(os.environ.get("VISUALNAV_ROOT", SCRIPT_DIR.parents[1])).expanduser().resolve()
DEFAULT_LOG_DIR = VINT_ROOT / "deployment" / "logs" / "carla_runs"


def _latest_log(log_dir: Path) -> Path:
    logs = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No *.jsonl logs found under {log_dir}")
    return logs[-1]


def _iter_records(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from exc


def _distance_xy(records: List[dict]) -> float:
    total = 0.0
    prev = None
    for rec in records:
        pose = rec.get("pose") or {}
        cur = (pose.get("x"), pose.get("y"))
        if cur[0] is None or cur[1] is None:
            continue
        if prev is not None:
            total += math.hypot(float(cur[0]) - prev[0], float(cur[1]) - prev[1])
        prev = (float(cur[0]), float(cur[1]))
    return total


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _fmt(value: Optional[float], suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}{suffix}"


def summarize(path: Path) -> None:
    records = list(_iter_records(path))
    start = next((r for r in records if r.get("event") == "start"), None)
    controls = [r for r in records if r.get("event") == "control"]
    if not controls:
        print(f"No control records found in {path}")
        return

    speeds = [float((r.get("velocity") or {}).get("speed_mps", 0.0)) for r in controls]
    steers = [float((r.get("control") or {}).get("steer", 0.0)) for r in controls]
    throttles = [float((r.get("control") or {}).get("throttle", 0.0)) for r in controls]
    brakes = [float((r.get("control") or {}).get("brake", 0.0)) for r in controls]
    proposed_controls = [r.get("proposed_control") or {} for r in controls if r.get("proposed_control")]
    proposed_steers = [float(c.get("steer", 0.0)) for c in proposed_controls]
    proposed_throttles = [float(c.get("throttle", 0.0)) for c in proposed_controls]
    proposed_steer_limit_pairs = []
    for rec in controls:
        proposed = rec.get("proposed_control") or {}
        if not proposed:
            continue
        effective_limit = (rec.get("chosen") or {}).get("effective_max_steer")
        if effective_limit is None and start is not None:
            effective_limit = start.get("max_steer")
        if effective_limit is not None:
            proposed_steer_limit_pairs.append(
                (float(proposed.get("steer", 0.0)), abs(float(effective_limit)))
            )
    applied_flags = [bool(r.get("control_applied", True)) for r in controls]
    min_dists = [
        float((r.get("topomap") or {}).get("min_dist"))
        for r in controls
        if (r.get("topomap") or {}).get("min_dist") is not None
    ]
    localization_modes = [
        (r.get("topomap") or {}).get("localization_mode")
        for r in controls
        if (r.get("topomap") or {}).get("localization_mode") is not None
    ]
    closest_nodes = [
        int((r.get("topomap") or {}).get("closest_node"))
        for r in controls
        if (r.get("topomap") or {}).get("closest_node") is not None
    ]
    vision_closest_nodes = [
        int((r.get("topomap") or {}).get("vision_closest_node"))
        for r in controls
        if (r.get("topomap") or {}).get("vision_closest_node") is not None
    ]
    vision_raw_closest_nodes = [
        int((r.get("topomap") or {}).get("vision_raw_closest_node"))
        for r in controls
        if (r.get("topomap") or {}).get("vision_raw_closest_node") is not None
    ]
    vision_clamped = [
        bool((r.get("topomap") or {}).get("vision_node_was_clamped"))
        for r in controls
        if (r.get("topomap") or {}).get("vision_node_was_clamped") is not None
    ]
    gt_closest_nodes = [
        int((r.get("topomap") or {}).get("gt_closest_node"))
        for r in controls
        if (r.get("topomap") or {}).get("gt_closest_node") is not None
    ]
    gt_dists = [
        float((r.get("topomap") or {}).get("gt_distance_m"))
        for r in controls
        if (r.get("topomap") or {}).get("gt_distance_m") is not None
    ]
    selected_nodes = [
        int((r.get("topomap") or {}).get("selected_subgoal_node"))
        for r in controls
        if (r.get("topomap") or {}).get("selected_subgoal_node") is not None
    ]
    waypoints = [
        (r.get("chosen") or {}).get("waypoint_xy")
        for r in controls
        if (r.get("chosen") or {}).get("waypoint_xy") is not None
    ]

    duration = None
    if controls[0].get("elapsed_wall_time") is not None and controls[-1].get("elapsed_wall_time") is not None:
        duration = float(controls[-1]["elapsed_wall_time"]) - float(controls[0]["elapsed_wall_time"])

    print(f"Log: {path}")
    if start:
        print(
            f"Mode: {start.get('mode', 'navigate')}  Topomap: {start.get('topomap')}  "
            f"Map: {start.get('map')}  Model: {start.get('model')}"
        )
        print(
            "Settings: "
            f"max_v={start.get('max_v')} max_steer={start.get('max_steer')} "
            f"rate_hz={start.get('rate_hz')} "
            f"metric_waypoint_spacing={start.get('metric_waypoint_spacing')} "
            f"selection={start.get('trajectory_selection')}"
        )
        navigation_keys = (
            "goal_node",
            "radius",
            "close_threshold",
            "visual_subgoal_offset",
            "max_visual_node_advance",
        )
        if start.get("topomap") is not None and any(
            start.get(key) is not None for key in navigation_keys
        ):
            print(
                "Navigation: "
                f"goal_node={start.get('goal_node')} radius={start.get('radius')} "
                f"close_threshold={start.get('close_threshold')} "
                f"visual_subgoal_offset={start.get('visual_subgoal_offset')} "
                f"max_visual_node_advance={start.get('max_visual_node_advance')}"
            )
        if start.get("seed") is not None:
            print(
                "Diffusion: "
                f"seed={start.get('seed')} samples={start.get('num_samples')} "
                f"steps={start.get('num_diffusion_iters')} "
                f"guidance={start.get('guidance_min')}..{start.get('guidance_max')} "
                f"power={start.get('guidance_power')}"
            )
        if start.get("steering_controller") is not None:
            print(
                "Controller: "
                f"{start.get('steering_controller')} vehicle={start.get('vehicle')} "
                f"wheelbase={start.get('wheelbase')}m "
                f"max_wheel_angle={start.get('max_wheel_angle_deg')}deg"
            )
        if start.get("min_throttle") is not None or start.get("preview_source") is not None:
            print(
                "Runtime: "
                f"min_throttle={start.get('min_throttle')} "
                f"preview_source={start.get('preview_source')}"
            )
        if start.get("high_speed_max_steer") is not None:
            print(
                "Speed steer limit: "
                f"start={start.get('high_speed_steer_start')}m/s "
                f"full={start.get('high_speed_steer_full')}m/s "
                f"high_speed_max={start.get('high_speed_max_steer')}"
            )
        debug = start.get("debug") or {}
        if debug:
            print(
                "Debug: "
                f"no_control={debug.get('no_control')} "
                f"gt_topomap_localization={debug.get('gt_topomap_localization')} "
                f"gt_subgoal_offset={debug.get('gt_subgoal_offset')} "
                f"draw_world_trajectories={debug.get('draw_world_trajectories')}"
            )
    print(f"Control steps: {len(controls)}")
    print(f"Duration: {_fmt(duration, ' s')}")
    print(f"Path length: {_fmt(_distance_xy(controls), ' m')}")
    print(f"Speed mean/max: {_fmt(_mean(speeds), ' m/s')} / {_fmt(max(speeds), ' m/s')}")
    print(f"Abs steer mean/max: {_fmt(_mean([abs(v) for v in steers]))} / {_fmt(max(abs(v) for v in steers))}")
    print(f"Throttle mean/max: {_fmt(_mean(throttles))} / {_fmt(max(throttles))}")
    print(f"Brake mean/max: {_fmt(_mean(brakes))} / {_fmt(max(brakes))}")
    if proposed_controls:
        print(f"Control applied steps: {sum(applied_flags)} / {len(applied_flags)}")
        print(
            "Proposed throttle mean/max: "
            f"{_fmt(_mean(proposed_throttles))} / {_fmt(max(proposed_throttles))}"
        )
        print(
            "Proposed abs steer mean/max: "
            f"{_fmt(_mean([abs(v) for v in proposed_steers]))} / {_fmt(max(abs(v) for v in proposed_steers))}"
        )
        if proposed_steer_limit_pairs:
            saturated = sum(
                limit > 0 and abs(steer) >= 0.98 * limit
                for steer, limit in proposed_steer_limit_pairs
            )
            print(
                "Proposed steer saturation (effective limit): "
                f"{saturated} / {len(proposed_steer_limit_pairs)} "
                f"({100.0 * saturated / len(proposed_steer_limit_pairs):.1f}%)"
            )
    if localization_modes:
        unique_modes = sorted(set(str(v) for v in localization_modes))
        print(f"Topomap localization mode(s): {', '.join(unique_modes)}")
    if min_dists:
        print(f"Topomap min_dist mean/min/max: {_fmt(_mean(min_dists))} / {_fmt(min(min_dists))} / {_fmt(max(min_dists))}")
    if closest_nodes:
        print(f"Closest node first/last/max: {closest_nodes[0]} / {closest_nodes[-1]} / {max(closest_nodes)}")
    if vision_closest_nodes:
        node_label = "Vision stabilized" if vision_raw_closest_nodes else "Vision"
        print(
            f"{node_label} closest node first/last/max: "
            f"{vision_closest_nodes[0]} / {vision_closest_nodes[-1]} / {max(vision_closest_nodes)}"
        )
        regressions = sum(
            current < previous
            for previous, current in zip(vision_closest_nodes, vision_closest_nodes[1:])
        )
        print(f"{node_label} node regressions: {regressions}")
    if vision_raw_closest_nodes:
        print(
            "Vision raw closest node first/last/min/max: "
            f"{vision_raw_closest_nodes[0]} / {vision_raw_closest_nodes[-1]} / "
            f"{min(vision_raw_closest_nodes)} / {max(vision_raw_closest_nodes)}"
        )
    if vision_clamped:
        clamped_count = sum(vision_clamped)
        print(
            "Vision node gate clamped: "
            f"{clamped_count} / {len(vision_clamped)} "
            f"({100.0 * clamped_count / len(vision_clamped):.1f}%)"
        )
    if gt_closest_nodes:
        print(f"GT closest node first/last/max: {gt_closest_nodes[0]} / {gt_closest_nodes[-1]} / {max(gt_closest_nodes)}")
    if gt_dists:
        print(f"GT pose-to-route dist mean/min/max: {_fmt(_mean(gt_dists), ' m')} / {_fmt(min(gt_dists), ' m')} / {_fmt(max(gt_dists), ' m')}")
    if selected_nodes:
        print(f"Selected node first/last/max: {selected_nodes[0]} / {selected_nodes[-1]} / {max(selected_nodes)}")
    if waypoints:
        dx = [float(wp[0]) for wp in waypoints]
        dy = [float(wp[1]) for wp in waypoints]
        print(f"Chosen waypoint x mean/min/max: {_fmt(_mean(dx))} / {_fmt(min(dx))} / {_fmt(max(dx))}")
        print(f"Chosen waypoint y mean/min/max: {_fmt(_mean(dy))} / {_fmt(min(dy))} / {_fmt(max(dy))}")

    debug = (start or {}).get("debug") or {}
    no_control = bool(debug.get("no_control", False))
    gt_mode = bool(debug.get("gt_topomap_localization", False))
    if no_control:
        print("Note: no-control log; vehicle was intentionally held, so path length and closest-node progress may stay near zero.")
    if min_dists and min(min_dists) >= 8.0:
        print("Warning: topomap distances stayed high; check map/start pose/camera/topomap consistency.")
    if closest_nodes and max(closest_nodes) == closest_nodes[0] and not no_control:
        print("Warning: closest topomap node did not advance; localization likely failed or the vehicle did not move along the route.")
    if gt_mode and vision_closest_nodes and max(vision_closest_nodes) == vision_closest_nodes[0]:
        print("Warning: vision closest node did not advance even though GT localization mode was available for comparison.")
    steer_for_warning = proposed_steers if proposed_steers else steers
    if steer_for_warning:
        max_abs_steer = max(abs(v) for v in steer_for_warning)
        saturated_dynamic = any(
            limit > 0 and abs(steer) >= 0.98 * limit
            for steer, limit in proposed_steer_limit_pairs
        )
        if saturated_dynamic:
            print("Warning: steer saturated near its speed-dependent limit; inspect waypoint y.")
        elif max_abs_steer >= 0.95:
            print("Warning: steer saturated near +/-1; reduce max_steer/steer_gain or inspect waypoint y.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latest or specified CARLA deployment motion log.")
    parser.add_argument("log", nargs="?", default=None, help="Path to a *.jsonl log. Defaults to latest.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    args = parser.parse_args()

    path = Path(args.log).expanduser().resolve() if args.log else _latest_log(Path(args.log_dir).expanduser().resolve())
    summarize(path)


if __name__ == "__main__":
    main()
