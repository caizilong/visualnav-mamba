#!/usr/bin/env python3
"""
在 CARLA 仿真中部署 NoMaD-Mamba，复用 deployment/src/utils.py，但不依赖 ROS。

功能要点：
- 加载 CARLA 独立模型注册表中的 checkpoint 与模型配置；
- 前视相机观测 + 拓扑图子目标，与 navigate.py 相同的扩散采样与 guidance；
- 在车辆当前位姿下，将每条轨迹的 8 个二维路点（CARLA 车体坐标：前 x、右 y）变换到世界坐标，
  用 CARLA DebugDraw 绘制 8 条不同颜色的折线（8 条轨迹 × 每条约 8 个点）；
- 使用与 pd_controller 相同的局部 waypoint 规则驱动车辆（映射到 throttle / steer）。

用法（先启动 CARLA，例如 CARLA_Latest 下 ./CarlaUE4.sh）::

    cd /path/to/visualnav-mamba
    bash scripts/carla/carla_vint_quickstart.sh deploy \\
        --dir town01_route01_topomap --no-preview-window

需准备与 navigate 相同的 topomap：deployment/topomaps/images/<dir>/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

try:
    import cv2
except ImportError:
    cv2 = None

# ---------- 路径：脚本位于 visualnav-mamba/scripts/carla ----------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VINT_ROOT = os.path.abspath(
    os.environ.get("VISUALNAV_ROOT", os.path.join(_SCRIPT_DIR, "..", ".."))
)
_CARLA_WORKSPACE = os.path.abspath(
    os.environ.get("CARLA_WORKSPACE", "/home/czl/CARLA")
)
_DEPLOY_ROOT = os.path.join(_VINT_ROOT, "deployment")
_DEPLOY_SRC = os.path.join(_DEPLOY_ROOT, "src")
_TRAIN_ROOT = os.path.join(_VINT_ROOT, "train")
if _TRAIN_ROOT not in sys.path:
    sys.path.insert(0, _TRAIN_ROOT)
if _DEPLOY_SRC not in sys.path:
    sys.path.insert(0, _DEPLOY_SRC)

from utils import clip_angle, load_model, to_numpy, transform_images  # noqa: E402
from carla_camera_utils import (  # noqa: E402
    add_camera_arguments,
    build_camera_blueprint,
    camera_metadata,
    make_camera_transform,
)


def _load_action_stats(train_root: str) -> dict:
    """Load diffusion action normalization stats from training data config."""
    data_cfg = os.path.join(train_root, "vint_train", "data", "data_config.yaml")
    with open(data_cfg, "r") as f:
        cfg = yaml.safe_load(f) or {}
    action_stats = cfg.get("action_stats", {})
    if "min" not in action_stats or "max" not in action_stats:
        raise KeyError(f"action_stats.min/max missing in {data_cfg}")
    return {
        "min": np.array(action_stats["min"], dtype=np.float32),
        "max": np.array(action_stats["max"], dtype=np.float32),
    }


def _unnormalize_data(ndata: np.ndarray, stats: dict) -> np.ndarray:
    ndata = (ndata + 1.0) / 2.0
    return ndata * (stats["max"] - stats["min"]) + stats["min"]


def get_action(diffusion_output: torch.Tensor, action_stats: dict) -> torch.Tensor:
    """
    Convert diffusion output (normalized deltas) to cumulative trajectory actions.
    """
    device = diffusion_output.device
    ndeltas = to_numpy(diffusion_output.reshape(diffusion_output.shape[0], -1, 2))
    deltas = _unnormalize_data(ndeltas, action_stats)
    actions = np.cumsum(deltas, axis=1).astype(np.float32)
    return torch.from_numpy(actions).to(device)


def diffusion_guidance_scale(
    step_idx: int,
    total_steps: int,
    min_scale: float = 0.25,
    max_scale: float = 1.75,
    power: float = 1.5,
) -> float:
    """Early weak guidance, late strong guidance."""
    if total_steps <= 1:
        return max_scale
    progress = step_idx / float(total_steps - 1)
    return min_scale + (max_scale - min_scale) * (progress ** power)


def _ensure_carla(carla_root: Optional[str]) -> None:
    """
    加载 CARLA Python API：优先使用安装目录下的 .egg（与仿真器主版本一致），
    否则使用当前环境已安装的 carla（例如 carla_vint 中 `pip install carla`）。
    """
    roots: List[str] = []
    if carla_root:
        roots.append(os.path.abspath(carla_root))
    env_root = os.environ.get("CARLA_ROOT")
    if env_root and os.path.abspath(env_root) not in [os.path.abspath(r) for r in roots]:
        roots.append(os.path.abspath(env_root))
    for root in roots:
        egg_dir = os.path.join(root, "PythonAPI", "carla", "dist")
        if not os.path.isdir(egg_dir):
            continue
        for name in sorted(os.listdir(egg_dir)):
            if name.endswith(".egg"):
                p = os.path.join(egg_dir, name)
                if p not in sys.path:
                    sys.path.insert(0, p)
                break
    try:
        import carla  # noqa: F401
        return
    except ImportError as e:
        raise ImportError(
            "无法导入 carla。请任选其一：\n"
            "  1) 设置环境变量 CARLA_ROOT 或参数 --carla-root 指向 CARLA 安装目录"
            "（含 PythonAPI/carla/dist/*.egg）；\n"
            "  2) 在与录制数据相同的环境中执行: pip install carla"
            "（主版本号需与 CARLA 服务器一致）。\n"
        ) from e


def _short_map_name(map_name: Optional[str]) -> Optional[str]:
    if not map_name:
        return None
    text = str(map_name).rstrip("/")
    return os.path.basename(text)


def _resolve_repo_path(path: str) -> str:
    """Resolve user paths consistently, independent of the current directory."""
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_VINT_ROOT, path))


def _load_topomap_metadata(topomap_root: str, topomap_name: str) -> Optional[dict]:
    meta_path = os.path.join(topomap_root, "meta", f"{topomap_name}.json")
    if not os.path.isfile(meta_path):
        print(
            f"[topomap] 未找到 meta 文件: {meta_path}。"
            "将无法自动加载地图或从 topomap 起点 spawn；建议用 collect-topomap 重新生成。"
        )
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    print(f"[topomap] Loaded metadata: {meta_path}")
    return meta


def _topomap_spawn_transform(carla_module, world, topomap_meta: dict, node_idx: int, z_offset: float):
    nodes = topomap_meta.get("nodes") or []
    if not nodes:
        raise ValueError("topomap metadata has no nodes")
    node_idx = int(np.clip(node_idx, 0, len(nodes) - 1))
    node = nodes[node_idx]
    x = float(node["x"])
    y = float(node["y"])
    yaw_deg = float(np.degrees(float(node["yaw"])))
    z = float(z_offset)
    try:
        wp = world.get_map().get_waypoint(
            carla_module.Location(x=x, y=y, z=1.0),
            project_to_road=True,
        )
        if wp is not None:
            z = float(wp.transform.location.z) + float(z_offset)
    except RuntimeError:
        pass
    return carla_module.Transform(
        carla_module.Location(x=x, y=y, z=z),
        carla_module.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0),
    )


def _topomap_nodes_xy(topomap_meta: Optional[dict]) -> Optional[np.ndarray]:
    if topomap_meta is None:
        return None
    nodes = topomap_meta.get("nodes") or []
    if not nodes:
        return None
    return np.array([[float(node["x"]), float(node["y"])] for node in nodes], dtype=np.float32)


def _nearest_topomap_node_xy(
    nodes_xy: np.ndarray,
    x: float,
    y: float,
    center_idx: int,
    window: int,
    goal_node: int,
) -> Tuple[int, float, Tuple[int, int]]:
    if nodes_xy.size == 0:
        raise ValueError("topomap metadata has no node positions")
    max_idx = min(int(goal_node), len(nodes_xy) - 1)
    if int(window) > 0 and center_idx >= 0:
        start = max(0, int(center_idx) - int(window))
        end = min(max_idx, int(center_idx) + int(window))
    else:
        start = 0
        end = max_idx
    candidates = nodes_xy[start : end + 1]
    dists = np.linalg.norm(candidates - np.array([[float(x), float(y)]], dtype=np.float32), axis=1)
    local_idx = int(np.argmin(dists))
    return start + local_idx, float(dists[local_idx]), (start, end)


def _select_trajectory_index(
    trajectories: np.ndarray,
    waypoint_idx: int,
    mode: str = "median",
) -> int:
    traj_count = int(trajectories.shape[0])
    if traj_count <= 1 or mode == "first":
        return 0
    if mode == "random":
        return int(np.random.randint(0, max(traj_count, 1)))

    points = trajectories[:, waypoint_idx, :2]
    finite_mask = np.isfinite(points).all(axis=1)
    forward_mask = points[:, 0] > 1e-3
    valid = np.where(finite_mask & forward_mask)[0]
    if len(valid) == 0:
        valid = np.where(finite_mask)[0]
    if len(valid) == 0:
        return 0

    valid_points = points[valid]
    median_xy = np.median(valid_points, axis=0)
    # Prefer the representative trajectory instead of a random diffusion sample.
    scores = np.abs(valid_points[:, 1] - median_xy[1]) + 0.25 * np.abs(
        valid_points[:, 0] - median_xy[0]
    )
    return int(valid[int(np.argmin(scores))])


def _provided_cli_attrs(parser: argparse.ArgumentParser, argv: Sequence[str]) -> set:
    provided = set()
    for action in parser._actions:
        for opt in action.option_strings:
            for arg in argv:
                if arg == opt or arg.startswith(f"{opt}="):
                    provided.add(action.dest)
    return provided


def apply_benchmark_config(
    args: argparse.Namespace,
    section: str,
    protected_attrs: Optional[set] = None,
) -> argparse.Namespace:
    if not args.benchmark_config:
        return args
    protected_attrs = protected_attrs or set()
    cfg_path = args.benchmark_config
    if not os.path.isabs(cfg_path):
        cfg_path = _resolve_repo_path(cfg_path)
    with open(cfg_path, "r") as f:
        benchmark_cfg = yaml.safe_load(f) or {}
    merged_cfg = {}
    merged_cfg.update(benchmark_cfg.get("common", {}))
    merged_cfg.update(benchmark_cfg.get(section, {}))
    for key, value in merged_cfg.items():
        attr_name = key.replace("-", "_")
        if hasattr(args, attr_name):
            if attr_name in protected_attrs:
                continue
            setattr(args, attr_name, value)
    print(f"Loaded benchmark config from {cfg_path} ({section})")
    return args


def pd_controller(
    waypoint: np.ndarray, dt: float, max_v: float, max_w: float
) -> Tuple[float, float]:
    """Convert a waypoint at ``dt`` seconds in the future to target (v, w)."""
    eps = 1e-8
    assert len(waypoint) in (2, 4), "waypoint must be 2D or 4D"
    if len(waypoint) == 2:
        dx, dy = float(waypoint[0]), float(waypoint[1])
    else:
        dx, dy = float(waypoint[0]), float(waypoint[1])
        hx, hy = float(waypoint[2]), float(waypoint[3])
    if len(waypoint) == 4 and np.abs(dx) < eps and np.abs(dy) < eps:
        v = 0.0
        w = clip_angle(np.arctan2(hy, hx)) / dt
    elif np.abs(dx) < eps:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt
    v = float(np.clip(v, 0, max_v))
    w = float(np.clip(w, -max_w, max_w))
    return v, w


def pure_pursuit_steer(
    waypoint: np.ndarray,
    wheelbase: float,
    max_wheel_angle_deg: float,
    steer_gain: float = 1.0,
    max_steer: float = 0.35,
) -> Tuple[float, float, float]:
    """Map a CARLA-local waypoint (+x forward, +y right) to normalized steer.

    Returns ``(steer, curvature, wheel_angle_rad)``. CARLA's normalized steer and
    the dataset's local y coordinate are both positive to the right.
    """
    x, y = float(waypoint[0]), float(waypoint[1])
    lookahead_sq = x * x + y * y
    if lookahead_sq <= 1e-8:
        return 0.0, 0.0, 0.0

    curvature = 2.0 * y / lookahead_sq
    wheel_angle = float(np.arctan(float(wheelbase) * curvature))
    max_wheel_angle = np.deg2rad(max(float(max_wheel_angle_deg), 1e-3))
    normalized_steer = float(steer_gain) * wheel_angle / max_wheel_angle
    steer = float(np.clip(normalized_steer, -abs(max_steer), abs(max_steer)))
    return steer, curvature, wheel_angle


def stabilize_visual_node(
    raw_node: int,
    previous_node: int,
    goal_node: int,
    max_advance: int,
) -> int:
    """Keep visual topomap localization monotonic and reject one-frame jumps."""
    previous = max(0, min(int(previous_node), int(goal_node)))
    upper = min(previous + max(1, int(max_advance)), int(goal_node))
    return max(previous, min(int(raw_node), upper))


def speed_dependent_steer_limit(
    speed_mps: float,
    base_max_steer: float,
    limit_start_mps: float,
    full_limit_mps: float,
    high_speed_max_steer: float,
) -> float:
    """Linearly reduce the normalized steering limit as vehicle speed rises."""
    base_limit = abs(float(base_max_steer))
    high_speed_limit = min(abs(float(high_speed_max_steer)), base_limit)
    start = float(limit_start_mps)
    full = max(float(full_limit_mps), start + 1e-6)
    ratio = float(np.clip((float(speed_mps) - start) / (full - start), 0.0, 1.0))
    return (1.0 - ratio) * base_limit + ratio * high_speed_limit


def vehicle_steering_geometry(
    vehicle,
    fallback_wheelbase: float,
    fallback_max_wheel_angle_deg: float,
) -> Tuple[float, float]:
    """Read wheelbase and maximum wheel angle from CARLA vehicle physics."""
    wheelbase = float(fallback_wheelbase)
    max_wheel_angle_deg = float(fallback_max_wheel_angle_deg)
    try:
        wheels = list(vehicle.get_physics_control().wheels)
        wheel_x_cm = [float(wheel.position.x) for wheel in wheels]
        if len(wheel_x_cm) >= 2:
            measured_wheelbase = (max(wheel_x_cm) - min(wheel_x_cm)) / 100.0
            if 0.5 <= measured_wheelbase <= 10.0:
                wheelbase = measured_wheelbase
        measured_max_angle = max(float(wheel.max_steer_angle) for wheel in wheels)
        if measured_max_angle > 1e-3:
            max_wheel_angle_deg = measured_max_angle
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass
    return wheelbase, max_wheel_angle_deg


def local_xy_to_world_loc(transform, lx: float, ly: float, z_lift: float = 0.35):
    import carla

    # CARLA 数据采集直接保存原生 world x/y 和 yaw；训练时转换出的局部坐标
    # 因而也是 CARLA 车体坐标：+X 向前、+Y 向右。这里不能再翻转 Y。
    local_loc = carla.Location(x=float(lx), y=float(ly), z=float(z_lift))

    # 2. 完整的 3D 旋转与平移矩阵变换：
    # 底层原生包含 Pitch 和 Roll 变换，解决刹车点头时轨迹“飘向 Z 轴”的透视异常
    transform.transform(local_loc)

    return local_loc


def draw_predicted_trajectories(
    world,
    vehicle_transform,
    trajectories: np.ndarray,
    colors: Sequence[Tuple[int, int, int]],
    life_time: float = 0.15,
) -> None:
    """
    trajectories: (num_traj, num_points, 2) 车体坐标系下的累积路点。
    绘制 num_traj 条折线，每条连接 num_points 个点。
    """
    import carla

    num_traj, num_points, _ = trajectories.shape
    for t in range(num_traj):
        color = carla.Color(colors[t][0], colors[t][1], colors[t][2])
        for i in range(num_points - 1):
            lx0, ly0 = trajectories[t, i]
            lx1, ly1 = trajectories[t, i + 1]
            a = local_xy_to_world_loc(vehicle_transform, float(lx0), float(ly0))
            b = local_xy_to_world_loc(vehicle_transform, float(lx1), float(ly1))
            world.debug.draw_line(a, b, thickness=0.08, color=color, life_time=life_time)
        for i in range(num_points):
            lx, ly = trajectories[t, i]
            c = local_xy_to_world_loc(vehicle_transform, float(lx), float(ly))
            world.debug.draw_point(c, size=0.12, color=color, life_time=life_time)


def _camera_intrinsic(image_w: int, image_h: int, fov_deg: float) -> np.ndarray:
    focal = image_w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    k = np.eye(3, dtype=np.float32)
    k[0, 0] = focal
    k[1, 1] = focal
    k[0, 2] = image_w / 2.0
    k[1, 2] = image_h / 2.0
    return k


def _project_world_to_image(
    world_loc,
    camera_transform,
    k: np.ndarray,
    image_w: int,
    image_h: int,
) -> Optional[Tuple[int, int]]:
    """
    Project a CARLA world point to camera image pixel.
    """
    world_pt = np.array([world_loc.x, world_loc.y, world_loc.z, 1.0], dtype=np.float32)
    w2c = np.array(camera_transform.get_inverse_matrix(), dtype=np.float32)
    camera_pt = np.dot(w2c, world_pt)
    # UE coordinate -> standard camera coordinate
    cam_xyz = np.array([camera_pt[1], -camera_pt[2], camera_pt[0]], dtype=np.float32)
    if cam_xyz[2] <= 1e-3:
        return None
    uvw = np.dot(k, cam_xyz)
    u = int(uvw[0] / uvw[2])
    v = int(uvw[1] / uvw[2])
    if u < 0 or u >= image_w or v < 0 or v >= image_h:
        return None
    return (u, v)


def _draw_bev_inset(
    vis: np.ndarray,
    trajectories: np.ndarray,
    colors: Sequence[Tuple[int, int, int]],
    chosen_traj_idx: Optional[int],
    waypoint_idx: Optional[int],
) -> None:
    """
    Draw a bird's-eye-view inset using CARLA local coordinates (x forward, y right).
    This view avoids first-person projection overlap and makes direction clear.
    """
    h, w = vis.shape[:2]
    inset_w = max(280, int(w * 0.34))
    inset_h = max(220, int(h * 0.34))
    x0 = w - inset_w - 12
    y0 = 12

    overlay = vis.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + inset_w, y0 + inset_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)
    cv2.rectangle(vis, (x0, y0), (x0 + inset_w, y0 + inset_h), (230, 230, 230), 1)

    center_x = x0 + inset_w // 2
    origin_y = y0 + inset_h - 26

    max_forward = float(np.max(np.abs(trajectories[:, :, 0]))) if trajectories.size else 2.0
    max_side = float(np.max(np.abs(trajectories[:, :, 1]))) if trajectories.size else 2.0
    max_forward = max(2.0, max_forward)
    max_side = max(1.0, max_side)
    scale_x = (inset_w - 40) / (2.0 * max_side)
    scale_y = (inset_h - 48) / max_forward
    scale = min(scale_x, scale_y)

    # axes: x-forward(up), y-right(right)
    cv2.arrowedLine(
        vis,
        (center_x, origin_y),
        (center_x, y0 + 20),
        (120, 240, 120),
        2,
        cv2.LINE_AA,
        tipLength=0.05,
    )
    cv2.arrowedLine(
        vis,
        (center_x, origin_y),
        (x0 + inset_w - 20, origin_y),
        (240, 180, 80),
        2,
        cv2.LINE_AA,
        tipLength=0.05,
    )
    cv2.putText(vis, "x+", (center_x + 6, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 240, 120), 1, cv2.LINE_AA)
    cv2.putText(vis, "y+", (x0 + inset_w - 48, origin_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 180, 80), 1, cv2.LINE_AA)

    for t in range(trajectories.shape[0]):
        bgr = (colors[t][2], colors[t][1], colors[t][0])
        is_chosen = chosen_traj_idx is not None and t == chosen_traj_idx
        pts: List[Tuple[int, int]] = []
        for i in range(trajectories.shape[1]):
            lx, ly = float(trajectories[t, i, 0]), float(trajectories[t, i, 1])
            px = int(round(center_x + ly * scale))
            py = int(round(origin_y - lx * scale))
            pts.append((px, py))

        if len(pts) >= 2:
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i + 1], (0, 0, 0), 4 if is_chosen else 3, cv2.LINE_AA)
                cv2.line(vis, pts[i], pts[i + 1], bgr, 2 if is_chosen else 1, cv2.LINE_AA)
            cv2.arrowedLine(vis, pts[-2], pts[-1], bgr, 2 if is_chosen else 1, cv2.LINE_AA, tipLength=0.35)

        for p in pts:
            cv2.circle(vis, p, 3 if is_chosen else 2, (0, 0, 0), -1)
            cv2.circle(vis, p, 2 if is_chosen else 1, bgr, -1)

        if waypoint_idx is not None and 0 <= waypoint_idx < len(pts):
            wp = pts[waypoint_idx]
            cv2.circle(vis, wp, 6, (255, 255, 255), 1)

        if pts:
            cv2.putText(
                vis,
                str(t),
                (pts[-1][0] + 4, pts[-1][1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                bgr,
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        vis,
        "BEV (x-forward, y-right)",
        (x0 + 8, y0 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def render_preview_overlay(
    image_bgr: np.ndarray,
    preview_camera_transform,
    trajectories: np.ndarray,
    colors: Sequence[Tuple[int, int, int]],
    chosen_traj_idx: Optional[int],
    waypoint_idx: Optional[int],
    vehicle_transform,
    speed_mps: float,
    image_w: int,
    image_h: int,
    fov_deg: float,
) -> np.ndarray:
    """
    Draw predicted trajectories and speed text on first-person camera image.
    """
    vis = image_bgr.copy()
    k = _camera_intrinsic(image_w, image_h, fov_deg)
    for t in range(trajectories.shape[0]):
        bgr = (colors[t][2], colors[t][1], colors[t][0])
        is_chosen = chosen_traj_idx is not None and t == chosen_traj_idx
        pts_2d: List[Tuple[int, int]] = []
        for i in range(trajectories.shape[1]):
            lx, ly = trajectories[t, i]
            world_loc = local_xy_to_world_loc(vehicle_transform, float(lx), float(ly), z_lift=0.15)
            uv = _project_world_to_image(
                world_loc, preview_camera_transform, k, image_w, image_h
            )
            if uv is not None:
                pts_2d.append(uv)
        if len(pts_2d) >= 2:
            # 先给所有轨迹画黑色描边，提升复杂背景中的可见性
            for i in range(len(pts_2d) - 1):
                cv2.line(vis, pts_2d[i], pts_2d[i + 1], (0, 0, 0), 4 if is_chosen else 3, cv2.LINE_AA)
            if is_chosen:
                # 选中轨迹先画白色描边，增强第一人称画面的可读性
                for i in range(len(pts_2d) - 1):
                    cv2.line(vis, pts_2d[i], pts_2d[i + 1], (255, 255, 255), 5, cv2.LINE_AA)
            for i in range(len(pts_2d) - 1):
                cv2.line(vis, pts_2d[i], pts_2d[i + 1], bgr, 3 if is_chosen else 2, cv2.LINE_AA)
            cv2.arrowedLine(
                vis,
                pts_2d[-2],
                pts_2d[-1],
                bgr,
                3 if is_chosen else 2,
                cv2.LINE_AA,
                tipLength=0.3,
            )
        for p in pts_2d:
            cv2.circle(vis, p, 5 if is_chosen else 4, (0, 0, 0), -1)
            if is_chosen:
                cv2.circle(vis, p, 6, (255, 255, 255), -1)
            cv2.circle(vis, p, 4 if is_chosen else 3, bgr, -1)
        if pts_2d:
            cv2.putText(
                vis,
                str(t),
                (pts_2d[-1][0] + 5, pts_2d[-1][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                bgr,
                1,
                cv2.LINE_AA,
            )
        if waypoint_idx is not None and 0 <= waypoint_idx < len(pts_2d):
            wp = pts_2d[waypoint_idx]
            cv2.circle(vis, wp, 8 if is_chosen else 6, (255, 255, 255), 1)
    cv2.putText(
        vis,
        f"speed: {speed_mps:.2f} m/s",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (50, 255, 50),
        2,
        cv2.LINE_AA,
    )
    if chosen_traj_idx is not None:
        cv2.putText(
            vis,
            f"chosen_traj: {chosen_traj_idx}",
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    _draw_bev_inset(vis, trajectories, colors, chosen_traj_idx, waypoint_idx)
    return vis


def resize_with_letterbox(
    image_bgr: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """Resize an image without changing its aspect ratio, padding with black."""
    if target_w <= 0 or target_h <= 0:
        raise ValueError("preview width and height must be positive")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR image, got shape={image_bgr.shape}")

    source_h, source_w = image_bgr.shape[:2]
    if source_w <= 0 or source_h <= 0:
        raise ValueError(f"invalid source image shape={image_bgr.shape}")

    scale = min(float(target_w) / source_w, float(target_h) / source_h)
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(
        image_bgr,
        (resized_w, resized_h),
        interpolation=interpolation,
    )

    canvas = np.zeros((target_h, target_w, 3), dtype=image_bgr.dtype)
    x0 = (target_w - resized_w) // 2
    y0 = (target_h - resized_h) // 2
    canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized
    return canvas


def render_observation_preview(
    obs_pil: PILImage.Image,
    trajectories: np.ndarray,
    colors: Sequence[Tuple[int, int, int]],
    chosen_traj_idx: Optional[int],
    waypoint_idx: Optional[int],
    speed_mps: float,
    preview_w: int,
    preview_h: int,
) -> np.ndarray:
    """
    Render the actual model observation image. For fisheye observations, only draw
    a BEV inset because the pinhole projection overlay is geometrically invalid.
    Preserve the square observation aspect ratio instead of stretching it to the
    preview window's usually widescreen dimensions.
    """
    rgb = np.asarray(obs_pil.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis = resize_with_letterbox(bgr, preview_w, preview_h)
    cv2.putText(
        vis,
        f"observation fisheye | speed: {speed_mps:.2f} m/s",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 255, 50),
        2,
        cv2.LINE_AA,
    )
    if chosen_traj_idx is not None:
        cv2.putText(
            vis,
            f"chosen_traj: {chosen_traj_idx}",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    _draw_bev_inset(vis, trajectories, colors, chosen_traj_idx, waypoint_idx)
    return vis


def _default_traj_colors(n: int) -> List[Tuple[int, int, int]]:
    # 高对比度固定调色盘（RGB），确保 8 条轨迹在第一人称画面里更容易区分
    base = [
        (255, 48, 48),    # vivid red
        (48, 255, 48),    # vivid green
        (48, 128, 255),   # vivid blue
        (255, 220, 0),    # amber
        (255, 48, 255),   # magenta
        (0, 245, 245),    # cyan
        (255, 120, 0),    # orange
        (160, 80, 255),   # purple
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        hue = int(255 * i / max(n, 1))
        out.append((hue, 255 - hue, 128))
    return out


def carla_control_from_twist(
    v: float,
    w: float,
    throttle_scale: float,
    steer_gain: float,
    current_speed_mps: Optional[float] = None,
    speed_limit_mps: Optional[float] = None,
    speed_kp: float = 0.8,
    max_brake: float = 0.7,
    max_steer: float = 0.35,
    min_throttle: float = 0.0,
    previous_steer: Optional[float] = None,
    steer_smoothing: float = 0.35,
    steer_command: Optional[float] = None,
) -> "carla.VehicleControl":
    import carla

    c = carla.VehicleControl()
    c.throttle = float(np.clip(abs(v) * throttle_scale, 0.0, 1.0))
    if v > 1e-3 and c.throttle > 0.0:
        c.throttle = float(max(c.throttle, np.clip(min_throttle, 0.0, 1.0)))
    c.brake = 0.0
    if (
        current_speed_mps is not None
        and speed_limit_mps is not None
        and speed_limit_mps > 0.0
    ):
        # Closed-loop speed limiting: if overspeed, cut throttle and apply brake.
        if current_speed_mps > speed_limit_mps:
            overspeed = current_speed_mps - speed_limit_mps
            c.throttle = 0.0
            c.brake = float(np.clip(speed_kp * overspeed, 0.0, max_brake))
        elif current_speed_mps > 0.9 * speed_limit_mps:
            # Near limit, smoothly reduce throttle to avoid persistent overshoot.
            remain = max(speed_limit_mps - current_speed_mps, 0.0)
            ratio = remain / max(0.1 * speed_limit_mps, 1e-3)
            c.throttle *= float(np.clip(ratio, 0.0, 1.0))
    if steer_command is None:
        # Backward-compatible fallback for callers that only provide target yaw rate.
        target_steer = float(np.clip(w * steer_gain, -abs(max_steer), abs(max_steer)))
    else:
        target_steer = float(np.clip(steer_command, -abs(max_steer), abs(max_steer)))
    if previous_steer is not None:
        alpha = float(np.clip(steer_smoothing, 0.0, 1.0))
        target_steer = (1.0 - alpha) * float(previous_steer) + alpha * target_steer
    c.steer = float(np.clip(target_steer, -abs(max_steer), abs(max_steer)))
    c.hand_brake = False
    c.manual_gear_shift = False
    return c


def main() -> None:
    parser = argparse.ArgumentParser(description="NoMaD / NoMaD-Mamba in CARLA (no ROS)")
    parser.add_argument(
        "--mode",
        choices=["navigate", "explore"],
        default="navigate",
        help="navigate 使用 topomap 目标；explore 使用无目标条件扩散策略",
    )
    parser.add_argument("--model", "-m", default="nomad_mamba_carla", type=str)
    parser.add_argument(
        "--model-config",
        default=os.path.join(_DEPLOY_ROOT, "config", "models_carla.yaml"),
        type=str,
        help="模型注册表；相对路径按 visualnav-mamba 仓库根目录解析",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="临时覆盖注册表中的 checkpoint；相对路径按仓库根目录解析",
    )
    parser.add_argument(
        "--topomap-root",
        default=os.path.join(_DEPLOY_ROOT, "topomaps"),
        type=str,
        help="包含 images/ 与 meta/ 的 topomap 根目录",
    )
    parser.add_argument(
        "--benchmark-config",
        default=None,
        type=str,
        help="benchmark YAML（可为相对 CARLA 根目录的路径）",
    )
    parser.add_argument(
        "--carla-root",
        default=None,
        type=str,
        help="CARLA 安装根目录（含 PythonAPI/carla/dist/*.egg）；不设则使用环境变量 CARLA_ROOT，或直接依赖 pip 安装的 carla",
    )
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=2000, type=int)
    parser.add_argument(
        "--vehicle-filter",
        default="vehicle.mini.cooper_s",
        type=str,
        help="CARLA vehicle blueprint filter; defaults to the vehicle used for training data",
    )
    parser.add_argument(
        "--spawn-index",
        default=-1,
        type=int,
        help="可选：优先使用的出生点下标（-1 表示自动选择）",
    )
    parser.add_argument(
        "--spawn-retries",
        default=3,
        type=int,
        help="出生点碰撞时的重试轮数",
    )
    parser.add_argument(
        "--spawn-retry-delay",
        default=0.5,
        type=float,
        help="出生点重试间隔（秒）",
    )
    parser.add_argument(
        "--cleanup-vehicles",
        action="store_true",
        help="spawn 前清理当前世界已有车辆，减少出生点碰撞",
    )
    parser.add_argument(
        "--map",
        default=None,
        type=str,
        help="可选：加载指定地图名，如 Town01（默认使用当前世界）",
    )
    parser.add_argument(
        "--spawn-from-topomap",
        dest="spawn_from_topomap",
        action="store_true",
        default=True,
        help="使用 topomap meta 的起点位姿 spawn 车辆（默认启用）",
    )
    parser.add_argument(
        "--no-spawn-from-topomap",
        dest="spawn_from_topomap",
        action="store_false",
        help="禁用 topomap 起点 spawn，改用 --spawn-index/默认出生点",
    )
    parser.add_argument("--topomap-spawn-node", default=0, type=int, help="用作 spawn 起点的 topomap 节点")
    parser.add_argument(
        "--topomap-spawn-z-offset",
        default=0.5,
        type=float,
        help="topomap 起点投影到道路后的车辆 z 偏移",
    )
    parser.add_argument("--dir", "-d", default="topomap", type=str, help="topomaps/images 下的子目录名")
    parser.add_argument(
        "--no-topomap",
        action="store_true",
        help="启用纯端到端模式（不使用 topomap 全局定位）",
    )
    parser.add_argument("--goal-node", "-g", default=-1, type=int)
    parser.add_argument("--waypoint", "-w", default=2, type=int)
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="兼容旧配置并写入日志；视觉控制子目标不再依赖该距离阈值",
    )
    parser.add_argument("--radius", "-r", default=4, type=int, help="视觉定位向后搜索的节点半径")
    parser.add_argument(
        "--visual-subgoal-offset",
        default=3,
        type=int,
        help="视觉定位节点之后固定前视多少个节点作为控制子目标",
    )
    parser.add_argument(
        "--max-visual-node-advance",
        default=2,
        type=int,
        help="每次推理视觉定位节点最多前进多少个节点（且不允许回退，必须 >=1）",
    )
    parser.add_argument("--num-samples", "-n", default=8, type=int)
    parser.add_argument("--guidance-min", default=None, type=float)
    parser.add_argument("--guidance-max", default=None, type=float)
    parser.add_argument("--guidance-power", default=None, type=float)
    parser.add_argument("--seed", default=0, type=int, help="diffusion、NumPy 与 PyTorch 随机种子")
    parser.add_argument(
        "--frame-rate",
        default=8.0,
        type=float,
        help="模型观测上下文与控制循环频率（Hz）；CARLA 训练数据按 8 Hz 采集",
    )
    parser.add_argument(
        "--metric-waypoint-spacing",
        default=0.5,
        type=float,
        help="把模型归一化 waypoint 还原为米的尺度；CARLA 训练配置为 0.5 m",
    )
    parser.add_argument(
        "--max-v-override",
        default=4.0,
        type=float,
        help="覆盖 robot.yaml 的 max_v（m/s），默认 4.0",
    )
    parser.add_argument(
        "--throttle-scale",
        default=0.2,
        type=float,
        help="CARLA 油门缩放（建议值约 1/max_v；默认 0.2）",
    )
    parser.add_argument(
        "--min-throttle",
        default=0.0,
        type=float,
        help="v>0 时的最小油门，用于低速调试时克服车辆起步静摩擦；0 表示禁用",
    )
    parser.add_argument(
        "--steer-gain",
        default=1.0,
        type=float,
        help="pure-pursuit 归一化方向盘增益",
    )
    parser.add_argument("--max-steer", default=0.35, type=float, help="CARLA steer 绝对值限幅，避免猛打方向")
    parser.add_argument(
        "--high-speed-steer-start",
        default=2.5,
        type=float,
        help="从该速度（m/s）开始线性收紧 steer 限幅",
    )
    parser.add_argument(
        "--high-speed-steer-full",
        default=4.0,
        type=float,
        help="达到该速度（m/s）时使用 --high-speed-max-steer",
    )
    parser.add_argument(
        "--high-speed-max-steer",
        default=0.18,
        type=float,
        help="高速时 CARLA steer 绝对值限幅",
    )
    parser.add_argument(
        "--wheelbase",
        default=2.5,
        type=float,
        help="车辆轴距回退值（米）；运行时优先读取 CARLA physics_control",
    )
    parser.add_argument(
        "--max-wheel-angle",
        default=70.0,
        type=float,
        help="前轮最大转角回退值（度）；运行时优先读取 CARLA physics_control",
    )
    parser.add_argument(
        "--steer-smoothing",
        default=0.35,
        type=float,
        help="steer 一阶平滑系数，1 表示不平滑，0 表示保持上一帧",
    )
    parser.add_argument(
        "--trajectory-selection",
        choices=["first", "median", "random"],
        default="median",
        help="多条 diffusion 轨迹中的执行轨迹选择策略",
    )
    parser.add_argument(
        "--topomap-warn-distance",
        default=8.0,
        type=float,
        help="topomap 最小距离高于该值时提示地图/起点可能不匹配；<=0 禁用",
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="调试模式：正常跑模型、绘制轨迹并写日志，但不执行模型控制，只保持车辆刹停",
    )
    parser.add_argument(
        "--gt-topomap-localization",
        action="store_true",
        help="调试模式：使用 CARLA 真值位姿选择最近 topomap 节点，视觉距离仍会写入日志用于对比",
    )
    parser.add_argument(
        "--gt-localization-window",
        default=20,
        type=int,
        help="GT topomap 定位时在当前节点前后多少个节点内搜索；<=0 表示全局搜索",
    )
    parser.add_argument(
        "--gt-subgoal-offset",
        default=1,
        type=int,
        help="GT topomap 定位时选择 closest_node 之后第几个节点作为子目标",
    )
    parser.add_argument("--control-debug", action="store_true", help="打印 waypoint/v/w/steer 调试信息")
    parser.add_argument(
        "--motion-log-dir",
        default=os.path.join(_DEPLOY_ROOT, "logs", "carla_runs"),
        type=str,
        help="部署运动日志输出目录，JSONL 格式",
    )
    parser.add_argument(
        "--no-motion-log",
        dest="motion_log",
        action="store_false",
        help="禁用部署运动日志保存",
    )
    parser.set_defaults(motion_log=True)
    parser.add_argument(
        "--speed-kp",
        default=0.8,
        type=float,
        help="超速制动比例增益（越大制动越积极）",
    )
    parser.add_argument(
        "--max-brake",
        default=0.7,
        type=float,
        help="超速时最大制动强度（0~1）",
    )
    parser.add_argument(
        "--preview-window",
        dest="preview_window",
        action="store_true",
        help="显示前视小窗（第一人称画面 + 预测轨迹）",
    )
    parser.add_argument(
        "--no-preview-window",
        dest="preview_window",
        action="store_false",
        help="禁用前视预览小窗（无图形界面环境建议启用）",
    )
    parser.set_defaults(preview_window=True)
    parser.add_argument(
        "--preview-width",
        default=800,
        type=int,
        help="预览小窗宽度",
    )
    parser.add_argument(
        "--preview-height",
        default=450,
        type=int,
        help="预览小窗高度",
    )
    parser.add_argument(
        "--preview-fov",
        default=90.0,
        type=float,
        help="预览小窗相机 FOV",
    )
    parser.add_argument(
        "--preview-source",
        choices=["pinhole", "observation"],
        default="pinhole",
        help="preview-window 画面来源：pinhole 为普通 RGB 可视化相机；observation 为模型实际鱼眼观测画面",
    )
    parser.add_argument(
        "--draw-life-time",
        default=0.2,
        type=float,
        help="CARLA 世界轨迹的生命周期（秒）；仅在 --draw-world-trajectories 时生效",
    )
    parser.add_argument(
        "--draw-world-trajectories",
        action="store_true",
        help=(
            "在 CARLA 三维世界中绘制预测轨迹。仅用于调试；轨迹会被 observation "
            "相机捕获并污染模型输入，默认关闭"
        ),
    )
    add_camera_arguments(parser)
    raw_argv = sys.argv[1:]
    args = parser.parse_args()
    cli_provided_attrs = _provided_cli_attrs(parser, raw_argv)
    args = apply_benchmark_config(args, args.mode, protected_attrs=cli_provided_attrs)
    if args.mode == "explore":
        args.no_topomap = True
        args.spawn_from_topomap = False
    if int(args.radius) < 0:
        raise ValueError(f"radius 必须 >= 0，当前值为 {args.radius}")
    if int(args.visual_subgoal_offset) < 0:
        raise ValueError(
            f"visual_subgoal_offset 必须 >= 0，当前值为 {args.visual_subgoal_offset}"
        )
    if int(args.max_visual_node_advance) < 1:
        raise ValueError(
            "max_visual_node_advance 必须 >= 1，"
            f"当前值为 {args.max_visual_node_advance}"
        )
    if float(args.high_speed_steer_full) <= float(args.high_speed_steer_start):
        raise ValueError("high_speed_steer_full 必须大于 high_speed_steer_start")
    if float(args.high_speed_max_steer) < 0.0:
        raise ValueError("high_speed_max_steer 必须 >= 0")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"CARLA policy mode: {args.mode}")
    print(f"Random seed: {seed}")

    carla_root = args.carla_root or os.environ.get("CARLA_ROOT")
    _ensure_carla(carla_root)
    import carla

    robot_config_path = os.path.join(_DEPLOY_ROOT, "config", "robot.yaml")
    with open(robot_config_path, "r") as f:
        robot_config = yaml.safe_load(f)
    max_v = float(robot_config["max_v"])
    if args.max_v_override is not None:
        max_v = float(args.max_v_override)
    max_w = float(robot_config["max_w"])
    rate_hz = float(
        args.frame_rate
        if args.frame_rate is not None
        else robot_config["frame_rate"]
    )
    if rate_hz <= 0:
        raise ValueError(f"frame_rate 必须大于 0，当前值为 {rate_hz}")
    dt = 1.0 / rate_hz

    model_config_path = _resolve_repo_path(args.model_config)
    model_config_dir = os.path.dirname(model_config_path)
    with open(model_config_path, "r") as f:
        model_paths = yaml.safe_load(f)
    if args.model not in model_paths:
        raise KeyError(f"模型 {args.model} 未在 {model_config_path} 中配置")
    ckpt_path = args.checkpoint or model_paths[args.model]["ckpt_path"]
    if args.checkpoint:
        ckpt_path = _resolve_repo_path(ckpt_path)
    elif not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(model_config_dir, ckpt_path))
    cfg_path = model_paths[args.model]["config_path"]
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.normpath(os.path.join(model_config_dir, cfg_path))
    with open(cfg_path, "r") as f:
        model_params = yaml.safe_load(f)
    config_overrides = model_paths[args.model].get("config_overrides", {}) or {}
    if config_overrides:
        model_params.update(config_overrides)
        print(f"Applied model config overrides for {args.model}: {config_overrides}")
    action_stats = _load_action_stats(_TRAIN_ROOT)
    metric_waypoint_spacing = float(
        args.metric_waypoint_spacing
        if args.metric_waypoint_spacing is not None
        else model_params.get("metric_waypoint_spacing", max_v / rate_hz)
    )
    if metric_waypoint_spacing <= 0:
        raise ValueError(
            "metric_waypoint_spacing 必须大于 0，"
            f"当前值为 {metric_waypoint_spacing}"
        )
    print(
        "Policy timing/action scale:",
        f"frame_rate={rate_hz:g} Hz,",
        f"metric_waypoint_spacing={metric_waypoint_spacing:g} m,",
        f"max_v={max_v:g} m/s",
    )
    if args.mode == "explore" and float(model_params.get("goal_mask_prob", 0.0)) <= 0.0:
        print(
            "[model] 警告：当前模型注册为 goal_mask_prob=0，Explore 的无目标分支"
            "没有对应训练样本；建议使用 nomad_mamba_carla。"
        )

    context_size = model_params["context_size"]
    guidance_min = args.guidance_min
    if guidance_min is None:
        guidance_min = model_params.get("goal_guidance_min", 1.0)
    guidance_max = args.guidance_max
    if guidance_max is None:
        guidance_max = model_params.get("goal_guidance_max", 1.0)
    guidance_power = args.guidance_power
    if guidance_power is None:
        guidance_power = model_params.get("goal_guidance_power", 1.0)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"未找到权重: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    diffusion_generator = torch.Generator(device=device)
    diffusion_generator.manual_seed(seed)
    model = load_model(ckpt_path, model_params, device)
    model.eval()

    use_topomap = not args.no_topomap
    topomap_meta: Optional[dict] = None
    topomap_nodes_xy: Optional[np.ndarray] = None
    topomap_root = _resolve_repo_path(args.topomap_root)
    if use_topomap:
        topomap_dir = os.path.join(topomap_root, "images", args.dir)
        if not os.path.isdir(topomap_dir):
            raise FileNotFoundError(
                f"未找到拓扑图目录: {topomap_dir}（请先按 deployment 说明生成 topomap）"
            )
        topomap_meta = _load_topomap_metadata(topomap_root, args.dir)
        topomap_filenames = sorted(os.listdir(topomap_dir), key=lambda x: int(x.split(".")[0]))
        num_nodes = len(topomap_filenames)
        topomap: List[PILImage.Image] = []
        for i in range(num_nodes):
            topomap.append(PILImage.open(os.path.join(topomap_dir, topomap_filenames[i])))
        goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node
        assert 0 <= goal_node < len(topomap), "Invalid goal index"
        closest_node = 0
        reached_goal = False
        topomap_nodes_xy = _topomap_nodes_xy(topomap_meta)
    else:
        if args.mode == "explore":
            print("Running in explore mode (goal-masked, no topomap).")
        else:
            print("Running in navigate/no-topomap compatibility mode.")
        topomap = []
        goal_node = -1
        closest_node = -1
        reached_goal = False

    if args.gt_topomap_localization:
        if not use_topomap:
            raise ValueError("--gt-topomap-localization 需要启用 topomap，不能与 --no-topomap 同时使用")
        if topomap_nodes_xy is None:
            raise ValueError("--gt-topomap-localization 需要 topomap meta 中包含节点 x/y")
        print(
            "[debug] GT topomap localization enabled: "
            f"window={args.gt_localization_window}, subgoal_offset={args.gt_subgoal_offset}"
        )
    if args.no_control:
        print("[debug] No-control mode enabled: model outputs are logged, vehicle is held with brake.")

    num_samples = args.num_samples
    if num_samples < 1:
        print(f"num_samples={num_samples} 非法，自动回退为 1。")
        num_samples = 1
    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=model_params["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    else:
        raise NotImplementedError("当前 CARLA 脚本仅实现 model_type == nomad")

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    requested_map = args.map
    meta_map = None
    if topomap_meta is not None:
        meta_map = (topomap_meta.get("camera_meta") or {}).get("map")
        if requested_map is None and meta_map:
            requested_map = _short_map_name(meta_map)
            print(f"[topomap] 使用 meta 中的地图加载 CARLA world: {requested_map}")

    if requested_map:
        world = client.load_world(requested_map)
    else:
        world = client.get_world()
    print("CARLA world:", world.get_map().name)
    if meta_map and _short_map_name(world.get_map().name) != _short_map_name(meta_map):
        print(
            "[topomap] 警告：当前 CARLA 地图与 topomap meta 不一致："
            f"world={world.get_map().name}, topomap={meta_map}"
        )
    if args.cleanup_vehicles:
        actors = world.get_actors().filter("vehicle.*")
        cleanup_count = 0
        for actor in actors:
            try:
                actor.destroy()
                cleanup_count += 1
            except RuntimeError:
                pass
        # 清理后等待一个 tick，确保碰撞体状态更新
        time.sleep(0.2)
        print(f"Cleaned up {cleanup_count} existing vehicles before spawn.")

    blueprint_library = world.get_blueprint_library()
    vehicle_candidates = blueprint_library.filter(args.vehicle_filter)
    if not vehicle_candidates:
        raise ValueError(f"未找到车辆 blueprint: {args.vehicle_filter}")
    vehicle_bp = vehicle_candidates[0]
    spawn_points = world.get_map().get_spawn_points()
    vehicle = None
    if use_topomap and args.spawn_from_topomap:
        if topomap_meta is None:
            print("[topomap] 未找到 meta，无法从 topomap 起点 spawn；回退到普通 spawn point。")
        else:
            try:
                spawn_tf = _topomap_spawn_transform(
                    carla,
                    world,
                    topomap_meta,
                    args.topomap_spawn_node,
                    args.topomap_spawn_z_offset,
                )
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_tf)
                if vehicle is not None:
                    print(
                        "[topomap] Spawned vehicle at topomap "
                        f"node {args.topomap_spawn_node}: "
                        f"x={spawn_tf.location.x:.2f}, y={spawn_tf.location.y:.2f}, "
                        f"yaw={spawn_tf.rotation.yaw:.1f}"
                    )
                else:
                    print("[topomap] topomap 起点被占用或不可生成，回退到普通 spawn point。")
            except (KeyError, ValueError, RuntimeError) as exc:
                print(f"[topomap] topomap 起点 spawn 失败: {exc}；回退到普通 spawn point。")
    if spawn_points:
        base_indices = list(range(len(spawn_points)))
        if args.spawn_index >= 0:
            preferred = args.spawn_index % len(spawn_points)
            base_indices = [preferred] + [i for i in base_indices if i != preferred]
        for attempt in range(max(1, int(args.spawn_retries))):
            if vehicle is not None:
                break
            # 首轮按固定顺序，后续轮次打乱以避开拥堵点
            if attempt == 0:
                candidate_indices = base_indices
            else:
                candidate_indices = list(np.random.permutation(base_indices))
            for idx in candidate_indices:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[idx])
                if vehicle is not None:
                    print(f"Spawned vehicle at spawn index {idx} (attempt {attempt + 1}).")
                    break
            if vehicle is not None:
                break
            time.sleep(max(0.0, float(args.spawn_retry_delay)))
    else:
        if vehicle is None:
            vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform())

    if vehicle is None:
        raise RuntimeError(
            "Spawn failed because all candidate spawn positions are occupied. "
            "请稍后重试，或切换地图/增大 --spawn-retries，或指定 --spawn-index。"
        )
    vehicle.set_autopilot(False)
    wheelbase, max_wheel_angle_deg = vehicle_steering_geometry(
        vehicle,
        fallback_wheelbase=args.wheelbase,
        fallback_max_wheel_angle_deg=args.max_wheel_angle,
    )
    print(
        "Vehicle/controller:",
        vehicle_bp.id,
        "pure_pursuit,",
        f"wheelbase={wheelbase:.3f} m,",
        f"max_wheel_angle={max_wheel_angle_deg:.1f} deg",
    )

    obs_image_size = (int(args.record_width), int(args.record_height))
    cam_bp = build_camera_blueprint(
        blueprint_library,
        camera_type=args.camera_type,
        image_size=obs_image_size,
        rgb_fov=args.rgb_fov,
        fisheye_fov=args.fisheye_fov,
        fisheye_model=args.fisheye_model,
        fov_mask=args.fov_mask,
        fov_fade_size=args.fov_fade_size,
        sensor_tick=1.0 / rate_hz,
    )
    # 与 carla_simple_control.py 录制前视相机位姿保持一致
    cam_tf = make_camera_transform(carla)
    obs_camera_meta = camera_metadata(
        cam_bp,
        args.camera_type,
        obs_image_size,
        cam_tf,
        extra={"role": "deployment_observation"},
    )
    print(
        "Observation camera:",
        obs_camera_meta["blueprint"],
        f"{obs_image_size[0]}x{obs_image_size[1]}",
        obs_camera_meta["attributes"],
    )
    context_queue: deque = deque(maxlen=context_size + 1)
    context_lock = threading.Lock()

    def _on_image(img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        rgb = arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        with context_lock:
            context_queue.append(PILImage.fromarray(rgb))

    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
    camera.listen(_on_image)

    preview_queue: deque = deque(maxlen=1)
    preview_camera = None
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    preview_enabled = args.preview_window and cv2 is not None and has_display
    preview_w = int(args.preview_width)
    preview_h = int(args.preview_height)
    preview_fov = float(args.preview_fov)
    preview_source = str(args.preview_source)
    if args.preview_window and cv2 is None:
        print("OpenCV (cv2) 未安装，禁用预览小窗。")
    elif args.preview_window and not has_display:
        print("未检测到图形显示环境（DISPLAY/WAYLAND_DISPLAY），禁用预览小窗。")

    if preview_enabled and preview_source == "pinhole":
        pv_bp = blueprint_library.find("sensor.camera.rgb")
        pv_bp.set_attribute("image_size_x", str(preview_w))
        pv_bp.set_attribute("image_size_y", str(preview_h))
        pv_bp.set_attribute("fov", str(preview_fov))
        # 与录制脚本 carla_simple_control.py 的前视相机保持一致
        pv_tf = carla.Transform(
            carla.Location(x=2.0, y=0.0, z=1.6),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )

        def _on_preview_image(img):
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
            bgr = arr[:, :, :3]
            preview_queue.append(bgr.copy())

        preview_camera = world.spawn_actor(pv_bp, pv_tf, attach_to=vehicle)
        preview_camera.listen(_on_preview_image)
        cv2.namedWindow("CARLA NoMaD Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA NoMaD Preview", preview_w, preview_h)
    elif preview_enabled:
        print(
            "[preview] 使用 observation 鱼眼观测画面；保持原始宽高比，"
            "轨迹仅显示在 BEV inset 中。"
        )
        cv2.namedWindow("CARLA NoMaD Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CARLA NoMaD Preview", preview_w, preview_h)

    if args.draw_world_trajectories:
        print(
            "[debug] 警告：已启用 CARLA 世界轨迹绘制；这些轨迹会被 observation "
            "相机捕获并进入后续模型输入。不要用于正式闭环评估。"
        )

    traj_colors = _default_traj_colors(num_samples)
    waypoint_clip_warned = False
    topomap_distance_warned = False
    last_steer = 0.0
    motion_log_file = None
    motion_log_path = None
    run_started_wall_time = time.time()
    if args.motion_log:
        os.makedirs(args.motion_log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_label = args.dir if use_topomap else args.mode
        log_name = f"{timestamp}_{log_label}.jsonl"
        motion_log_path = os.path.join(args.motion_log_dir, log_name)
        motion_log_file = open(motion_log_path, "w", encoding="utf-8")
        motion_log_file.write(
            json.dumps(
                {
                    "event": "start",
                    "wall_time": run_started_wall_time,
                    "mode": args.mode,
                    "topomap": args.dir if use_topomap else None,
                    "model": args.model,
                    "checkpoint": str(ckpt_path),
                    "model_goal_mask_prob": float(model_params.get("goal_mask_prob", 0.0)),
                    "model_drop_backbone_prefix_tokens": bool(
                        model_params.get("drop_backbone_prefix_tokens", True)
                    ),
                    "map": world.get_map().name,
                    "vehicle": vehicle_bp.id,
                    "spawn_from_topomap": bool(args.spawn_from_topomap),
                    "topomap_spawn_node": int(args.topomap_spawn_node),
                    "goal_node": int(goal_node) if use_topomap else None,
                    "close_threshold": int(args.close_threshold),
                    "radius": int(args.radius),
                    "visual_subgoal_offset": int(args.visual_subgoal_offset),
                    "max_visual_node_advance": int(args.max_visual_node_advance),
                    "max_v": float(max_v),
                    "max_w": float(max_w),
                    "rate_hz": float(rate_hz),
                    "metric_waypoint_spacing": float(metric_waypoint_spacing),
                    "waypoint": int(args.waypoint),
                    "trajectory_selection": args.trajectory_selection,
                    "seed": int(seed),
                    "num_samples": int(num_samples),
                    "num_diffusion_iters": int(num_diffusion_iters),
                    "guidance_min": float(guidance_min),
                    "guidance_max": float(guidance_max),
                    "guidance_power": float(guidance_power),
                    "max_steer": float(args.max_steer),
                    "high_speed_steer_start": float(args.high_speed_steer_start),
                    "high_speed_steer_full": float(args.high_speed_steer_full),
                    "high_speed_max_steer": float(args.high_speed_max_steer),
                    "steer_smoothing": float(args.steer_smoothing),
                    "steer_gain": float(args.steer_gain),
                    "steering_controller": "pure_pursuit",
                    "wheelbase": float(wheelbase),
                    "max_wheel_angle_deg": float(max_wheel_angle_deg),
                    "throttle_scale": float(args.throttle_scale),
                    "min_throttle": float(args.min_throttle),
                    "preview_source": str(args.preview_source),
                    "debug": {
                        "no_control": bool(args.no_control),
                        "gt_topomap_localization": bool(args.gt_topomap_localization),
                        "gt_localization_window": int(args.gt_localization_window),
                        "gt_subgoal_offset": int(args.gt_subgoal_offset),
                        "draw_world_trajectories": bool(args.draw_world_trajectories),
                    },
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        motion_log_file.flush()
        print(f"[motion-log] Writing deployment log to {motion_log_path}")

    rate = 1.0 / rate_hz

    try:
        while True:
            if use_topomap and reached_goal:
                vehicle.apply_control(carla.VehicleControl())
                time.sleep(0.05)
                continue

            with context_lock:
                context_images = list(context_queue)
            if len(context_images) <= context_size:
                time.sleep(0.01)
                continue
            obs_pil = context_images[-1]

            obs_images = transform_images(
                context_images, model_params["image_size"], center_crop=False
            )
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)
            topomap_log = {
                "localization_mode": ("gt" if args.gt_topomap_localization else "vision")
                if use_topomap
                else None,
                "candidate_node_ids": None,
                "dists": None,
                "closest_node": int(closest_node) if use_topomap else None,
                "selected_subgoal_node": None,
                "min_dist": None,
                "vision_raw_closest_node": None,
                "vision_closest_node": None,
                "vision_node_was_clamped": None,
                "vision_selected_subgoal_node": None,
                "gt_closest_node": None,
                "gt_distance_m": None,
                "gt_search_range": None,
            }
            with torch.inference_mode():
                if use_topomap:
                    gt_closest_node = None
                    gt_distance_m = None
                    gt_search_range = None
                    previous_closest_node = int(closest_node)
                    localization_center_node = closest_node
                    if args.gt_topomap_localization:
                        vehicle_loc = vehicle.get_transform().location
                        gt_closest_node, gt_distance_m, gt_search_range = _nearest_topomap_node_xy(
                            topomap_nodes_xy,
                            float(vehicle_loc.x),
                            float(vehicle_loc.y),
                            closest_node,
                            int(args.gt_localization_window),
                            goal_node,
                        )
                        localization_center_node = int(gt_closest_node)

                    mask = torch.zeros(1, dtype=torch.long, device=device)
                    start = max(int(localization_center_node) - args.radius, 0)
                    if args.gt_topomap_localization:
                        search_forward_nodes = max(
                            int(args.radius) + 1,
                            max(0, int(args.gt_subgoal_offset)),
                        )
                    else:
                        # The candidate range must contain both every accepted
                        # localization node and its fixed forward control goal.
                        search_forward_nodes = max(
                            int(args.radius) + 1,
                            int(args.max_visual_node_advance)
                            + int(args.visual_subgoal_offset),
                        )
                    end = min(
                        int(localization_center_node) + search_forward_nodes,
                        goal_node,
                    )
                    goal_image = [
                        transform_images(g_img, model_params["image_size"], center_crop=False).to(device)
                        for g_img in topomap[start : end + 1]
                    ]
                    goal_image = torch.concat(goal_image, dim=0)
                    obsgoal_cond = model(
                        "vision_encoder",
                        obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                        goal_img=goal_image,
                        input_goal_mask=mask.repeat(len(goal_image)),
                    )
                    dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                    dists = to_numpy(dists.flatten())
                    candidate_node_ids = list(range(start, end + 1))
                    min_idx = int(np.argmin(dists))
                    if (
                        args.topomap_warn_distance > 0
                        and not topomap_distance_warned
                        and float(dists[min_idx]) >= float(args.topomap_warn_distance)
                    ):
                        print(
                            "[topomap] 警告：最小匹配距离较高 "
                            f"({float(dists[min_idx]):.2f})，"
                            "可能是 topomap 与当前地图/起点/相机不一致。"
                        )
                        topomap_distance_warned = True
                    vision_raw_closest_node = min_idx + start
                    if args.gt_topomap_localization:
                        # GT remains the control localization source in this debug
                        # mode; raw vision output is logged without stabilization.
                        vision_closest_node = int(vision_raw_closest_node)
                        vision_selected_subgoal_node = min(
                            vision_closest_node + int(args.visual_subgoal_offset),
                            goal_node,
                        )
                        closest_node = int(gt_closest_node)
                        selected_subgoal_node = min(
                            int(closest_node) + max(0, int(args.gt_subgoal_offset)),
                            goal_node,
                        )
                        selected_subgoal_node = int(np.clip(selected_subgoal_node, start, end))
                        sg_idx = int(selected_subgoal_node - start)
                    else:
                        vision_closest_node = stabilize_visual_node(
                            vision_raw_closest_node,
                            previous_closest_node,
                            goal_node,
                            int(args.max_visual_node_advance),
                        )
                        closest_node = int(vision_closest_node)
                        # Localization and control targeting are deliberately
                        # separated: distance threshold no longer changes lookahead.
                        selected_subgoal_node = min(
                            closest_node + int(args.visual_subgoal_offset),
                            goal_node,
                        )
                        vision_selected_subgoal_node = int(selected_subgoal_node)
                        sg_idx = int(selected_subgoal_node - start)
                    if not 0 <= sg_idx < len(obsgoal_cond):
                        raise RuntimeError(
                            "selected topomap subgoal is outside the encoded candidate range: "
                            f"selected={selected_subgoal_node}, range=[{start}, {end}]"
                        )
                    topomap_log = {
                        "localization_mode": "gt" if args.gt_topomap_localization else "vision",
                        "candidate_node_ids": [int(v) for v in candidate_node_ids],
                        "dists": [float(v) for v in dists.tolist()],
                        "closest_node": int(closest_node),
                        "selected_subgoal_node": int(selected_subgoal_node),
                        "min_dist": float(dists[min_idx]),
                        "vision_raw_closest_node": int(vision_raw_closest_node),
                        "vision_closest_node": int(vision_closest_node),
                        "vision_node_was_clamped": bool(
                            vision_closest_node != vision_raw_closest_node
                        ),
                        "vision_selected_subgoal_node": int(vision_selected_subgoal_node),
                        "gt_closest_node": int(gt_closest_node) if gt_closest_node is not None else None,
                        "gt_distance_m": float(gt_distance_m) if gt_distance_m is not None else None,
                        "gt_search_range": [int(gt_search_range[0]), int(gt_search_range[1])]
                        if gt_search_range is not None
                        else None,
                    }
                    if args.gt_topomap_localization:
                        print(
                            "[topomap/gt] "
                            f"gt_closest={closest_node} "
                            f"gt_dist={float(gt_distance_m):.2f}m "
                            f"vision_raw_closest={vision_raw_closest_node} "
                            f"selected_subgoal_node={selected_subgoal_node} "
                            f"candidate_node_ids={candidate_node_ids} "
                            f"dists={np.array2string(dists, precision=2, suppress_small=True)}"
                        )
                    else:
                        print(
                            "[topomap] "
                            f"candidate_node_ids={candidate_node_ids} "
                            f"dists={np.array2string(dists, precision=2, suppress_small=True)} "
                            f"raw_closest_node={vision_raw_closest_node} "
                            f"closest_node={closest_node} "
                            f"selected_subgoal_node={selected_subgoal_node}"
                        )
                    cond_obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
                    selected_goal_img = goal_image[sg_idx].unsqueeze(0)
                    no_goal_mask = torch.ones(1, dtype=torch.long, device=device)
                    obs_cond = model(
                        "vision_encoder",
                        obs_img=obs_images,
                        goal_img=selected_goal_img,
                        input_goal_mask=no_goal_mask,
                    )

                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(num_samples, 1)
                        cond_obs_cond = cond_obs_cond.repeat(num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(num_samples, 1, 1)
                        cond_obs_cond = cond_obs_cond.repeat(num_samples, 1, 1)

                    noisy_action = torch.randn(
                        (num_samples, model_params["len_traj_pred"], 2),
                        device=device,
                        generator=diffusion_generator,
                    )
                    naction = noisy_action
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                    total_steps = len(noise_scheduler.timesteps)
                    for step_idx, k in enumerate(noise_scheduler.timesteps):
                        unconditional_noise = model(
                            "noise_pred_net",
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond,
                        )
                        conditional_noise = model(
                            "noise_pred_net",
                            sample=naction,
                            timestep=k,
                            global_cond=cond_obs_cond,
                        )
                        gs = diffusion_guidance_scale(
                            step_idx,
                            total_steps,
                            guidance_min,
                            guidance_max,
                            guidance_power,
                        )
                        noise_pred = unconditional_noise + gs * (
                            conditional_noise - unconditional_noise
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction,
                            generator=diffusion_generator,
                        ).prev_sample
                else:
                    img_w, img_h = model_params["image_size"]
                    fake_goal = torch.randn(
                        (1, 3, img_h, img_w),
                        device=device,
                        generator=diffusion_generator,
                    )
                    no_goal_mask = torch.ones(1, dtype=torch.long, device=device)
                    obs_cond = model(
                        "vision_encoder",
                        obs_img=obs_images,
                        goal_img=fake_goal,
                        input_goal_mask=no_goal_mask,
                    )

                    if len(obs_cond.shape) == 2:
                        obs_cond = obs_cond.repeat(num_samples, 1)
                    else:
                        obs_cond = obs_cond.repeat(num_samples, 1, 1)

                    noisy_action = torch.randn(
                        (num_samples, model_params["len_traj_pred"], 2),
                        device=device,
                        generator=diffusion_generator,
                    )
                    naction = noisy_action
                    noise_scheduler.set_timesteps(num_diffusion_iters)
                    for k in noise_scheduler.timesteps:
                        noise_pred = model(
                            "noise_pred_net",
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond,
                        )
                        naction = noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction,
                            generator=diffusion_generator,
                        ).prev_sample

            naction = to_numpy(get_action(naction, action_stats))
            if model_params.get("normalize"):
                # Training normalizes CARLA waypoints by metric_waypoint_spacing.
                # Keep this spatial scale independent from controller speed limits.
                naction *= metric_waypoint_spacing

            if args.draw_world_trajectories:
                draw_predicted_trajectories(
                    world,
                    vehicle.get_transform(),
                    naction,
                    traj_colors,
                    life_time=args.draw_life_time,
                )

            traj_count = int(naction.shape[0])
            traj_len = int(naction.shape[1]) if naction.ndim >= 2 else 0
            if traj_count <= 0 or traj_len <= 0:
                time.sleep(rate)
                continue
            waypoint_idx = int(np.clip(args.waypoint, 0, traj_len - 1))
            if waypoint_idx != args.waypoint and not waypoint_clip_warned:
                print(
                    f"waypoint={args.waypoint} 超出范围 [0, {traj_len - 1}]，"
                    f"自动使用 waypoint={waypoint_idx}。"
                )
                waypoint_clip_warned = True
            chosen_traj_idx = _select_trajectory_index(
                naction,
                waypoint_idx,
                mode=args.trajectory_selection,
            )
            chosen = naction[chosen_traj_idx, waypoint_idx]
            lookahead_time = (waypoint_idx + 1) * dt
            v, w = pd_controller(
                waypoint=chosen[:2],
                dt=lookahead_time,
                max_v=max_v,
                max_w=max_w,
            )
            vehicle_velocity = vehicle.get_velocity()
            speed_mps = float(
                np.linalg.norm(
                    [
                        vehicle_velocity.x,
                        vehicle_velocity.y,
                        vehicle_velocity.z,
                    ]
                )
            )
            effective_max_steer = speed_dependent_steer_limit(
                speed_mps,
                base_max_steer=args.max_steer,
                limit_start_mps=args.high_speed_steer_start,
                full_limit_mps=args.high_speed_steer_full,
                high_speed_max_steer=args.high_speed_max_steer,
            )
            steer_command, curvature, wheel_angle = pure_pursuit_steer(
                chosen[:2],
                wheelbase=wheelbase,
                max_wheel_angle_deg=max_wheel_angle_deg,
                steer_gain=args.steer_gain,
                max_steer=effective_max_steer,
            )
            ctrl = carla_control_from_twist(
                v,
                w,
                args.throttle_scale,
                args.steer_gain,
                current_speed_mps=speed_mps,
                speed_limit_mps=max_v,
                speed_kp=args.speed_kp,
                max_brake=args.max_brake,
                max_steer=effective_max_steer,
                min_throttle=args.min_throttle,
                previous_steer=last_steer,
                steer_smoothing=args.steer_smoothing,
                steer_command=steer_command,
            )
            proposed_ctrl = ctrl
            applied_ctrl = proposed_ctrl
            control_applied = True
            if args.no_control:
                applied_ctrl = carla.VehicleControl()
                applied_ctrl.throttle = 0.0
                applied_ctrl.steer = 0.0
                applied_ctrl.brake = 1.0
                control_applied = False
            last_steer = float(applied_ctrl.steer)
            if args.control_debug:
                print(
                    "[control] "
                    f"traj={chosen_traj_idx} wp={waypoint_idx} "
                    f"xy=({float(chosen[0]):.2f},{float(chosen[1]):.2f}) "
                    f"lookahead={lookahead_time:.3f}s v={v:.2f} w={w:.2f} "
                    f"curvature={curvature:.3f} wheel_angle={np.rad2deg(wheel_angle):.1f}deg "
                    f"steer_limit={effective_max_steer:.2f} "
                    f"proposed_steer={proposed_ctrl.steer:.2f} "
                    f"applied_steer={applied_ctrl.steer:.2f} "
                    f"throttle={applied_ctrl.throttle:.2f} brake={applied_ctrl.brake:.2f} "
                    f"speed={speed_mps:.2f} applied={control_applied}"
                )
            vehicle.apply_control(applied_ctrl)
            if motion_log_file is not None:
                transform = vehicle.get_transform()
                loc = transform.location
                rot = transform.rotation
                vel = vehicle.get_velocity()
                acc = vehicle.get_acceleration()
                ang_vel = vehicle.get_angular_velocity()
                try:
                    snapshot = world.get_snapshot()
                    frame = int(snapshot.frame)
                    sim_time = float(snapshot.timestamp.elapsed_seconds)
                except RuntimeError:
                    frame = None
                    sim_time = None
                motion_log_file.write(
                    json.dumps(
                        {
                            "event": "control",
                            "wall_time": time.time(),
                            "elapsed_wall_time": time.time() - run_started_wall_time,
                            "frame": frame,
                            "sim_time": sim_time,
                            "pose": {
                                "x": float(loc.x),
                                "y": float(loc.y),
                                "z": float(loc.z),
                                "pitch": float(rot.pitch),
                                "yaw": float(rot.yaw),
                                "roll": float(rot.roll),
                            },
                            "velocity": {
                                "x": float(vel.x),
                                "y": float(vel.y),
                                "z": float(vel.z),
                                "speed_mps": float(speed_mps),
                            },
                            "acceleration": {
                                "x": float(acc.x),
                                "y": float(acc.y),
                                "z": float(acc.z),
                            },
                            "angular_velocity": {
                                "x": float(ang_vel.x),
                                "y": float(ang_vel.y),
                                "z": float(ang_vel.z),
                            },
                            "topomap": topomap_log,
                            "chosen": {
                                "trajectory_index": int(chosen_traj_idx),
                                "waypoint_index": int(waypoint_idx),
                                "waypoint_xy": [float(chosen[0]), float(chosen[1])],
                                "lookahead_time": float(lookahead_time),
                                "v": float(v),
                                "w": float(w),
                                "curvature": float(curvature),
                                "wheel_angle_deg": float(np.rad2deg(wheel_angle)),
                                "steer_command": float(steer_command),
                                "effective_max_steer": float(effective_max_steer),
                            },
                            "control_applied": bool(control_applied),
                            "proposed_control": {
                                "throttle": float(proposed_ctrl.throttle),
                                "steer": float(proposed_ctrl.steer),
                                "brake": float(proposed_ctrl.brake),
                                "hand_brake": bool(proposed_ctrl.hand_brake),
                                "reverse": bool(proposed_ctrl.reverse),
                            },
                            "control": {
                                "throttle": float(applied_ctrl.throttle),
                                "steer": float(applied_ctrl.steer),
                                "brake": float(applied_ctrl.brake),
                                "hand_brake": bool(applied_ctrl.hand_brake),
                                "reverse": bool(applied_ctrl.reverse),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                motion_log_file.flush()

            if preview_enabled and (
                (preview_source == "pinhole" and len(preview_queue) > 0)
                or preview_source == "observation"
            ):
                if preview_source == "observation":
                    vis = render_observation_preview(
                        obs_pil,
                        naction,
                        traj_colors,
                        chosen_traj_idx,
                        waypoint_idx,
                        speed_mps,
                        preview_w,
                        preview_h,
                    )
                else:
                    vis = render_preview_overlay(
                        preview_queue[-1],
                        preview_camera.get_transform(),
                        naction,
                        traj_colors,
                        chosen_traj_idx,
                        waypoint_idx,
                        vehicle.get_transform(),
                        speed_mps,
                        preview_w,
                        preview_h,
                        preview_fov,
                    )
                cv2.imshow("CARLA NoMaD Preview", vis)
                # q/Q 退出
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break

            if use_topomap:
                reached_goal = closest_node == goal_node
                if reached_goal:
                    print("Reached goal (topomap 索引). 车辆已停止。")

            time.sleep(rate)
    except KeyboardInterrupt:
        print("\n收到中断，正在停止部署。")
    finally:
        print("清理传感器与车辆…")
        if motion_log_file is not None:
            motion_log_file.write(
                json.dumps(
                    {
                        "event": "end",
                        "wall_time": time.time(),
                        "elapsed_wall_time": time.time() - run_started_wall_time,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            motion_log_file.close()
            print(f"[motion-log] Saved deployment log: {motion_log_path}")
        if preview_enabled:
            cv2.destroyAllWindows()
        if preview_camera is not None and preview_camera.is_alive:
            preview_camera.stop()
            preview_camera.destroy()
        if camera.is_alive:
            camera.stop()
            camera.destroy()
        if vehicle.is_alive:
            vehicle.destroy()


if __name__ == "__main__":
    main()
