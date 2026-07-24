#!/usr/bin/env python3
"""Shared CARLA camera helpers for fisheye data collection and deployment."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


DEFAULT_CAMERA_TYPE = "fisheye"
DEFAULT_FISHEYE_MODEL = "equidistant"
DEFAULT_RGB_FOV = 90.0
DEFAULT_FISHEYE_FOV = 180.0
DEFAULT_FOV_MASK = True
DEFAULT_FOV_FADE_SIZE = 0.0
DEFAULT_RECORD_IMAGE_SIZE = (224, 224)
DEFAULT_CAMERA_LOCATION = (2.0, 0.0, 1.6)
DEFAULT_CAMERA_ROTATION = (0.0, 0.0, 0.0)


def str_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def normalize_camera_type(camera_type: str) -> str:
    camera_type = str(camera_type).strip().lower()
    aliases = {
        "rgb": "rgb",
        "pinhole": "rgb",
        "front": "rgb",
        "fisheye": "fisheye",
        "wide": "fisheye",
        "wide_angle": "fisheye",
        "wide-angle": "fisheye",
    }
    if camera_type not in aliases:
        raise ValueError(
            f"Unsupported camera_type={camera_type!r}; expected 'rgb' or 'fisheye'."
        )
    return aliases[camera_type]


def bool_attr(value: bool) -> str:
    return "true" if bool(value) else "false"


def make_camera_transform(
    carla_module,
    location: Tuple[float, float, float] = DEFAULT_CAMERA_LOCATION,
    rotation: Tuple[float, float, float] = DEFAULT_CAMERA_ROTATION,
):
    pitch, yaw, roll = rotation
    return carla_module.Transform(
        carla_module.Location(x=float(location[0]), y=float(location[1]), z=float(location[2])),
        carla_module.Rotation(pitch=float(pitch), yaw=float(yaw), roll=float(roll)),
    )


def _set_if_present(bp, name: str, value: Any) -> None:
    if bp.has_attribute(name):
        bp.set_attribute(name, str(value))


def build_camera_blueprint(
    blueprint_library,
    camera_type: str = DEFAULT_CAMERA_TYPE,
    image_size: Tuple[int, int] = DEFAULT_RECORD_IMAGE_SIZE,
    rgb_fov: float = DEFAULT_RGB_FOV,
    fisheye_fov: float = DEFAULT_FISHEYE_FOV,
    fisheye_model: str = DEFAULT_FISHEYE_MODEL,
    fov_mask: bool = DEFAULT_FOV_MASK,
    fov_fade_size: float = DEFAULT_FOV_FADE_SIZE,
    gamma: Optional[float] = None,
    kannala_brandt_params: Optional[Iterable[float]] = None,
    sensor_tick: Optional[float] = None,
):
    camera_type = normalize_camera_type(camera_type)
    if camera_type == "fisheye":
        bp = blueprint_library.find("sensor.camera.rgb.wide_angle_lens")
        _set_if_present(bp, "fov", float(fisheye_fov))
        _set_if_present(bp, "camera_model", fisheye_model)
        _set_if_present(bp, "fov_mask", bool_attr(fov_mask))
        _set_if_present(bp, "fov_fade_size", float(fov_fade_size))
        _set_if_present(bp, "equirectangular", "false")
        _set_if_present(bp, "perspective", "false")
        if kannala_brandt_params is not None:
            for idx, value in enumerate(kannala_brandt_params):
                _set_if_present(bp, f"k{idx}", float(value))
    else:
        bp = blueprint_library.find("sensor.camera.rgb")
        _set_if_present(bp, "fov", float(rgb_fov))

    width, height = image_size
    bp.set_attribute("image_size_x", str(int(width)))
    bp.set_attribute("image_size_y", str(int(height)))
    if gamma is not None:
        _set_if_present(bp, "gamma", float(gamma))
    if sensor_tick is not None:
        _set_if_present(bp, "sensor_tick", float(sensor_tick))
    return bp


def transform_to_metadata(transform) -> Dict[str, Dict[str, float]]:
    loc = transform.location
    rot = transform.rotation
    return {
        "location": {"x": float(loc.x), "y": float(loc.y), "z": float(loc.z)},
        "rotation": {
            "pitch": float(rot.pitch),
            "yaw": float(rot.yaw),
            "roll": float(rot.roll),
        },
    }


def camera_metadata(
    bp,
    camera_type: str,
    image_size: Tuple[int, int],
    transform,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    camera_type = normalize_camera_type(camera_type)
    attrs = {}
    for attr_name in (
        "image_size_x",
        "image_size_y",
        "fov",
        "camera_model",
        "fov_mask",
        "fov_fade_size",
        "equirectangular",
        "perspective",
        "gamma",
        "sensor_tick",
        "k0",
        "k1",
        "k2",
        "k3",
    ):
        if bp.has_attribute(attr_name):
            attrs[attr_name] = str(bp.get_attribute(attr_name))

    metadata: Dict[str, Any] = {
        "camera_type": camera_type,
        "blueprint": bp.id,
        "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
        "transform": transform_to_metadata(transform),
        "attributes": attrs,
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def write_camera_metadata(output_dir: str, metadata: Mapping[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "camera_meta.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, sort_keys=True)

    try:
        import yaml
    except ImportError:
        return

    yaml_path = os.path.join(output_dir, "camera_meta.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(metadata), f, sort_keys=True, allow_unicode=True)


def add_camera_arguments(parser, default_camera_type: str = DEFAULT_CAMERA_TYPE) -> None:
    parser.add_argument(
        "--camera-type",
        choices=["rgb", "fisheye"],
        default=default_camera_type,
        help="Front observation camera type.",
    )
    parser.add_argument("--record-width", type=int, default=DEFAULT_RECORD_IMAGE_SIZE[0])
    parser.add_argument("--record-height", type=int, default=DEFAULT_RECORD_IMAGE_SIZE[1])
    parser.add_argument("--rgb-fov", type=float, default=DEFAULT_RGB_FOV)
    parser.add_argument("--fisheye-fov", type=float, default=DEFAULT_FISHEYE_FOV)
    parser.add_argument("--fisheye-model", default=DEFAULT_FISHEYE_MODEL)
    parser.add_argument(
        "--fov-mask",
        nargs="?",
        const=True,
        type=str_to_bool,
        default=DEFAULT_FOV_MASK,
        help="Draw pixels outside fisheye FOV as black.",
    )
    parser.add_argument("--fov-fade-size", type=float, default=DEFAULT_FOV_FADE_SIZE)
