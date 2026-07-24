import io
import os
from typing import Tuple, Union

import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF

VISUALIZATION_IMAGE_SIZE = (160, 120)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # kept for deployment utility compatibility; training no longer crops images


def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    # 确保 yaw 是标量值，处理可能的数组输入
    if isinstance(yaw, np.ndarray):
        yaw = float(yaw.flatten()[0])
    else:
        yaw = float(yaw)
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def img_path_to_data(
    path: Union[str, io.BytesIO],
    image_resize_size: Tuple[int, int],
) -> torch.Tensor:
    """Load an RGB image and resize directly as a CHW uint8 tensor."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        return TF.pil_to_tensor(img.resize(image_resize_size))
