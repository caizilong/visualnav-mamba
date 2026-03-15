import numpy as np
import os
from PIL import Image
from typing import Any, Iterable, Tuple

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import io
from typing import Union

VISUALIZATION_IMAGE_SIZE = (160, 120)   # 训练与可视化中使用的小分辨率图像 (宽, 高)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # 所有原始图像在进入模型前都会居中裁剪为 4:3 的宽高比



def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")


def yaw_rotmat(yaw: float) -> np.ndarray:
    # 确保 yaw 为 Python 标量，避免 yaw[0] 为 0 维数组时 np.cos/sin 返回数组导致 inhomogeneous array
    yaw = float(np.asarray(yaw).flat[0])
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    将一组全局坐标系下的位置，转换到以 (curr_pos, curr_yaw) 为原点/朝向的局部坐标系中。

    Args:
        positions: 要转换的点，shape (..., 2) 或 (..., 3)
        curr_pos: 当前机器人的位置（全局坐标）
        curr_yaw: 当前机器人的航向角（全局坐标）
    Returns:
        np.ndarray: 转换到局部坐标系后的点
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    计算相邻 waypoint 之间的差分（增量）。

    - 若每个 waypoint 有 2 维 (x, y)，则返回 (dx, dy)
    - 若有 3 维 (x, y, yaw)，则对 yaw 部分额外转为 (cos, sin) 表示
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    将最后一维角度（弧度）替换为 (cos(theta), sin(theta)) 的 2 维向量，
    得到 (dx, dy, cos(theta), sin(theta)) 形式的动作表示。
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def transform_images(
    img: Image.Image, transform: transforms, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    """
    对单张 PIL 图像做：
    1) 按 4:3 宽高比做中心裁剪
    2) 生成一个用于可视化的小尺寸版本（VISUALIZATION_IMAGE_SIZE）
    3) 再 resize 到模型输入所需的 image_resize_size，并应用给定 transform

    Returns:
        viz_img: 供可视化使用的小图，Tensor 形式
        transf_img: 供模型输入使用的标准化后图像 Tensor
    """
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    viz_img = img.resize(VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    img = img.resize(image_resize_size)
    transf_img = transform(img)
    return viz_img, transf_img


def resize_and_aspect_crop(
    img: Image.Image, image_resize_size: Tuple[int, int], aspect_ratio: float = IMAGE_ASPECT_RATIO
):
    """
    只做裁剪到目标宽高比 + resize 的版本（不做额外归一化），
    常用于离线数据处理或在 Dataset 中快速得到 channels-first 图像张量。
    """
    w, h = img.size
    if w > h:
        img = TF.center_crop(img, (h, int(h * aspect_ratio)))  # crop to the right ratio
    else:
        img = TF.center_crop(img, (int(w / aspect_ratio), w))
    img = img.resize(image_resize_size)
    resize_img = TF.to_tensor(img)
    return resize_img


def img_path_to_data(path: Union[str, io.BytesIO], image_resize_size: Tuple[int, int]) -> torch.Tensor:
    """
    从磁盘/内存路径读取一张图像，并按统一的裁剪 + resize 规则转换为 Tensor。

    Args:
        path: 图像路径或 BytesIO 对象
        image_resize_size: 模型输入所需的图像尺寸
    Returns:
        torch.Tensor: 处理后的图像张量（channels-first, 已缩放到 [0,1]）
    """
    # return transform_images(Image.open(path), transform, image_resize_size, aspect_ratio)
    return resize_and_aspect_crop(Image.open(path), image_resize_size)    

