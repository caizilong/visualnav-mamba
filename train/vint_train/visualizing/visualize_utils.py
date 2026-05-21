import numpy as np
from PIL import Image
import torch

VIZ_IMAGE_SIZE = (640, 480)          # 所有可视化图像统一 resize 的分辨率 (宽, 高)
RED = np.array([1, 0, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
CYAN = np.array([0, 1, 1])
YELLOW = np.array([1, 1, 0])
MAGENTA = np.array([1, 0, 1])


def numpy_to_img(arr: np.ndarray) -> Image:
    """
    将 [C, H, W]、数值范围 [0, 1] 的 numpy 图像张量转换为 PIL.Image，
    并统一 resize 到 VIZ_IMAGE_SIZE。
    """
    img = Image.fromarray(np.transpose(np.uint8(255 * arr), (1, 2, 0)))
    img = img.resize(VIZ_IMAGE_SIZE)
    return img


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """torch.Tensor → numpy.ndarray 的便捷转换（自动 detach + cpu）。"""
    return tensor.detach().cpu().numpy()


def from_numpy(array: np.ndarray) -> torch.Tensor:
    """numpy.ndarray → torch.FloatTensor 的便捷转换。"""
    return torch.from_numpy(array).float()
