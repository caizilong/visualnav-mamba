# modified from PyTorch torchvision library
from typing import Callable, Any, Optional, List

import torch
from torch import Tensor
from torch import nn

from torchvision.ops.misc import ConvNormActivation
from torchvision.models._utils import _make_divisible
from torchvision.models.mobilenetv2 import InvertedResidual


class MobileNetEncoder(nn.Module):
    def __init__(
        self,
        num_images: int = 1,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        用于 GNM 的 MobileNetV2 编码器，来自 torchvision 源码的轻量修改版本。

        与标准 MobileNetV2 的主要区别：
        - 输入通道数变为 `num_images * 3`，用于将多帧 RGB 图像在通道维拼接后
          一起送入网络编码（用于上下文 + 当前观测 + 目标的联合编码）。

        Args:
            num_images (int): 输入中拼接的图像数量（每张 3 通道）
            num_classes (int): 分类头输出类别数（GNM 这里只使用 features 部分）
            width_mult (float): 宽度倍率（控制通道数）
            inverted_residual_setting: 主体网络结构配置
            round_nearest (int): 通道数向上取整到该整数的倍数
            block: InvertedResidual block 实现
            norm_layer: 归一化层类型（默认为 BatchNorm2d）
            dropout (float): 分类头使用的 dropout 概率
        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        # 首层卷积：输入通道 = num_images * 3
        features: List[nn.Module] = [
            ConvNormActivation(
                num_images * 3,
                input_channel,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        ]
        # 构建一系列 Inverted Residual block
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                )
                input_channel = output_channel
        # 构建最后一个 1x1 卷积层，将通道数扩展到 last_channel
        features.append(
            # Conv2dNormActivation(
            ConvNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        # 将所有特征层封装成 nn.Sequential
        self.features = nn.Sequential(*features)

        # 分类头（在 GNM 中一般不会用到，仅保留完整性）
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        # 权重初始化（与 torchvision 实现一致）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # 该函数保留自 torchvision实现，便于 TorchScript 支持。
        # 实际 forward 中直接调用此函数。
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
