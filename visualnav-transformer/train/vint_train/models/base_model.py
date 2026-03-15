import torch
import torch.nn as nn

from typing import List, Dict, Optional, Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        """
        视觉导航模型的基础父类，封装了一些通用配置：

        Args:
            context_size (int): 作为上下文使用的历史观测帧数（不含当前帧）
            len_traj_pred (int): 未来需要预测的 waypoint 数量
            learn_angle (bool): 是否预测机器人的朝向（yaw），
                                若为 True，则每个 waypoint 多出 2 维 sin/cos 表示角度
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size
        self.learn_angle = learn_angle
        self.len_trajectory_pred = len_traj_pred
        if self.learn_angle:
            # 每个 waypoint 的动作参数为 (dx, dy, cos(yaw), sin(yaw))
            self.num_action_params = 4  # last two dims are the cos and sin of the angle
        else:
            # 只预测位移 (dx, dy)
            self.num_action_params = 2

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        """
        将 CNN 提取的特征图做全局平均池化并展平为向量。

        典型输入形状: [B, C, H, W]
        输出形状: [B, C]
        """
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        所有子类都应实现的前向接口。

        Args:
            obs_img (torch.Tensor): 观测图像 batch，
                通常形状为 [B, 3 * context_size, H, W]（多帧按通道拼接）
            goal_img (torch.Tensor): 目标图像 batch，
                通常形状为 [B, 3, H, W]

        Returns:
            dist_pred (torch.Tensor): 预测的“观测到目标”的时间距离，
                形状一般为 [B, 1]
            action_pred (torch.Tensor): 预测的局部轨迹，
                形状为 [B, len_traj_pred, num_action_params]
        """
        raise NotImplementedError
