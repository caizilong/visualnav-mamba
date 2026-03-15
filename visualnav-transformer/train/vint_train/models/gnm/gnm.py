import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from vint_train.models.gnm.modified_mobilenetv2 import MobileNetEncoder
from vint_train.models.base_model import BaseModel


class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        """
        GNM（Goal-Conditioned Navigation Model）主类。

        整体结构：
        - 使用一个 MobileNetEncoder 对“上下文 + 当前观测”做编码，得到 obs_encoding
        - 使用另一个 MobileNetEncoder 对“上下文 + 当前观测 + 目标”一起编码，得到 goal_encoding
        - 将二者拼接后送入全连接层，输出：
          - 到目标的距离 dist_pred
          - 未来轨迹 action_pred（在局部坐标下）

        Args:
            context_size (int): 历史观测帧数
            len_traj_pred (int): 需要预测的未来 waypoint 数
            learn_angle (bool): 是否预测 yaw（角度）
            obs_encoding_size (int): 观测编码向量维度
            goal_encoding_size (int): 目标编码向量维度
        """
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)
        # 用于编码 “上下文 + 当前观测” 图像序列的 MobileNet 编码器
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        # 将 MobileNet 的输出通道压缩到 obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        # 用于编码 “上下文 + 当前观测 + 目标” 的 MobileNet 编码器
        # 注意 num_images = context_size + 当前观测 + 目标 = 2 + context_size
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        # 将 goal 编码压缩到 goal_encoding_size
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )
        # 将 obs_encoding 与 goal_encoding 融合后的向量变换到一个紧凑表示 z
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        # 预测距离（标量）
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        # 预测未来 len_trajectory_pred 个 waypoint，每个 waypoint 有 num_action_params 维
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_img: [B, 3 * (context_size + 1), H, W]
                上下文 + 当前观测图像，按时间在通道维拼接
            goal_img: [B, 3, H, W]
                目标图像

        Returns:
            dist_pred: [B, 1]，到目标的离散时间距离预测
            action_pred: [B, len_traj_pred, num_action_params]，局部轨迹
        """
        # 1) 仅基于 “上下文 + 当前观测” 的编码
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        # 2) 将 obs 与 goal 在通道维拼接，再进行编码，学习两者之间的关系
        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        # 3) 融合两个编码，经过全连接网络得到紧凑表征 z
        z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)

        # 4) 调整形状并对轨迹做后处理
        #    - 先将线性层输出 reshape 成 [B, T, num_action_params]
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        #    - 前两维是 (dx, dy)，通过 cumsum 将“位移增量”变成相对原点的 waypoint
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            # 若预测角度，则 (cos, sin) 向量需要归一化到单位圆上，避免数值漂移
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred
