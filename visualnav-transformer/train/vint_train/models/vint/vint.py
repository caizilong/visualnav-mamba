import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import timm
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder


class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        ViNT 模型：使用 EfficientNet 编码视觉观测与目标，再通过 Transformer 解码出
        - 当前观测到目标的时间距离（dist_pred）
        - 在机器人局部坐标系下的未来轨迹（action_pred）

        Args:
            context_size (int): 用作上下文的历史观测帧数
            len_traj_pred (int): 未来需要预测的 waypoint 数量
            learn_angle (bool): 是否同时预测机器人的朝向（yaw）
            obs_encoder (str): 用于编码观测图像的 EfficientNet 架构名称（如 "efficientnet-b0"）
            obs_encoding_size (int): 观测编码维度
            late_fusion (bool): 为 True 时，obs 与 goal 分别编码；否则早期拼接后再编码
            mha_num_attention_heads (int): Transformer 中多头注意力的头数
            mha_num_attention_layers (int): Transformer encoder 堆叠层数
            mha_ff_dim_factor (int): Transformer 前馈层宽度相对 embed_dim 的放大倍数
        """
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size

        self.late_fusion = late_fusion
        # 根据 obs_encoder 名称构建 timm EfficientNet 特征提取 backbone
        if obs_encoder.split("-")[0] == "efficientnet":
            # timm 中 EfficientNet-B0 的名称为 "efficientnet_b0"
            obs_model_name = obs_encoder.replace("-", "_")
            self.obs_encoder = timm.create_model(
                obs_model_name,
                pretrained=True,
                in_chans=3,
                num_classes=0,  # 移除分类头，forward_features 输出特征
            )
            self.num_obs_features = self.obs_encoder.num_features

            goal_model_name = "efficientnet_b0"
            in_chans_goal = 3 if self.late_fusion else 6
            self.goal_encoder = timm.create_model(
                goal_model_name,
                pretrained=True,
                in_chans=in_chans_goal,
                num_classes=0,
            )
            self.num_goal_features = self.goal_encoder.num_features
        else:
            raise NotImplementedError(f"Unsupported obs_encoder: {obs_encoder}")
        
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # 基于所有 token（上下文 + 当前观测 + 目标）的特征，输出一个紧凑表征
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        输入:
            obs_img: shape [B, 3 * context_size, H, W] 的上下文图像序列（按时间拼在通道维）
            goal_img: shape [B, 3, H, W] 的目标图像

        输出:
            dist_pred: shape [B, 1] 的距离预测
            action_pred: shape [B, len_traj_pred, num_action_params] 的局部轨迹
        """

        # ---------- 编码目标图像 ----------
        if self.late_fusion:
            goal_feats = self.goal_encoder.forward_features(goal_img)
        else:
            obsgoal_img = torch.cat(
                [obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1
            )
            goal_feats = self.goal_encoder.forward_features(obsgoal_img)

        # 统一使用自适应平均池化将空间特征压缩为一维向量
        if goal_feats.ndim == 4:
            goal_encoding = F.adaptive_avg_pool2d(goal_feats, (1, 1)).flatten(start_dim=1)
        else:
            goal_encoding = goal_feats

        # 现在 goal_encoding: [batch_size, num_goal_features]
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)
        # 现在 goal_encoding: [batch_size, 1, self.goal_encoding_size]
        assert goal_encoding.shape[2] == self.goal_encoding_size
        
        # ---------- 编码上下文观测 ----------
        # 将 [B, 3 * context_size, H, W] 拆成 context_size 张 [B, 3, H, W]
        obs_img = torch.split(obs_img, 3, dim=1)

        # image size is [batch_size*self.context_size, 3, H, W]
        obs_img = torch.concat(obs_img, dim=0)

        # get the observation encoding
        obs_feats = self.obs_encoder.forward_features(obs_img)
        if obs_feats.ndim == 4:
            obs_encoding = F.adaptive_avg_pool2d(obs_feats, (1, 1)).flatten(start_dim=1)
        else:
            obs_encoding = obs_feats

        obs_encoding = self.compress_obs_enc(obs_encoding)
        # currently, the size is [batch_size*(self.context_size + 1), self.obs_encoding_size]
        # reshape the obs_encoding to [context + 1, batch, encoding_size], note that the order is flipped
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        # currently, the size is [batch_size, self.context_size+1, self.obs_encoding_size]

        # concatenate the goal encoding to the observation encoding
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)
        # currently, the size is [batch_size, 32]

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred