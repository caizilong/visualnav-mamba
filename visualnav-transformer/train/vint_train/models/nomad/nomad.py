import os
import argparse
import time
import pdb

import torch
import torch.nn as nn


class NoMaD(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        """
        NoMaD 主模型容器。

        其本质是对三个子网络的封装：
        - vision_encoder: 将 (obs, goal) 图像编码成一个 embedding（可选带 goal mask）
        - noise_pred_net: 条件扩散模型中的噪声预测 UNet（输入轨迹 + 条件 embedding）
        - dist_pred_net: MLP，用于根据 embedding 预测距离标签

        训练与推理过程中，通过字符串参数 `func_name` 来选择调用哪个子网络。
        """
        super(NoMaD, self).__init__()


        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        """
        统一的前向接口。

        Args:
            func_name (str): 指定调用的子模块：
                - "vision_encoder": kwargs 需要包含 obs_img, goal_img, input_goal_mask
                - "noise_pred_net": kwargs 需要包含 sample, timestep, global_cond
                - "dist_pred_net": kwargs 需要包含 obsgoal_cond
        """
        if func_name == "vision_encoder" :
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        """
        简单的多层感知机，用于从 embedding 中预测一个标量（例如距离）。

        结构：embedding_dim → embedding_dim/4 → embedding_dim/16 → 1
        """
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        # 将任意形状的输入展平成 [N, embedding_dim]，逐样本预测一个标量
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output



