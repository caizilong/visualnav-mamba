import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

import timm

from vint_train.models.vint.self_attention import PositionalEncoding
from .nomad_vint import replace_bn_with_gn
from .mamba2 import Mamba2, MambaConfig


def _normalize_model_name(model_name: str) -> str:
    """
    将模型名称标准化为 timm 格式。
    例如: 'efficientnet-b0' -> 'efficientnet_b0'
    """
    return model_name.replace("-", "_")


def _create_timm_encoder(
    model_name: str,
    in_chans: int = 3,
    pretrained: bool = True,
    use_gn: bool = True,
) -> Tuple[nn.Module, int]:
    """
    使用 timm 创建视觉编码器。
    
    支持的模型类型包括但不限于：
    - EfficientNet 系列: efficientnet_b0, efficientnet_b1, ...
    - ResNet 系列: resnet18, resnet50, ...
    - ViT 系列: vit_tiny_patch16_224, vit_small_patch16_224, ...
    - DINOv2 系列: vit_small_patch14_dinov2, vit_base_patch14_dinov2, ...
    - ConvNeXt 系列: convnext_tiny, convnext_small, ...
    
    Args:
        model_name: timm 模型名称
        in_chans: 输入通道数（3 为 RGB，6 为拼接图像）
        pretrained: 是否使用预训练权重
        use_gn: 是否将 BatchNorm 替换为 GroupNorm（对小 batch size 更稳定）
    
    Returns:
        encoder: 视觉编码器模型
        num_features: 输出特征维度
    """
    # 标准化模型名称
    model_name = _normalize_model_name(model_name)
    
    # 创建 timm 模型
    encoder = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_chans,
        num_classes=0,  # 移除分类头，只保留特征提取部分
    )
    
    # 可选：将 BatchNorm 替换为 GroupNorm
    if use_gn:
        encoder = replace_bn_with_gn(encoder)
    
    num_features = encoder.num_features
    return encoder, num_features


def _extract_features(encoder: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    从 timm 编码器中提取特征，统一处理不同模型的输出格式。
    
    不同模型的 forward_features 输出：
    - CNN (EfficientNet, ResNet, ConvNeXt): [B, C, H, W]
    - ViT (包括 DINOv2): [B, num_tokens, dim] 或 [B, dim]
    
    Returns:
        features: [B, num_features] 的特征向量
    """
    feats = encoder.forward_features(x)
    
    if feats.ndim == 4:
        # CNN 输出: [B, C, H, W] -> [B, C]
        feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(start_dim=1)
    elif feats.ndim == 3:
        # ViT 输出: [B, num_tokens, dim]
        # 通常第一个 token 是 CLS token，或者做平均池化
        # 这里使用平均池化以保持通用性
        feats = feats.mean(dim=1)  # [B, dim]
    # 如果 feats.ndim == 2，已经是 [B, dim] 格式，无需处理
    
    return feats


class NoMaD_Mamba(nn.Module):
    """
    使用 Mamba2 代替 Transformer 的 NoMaD 视觉编码器版本。

    与原始 `NoMaD_ViNT` 的主要差别：
    - 使用 timm 库加载视觉编码器，支持多种 backbone（EfficientNet, ViT, DINOv2 等）；
    - 将 `TransformerEncoder` 替换为若干层 Mamba2 block，用于对
      [context_tokens + goal_token] 这一短序列进行建模；
    - 其余接口（输入/输出张量形状）保持不变，方便直接替换 `vision_encoder`。
    
    支持的 obs_encoder 类型（通过 timm 库）：
    - 'efficientnet-b0', 'efficientnet-b1', ...
    - 'resnet18', 'resnet50', ...
    - 'vit_tiny_patch16_224', 'vit_small_patch16_224', ...
    - 'vit_small_patch14_dinov2', 'vit_base_patch14_dinov2', ...
    - 'convnext_tiny', 'convnext_small', ...
    """

    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        goal_encoder: Optional[str] = None,  # 新增：可单独指定 goal 编码器，默认与 obs_encoder 相同
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,   # 保留接口但目前未使用
        mha_num_attention_layers: Optional[int] = 2,  # 对应为 Mamba 层数
        mha_ff_dim_factor: Optional[int] = 4,         # 未使用，仅为兼容
        mamba_cfg: Optional["MambaConfig"] = None,
    ) -> None:
        super().__init__()

        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size
        
        # 如果未指定 goal_encoder，则使用与 obs_encoder 相同的类型
        if goal_encoder is None:
            goal_encoder = obs_encoder

        # -------- 观测编码器：对单帧 RGB 图像做特征提取（timm） --------
        self.obs_encoder, self.num_obs_features = _create_timm_encoder(
            model_name=obs_encoder,
            in_chans=3,
            pretrained=True,
            use_gn=True,
        )
        self.obs_encoder_type = _normalize_model_name(obs_encoder)

        # -------- 目标编码器：对 "当前观测 + 目标" 拼接后的 6 通道图像编码 --------
        self.goal_encoder, self.num_goal_features = _create_timm_encoder(
            model_name=goal_encoder,
            in_chans=6,  # 当前观测(3) + 目标图像(3) 拼接
            pretrained=True,
            use_gn=True,
        )

        # 若 EfficientNet 输出维度与期望的 encoding_size 不一致，则用线性层压缩到统一维度
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # -------- 位置编码 + Mamba2 序列建模 --------
        self.positional_encoding = PositionalEncoding(
            self.obs_encoding_size,
            max_seq_len=self.context_size + 2,
        )

        if mamba_cfg is None or not isinstance(mamba_cfg, MambaConfig):
            mamba_cfg = MambaConfig()
        self.mamba_cfg = mamba_cfg

        def _make_mamba_layer(dim: int) -> "Mamba2":
            # 使用与 MTIL 中 MambaPolicy 相同的配置字段
            return Mamba2(
                d_model=dim,
                d_state=self.mamba_cfg.d_state,
                d_conv=self.mamba_cfg.d_conv,
                expand=self.mamba_cfg.expand,
                headdim=self.mamba_cfg.headdim,
                ngroups=self.mamba_cfg.ngroups,
                A_init_range=self.mamba_cfg.A_init_range,
                dt_min=self.mamba_cfg.dt_min,
                dt_max=self.mamba_cfg.dt_max,
                dt_init_floor=self.mamba_cfg.dt_init_floor,
                dt_limit=self.mamba_cfg.dt_limit,
                bias=self.mamba_cfg.mamba_bias,     #123
                conv_bias=self.mamba_cfg.mamba_conv_bias,       #123
                chunk_size=self.mamba_cfg.chunk_size,
                use_mem_eff_path=self.mamba_cfg.use_mem_eff_path,
            )

        # 使用 mha_num_attention_layers 作为 Mamba block 的层数
        self.mamba_layers = nn.ModuleList(
            [_make_mamba_layer(self.obs_encoding_size) for _ in range(mha_num_attention_layers)]
        )
        # Pre-LN：每层 Mamba 前做 LayerNorm，提升深层网络训练稳定性
        self.mamba_norms = nn.ModuleList(
            [nn.LayerNorm(self.obs_encoding_size) for _ in range(mha_num_attention_layers)]
        )

        # goal mask 相关，与 NoMaD_ViNT 保持一致；注册为 buffer 随模型自动迁移设备
        goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        goal_mask[:, -1] = True  # mask 掉最后一个 token（goal）
        no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        all_masks = torch.cat([no_mask, goal_mask], dim=0)
        avg_pool_mask = torch.cat(
            [
                1 - no_mask.float(),
                (1 - goal_mask.float()) * ((self.context_size + 2) / (self.context_size + 1)),
            ],
            dim=0,
        )
        self.register_buffer("goal_mask", goal_mask)
        self.register_buffer("no_mask", no_mask)
        self.register_buffer("all_masks", all_masks)
        self.register_buffer("avg_pool_mask", avg_pool_mask)

    def forward(
        self,
        obs_img: torch.Tensor,
        goal_img: torch.Tensor,
        input_goal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            obs_img: [B, 3 * (context_size+1), H, W] 的上下文观测图像（含当前帧）
            goal_img: [B, 3, H, W] 的目标图像
            input_goal_mask: [B]，取值 0 或 1，控制是否屏蔽 goal token

        Returns:
            obs_encoding_tokens: [B, obs_encoding_size] 的视觉 embedding，
                                 作为 NoMaD 主模型的条件输入。
        """
        device = obs_img.device

        # ------- 1) 目标编码 -------
        goal_mask = None
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # 当前观测的图像位于 obs_img 的最后一段通道上
        obsgoal_img = torch.cat(
            [obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1
        )  # [B, 6, H, W]
        # 使用统一的特征提取函数，支持 CNN 和 ViT 等不同架构
        obsgoal_encoding = _extract_features(self.goal_encoder, obsgoal_img)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if obsgoal_encoding.ndim == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)  # [B, 1, C]
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding  # [B, 1, C]

        # ------- 2) 观测编码：将多帧观测拆成单帧再堆叠 -------
        obs_split = torch.split(obs_img, 3, dim=1)  # context_size+1 张 [B, 3, H, W]
        obs_stack = torch.cat(obs_split, dim=0)     # [B*(context_size+1), 3, H, W]

        # 使用统一的特征提取函数
        obs_encoding = _extract_features(self.obs_encoder, obs_stack)
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # 形状恢复为 [B, context_size+1, C]，并拼接 goal token 得到序列
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape(
            (self.context_size + 1, -1, self.obs_encoding_size)
        )
        obs_encoding = torch.transpose(obs_encoding, 0, 1)  # [B, context_size+1, C]

        # ------- 3) 处理 goal mask -------
        # 与 Transformer 不同，Mamba 没有原生的 padding mask 机制
        # 但由于 Mamba 是因果模型，goal token 在序列末尾，不会影响前面观测 token 的计算
        # 为了更好地模拟 Transformer 的 mask 行为，当 goal_mask=1 时将 goal_encoding 置零
        # 这样 goal token 对最终 pooling 的贡献最小化
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()  # 0 or 1
            # 将 goal_mask 扩展为 [B, 1, C] 形状，用于逐元素乘法
            goal_mask_expand = goal_mask.float().unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            # 当 goal_mask=1 时，将 goal_encoding 置零；goal_mask=0 时保持不变
            goal_encoding = goal_encoding * (1 - goal_mask_expand)
            # all_masks 已注册为 buffer，自动与模型同设备
            src_key_padding_mask = torch.index_select(
                self.all_masks, 0, no_goal_mask
            )  # [B, seq_len]
        else:
            src_key_padding_mask = None

        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)  # [B, context_size+2, C]

        # ------- 4) 位置编码 + Mamba2 序列建模 -------
        x = tokens
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)

        # Pre-LN 残差：x = x + layer(LayerNorm(x))，先归一化再进 Mamba，梯度更稳定
        for layer, norm in zip(self.mamba_layers, self.mamba_norms):
            x = x + layer(norm(x))

        # ------- 5) 按 mask 做加权平均池化，得到最终 embedding -------
        if src_key_padding_mask is not None:
            # avg_pool_mask 已注册为 buffer，自动与模型同设备
            avg_mask = torch.index_select(
                self.avg_pool_mask, 0, no_goal_mask
            ).unsqueeze(-1)  # [B, seq_len, 1]
            x = x * avg_mask

        obs_encoding_tokens = x.mean(dim=1)  # [B, C]
        return obs_encoding_tokens

