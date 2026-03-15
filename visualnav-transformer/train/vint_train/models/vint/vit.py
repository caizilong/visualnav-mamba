import torch
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from vit_pytorch import SimpleViT
import pdb


class ViT(nn.Module):
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,
        context_size: int = 5,
        image_size: int = 128,
        patch_size: int = 16,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 4,
    ) -> None:
        """
        基于 Vision Transformer 的视觉编码器，用于 NoMaD 中的 ViT 版本。

        思路：
        - 将多帧观测和目标图像在宽度方向上拼接成一张大图
        - 使用 `MaskedGoalViT` 在 patch 级别上做自注意力建模
        - 通过 goal mask 控制是否让模型看到目标区域的 patch
        """
        super(ViT, self).__init__()
        self.context_size = context_size
        self.patch_size = patch_size
        if type(image_size) == int:
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_width = image_size[0]
            self.image_height = image_size[1]
        # MaskedGoalViT 的输入是单张大图：[B, 3, H, W*(context_size+2)]
        self.ViT = MaskedGoalViT(
            context_size=context_size,
            image_size=(self.image_height, self.image_width*(self.context_size + 2)),
            patch_size=self.patch_size,
            # embedding 维度设为 obs_encoding_size
            dim=encoding_size,
            depth = mha_num_attention_layers,
            heads = mha_num_attention_heads,
            mlp_dim = encoding_size
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_img: [B, 3 * context_size, H, W]，多帧观测
            goal_img: [B, 3, H, W]，目标图像
            input_goal_mask: [B]，取值 0 或 1，控制是否对目标区域做 mask

        Returns:
            final_repr: [B, obs_encoding_size]，整张拼接图像的全局表征
        """

        # 将多帧观测拆成单帧列表，再与目标图像一起在宽度维拼接：
        # obs_img_list: context_size 张 [B, 3, H, W]
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        obsgoal_img_list = obs_img_list + [goal_img]
        # x: [B, 3, H, W * (context_size+1 + 1)] = [B, 3, H, W * (context_size+2)]
        x = torch.cat(obsgoal_img_list, dim=-1)
        assert len(x.shape) == 4, "input image shape is not 4D"
        assert x.shape[1] == 3, "input image channel is not 3"
        assert x.shape[2] == self.image_height, f"input image height is not {self.image_height}"
        assert x.shape[3] == self.image_width*(self.context_size + 2), f"input image width is not {self.image_width}*(context_size + 2)"
       
        final_repr = self.ViT(x)
        
        return final_repr

# Helper Functions for ViT

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# Classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask):
        # 标准多头自注意力模块，额外支持一个加性 mask 用于屏蔽部分 patch 间的交互。
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        attn = self.attend(dots + mask) 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, mask):
        # 多层 stack，每层都是 Attention + FeedForward 的残差连接。
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return x


# Implementation of ViT with goal masking
class MaskedGoalViT(nn.Module):
    def __init__(self, *, context_size, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.h = image_height // patch_height
        self.w = image_width // patch_width
        patch_dim = channels * patch_height * patch_width

        # 将整张图像切分为 [h, w] 个 patch，并对每个 patch 做线性投影得到初始 token 表示
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

        # goal_mask: 标记哪些 patch 属于“观测区域”，哪些属于“目标区域”
        self.goal_mask = torch.ones((self.h, self.w))
        assert self.w % (context_size + 2) == 0, "context_size must be a factor of numbers of patches in width"
        self.goal_mask[:, -self.w//(context_size + 2):] = 0
        self.goal_mask = rearrange(self.goal_mask, 'h w -> (h w)')
        self.no_mask = torch.ones(self.h*self.w)
        # all_masks[0]: 不做空间 mask；all_masks[1]: 只允许在“非目标” patch 上进行平均
        self.all_masks = torch.stack([self.no_mask, self.goal_mask,], dim=0)
        self.no_cross_mask = torch.ones((self.h*self.w, self.h*self.w))
        self.goal_cross_mask = torch.ones((self.h*self.w, self.h*self.w))
        # 构造 cross-attention 的 pairwise mask：
        #   goal_cross_mask[i, j] = 0 表示 i 或 j 有一个属于“目标区域”，不允许它们互相注意。
        for i in range(self.h*self.w):
            for j in range(self.h*self.w):
                if self.goal_mask[i] + self.goal_mask[j] < 2:
                    self.goal_cross_mask[i, j] = 0
        self.all_cross_masks = torch.stack([self.no_cross_mask, self.goal_cross_mask], dim=0)
        # mean_mask 用于在做全局平均时，避免被 mask 掉的 patch 影响均值
        self.mean_mask = self.all_masks / self.all_masks.mean(dim=1, keepdim=True)

        # 将 0/1 mask 转成加性 mask，0 → 0, 1 → -1e9（相当于 softmax 前强行压制）
        self.all_cross_masks = torch.where(self.all_cross_masks == 0, -1e9, 0.0)
        self.all_masks = torch.where(self.all_masks == 0, -1e9, 0.0)


    def forward(self, img, input_goal_mask=None):
        """
        Args:
            img: [B, 3, H, W_total]，已经在宽度维拼接好的大图
            input_goal_mask: [B]，0 表示不 mask 目标区域，1 表示 mask 目标区域

        Returns:
            x: [B, dim]，对整张图像的全局编码
        """
        b, c, h, w, dtype = *img.shape, img.dtype
        device = img.device

        if input_goal_mask is None:
            # 默认不 mask（即全部可见）
            input_goal_mask = torch.zeros(b, dtype=torch.int64)

        # 根据 input_goal_mask 选择对应的 cross-attention mask
        final_mask = torch.index_select(self.all_cross_masks.to(device), 0, input_goal_mask.to(device))

        # 1) 切 patch + 线性投影 + 正余弦位置编码
        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        # 2) 送入 Transformer，带上 cross-attention mask
        x = self.transformer(x, mask=final_mask)

        # 3) 对输出 token 做加权平均，权重由 mean_mask 决定
        final_mask = torch.index_select(self.mean_mask.to(device), 0, input_goal_mask.to(device)).unsqueeze(-1)
        x = x * final_mask 
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return x