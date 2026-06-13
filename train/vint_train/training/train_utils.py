import itertools
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import tqdm
import wandb
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE
from vint_train.training.logger import Logger
from vint_train.visualizing.action_utils import plot_trajs_and_points
from vint_train.visualizing.visualize_utils import to_numpy

with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

ACTION_STATS = {}
for key in data_config["action_stats"]:
    ACTION_STATS[key] = np.array(data_config["action_stats"][key])

def _action_stat_tensor(stats, key: str, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(stats[key], dtype=torch.float32, device=ref.device)


def normalize_data_torch(data: torch.Tensor, stats=ACTION_STATS) -> torch.Tensor:
    data = data.float()
    stat_min = _action_stat_tensor(stats, "min", data)
    stat_max = _action_stat_tensor(stats, "max", data)
    ndata = (data - stat_min) / (stat_max - stat_min)
    return ndata * 2 - 1


def unnormalize_data_torch(ndata: torch.Tensor, stats=ACTION_STATS) -> torch.Tensor:
    ndata = ndata.float()
    stat_min = _action_stat_tensor(stats, "min", ndata)
    stat_max = _action_stat_tensor(stats, "max", ndata)
    data = (ndata + 1) / 2
    return data * (stat_max - stat_min) + stat_min


def get_delta_torch(actions: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros(
        (actions.shape[0], 1, actions.shape[-1]),
        dtype=actions.dtype,
        device=actions.device,
    )
    ex_actions = torch.cat((zeros, actions), dim=1)
    return ex_actions[:, 1:] - ex_actions[:, :-1]


def normalize_image_tensor(images: torch.Tensor) -> torch.Tensor:
    channels = images.shape[1]
    if channels % 3 != 0:
        raise ValueError(f"Expected image channels to be a multiple of 3, got {channels}")
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], dtype=images.dtype, device=images.device
    ).view(1, 3, 1, 1)
    repeats = channels // 3
    if repeats > 1:
        mean = mean.repeat(1, repeats, 1, 1)
        std = std.repeat(1, repeats, 1, 1)
    return (images - mean) / std


# Train utils for NOMAD

def _compute_losses_nomad(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    batch_dist_label: torch.Tensor,
    batch_action_label: torch.Tensor,
    device: torch.device,
    action_mask: torch.Tensor,
    guidance_scale_min: float = 0.25,
    guidance_scale_max: float = 1.75,
    guidance_scale_power: float = 1.5,
    use_adaptive_guidance: bool = True,
    guidance_confidence_weight: float = 0.35,
    guidance_uncertainty_weight: float = 0.25,
    guidance_distance_scale: float = 10.0,
    generator: Optional[torch.Generator] = None,
):
    """
    对 NoMaD 的 EMA 模型进行一次前向推理并计算损失与指标。

    - uc_*: 在“只看当前观测（目标 mask 掉）”条件下采样得到的动作
    - gc_*: 在“观测 + 目标都可见”的条件下采样得到的动作与距离
    """

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        batch_goal_images,
        pred_horizon,
        action_dim,
        num_samples=1,
        device=device,
        guidance_scale_min=guidance_scale_min,
        guidance_scale_max=guidance_scale_max,
        guidance_scale_power=guidance_scale_power,
        use_adaptive_guidance=use_adaptive_guidance,
        guidance_confidence_weight=guidance_confidence_weight,
        guidance_uncertainty_weight=guidance_uncertainty_weight,
        guidance_distance_scale=guidance_distance_scale,
        generator=generator,
    )
    uc_actions = model_output_dict['uc_actions']
    gc_actions = model_output_dict['gc_actions']
    gc_distance = model_output_dict['gc_distance']

    gc_dist_loss = F.mse_loss(gc_distance, batch_dist_label.unsqueeze(-1))

    def action_reduce(unreduced_loss: torch.Tensor):
        # 在时间和维度上平均，再用 action_mask 做归一化加权。
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        assert unreduced_loss.shape == action_mask.shape, f"{unreduced_loss.shape} != {action_mask.shape}"
        return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)

    # Mask out invalid inputs (for negatives, or when the distance between obs and goal is large)
    assert uc_actions.shape == batch_action_label.shape, f"{uc_actions.shape} != {batch_action_label.shape}"
    assert gc_actions.shape == batch_action_label.shape, f"{gc_actions.shape} != {batch_action_label.shape}"

    uc_action_loss = action_reduce(F.mse_loss(uc_actions, batch_action_label, reduction="none"))
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, batch_action_label, reduction="none"))

    uc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        uc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    uc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(uc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    gc_action_waypts_cos_similairity = action_reduce(F.cosine_similarity(
        gc_actions[:, :, :2], batch_action_label[:, :, :2], dim=-1
    ))
    gc_multi_action_waypts_cos_sim = action_reduce(F.cosine_similarity(
        torch.flatten(gc_actions[:, :, :2], start_dim=1),
        torch.flatten(batch_action_label[:, :, :2], start_dim=1),
        dim=-1,
    ))

    results = {
        "uc_action_loss": uc_action_loss,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_similairity,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim,
        "gc_dist_loss": gc_dist_loss,
        "gc_action_loss": gc_action_loss,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_similairity,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim,
    }

    return results


def _compute_navigation_aux_loss(
    aux_outputs,
    goal_pos: torch.Tensor,
    distance: torch.Tensor,
    goal_mask: torch.Tensor,
    negative_distance_threshold: float,
    goal_pos_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
):
    if aux_outputs is None:
        zero = goal_pos.new_tensor(0.0)
        return zero, {"nav_goal_pos_loss": zero, "nav_contrastive_loss": zero}

    valid_goal_mask = (
        (1 - goal_mask.float()) * (distance.float() < negative_distance_threshold).float()
    )
    denom = valid_goal_mask.sum() + 1e-2

    goal_pos_pred = aux_outputs.get("goal_pos_pred")
    if goal_pos_pred is not None and goal_pos_weight != 0:
        goal_pos_loss = F.mse_loss(goal_pos_pred, goal_pos, reduction="none").mean(dim=-1)
        goal_pos_loss = (goal_pos_loss * valid_goal_mask).sum() / denom
    else:
        goal_pos_loss = goal_pos.new_tensor(0.0)

    contrastive_loss = goal_pos.new_tensor(0.0)
    obs_embedding = aux_outputs.get("obs_nav_embedding")
    goal_embedding = aux_outputs.get("goal_nav_embedding")
    pos_idx = torch.nonzero(valid_goal_mask > 0, as_tuple=False).flatten()
    if (
        obs_embedding is not None
        and goal_embedding is not None
        and contrastive_weight != 0
        and pos_idx.numel() > 1
    ):
        obs_embedding = F.normalize(obs_embedding.index_select(0, pos_idx), dim=-1)
        goal_embedding = F.normalize(goal_embedding.index_select(0, pos_idx), dim=-1)
        logits = obs_embedding @ goal_embedding.T / max(float(contrastive_temperature), 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )

    aux_loss = goal_pos_weight * goal_pos_loss + contrastive_weight * contrastive_loss
    return aux_loss, {
        "nav_goal_pos_loss": goal_pos_loss.detach(),
        "nav_contrastive_loss": contrastive_loss.detach(),
    }


def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    sampling_metrics_freq: int = 1000,
    use_wandb: bool = True,
    goal_guidance_min: float = 0.25,
    goal_guidance_max: float = 1.75,
    goal_guidance_power: float = 1.5,
    use_adaptive_guidance: bool = True,
    guidance_confidence_weight: float = 0.35,
    guidance_uncertainty_weight: float = 0.25,
    guidance_distance_scale: float = 10.0,
    nav_goal_pos_loss_weight: float = 0.05,
    nav_contrastive_loss_weight: float = 0.01,
    nav_contrastive_temperature: float = 0.1,
    aux_negative_distance_threshold: float = 20.0,
    max_grad_norm: Optional[float] = None,
):
    """
    训练 NoMaD 扩散策略模型一个 epoch。

    训练目标：
    - 使用噪声调度器将“归一化动作序列”加噪到不同时间步
    - 训练模型预测噪声残差（diffusion_loss）
    - 同时预测距离（dist_loss），并用 alpha 组合二者：
        loss = alpha * dist_loss + (1 - alpha) * diffusion_loss

    Args:
        model: 主模型（含 vision_encoder / noise_pred_net / dist_pred_net）
        ema_model: 模型参数的指数滑动平均，用于更稳定的评估和可视化
        optimizer: 优化器
        dataloader: 训练集 DataLoader
        transform: 图像预处理
        device: 训练设备
        noise_scheduler: DDPM 噪声调度器
        goal_mask_prob: 训练时随机 mask 掉目标 token 的概率
        project_folder: 工程目录
        epoch: 当前 epoch
        alpha: 距离损失在总损失中的权重（数值一般较小）
        print_log_freq: 打印日志频率
        image_log_freq: 可视化频率
        num_images_log: 可视化时使用的样本数
        use_wandb: 是否记录到 wandb
        goal_guidance_*: 与配置 goal_guidance_* 一致，用于日志/可视化中的扩散采样
    """
    goal_mask_prob = float(torch.clip(torch.tensor(goal_mask_prob), 0, 1).item())
    model.train()
    num_batches = len(dataloader)
    non_blocking = device.type == "cuda"
    log_window_size = max(int(print_log_freq), 1)

    ema_eval_model = ema_model.averaged_model
    epoch_loss_sum = 0.0

    uc_action_loss_logger = Logger("uc_action_loss", "train", window_size=log_window_size)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", "train", window_size=log_window_size
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", "train", window_size=log_window_size
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", "train", window_size=log_window_size)
    gc_action_loss_logger = Logger("gc_action_loss", "train", window_size=log_window_size)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", "train", window_size=log_window_size
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", "train", window_size=log_window_size
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image, 
                goal_image,
                actions,
                distance,
                goal_pos,
                dataset_idx,
                action_mask, 
            ) = data
            
            # -------- 1) 图像预处理与可视化图像抽取 --------
            should_log_images = image_log_freq != 0 and i % image_log_freq == 0
            if should_log_images:
                last_obs_image = obs_image[:, -3:, :, :]
                batch_viz_obs_images = TF.resize(last_obs_image, VISUALIZATION_IMAGE_SIZE[::-1])
                batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
            else:
                batch_viz_obs_images = None
                batch_viz_goal_images = None
            batch_obs_images = normalize_image_tensor(
                obs_image.to(device, non_blocking=non_blocking)
            )
            batch_goal_images = normalize_image_tensor(
                goal_image.to(device, non_blocking=non_blocking)
            )
            actions = actions.to(device, non_blocking=non_blocking)
            action_mask = action_mask.to(device, non_blocking=non_blocking)
            goal_pos = goal_pos.float().to(device, non_blocking=non_blocking)
            distance = distance.float().to(device, non_blocking=non_blocking)

            B = actions.shape[0]

            # Generate random goal mask
            goal_mask = torch.rand((B,), device=device) < goal_mask_prob
            goal_mask = goal_mask.long()
            obsgoal_cond = model(
                "vision_encoder",
                obs_img=batch_obs_images,
                goal_img=batch_goal_images,
                input_goal_mask=goal_mask,
            )
            aux_outputs = model("vision_aux")

            # 将绝对动作序列转成相邻 step 之间的增量，再根据数据集统计量归一化到 [-1, 1]
            deltas = get_delta_torch(actions)
            naction = normalize_data_torch(deltas, ACTION_STATS)
            assert naction.shape[-1] == 2, "action dim must be 2"

            # -------- 2) 距离预测损失 --------
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            # 逐样本计算距离损失，再屏蔽掉训练时被 goal-mask 的样本。
            dist_loss_per_sample = nn.functional.mse_loss(
                dist_pred.squeeze(-1),
                distance,
                reduction="none",
            )
            valid_dist_mask = 1 - goal_mask.float()
            dist_loss = (dist_loss_per_sample * valid_dist_mask).sum() / (
                valid_dist_mask.sum() + 1e-2
            )
            nav_aux_loss, nav_aux_logs = _compute_navigation_aux_loss(
                aux_outputs,
                goal_pos,
                distance,
                goal_mask,
                aux_negative_distance_threshold,
                nav_goal_pos_loss_weight,
                nav_contrastive_loss_weight,
                nav_contrastive_temperature,
            )

            def action_reduce(unreduced_loss: torch.Tensor):
                # Reduce over non-batch dimensions to get loss per batch element
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                assert unreduced_loss.shape == action_mask.shape, (
                    f"{unreduced_loss.shape} != {action_mask.shape}"
                )
                return (unreduced_loss * action_mask).mean() / (
                    action_mask.mean() + 1e-2
                )

            # Sample noise to add to actions：为每条轨迹采样高斯噪声
            noise = torch.randn_like(naction)

            # Sample a diffusion iteration for each data point：每个样本随机选择一个时间步
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            ).long()

            # Add noise：根据对应时间步的噪声水平，把轨迹从“干净”推到“噪声域”
            noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)

            # 预测噪声残差，使得在该时间步可以从 noisy_action 恢复出原始 clean action
            noise_pred = model(
                "noise_pred_net",
                sample=noisy_action,
                timestep=timesteps,
                global_cond=obsgoal_cond,
            )

            # L2 loss：在时间和维度上求平均，再用 action_mask 做样本级加权
            diffusion_loss = action_reduce(
                F.mse_loss(noise_pred, noise, reduction="none")
            )

            # Total loss：距离、扩散与导航辅助损失的加权和
            loss = alpha * dist_loss + (1 - alpha) * diffusion_loss + nav_aux_loss

            # 反向传播并更新参数
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # 同步更新 EMA 模型，用于评估与可视化
            ema_model.step(model)

            # Logging
            loss_cpu = loss.item()
            epoch_loss_sum += loss_cpu
            tepoch.set_postfix(loss=loss_cpu)
            wandb_payload = None
            if use_wandb and wandb_log_freq != 0 and i % wandb_log_freq == 0:
                wandb_payload = {
                    "epoch": epoch,
                    "total_loss": loss_cpu,
                    "dist_loss": dist_loss.item(),
                    "diffusion_loss": diffusion_loss.item(),
                    "nav_aux_loss": nav_aux_loss.item(),
                    "nav_goal_pos_loss": nav_aux_logs["nav_goal_pos_loss"].item(),
                    "nav_contrastive_loss": nav_aux_logs["nav_contrastive_loss"].item(),
                }


            should_sample_metrics = sampling_metrics_freq != 0 and i % sampling_metrics_freq == 0
            if should_sample_metrics:
                ema_was_training = ema_eval_model.training
                ema_eval_model.eval()
                with torch.inference_mode():
                    losses = _compute_losses_nomad(
                        ema_eval_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        distance,
                        actions,
                        device,
                        action_mask,
                        guidance_scale_min=goal_guidance_min,
                        guidance_scale_max=goal_guidance_max,
                        guidance_scale_power=goal_guidance_power,
                        use_adaptive_guidance=use_adaptive_guidance,
                        guidance_confidence_weight=guidance_confidence_weight,
                        guidance_uncertainty_weight=guidance_uncertainty_weight,
                        guidance_distance_scale=guidance_distance_scale,
                    )
                if ema_was_training:
                    ema_eval_model.train()
                
                for key, value in losses.items():
                    if key in loggers:
                        logger = loggers[key]
                        logger.log_data(value.item())
            
                data_log = {}
                for key, logger in loggers.items():
                    data_log[logger.full_name()] = logger.latest()
                    print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                if wandb_payload is not None:
                    wandb_payload.update(data_log)

            if wandb_payload is not None:
                wandb.log(wandb_payload, commit=True)

            if should_log_images:
                ema_was_training = ema_eval_model.training
                ema_eval_model.eval()
                with torch.inference_mode():
                    visualize_diffusion_action_distribution(
                        ema_eval_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        batch_viz_obs_images,
                        batch_viz_goal_images,
                        actions,
                        distance,
                        goal_pos,
                        device,
                        "train",
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_wandb,
                        goal_guidance_min=goal_guidance_min,
                        goal_guidance_max=goal_guidance_max,
                        goal_guidance_power=goal_guidance_power,
                        use_adaptive_guidance=use_adaptive_guidance,
                        guidance_confidence_weight=guidance_confidence_weight,
                        guidance_uncertainty_weight=guidance_uncertainty_weight,
                        guidance_distance_scale=guidance_distance_scale,
                    )
                if ema_was_training:
                    ema_eval_model.train()
    return epoch_loss_sum / max(num_batches, 1)


def evaluate_nomad(
    eval_type: str,
    ema_model: EMAModel,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    eval_fraction: float = 0.25,
    use_wandb: bool = True,
    sampling_metrics_freq: int = 1000,
    goal_guidance_min: float = 0.25,
    goal_guidance_max: float = 1.75,
    goal_guidance_power: float = 1.5,
    use_adaptive_guidance: bool = True,
    guidance_confidence_weight: float = 0.35,
    guidance_uncertainty_weight: float = 0.25,
    guidance_distance_scale: float = 10.0,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        ema_model (nn.Module): exponential moving average version of model to evaluate
        dataloader (DataLoader): dataloader for eval
        transform (transforms): transform to apply to images
        device (torch.device): device to use for evaluation
        noise_scheduler: noise scheduler to evaluate with 
        project_folder (string): path to project folder
        epoch (int): current epoch
        print_log_freq (int): how often to print logs 
        wandb_log_freq (int): how often to log to wandb
        image_log_freq (int): how often to log images
        alpha (float): weight for action loss
        num_images_log (int): number of images to log
        eval_fraction (float): fraction of data to use for evaluation
        use_wandb (bool): whether to use wandb for logging
    """
    goal_mask_prob = float(torch.clip(torch.tensor(goal_mask_prob), 0, 1).item())
    ema_model = ema_model.averaged_model
    ema_model.eval()
    non_blocking = device.type == "cuda"
    log_window_size = max(int(print_log_freq), 1)
    num_batches = len(dataloader)
    eval_generator = torch.Generator()
    eval_generator.manual_seed(0)

    uc_action_loss_logger = Logger("uc_action_loss", eval_type, window_size=log_window_size)
    uc_action_waypts_cos_sim_logger = Logger(
        "uc_action_waypts_cos_sim", eval_type, window_size=log_window_size
    )
    uc_multi_action_waypts_cos_sim_logger = Logger(
        "uc_multi_action_waypts_cos_sim", eval_type, window_size=log_window_size
    )
    gc_dist_loss_logger = Logger("gc_dist_loss", eval_type, window_size=log_window_size)
    gc_action_loss_logger = Logger("gc_action_loss", eval_type, window_size=log_window_size)
    gc_action_waypts_cos_sim_logger = Logger(
        "gc_action_waypts_cos_sim", eval_type, window_size=log_window_size
    )
    gc_multi_action_waypts_cos_sim_logger = Logger(
        "gc_multi_action_waypts_cos_sim", eval_type, window_size=log_window_size
    )
    loggers = {
        "uc_action_loss": uc_action_loss_logger,
        "uc_action_waypts_cos_sim": uc_action_waypts_cos_sim_logger,
        "uc_multi_action_waypts_cos_sim": uc_multi_action_waypts_cos_sim_logger,
        "gc_dist_loss": gc_dist_loss_logger,
        "gc_action_loss": gc_action_loss_logger,
        "gc_action_waypts_cos_sim": gc_action_waypts_cos_sim_logger,
        "gc_multi_action_waypts_cos_sim": gc_multi_action_waypts_cos_sim_logger,
    }
    num_batches = max(int(num_batches * eval_fraction), 1)

    with torch.inference_mode():
        with tqdm.tqdm(
            itertools.islice(dataloader, num_batches), 
            total=num_batches, 
            dynamic_ncols=True, 
            desc=f"Evaluating {eval_type} for epoch {epoch}", 
            leave=False) as tepoch:
            for i, data in enumerate(tepoch):
                (
                    obs_image, 
                    goal_image,
                    actions,
                    distance,
                    goal_pos,
                    dataset_idx,
                    action_mask,
                ) = data
                
                should_log_images = image_log_freq != 0 and i % image_log_freq == 0
                if should_log_images:
                    last_obs_image = obs_image[:, -3:, :, :]
                    batch_viz_obs_images = TF.resize(last_obs_image, VISUALIZATION_IMAGE_SIZE[::-1])
                    batch_viz_goal_images = TF.resize(goal_image, VISUALIZATION_IMAGE_SIZE[::-1])
                else:
                    batch_viz_obs_images = None
                    batch_viz_goal_images = None
                batch_obs_images = normalize_image_tensor(
                    obs_image.to(device, non_blocking=non_blocking)
                )
                batch_goal_images = normalize_image_tensor(
                    goal_image.to(device, non_blocking=non_blocking)
                )
                actions = actions.to(device, non_blocking=non_blocking)
                action_mask = action_mask.to(device, non_blocking=non_blocking)
                distance = distance.float().to(device, non_blocking=non_blocking)

                B = actions.shape[0]

                # Generate random goal mask
                rand_goal_mask = (
                    torch.rand((B,), generator=eval_generator) < goal_mask_prob
                ).long().to(device, non_blocking=non_blocking)
                goal_mask = torch.ones_like(rand_goal_mask).long().to(device, non_blocking=non_blocking)
                no_mask = torch.zeros_like(rand_goal_mask).long().to(device, non_blocking=non_blocking)

                rand_mask_cond = ema_model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    goal_img=batch_goal_images,
                    input_goal_mask=rand_goal_mask,
                )

                obsgoal_cond = ema_model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    goal_img=batch_goal_images,
                    input_goal_mask=no_mask,
                )
                obsgoal_cond = obsgoal_cond.flatten(start_dim=1)

                goal_mask_cond = ema_model(
                    "vision_encoder",
                    obs_img=batch_obs_images,
                    goal_img=batch_goal_images,
                    input_goal_mask=goal_mask,
                )

                deltas = get_delta_torch(actions)
                naction = normalize_data_torch(deltas, ACTION_STATS)
                assert naction.shape[-1] == 2, "action dim must be 2"

                # Sample noise to add to actions
                noise = torch.randn(naction.shape, generator=eval_generator).to(
                    device, non_blocking=non_blocking
                )

                # Sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (B,),
                    generator=eval_generator,
                ).long().to(device, non_blocking=non_blocking)

                noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                ### RANDOM MASK ERROR ###
                # Predict the noise residual
                rand_mask_noise_pred = ema_model(
                    "noise_pred_net",
                    sample=noisy_actions,
                    timestep=timesteps,
                    global_cond=rand_mask_cond,
                )

                # L2 loss
                rand_mask_loss = nn.functional.mse_loss(rand_mask_noise_pred, noise)

                ### NO MASK ERROR ###
                # Predict the noise residual
                no_mask_noise_pred = ema_model(
                    "noise_pred_net",
                    sample=noisy_actions,
                    timestep=timesteps,
                    global_cond=obsgoal_cond,
                )

                # L2 loss
                no_mask_loss = nn.functional.mse_loss(no_mask_noise_pred, noise)

                ### GOAL MASK ERROR ###
                # predict the noise residual
                goal_mask_noise_pred = ema_model(
                    "noise_pred_net",
                    sample=noisy_actions,
                    timestep=timesteps,
                    global_cond=goal_mask_cond,
                )

                # L2 loss
                goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)

                # Logging
                loss_cpu = rand_mask_loss.item()
                tepoch.set_postfix(loss=loss_cpu)

                wandb_payload = None
                if use_wandb and wandb_log_freq != 0 and i % wandb_log_freq == 0:
                    wandb_payload = {
                        "epoch": epoch,
                        "diffusion_eval_loss (random masking)": rand_mask_loss.item(),
                        "diffusion_eval_loss (no masking)": no_mask_loss.item(),
                        "diffusion_eval_loss (goal masking)": goal_mask_loss.item(),
                    }

                should_sample_metrics = sampling_metrics_freq != 0 and i % sampling_metrics_freq == 0
                if should_sample_metrics:
                    losses = _compute_losses_nomad(
                        ema_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        distance,
                        actions,
                        device,
                        action_mask,
                        guidance_scale_min=goal_guidance_min,
                        guidance_scale_max=goal_guidance_max,
                        guidance_scale_power=goal_guidance_power,
                        use_adaptive_guidance=use_adaptive_guidance,
                        guidance_confidence_weight=guidance_confidence_weight,
                        guidance_uncertainty_weight=guidance_uncertainty_weight,
                        guidance_distance_scale=guidance_distance_scale,
                        generator=eval_generator,
                    )
                    
                    for key, value in losses.items():
                        if key in loggers:
                            logger = loggers[key]
                            logger.log_data(value.item())
                
                    data_log = {}
                    for key, logger in loggers.items():
                        data_log[logger.full_name()] = logger.latest()
                        print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {logger.display()}")

                    if wandb_payload is not None:
                        wandb_payload.update(data_log)

                if wandb_payload is not None:
                    wandb.log(wandb_payload, commit=True)

                if should_log_images:
                    visualize_diffusion_action_distribution(
                        ema_model,
                        noise_scheduler,
                        batch_obs_images,
                        batch_goal_images,
                        batch_viz_obs_images,
                        batch_viz_goal_images,
                        actions,
                        distance,
                        goal_pos,
                        device,
                        eval_type,
                        project_folder,
                        epoch,
                        num_images_log,
                        30,
                        use_wandb,
                        goal_guidance_min=goal_guidance_min,
                        goal_guidance_max=goal_guidance_max,
                        goal_guidance_power=goal_guidance_power,
                        use_adaptive_guidance=use_adaptive_guidance,
                        guidance_confidence_weight=guidance_confidence_weight,
                        guidance_uncertainty_weight=guidance_uncertainty_weight,
                        guidance_distance_scale=guidance_distance_scale,
                        generator=eval_generator,
                    )


# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_delta(actions):
    # append zeros to first action
    ex_actions = np.concatenate([np.zeros((actions.shape[0],1,actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:,1:] - ex_actions[:,:-1]
    return delta

def get_action(diffusion_output, action_stats=ACTION_STATS):
    # diffusion_output: (B, 2*T+1, 1)
    # return: (B, T-1)
    ndeltas = diffusion_output
    ndeltas = ndeltas.reshape(ndeltas.shape[0], -1, 2)
    ndeltas = unnormalize_data_torch(ndeltas, action_stats)
    return torch.cumsum(ndeltas, dim=1)


def diffusion_guidance_scale(
    step_idx: int,
    total_steps: int,
    min_scale: float = 0.25,
    max_scale: float = 1.75,
    power: float = 1.5,
    goal_confidence: Optional[torch.Tensor] = None,
    action_uncertainty: Optional[torch.Tensor] = None,
    confidence_weight: float = 0.0,
    uncertainty_weight: float = 0.0,
):
    """前期更接近无条件分支，后期增强目标 guidance，可叠加置信度/不确定性自适应项。"""
    if total_steps <= 1:
        base_scale = max_scale
    else:
        progress = step_idx / float(total_steps - 1)
        base_scale = min_scale + (max_scale - min_scale) * (progress ** power)

    if goal_confidence is None and action_uncertainty is None:
        return base_scale

    scale = torch.full_like(
        goal_confidence if goal_confidence is not None else action_uncertainty,
        float(base_scale),
    )

    if goal_confidence is not None and confidence_weight != 0:
        confidence = goal_confidence.clamp(0, 1)
        scale = scale * (1 + confidence_weight * (2 * confidence - 1))

    if action_uncertainty is not None and uncertainty_weight != 0:
        uncertainty = action_uncertainty.clamp_min(0)
        uncertainty = uncertainty / (uncertainty.detach().mean() + 1e-6)
        scale = scale / (1 + uncertainty_weight * uncertainty)

    return scale.clamp(min_scale, max_scale)


def _repeat_group_stat(stat: torch.Tensor, num_samples: int) -> torch.Tensor:
    if num_samples <= 1:
        return stat
    return stat.repeat_interleave(num_samples, dim=0)


def _sample_uncertainty(diffusion_output: torch.Tensor, num_samples: int) -> Optional[torch.Tensor]:
    if num_samples <= 1:
        return None
    batch_size = diffusion_output.shape[0] // num_samples
    grouped = diffusion_output.reshape(batch_size, num_samples, *diffusion_output.shape[1:])
    uncertainty = grouped.var(dim=1, unbiased=False).mean(dim=tuple(range(1, grouped.dim() - 1)))
    return uncertainty.repeat_interleave(num_samples, dim=0)


def model_output(
    model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    num_samples: int,
    device: torch.device,
    guidance_scale_min: float = 0.25,
    guidance_scale_max: float = 1.75,
    guidance_scale_power: float = 1.5,
    use_adaptive_guidance: bool = True,
    guidance_confidence_weight: float = 0.35,
    guidance_uncertainty_weight: float = 0.25,
    guidance_distance_scale: float = 10.0,
    generator: Optional[torch.Generator] = None,
):
    noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps)

    goal_mask = torch.ones((batch_goal_images.shape[0],)).long().to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=goal_mask)
    # obs_cond = obs_cond.flatten(start_dim=1)
    obs_cond = obs_cond.repeat_interleave(num_samples, dim=0)

    no_mask = torch.zeros((batch_goal_images.shape[0],)).long().to(device)
    obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images, goal_img=batch_goal_images, input_goal_mask=no_mask)
    gc_distance_base = model("dist_pred_net", obsgoal_cond=obsgoal_cond).flatten()
    # obsgoal_cond = obsgoal_cond.flatten(start_dim=1)  
    obsgoal_cond = obsgoal_cond.repeat_interleave(num_samples, dim=0)
    goal_confidence = torch.exp(
        -gc_distance_base.float().clamp_min(0) / max(float(guidance_distance_scale), 1e-6)
    )
    goal_confidence = _repeat_group_stat(goal_confidence, num_samples)

    # initialize action from Gaussian noise
    if generator is None:
        noisy_diffusion_output = torch.randn(
            (len(obs_cond), pred_horizon, action_dim), device=device
        )
    else:
        noisy_diffusion_output = torch.randn(
            (len(obs_cond), pred_horizon, action_dim), generator=generator
        ).to(device)
    diffusion_output = noisy_diffusion_output


    total_steps = len(noise_scheduler.timesteps)
    for k in noise_scheduler.timesteps[:]:
        # predict noise
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample

    uc_actions = get_action(diffusion_output, ACTION_STATS)

    # initialize action from Gaussian noise
    if generator is None:
        noisy_diffusion_output = torch.randn(
            (len(obs_cond), pred_horizon, action_dim), device=device
        )
    else:
        noisy_diffusion_output = torch.randn(
            (len(obs_cond), pred_horizon, action_dim), generator=generator
        ).to(device)
    diffusion_output = noisy_diffusion_output

    for step_idx, k in enumerate(noise_scheduler.timesteps[:]):
        unconditional_noise = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )
        conditional_noise = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obsgoal_cond
        )
        guidance_scale = diffusion_guidance_scale(
            step_idx,
            total_steps,
            guidance_scale_min,
            guidance_scale_max,
            guidance_scale_power,
            goal_confidence=goal_confidence if use_adaptive_guidance else None,
            action_uncertainty=_sample_uncertainty(diffusion_output, num_samples)
            if use_adaptive_guidance
            else None,
            confidence_weight=guidance_confidence_weight,
            uncertainty_weight=guidance_uncertainty_weight,
        )
        if torch.is_tensor(guidance_scale):
            guidance_scale = guidance_scale.reshape(-1, 1, 1)
        noise_pred = unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )

        # inverse diffusion step (remove noise)
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    obsgoal_cond = obsgoal_cond.flatten(start_dim=1)
    gc_actions = get_action(diffusion_output, ACTION_STATS)
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)

    return {
        'uc_actions': uc_actions,
        'gc_actions': gc_actions,
        'gc_distance': gc_distance,
    }


def visualize_diffusion_action_distribution(
    ema_model: nn.Module,
    noise_scheduler: DDPMScheduler,
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_viz_obs_images: torch.Tensor,
    batch_viz_goal_images: torch.Tensor,
    batch_action_label: torch.Tensor,
    batch_distance_labels: torch.Tensor,
    batch_goal_pos: torch.Tensor,
    device: torch.device,
    eval_type: str,
    project_folder: str,
    epoch: int,
    num_images_log: int,
    num_samples: int = 30,
    use_wandb: bool = True,
    goal_guidance_min: float = 0.25,
    goal_guidance_max: float = 1.75,
    goal_guidance_power: float = 1.5,
    use_adaptive_guidance: bool = True,
    guidance_confidence_weight: float = 0.35,
    guidance_uncertainty_weight: float = 0.25,
    guidance_distance_scale: float = 10.0,
    generator: Optional[torch.Generator] = None,
):
    """Plot samples from the exploration model."""

    visualize_path = os.path.join(
        project_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "action_sampling_prediction",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)

    max_batch_size = batch_obs_images.shape[0]

    num_images_log = min(num_images_log, batch_obs_images.shape[0], batch_goal_images.shape[0], batch_action_label.shape[0], batch_goal_pos.shape[0])
    batch_obs_images = batch_obs_images[:num_images_log]
    batch_goal_images = batch_goal_images[:num_images_log]
    batch_action_label = batch_action_label[:num_images_log]
    batch_goal_pos = batch_goal_pos[:num_images_log]
    
    wandb_list = []

    pred_horizon = batch_action_label.shape[1]
    action_dim = batch_action_label.shape[2]

    # split into batches
    batch_obs_images_list = torch.split(batch_obs_images, max_batch_size, dim=0)
    batch_goal_images_list = torch.split(batch_goal_images, max_batch_size, dim=0)

    uc_actions_list = []
    gc_actions_list = []
    gc_distances_list = []

    for obs, goal in zip(batch_obs_images_list, batch_goal_images_list):
        model_output_dict = model_output(
            ema_model,
            noise_scheduler,
            obs,
            goal,
            pred_horizon,
            action_dim,
            num_samples,
            device,
            guidance_scale_min=goal_guidance_min,
            guidance_scale_max=goal_guidance_max,
            guidance_scale_power=goal_guidance_power,
            use_adaptive_guidance=use_adaptive_guidance,
            guidance_confidence_weight=guidance_confidence_weight,
            guidance_uncertainty_weight=guidance_uncertainty_weight,
            guidance_distance_scale=guidance_distance_scale,
            generator=generator,
        )
        uc_actions_list.append(to_numpy(model_output_dict['uc_actions']))
        gc_actions_list.append(to_numpy(model_output_dict['gc_actions']))
        gc_distances_list.append(to_numpy(model_output_dict['gc_distance']))

    # concatenate
    uc_actions_list = np.concatenate(uc_actions_list, axis=0)
    gc_actions_list = np.concatenate(gc_actions_list, axis=0)
    gc_distances_list = np.concatenate(gc_distances_list, axis=0)

    # split into actions per observation
    uc_actions_list = np.split(uc_actions_list, num_images_log, axis=0)
    gc_actions_list = np.split(gc_actions_list, num_images_log, axis=0)
    gc_distances_list = np.split(gc_distances_list, num_images_log, axis=0)

    gc_distances_avg = [np.mean(dist) for dist in gc_distances_list]
    gc_distances_std = [np.std(dist) for dist in gc_distances_list]

    assert len(uc_actions_list) == len(gc_actions_list) == num_images_log

    np_distance_labels = to_numpy(batch_distance_labels)

    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        uc_actions = uc_actions_list[i]
        gc_actions = gc_actions_list[i]
        action_label = to_numpy(batch_action_label[i])

        traj_list = np.concatenate([
            uc_actions,
            gc_actions,
            action_label[None],
        ], axis=0)
        # traj_labels = ["r", "GC", "GC_mean", "GT"]
        traj_colors = ["red"] * len(uc_actions) + ["green"] * len(gc_actions) + ["magenta"]
        traj_alphas = [0.1] * (len(uc_actions) + len(gc_actions)) + [1.0]

        # make points numpy array of robot positions (0, 0) and goal positions
        point_list = [np.array([0, 0]), to_numpy(batch_goal_pos[i])]
        point_colors = ["green", "red"]
        point_alphas = [1.0, 1.0]

        plot_trajs_and_points(
            ax[0],
            traj_list,
            point_list,
            traj_colors,
            point_colors,
            traj_labels=None,
            point_labels=None,
            quiver_freq=0,
            traj_alphas=traj_alphas,
            point_alphas=point_alphas, 
        )
        
        obs_image = to_numpy(batch_viz_obs_images[i])
        goal_image = to_numpy(batch_viz_goal_images[i])
        # move channel to last dimension
        obs_image = np.moveaxis(obs_image, 0, -1)
        goal_image = np.moveaxis(goal_image, 0, -1)
        ax[1].imshow(obs_image)
        ax[2].imshow(goal_image)

        # set title
        ax[0].set_title(f"diffusion action predictions")
        ax[1].set_title(f"observation")
        ax[2].set_title(f"goal: label={np_distance_labels[i]} gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
        
        # make the plot large
        fig.set_size_inches(18.5, 10.5)

        save_path = os.path.join(visualize_path, f"sample_{i}.png")
        plt.savefig(save_path)
        wandb_list.append(wandb.Image(save_path))
        plt.close(fig)
    if len(wandb_list) > 0 and use_wandb:
        wandb.log({"epoch": epoch, f"{eval_type}_action_samples": wandb_list}, commit=False)
