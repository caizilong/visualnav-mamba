import wandb
import os
import numpy as np
import copy
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate
from vint_train.training.train_utils import train_nomad, evaluate_nomad

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _ema_device(ema_model: EMAModel) -> torch.device:
    averaged_model = _unwrap_model(ema_model.averaged_model)
    try:
        return next(averaged_model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def serialize_ema_model(ema_model: EMAModel) -> dict:
    state = {
        "averaged_model": _unwrap_model(ema_model.averaged_model).state_dict(),
    }

    if hasattr(ema_model, "shadow_params"):
        state["shadow_params"] = [
            param.detach().cpu().clone() for param in ema_model.shadow_params
        ]

    scalar_attrs = [
        "decay",
        "min_decay",
        "optimization_step",
        "update_after_step",
        "use_ema_warmup",
        "inv_gamma",
        "power",
        "cur_decay_value",
    ]
    for attr_name in scalar_attrs:
        if hasattr(ema_model, attr_name):
            state[attr_name] = copy.deepcopy(getattr(ema_model, attr_name))

    return state


def _apply_nomad_backbone_schedule(
    model: nn.Module,
    epoch: int,
    use_wandb: bool = True,
    force_log: bool = False,
) -> None:
    raw_model = _unwrap_model(model)
    vision_encoder = getattr(raw_model, "vision_encoder", None)
    if vision_encoder is None or not hasattr(vision_encoder, "apply_backbone_unfreeze_schedule"):
        return

    schedule_info = vision_encoder.apply_backbone_unfreeze_schedule(epoch)
    if not schedule_info.get("changed", False) and not force_log:
        return

    trainable_blocks = schedule_info["trainable_blocks"]
    total_blocks = schedule_info["total_blocks"]
    trainable_params_m = schedule_info["trainable_backbone_params"] / 1e6
    total_params_m = schedule_info["total_backbone_params"] / 1e6
    print(
        "Backbone freeze schedule:"
        f" epoch={epoch}, trainable_blocks={trainable_blocks}/{total_blocks},"
        f" trainable_backbone_params={trainable_params_m:.2f}M/{total_params_m:.2f}M"
    )
    if use_wandb:
        wandb.log(
            {
                "epoch": epoch,
                "backbone_trainable_blocks": trainable_blocks,
                "backbone_total_blocks": total_blocks,
                "backbone_trainable_params_m": trainable_params_m,
                "backbone_total_params_m": total_params_m,
            },
            commit=False,
        )


def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
):
    """
    统一的训练 + 测试循环（适用于 ViNT / GNM 模型）。

    Args:
        train_model: 是否进行训练（False 时只跑评估）
        model: 需要训练 / 评估的模型
        optimizer: 优化器
        scheduler: 学习率调度器
        dataloader: 训练集 DataLoader
        test_dataloaders: 测试集 DataLoader 字典（键为数据集名称）
        transform: 图像预处理（如归一化等）
        epochs: 训练轮数
        device: 训练设备
        project_folder: 日志与 checkpoint 的保存目录
        normalized: 动作空间是否做过归一化
        wandb_log_freq: 记录 scalar 到 wandb 的频率
        print_log_freq: 在终端打印日志的频率
        image_log_freq: 向 wandb 记录图像的频率
        num_images_log: 每次记录的图像数量
        current_epoch: 当前起始 epoch（用于从 checkpoint 继续训练）
        alpha: 距离损失与动作损失之间的权重系数
        learn_angle: 是否预测 yaw（角度）
        use_wandb: 是否启用 wandb 记录
        eval_fraction: 只使用部分训练数据做快速评估的比例
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        # ---------- 训练阶段 ----------
        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
            )

        avg_total_test_loss = []
        # ---------- 在所有测试集上做评估 ----------
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        # log average eval loss
        if use_wandb:
            wandb.log({"epoch": epoch}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "avg_total_test_loss": np.mean(avg_total_test_loss),
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    # Flush the last set of eval logs
    if use_wandb:
        wandb.log({"epoch": current_epoch + epochs - 1})
    print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    resume_checkpoint: Optional[dict] = None,
):
    """
    NoMaD（基于 diffusion policy）模型的训练 + 评估循环。

    相比 ViNT / GNM，多了：
    - 噪声调度器 `noise_scheduler`（DDPM）
    - EMA 模型 `ema_model`（用于更稳定的评估）

    Args:
        model: 主模型（包含视觉编码器 + 噪声预测网络 + 距离预测网络）
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        noise_scheduler: diffusion 噪声调度器
        train_loader: 训练集 DataLoader
        test_dataloaders: 测试集 DataLoader 字典
        transform: 图像预处理
        goal_mask_prob: 训练时对目标 token 做 masking 的概率
        epochs: 训练轮数
        device: 训练设备
        project_folder: 日志与 checkpoint 目录
        wandb_log_freq: wandb 记录频率
        print_log_freq: 终端打印频率
        image_log_freq: 记录图像的频率
        num_images_log: 记录的图像数量
        current_epoch: 起始 epoch
        alpha: 各损失项之间的权重
        use_wandb: 是否启用 wandb
        eval_fraction: 每轮评估时，使用训练数据子集的比例
        eval_freq: 每多少个 epoch 进行一次评估
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    training_latest_path = os.path.join(project_folder, "training_latest.pth")
    ema_model = EMAModel(model=model,power=0.75)
    if isinstance(resume_checkpoint, dict) and "ema_state_dict" in resume_checkpoint:
        load_ema_model(ema_model, resume_checkpoint["ema_state_dict"])
        print("Loaded EMA state from training checkpoint")
    
    for epoch in range(current_epoch, current_epoch + epochs):
        _apply_nomad_backbone_schedule(
            model=model,
            epoch=epoch,
            use_wandb=use_wandb,
            force_log=(epoch == current_epoch),
        )
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            if lr_scheduler is not None:
                lr_scheduler.step()

        ema_epoch_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        ema_latest_path = os.path.join(project_folder, "ema_latest.pth")
        ema_model_state_dict = _unwrap_model(ema_model.averaged_model).state_dict()
        torch.save(ema_model_state_dict, ema_epoch_path)
        torch.save(ema_model_state_dict, ema_latest_path)
        print(f"Saved EMA model to {ema_epoch_path} and {ema_latest_path}")

        model_epoch_path = os.path.join(project_folder, f"{epoch}.pth")
        raw_model = _unwrap_model(model)
        model_state_dict = raw_model.state_dict()
        torch.save(model_state_dict, model_epoch_path)
        torch.save(model_state_dict, latest_path)
        print(f"Saved model to {model_epoch_path}")

        training_state = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "ema_state_dict": serialize_ema_model(ema_model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "backbone_trainable_blocks": getattr(
                getattr(raw_model, "vision_encoder", None),
                "_current_trainable_backbone_blocks",
                None,
            ),
        }
        torch.save(training_state, training_latest_path)
        print(f"Saved training state to {training_latest_path}")

        # save optimizer
        latest_optimizer_path = os.path.join(project_folder, "optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        if lr_scheduler is not None:
            latest_scheduler_path = os.path.join(project_folder, "scheduler_latest.pth")
            torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    image_log_freq=image_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)

        
    # Flush the last set of eval logs
    if use_wandb:
        wandb.log({"epoch": current_epoch + epochs - 1})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    if not isinstance(state_dict, dict):
        _unwrap_model(ema_model.averaged_model).load_state_dict(state_dict, strict=False)
        if hasattr(ema_model, "shadow_params"):
            ema_model.shadow_params = [
                param.detach().clone()
                for param in _unwrap_model(ema_model.averaged_model).parameters()
            ]
        return

    averaged_model_state = state_dict.get("averaged_model", state_dict)
    _unwrap_model(ema_model.averaged_model).load_state_dict(averaged_model_state, strict=False)

    device = _ema_device(ema_model)
    if hasattr(ema_model, "shadow_params"):
        if "shadow_params" in state_dict:
            ema_model.shadow_params = [
                tensor.detach().to(device).clone() for tensor in state_dict["shadow_params"]
            ]
        else:
            ema_model.shadow_params = [
                param.detach().clone()
                for param in _unwrap_model(ema_model.averaged_model).parameters()
            ]

    scalar_attrs = [
        "decay",
        "min_decay",
        "optimization_step",
        "update_after_step",
        "use_ema_warmup",
        "inv_gamma",
        "power",
        "cur_decay_value",
    ]
    for attr_name in scalar_attrs:
        if attr_name in state_dict and hasattr(ema_model, attr_name):
            setattr(ema_model, attr_name, copy.deepcopy(state_dict[attr_name]))


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params
