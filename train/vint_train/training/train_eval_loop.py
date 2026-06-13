import copy
import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from prettytable import PrettyTable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from vint_train.training.train_utils import evaluate_nomad, train_nomad


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


def _atomic_torch_save(obj, path: str) -> None:
    tmp_path = f"{path}.tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise RuntimeError(
            f"Failed to save checkpoint to {path}. Check disk space, quota, "
            "and filesystem write stability."
        ) from exc


def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    train_stage: str = "finetune",
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    sampling_metrics_freq: int = 1000,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
    resume_checkpoint: Optional[dict] = None,
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
    ema_model: Optional[EMAModel] = None,
    save_epoch_checkpoints: bool = False,
    save_training_state: bool = False,
    save_optimizer_state: bool = False,
):
    latest_path = os.path.join(project_folder, "latest.pth")
    training_latest_path = os.path.join(project_folder, "training_latest.pth")
    progress_latest_path = os.path.join(project_folder, "training_progress_latest.pth")

    if ema_model is None:
        ema_model = EMAModel(model=model, power=0.75)
        if isinstance(resume_checkpoint, dict) and "ema_state_dict" in resume_checkpoint:
            load_ema_model(ema_model, resume_checkpoint["ema_state_dict"])
            print("Loaded EMA state from training checkpoint")

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(f"Start NoMaD-Mamba training epoch {epoch}/{current_epoch + epochs - 1}")
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
                train_stage=train_stage,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                sampling_metrics_freq=sampling_metrics_freq,
                use_wandb=use_wandb,
                alpha=alpha,
                goal_guidance_min=goal_guidance_min,
                goal_guidance_max=goal_guidance_max,
                goal_guidance_power=goal_guidance_power,
                use_adaptive_guidance=use_adaptive_guidance,
                guidance_confidence_weight=guidance_confidence_weight,
                guidance_uncertainty_weight=guidance_uncertainty_weight,
                guidance_distance_scale=guidance_distance_scale,
                nav_goal_pos_loss_weight=nav_goal_pos_loss_weight,
                nav_contrastive_loss_weight=nav_contrastive_loss_weight,
                nav_contrastive_temperature=nav_contrastive_temperature,
                aux_negative_distance_threshold=aux_negative_distance_threshold,
                max_grad_norm=max_grad_norm,
            )
            if lr_scheduler is not None:
                lr_scheduler.step()

        ema_latest_path = os.path.join(project_folder, "ema_latest.pth")
        ema_model_state_dict = _unwrap_model(ema_model.averaged_model).state_dict()
        if save_epoch_checkpoints:
            ema_epoch_path = os.path.join(project_folder, f"ema_{epoch}.pth")
            _atomic_torch_save(ema_model_state_dict, ema_epoch_path)
        _atomic_torch_save(ema_model_state_dict, ema_latest_path)

        raw_model = _unwrap_model(model)
        model_state_dict = raw_model.state_dict()
        if save_epoch_checkpoints:
            model_epoch_path = os.path.join(project_folder, f"{epoch}.pth")
            _atomic_torch_save(model_state_dict, model_epoch_path)
        _atomic_torch_save(model_state_dict, latest_path)

        if save_optimizer_state:
            latest_optimizer_path = os.path.join(project_folder, "optimizer_latest.pth")
            _atomic_torch_save(optimizer.state_dict(), latest_optimizer_path)

            if lr_scheduler is not None:
                latest_scheduler_path = os.path.join(project_folder, "scheduler_latest.pth")
                _atomic_torch_save(lr_scheduler.state_dict(), latest_scheduler_path)

        progress_state = {"epoch": epoch}
        _atomic_torch_save(progress_state, progress_latest_path)

        if save_training_state:
            training_state = {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "ema_state_dict": serialize_ema_model(ema_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            }
            _atomic_torch_save(training_state, training_latest_path)

        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                print(f"Start {dataset_type} evaluation epoch {epoch}/{current_epoch + epochs - 1}")
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
                    sampling_metrics_freq=sampling_metrics_freq,
                    eval_fraction=eval_fraction,
                    goal_guidance_min=goal_guidance_min,
                    goal_guidance_max=goal_guidance_max,
                    goal_guidance_power=goal_guidance_power,
                    use_adaptive_guidance=use_adaptive_guidance,
                    guidance_confidence_weight=guidance_confidence_weight,
                    guidance_uncertainty_weight=guidance_uncertainty_weight,
                    guidance_distance_scale=guidance_distance_scale,
                )

        if use_wandb:
            lr_log = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if len(optimizer.param_groups) > 1:
                lr_log["backbone_lr"] = optimizer.param_groups[1]["lr"]
            wandb.log(lr_log, commit=False)

    if use_wandb:
        wandb.log({"epoch": current_epoch + epochs - 1})
    print()
    return ema_model


def load_model(model, model_type, checkpoint: dict) -> None:
    if model_type != "nomad":
        raise ValueError("Only NoMaD checkpoints are supported in this slimmed repository.")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)


def load_ema_model(ema_model, state_dict: dict) -> None:
    if not isinstance(state_dict, dict):
        _unwrap_model(ema_model.averaged_model).load_state_dict(state_dict, strict=True)
        if hasattr(ema_model, "shadow_params"):
            ema_model.shadow_params = [
                param.detach().clone()
                for param in _unwrap_model(ema_model.averaged_model).parameters()
            ]
        return

    averaged_model_state = state_dict.get("averaged_model", state_dict)
    _unwrap_model(ema_model.averaged_model).load_state_dict(averaged_model_state, strict=True)

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
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params
