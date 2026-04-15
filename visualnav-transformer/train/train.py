import os
import atexit
import gc
import wandb
import argparse
import numpy as np
import yaml
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

"""
IMPORT YOUR MODEL HERE

该文件是整个视觉导航模型的训练入口：
- 解析配置、构建数据集和数据加载器
- 根据 config 选择并实例化不同类型的模型（GNM / ViNT / NoMaD）
- 创建优化器与学习率调度器
- 调用 train_eval_loop / train_eval_loop_nomad 完成训练与评估
"""
from vint_train.models.gnm.gnm import GNM
from vint_train.models.vint.vint import ViNT
from vint_train.models.vint.vit import ViT
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.nomad.nomad_mamba import NoMaD_Mamba
from vint_train.models.nomad.mamba2 import MambaConfig
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.training.train_eval_loop import (
    train_eval_loop,
    train_eval_loop_nomad,
    load_model,
)


def _shutdown_dataloader_workers(loader):
    """
    显式关闭 DataLoader 的 worker 进程，避免进程退出时 multiprocessing resource_tracker
    报告 leaked semaphore（常见于 num_workers>0 且 persistent_workers=True）。
    """
    if loader is None:
        return
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None and hasattr(it, "_shutdown_workers"):
            it._shutdown_workers()
    except Exception:
        pass
    try:
        loader._iterator = None
    except Exception:
        pass


def _build_optimizer_param_groups(config, model: nn.Module, base_lr: float):
    if not (
        config["model_type"] == "nomad"
        and config.get("vision_encoder") == "nomad_mamba"
        and config.get("use_differential_lr", False)
    ):
        return None

    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is None:
        return None

    backbone_lr = float(config.get("backbone_lr", base_lr / 5.0))
    backbone_param_ids = set()
    backbone_params = []
    for encoder_name in ("obs_encoder", "goal_encoder"):
        encoder = getattr(vision_encoder, encoder_name, None)
        if encoder is None:
            continue
        for parameter in encoder.parameters():
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in backbone_param_ids:
                continue
            backbone_param_ids.add(parameter_id)
            backbone_params.append(parameter)

    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_param_ids
    ]

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "group_name": "main"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "group_name": "backbone"})

    if len(param_groups) <= 1:
        return None

    print(
        "Using differential LR:"
        f" main_lr={base_lr:.2e}, backbone_lr={backbone_lr:.2e},"
        f" main_params={sum(p.numel() for p in other_params) / 1e6:.2f}M,"
        f" backbone_params={sum(p.numel() for p in backbone_params) / 1e6:.2f}M"
    )
    return param_groups


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    logical_gpu_ids = list(range(len(config["gpu_ids"]))) if torch.cuda.is_available() else []
    first_gpu_id = logical_gpu_ids[0] if logical_gpu_ids else 0
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = not cudnn.deterministic
    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    # Load the data
    train_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    if "clip_goals" not in config:
        config["clip_goals"] = False

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1

        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                    dataset = ViNT_Dataset(
                        data_folder=data_config["data_folder"],
                        data_split_folder=data_config[data_split_type],
                        dataset_name=dataset_name,
                        image_size=config["image_size"],
                        waypoint_spacing=data_config["waypoint_spacing"],
                        min_dist_cat=config["distance"]["min_dist_cat"],
                        max_dist_cat=config["distance"]["max_dist_cat"],
                        min_action_distance=config["action"]["min_dist_cat"],
                        max_action_distance=config["action"]["max_dist_cat"],
                        negative_mining=data_config["negative_mining"],
                        len_traj_pred=config["len_traj_pred"],
                        learn_angle=config["learn_angle"],
                        context_size=config["context_size"],
                        context_type=config["context_type"],
                        end_slack=data_config["end_slack"],
                        goals_per_obs=data_config["goals_per_obs"],
                        normalize=config["normalize"],
                        goal_type=config["goal_type"],
                    )
                    if data_split_type == "train":
                        train_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type] = dataset

    # combine all the datasets from different robots
    train_dataset = ConcatDataset(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=config["num_workers"] > 0,
        pin_memory=torch.cuda.is_available(),
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    def _training_cleanup():
        _shutdown_dataloader_workers(train_loader)
        for _loader in test_dataloaders.values():
            _shutdown_dataloader_workers(_loader)
        gc.collect()

    atexit.register(_training_cleanup)

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model_type"] == "vint":
        model = ViNT(
            context_size=config["context_size"],
            len_traj_pred=config["len_traj_pred"],
            learn_angle=config["learn_angle"],
            obs_encoder=config["obs_encoder"],
            obs_encoding_size=config["obs_encoding_size"],
            late_fusion=config["late_fusion"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["model_type"] == "nomad":
        if config["vision_encoder"] == "nomad_vint":
            vision_encoder = NoMaD_ViNT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        elif config["vision_encoder"] == "nomad_mamba":
            # 使用 Mamba2 作为序列建模模块的视觉编码器
            # 支持 timm 库中的多种视觉编码器：EfficientNet, ResNet, ViT, DINOv2, ConvNeXt 等
            # 配置中 image_size 是 [宽, 高]，但 timm 需要 (高, 宽)
            img_size_hw = (config["image_size"][1], config["image_size"][0])
            vision_encoder = NoMaD_Mamba(
                context_size=config["context_size"],
                obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
                goal_encoder=config.get("goal_encoder", None),  # 可选，默认与 obs_encoder 相同
                obs_encoding_size=config["encoding_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
                mha_ff_dim_factor=config["mha_ff_dim_factor"],
                mamba_cfg=MambaConfig.from_dict(config),
                img_size=img_size_hw,  # 传递图像尺寸给 ViT 类模型
                bidirectional_mamba=config.get("bidirectional_mamba", True),
                use_goal_gate=config.get("use_goal_gate", True),
                use_goal_film=config.get("use_goal_film", True),
                goal_fusion_hidden_dim=config.get("goal_fusion_hidden_dim", None),
                share_visual_backbone=config.get("share_visual_backbone", False),
                adapter_hidden_dim=config.get("adapter_hidden_dim", None),
                adapter_scale=config.get("adapter_scale", 0.1),
            )
            # 注：_create_timm_encoder 内部已调用 replace_bn_with_gn，无需重复调用
        elif config["vision_encoder"] == "vib":
            raise NotImplementedError(
                "`vision_encoder: vib` 尚未在当前代码库中实现，请改用 `nomad_vint`、`nomad_mamba` 或 `vit`。"
            )
        elif config["vision_encoder"] == "vit": 
            vision_encoder = ViT(
                obs_encoding_size=config["encoding_size"],
                context_size=config["context_size"],
                image_size=config["image_size"],
                patch_size=config["patch_size"],
                mha_num_attention_heads=config["mha_num_attention_heads"],
                mha_num_attention_layers=config["mha_num_attention_layers"],
            )
            vision_encoder = replace_bn_with_gn(vision_encoder)
        else: 
            raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")
        
        # 条件一维 UNet，用于 diffusion policy 中预测 action 噪声
        noise_pred_net = ConditionalUnet1D(
                input_dim=2,
                global_cond_dim=config["encoding_size"],
                down_dims=config["down_dims"],
                cond_predict_scale=config["cond_predict_scale"],
            )
        # 预测距离的 MLP 网络
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
    
        # Diffusion policy 中使用的 DDPM 调度器（只在 NoMaD 模型下需要）
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    else:
        raise ValueError(f"Model type {config['model_type']} not supported")

    max_grad_norm = None
    if config["clipping"]:
        max_grad_norm = float(config["max_norm"])
        print("Clipping gradients by global norm:", max_grad_norm)

    lr = float(config["lr"])
    optimizer_param_groups = _build_optimizer_param_groups(config, model, lr)
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(
            optimizer_param_groups if optimizer_param_groups is not None else model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
        )
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(
            optimizer_param_groups if optimizer_param_groups is not None else model.parameters(),
            lr=lr,
        )
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_param_groups if optimizer_param_groups is not None else model.parameters(),
            lr=lr,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.,
                max_lr=lr,
                step_size_up=config["cyclic_period"] // 2,
                cycle_momentum=False,
            )
        elif config["scheduler"] == "plateau":
            print("Using ReduceLROnPlateau")
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=config["plateau_factor"],
                patience=config["plateau_patience"],
                verbose=True,
            )
        else:
            raise ValueError(f"Scheduler {config['scheduler']} not supported")

        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )

    current_epoch = 0
    latest_checkpoint = None
    resume_checkpoint = None
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        if config["model_type"] == "nomad":
            training_latest_path = os.path.join(load_project_folder, "training_latest.pth")
            if os.path.exists(training_latest_path):
                latest_path = training_latest_path
        latest_checkpoint = torch.load(latest_path, map_location="cpu")
        if (
            config["model_type"] == "nomad"
            and isinstance(latest_checkpoint, dict)
            and ("model_state_dict" in latest_checkpoint or "ema_state_dict" in latest_checkpoint)
        ):
            resume_checkpoint = latest_checkpoint
        load_model(model, config["model_type"], latest_checkpoint)
        if isinstance(latest_checkpoint, dict) and "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    # Multi-GPU
    if len(logical_gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=logical_gpu_ids)
    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        def _resolve_state_dict(checkpoint_entry):
            if checkpoint_entry is None:
                return None
            if hasattr(checkpoint_entry, "state_dict"):
                return checkpoint_entry.state_dict()
            return checkpoint_entry

        optimizer_state = None
        scheduler_state = None
        if isinstance(latest_checkpoint, dict):
            optimizer_state = _resolve_state_dict(
                latest_checkpoint.get("optimizer_state_dict", latest_checkpoint.get("optimizer"))
            )
            scheduler_state = _resolve_state_dict(
                latest_checkpoint.get("scheduler_state_dict", latest_checkpoint.get("scheduler"))
            )

        if config["model_type"] == "nomad":
            if optimizer_state is None:
                optimizer_latest_path = os.path.join(load_project_folder, "optimizer_latest.pth")
                if os.path.exists(optimizer_latest_path):
                    optimizer_state = torch.load(optimizer_latest_path, map_location="cpu")
            if scheduler is not None and scheduler_state is None:
                scheduler_latest_path = os.path.join(load_project_folder, "scheduler_latest.pth")
                if os.path.exists(scheduler_latest_path):
                    scheduler_state = torch.load(scheduler_latest_path, map_location="cpu")

        if optimizer_state is not None:
            try:
                optimizer.load_state_dict(optimizer_state)
            except ValueError as exc:
                print(f"Skipping optimizer state restore due to param-group mismatch: {exc}")
        if scheduler is not None and scheduler_state is not None:
            try:
                scheduler.load_state_dict(scheduler_state)
            except ValueError as exc:
                print(f"Skipping scheduler state restore due to mismatch: {exc}")

    # ---------- 进入统一的训练 / 评估循环 ----------
    if config["model_type"] == "vint" or config["model_type"] == "gnm": 
        train_eval_loop(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            max_grad_norm=max_grad_norm,
        )
    else:
        train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            resume_checkpoint=resume_checkpoint,
            goal_guidance_min=float(config.get("goal_guidance_min", 0.25)),
            goal_guidance_max=float(config.get("goal_guidance_max", 1.75)),
            goal_guidance_power=float(config.get("goal_guidance_power", 1.5)),
            max_grad_norm=max_grad_norm,
        )

    # 在打印结束前显式关闭 worker，避免仅依赖 atexit 时仍出现 semaphore 泄漏提示
    _training_cleanup()

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Visual Navigation Transformer")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/vint.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        # 与下方 torch.multiprocessing.set_start_method("spawn") 一致，避免 fork/spawn 混用导致
        # multiprocessing 资源（含 semaphore）回收异常。
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="spawn"),
            entity="coisinic243-beijing-university-of-technology", # TODO: change this to your wandb entity
        )

        def _wandb_finish_atexit():
            try:
                wandb.finish()
            except Exception:
                pass

        atexit.register(_wandb_finish_atexit)

        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        wandb.save(args.config, policy="now")  # save the config file
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
