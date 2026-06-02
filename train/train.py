import argparse
import atexit
import gc
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import wandb
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torch.optim import Adam, AdamW
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.nomad.mamba2 import MambaConfig
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_mamba import NoMaD_Mamba
from vint_train.training.train_eval_loop import load_model, train_eval_loop_nomad


def _shutdown_dataloader_workers(loader):
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


def _assert_nomad_mamba_config(config: dict) -> None:
    if config.get("model_type") != "nomad":
        raise ValueError(
            "This repository is slimmed to NoMaD-Mamba only. Set `model_type: nomad`."
        )
    if config.get("vision_encoder") != "nomad_mamba":
        raise ValueError(
            "This repository is slimmed to NoMaD-Mamba only. "
            "Set `vision_encoder: nomad_mamba`."
        )


def _build_optimizer_param_groups(config, model: nn.Module, base_lr: float):
    if not config.get("use_differential_lr", False):
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


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = enabled


def _apply_train_stage(config: dict, model: NoMaD) -> None:
    vision_encoder = model.vision_encoder
    if config.get("ablation_unfreeze_all", False):
        _set_requires_grad(model, True)
        train_stage = "ablation_unfreeze_all (finetune)"
        freeze_backbone = False
    else:
        train_stage = config.get("train_stage", "finetune")
        valid_stages = {"representation_warmup", "diffusion_tuning", "finetune"}
        if train_stage not in valid_stages:
            raise ValueError(f"train_stage must be one of {sorted(valid_stages)}, got {train_stage}")

        freeze_backbone = bool(config.get("freeze_backbone", False))

        if freeze_backbone or train_stage in {"representation_warmup", "diffusion_tuning"}:
            for encoder_name in ("obs_encoder", "goal_encoder"):
                encoder = getattr(vision_encoder, encoder_name, None)
                if encoder is not None:
                    _set_requires_grad(encoder, False)

        if train_stage == "representation_warmup":
            _set_requires_grad(model.noise_pred_net, False)
        elif train_stage == "diffusion_tuning":
            _set_requires_grad(model.noise_pred_net, True)
        else:
            _set_requires_grad(model.noise_pred_net, True)

        if train_stage == "finetune" and not freeze_backbone:
            for encoder_name in ("obs_encoder", "goal_encoder"):
                encoder = getattr(vision_encoder, encoder_name, None)
                if encoder is not None:
                    _set_requires_grad(encoder, True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    backbone_trainable = 0
    backbone_total = 0
    seen_param_ids = set()
    for encoder_name in ("obs_encoder", "goal_encoder"):
        encoder = getattr(vision_encoder, encoder_name, None)
        if encoder is None:
            continue
        for parameter in encoder.parameters():
            parameter_id = id(parameter)
            if parameter_id in seen_param_ids:
                continue
            seen_param_ids.add(parameter_id)
            backbone_total += parameter.numel()
            if parameter.requires_grad:
                backbone_trainable += parameter.numel()

    print(
        "Training stage:"
        f" {train_stage}, freeze_backbone={freeze_backbone},"
        f" trainable={trainable / 1e6:.2f}M/{total / 1e6:.2f}M,"
        f" backbone_trainable={backbone_trainable / 1e6:.2f}M/{backbone_total / 1e6:.2f}M"
    )


def main(config):
    _assert_nomad_mamba_config(config)

    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif isinstance(config["gpu_ids"], int):
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["gpu_ids"]])
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    logical_gpu_ids = list(range(len(config["gpu_ids"]))) if torch.cuda.is_available() else []
    first_gpu_id = logical_gpu_ids[0] if logical_gpu_ids else 0
    device = torch.device(f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = not cudnn.deterministic
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = []
    test_datasets = {}

    config.setdefault("context_type", "temporal")
    config.setdefault("clip_goals", False)

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        data_config.setdefault("negative_mining", True)
        data_config.setdefault("goals_per_obs", 1)
        data_config.setdefault("end_slack", 0)
        data_config.setdefault("waypoint_spacing", 1)

        for data_split_type in ["train", "test"]:
            if data_split_type not in data_config:
                continue
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
                test_datasets[f"{dataset_name}_{data_split_type}"] = dataset

    train_dataset = ConcatDataset(train_dataset)

    train_num_workers = int(config["num_workers"])
    train_loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": True,
        "num_workers": train_num_workers,
        "drop_last": False,
        "persistent_workers": train_num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    # Only set prefetch_factor when explicitly configured; otherwise let PyTorch
    # default to 2 (minimum, most memory-efficient). Large prefetch_factor values
    # can cause significant memory bloat in persistent worker processes.
    _prefetch = config.get("prefetch_factor")
    if train_num_workers > 0 and _prefetch is not None and int(_prefetch) > 0:
        train_loader_kwargs["prefetch_factor"] = int(_prefetch)
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]
    if config.get("eval_num_workers") is None:
        config["eval_num_workers"] = min(4, int(config["num_workers"]))

    test_dataloaders = {}
    for dataset_type, dataset in test_datasets.items():
        eval_num_workers = int(config["eval_num_workers"])
        eval_loader_kwargs = {
            "batch_size": config["eval_batch_size"],
            "shuffle": False,
            "num_workers": eval_num_workers,
            "drop_last": False,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": eval_num_workers > 0,
        }
        # Only set prefetch_factor when explicitly configured; otherwise let
        # PyTorch default to 2. Use eval-specific override if provided,
        # falling back to the general prefetch_factor.
        if eval_num_workers > 0:
            _eval_prefetch = config.get("eval_prefetch_factor", config.get("prefetch_factor"))
            if _eval_prefetch is not None and int(_eval_prefetch) > 0:
                eval_loader_kwargs["prefetch_factor"] = int(_eval_prefetch)
        test_dataloaders[dataset_type] = DataLoader(dataset, **eval_loader_kwargs)

    def _training_cleanup():
        _shutdown_dataloader_workers(train_loader)
        for _loader in test_dataloaders.values():
            _shutdown_dataloader_workers(_loader)
        gc.collect()

    atexit.register(_training_cleanup)

    img_size_hw = (config["image_size"][1], config["image_size"][0])
    vision_encoder = NoMaD_Mamba(
        context_size=config["context_size"],
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        goal_encoder=config.get("goal_encoder", None),
        pretrained_backbone=config.get("pretrained_backbone", True),
        obs_encoding_size=config["encoding_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
        mamba_cfg=MambaConfig.from_dict(config),
        img_size=img_size_hw,
        bidirectional_mamba=config.get("bidirectional_mamba", True),
        use_goal_gate=config.get("use_goal_gate", True),
        use_goal_film=config.get("use_goal_film", True),
        use_goal_mamba_fusion=config.get("use_goal_mamba_fusion", True),
        goal_fusion_hidden_dim=config.get("goal_fusion_hidden_dim", None),
        goal_mamba_fusion_hidden_dim=config.get("goal_mamba_fusion_hidden_dim", None),
        share_visual_backbone=config.get("share_visual_backbone", False),
        use_visual_adapter=config.get("use_visual_adapter", True),
        adapter_hidden_dim=config.get("adapter_hidden_dim", None),
        adapter_scale=config.get("adapter_scale", 0.1),
        use_navigation_aux=config.get("use_navigation_aux", True),
        nav_aux_hidden_dim=config.get("nav_aux_hidden_dim", None),
        use_spatial_mamba_tokens=config.get("use_spatial_mamba_tokens", False),
        drop_backbone_prefix_tokens=config.get("drop_backbone_prefix_tokens", True),
        vit_global_pool=config.get("vit_global_pool", "all_mean"),
    )

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    max_grad_norm = None
    if config["clipping"]:
        max_grad_norm = float(config["max_norm"])
        print("Clipping gradients by global norm:", max_grad_norm)

    current_epoch = 0
    latest_checkpoint = None
    resume_checkpoint = None
    if "load_run" in config:
        load_project_folder = os.path.join("/logs", config["load_run"])
        print("Loading model from", load_project_folder)

        training_latest_path = os.path.join(load_project_folder, "training_latest.pth")
        latest_path = os.path.join(load_project_folder, "latest.pth")
        progress_latest_path = os.path.join(load_project_folder, "training_progress_latest.pth")
        checkpoint_candidates = (
            [training_latest_path, latest_path]
            if os.path.exists(training_latest_path)
            else [latest_path]
        )
        latest_checkpoint = None
        for checkpoint_path in checkpoint_candidates:
            try:
                latest_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                latest_path = checkpoint_path
                break
            except Exception as exc:
                if checkpoint_path == checkpoint_candidates[-1]:
                    raise
                print(
                    f"Skipping unreadable checkpoint {checkpoint_path}: {exc}. "
                    f"Falling back to {checkpoint_candidates[-1]}."
                )
        if isinstance(latest_checkpoint, dict) and (
            "model_state_dict" in latest_checkpoint or "ema_state_dict" in latest_checkpoint
        ):
            resume_checkpoint = latest_checkpoint

        load_model(model, "nomad", latest_checkpoint)
        if isinstance(latest_checkpoint, dict) and "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1
        elif os.path.exists(progress_latest_path):
            progress_checkpoint = torch.load(progress_latest_path, map_location="cpu")
            if isinstance(progress_checkpoint, dict) and "epoch" in progress_checkpoint:
                current_epoch = int(progress_checkpoint["epoch"]) + 1

        if resume_checkpoint is None:
            ema_latest_path = os.path.join(load_project_folder, "ema_latest.pth")
            if os.path.exists(ema_latest_path):
                resume_checkpoint = {
                    "ema_state_dict": torch.load(ema_latest_path, map_location="cpu")
                }

    if len(logical_gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=logical_gpu_ids)
    model = model.to(device)

    stages = []
    if config.get("ablation_unfreeze_all", False):
        stages.append({
            "name": "finetune",
            "epochs": config.get("epochs", 100),
            "freeze_backbone": False,
        })
    elif config.get("multi_stage", False):
        stages.append({
            "name": "representation_warmup",
            "epochs": config.get("stage1_epochs", 10),
            "freeze_backbone": True,
        })
        stages.append({
            "name": "diffusion_tuning",
            "epochs": config.get("stage2_epochs", 50),
            "freeze_backbone": True,
        })
        stages.append({
            "name": "finetune",
            "epochs": config.get("stage3_epochs", 20),
            "freeze_backbone": False,
        })
    else:
        stages.append({
            "name": config.get("train_stage", "finetune"),
            "epochs": config.get("epochs", 100),
            "freeze_backbone": config.get("freeze_backbone", False),
        })

    ema_model = None
    for stage_idx, stage_info in enumerate(stages):
        stage_name = stage_info["name"]
        stage_epochs = stage_info["epochs"]
        
        print(f"=== Starting Stage {stage_idx + 1}/{len(stages)}: {stage_name} for {stage_epochs} epochs ===")
        config["train_stage"] = stage_name
        config["freeze_backbone"] = stage_info["freeze_backbone"]
        config["epochs"] = stage_epochs
        
        raw_model = model.module if hasattr(model, "module") else model
        _apply_train_stage(config, raw_model)

        lr = float(config["lr"])
        optimizer_param_groups = _build_optimizer_param_groups(config, raw_model, lr)
        config_optimizer = config["optimizer"].lower()
        if config_optimizer == "adam":
            optimizer = Adam(
                optimizer_param_groups if optimizer_param_groups is not None else raw_model.parameters(),
                lr=lr,
                betas=(0.9, 0.98),
            )
        elif config_optimizer == "adamw":
            optimizer = AdamW(
                optimizer_param_groups if optimizer_param_groups is not None else raw_model.parameters(),
                lr=lr,
            )
        elif config_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_param_groups if optimizer_param_groups is not None else raw_model.parameters(),
                lr=lr,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Optimizer {config_optimizer} not supported")

        scheduler = None
        if config["scheduler"] is not None:
            config_scheduler = config["scheduler"].lower()
            if config_scheduler == "cosine":
                print("Using cosine annealing with T_max", stage_epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_epochs)
            elif config_scheduler == "cyclic":
                print("Using cyclic LR with cycle", config["cyclic_period"])
                scheduler = torch.optim.lr_scheduler.CyclicLR(
                    optimizer,
                    base_lr=lr / 10.0,
                    max_lr=lr,
                    step_size_up=config["cyclic_period"] // 2,
                    cycle_momentum=False,
                )
            elif config_scheduler == "plateau":
                print("Using ReduceLROnPlateau")
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    factor=config["plateau_factor"],
                    patience=config["plateau_patience"],
                    verbose=True,
                )
            else:
                raise ValueError(f"Scheduler {config_scheduler} not supported")

            if config["warmup"]:
                print("Using warmup scheduler")
                scheduler = GradualWarmupScheduler(
                    optimizer,
                    multiplier=1,
                    total_epoch=config["warmup_epochs"],
                    after_scheduler=scheduler,
                )

        if stage_idx == 0 and "load_run" in config and latest_checkpoint is not None:
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

            load_project_folder = os.path.join("/logs", config["load_run"])
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

        ema_model = train_eval_loop_nomad(
            train_model=config["train"],
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            test_dataloaders=test_dataloaders,
            transform=transform,
            goal_mask_prob=config["goal_mask_prob"],
            epochs=stage_epochs,
            device=device,
            project_folder=config["project_folder"],
            train_stage=stage_name,
            print_log_freq=config["print_log_freq"],
            wandb_log_freq=config["wandb_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            sampling_metrics_freq=int(config.get("sampling_metrics_freq", config["print_log_freq"])),
            current_epoch=current_epoch,
            alpha=float(config["alpha"]),
            use_wandb=config["use_wandb"],
            use_amp=bool(config.get("use_amp", torch.cuda.is_available())),
            eval_fraction=config["eval_fraction"],
            eval_freq=config["eval_freq"],
            resume_checkpoint=resume_checkpoint if stage_idx == 0 else None,
            goal_guidance_min=float(config.get("goal_guidance_min", 0.25)),
            goal_guidance_max=float(config.get("goal_guidance_max", 1.75)),
            goal_guidance_power=float(config.get("goal_guidance_power", 1.5)),
            use_adaptive_guidance=config.get("use_adaptive_guidance", True),
            guidance_confidence_weight=float(config.get("guidance_confidence_weight", 0.35)),
            guidance_uncertainty_weight=float(config.get("guidance_uncertainty_weight", 0.25)),
            guidance_distance_scale=float(config.get("guidance_distance_scale", 10.0)),
            nav_goal_pos_loss_weight=float(config.get("nav_goal_pos_loss_weight", 0.05)),
            nav_contrastive_loss_weight=float(config.get("nav_contrastive_loss_weight", 0.01)),
            nav_contrastive_temperature=float(config.get("nav_contrastive_temperature", 0.1)),
            aux_negative_distance_threshold=float(
                config.get("aux_negative_distance_threshold", config["distance"]["max_dist_cat"])
            ),
            max_grad_norm=max_grad_norm,
            ema_model=ema_model,
            save_epoch_checkpoints=bool(config.get("save_epoch_checkpoints", False)),
            save_training_state=bool(config.get("save_training_state", False)),
            save_optimizer_state=bool(config.get("save_optimizer_state", False)),
        )
        current_epoch += stage_epochs

    _training_cleanup()
    print("FINISHED TRAINING")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="NoMaD-Mamba Training")
    parser.add_argument(
        "--config",
        "-c",
        default="config/nomad_mamba.yaml",
        type=str,
        help="Path to the training config file",
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
        "/logs",
        config["project_name"],
        config["run_name"],
    )
    os.makedirs(config["project_folder"], exist_ok=True)

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="spawn"),
            entity="coisinic243-beijing-university-of-technology",
        )

        def _wandb_finish_atexit():
            try:
                wandb.finish()
            except Exception:
                pass

        atexit.register(_wandb_finish_atexit)
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        wandb.save(args.config, policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
