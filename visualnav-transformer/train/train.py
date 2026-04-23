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

    test_dataloaders = {}
    for dataset_type, dataset in test_datasets.items():
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
        goal_fusion_hidden_dim=config.get("goal_fusion_hidden_dim", None),
        share_visual_backbone=config.get("share_visual_backbone", False),
        adapter_hidden_dim=config.get("adapter_hidden_dim", None),
        adapter_scale=config.get("adapter_scale", 0.1),
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        elif config["scheduler"] == "cyclic":
            print("Using cyclic LR with cycle", config["cyclic_period"])
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=lr / 10.0,
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
        print("Loading model from", load_project_folder)

        latest_path = os.path.join(load_project_folder, "latest.pth")
        training_latest_path = os.path.join(load_project_folder, "training_latest.pth")
        if os.path.exists(training_latest_path):
            latest_path = training_latest_path

        latest_checkpoint = torch.load(latest_path, map_location="cpu")
        if isinstance(latest_checkpoint, dict) and (
            "model_state_dict" in latest_checkpoint or "ema_state_dict" in latest_checkpoint
        ):
            resume_checkpoint = latest_checkpoint

        load_model(model, "nomad", latest_checkpoint)
        if isinstance(latest_checkpoint, dict) and "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    if len(logical_gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=logical_gpu_ids)
    model = model.to(device)

    if "load_run" in config:
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

        load_project_folder = os.path.join("logs", config["load_run"])
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
    config["project_folder"] = os.path.join("logs", config["project_name"], config["run_name"])
    os.makedirs(config["project_folder"])

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
