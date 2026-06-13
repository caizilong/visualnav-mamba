import os
import sys

# Ensure local diffusion_policy package is importable in deployment runtime.
_DEPLOY_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_VINT_ROOT = os.path.dirname(os.path.dirname(_DEPLOY_SRC_DIR))
_DIFFUSION_POLICY_CANDIDATES = [
    os.path.join(_VINT_ROOT, "diffusion_policy"),
    os.path.join(os.path.dirname(_VINT_ROOT), "diffusion_policy"),
]
for _root in _DIFFUSION_POLICY_CANDIDATES:
    expected_module = os.path.join(
        _root, "diffusion_policy", "model", "diffusion", "conditional_unet1d.py"
    )
    if os.path.isfile(expected_module) and _root not in sys.path:
        sys.path.insert(0, _root)
        break

try:
    from sensor_msgs.msg import Image
except ImportError:
    Image = None

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image as PILImage
from torchvision import transforms
from typing import List

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO
from vint_train.models.nomad.mamba2 import MambaConfig
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_mamba import NoMaD_Mamba


def load_model(
    model_path: str,
    config: dict,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """Build NoMaD-Mamba and load checkpoint weights."""
    if config.get("model_type") != "nomad" or config.get("vision_encoder") != "nomad_mamba":
        raise ValueError(
            "This deployment runtime only supports `model_type: nomad` + "
            "`vision_encoder: nomad_mamba`."
        )

    img_size_hw = (config["image_size"][1], config["image_size"][0])
    vision_encoder = NoMaD_Mamba(
        context_size=config["context_size"],
        obs_encoder=config.get("obs_encoder", "efficientnet-b0"),
        goal_encoder=config.get("goal_encoder", None),
        pretrained_backbone=config.get("pretrained_backbone", False),
        obs_encoding_size=config["encoding_size"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
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

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model


def msg_to_pil(msg: Image) -> PILImage.Image:
    if Image is None:
        raise ImportError("sensor_msgs is required for msg_to_pil; install ROS sensor_msgs first.")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return PILImage.fromarray(img)


def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    if Image is None:
        raise ImportError("sensor_msgs is required for pil_to_msg; install ROS sensor_msgs first.")
    img = np.asarray(pil_img)
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes()
    ros_image.step = ros_image.width
    return ros_image


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def transform_images(
    pil_imgs: List[PILImage.Image],
    image_size: List[int],
    center_crop: bool = False,
) -> torch.Tensor:
    """Convert one or more PIL images into normalized tensor [1, 3*N, H, W]."""
    transform_type = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if type(pil_imgs) != list:
        pil_imgs = [pil_imgs]
    transf_imgs = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(image_size)
        transf_img = transform_type(pil_img)
        transf_imgs.append(torch.unsqueeze(transf_img, 0))
    return torch.cat(transf_imgs, dim=1)


def clip_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi
