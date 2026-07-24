import argparse
import os
import time

import cv2
import numpy as np
import rospy
import torch
import yaml
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

from topic_names import IMAGE_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC, VIZ_NAV_IMAGE_TOPIC
from utils import load_model, msg_to_pil, pil_to_msg, to_numpy, transform_images
from vint_train.training.train_utils import get_action


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_CONFIG_PATH = os.environ.get(
    "ROBOT_CONFIG_PATH",
    os.path.normpath(os.path.join(BASE_DIR, "../config/robot.yaml")),
)
MODEL_CONFIG_PATH = os.environ.get(
    "MODEL_CONFIG_PATH",
    os.path.normpath(os.path.join(BASE_DIR, "../config/models.yaml")),
)

with open(ROBOT_CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)
MAX_V = robot_config["max_v"]
RATE = robot_config["frame_rate"]

context_queue = []
context_size = None
latest_raw_img = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    global latest_raw_img
    obs_img = msg_to_pil(msg)
    latest_raw_img = obs_img
    if context_size is None:
        return
    if len(context_queue) < context_size + 1:
        context_queue.append(obs_img)
    else:
        context_queue.pop(0)
        context_queue.append(obs_img)


def draw_trajectory_minimap(img, naction_all, selected_idx, chosen_wp_idx):
    """Draw sampled trajectories as a top-down minimap overlay on a BGR image."""
    H, W = img.shape[:2]
    num_samples = len(naction_all)

    mm_w, mm_h = 300, 280
    mm_margin = 20
    mm_x1 = W - mm_w - mm_margin
    mm_y1 = H - mm_h - mm_margin
    mm_x2 = mm_x1 + mm_w
    mm_y2 = mm_y1 + mm_h

    colors = [
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (0, 165, 255),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 255),
    ]

    all_pts = naction_all[:, :, :2].reshape(-1, 2)
    max_extent = max(float(np.abs(all_pts).max()), 0.3)
    scale = (mm_h * 0.35) / max_extent

    ox = mm_x1 + mm_w // 2
    oy = mm_y1 + int(mm_h * 0.85)

    def robot_to_pixel(x, y):
        px = int(ox - y * scale)
        py = int(oy - x * scale)
        return px, py

    overlay = img.copy()
    cv2.rectangle(overlay, (mm_x1, mm_y1), (mm_x2, mm_y2), (35, 35, 35), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.rectangle(img, (mm_x1, mm_y1), (mm_x2, mm_y2), (100, 100, 100), 1)

    grid_step = 0.5
    grid_extent = max_extent + grid_step
    for g in np.arange(-grid_extent, grid_extent + grid_step, grid_step):
        gx, _ = robot_to_pixel(g, 0.0)
        if mm_x1 < gx < mm_x2:
            cv2.line(img, (gx, mm_y1), (gx, mm_y2), (55, 55, 55), 1, cv2.LINE_AA)
        _, gy = robot_to_pixel(0.0, g)
        if mm_y1 < gy < mm_y2:
            cv2.line(img, (mm_x1, gy), (mm_x2, gy), (55, 55, 55), 1, cv2.LINE_AA)

    rx, ry = robot_to_pixel(0, 0)
    tri_size = 8
    tri_pts = np.array(
        [
            [rx, ry - tri_size],
            [rx - tri_size // 2, ry + tri_size // 2],
            [rx + tri_size // 2, ry + tri_size // 2],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [tri_pts], (0, 200, 0))

    chosen_wp_idx = int(np.clip(chosen_wp_idx, 0, naction_all.shape[1] - 1))
    selected_idx = int(np.clip(selected_idx, 0, num_samples - 1))

    for i in range(num_samples):
        traj = naction_all[i]
        color = colors[i % len(colors)]
        thickness = 2 if i == selected_idx else 1
        pts = [robot_to_pixel(float(wp[0]), float(wp[1])) for wp in traj]

        for j in range(len(pts) - 1):
            cv2.line(img, pts[j], pts[j + 1], color, thickness, cv2.LINE_AA)

        for j, pt in enumerate(pts):
            if i == selected_idx and j == chosen_wp_idx:
                cv2.circle(img, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 2, color, -1, cv2.LINE_AA)

    cv2.putText(
        img,
        "Trajectories (top-down)",
        (mm_x1 + 6, mm_y1 + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"scale: {scale:.0f} px/m  |  samples: {num_samples}",
        (mm_x1 + 6, mm_y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"selected wp: {chosen_wp_idx}",
        (mm_x1 + 6, mm_y1 + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    return img


def apply_benchmark_config(args: argparse.Namespace, section: str) -> argparse.Namespace:
    if not args.benchmark_config:
        return args

    benchmark_path = args.benchmark_config
    if not os.path.isabs(benchmark_path):
        benchmark_path = os.path.normpath(os.path.join(BASE_DIR, benchmark_path))
    with open(benchmark_path, "r") as f:
        benchmark_cfg = yaml.safe_load(f) or {}

    merged_cfg = {}
    merged_cfg.update(benchmark_cfg.get("common", {}))
    merged_cfg.update(benchmark_cfg.get(section, {}))
    for key, value in merged_cfg.items():
        attr_name = key.replace("-", "_")
        if hasattr(args, attr_name):
            setattr(args, attr_name, value)

    print(f"Loaded benchmark config from {benchmark_path} ({section})")
    return args


def _load_model_params(model_name: str):
    with open(MODEL_CONFIG_PATH, "r") as f:
        model_paths = yaml.safe_load(f)

    if model_name not in model_paths:
        raise KeyError(f"Unknown model '{model_name}'. Check {MODEL_CONFIG_PATH}.")

    model_config_path = model_paths[model_name]["config_path"]
    if not os.path.isabs(model_config_path):
        model_config_path = os.path.normpath(os.path.join(BASE_DIR, model_config_path))
    with open(model_config_path, "r") as f:
        model_params = yaml.safe_load(f)

    if model_params.get("model_type") != "nomad" or model_params.get("vision_encoder") != "nomad_mamba":
        raise ValueError(
            "explore.py only supports NoMaD-Mamba configs: "
            "`model_type: nomad` and `vision_encoder: nomad_mamba`."
        )

    ckpt_path = model_paths[model_name]["ckpt_path"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.normpath(os.path.join(BASE_DIR, ckpt_path))
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model weights not found at {ckpt_path}")

    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, model_params, device).to(device)
    model.eval()
    return model, model_params


def main(args: argparse.Namespace):
    global context_size

    model, model_params = _load_model_params(args.model)
    context_size = model_params["context_size"]

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    viz_pub = rospy.Publisher(VIZ_NAV_IMAGE_TOPIC, Image, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    naction_all = np.zeros((0,))
    while not rospy.is_shutdown():
        if len(context_queue) > model_params["context_size"]:
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False).to(device)
            fake_goal = torch.randn(
                (1, 3, *obs_images.shape[-2:]),
                dtype=obs_images.dtype,
                device=device,
            )
            mask = torch.ones(1).long().to(device)

            with torch.no_grad():
                obs_cond = model(
                    "vision_encoder",
                    obs_img=obs_images,
                    goal_img=fake_goal,
                    input_goal_mask=mask,
                )

                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)

                naction = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = time.time()
                for k in noise_scheduler.timesteps[:]:
                    noise_pred = model(
                        "noise_pred_net",
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                    )
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction,
                    ).prev_sample
                print("time elapsed:", time.time() - start_time)

            naction = to_numpy(get_action(naction))
            naction_all = naction.copy()
            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
            sampled_actions_pub.publish(sampled_actions_msg)

            chosen_waypoint = naction[0][args.waypoint]
            if model_params["normalize"]:
                chosen_waypoint *= (MAX_V / RATE)
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint
            waypoint_pub.publish(waypoint_msg)

            if latest_raw_img is not None and naction_all.size > 1:
                try:
                    viz_img = np.array(latest_raw_img.convert("RGB"))[:, :, ::-1].copy()
                    viz_img = draw_trajectory_minimap(
                        viz_img,
                        naction_all,
                        selected_idx=0,
                        chosen_wp_idx=args.waypoint,
                    )
                    viz_msg = pil_to_msg(PILImage.fromarray(viz_img[:, :, ::-1]), encoding="rgb8")
                    viz_pub.publish(viz_msg)
                except Exception:
                    pass
            print("Published waypoint")

        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoMaD-Mamba exploration on the LoCoBot")
    parser.add_argument(
        "--model",
        "-m",
        default="nomad_mamba",
        type=str,
        help="model name (check ../config/models.yaml)",
    )
    parser.add_argument(
        "--benchmark-config",
        default=None,
        type=str,
        help="optional YAML file with reproducible benchmark defaults",
    )
    parser.add_argument(
        "--waypoint",
        "-w",
        default=2,
        type=int,
        help="index of waypoint used for control",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="number of sampled trajectories",
    )
    args = parser.parse_args()
    args = apply_benchmark_config(args, "explore")
    print(f"Using {device}")
    main(args)
