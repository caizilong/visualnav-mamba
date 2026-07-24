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
from std_msgs.msg import Bool, Float32MultiArray

from topic_names import IMAGE_TOPIC, SAMPLED_ACTIONS_TOPIC, WAYPOINT_TOPIC, VIZ_NAV_IMAGE_TOPIC
from utils import load_model, msg_to_pil, to_numpy, transform_images, pil_to_msg
from vint_train.training.train_utils import get_action


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOPOMAP_IMAGES_DIR = os.path.normpath(os.path.join(BASE_DIR, "../topomaps/images"))
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
latest_raw_img = None  # stores latest raw camera image for trajectory visualization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def callback_obs(msg):
    global latest_raw_img
    obs_img = msg_to_pil(msg)
    latest_raw_img = obs_img  # keep raw image for visualization overlay
    if context_size is None:
        return
    if len(context_queue) < context_size + 1:
        context_queue.append(obs_img)
    else:
        context_queue.pop(0)
        context_queue.append(obs_img)


def draw_trajectory_minimap(img, naction_all, selected_idx, chosen_wp_idx):
    """Draw all 8 sampled trajectories as a top-down minimap overlay on the camera image.

    Args:
        img: numpy array (H, W, 3) BGR image
        naction_all: (num_samples, len_traj_pred, 2) — all trajectories in robot-frame meters
        selected_idx: which trajectory index is the selected one
        chosen_wp_idx: which waypoint index is the chosen control point

    Returns:
        annotated BGR image (numpy array)
    """
    H, W = img.shape[:2]
    num_samples = len(naction_all)

    # Minimap position (bottom-right corner)
    mm_w, mm_h = 300, 280
    mm_margin = 20
    mm_x1 = W - mm_w - mm_margin
    mm_y1 = H - mm_h - mm_margin
    mm_x2 = mm_x1 + mm_w
    mm_y2 = mm_y1 + mm_h

    # Colors for up to 8 trajectories (BGR order)
    colors = [
        (0, 255, 0),     # green  — selected
        (255, 255, 0),   # cyan
        (255, 0, 255),   # magenta
        (0, 255, 255),   # yellow
        (0, 165, 255),   # orange
        (255, 0, 0),     # blue
        (0, 0, 255),     # red
        (255, 255, 255), # white
    ]

    # Adapt scale to trajectory extent
    all_pts = naction_all.reshape(-1, 2)
    max_extent = max(float(np.abs(all_pts).max()), 0.3)
    scale = (mm_h * 0.35) / max_extent  # px per meter

    # Origin (robot position) in minimap pixel coords
    ox = mm_x1 + mm_w // 2
    oy = mm_y1 + int(mm_h * 0.85)

    def robot_to_pixel(x, y):
        """Convert robot-frame (x=forward, y=left) to minimap (x=right, y=up)."""
        px = int(ox - y * scale)
        py = int(oy - x * scale)
        return px, py

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (mm_x1, mm_y1), (mm_x2, mm_y2), (35, 35, 35), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # Border
    cv2.rectangle(img, (mm_x1, mm_y1), (mm_x2, mm_y2), (100, 100, 100), 1)

    # Grid lines every 0.5 m
    grid_step = 0.5
    grid_extent = max_extent + grid_step
    for g in np.arange(-grid_extent, grid_extent + grid_step, grid_step):
        gx, _ = robot_to_pixel(g, 0.0)
        if mm_x1 < gx < mm_x2:
            cv2.line(img, (gx, mm_y1), (gx, mm_y2), (55, 55, 55), 1, cv2.LINE_AA)
        _, gy = robot_to_pixel(0.0, g)
        if mm_y1 < gy < mm_y2:
            cv2.line(img, (mm_x1, gy), (mm_x2, gy), (55, 55, 55), 1, cv2.LINE_AA)

    # Robot position triangle
    rx, ry = robot_to_pixel(0, 0)
    tri_size = 8
    tri_pts = np.array([
        [rx, ry - tri_size],
        [rx - tri_size // 2, ry + tri_size // 2],
        [rx + tri_size // 2, ry + tri_size // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [tri_pts], (0, 200, 0))

    # Draw each trajectory
    for i in range(num_samples):
        traj = naction_all[i]
        color = colors[i % len(colors)]
        thickness = 2 if i == selected_idx else 1

        # Collect pixel positions
        pts = [robot_to_pixel(float(wp[0]), float(wp[1])) for wp in traj]

        # Draw line segments
        for j in range(len(pts) - 1):
            cv2.line(img, pts[j], pts[j + 1], color, thickness, cv2.LINE_AA)

        # Draw waypoint markers
        for j, pt in enumerate(pts):
            if i == selected_idx and j == chosen_wp_idx:
                cv2.circle(img, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, pt, 2, color, -1, cv2.LINE_AA)

    # Labels
    cv2.putText(img, "Trajectories (top-down)", (mm_x1 + 6, mm_y1 + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"scale: {scale:.0f} px/m  |  samples: {num_samples}",
                (mm_x1 + 6, mm_y1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(img, "selected", (mm_x1 + 6, mm_y1 + 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)

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


def diffusion_guidance_scale(
    step_idx: int,
    total_steps: int,
    min_scale: float,
    max_scale: float,
    power: float,
    goal_confidence: float = None,
    action_uncertainty: torch.Tensor = None,
    confidence_weight: float = 0.0,
    uncertainty_weight: float = 0.0,
):
    if total_steps <= 1:
        base_scale = max_scale
    else:
        progress = step_idx / float(total_steps - 1)
        base_scale = min_scale + (max_scale - min_scale) * (progress ** power)

    if goal_confidence is None and action_uncertainty is None:
        return base_scale

    scale = torch.full(
        (1,),
        float(base_scale),
        device=action_uncertainty.device if action_uncertainty is not None else device,
    )
    if goal_confidence is not None and confidence_weight != 0:
        confidence = float(np.clip(goal_confidence, 0.0, 1.0))
        scale = scale * (1 + confidence_weight * (2 * confidence - 1))
    if action_uncertainty is not None and uncertainty_weight != 0:
        uncertainty = action_uncertainty.clamp_min(0)
        uncertainty = uncertainty / (uncertainty.detach().mean() + 1e-6)
        scale = scale / (1 + uncertainty_weight * uncertainty)
    return scale.clamp(min_scale, max_scale)


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

    valid_model = model_params.get("model_type") == "nomad"
    valid_encoder = model_params.get("vision_encoder") in ("nomad_mamba", "nomad_vint")
    if not valid_model or not valid_encoder:
        raise ValueError(
            "navigate.py only supports NoMaD configs: "
            "`model_type: nomad` and `vision_encoder: nomad_mamba` or `nomad_vint`."
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
    guidance_min = args.guidance_min if args.guidance_min is not None else model_params.get("goal_guidance_min", 1.0)
    guidance_max = args.guidance_max if args.guidance_max is not None else model_params.get("goal_guidance_max", 1.0)
    guidance_power = args.guidance_power if args.guidance_power is not None else model_params.get("goal_guidance_power", 1.0)
    use_adaptive_guidance = model_params.get("use_adaptive_guidance", True)
    guidance_confidence_weight = model_params.get("guidance_confidence_weight", 0.35)
    guidance_uncertainty_weight = model_params.get("guidance_uncertainty_weight", 0.25)
    guidance_distance_scale = max(float(model_params.get("guidance_distance_scale", 10.0)), 1e-6)

    topomap_dir = os.path.join(TOPOMAP_IMAGES_DIR, args.dir)
    topomap_filenames = sorted(
        [
            name
            for name in os.listdir(topomap_dir)
            if name.lower().endswith(".png") and os.path.splitext(name)[0].isdigit()
        ],
        key=lambda x: int(os.path.splitext(x)[0]),
    )
    if not topomap_filenames:
        raise FileNotFoundError(f"No numeric PNG topomap nodes found in {topomap_dir}")
    num_nodes = len(topomap_filenames)
    topomap = []
    for i in range(num_nodes):
        image_path = os.path.join(topomap_dir, topomap_filenames[i])
        topomap.append(PILImage.open(image_path))

    closest_node = 0
    goal_hit_count = 0
    assert -1 <= args.goal_node < len(topomap), "Invalid goal index"
    goal_node = len(topomap) - 1 if args.goal_node == -1 else args.goal_node

    rospy.init_node("EXPLORATION", anonymous=False)
    rate = rospy.Rate(RATE)
    rospy.Subscriber(IMAGE_TOPIC, Image, callback_obs, queue_size=1)
    waypoint_pub = rospy.Publisher(WAYPOINT_TOPIC, Float32MultiArray, queue_size=1)
    sampled_actions_pub = rospy.Publisher(SAMPLED_ACTIONS_TOPIC, Float32MultiArray, queue_size=1)
    goal_pub = rospy.Publisher("/topoplan/reached_goal", Bool, queue_size=1)
    viz_pub = rospy.Publisher(VIZ_NAV_IMAGE_TOPIC, Image, queue_size=1)

    print("Registered with master node. Waiting for image observations...")

    num_diffusion_iters = model_params["num_diffusion_iters"]
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    naction_all = np.zeros((0,))  # placeholder for trajectory visualization
    while not rospy.is_shutdown():
        chosen_waypoint = np.zeros(4)
        if len(context_queue) > model_params["context_size"]:
            obs_images = transform_images(context_queue, model_params["image_size"], center_crop=False)
            obs_images = torch.split(obs_images, 3, dim=1)
            obs_images = torch.cat(obs_images, dim=1).to(device)
            mask = torch.zeros(1).long().to(device)

            start = max(closest_node - args.radius, 0)
            end = min(closest_node + args.radius + 1, goal_node)
            goal_image = [
                transform_images(g_img, model_params["image_size"], center_crop=False).to(device)
                for g_img in topomap[start : end + 1]
            ]
            goal_image = torch.concat(goal_image, dim=0)

            with torch.inference_mode():
                obsgoal_cond = model(
                    "vision_encoder",
                    obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                    goal_img=goal_image,
                    input_goal_mask=mask.repeat(len(goal_image)),
                )
                dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
                dists = to_numpy(dists.flatten())
                min_idx = np.argmin(dists)
                closest_node = min_idx + start
                print("closest node:", closest_node)
                sg_idx = min(min_idx + int(dists[min_idx] < args.close_threshold), len(obsgoal_cond) - 1)
                goal_confidence = float(np.exp(-max(float(dists[sg_idx]), 0.0) / guidance_distance_scale))
                cond_obs_cond = obsgoal_cond[sg_idx].unsqueeze(0)
                selected_goal_img = goal_image[sg_idx].unsqueeze(0)
                no_goal_mask = torch.ones(1, dtype=torch.long, device=device)
                obs_cond = model(
                    "vision_encoder",
                    obs_img=obs_images,
                    goal_img=selected_goal_img,
                    input_goal_mask=no_goal_mask,
                )

                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)
                    cond_obs_cond = cond_obs_cond.repeat(args.num_samples, 1)
                else:
                    obs_cond = obs_cond.repeat(args.num_samples, 1, 1)
                    cond_obs_cond = cond_obs_cond.repeat(args.num_samples, 1, 1)

                naction = torch.randn((args.num_samples, model_params["len_traj_pred"], 2), device=device)
                noise_scheduler.set_timesteps(num_diffusion_iters)

                start_time = time.time()
                total_steps = len(noise_scheduler.timesteps)
                for step_idx, k in enumerate(noise_scheduler.timesteps[:]):
                    unconditional_noise = model(
                        "noise_pred_net",
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond,
                    )
                    conditional_noise = model(
                        "noise_pred_net",
                        sample=naction,
                        timestep=k,
                        global_cond=cond_obs_cond,
                    )
                    guidance_scale = diffusion_guidance_scale(
                        step_idx,
                        total_steps,
                        guidance_min,
                        guidance_max,
                        guidance_power,
                        goal_confidence=goal_confidence if use_adaptive_guidance else None,
                        action_uncertainty=naction.var(dim=0, unbiased=False).mean().reshape(1)
                        if use_adaptive_guidance and args.num_samples > 1
                        else None,
                        confidence_weight=guidance_confidence_weight,
                        uncertainty_weight=guidance_uncertainty_weight,
                    )
                    noise_pred = unconditional_noise + guidance_scale * (conditional_noise - unconditional_noise)
                    naction = noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
                print("time elapsed:", time.time() - start_time)

            naction = to_numpy(get_action(naction))
            naction_all = naction.copy()  # save all trajectories for visualization
            sampled_actions_msg = Float32MultiArray()
            sampled_actions_msg.data = np.concatenate((np.array([0]), naction.flatten()))
            sampled_actions_pub.publish(sampled_actions_msg)

            naction = naction[0]
            chosen_waypoint = naction[args.waypoint]

        if model_params["normalize"]:
            chosen_waypoint[:2] *= (MAX_V / RATE)
        waypoint_msg = Float32MultiArray()
        waypoint_msg.data = chosen_waypoint
        waypoint_pub.publish(waypoint_msg)

        if closest_node == goal_node:
            goal_hit_count += 1
        else:
            goal_hit_count = 0
        reached_goal = goal_hit_count >= args.goal_confirmations
        goal_pub.publish(reached_goal)
        if reached_goal:
            print(f"Reached goal after {goal_hit_count} consecutive confirmations. Stopping...")
            break

        # --- Trajectory visualization overlay ---
        if latest_raw_img is not None and naction_all.size > 1:
            try:
                viz_img = np.array(latest_raw_img.convert('RGB'))[:, :, ::-1].copy()
                viz_img = draw_trajectory_minimap(viz_img, naction_all, selected_idx=0, chosen_wp_idx=args.waypoint)
                viz_msg = pil_to_msg(PILImage.fromarray(viz_img[:, :, ::-1]), encoding="rgb8")
                viz_pub.publish(viz_msg)
            except Exception:
                pass  # never let visualization errors crash navigation

        rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NoMaD-Mamba navigation on the LoCoBot")
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
        help="index of the waypoint used for navigation",
    )
    parser.add_argument(
        "--dir",
        "-d",
        default="topomap",
        type=str,
        help="path to topomap images",
    )
    parser.add_argument(
        "--goal-node",
        "-g",
        default=-1,
        type=int,
        help="goal node index in the topomap (if -1, use the last node)",
    )
    parser.add_argument(
        "--close-threshold",
        "-t",
        default=3,
        type=int,
        help="distance threshold for moving to the next sub-goal",
    )
    parser.add_argument(
        "--radius",
        "-r",
        default=4,
        type=int,
        help="temporal search radius over topological nodes",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        default=8,
        type=int,
        help="number of sampled trajectories",
    )
    parser.add_argument(
        "--goal-confirmations",
        default=5,
        type=int,
        help="number of consecutive goal-node localizations required before publishing reached_goal",
    )
    parser.add_argument(
        "--guidance-min",
        default=None,
        type=float,
        help="minimum goal guidance scale at early denoising steps",
    )
    parser.add_argument(
        "--guidance-max",
        default=None,
        type=float,
        help="maximum goal guidance scale at late denoising steps",
    )
    parser.add_argument(
        "--guidance-power",
        default=None,
        type=float,
        help="power used by the guidance schedule",
    )
    args = parser.parse_args()
    args = apply_benchmark_config(args, "navigate")
    print(f"Using {device}")
    main(args)
