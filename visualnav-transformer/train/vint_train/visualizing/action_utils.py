import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List
import wandb
import yaml
import torch
import torch.nn as nn
from vint_train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

# 加载数据集配置，用于获取不同数据集的 metric_waypoint_spacing、相机内参等信息
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)


def visualize_traj_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    wandb_step: Optional[int] = None,
    wandb_epoch: Optional[float] = None,
):
    """
    将预测的轨迹与 GT 轨迹在 ego 坐标系下进行可视化对比（使用最后一个 batch）。

    Args:
        batch_obs_images: 观测图像 batch，shape [B, H, W, C]
        batch_goal_images: 目标图像 batch，shape [B, H, W, C]
        dataset_indices: 每条样本所属数据集在 data_config.yaml 中的索引
        batch_goals: 局部坐标系下的目标位置 [B, 2]
        batch_pred_waypoints: 预测轨迹 [B, T, 2/4] 或 [B, num_samples, T, 2/4]
        batch_label_waypoints: GT 轨迹 [B, T, 2/4]
        eval_type: 评估类型字符串（用于日志前缀）
        normalized: 轨迹是否在训练时被归一化（若是，则这里会乘回 metric_waypoint_spacing）
        save_folder: 保存图片的工程目录
        epoch: 当前 epoch 编号
        num_images_preds: 最多可视化多少个样本
        use_wandb: 是否将图片上传到 wandb
        display: 是否直接在窗口中展示（一般在训练中为 False）
    """
    visualize_path = None
    if save_folder is not None:
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_goals)
        == len(batch_pred_waypoints)
        == len(batch_label_waypoints)
    )

    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        obs_img = numpy_to_img(batch_obs_images[i])
        goal_img = numpy_to_img(batch_goal_images[i])
        dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]

        if normalized:
            pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        compare_waypoints_pred_to_label(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            pred_waypoints,
            label_waypoints,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        log_dict = {f"{eval_type}_action_prediction": wandb_list}
        if wandb_step is not None:
            log_dict["epoch"] = wandb_epoch if wandb_epoch is not None else wandb_step
            wandb.log(log_dict, step=int(wandb_step), commit=False)
        else:
            wandb.log(log_dict, commit=False)


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    对单个样本，生成一个包含 3 列的可视化图：
    - 左：在平面上画出预测轨迹与 GT 轨迹
    - 中：将轨迹投影到观测图像上（若有相机内参）
    - 右：显示目标图像

    Args:
        obs_img: 当前观测的图像（PIL 或 numpy 格式）
        goal_img: 目标图像
        dataset_name: 在 data_config.yaml 中的数据集名称，如 "recon"
        goal_pos: 目标在局部坐标下的位置
        pred_waypoints: 预测轨迹（可以是多条采样轨迹或单条）
        label_waypoints: GT 轨迹
        save_path: 若不为 None，则保存到该路径
        display: 是否在屏幕上显示（一般在离线调试时使用）
    """

    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])
    if len(pred_waypoints.shape) > 2:
        trajs = [*pred_waypoints, label_waypoints]
    else:
        trajs = [pred_waypoints, label_waypoints]
    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    ax[2].imshow(goal_img)

    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    if not display:
        plt.close(fig)


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    dataset_name: str,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
):
    """
    在一张图像上画出轨迹与关键点。

    若当前数据集在 data_config.yaml 中配置了相机内外参（camera_metrics），则会：
    - 使用 `get_pos_pixels` 将局部坐标系下的轨迹点投影到像素坐标
    - 在图像上绘制轨迹折线与关键点
    若未配置内参，则仅简单调用 `imshow` 展示原图。
    """
    assert len(list_trajs) <= len(traj_colors), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        dataset_name in data_config
    ), f"Dataset {dataset_name} not found in data/data_config.yaml"

    ax.imshow(img)
    if (
        "camera_metrics" in data_config[dataset_name]
        and "camera_height" in data_config[dataset_name]["camera_metrics"]
        and "camera_matrix" in data_config[dataset_name]["camera_metrics"]
        and "dist_coeffs" in data_config[dataset_name]["camera_metrics"]
    ):
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]

        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
        camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

        k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
        k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
        p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
        p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
        k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
        dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

        for i, traj in enumerate(list_trajs):
            xy_coords = traj[:, :2]  # (horizon, 2)
            traj_pixels = get_pos_pixels(
                xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
            )
            if len(traj_pixels.shape) == 2:
                ax.plot(
                    traj_pixels[:250, 0],
                    traj_pixels[:250, 1],
                    color=traj_colors[i],
                    lw=2.5,
                )

        for i, point in enumerate(list_points):
            if len(point.shape) == 1:
                # add a dimension to the front of point
                point = point[None, :2]
            else:
                point = point[:, :2]
            pt_pixels = get_pos_pixels(
                point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
            )
            ax.plot(
                pt_pixels[:250, 0],
                pt_pixels[:250, 1],
                color=point_colors[i],
                marker="o",
                markersize=10.0,
            )
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    在 2D 平面上绘制多条轨迹与若干点，支持可选的 yaw 向量可视化。

    Args:
        ax: matplotlib axis
        list_trajs: 轨迹列表，每条为 (T, 2) 或 (T, 4)（后两维为 yaw 的 sin/cos 或角度）
        list_points: 关键点列表，每个为 (2,)
        traj_colors / point_colors: 颜色列表
        traj_labels / point_labels: 图例标签
        traj_alphas / point_alphas: 透明度
        quiver_freq: 若轨迹包含 yaw，隔多少个点画一次箭头
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "Not enough colors for trajectories"
    assert len(list_points) <= len(point_colors), "Not enough colors for points"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "Not enough labels for trajectories"
    assert point_labels is None or len(list_points) == len(point_labels), "Not enough labels for points"

    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        if traj.shape[1] > 2 and quiver_freq > 0:  # traj data also includes yaw of the robot
            bearings = gen_bearings_from_waypoints(traj)
            ax.quiver(
                traj[::quiver_freq, 0],
                traj[::quiver_freq, 1],
                bearings[::quiver_freq, 0],
                bearings[::quiver_freq, 1],
                color=traj_colors[i] * 0.5,
                scale=1.0,
            )
    for i, pt in enumerate(list_points):
        if point_labels is None:
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    
    # put the legend below the plot
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    ax.set_aspect("equal", "box")


def angle_to_unit_vector(theta):
    """Converts an angle to a unit vector."""
    return np.array([np.cos(theta), np.sin(theta)])


def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """Generate bearings from waypoints, (x, y, sin(theta), cos(theta))."""
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # label is sin/cos repr
            v = waypoints[i, 2:]
            # normalize v
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # label is radians repr
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    使用给定相机参数，将 2D 地面坐标 (x, y) 投影到图像平面上的像素坐标 (u, v)。

    Args:
        xy: (batch_size, horizon, 2) 的坐标序列
        camera_height: 相机离地高度（米）
        camera_x_offset: 相机在车体坐标中的 x 方向偏移（米）
        camera_matrix: 3x3 相机内参矩阵
        dist_coeffs: 镜头畸变系数
    Returns:
        uv: (batch_size, horizon, 2) 的像素坐标
    """
    batch_size, horizon, _ = xy.shape

    # create 3D coordinates with the camera positioned at the given height
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # create dummy rotation and translation vectors
    rvec = tvec = (0, 0, 0)

    xyz[..., 0] += camera_x_offset
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    将局部 (x, y) 坐标投影到图像平面，并根据 clip 参数决定是否裁剪到图像边界。

    - clip=True: 将所有点坐标裁剪到可视化图像范围内
    - clip=False: 只保留落在图像内部的点
    """
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels


def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    Args:
        fx: focal length in x direction
        fy: focal length in y direction
        cx: principal point x coordinate
        cy: principal point y coordinate
    Returns:
        camera matrix
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
