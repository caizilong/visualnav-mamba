import os
import wandb
import numpy as np
from typing import List, Optional, Tuple
from vint_train.visualizing.visualize_utils import numpy_to_img
import matplotlib.pyplot as plt


def visualize_dist_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    batch_dist_preds: np.ndarray,
    batch_dist_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
    rounding: int = 4,
    dist_error_threshold: float = 3.0,
    wandb_step: Optional[int] = None,
    wandb_epoch: Optional[float] = None,
):
    """
    将“观测-目标对”的距离预测与标签可视化成图像（单样本两张图：Observation / Goal）。

    若预测误差超过 dist_error_threshold，则标题文字用红色，便于快速发现误差较大的样本。
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_dist_preds)
        == len(batch_dist_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        dist_pred = np.round(batch_dist_preds[i], rounding)
        dist_label = np.round(batch_dist_labels[i], rounding)
        obs_image = numpy_to_img(batch_obs_images[i])
        goal_image = numpy_to_img(batch_goal_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")
        text_color = "black"
        if abs(dist_pred - dist_label) > dist_error_threshold:
            text_color = "red"

        display_distance_pred(
            [obs_image, goal_image],
            ["Observation", "Goal"],
            dist_pred,
            dist_label,
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        log_dict = {f"{eval_type}_dist_prediction": wandb_list}
        if wandb_step is not None:
            log_dict["epoch"] = wandb_epoch if wandb_epoch is not None else wandb_step
            wandb.log(log_dict, step=int(wandb_step), commit=False)
        else:
            wandb.log(log_dict, commit=False)


def visualize_dist_pairwise_pred(
    batch_obs_images: np.ndarray,
    batch_close_images: np.ndarray,
    batch_far_images: np.ndarray,
    batch_close_preds: np.ndarray,
    batch_far_preds: np.ndarray,
    batch_close_labels: np.ndarray,
    batch_far_labels: np.ndarray,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    wandb_step: Optional[int] = None,
    wandb_epoch: Optional[float] = None,
    display: bool = False,
    rounding: int = 4,
):
    """
    针对“成对距离比较”任务的可视化：Observation + Close Goal + Far Goal。

    - 标题中会打印 close / far 的预测与标签
    - 若模型未能满足 close_pred < far_pred，则用红色标记文本
    """
    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "pairwise_dist_classification",
    )
    if not os.path.isdir(visualize_path):
        os.makedirs(visualize_path)
    assert (
        len(batch_obs_images)
        == len(batch_close_images)
        == len(batch_far_images)
        == len(batch_close_preds)
        == len(batch_far_preds)
        == len(batch_close_labels)
        == len(batch_far_labels)
    )
    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    for i in range(min(batch_size, num_images_preds)):
        close_dist_pred = np.round(batch_close_preds[i], rounding)
        far_dist_pred = np.round(batch_far_preds[i], rounding)
        close_dist_label = np.round(batch_close_labels[i], rounding)
        far_dist_label = np.round(batch_far_labels[i], rounding)
        obs_image = numpy_to_img(batch_obs_images[i])
        close_image = numpy_to_img(batch_close_images[i])
        far_image = numpy_to_img(batch_far_images[i])

        save_path = None
        if save_folder is not None:
            save_path = os.path.join(visualize_path, f"{i}.png")

        if close_dist_pred < far_dist_pred:
            text_color = "black"
        else:
            text_color = "red"

        display_distance_pred(
            [obs_image, close_image, far_image],
            ["Observation", "Close Goal", "Far Goal"],
            f"close_pred = {close_dist_pred}, far_pred = {far_dist_pred}",
            f"close_label = {close_dist_label}, far_label = {far_dist_label}",
            text_color,
            save_path,
            display,
        )
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    if use_wandb:
        log_dict = {f"{eval_type}_pairwise_classification": wandb_list}
        if wandb_step is not None:
            log_dict["epoch"] = wandb_epoch if wandb_epoch is not None else wandb_step
            wandb.log(log_dict, step=int(wandb_step), commit=False)
        else:
            wandb.log(log_dict, commit=False)


def display_distance_pred(
    imgs: list,
    titles: list,
    dist_pred: float,
    dist_label: float,
    text_color: str = "black",
    save_path: Optional[str] = None,
    display: bool = False,
):
    """底层绘图函数：将若干图像 + 文本整体排版成一张图并保存/显示。"""
    plt.figure()
    fig, ax = plt.subplots(1, len(imgs))

    plt.suptitle(f"prediction: {dist_pred}\nlabel: {dist_label}", color=text_color)

    for axis, img, title in zip(ax, imgs, titles):
        axis.imshow(img)
        axis.set_title(title)
        axis.xaxis.set_visible(False)
        axis.yaxis.set_visible(False)

    # make the plot large
    fig.set_size_inches((18.5 / 3) * len(imgs), 10.5)

    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )
    if not display:
        plt.close(fig)
