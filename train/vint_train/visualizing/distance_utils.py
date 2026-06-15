import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def _image_to_hwc(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu()
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected a CHW RGB image, got shape {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        return image.permute(1, 2, 0).numpy()
    image = image.float()
    if image.numel() > 0 and image.max().item() <= 1.0:
        image = image * 255.0
    return image.clamp(0, 255).byte().permute(1, 2, 0).numpy()


def visualize_distance_predictions(
    batch_obs_images: torch.Tensor,
    batch_goal_images: torch.Tensor,
    batch_dist_preds: torch.Tensor,
    batch_dist_labels: torch.Tensor,
    eval_type: str,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    dist_error_threshold: float = 3.0,
) -> None:
    """Save observation-goal pairs annotated with predicted and target distance."""
    batch_dist_preds = batch_dist_preds.detach().float().cpu().reshape(-1)
    batch_dist_labels = batch_dist_labels.detach().float().cpu().reshape(-1)
    batch_size = min(
        batch_obs_images.shape[0],
        batch_goal_images.shape[0],
        batch_dist_preds.shape[0],
        batch_dist_labels.shape[0],
        num_images_preds,
    )
    if batch_size == 0:
        return

    visualize_path = os.path.join(
        save_folder,
        "visualize",
        eval_type,
        f"epoch{epoch}",
        "distance_prediction",
    )
    os.makedirs(visualize_path, exist_ok=True)

    wandb_images = []
    for i in range(batch_size):
        dist_pred = float(batch_dist_preds[i])
        dist_label = float(batch_dist_labels[i])
        dist_error = abs(dist_pred - dist_label)
        title_color = "red" if dist_error > dist_error_threshold else "black"

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(_image_to_hwc(batch_obs_images[i]))
        axes[0].set_title("Observation")
        axes[1].imshow(_image_to_hwc(batch_goal_images[i]))
        axes[1].set_title("Goal")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(
            f"distance prediction={dist_pred:.2f}, label={dist_label:.2f}, "
            f"absolute error={dist_error:.2f}",
            color=title_color,
        )
        fig.set_size_inches(12, 5)

        save_path = os.path.join(visualize_path, f"sample_{i:04d}.png")
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        if use_wandb:
            wandb_images.append(wandb.Image(save_path))

    if wandb_images:
        wandb.log(
            {
                "epoch": epoch,
                f"{eval_type}_distance_predictions": wandb_images,
            },
            commit=False,
        )
