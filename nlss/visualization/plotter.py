"""
CFD visualization and comparison utilities for OpenFOAM vs AI Surrogate rollouts.
Plots side-by-side velocity streamlines, pressure contours, vorticity, and error trajectories.
"""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..physics.losses import compute_vorticity_2d


def plot_field_comparison(
    gt_state: Union[np.ndarray, torch.Tensor],
    pred_state: Union[np.ndarray, torch.Tensor],
    time_val: float,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots side-by-side comparison of Ground Truth, AI Surrogate Prediction, and Difference.
    gt_state, pred_state shape: (3, H, W) where channels are (u, v, p).
    """
    if isinstance(gt_state, torch.Tensor):
        gt = gt_state.detach().cpu().numpy()
    else:
        gt = gt_state

    if isinstance(pred_state, torch.Tensor):
        pred = pred_state.detach().cpu().numpy()
    else:
        pred = pred_state

    # Velocity magnitude
    u_mag_gt = np.sqrt(gt[0] ** 2 + gt[1] ** 2)
    u_mag_pred = np.sqrt(pred[0] ** 2 + pred[1] ** 2)
    u_mag_diff = np.abs(u_mag_pred - u_mag_gt)

    # Pressure
    p_gt = gt[2]
    p_pred = pred[2]
    p_diff = np.abs(p_pred - p_gt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Velocity Magnitude Row
    im0 = axes[0, 0].imshow(u_mag_gt, cmap="viridis", origin="lower")
    axes[0, 0].set_title(f"OpenFOAM / CFD Ground Truth |U| (t={time_val:.3f}s)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(u_mag_pred, cmap="viridis", origin="lower")
    axes[0, 1].set_title(f"AI Surrogate Prediction |U| (t={time_val:.3f}s)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(u_mag_diff, cmap="inferno", origin="lower")
    axes[0, 2].set_title(f"Absolute Error |U_pred - U_gt|")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Pressure Row
    im3 = axes[1, 0].imshow(p_gt, cmap="coolwarm", origin="lower")
    axes[1, 0].set_title(f"CFD Ground Truth Pressure p")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im4 = axes[1, 1].imshow(p_pred, cmap="coolwarm", origin="lower")
    axes[1, 1].set_title(f"AI Surrogate Pressure p")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(p_diff, cmap="inferno", origin="lower")
    axes[1, 2].set_title(f"Absolute Error |p_pred - p_gt|")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        ax.set_xlabel("X (grid)")
        ax.set_ylabel("Y (grid)")

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_rollout_metrics(
    metrics_dict: dict,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
):
    """
    Plots error trajectory over multi-step autoregressive rollout.
    """
    rel_l2 = metrics_dict["relative_l2_trajectory"]
    rmse = metrics_dict["rmse_trajectory"]
    steps = np.arange(len(rel_l2))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    color = "tab:blue"
    ax1.set_xlabel("Rollout Time Steps")
    ax1.set_ylabel("Relative L2 Error", color=color)
    ax1.plot(steps, rel_l2, color=color, linewidth=2, label="Relative L2 Error")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("RMSE", color=color)
    ax2.plot(steps, rmse, color=color, linewidth=2, linestyle="--", label="RMSE")
    ax2.tick_params(axis="y", labelcolor=color)

    speedup = metrics_dict.get("speedup_factor", 1.0)
    plt.title(f"AI Autoregressive Rollout Performance (Speedup: {speedup:.1f}x)")

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_p, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
