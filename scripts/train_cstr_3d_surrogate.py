#!/usr/bin/env python3
"""
3D Volumetric Neural Operator Training Pipeline (BioFlo-CFD HybridCFDNet3D):
Trains the full 3D volumetric surrogate on the 59-snapshot OpenFOAM CSTR multiphase dataset.
Channels: [ux, uy, uz, p, alpha.water] on (32 x 32 x 32) grid.

Loss:
  Multiphase Scale-Invariant Relative L2 Loss (Velocity + Pressure + Alpha) + 3D Mass Continuity Penalty (div_3D ~ 0)
Hardware Safeguard:
  Monitors NVIDIA GPU thermal sensor; pauses if temperature exceeds 80°C until safe.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure local package path is available
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlss.models.hybrid_cfd_net_3d import HybridCFDNet3D, compute_divergence_3d


# =========================================================================
# GPU THERMAL MONITORING
# =========================================================================

def get_gpu_temperature() -> Optional[float]:
    """Queries current NVIDIA GPU core temperature via nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
        return float(res.stdout.strip().split()[0])
    except Exception:
        return None


def thermal_safeguard(threshold_c: float = 80.0, cooldown_sec: float = 10.0):
    """Pauses execution if GPU temperature exceeds safety threshold."""
    temp = get_gpu_temperature()
    if temp is not None and temp >= threshold_c:
        print(f"\n⚠️  [GPU THERMAL SAFEGUARD] Temperature reached {temp:.1f}°C (Threshold: {threshold_c:.1f}°C). Cooling down for {cooldown_sec:.1f}s...")
        time.sleep(cooldown_sec)
        while True:
            t = get_gpu_temperature()
            if t is None or t < (threshold_c - 5.0):
                print(f"❄️  [GPU COOLDOWN COMPLETED] Resuming training at {t:.1f}°C.")
                break
            time.sleep(5.0)


# =========================================================================
# MULTIPHASE 3D LOSS FUNCTION
# =========================================================================

class MultiphaseScaleInvariantLoss3D(nn.Module):
    """
    Scale-invariant loss for multiphase 3D CFD [ux, uy, uz, p, alpha]:
      L = L_rel(u) + 0.5 * L_rel(p) + 1.0 * L_rel(alpha) + lambda_div * ||div(u)||_1
    """

    def __init__(self, eps: float = 1e-4, div_weight: float = 0.05):
        super().__init__()
        self.eps = eps
        self.div_weight = div_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # pred, target: (B, 5, D, H, W)
        pred_vel = pred[:, :3]
        targ_vel = target[:, :3]
        pred_p = pred[:, 3:4]
        targ_p = target[:, 3:4]
        pred_alpha = pred[:, 4:5]
        targ_alpha = target[:, 4:5]

        # Velocity Relative L2
        vel_diff = torch.norm(pred_vel - targ_vel, p=2, dim=(1, 2, 3, 4))
        vel_base = torch.norm(targ_vel, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_vel = torch.mean(vel_diff / vel_base)

        # Pressure Relative L2
        p_diff = torch.norm(pred_p - targ_p, p=2, dim=(1, 2, 3, 4))
        p_base = torch.norm(targ_p, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_p = torch.mean(p_diff / p_base)

        # Alpha Relative L2 (and clamp between 0 and 1)
        alpha_diff = torch.norm(pred_alpha - targ_alpha, p=2, dim=(1, 2, 3, 4))
        alpha_base = torch.norm(targ_alpha, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_alpha = torch.mean(alpha_diff / alpha_base)

        # 3D Incompressibility constraint div(u) ~ 0
        div_pred = compute_divergence_3d(pred_vel)
        loss_div = torch.mean(torch.abs(div_pred))

        total_loss = loss_vel + 0.5 * loss_p + 1.0 * loss_alpha + self.div_weight * loss_div

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_p": float(loss_p.item()),
            "loss_alpha": float(loss_alpha.item()),
            "loss_div": float(loss_div.item()),
        }
        return total_loss, metrics


# =========================================================================
# MAIN TRAINING PIPELINE
# =========================================================================

def train_cstr_3d_surrogate(
    dataset_path: str = "data/cstr_3d_real_openfoam_dataset.pt",
    model_save_path: str = "models/hybrid_cfd_net_3d_cstr_best.pt",
    figure_save_path: str = "outputs/figures/trained_cstr_3d_surrogate_evaluation.png",
    epochs: int = 60,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    modes: int = 8,
    num_layers: int = 3,
):
    print("=" * 75)
    print("🧠 BioFlo-CFD: Training 3D Volumetric Surrogate on Real OpenFOAM CSTR Dataset")
    print("=" * 75)

    os.makedirs(Path(model_save_path).parent, exist_ok=True)
    os.makedirs(Path(figure_save_path).parent, exist_ok=True)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Compute Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        temp = get_gpu_temperature()
        print(f"   GPU: {gpu_name} ({gpu_vram:.2f} GB VRAM) | Temp: {temp}°C")

    # 2. Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"📂 Loading dataset from {dataset_path}...")
    data = torch.load(dataset_path, weights_only=False)
    inputs = data["input_pairs"]    # (N, 5, D, H, W)
    targets = data["target_pairs"]  # (N, 5, D, H, W)
    times = data["times"]
    channels = data["channels"]
    grid_dim = data["grid_dim"]
    n_samples = inputs.shape[0]

    print(f"   Loaded {n_samples} state-transition pairs across {len(times)} snapshots")
    print(f"   Grid Resolution: {grid_dim}³ = {grid_dim**3:,} voxels")
    print(f"   Channels: {channels}")

    # 3. Train / Validation split (80% train, 20% validation temporally)
    n_train = int(n_samples * 0.8)
    n_val = n_samples - n_train
    print(f"   Split: {n_train} Train pairs (t = {times[0]:.2f}s -> {times[n_train]:.2f}s) | {n_val} Val pairs (t = {times[n_train]:.2f}s -> {times[-1]:.2f}s)")

    train_inputs, val_inputs = inputs[:n_train], inputs[n_train:]
    train_targets, val_targets = targets[:n_train], targets[n_train:]

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_inputs, val_targets), batch_size=batch_size, shuffle=False)

    # 4. Model instantiation
    print("\n🏗️  Instantiating HybridCFDNet3D model...")
    model = HybridCFDNet3D(
        in_channels=5,
        out_channels=5,
        hidden_dim=hidden_dim,
        modes=modes,
        num_layers=num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable Parameters: {n_params:,}")

    # 5. Optimizer, Scheduler, Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = MultiphaseScaleInvariantLoss3D(eps=1e-4, div_weight=0.05).to(device)

    # 6. Training loop
    history = {"train_loss": [], "val_loss": [], "val_vel": [], "val_p": [], "val_alpha": [], "val_div": []}
    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\n🚀 Beginning training for {epochs} epochs (Batch size: {batch_size})...\n")

    for epoch in range(1, epochs + 1):
        thermal_safeguard(threshold_c=80.0, cooldown_sec=10.0)

        model.train()
        train_loss_epoch = 0.0
        n_batches = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss, metrics = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_epoch += metrics["loss_total"]
            n_batches += 1

        train_loss_epoch /= max(1, n_batches)
        scheduler.step()

        # Validation phase
        model.eval()
        val_metrics = {"loss_total": 0.0, "loss_vel": 0.0, "loss_p": 0.0, "loss_alpha": 0.0, "loss_div": 0.0}
        n_val_batches = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                _, metrics = criterion(pred, by)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]
                n_val_batches += 1

        for k in val_metrics:
            val_metrics[k] /= max(1, n_val_batches)

        history["train_loss"].append(train_loss_epoch)
        history["val_loss"].append(val_metrics["loss_total"])
        history["val_vel"].append(val_metrics["loss_vel"])
        history["val_p"].append(val_metrics["loss_p"])
        history["val_alpha"].append(val_metrics["loss_alpha"])
        history["val_div"].append(val_metrics["loss_div"])

        # Checkpoint best model
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "grid_dim": grid_dim,
                    "channels": channels,
                },
                model_save_path,
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            temp = get_gpu_temperature()
            temp_str = f"{temp:.0f}°C" if temp else "N/A"
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss_epoch:.4f} | "
                f"Val Loss: {val_metrics['loss_total']:.4f} (Vel: {val_metrics['loss_vel']*100:.1f}%, P: {val_metrics['loss_p']*100:.1f}%, Alpha: {val_metrics['loss_alpha']*100:.1f}%) | "
                f"GPU: {temp_str}"
            )

    total_training_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins)!")
    print(f"   Best Validation Loss: {best_val_loss:.4f} (Saved to {model_save_path})")

    # =========================================================================
    # MULTI-STEP AUTOREGRESSIVE ROLLOUT EVALUATION
    # =========================================================================
    print("\n🔍 Conducting Multi-Step Autoregressive Rollout Evaluation...")
    model.eval()

    # Load best checkpoint
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Rollout across validation steps
    current_state = val_inputs[0:1].to(device)  # initial condition (1, 5, D, H, W)
    rollout_preds = []
    rollout_targets = []
    rollout_errors = []

    with torch.no_grad():
        for step in range(len(val_inputs)):
            target_step = val_targets[step:step+1].to(device)
            next_state = model(current_state)

            # Clamp phase fraction between 0 and 1
            next_state[:, 4:5] = torch.clamp(next_state[:, 4:5], 0.0, 1.0)

            # Measure relative L2 error on velocity
            vel_err = float(torch.norm(next_state[:, :3] - target_step[:, :3]) / (torch.norm(target_step[:, :3]) + 1e-4))
            rollout_errors.append(vel_err)

            rollout_preds.append(next_state.cpu().numpy().squeeze(0))
            rollout_targets.append(target_step.cpu().numpy().squeeze(0))

            # Autoregressive feedback: feed prediction as input to next step!
            current_state = next_state

    print(f"   1-Step Relative Velocity Error: {rollout_errors[0]*100:.2f}%")
    print(f"   {len(rollout_errors)}-Step Accumulated Rollout Error: {rollout_errors[-1]*100:.2f}%")
    print(f"   Mean Rollout Error over trajectory: {np.mean(rollout_errors)*100:.2f}%")

    # =========================================================================
    # PUBLICATION DIAGNOSTIC FIGURE
    # =========================================================================
    print(f"\n🎨 Generating evaluation figure: {figure_save_path}...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    # Panel A: Loss Curves
    axes[0, 0].plot(range(1, epochs + 1), history["train_loss"], label="Train Loss (Total)", color="#1f77b4", lw=2)
    axes[0, 0].plot(range(1, epochs + 1), history["val_loss"], label="Val Loss (Total)", color="#d62728", lw=2, linestyle="--")
    axes[0, 0].set_title("(A) 3D Neural Surrogate Convergence History", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=10)
    axes[0, 0].set_ylabel("Scale-Invariant Relative Loss", fontsize=10)
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # Panel B: Autoregressive Rollout Error
    rollout_steps = np.arange(1, len(rollout_errors) + 1)
    rollout_times = [times[n_train + i] for i in range(len(rollout_errors))]
    axes[0, 1].plot(rollout_times, [e * 100 for e in rollout_errors], marker="o", color="#2ca02c", lw=2, label="Surrogate Rollout Error (%)")
    axes[0, 1].axhline(y=np.mean(rollout_errors)*100, color="#7f7f7f", linestyle=":", label=f"Mean Error ({np.mean(rollout_errors)*100:.1f}%)")
    axes[0, 1].set_title("(B) Autoregressive Multi-Step Rollout Stability", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Physical Time t (s)", fontsize=10)
    axes[0, 1].set_ylabel("Relative Velocity L2 Error (%)", fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    # Mid-plane slice index (axial plane through impeller)
    slice_idx = grid_dim // 2
    eval_step = min(5, len(rollout_preds) - 1)
    gt_eval = rollout_targets[eval_step]    # (5, D, H, W)
    pr_eval = rollout_preds[eval_step]      # (5, D, H, W)

    # Panel C: Velocity magnitude |U| comparison at slice Y = mid
    gt_vel_mag = np.linalg.norm(gt_eval[:3, :, slice_idx, :], axis=0)  # (D, W)
    pr_vel_mag = np.linalg.norm(pr_eval[:3, :, slice_idx, :], axis=0)  # (D, W)
    vmin_u, vmax_u = 0.0, max(gt_vel_mag.max(), pr_vel_mag.max())

    comp_u = np.concatenate([gt_vel_mag, pr_vel_mag], axis=1)  # side by side
    im_c = axes[1, 0].imshow(comp_u.T, origin="lower", cmap="turbo", vmin=vmin_u, vmax=vmax_u)
    axes[1, 0].set_title(f"(C) 3D Velocity Magnitude |U| at t = {rollout_times[eval_step]:.2f}s\n[Left: OpenFOAM Ground Truth | Right: HybridCFDNet3D]", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Z-Axis (Vertical Domain) [Left: CFD, Right: AI]", fontsize=9)
    axes[1, 0].set_ylabel("X-Axis (Radial Span)", fontsize=9)
    fig.colorbar(im_c, ax=axes[1, 0], fraction=0.046, pad=0.04, label="|U| (m/s)")

    # Panel D: Multiphase Free Surface (alpha.water) comparison
    gt_alpha = gt_eval[4, :, slice_idx, :]  # (D, W)
    pr_alpha = pr_eval[4, :, slice_idx, :]  # (D, W)
    comp_alpha = np.concatenate([gt_alpha, pr_alpha], axis=1)

    im_d = axes[1, 1].imshow(comp_alpha.T, origin="lower", cmap="Blues_r", vmin=0.0, vmax=1.0)
    axes[1, 1].set_title(f"(D) Multiphase Volume Fraction alpha.water (Vortex Funnel)\n[Left: OpenFOAM Ground Truth | Right: HybridCFDNet3D]", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Z-Axis (Vertical Domain) [Left: CFD, Right: AI]", fontsize=9)
    axes[1, 1].set_ylabel("X-Axis (Radial Span)", fontsize=9)
    fig.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04, label="alpha.water (1=Liquid, 0=Air)")

    plt.tight_layout()
    plt.savefig(figure_save_path, bbox_inches="tight")
    plt.close()

    print(f"✅ Published evaluation figure to: {figure_save_path}")
    print("=" * 75)


if __name__ == "__main__":
    train_cstr_3d_surrogate()
