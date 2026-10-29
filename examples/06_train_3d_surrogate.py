#!/usr/bin/env python3
"""
3D Volumetric Neural Operator Training Pipeline (BioFlo-CFD HybridCFDNet3D):
Trains full 3D volumetric surrogate on unified 3D CFD dataset:
  - 3D CSTR Bioreactor suite (Rushton turbine sweeps, 4 baffles, dual toroidal loops)
  - OpenFOAM 3D Tutorial cases (TJunction flow split, rotatingCylinders)
  - 3D non-linear Navier-Stokes Taylor-Green and Beltrami helical regimes

Loss:
  Scale-Invariant Relative L2 Loss (Velocity + Pressure) + 3D Mass Continuity Penalty (div_3D ~ 0)
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

from nlss.models.hybrid_cfd_net_3d import HybridCFDNet3D, ScaleInvariantRelativeLoss3D


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


def enforce_gpu_thermal_safety(max_temp: float = 80.0, cooldown_target: float = 72.0):
    """Enforces GPU thermal ceiling; sleeps if temperature exceeds max_temp."""
    temp = get_gpu_temperature()
    if temp is not None and temp >= max_temp:
        print(f"\n🔥 [THERMAL WARNING] GPU temp reached {temp:.1f}°C (Limit: {max_temp:.1f}°C).")
        print(f"   Pausing 3D training to protect hardware until cooled below {cooldown_target:.1f}°C...")
        while temp is not None and temp > cooldown_target:
            time.sleep(4.0)
            temp = get_gpu_temperature()
        print(f"❄️  [COOLDOWN COMPLETE] GPU cooled down to {temp:.1f}°C. Resuming training.\n")


# =========================================================================
# TRAINING ENGINE
# =========================================================================

def train_3d_surrogate(
    dataset_path: Path,
    output_dir: Path,
    preset: str = "small",
    epochs: int = 150,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    max_temp: float = 80.0,
    device: Optional[torch.device] = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "hybrid_cfd_net_3d_best.pt"
    latest_ckpt_path = output_dir / "hybrid_cfd_net_3d_latest.pt"

    print("=" * 80)
    print("      BioFlo-CFD: FULL VOLUMETRIC 3D NEURAL OPERATOR TRAINING")
    print("=" * 80)
    print(f"• Hardware Device:               {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"• Thermal Ceiling:               {max_temp:.1f}°C")
    print(f"• Dataset Source:                {dataset_path}")
    print(f"• Model Preset:                  '{preset}'")
    print(f"• Target Epochs:                 {epochs}")
    print(f"• Batch Size:                    {batch_size}")
    print(f"• Base Learning Rate:            {learning_rate}")

    # 1. Load Dataset
    if not dataset_path.exists():
        print(f"Error: Dataset {dataset_path} does not exist. Run scripts/build_3d_cfd_dataset.py first.")
        sys.exit(1)

    data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    mean_3d = data["mean_3d"].to(device)
    std_3d = data["std_3d"].to(device)

    print(f"✓ Loaded 3D Volumetric Dataset: {x_train.shape[0]} train pairs, {x_val.shape[0]} val pairs ({data['grid_dim']}³ grid)")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False
    )

    # 2. Initialize Model
    model = HybridCFDNet3D.from_preset(preset, in_channels=4, out_channels=4).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ HybridCFDNet3D initialized with {param_count:,} trainable parameters.")

    # 3. Loss & Optimizer
    criterion = ScaleInvariantRelativeLoss3D(eps=1e-4, div_weight=0.05).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5
    )

    # Resume from checkpoint if present
    start_epoch = 1
    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_vel_loss": [], "val_p_loss": [], "val_div_loss": []
    }

    if latest_ckpt_path.exists():
        print(f"📦 Resuming from checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if "history" in ckpt:
            history = ckpt["history"]
        print(f"✓ Resumed from Epoch {ckpt['epoch']} (Best Val Loss: {best_val_loss:.4f})")

    if start_epoch > epochs:
        print(f"Checkpoint already at epoch {start_epoch-1} >= target {epochs}. Exiting.")
        return

    # Normalization helper
    def norm_fn(t: torch.Tensor) -> torch.Tensor:
        return (t - mean_3d) / std_3d

    def denorm_fn(t: torch.Tensor) -> torch.Tensor:
        return t * std_3d + mean_3d

    # 4. Training Loop
    t_start = time.perf_counter()

    for epoch in range(start_epoch, epochs + 1):
        # Thermal check
        if epoch % 5 == 0:
            enforce_gpu_thermal_safety(max_temp=max_temp)

        model.train()
        train_losses = []

        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            # Online symmetry augmentation (reflection along X and Y axes)
            if torch.rand(1).item() > 0.5:
                x_b = torch.flip(x_b, dims=[-1])
                y_b = torch.flip(y_b, dims=[-1])
                x_b[:, 0] = -x_b[:, 0]
                y_b[:, 0] = -y_b[:, 0]

            optimizer.zero_grad()
            x_norm = norm_fn(x_b)
            pred_norm = model(x_norm)
            pred_phys = denorm_fn(pred_norm)

            loss, _ = criterion(pred_phys, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        epoch_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        val_vel = []
        val_p = []
        val_div = []

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b = x_b.to(device)
                y_b = y_b.to(device)

                x_norm = norm_fn(x_b)
                pred_norm = model(x_norm)
                pred_phys = denorm_fn(pred_norm)

                loss, metrics = criterion(pred_phys, y_b)
                val_losses.append(metrics["loss_total"])
                val_vel.append(metrics["loss_vel"])
                val_p.append(metrics["loss_p"])
                val_div.append(metrics["loss_div"])

        epoch_val_loss = float(np.mean(val_losses))
        epoch_val_vel = float(np.mean(val_vel))
        epoch_val_p = float(np.mean(val_p))
        epoch_val_div = float(np.mean(val_div))

        scheduler.step(epoch_val_loss)
        lr_curr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_vel_loss"].append(epoch_val_vel)
        history["val_p_loss"].append(epoch_val_p)
        history["val_div_loss"].append(epoch_val_div)

        is_best = epoch_val_loss < best_val_loss
        if is_best:
            best_val_loss = epoch_val_loss

        # Periodic logging
        if epoch % 5 == 0 or epoch == start_epoch or is_best:
            temp_str = f" | GPU: {get_gpu_temperature():.0f}°C" if device.type == "cuda" else ""
            best_tag = " \033[1;32m[★ BEST]\033[0m" if is_best else ""
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Train: {epoch_train_loss:.4f} | "
                f"Val: {epoch_val_loss:.4f} (Vel: {epoch_val_vel:.4f}, P: {epoch_val_p:.4f}, Div: {epoch_val_div:.4f}) | "
                f"LR: {lr_curr:.1e}{temp_str}{best_tag}"
            )

        # Save checkpoint dictionary
        ckpt_dict = {
            "epoch": epoch,
            "preset": preset,
            "num_parameters": param_count,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": epoch_val_loss,
            "mean_3d": mean_3d.cpu(),
            "std_3d": std_3d.cpu(),
            "history": history,
        }

        torch.save(ckpt_dict, latest_ckpt_path)
        if is_best:
            torch.save(ckpt_dict, best_ckpt_path)

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print("                    3D TRAINING COMPLETED")
    print("=" * 80)
    print(f"• Total Duration:        {total_time:.2f} s ({(total_time / (epochs - start_epoch + 1)):.2f} s/epoch)")
    print(f"• Best Validation Loss:  \033[1;32m{best_val_loss:.4f}\033[0m")
    print(f"• Best Checkpoint:       {best_ckpt_path}")
    print(f"• Latest Checkpoint:     {latest_ckpt_path}")

    # Plot 3D Loss History
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=200)

    ep_range = list(range(1, len(history["train_loss"]) + 1))
    ax1.plot(ep_range, history["train_loss"], label="Train Loss (Rel L2 + Div)", color="navy", lw=2.0)
    ax1.plot(ep_range, history["val_loss"], label="Val Loss", color="crimson", lw=2.0, linestyle="--")
    ax1.set_title("HybridCFDNet3D Convergence History", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Composite Relative Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.plot(ep_range, history["val_vel_loss"], label="Velocity Rel L2", color="darkgreen", lw=1.8)
    ax2.plot(ep_range, history["val_p_loss"], label="Pressure Rel L2", color="purple", lw=1.8)
    ax2.plot(ep_range, history["val_div_loss"], label="Mass Div Penalty ||div||", color="darkorange", lw=1.8)
    ax2.set_title("Multi-Channel Error Decomposition", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Loss Component", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    loss_fig = output_dir / "training_loss_history_3d.png"
    plt.savefig(loss_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved 3D Training Loss History to: {loss_fig}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train 3D Volumetric Neural Operator for CFD")
    parser.add_argument("--dataset", type=str, default="data/cfd_3d_dataset.pt", help="Path to 3D dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints_3d_openfoam", help="Output directory for checkpoints")
    parser.add_argument("--preset", type=str, default="small", choices=["small", "medium", "large"], help="Model preset")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--max-temp", type=float, default=80.0, help="Maximum allowable GPU temperature in C")

    args = parser.parse_args()

    train_3d_surrogate(
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        preset=args.preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_temp=args.max_temp,
    )


if __name__ == "__main__":
    main()
