#!/usr/bin/env python3
"""
High-Fidelity 3D Volumetric Surrogate Training Pipeline (BioFlo-CFD HybridCFDNet3D):
Implements:
  1. 2-Step Autoregressive Unrolled Training (BPTT): Teaches network to self-correct drift.
  2. 3D Vorticity / Enstrophy Conservation Penalty: ||curl(u) - curl(u_gt)||^2 preserves Rushton vortices.
  3. Multiphase Interface Sharpening Loss (BCE + Interface Gradient Matching) for sharp vortex core.
  4. 3D Mass Continuity Penalty: ||div(u)||_1 ~ 0.
  5. Cosine Annealing Learning Rate Schedule with Warmup.
  6. Real-time GPU Thermal Safeguard (< 80°C).
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlss.models.hybrid_cfd_net_3d import HybridCFDNet3D, compute_divergence_3d


# =========================================================================
# GPU THERMAL MONITORING
# =========================================================================

def get_gpu_temperature() -> Optional[float]:
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
    temp = get_gpu_temperature()
    if temp is not None and temp >= threshold_c:
        print(f"\n⚠️  [GPU THERMAL SAFEGUARD] Temperature reached {temp:.1f}°C. Cooling for {cooldown_sec:.1f}s...")
        time.sleep(cooldown_sec)
        while True:
            t = get_gpu_temperature()
            if t is None or t < (threshold_c - 5.0):
                print(f"❄️  [GPU COOLDOWN COMPLETED] Resuming training at {t:.1f}°C.")
                break
            time.sleep(5.0)


# =========================================================================
# PHYSICAL LOSS OPERATORS: 3D VORTICITY & SHARP INTERFACE
# =========================================================================

def compute_vorticity_3d(vel: torch.Tensor, dx: float = 1.0) -> torch.Tensor:
    """
    vel: (B, 3, D, H, W) [ux, uy, uz]
    D corresponds to z (axis 2), H to y (axis 3), W to x (axis 4)
    omega = [duz/dy - duy/dz, dux/dz - duz/dx, duy/dx - dux/dy]
    """
    ux = vel[:, 0:1]
    uy = vel[:, 1:2]
    uz = vel[:, 2:3]

    duz_dy = (torch.roll(uz, -1, dims=3) - torch.roll(uz, 1, dims=3)) / (2 * dx)
    duy_dz = (torch.roll(uy, -1, dims=2) - torch.roll(uy, 1, dims=2)) / (2 * dx)
    omega_x = duz_dy - duy_dz

    dux_dz = (torch.roll(ux, -1, dims=2) - torch.roll(ux, 1, dims=2)) / (2 * dx)
    duz_dx = (torch.roll(uz, -1, dims=4) - torch.roll(uz, 1, dims=4)) / (2 * dx)
    omega_y = dux_dz - duz_dx

    duy_dx = (torch.roll(uy, -1, dims=4) - torch.roll(uy, 1, dims=4)) / (2 * dx)
    dux_dy = (torch.roll(ux, -1, dims=3) - torch.roll(ux, 1, dims=3)) / (2 * dx)
    omega_z = duy_dx - dux_dy

    return torch.cat([omega_x, omega_y, omega_z], dim=1)


class HighFidelityCFDLoss3D(nn.Module):
    """
    High-Fidelity Multiphase 3D Loss:
      - Relative L2 for Velocity & Pressure
      - Interface Sharpening: Binary Cross-Entropy + Interface Normal Gradient Matching
      - 3D Vorticity / Turbulent Eddy Preservation: ||curl(u) - curl(u_gt)||_2
      - 3D Incompressibility Constraint: ||div(u)||_1
    """

    def __init__(
        self,
        eps: float = 1e-4,
        div_weight: float = 0.05,
        vort_weight: float = 0.10,
        interface_weight: float = 0.20,
    ):
        super().__init__()
        self.eps = eps
        self.div_weight = div_weight
        self.vort_weight = vort_weight
        self.interface_weight = interface_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_vel = pred[:, :3]
        targ_vel = target[:, :3]
        pred_p = pred[:, 3:4]
        targ_p = target[:, 3:4]
        pred_alpha = pred[:, 4:5]
        targ_alpha = target[:, 4:5]

        # 1. Velocity Relative L2
        vel_diff = torch.norm(pred_vel - targ_vel, p=2, dim=(1, 2, 3, 4))
        vel_base = torch.norm(targ_vel, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_vel = torch.mean(vel_diff / vel_base)

        # 2. Pressure Relative L2
        p_diff = torch.norm(pred_p - targ_p, p=2, dim=(1, 2, 3, 4))
        p_base = torch.norm(targ_p, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_p = torch.mean(p_diff / p_base)

        # 3. Alpha Relative L2
        alpha_diff = torch.norm(pred_alpha - targ_alpha, p=2, dim=(1, 2, 3, 4))
        alpha_base = torch.norm(targ_alpha, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_alpha = torch.mean(alpha_diff / alpha_base)

        # 4. Multiphase Interface Sharpening (BCE + Gradient Alignment)
        clamped_alpha = torch.clamp(pred_alpha, 1e-4, 1.0 - 1e-4)
        clamped_targ = torch.clamp(targ_alpha, 0.0, 1.0)
        bce_loss = F.binary_cross_entropy(clamped_alpha, clamped_targ)

        # Interface spatial gradient matching
        grad_pred_z = (torch.roll(clamped_alpha, -1, dims=2) - torch.roll(clamped_alpha, 1, dims=2)) * 0.5
        grad_pred_y = (torch.roll(clamped_alpha, -1, dims=3) - torch.roll(clamped_alpha, 1, dims=3)) * 0.5
        grad_pred_x = (torch.roll(clamped_alpha, -1, dims=4) - torch.roll(clamped_alpha, 1, dims=4)) * 0.5

        grad_targ_z = (torch.roll(clamped_targ, -1, dims=2) - torch.roll(clamped_targ, 1, dims=2)) * 0.5
        grad_targ_y = (torch.roll(clamped_targ, -1, dims=3) - torch.roll(clamped_targ, 1, dims=3)) * 0.5
        grad_targ_x = (torch.roll(clamped_targ, -1, dims=4) - torch.roll(clamped_targ, 1, dims=4)) * 0.5

        grad_diff = torch.mean(
            torch.abs(grad_pred_z - grad_targ_z)
            + torch.abs(grad_pred_y - grad_targ_y)
            + torch.abs(grad_pred_x - grad_targ_x)
        )
        loss_interface = bce_loss + grad_diff

        # 5. Vorticity Conservation Loss
        vort_pred = compute_vorticity_3d(pred_vel)
        vort_targ = compute_vorticity_3d(targ_vel)
        vort_diff = torch.norm(vort_pred - vort_targ, p=2, dim=(1, 2, 3, 4))
        vort_base = torch.norm(vort_targ, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_vort = torch.mean(vort_diff / vort_base)

        # 6. Divergence Penalty
        div_pred = compute_divergence_3d(pred_vel)
        loss_div = torch.mean(torch.abs(div_pred))

        total_loss = (
            loss_vel
            + 0.5 * loss_p
            + 0.8 * loss_alpha
            + self.interface_weight * loss_interface
            + self.vort_weight * loss_vort
            + self.div_weight * loss_div
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_p": float(loss_p.item()),
            "loss_alpha": float(loss_alpha.item()),
            "loss_interface": float(loss_interface.item()),
            "loss_vort": float(loss_vort.item()),
            "loss_div": float(loss_div.item()),
        }
        return total_loss, metrics


# =========================================================================
# MULTI-STEP UNROLLED SEQUENCE DATASET
# =========================================================================

class MultiStepCFDDataset(Dataset):
    """
    Yields 3-step sequence tuples: (S_t, S_{t+1}, S_{t+2})
    Enables multi-step unrolled training to penalize compounding errors.
    """

    def __init__(self, states: torch.Tensor):
        # states: (T, C, D, H, W)
        self.states = states
        self.num_sequences = states.shape[0] - 2

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.states[idx + 1], self.states[idx + 2]


# =========================================================================
# TRAINING PIPELINE
# =========================================================================

def train_high_fidelity_surrogate(
    dataset_path: str = "data/cstr_3d_real_openfoam_dataset.pt",
    model_save_path: str = "models/hybrid_cfd_net_3d_cstr_high_fidelity.pt",
    figure_save_path: str = "outputs/figures/trained_cstr_3d_high_fidelity_evaluation.png",
    epochs: int = 70,
    batch_size: int = 2,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    modes: int = 8,
):
    print("=" * 80)
    print("🌟 BioFlo-CFD: High-Fidelity 3D Volumetric Surrogate Training Pipeline")
    print("   Features: 2-Step BPTT Unrolling | Vorticity Loss | Sharp Interface BCE")
    print("=" * 80)

    os.makedirs(Path(model_save_path).parent, exist_ok=True)
    os.makedirs(Path(figure_save_path).parent, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Compute Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        temp = get_gpu_temperature()
        print(f"   GPU: {gpu_name} ({gpu_vram:.2f} GB VRAM) | Temp: {temp}°C")

    # Load continuous states
    data = torch.load(dataset_path, weights_only=False)
    states = data["states"]  # (81, 5, 32, 32, 32)
    T = states.shape[0]

    # Split: Train first 65 snapshots, Validation last 16 snapshots
    n_train_snaps = 65
    train_states = states[:n_train_snaps]
    val_states = states[n_train_snaps - 2:]  # overlapping so first val seq starts at n_train_snaps - 2

    train_ds = MultiStepCFDDataset(train_states)
    val_ds = MultiStepCFDDataset(val_states)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"   Total Snapshots: {T} | Train Sequences: {len(train_ds)} | Val Sequences: {len(val_ds)}")

    # Instantiate model
    model = HybridCFDNet3D(
        in_channels=5,
        out_channels=5,
        hidden_dim=hidden_dim,
        modes=modes,
        num_layers=3,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️  HybridCFDNet3D Parameters: {total_params:,}")

    criterion = HighFidelityCFDLoss3D()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_vel_err": [],
        "val_p_err": [],
        "val_alpha_err": [],
        "val_vort_err": [],
    }

    t_train_start = time.time()

    print(f"\n🚀 Launching {epochs} epochs of multi-step unrolled training...")
    for epoch in range(1, epochs + 1):
        thermal_safeguard(threshold_c=78.0)
        t_ep_start = time.time()

        # Training
        model.train()
        train_loss_accum = 0.0
        n_train_batches = 0

        for s_t0, s_t1, s_t2 in train_loader:
            s_t0 = s_t0.to(device)
            s_t1 = s_t1.to(device)
            s_t2 = s_t2.to(device)

            optimizer.zero_grad()

            # Step 1: S_t0 -> Pred_1
            pred_1 = model(s_t0)
            loss_1, _ = criterion(pred_1, s_t1)

            # Step 2 (Unrolled): Pred_1 -> Pred_2 (BPTT)
            # Physical bounds clamp on alpha
            pred_1_phys = torch.cat([pred_1[:, :4], torch.clamp(pred_1[:, 4:5], 0.0, 1.0)], dim=1)
            pred_2 = model(pred_1_phys)
            loss_2, _ = criterion(pred_2, s_t2)

            # Combined unrolled loss (discount factor gamma = 0.6 on step 2)
            loss = loss_1 + 0.6 * loss_2
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accum += loss.item()
            n_train_batches += 1

        scheduler.step()
        train_loss_mean = train_loss_accum / max(1, n_train_batches)

        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_vel_err_accum = 0.0
        val_p_err_accum = 0.0
        val_alpha_err_accum = 0.0
        val_vort_err_accum = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for s_t0, s_t1, s_t2 in val_loader:
                s_t0 = s_t0.to(device)
                s_t1 = s_t1.to(device)
                s_t2 = s_t2.to(device)

                pred_1 = model(s_t0)
                pred_1_phys = torch.cat([pred_1[:, :4], torch.clamp(pred_1[:, 4:5], 0.0, 1.0)], dim=1)
                pred_2 = model(pred_1_phys)

                loss_1, m1 = criterion(pred_1, s_t1)
                loss_2, m2 = criterion(pred_2, s_t2)
                loss = loss_1 + 0.6 * loss_2

                val_loss_accum += loss.item()
                val_vel_err_accum += m1["loss_vel"]
                val_p_err_accum += m1["loss_p"]
                val_alpha_err_accum += m1["loss_alpha"]
                val_vort_err_accum += m1["loss_vort"]
                n_val_batches += 1

        val_loss_mean = val_loss_accum / max(1, n_val_batches)
        val_vel_err_mean = (val_vel_err_accum / max(1, n_val_batches)) * 100.0
        val_p_err_mean = (val_p_err_accum / max(1, n_val_batches)) * 100.0
        val_alpha_err_mean = (val_alpha_err_accum / max(1, n_val_batches)) * 100.0
        val_vort_err_mean = (val_vort_err_accum / max(1, n_val_batches)) * 100.0

        history["train_loss"].append(train_loss_mean)
        history["val_loss"].append(val_loss_mean)
        history["val_vel_err"].append(val_vel_err_mean)
        history["val_p_err"].append(val_p_err_mean)
        history["val_alpha_err"].append(val_alpha_err_mean)
        history["val_vort_err"].append(val_vort_err_mean)

        cur_temp = get_gpu_temperature()
        ep_time = time.time() - t_ep_start

        if val_loss_mean < best_val_loss:
            best_val_loss = val_loss_mean
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_mean,
                    "val_vel_err": val_vel_err_mean,
                    "val_vort_err": val_vort_err_mean,
                    "config": {
                        "hidden_dim": hidden_dim,
                        "modes": modes,
                        "num_layers": 3,
                    },
                },
                model_save_path,
            )
            saved_indicator = "★ BEST SAVED"
        else:
            saved_indicator = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs or saved_indicator:
            temp_str = f"{cur_temp:.0f}°C" if cur_temp is not None else "N/A"
            lr_cur = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
                f"Train: {train_loss_mean:.4f} | Val: {val_loss_mean:.4f} "
                f"(Vel: {val_vel_err_mean:.1f}%, Vort: {val_vort_err_mean:.1f}%, Alpha: {val_alpha_err_mean:.1f}%) | "
                f"LR: {lr_cur:.1e} | GPU: {temp_str} {saved_indicator}"
            )

    total_training_time = time.time() - t_train_start
    print(f"\n✅ High-Fidelity Training completed in {total_training_time:.1f}s ({total_training_time/60:.2f} mins)!")
    print(f"   Best Validation Loss: {best_val_loss:.4f} (Saved to {model_save_path})")

    # Autoregressive multi-step rollout evaluation on held-out validation window
    print("\n🔍 Conducting 16-Step Autoregressive Rollout Evaluation...")
    best_ckpt = torch.load(model_save_path, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_trajectory = states[65:].to(device)  # 16 snapshots
    n_rollout_steps = val_trajectory.shape[0] - 1

    current_state = val_trajectory[0:1]  # (1, 5, D, H, W)
    rollout_errors = []

    with torch.no_grad():
        for step in range(n_rollout_steps):
            pred_next = model(current_state)
            pred_next[:, 4:5] = torch.clamp(pred_next[:, 4:5], 0.0, 1.0)

            targ_next = val_trajectory[step + 1 : step + 2]
            err = (
                torch.norm(pred_next[:, :3] - targ_next[:, :3])
                / (torch.norm(targ_next[:, :3]) + 1e-4)
            ).item() * 100.0
            rollout_errors.append(err)
            current_state = pred_next

    print(f"   1-Step Relative Velocity Error: {rollout_errors[0]:.2f}%")
    print(f"   16-Step Accumulated Rollout Error: {rollout_errors[-1]:.2f}%")
    print(f"   Mean Rollout Error over trajectory: {np.mean(rollout_errors):.2f}%")

    # Plot evaluation and loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)

    # Panel A: Loss & Metric Convergence
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train Loss (2-Step BPTT)", color="#1f77b4", lw=2)
    ax.plot(history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax.set_title("Training Loss Convergence with Cosine Annealing", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    # Panel B: Autoregressive Rollout Stability
    ax = axes[1]
    steps = np.arange(1, len(rollout_errors) + 1)
    ax.plot(steps, rollout_errors, marker="o", color="#d62728", lw=2.5, label="High-Fidelity Surrogate (2-Step BPTT)")
    ax.axhline(np.mean(rollout_errors), color="#2ca02c", linestyle="--", lw=2, label=f"Mean Error: {np.mean(rollout_errors):.1f}%")
    ax.set_xlabel("Autoregressive Rollout Step", fontsize=11, fontweight="bold")
    ax.set_ylabel("Velocity Relative L2 Error (%)", fontsize=11, fontweight="bold")
    ax.set_title("16-Step Autonomous Autoregressive Rollout Stability", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(figure_save_path, dpi=200)
    plt.close()
    print(f"✅ Published high-fidelity evaluation figure to: {figure_save_path}")


if __name__ == "__main__":
    train_high_fidelity_surrogate()
