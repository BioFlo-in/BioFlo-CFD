#!/usr/bin/env python3
"""
Audited Training Pipeline for BioFlo-CylindricalNet3D (Pitfall-Free):
Features:
  1. Anisotropic Finite-Difference Curl: Physical (hx, hy, hz) grid spacings.
  2. Potential Scale Normalization: Prevents O(1/h) gradient shock at epoch 0.
  3. Coulomb Gauge Regularization: Pins null-space div(A) = 0.
  4. Continuous Harmonic Polar Coordinates: [r, cos(theta), sin(theta), z].
  5. Parseval Orthonormal Spectral Layers: Isometry guaranteed.
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

from nlss.models.cylindrical_cfd_net_3d import BioFloCylindricalNet3D


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


def thermal_safeguard(threshold_c: float = 78.0, cooldown_sec: float = 10.0):
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


class MultiStepCFDDataset(Dataset):
    def __init__(self, states: torch.Tensor):
        self.states = states
        self.num_sequences = states.shape[0] - 2

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.states[idx + 1], self.states[idx + 2]


class AuditedCylindricalCFDLoss(nn.Module):
    def __init__(self, fluid_mask: torch.Tensor, eps: float = 1e-4, gauge_weight: float = 0.01):
        super().__init__()
        self.register_buffer("fluid_mask", fluid_mask)
        self.eps = eps
        self.gauge_weight = gauge_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        curl_op: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mask = self.fluid_mask.unsqueeze(0).unsqueeze(0)

        pred_vel = pred[:, :3] * mask
        targ_vel = target[:, :3] * mask
        pred_p = pred[:, 3:4] * mask
        targ_p = target[:, 3:4] * mask
        pred_alpha = pred[:, 4:5] * mask
        targ_alpha = target[:, 4:5] * mask

        # Velocity Relative L2
        vel_diff = torch.norm(pred_vel - targ_vel, p=2, dim=(1, 2, 3, 4))
        vel_base = torch.norm(targ_vel, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_vel = torch.mean(vel_diff / vel_base)

        # Pressure Relative L2
        p_diff = torch.norm(pred_p - targ_p, p=2, dim=(1, 2, 3, 4))
        p_base = torch.norm(targ_p, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_p = torch.mean(p_diff / p_base)

        # Alpha Relative L2 + BCE
        alpha_diff = torch.norm(pred_alpha - targ_alpha, p=2, dim=(1, 2, 3, 4))
        alpha_base = torch.norm(targ_alpha, p=2, dim=(1, 2, 3, 4)) + self.eps
        loss_alpha = torch.mean(alpha_diff / alpha_base)

        clamped_alpha = torch.clamp(pred_alpha, 1e-4, 1.0 - 1e-4)
        clamped_targ = torch.clamp(targ_alpha, 0.0, 1.0)
        bce_loss = F.binary_cross_entropy(clamped_alpha[mask.expand_as(clamped_alpha)], clamped_targ[mask.expand_as(clamped_targ)])

        # Dimensionless Coulomb Gauge Constraint: hx^2 * ||div(A)||^2
        loss_gauge = torch.tensor(0.0, device=pred.device)
        if A is not None and curl_op is not None:
            div_A = curl_op.compute_gauge_divergence(A)
            loss_gauge = (curl_op.hx ** 2) * torch.mean(div_A**2)

        total_loss = loss_vel + 0.5 * loss_p + 0.8 * loss_alpha + 0.2 * bce_loss + self.gauge_weight * loss_gauge

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_vel": float(loss_vel.item()),
            "loss_p": float(loss_p.item()),
            "loss_alpha": float(loss_alpha.item()),
            "loss_bce": float(bce_loss.item()),
            "loss_gauge": float(loss_gauge.item()),
        }
        return total_loss, metrics


def train_audited_cylindrical_surrogate(
    dataset_path: str = "data/cstr_3d_dataset_64.pt",
    model_save_path: str = "models/bioflo_cylindrical_net_64_best.pt",
    figure_save_path: str = "outputs/figures/bioflo_cylindrical_net_64_evaluation.png",
    epochs: int = 50,
    lr: float = 8e-4,
    hidden_dim: int = 28,
):
    print("=" * 80)
    print("🚀 BioFlo-CylindricalNet3D: Audited Physics-Equivariant Training Pipeline")
    print("   Pitfalls Resolved: Anisotropic Spacing | Potential Scaling | Coulomb Gauge")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    data = torch.load(dataset_path, weights_only=False)
    states = data["states"]  # (81, 5, D, H, W)
    fluid_mask = data["fluid_mask"].to(device)

    # Enforce strict non-overlapping temporal split:
    # 3-step sequences span [t, t+1, t+2].
    # train_states [:61] -> last window is (58, 59, 60)
    # val_states [64:]   -> first window is (64, 65, 66)
    # Gaps 61, 62, 63 ensure zero data leakage between train and validation.
    train_states = states[:61]
    val_states = states[64:]

    train_ds = MultiStepCFDDataset(train_states)
    val_ds = MultiStepCFDDataset(val_states)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    D, H, W = fluid_mask.shape
    hx = 0.524 / max(1, W - 1)
    hy = 0.524 / max(1, H - 1)
    hz = 1.398 / max(1, D - 1)

    model = BioFloCylindricalNet3D(
        hidden_dim=hidden_dim,
        modes_xy=8,
        modes_z=8,
        num_layers=3,
        hx=hx,
        hy=hy,
        hz=hz,
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🏗️  Trainable Parameters: {params:,}")
    print(f"📏 Grid Dimensions: ({D}x{H}x{W}) | Spacings: hx={hx:.4f}m, hy={hy:.4f}m, hz={hz:.4f}m")

    criterion = AuditedCylindricalCFDLoss(fluid_mask=fluid_mask)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_vel_err": []}
    t0_all = time.time()

    for epoch in range(1, epochs + 1):
        thermal_safeguard(threshold_c=78.0)
        t_ep = time.time()
        model.train()
        train_loss = 0.0

        for s_t0, s_t1, s_t2 in train_loader:
            s_t0 = s_t0.to(device)
            s_t1 = s_t1.to(device)
            s_t2 = s_t2.to(device)

            optimizer.zero_grad()
            # Step 1: Predict t+dt. The curl decoder guarantees div(u)=0.
            # Loss evaluates within fluid_mask; we avoid discontinuous step-masking on velocity
            # to preserve exact divergence-free continuity across the domain.
            p1, A1 = model(s_t0, return_potential=True)
            l1, _ = criterion(p1, s_t1, A1, model.curl_op)

            # Step 2: Autoregressive unroll t+2dt
            p1_phys = torch.cat([p1[:, :4], torch.clamp(p1[:, 4:5], 0.0, 1.0)], dim=1)
            p2, A2 = model(p1_phys, return_potential=True)
            l2, _ = criterion(p2, s_t2, A2, model.curl_op)

            loss = l1 + 0.5 * l2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_vel_err = 0.0
        with torch.no_grad():
            for s_t0, s_t1, s_t2 in val_loader:
                s_t0 = s_t0.to(device)
                s_t1 = s_t1.to(device)
                s_t2 = s_t2.to(device)

                p1, A1 = model(s_t0, return_potential=True)
                l, m = criterion(p1, s_t1, A1, model.curl_op)
                val_loss += l.item()
                val_vel_err += m["loss_vel"]

        val_loss /= len(val_loader)
        val_vel_err = (val_vel_err / len(val_loader)) * 100.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_vel_err"].append(val_vel_err)

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(Path(model_save_path).parent, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_vel_err": val_vel_err,
                },
                model_save_path,
            )
            saved = "★ BEST"

        cur_temp = get_gpu_temperature()
        temp_str = f"{cur_temp:.0f}°C" if cur_temp else "N/A"
        if epoch % 5 == 0 or epoch == 1 or saved:
            scale_val = model.potential_scale.item()
            print(f"Epoch [{epoch:02d}/{epochs:02d}] ({time.time()-t_ep:.1f}s) | Train: {train_loss:.4f} | Val: {val_loss:.4f} (Vel: {val_vel_err:.1f}%) | Scale: {scale_val:.4f} | GPU: {temp_str} {saved}")

    total_time = time.time() - t0_all
    print(f"\n✅ Audited BioFlo-CylindricalNet3D Training completed in {total_time:.1f}s ({total_time/60:.2f} mins)!")
    print(f"   Best Validation Loss: {best_val_loss:.4f} (Saved to {model_save_path})")


if __name__ == "__main__":
    train_audited_cylindrical_surrogate()
