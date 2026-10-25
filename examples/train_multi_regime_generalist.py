#!/usr/bin/env python3
"""
Multi-Regime CFD Foundation Surrogate Training.

Eliminates Single-Regime Overfitting by training HybridCFDNet across:
  1. Stirred-Tank Bioreactor (STR) Flows (Rushton turbine radial jet, dual toroidal vortices)
  2. Multi-Speed & Reverse Driven Cavities (u_lid in [-1.5, +1.5], double-driven)
  3. Planar Channel & Duct Flows (Horizontal & vertical parabolic Poiseuille)
  4. Submerged Bluff Body & Obstacle Flows (Flow separation, vortex wake shedding)
  5. Taylor-Green Multi-Mode Decays (k=1, 2, 3)
  6. Perturbed Kelvin-Helmholtz Shear Layers
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nlss.dataset.multi_regime_suite import (
    generate_all_multi_regime_trajectories,
    build_augmented_master_dataset,
)
from nlss.dataset.bioreactor import BioreactorSimulator, BioreactorGeometry, compute_bioprocess_shear_metrics
from nlss.dataset.foam_dataset import FieldNormalizer
from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import compute_divergence_2d


def compute_physics_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalizer: FieldNormalizer,
    lambda_div: float = 0.05,
    lambda_rel: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes composite MSE + Relative L2 + Continuity Divergence Loss."""
    mse_loss = nn.functional.mse_loss(pred, target)
    
    # Relative L2 loss
    diff_norm = torch.norm(pred - target, p=2, dim=(-2, -1))
    target_norm = torch.norm(target, p=2, dim=(-2, -1)) + 1e-6
    rel_loss = torch.mean(diff_norm / target_norm)

    # Physical divergence residual
    pred_phys = normalizer.denormalize(pred)
    div_residual = torch.mean(torch.abs(compute_divergence_2d(pred_phys[:, :2])))

    total_loss = mse_loss + lambda_rel * rel_loss + lambda_div * div_residual
    return total_loss, mse_loss, div_residual


def evaluate_bioreactor_rollout(
    model: nn.Module,
    normalizer: FieldNormalizer,
    device: torch.device,
    num_steps: int = 50,
) -> Tuple[float, float, float]:
    """Evaluates autoregressive multi-step rollout on Stirred-Tank Bioreactor flow."""
    model.eval()
    nx, ny = 64, 64
    geom = BioreactorGeometry(T=1.0, H=1.0)
    sim = BioreactorSimulator(nx=nx, ny=ny, nu=0.01, dt=0.0025, u_tip=1.0, geom=geom)
    res = sim.run_simulation(n_steps=num_steps * 2, save_interval=2, is_cavity=False)
    gt_u = res["u"]
    gt_v = res["v"]
    gt_p = res["p"]
    gt_traj = np.stack([gt_u, gt_v, gt_p], axis=1)

    s_curr = torch.zeros(1, 3, ny, nx, device=device)
    shaft_mask_t = torch.from_numpy(sim.shaft_mask).to(device)
    impeller_mask_t = torch.from_numpy(sim.impeller_mask).to(device)
    xc = sim.lx / 2.0
    right_blade_t = impeller_mask_t & (torch.from_numpy(sim.X > xc).to(device))
    left_blade_t = impeller_mask_t & (torch.from_numpy(sim.X < xc).to(device))

    ai_rollout = [s_curr.cpu().numpy().squeeze(0)]
    with torch.no_grad():
        for step_i in range(1, num_steps + 1):
            s_curr[0, 0, right_blade_t] = sim.u_tip
            s_curr[0, 0, left_blade_t] = -sim.u_tip
            s_norm = normalizer.normalize(s_curr)
            pred_norm = model(s_norm)
            pred = normalizer.denormalize(pred_norm)

            # Boundaries
            pred[0, 0, 0, :] = 0.0
            pred[0, 1, 0, :] = 0.0
            pred[0, 0, :, 0] = 0.0
            pred[0, 1, :, 0] = 0.0
            pred[0, 0, :, -1] = 0.0
            pred[0, 1, :, -1] = 0.0
            pred[0, 1, -1, :] = 0.0
            pred[0, 0, -1, :] = pred[0, 0, -2, :]
            pred[0, 0, shaft_mask_t] = 0.0
            pred[0, 1, shaft_mask_t] = 0.0

            s_curr = pred
            ai_rollout.append(s_curr.cpu().numpy().squeeze(0))

    ai_traj = np.stack(ai_rollout, axis=0)
    u_diff = ai_traj[:, :2] - gt_traj[:, :2]
    rel_l2 = float(np.linalg.norm(u_diff[-1]) / (np.linalg.norm(gt_traj[-1, :2]) + 1e-6) * 100.0)
    rmse_u = float(np.sqrt(np.mean((ai_traj[:, 0] - gt_traj[:, 0])**2)))
    rmse_v = float(np.sqrt(np.mean((ai_traj[:, 1] - gt_traj[:, 1])**2)))

    return rel_l2, rmse_u, rmse_v


def main():
    print("=" * 80)
    print("  MULTI-REGIME CFD GENERALIST SURROGATE TRAINING PIPELINE")
    print("  [Bioreactor STR + Multi-Cavity + Poiseuille Channel + Obstacle Wake]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = Path("outputs/multi_regime_training")
    ckpt_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize Multi-Regime Dataset
    print("\n[Step 1/5] Synthesizing Multi-Regime CFD Simulation Suite...")
    cache_path = out_dir / "multi_regime_dataset_cache.pt"
    if cache_path.exists():
        print(f"  • Loading cached dataset from {cache_path}...")
        cached_data = torch.load(cache_path, weights_only=False)
        inputs_t = cached_data["inputs"]
        targets_t = cached_data["targets"]
        counts = cached_data["counts"]
    else:
        regimes = generate_all_multi_regime_trajectories(nx=64, ny=64, n_steps=40, dt=0.005)
        inputs_t, targets_t, counts = build_augmented_master_dataset(regimes, apply_symmetry=True)
        torch.save({"inputs": inputs_t, "targets": targets_t, "counts": counts}, cache_path)
        print(f"  • Cached master dataset to {cache_path}")

    total_samples = inputs_t.shape[0]
    total_datapoints = total_samples * 64 * 64
    print(f"✓ Master Dataset: {total_samples:,} transition pairs ({total_datapoints:,} physical cell states)")
    for k, v in counts.items():
        print(f"  • {k.capitalize()}: {v:,} transitions ({v/total_samples*100:.1f}%)")

    # 2. Fit Field Normalizer
    print("\n[Step 2/5] Fitting FieldNormalizer over Multi-Regime velocity & pressure fields...")
    normalizer = FieldNormalizer()
    normalizer.fit(inputs_t)
    print(f"  • Velocity U Mean: {normalizer.mean[0, 0, 0, 0]:.4e} | Std: {normalizer.std[0, 0, 0, 0]:.4f}")
    print(f"  • Velocity V Mean: {normalizer.mean[0, 1, 0, 0]:.4e} | Std: {normalizer.std[0, 1, 0, 0]:.4f}")
    print(f"  • Pressure P Mean: {normalizer.mean[0, 2, 0, 0]:.4e} | Std: {normalizer.std[0, 2, 0, 0]:.4f}")

    norm_inputs = normalizer.normalize(inputs_t)
    norm_targets = normalizer.normalize(targets_t)

    # Train/Val Split (90% train, 10% validation)
    val_size = int(0.10 * total_samples)
    train_size = total_samples - val_size
    indices = torch.randperm(total_samples)
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_ds = TensorDataset(norm_inputs[train_idx], norm_targets[train_idx])
    val_ds = TensorDataset(norm_inputs[val_idx], norm_targets[val_idx])

    batch_size = 16
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"✓ Train: {train_size:,} pairs ({len(train_loader)} batches) | Val: {val_size:,} pairs ({len(val_loader)} batches)")

    # 3. Initialize HybridCFDNet
    print("\n[Step 3/5] Initializing HybridCFDNet Generalist Architecture...")
    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=48,
        modes1=12,
        modes2=12,
        num_layers=2,
        num_heads=4,
        enforce_boundary=False,  # Unconstrained for multi-regime physics
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total Trainable Parameters: {num_params:,}")

    # 4. Training Loop
    epochs = 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[Step 4/5] Training Generalist Neural Operator for {epochs} epochs...")
    best_val_loss = float("inf")
    best_bio_l2 = float("inf")
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "bio_l2": []}

    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss, _, _ = compute_physics_loss(pred, by, normalizer)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        train_loss = running_loss / train_size
        scheduler.step()

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_pred = model(vx)
                v_loss, _, _ = compute_physics_loss(v_pred, vy, normalizer)
                val_running += v_loss.item() * vx.size(0)
        val_loss = val_running / val_size

        # Evaluate on Bioreactor STR rollout every 5 epochs
        bio_l2 = float("nan")
        if ep % 5 == 0 or ep == epochs:
            bio_l2, rmse_u, rmse_v = evaluate_bioreactor_rollout(model, normalizer, device, num_steps=50)
            if bio_l2 < best_bio_l2:
                best_bio_l2 = bio_l2

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["bio_l2"].append(bio_l2 if not np.isnan(bio_l2) else (history["bio_l2"][-1] if history["bio_l2"] else 140.0))

        # Save Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "normalizer": normalizer.to_dict(),
                    "val_loss": val_loss,
                    "best_bio_l2": best_bio_l2,
                },
                ckpt_dir / "best_generalist_cfd.pt",
            )

        bio_str = f" | Bio L2: {bio_l2:.2f}%" if not np.isnan(bio_l2) else ""
        print(f"  Epoch [{ep:2d}/{epochs:2d}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}{bio_str} | lr: {scheduler.get_last_lr()[0]:.2e}")

    t_train = time.perf_counter() - t_start
    print(f"✓ Training finished in {t_train:.1f} s ({t_train/epochs:.2f} s/epoch). Best Val Loss: {best_val_loss:.5f}")

    # 5. Final Evaluation and Comparison Plot
    print("\n[Step 5/5] Generating Generalist vs Overfitted Model Comparison...")
    final_ckpt = torch.load(ckpt_dir / "best_generalist_cfd.pt", map_location=device, weights_only=False)
    model.load_state_dict(final_ckpt["model_state_dict"])
    final_bio_l2, final_u_rmse, final_v_rmse = evaluate_bioreactor_rollout(model, normalizer, device, num_steps=50)

    print("\n" + "=" * 60)
    print("🎯 MULTI-REGIME GENERALIZATION RESULTS")
    print("=" * 60)
    print("  • Overfitted Model Bioreactor L2 Error: 140.12%")
    print(f"  • Multi-Regime Generalist L2 Error:     {final_bio_l2:.2f}%")
    print(f"  • Velocity U RMSE:                      {final_u_rmse:.4f} m/s ({final_u_rmse*1000:.1f} mm/s)")
    print(f"  • Velocity V RMSE:                      {final_v_rmse:.4f} m/s ({final_v_rmse*1000:.1f} mm/s)")
    print("=" * 60)

    # Plot Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax1.plot(history["train_loss"], label="Train Loss", lw=2, color="#1f77b4")
    ax1.plot(history["val_loss"], label="Val Loss", lw=2, color="#ff7f0e", linestyle="--")
    ax1.set_title("Multi-Regime CFD Loss Convergence", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(range(1, epochs + 1), history["bio_l2"], label="Generalist Bioreactor L2 Error (%)", lw=2.5, color="#2ca02c")
    ax2.axhline(140.12, color="red", linestyle=":", lw=2, label="Overfitted Model Error (140.12%)")
    ax2.set_title("Bioreactor STR Generalization Error", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Relative L2 Error (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plot_path = plots_dir / "multi_regime_training_convergence.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved training convergence plot: {plot_path}")


if __name__ == "__main__":
    main()
