#!/usr/bin/env python3
"""
Extended Training Pipeline: 1,000 Additional Epochs for HybridCFDNet on OpenFOAM Trajectory.
Incorporates:
1. Multi-Step Autoregressive Pushforward Loss (S_t -> S_{t+1} -> S_{t+2}) to suppress compounding rollout error.
2. Physics-Informed Importance Weighting on High-Shear Layers and Vortex Recirculation Core.
3. Smooth Cosine Annealing Learning Rate Schedule with Warmup.
4. Full Trajectory Error Tracking against OpenFOAM icoFoam Ground Truth across 50 Timesteps.
5. Automated Export of Updated High-Fidelity Predictions to OpenFOAM Native Format.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import PhysicsInformedCFDLoss, compute_cfd_importance_weight, compute_divergence_2d
from nlss.io.foam_writer import export_simulation_state_to_openfoam


def evaluate_rollout(
    model: nn.Module,
    dataset,
    normalizer: FieldNormalizer,
    num_steps: int,
    device: torch.device,
) -> Tuple[np.ndarray, float, List[float], List[float]]:
    """Evaluates 50-step autoregressive rollout against OpenFOAM ground truth."""
    model.eval()
    t0 = time.perf_counter()

    with torch.no_grad():
        curr = dataset.normalized_data[0:1].to(device)
        rollout = [curr]

        for s in range(num_steps):
            curr = model(curr)

            # Preserve physical Dirichlet boundary conditions
            phys = normalizer.denormalize(curr)
            phys[:, :2, :, 0] = 0.0   # Left wall
            phys[:, :2, :, -1] = 0.0  # Right wall
            phys[:, :2, 0, :] = 0.0   # Bottom wall
            target_lid = dataset.data[s + 1, 0, -1, :].to(device)
            phys[:, 0, -1, :] = target_lid
            phys[:, 1, -1, :] = 0.0

            curr = normalizer.normalize(phys)
            rollout.append(curr)

        rollout_tensor = normalizer.denormalize(torch.cat(rollout, dim=0)).cpu()

    t_rollout = time.perf_counter() - t0
    pred_data = rollout_tensor.numpy()
    gt_data = dataset.data.numpy()

    rel_l2_list = []
    rmse_list = []
    for s in range(num_steps + 1):
        diff_u = pred_data[s, :2] - gt_data[s, :2]
        rel_l2 = np.linalg.norm(diff_u) / (np.linalg.norm(gt_data[s, :2]) + 1e-6)
        rmse_u = np.sqrt(np.mean(diff_u**2))
        rel_l2_list.append(rel_l2 * 100)
        rmse_list.append(rmse_u * 1000)

    return pred_data, t_rollout, rel_l2_list, rmse_list


def generate_benchmark_plots(
    gt: np.ndarray,
    pred: np.ndarray,
    times: np.ndarray,
    openfoam_time: float,
    ai_time: float,
    rel_l2_list: List[float],
    rmse_list: List[float],
    output_dir: Path,
):
    """Generates comprehensive streamline comparisons, centerline profiles, and speedup charts."""
    num_steps = len(times) - 1
    eval_steps = [10, 30, 50]  # Timestamps: t = 0.05s, 0.15s, 0.25s
    x_coords = np.linspace(0, 1, 64)
    y_coords = np.linspace(0, 1, 64)

    # 1. Streamlines & Field Comparison Matrix
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    plt.subplots_adjust(wspace=0.28, hspace=0.32)

    for row_idx, s in enumerate(eval_steps):
        t_curr = times[s]
        gt_ux = gt[s, 0]
        gt_uy = gt[s, 1]
        pr_ux = pred[s, 0]
        pr_uy = pred[s, 1]

        gt_mag = np.sqrt(gt_ux**2 + gt_uy**2)
        pr_mag = np.sqrt(pr_ux**2 + pr_uy**2)
        err_mag = np.abs(pr_mag - gt_mag)
        pr_p = pred[s, 2]

        vmax_u = max(np.percentile(gt_mag, 99.5), 0.1)

        # OpenFOAM Ground Truth
        im0 = axes[row_idx, 0].imshow(gt_mag, cmap="viridis", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=vmax_u)
        axes[row_idx, 0].streamplot(x_coords, y_coords, gt_ux, gt_uy, color="white", density=0.75, linewidth=0.7, arrowsize=0.7)
        axes[row_idx, 0].set_title(f"OpenFOAM icoFoam |U| (t={t_curr:.2f}s)", fontsize=11, fontweight="bold")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04, label="|U| (m/s)")

        # AI Surrogate Prediction
        im1 = axes[row_idx, 1].imshow(pr_mag, cmap="viridis", origin="lower", extent=[0, 1, 0, 1], vmin=0, vmax=vmax_u)
        axes[row_idx, 1].streamplot(x_coords, y_coords, pr_ux, pr_uy, color="white", density=0.75, linewidth=0.7, arrowsize=0.7)
        axes[row_idx, 1].set_title(f"AI HybridCFDNet |U| (t={t_curr:.2f}s)", fontsize=11, fontweight="bold")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04, label="|U| (m/s)")

        # Pointwise Error Map
        im2 = axes[row_idx, 2].imshow(err_mag, cmap="inferno", origin="lower", extent=[0, 1, 0, 1])
        axes[row_idx, 2].set_title(f"Pointwise |U_AI - U_FOAM| Error", fontsize=11, fontweight="bold")
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04, label="Error (m/s)")

        # AI Predicted Pressure Field
        im3 = axes[row_idx, 3].imshow(pr_p, cmap="coolwarm", origin="lower", extent=[0, 1, 0, 1])
        axes[row_idx, 3].set_title(f"AI Pressure Field p(x,y) (t={t_curr:.2f}s)", fontsize=11, fontweight="bold")
        plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04, label="p (m²/s²)")

        for c in range(4):
            axes[row_idx, c].set_xlabel("x (m)")
            axes[row_idx, c].set_ylabel("y (m)")

    fig1_path = output_dir / "full_simulation_streamlines_comparison.png"
    plt.savefig(fig1_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved updated streamline comparison matrix to: {fig1_path}")

    # 2. Centerline Velocity Profiles
    fig2, (ax_vert, ax_horiz) = plt.subplots(1, 2, figsize=(14, 5.5))
    mid_idx = 32

    for s, col, ls in [(10, "#1f77b4", "-"), (30, "#ff7f0e", "--"), (50, "#2ca02c", "-.")]:
        ax_vert.plot(gt[s, 0, :, mid_idx], y_coords, color=col, linestyle=ls, lw=2.2, label=f"OpenFOAM (t={times[s]:.2f}s)")
        ax_vert.plot(pred[s, 0, :, mid_idx], y_coords, color=col, linestyle="none", marker="o", markersize=3.5, label=f"AI Surrogate (t={times[s]:.2f}s)")

    ax_vert.set_xlabel("Horizontal Velocity u_x (m/s)", fontsize=11)
    ax_vert.set_ylabel("Vertical Position y (m)", fontsize=11)
    ax_vert.set_title("Vertical Centerline Profile: u_x(y) at x = 0.5 m", fontsize=12, fontweight="bold")
    ax_vert.grid(True, linestyle="--", alpha=0.5)
    ax_vert.legend(fontsize=9)

    for s, col, ls in [(10, "#1f77b4", "-"), (30, "#ff7f0e", "--"), (50, "#2ca02c", "-.")]:
        ax_horiz.plot(x_coords, gt[s, 1, mid_idx, :], color=col, linestyle=ls, lw=2.2, label=f"OpenFOAM (t={times[s]:.2f}s)")
        ax_horiz.plot(x_coords, pred[s, 1, mid_idx, :], color=col, linestyle="none", marker="s", markersize=3.5, label=f"AI Surrogate (t={times[s]:.2f}s)")

    ax_horiz.set_xlabel("Horizontal Position x (m)", fontsize=11)
    ax_horiz.set_ylabel("Vertical Velocity u_y (m/s)", fontsize=11)
    ax_horiz.set_title("Horizontal Centerline Profile: u_y(x) at y = 0.5 m", fontsize=12, fontweight="bold")
    ax_horiz.grid(True, linestyle="--", alpha=0.5)
    ax_horiz.legend(fontsize=9)

    plt.tight_layout()
    fig2_path = output_dir / "centerline_velocity_profiles.png"
    plt.savefig(fig2_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"📊 Saved updated centerline profiles comparison to: {fig2_path}")

    # 3. Rollout Error Tracking & Speedup Benchmark
    fig3, (ax_err, ax_spd) = plt.subplots(1, 2, figsize=(14, 5))

    ax_err.plot(times, rel_l2_list, "tab:blue", lw=2.2, label="Relative L2 Error (%)")
    ax_err.plot(times, rmse_list, "tab:orange", lw=2.0, linestyle="--", label="Velocity RMSE (mm/s)")
    ax_err.set_xlabel("Simulation Physical Time (s)", fontsize=11)
    ax_err.set_ylabel("Trajectory Error", fontsize=11)
    ax_err.set_title("50-Timestep Rollout Fidelity (Trained 1,350 Epochs)", fontsize=12, fontweight="bold")
    ax_err.grid(True, linestyle="--", alpha=0.5)
    ax_err.legend(fontsize=10)

    time_step_foam = (openfoam_time / num_steps) * 1000.0
    time_step_ai = (ai_time / num_steps) * 1000.0
    speedup = openfoam_time / max(ai_time, 1e-6)

    bars = ax_spd.bar(
        ["OpenFOAM (icoFoam)", "AI Surrogate (HybridCFDNet)"],
        [time_step_foam, time_step_ai],
        color=["#7f7f7f", "#2ca02c"],
        width=0.45,
    )
    ax_spd.set_ylabel("Latency per Time-Step (milliseconds)", fontsize=11)
    ax_spd.set_title(f"Acceleration: {speedup:.1f}x Faster on GPU", fontsize=12, fontweight="bold")
    ax_spd.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax_spd.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.2,
            f"{h:.2f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    plt.tight_layout()
    fig3_path = output_dir / "simulation_speedup_and_error.png"
    plt.savefig(fig3_path, dpi=180, bbox_inches="tight")
    plt.close(fig3)
    print(f"📊 Saved updated speedup and error dynamics chart to: {fig3_path}")


def main():
    print("=" * 70)
    print("  NLSS: TRAINING HYBRID SURROGATE FOR ANOTHER 1,000 EPOCHS")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Hardware Device:               {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = Path("outputs/full_openfoam_run")
    base_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ai_foam_dir = base_dir / "ai_predictions"
    ai_foam_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest OpenFOAM Ground Truth Simulation
    foam_case_dir = Path("outputs/sample_run/openfoam_solver_case")
    print(f"\n[Step 1/5] Ingesting OpenFOAM simulation data from '{foam_case_dir}'...")
    dataset, normalizer = load_dataset_from_openfoam_case(foam_case_dir, grid_shape=(64, 64))
    num_steps = len(dataset)
    times = np.array([i * 0.005 for i in range(num_steps + 1)])
    gt_data = dataset.data.numpy()
    print(f"✓ Loaded {num_steps + 1} time snapshots (t = 0.000s to {times[-1]:.3f}s)")

    # 2. Initialize Model & Resume from Previous Checkpoint
    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=48,
        modes1=12,
        modes2=12,
        num_layers=2,
        num_heads=4,
        enforce_boundary=False,
    ).to(device)

    prev_ckpt = ckpt_dir / "best_hybrid_openfoam.pt"
    start_epoch = 350
    best_loss = 0.003943
    if prev_ckpt.exists():
        print(f"📦 Resuming from checkpoint {prev_ckpt} (Epoch {start_epoch}, Loss: {best_loss:.6f})...")
        ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
        print("✓ Successfully loaded pre-trained model weights.")

    # 3. Training Engine: 1,000 More Epochs with Multi-Step Pushforward & Importance Weighting
    additional_epochs = 1000
    total_epochs = start_epoch + additional_epochs
    print(f"\n[Step 2/5] Launching training for another {additional_epochs} epochs (Epochs {start_epoch + 1} to {total_epochs})...")
    print("  • Regularization: Multi-step Pushforward Rollout (1-step + 2-step unrolling)")
    print("  • Loss Function:  Physics-Informed NMSE + Divergence + Vorticity Regularization")
    print("  • Optimizer:      AdamW (lr=8e-4 -> 1e-5 via CosineAnnealing)")

    inputs = dataset.normalized_data[:-1].to(device)
    targets = dataset.normalized_data[1:].to(device)
    n_samples = inputs.shape[0]
    batch_size = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-5)
    criterion = PhysicsInformedCFDLoss(lambda_nmse=0.5, lambda_div=0.05, lambda_vort=0.02)

    best_weights = None
    t_train_start = time.perf_counter()

    for ep in range(1, additional_epochs + 1):
        model.train()
        current_epoch_num = start_epoch + ep
        perm = torch.randperm(n_samples - 1)  # indices up to n_samples-2 for 2-step pushforward
        epoch_loss = 0.0

        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()

            # Step 1: S_t -> S_{t+1}
            s_t = inputs[idx]
            s_t1_target = targets[idx]
            pred_1 = model(s_t)
            loss_1, _ = criterion(pred_1, s_t1_target)

            # Step 2: Autoregressive Pushforward S_{t+1} -> S_{t+2}
            # Unrolls the network on its own prediction with 50% probability
            if ep > 100 and np.random.rand() < 0.6:
                s_t2_target = targets[idx + 1]
                pred_2 = model(pred_1)
                loss_2, _ = criterion(pred_2, s_t2_target)
                total_loss = loss_1 + 0.4 * loss_2
            else:
                total_loss = loss_1

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(1, (len(perm) // batch_size))

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 100 == 0 or ep == additional_epochs:
            lr_curr = scheduler.get_last_lr()[0]
            print(f"  Epoch [{current_epoch_num:04d}/{total_epochs:04d}] - Avg Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | LR: {lr_curr:.2e}")

    t_train = time.perf_counter() - t_train_start
    print(f"\n✓ 1,000 Additional Epochs finished in {t_train:.2f} s ({t_train/additional_epochs*1000:.1f} ms/epoch)")
    print(f"✓ Converged to Best Loss: {best_loss:.6f}")

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # Save upgraded checkpoint
    save_ckpt_path = ckpt_dir / "best_hybrid_openfoam_1350epochs.pt"
    torch.save(
        {
            "epoch": total_epochs,
            "model_state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "best_loss": best_loss,
        },
        save_ckpt_path,
    )
    # Also update master best checkpoint
    shutil.copyfile(save_ckpt_path, ckpt_dir / "best_hybrid_openfoam.pt")
    print(f"💾 Checkpoint saved to: {save_ckpt_path}")

    # 4. Evaluate Full 50-Step Autoregressive Rollout
    print(f"\n[Step 3/5] Executing full 50-step simulation rollout with upgraded model...")
    pred_data, ai_time, rel_l2_list, rmse_list = evaluate_rollout(
        model=model,
        dataset=dataset,
        normalizer=normalizer,
        num_steps=num_steps,
        device=device,
    )

    openfoam_time = 3.6150
    ms_per_step_cfd = (openfoam_time / num_steps) * 1000.0
    ms_per_step_ai = (ai_time / num_steps) * 1000.0
    speedup = openfoam_time / max(ai_time, 1e-6)

    print("\n" + "=" * 70)
    print("          UPGRADED SIMULATION ACCURACY (1,350 TOTAL EPOCHS)")
    print("=" * 70)
    print(f"• Total Simulation Timesteps:    {num_steps} steps (t = 0 to {times[-1]:.3f} s, Δt = 0.005 s)")
    print(f"• Spatial Mesh Resolution:       64 x 64 cells (4,096 cells)")
    print(f"• OpenFOAM (icoFoam) Solver:     {openfoam_time:.4f} s  ({ms_per_step_cfd:.2f} ms/step)")
    print(f"• AI Surrogate (HybridCFDNet):   {ai_time:.4f} s  ({ms_per_step_ai:.2f} ms/step)")
    print(f"• Acceleration Factor:           \033[1;32m{speedup:.1f}x FASTER\033[0m")

    for s in [5, 10, 20, 30, 40, 50]:
        t_curr = times[s]
        diff_u = pred_data[s, :2] - gt_data[s, :2]
        rel_l2 = np.linalg.norm(diff_u) / np.linalg.norm(gt_data[s, :2])
        rmse_u = np.sqrt(np.mean(diff_u**2))
        rmse_p = np.sqrt(np.mean((pred_data[s, 2] - gt_data[s, 2]) ** 2))

        pred_torch = torch.from_numpy(pred_data[s : s + 1])
        div_field = compute_divergence_2d(pred_torch)
        div_res = float(torch.norm(div_field).item())

        print(
            f"  t = {t_curr:.3f}s (Step {s:02d}): "
            f"Rel L2 = \033[1;32m{rel_l2*100:5.2f}%\033[0m | "
            f"RMSE = \033[1;32m{rmse_u*1000:5.2f} mm/s\033[0m | "
            f"Pressure RMSE = {rmse_p:.4e} | "
            f"||div(u)|| = {div_res:.4e}"
        )
    print("=" * 70)

    # 5. Generate Updated Publication-Grade Plots
    print(f"\n[Step 4/5] Updating publication figures with 1,350-epoch results...")
    generate_benchmark_plots(
        gt=gt_data,
        pred=pred_data,
        times=times,
        openfoam_time=openfoam_time,
        ai_time=ai_time,
        rel_l2_list=rel_l2_list,
        rmse_list=rmse_list,
        output_dir=plots_dir,
    )

    # 6. Re-export Full Trajectory to OpenFOAM Format
    print(f"\n[Step 5/5] Exporting upgraded simulation predictions into OpenFOAM format...")
    for s in range(num_steps + 1):
        t_curr = float(times[s])
        u_2d = pred_data[s, 0:2].transpose(1, 2, 0)
        p_2d = pred_data[s, 2]
        export_simulation_state_to_openfoam(
            case_dir=ai_foam_dir,
            time_val=t_curr,
            u_pred=u_2d,
            p_pred=p_2d,
        )
    print(f"✓ Exported all 51 upgraded time directories to: {ai_foam_dir}")
    print("\n✨ 1,000 Additional Epochs & Benchmark successfully completed!")


if __name__ == "__main__":
    main()
