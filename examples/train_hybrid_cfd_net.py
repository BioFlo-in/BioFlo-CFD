#!/usr/bin/env python3
"""
Full Training of Unified HybridCFDNet on >= 1,000,000 Physical Datapoints.
Architecture Hierarchy:
  1. Multi-Scale Neural Operator (Fourier Spectral Conv + Dense Dilated Conv)
  2. Mesh GraphNet (GNN cell-face flux message passing)
  3. DNN Inter-Layer Linkers with gating highways
  4. Spatiotemporal Transformer Head with patch attention & full-res skip
  5. Spatially-Adaptive Importance-Weighted Loss (quiescent bulk suppressed, active shear upweighted >2,400x)
"""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nlss.dataset.million_dataset import generate_million_datapoint_suite, MillionDatapointCFDDataset
from nlss.dataset.foam_dataset import FieldNormalizer
from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import compute_cfd_importance_weight, PhysicsInformedCFDLoss
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison


def main():
    print("=" * 80)
    print("  NLSS: FULL TRAINING OF UNIFIED HYBRID CFD-NET")
    print("  [Multi-Scale Operator + MeshGraphNet + DNN Linkers + Spatiotemporal Transformer]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = Path("outputs/hybrid_training")
    ckpt_dir = Path("checkpoints_hybrid")
    base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Million-Datapoint CFD Suite
    print("\n[Step 1/5] Loading multi-regime CFD dataset (>= 1,000,000 physical datapoints)...")
    master_tensor, total_datapoints = generate_million_datapoint_suite(
        output_dir=Path("outputs/million_run_1000epochs/dataset_cache"),
        nx=64,
        ny=64,
        dt=0.005,
    )
    assert total_datapoints >= 1_000_000, f"Dataset must have at least 1,000,000 datapoints, got {total_datapoints}"

    # 2. Build Dataset & Pre-load tensors on GPU
    dataset = MillionDatapointCFDDataset(master_tensor, rollout_steps=1)
    all_inputs = dataset.normalized_data[:-1].to(device)
    all_targets = dataset.normalized_data[1:].to(device)
    num_samples = all_inputs.shape[0]
    batch_size = 16
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"✓ GPU In-Memory Cache: {num_samples} transitions, {num_batches} batches/epoch (batch_size={batch_size})")

    # 3. Precompute Spatially-Adaptive Importance Weights W(x, y)
    print("\n[Step 2/5] Synthesizing Spatial Importance Weights W(x, y)...")
    all_weights = compute_cfd_importance_weight(
        target=all_targets,
        input_state=all_inputs,
        base_weight=0.1,
        alpha_grad=3.0,
        alpha_vort=3.0,
        alpha_dyn=4.0,
        power=1.5,
    )
    print(f"✓ Precomputed {all_weights.shape} weights (Min: {all_weights.min():.4f}, Max: {all_weights.max():.4f})")

    # 4. Initialize HybridCFDNet
    print("\n[Step 3/5] Initializing Unified HybridCFDNet Architecture...")
    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=48,
        modes1=12,
        modes2=12,
        num_layers=2,
        num_heads=4,
        enforce_boundary=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {total_params:,}")

    # 5. Training Engine: 1,000 Epochs
    total_epochs = 1000
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
    criterion = PhysicsInformedCFDLoss(
        lambda_nmse=0.5,
        lambda_div=0.1,
        lambda_vort=0.05,
        use_importance_weighting=True,
    )

    log_file_path = base_dir / "train_hybrid.log"
    log_file = open(log_file_path, "w", buffering=1)

    print(f"\n[Step 4/5] Launching Training for {total_epochs:,} Epochs on {total_datapoints:,} Physical Datapoints...")
    log_file.write(f"HybridCFDNet Training started on {total_datapoints:,} physical datapoints for {total_epochs} epochs\n")
    log_file.flush()

    loss_history = []
    rel_l2_history = []
    best_loss = float("inf")

    t_train_start = time.perf_counter()

    for epoch in range(1, total_epochs + 1):
        model.train()
        epoch_losses = []
        epoch_rel_l2 = []

        perm = torch.randperm(num_samples, device=device)
        for i in range(0, num_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            x_b = all_inputs[batch_idx]
            y_b = all_targets[batch_idx]
            w_b = all_weights[batch_idx]

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_b)
            loss, metrics = criterion(pred, y_b, input_state=x_b, weight=w_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(metrics["loss_total"])
            epoch_rel_l2.append(metrics["loss_rel_l2"])

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses))
        mean_rel = float(np.mean(epoch_rel_l2))
        loss_history.append(mean_loss)
        rel_l2_history.append(mean_rel)

        # Periodic logging: every 25 epochs (plus first 5 epochs)
        if epoch % 25 == 0 or epoch in [1, 2, 5, 10]:
            elapsed = time.perf_counter() - t_train_start
            eps = epoch / max(elapsed, 1e-5)
            rem_sec = (total_epochs - epoch) / max(eps, 1e-5)
            log_line = (
                f"Epoch [{epoch:04d}/{total_epochs:04d}] "
                f"Active Loss: {mean_loss:.4e} | Weighted Rel L2: {mean_rel:.4e} ({mean_rel*100:.2f}%) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | Speed: {eps:.2f} epoch/s | "
                f"ETA: {rem_sec/60:.1f} min"
            )
            print(log_line, flush=True)
            log_file.write(log_line + "\n")
            log_file.flush()

        # Checkpointing: best model and every 100 epochs
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "normalizer": dataset.normalizer.to_dict(),
                    "best_loss": best_loss,
                    "total_datapoints": total_datapoints,
                },
                ckpt_dir / "best_model.pt",
            )

        if epoch % 100 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "normalizer": dataset.normalizer.to_dict(),
                    "loss": mean_loss,
                },
                ckpt_dir / f"model_epoch_{epoch}.pt",
            )
            print(f"  [Checkpoint] Saved epoch {epoch} to {ckpt_dir / f'model_epoch_{epoch}.pt'}", flush=True)

    log_file.close()

    t_train_total = time.perf_counter() - t_train_start
    print(f"\n✓ Training Completed in {t_train_total:.1f} s ({t_train_total/60:.2f} min)")

    # Save final model
    torch.save(
        {
            "epoch": total_epochs,
            "model_state_dict": model.state_dict(),
            "normalizer": dataset.normalizer.to_dict(),
            "loss_history": loss_history,
            "total_datapoints": total_datapoints,
        },
        ckpt_dir / "final_model_hybrid.pt",
    )
    print(f"✓ Saved final checkpoint to: {ckpt_dir / 'final_model_hybrid.pt'}")

    # 6. Evaluate Multi-Step Rollout Performance
    print("\n[Step 5/5] Evaluating Rollout After 1,000 Epochs...")
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    engine = RolloutEngine(model=model, normalizer=dataset.normalizer, device=device)
    test_eval_traj = master_tensor[:50].to(device)
    init_state = test_eval_traj[0]

    num_eval_steps = 40
    pred_trajectory, ai_time = engine.rollout(init_state, num_steps=num_eval_steps)

    bench = benchmark_acceleration(
        rollout_engine=engine,
        ground_truth_trajectory=test_eval_traj.cpu(),
        cfd_solver_time_sec=2.8,
        num_steps=num_eval_steps,
    )

    print("\n" + "=" * 80)
    print("      HYBRID CFD-NET 1,000-EPOCH BENCHMARK RESULTS")
    print("=" * 80)
    print(f"• Total Training Epochs:         {total_epochs:,}")
    print(f"• Total Physical Datapoints:     {total_datapoints:,}")
    print(f"• Training Time:                 {t_train_total:.1f} s ({t_train_total/60:.2f} min)")
    print(f"• Rollout Steps:                 {bench['num_steps']}")
    print(f"• AI Rollout Execution Time:     {bench['ai_rollout_time_sec']:.4f} s ({bench['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Speedup Factor:                \033[1;32m{bench['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Relative L2 Error:        {bench['mean_relative_l2']:.4e} ({bench['mean_relative_l2']*100:.2f}%)")
    print(f"• Mean Velocity/Pressure RMSE:   {bench['mean_rmse']:.4e}")
    print("=" * 80 + "\n")

    # Plot Loss History
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, total_epochs + 1), loss_history, color="#2ca02c", lw=2, label="Hybrid Importance-Weighted Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training Epoch (1 to 1,000)")
    ax.set_ylabel("Active Loss (Log Scale)")
    ax.set_title(f"HybridCFDNet 1,000-Epoch Training Curve ({total_datapoints:,} Physical Datapoints)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    loss_curve_path = base_dir / "loss_trajectory_hybrid.png"
    plt.savefig(loss_curve_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved loss trajectory curve to: {loss_curve_path}")

    # Plot field comparison
    plot_field_comparison(
        gt_state=test_eval_traj[30].cpu(),
        pred_state=pred_trajectory[30].cpu(),
        time_val=30 * 0.005,
        save_path=base_dir / "field_comparison_hybrid.png",
    )
    print(f"📊 Saved field comparison to: {base_dir / 'field_comparison_hybrid.png'}")


if __name__ == "__main__":
    main()
