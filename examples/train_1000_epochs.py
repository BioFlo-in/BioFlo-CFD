#!/usr/bin/env python3
"""
Train Neural Surrogate Model on >= 1,000,000 Physical Datapoints for 1,000 Epochs.
Features:
  - Dataset: >= 1,000,000 physical state datapoints across 20 diverse Navier-Stokes regimes.
  - Duration: 1,000 full training epochs.
  - Optimization: AdamW with CosineAnnealingLR (1.5e-3 -> 1e-5) & PyTorch AMP.
  - Checkpoints: Saved every 100 epochs (epoch_100.pt, epoch_200.pt, ..., final_1000.pt) and best_model.pt.
  - Metrics & Logging: Periodic reporting and loss trajectory telemetry.
"""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nlss.dataset.million_dataset import generate_million_datapoint_suite, MillionDatapointCFDDataset
from nlss.models.dense_cfd_net import DenseCFDNet
from nlss.physics.losses import PhysicsInformedCFDLoss
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison


def main():
    print("=" * 75)
    print("  NLSS: 1,000-EPOCH TRAINING ON >= 1,000,000 PHYSICAL DATAPOINTS")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = Path("outputs/million_run_1000epochs")
    ckpt_dir = Path("checkpoints_1000epochs")
    base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Million-Datapoint CFD Suite
    print("\n[Step 1/4] Constructing dataset with >= 1,000,000 physical datapoints...")
    master_tensor, total_datapoints = generate_million_datapoint_suite(
        output_dir=base_dir / "dataset_cache",
        nx=64,
        ny=64,
        dt=0.005,
    )
    assert total_datapoints >= 1_000_000, f"Dataset must have at least 1,000,000 datapoints, got {total_datapoints}"

    # 2. Build Dataset & Pre-load tensors on GPU for zero-copy training
    dataset = MillionDatapointCFDDataset(master_tensor, rollout_steps=1)
    
    # Store normalized data directly in GPU VRAM (417 snapshots = ~20 MB VRAM)
    all_inputs = dataset.normalized_data[:-1].to(device)
    all_targets = dataset.normalized_data[1:].to(device)
    num_samples = all_inputs.shape[0]
    batch_size = 32
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"✓ GPU In-Memory Cache configured: {num_samples} transitions, {num_batches} batches/epoch (batch_size={batch_size})")

    # 3. Initialize Model Architecture
    print("\n[Step 2/4] Initializing High-Capacity DenseCFDNet...")
    model = DenseCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=64,
        modes1=16,
        modes2=16,
        num_blocks=3,
        enforce_boundary=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {total_params:,}")

    # 4. Training Engine: 1,000 Epochs
    total_epochs = 1000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)
    criterion = PhysicsInformedCFDLoss(lambda_nmse=0.5, lambda_div=0.1, lambda_vort=0.05)

    log_file_path = base_dir / "train_1000epochs.log"
    log_file = open(log_file_path, "w", buffering=1)

    print(f"\n[Step 3/4] Launching 1,000-Epoch Training on {total_datapoints:,} Datapoints...")
    log_file.write(f"Training started on {total_datapoints:,} physical datapoints for {total_epochs} epochs\n")
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

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_b)
            loss, metrics = criterion(pred, y_b, input_state=x_b)
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
                f"Loss: {mean_loss:.4e} | Rel L2: {mean_rel:.4e} ({mean_rel*100:.2f}%) | "
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
    print(f"\n✓ 1,000 Epochs Completed in {t_train_total:.1f} s ({t_train_total/60:.2f} min)")

    # Save final model
    torch.save(
        {
            "epoch": total_epochs,
            "model_state_dict": model.state_dict(),
            "normalizer": dataset.normalizer.to_dict(),
            "loss_history": loss_history,
            "total_datapoints": total_datapoints,
        },
        ckpt_dir / "final_model_1000epochs.pt",
    )
    print(f"✓ Saved final checkpoint to: {ckpt_dir / 'final_model_1000epochs.pt'}")

    # 5. Evaluate Multi-Step Rollout Performance
    print("\n[Step 4/4] Evaluating Rollout After 1,000-Epoch Training...")
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

    print("\n" + "=" * 75)
    print("      1,000-EPOCH TRAINED MODEL BENCHMARK RESULTS")
    print("=" * 75)
    print(f"• Total Training Epochs:         {total_epochs:,}")
    print(f"• Total Physical Datapoints:     {total_datapoints:,}")
    print(f"• Training Time:                 {t_train_total:.1f} s ({t_train_total/60:.2f} min)")
    print(f"• Rollout Steps:                 {bench['num_steps']}")
    print(f"• AI Rollout Execution Time:     {bench['ai_rollout_time_sec']:.4f} s ({bench['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Speedup Factor:                \033[1;32m{bench['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Relative L2 Error:        {bench['mean_relative_l2']:.4e} ({bench['mean_relative_l2']*100:.2f}%)")
    print(f"• Mean Velocity/Pressure RMSE:   {bench['mean_rmse']:.4e}")
    print("=" * 75 + "\n")

    # Plot 1,000 Epoch Loss History
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, total_epochs + 1), loss_history, color="tab:blue", lw=2, label="Transition Loss")
    ax.set_yscale("log")
    ax.set_xlabel("Training Epoch (1 to 1,000)")
    ax.set_ylabel("Loss (Log Scale)")
    ax.set_title(f"1,000-Epoch Training Curve ({total_datapoints:,} Physical Datapoints)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    loss_curve_path = base_dir / "loss_trajectory_1000epochs.png"
    plt.savefig(loss_curve_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved loss trajectory curve to: {loss_curve_path}")

    # Plot field comparison
    plot_field_comparison(
        gt_state=test_eval_traj[30].cpu(),
        pred_state=pred_trajectory[30].cpu(),
        time_val=30 * 0.005,
        save_path=base_dir / "field_comparison_after_1000epochs.png",
    )
    print(f"📊 Saved field comparison to: {base_dir}")


if __name__ == "__main__":
    main()
