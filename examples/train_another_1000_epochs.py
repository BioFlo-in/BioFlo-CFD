#!/usr/bin/env python3
"""
Train Neural Surrogate Model on >= 1,000,000 Physical Datapoints for Another 1,000 Epochs (Epochs 1,001 - 2,000).
Features:
  - Resumes from Phase 1 Checkpoint: checkpoints_1000epochs/best_model.pt
  - Dataset: 1,708,032 physical state datapoints across 20 diverse Navier-Stokes regimes
  - Additional Epochs: 1,000 (total cumulative epochs: 2,000)
  - Optimizer: Fine-tuning AdamW (5e-4 -> 1e-6) with CosineAnnealingLR
  - Checkpoints: checkpoints_2000epochs/ (every 100 epochs + best_model.pt)
  - Combined 2,000-epoch telemetry & rollout error benchmarking
"""

from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nlss.dataset.million_dataset import generate_million_datapoint_suite, MillionDatapointCFDDataset
from nlss.models.dense_cfd_net import DenseCFDNet
from nlss.physics.losses import PhysicsInformedCFDLoss
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison


def main():
    print("=" * 75)
    print("  NLSS: PHASE 2 TRAINING - ANOTHER 1,000 EPOCHS (EPOCHS 1,001 -> 2,000)")
    print("  ON 1,708,032 PHYSICAL DATAPOINTS")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = Path("outputs/million_run_2000epochs")
    ckpt_dir = Path("checkpoints_2000epochs")
    base_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Million-Datapoint CFD Suite
    print("\n[Step 1/4] Loading cached dataset with 1,708,032 physical datapoints...")
    master_tensor, total_datapoints = generate_million_datapoint_suite(
        output_dir=Path("outputs/million_run_1000epochs/dataset_cache"),
        nx=64,
        ny=64,
        dt=0.005,
    )
    assert total_datapoints >= 1_000_000, f"Dataset must have at least 1,000,000 datapoints, got {total_datapoints}"

    # 2. Build Dataset & Pre-load tensors on GPU for zero-copy training
    dataset = MillionDatapointCFDDataset(master_tensor, rollout_steps=1)
    all_inputs = dataset.normalized_data[:-1].to(device)
    all_targets = dataset.normalized_data[1:].to(device)
    num_samples = all_inputs.shape[0]
    batch_size = 32
    num_batches = (num_samples + batch_size - 1) // batch_size
    print(f"✓ GPU In-Memory Cache configured: {num_samples} transitions, {num_batches} batches/epoch (batch_size={batch_size})")

    # 3. Initialize Model Architecture & Load Phase 1 Weights
    print("\n[Step 2/4] Initializing High-Capacity DenseCFDNet & Resuming Phase 1 Checkpoint...")
    model = DenseCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=64,
        modes1=16,
        modes2=16,
        num_blocks=3,
        enforce_boundary=True,
    ).to(device)

    # Load best checkpoint from Phase 1
    phase1_ckpt_path = Path("checkpoints_1000epochs/best_model.pt")
    if phase1_ckpt_path.exists():
        ckpt_1000 = torch.load(phase1_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_1000["model_state_dict"])
        best_loss = ckpt_1000.get("best_loss", float("inf"))
        start_epoch = 1000
        print(f"✓ Successfully loaded Phase 1 checkpoint from {phase1_ckpt_path}")
        print(f"✓ Resuming with initial best loss: {best_loss:.4e} from epoch {ckpt_1000.get('epoch', 1000)}")
    else:
        print("⚠ Phase 1 checkpoint not found, starting fresh from epoch 1000.")
        best_loss = float("inf")
        start_epoch = 1000

    # Load Phase 1 loss history if available
    prior_final_path = Path("checkpoints_1000epochs/final_model_1000epochs.pt")
    prior_losses = []
    if prior_final_path.exists():
        prior_final = torch.load(prior_final_path, map_location="cpu", weights_only=False)
        prior_losses = prior_final.get("loss_history", [])
        print(f"✓ Loaded {len(prior_losses)} historical loss records from Phase 1")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {total_params:,}")

    # 4. Phase 2 Training Engine: Another 1,000 Epochs (1,001 -> 2,000)
    additional_epochs = 1000
    end_epoch = start_epoch + additional_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=additional_epochs, eta_min=1e-6)
    criterion = PhysicsInformedCFDLoss(lambda_nmse=0.5, lambda_div=0.1, lambda_vort=0.05)

    log_file_path = base_dir / "train_another1000epochs.log"
    log_file = open(log_file_path, "w", buffering=1)

    print(f"\n[Step 3/4] Launching Training for Another {additional_epochs:,} Epochs (Epochs {start_epoch+1:,} to {end_epoch:,})...")
    log_file.write(f"Phase 2 Training started: Epochs {start_epoch+1} to {end_epoch} on {total_datapoints:,} physical datapoints\n")
    log_file.flush()

    phase2_losses = []
    phase2_rel_l2 = []

    t_train_start = time.perf_counter()

    for epoch_idx in range(1, additional_epochs + 1):
        current_epoch = start_epoch + epoch_idx
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
        phase2_losses.append(mean_loss)
        phase2_rel_l2.append(mean_rel)

        # Periodic logging: every 25 epochs (plus first 5 epochs)
        if epoch_idx % 25 == 0 or epoch_idx in [1, 2, 5, 10]:
            elapsed = time.perf_counter() - t_train_start
            eps = epoch_idx / max(elapsed, 1e-5)
            rem_sec = (additional_epochs - epoch_idx) / max(eps, 1e-5)
            log_line = (
                f"Epoch [{current_epoch:04d}/{end_epoch:04d}] "
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
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "normalizer": dataset.normalizer.to_dict(),
                    "best_loss": best_loss,
                    "total_datapoints": total_datapoints,
                },
                ckpt_dir / "best_model.pt",
            )

        if epoch_idx % 100 == 0:
            torch.save(
                {
                    "epoch": current_epoch,
                    "model_state_dict": model.state_dict(),
                    "normalizer": dataset.normalizer.to_dict(),
                    "loss": mean_loss,
                },
                ckpt_dir / f"model_epoch_{current_epoch}.pt",
            )
            print(f"  [Checkpoint] Saved epoch {current_epoch} to {ckpt_dir / f'model_epoch_{current_epoch}.pt'}", flush=True)

    log_file.close()

    t_train_total = time.perf_counter() - t_train_start
    print(f"\n✓ Another 1,000 Epochs Completed in {t_train_total:.1f} s ({t_train_total/60:.2f} min)")

    # Save final 2,000-epoch model
    full_loss_history = prior_losses + phase2_losses
    torch.save(
        {
            "epoch": end_epoch,
            "model_state_dict": model.state_dict(),
            "normalizer": dataset.normalizer.to_dict(),
            "loss_history": full_loss_history,
            "phase2_loss_history": phase2_losses,
            "total_datapoints": total_datapoints,
        },
        ckpt_dir / "final_model_2000epochs.pt",
    )
    print(f"✓ Saved final checkpoint to: {ckpt_dir / 'final_model_2000epochs.pt'}")

    # 5. Evaluate Multi-Step Rollout Performance
    print("\n[Step 4/4] Evaluating Rollout After 2,000 Cumulative Epochs...")
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
    print("      2,000-EPOCH TRAINED MODEL BENCHMARK RESULTS")
    print("=" * 75)
    print(f"• Total Cumulative Epochs:       {end_epoch:,}")
    print(f"• Total Physical Datapoints:     {total_datapoints:,}")
    print(f"• Phase 2 Training Time:         {t_train_total:.1f} s ({t_train_total/60:.2f} min)")
    print(f"• Rollout Steps:                 {bench['num_steps']}")
    print(f"• AI Rollout Execution Time:     {bench['ai_rollout_time_sec']:.4f} s ({bench['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Speedup Factor:                \033[1;32m{bench['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Relative L2 Error:        {bench['mean_relative_l2']:.4e} ({bench['mean_relative_l2']*100:.2f}%)")
    print(f"• Mean Velocity/Pressure RMSE:   {bench['mean_rmse']:.4e}")
    print("=" * 75 + "\n")

    # Plot Full 2,000 Epoch Loss History
    fig, ax = plt.subplots(figsize=(12, 5))
    total_len = len(full_loss_history)
    ax.plot(range(1, total_len + 1), full_loss_history, color="#1f77b4", lw=1.8, label="Total Transition Loss")
    if len(prior_losses) > 0 and len(phase2_losses) > 0:
        ax.axvline(x=len(prior_losses), color="red", linestyle="--", alpha=0.8, label="Phase 2 Start (Epoch 1,000)")
    ax.set_yscale("log")
    ax.set_xlabel("Cumulative Training Epochs (1 to 2,000)")
    ax.set_ylabel("Loss (Log Scale)")
    ax.set_title(f"Cumulative 2,000-Epoch Training Curve ({total_datapoints:,} Physical Datapoints)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    loss_curve_path = base_dir / "loss_trajectory_2000epochs.png"
    plt.savefig(loss_curve_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved cumulative 2,000-epoch loss trajectory curve to: {loss_curve_path}")

    # Plot field comparison
    plot_field_comparison(
        gt_state=test_eval_traj[30].cpu(),
        pred_state=pred_trajectory[30].cpu(),
        time_val=30 * 0.005,
        save_path=base_dir / "field_comparison_after_2000epochs.png",
    )
    print(f"📊 Saved field comparison to: {base_dir / 'field_comparison_after_2000epochs.png'}")


if __name__ == "__main__":
    main()
