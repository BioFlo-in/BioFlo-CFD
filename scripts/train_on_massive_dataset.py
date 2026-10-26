#!/usr/bin/env python3
"""
Automated Training Daemon for Massive OpenFOAM Dataset.
Monitors the currently active background training process (PID 313314) and GPU VRAM.
As soon as the GPU is released, it automatically trains HybridCFDNet on the 154-regime
dataset package (7,636 transition pairs, velocity & surface boundary condition sweeps).
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Ensure NLSS package is in Python path
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from nlss.dataset.foam_dataset import FieldNormalizer
from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import compute_divergence_2d


def is_pid_running(pid: int) -> bool:
    """Checks whether a given PID is currently active using /proc on Linux."""
    proc_path = Path(f"/proc/{pid}")
    if not proc_path.exists():
        return False
    try:
        # Verify process is still responding
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_available_gpu_memory_mb() -> float:
    """Returns free GPU memory in MB using torch.cuda if available."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        # Use nvidia-smi query for total real-time system-wide free memory
        import subprocess
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return float(res.strip().split("\n")[0])
    except Exception:
        return 2048.0


def get_gpu_temperature() -> int:
    """Returns GPU temperature in Celsius."""
    if not torch.cuda.is_available():
        return 0
    try:
        import subprocess
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        return int(res.strip().split("\n")[0])
    except Exception:
        return 0


def check_and_throttle_gpu_temp(max_temp: int = 72, cool_down_target: int = 62):
    """Enforces active thermal ceiling, preventing laptop overheating."""
    temp = get_gpu_temperature()
    if temp >= max_temp:
        print(f"\n❄️  [THERMAL GUARD] GPU reached {temp}°C (limit: {max_temp}°C). Cooling down...")
        while temp > cool_down_target:
            time.sleep(3)
            temp = get_gpu_temperature()
        print(f"✓ Cooled down to {temp}°C. Resuming execution.")


def wait_for_gpu_release(target_pid: int = 313314, min_free_vram_mb: int = 3000, poll_interval_sec: int = 20):
    """Politely waits until the targeted PID finishes and sufficient GPU VRAM is free."""
    print("=" * 80)
    print("⏳ GPU TRAINING QUEUE ACTIVE")
    print(f"Monitoring active process PID : {target_pid}")
    print(f"Required free GPU memory       : {min_free_vram_mb} MB")
    print(f"Poll interval                  : {poll_interval_sec} seconds")
    print("=" * 80)

    wait_start = time.time()
    last_print = 0

    while True:
        pid_active = is_pid_running(target_pid)
        free_mb = get_available_gpu_memory_mb()

        elapsed_min = (time.time() - wait_start) / 60.0

        if (time.time() - last_print) >= 60:
            status_str = f"PID {target_pid} still running" if pid_active else f"PID {target_pid} terminated"
            print(f"[{time.strftime('%H:%M:%S')}] Waiting: {status_str} | Free VRAM: {free_mb:.0f} MB | Elapsed: {elapsed_min:.1f} min")
            last_print = time.time()

        if not pid_active and free_mb >= min_free_vram_mb:
            print(f"\n🎉 Process {target_pid} completed and {free_mb:.0f} MB VRAM is free!")
            print(f"Proceeding immediately to surrogate model training on massive OpenFOAM dataset.")
            time.sleep(5)  # Grace period for GPU buffers to fully clear
            break

        time.sleep(poll_interval_sec)


def train_surrogate(
    dataset_path: Path,
    output_dir: Path,
    preset: str = "medium",
    epochs: int = 200,
    batch_size: int = 16,
    grad_accum_steps: int = 2,
    lr: float = 1e-3,
    div_weight: float = 0.02,
    use_checkpointing: bool = True,
    device: Optional[str] = None,
    throttle_sleep: float = 0.0,
    max_temp: int = 72,
    resume: Optional[str] = None,
):
    """Trains HybridCFDNet on the massive dataset with configurable parameter scaling."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("🚀 STARTING TRAINING ON MASSIVE 154-REGIME OPENFOAM DATASET")
    print(f"Dataset path       : {dataset_path}")
    print(f"Model Preset       : {preset.upper()}")
    print(f"Target Epochs      : {epochs}")
    print(f"Batch size (micro) : {batch_size}")
    print(f"Grad Accumulation  : {grad_accum_steps} (Effective batch size: {batch_size * grad_accum_steps})")
    print(f"Learning rate      : {lr}")
    print(f"Checkpointing      : {use_checkpointing}")
    print(f"Device             : {device.upper()}")
    if throttle_sleep > 0:
        print(f"Thermal Throttle   : {throttle_sleep:.3f}s sleep per batch to prevent GPU overheating")
    print(f"Max GPU Temp Guard : {max_temp}°C (auto-cool pause if exceeded)")
    print("=" * 80)

    # 1. Load dataset package
    print("Loading pre-packaged PyTorch dataset...")
    t0 = time.time()
    pkg = torch.load(dataset_path, weights_only=False, map_location="cpu")
    inputs = pkg["inputs"]
    targets = pkg["targets"]
    if "normalizer" in pkg:
        normalizer = pkg["normalizer"]
    else:
        normalizer = FieldNormalizer()
        normalizer.mean = pkg["normalizer_mean"]
        normalizer.std = pkg["normalizer_std"]
    print(f"✓ Loaded {len(inputs):,} pairs in {time.time() - t0:.2f}s | Shape: {inputs.shape}")

    # 2. Normalize and Train/Val split
    inp_norm = normalizer.normalize(inputs)
    tgt_norm = normalizer.normalize(targets)

    full_ds = TensorDataset(inp_norm, tgt_norm)
    val_size = int(0.15 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Train split: {len(train_ds):,} pairs | Val split: {len(val_ds):,} pairs")

    # 3. Model & Optimizer with Scaling Preset
    model = HybridCFDNet.from_preset(
        preset,
        in_channels=3,
        out_channels=3,
        enforce_boundary=False,
        use_checkpointing=use_checkpointing,
    ).to(device)

    total_params = model.num_parameters
    print(f"HybridCFDNet [{preset.upper()}] trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_fn = nn.MSELoss()

    output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 1

    # Resume from checkpoint if requested
    if resume:
        resume_path = Path(resume)
        if resume_path.exists():
            print(f"🔄 Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                try:
                    model.load_state_dict(ckpt["model_state_dict"])
                    print("✓ Loaded model state dict.")
                except Exception as e:
                    print(f"⚠️ Could not load entire model state dict: {e}")
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    print("✓ Loaded optimizer state dict.")
                except Exception:
                    pass
            if "val_loss" in ckpt:
                best_val_loss = float(ckpt["val_loss"])
                print(f"✓ Resumed best val loss: {best_val_loss:.6f}")
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
                print(f"✓ Resuming at epoch: {start_epoch}")

    train_loss_hist = []
    val_loss_hist = []
    div_loss_hist = []

    start_train_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_train_mse = 0.0
        running_train_div = 0.0
        n_train_samples = 0
        optimizer.zero_grad()

        for step_idx, (b_inp, b_tgt) in enumerate(train_loader):
            b_inp = b_inp.to(device)
            b_tgt = b_tgt.to(device)

            b_pred = model(b_inp)
            loss_mse = mse_fn(b_pred, b_tgt)

            # Physics divergence penalty on denormalized physical velocity
            b_pred_phys = normalizer.denormalize(b_pred)
            div = compute_divergence_2d(b_pred_phys[:, :2])
            loss_div = torch.mean(div ** 2)

            total_loss = (loss_mse + div_weight * loss_div) / grad_accum_steps
            total_loss.backward()

            if (step_idx + 1) % grad_accum_steps == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                if throttle_sleep > 0:
                    time.sleep(throttle_sleep)

            # Check thermal guard every 25 steps
            if (step_idx + 1) % 25 == 0 and device == "cuda":
                check_and_throttle_gpu_temp(max_temp=max_temp)

            bs = len(b_inp)
            running_train_mse += loss_mse.item() * bs
            running_train_div += loss_div.item() * bs
            n_train_samples += bs

        scheduler.step()

        epoch_train_mse = running_train_mse / n_train_samples
        epoch_train_div = running_train_div / n_train_samples
        train_loss_hist.append(epoch_train_mse)
        div_loss_hist.append(epoch_train_div)

        # Validation
        model.eval()
        running_val_mse = 0.0
        n_val_samples = 0
        with torch.no_grad():
            for b_inp, b_tgt in val_loader:
                b_inp = b_inp.to(device)
                b_tgt = b_tgt.to(device)
                b_pred = model(b_inp)
                running_val_mse += mse_fn(b_pred, b_tgt).item() * len(b_inp)
                n_val_samples += len(b_inp)

        epoch_val_mse = running_val_mse / n_val_samples
        val_loss_hist.append(epoch_val_mse)

        # End of epoch thermal guard
        if device == "cuda":
            check_and_throttle_gpu_temp(max_temp=max_temp)

        # Save best model
        if epoch_val_mse < best_val_loss:
            best_val_loss = epoch_val_mse
            torch.save(
                {
                    "epoch": epoch,
                    "preset": preset,
                    "num_parameters": total_params,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "normalizer": normalizer,
                    "normalizer_mean": normalizer.mean,
                    "normalizer_std": normalizer.std,
                },
                output_dir / f"hybrid_cfd_net_{preset}_best.pt",
            )
            # Also keep latest generic pointer
            torch.save(
                {
                    "epoch": epoch,
                    "preset": preset,
                    "num_parameters": total_params,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "normalizer": normalizer,
                    "normalizer_mean": normalizer.mean,
                    "normalizer_std": normalizer.std,
                },
                output_dir / "hybrid_cfd_net_best.pt",
            )
            is_best = " ⭐ (NEW BEST)"
        else:
            is_best = ""

        # Periodic logging
        if epoch % 5 == 0 or epoch == start_epoch or epoch == epochs:
            elapsed = time.time() - start_train_time
            curr_temp = get_gpu_temperature() if device == "cuda" else 0
            temp_str = f" | GPU: {curr_temp}°C" if device == "cuda" else ""
            print(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Train MSE: {epoch_train_mse:.5f} | "
                f"Div Loss: {epoch_train_div:.5f} | "
                f"Val MSE: {epoch_val_mse:.5f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}{temp_str} | "
                f"Time: {elapsed:.1f}s{is_best}"
            )

    # Save final latest model
    torch.save(
        {
            "epoch": epochs,
            "preset": preset,
            "num_parameters": total_params,
            "model_state_dict": model.state_dict(),
            "val_loss": epoch_val_mse,
            "normalizer": normalizer,
            "normalizer_mean": normalizer.mean,
            "normalizer_std": normalizer.std,
        },
        output_dir / f"hybrid_cfd_net_{preset}_latest.pt",
    )

    # Plot loss curves
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(range(1, epochs + 1), train_loss_hist, label="Train MSE", color="#1f77b4", lw=2)
    ax1.plot(range(1, epochs + 1), val_loss_hist, label="Val MSE", color="#ff7f0e", lw=2)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Normalized Mean Squared Error", fontweight="bold", color="#1f77b4")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(range(1, epochs + 1), div_loss_hist, label="Continuity Div Loss", color="#2ca02c", linestyle="--", lw=1.5)
    ax2.set_ylabel("Divergence Penalty ||div(u)||^2", fontweight="bold", color="#2ca02c")
    ax2.set_yscale("log")

    plt.title(f"HybridCFDNet Training on Massive OpenFOAM Dataset (154 Regimes)", fontweight="bold")
    fig.tight_layout()
    plot_path = output_dir / "training_loss_history.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"✓ Saved loss history plot to {plot_path}")

    # Save summary metadata
    summary = {
        "dataset": str(dataset_path),
        "total_pairs": len(full_ds),
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "epochs": epochs,
        "batch_size": batch_size,
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(epoch_val_mse),
        "total_train_time_sec": float(time.time() - start_train_time),
        "device": device,
    }
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print(f"Best Val MSE: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Queue and train surrogate after active training finishes.")
    parser.add_argument("--preset", type=str, default="large", choices=["base", "medium", "large", "xlarge", "xxlarge"], help="Model scaling preset")
    parser.add_argument("--wait-pid", type=int, default=804682, help="PID to wait for")
    parser.add_argument("--min-vram", type=int, default=2500, help="Minimum free VRAM in MB")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Micro-batch size per step")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--nowait", action="store_true", help="Skip waiting for PID")
    parser.add_argument("--no-checkpointing", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--throttle-sleep", type=float, default=0.0, help="Thermal throttle delay in seconds per step to prevent GPU overheating (e.g. 0.05)")
    parser.add_argument("--max-temp", type=int, default=70, help="Maximum GPU temperature before auto-pause (default: 70°C)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    dataset_path = WORKSPACE / "outputs/massive_openfoam_dataset/massive_openfoam_dataset.pt"
    if not dataset_path.exists():
        print(f"Error: dataset not found at {dataset_path}")
        sys.exit(1)

    output_dir = WORKSPACE / "checkpoints_massive_openfoam"

    if not args.nowait:
        wait_for_gpu_release(target_pid=args.wait_pid, min_free_vram_mb=args.min_vram)

    train_surrogate(
        dataset_path=dataset_path,
        output_dir=output_dir,
        preset=args.preset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        use_checkpointing=not args.no_checkpointing,
        device=args.device,
        throttle_sleep=args.throttle_sleep,
        max_temp=args.max_temp,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
