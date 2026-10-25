#!/usr/bin/env python3
"""
NLSS End-to-End Demo:
1. Generates OpenFOAM simulation case data (Lid-Driven Cavity)
2. Trains a Fourier Neural Operator (FNO-2D) with continuity & vorticity regularization
3. Executes multi-step autoregressive rollout
4. Computes speedup factor and accuracy metrics
5. Exports predictions to OpenFOAM format and saves visualization plots
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from nlss.dataset.generator import generate_and_export_openfoam_benchmark
from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, CFDTrajectoryDataset
from nlss.models.fno import FNO2d
from nlss.train import train_surrogate_model
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison, plot_rollout_metrics


def main():
    print("=" * 60)
    print("  NLSS: OpenFOAM CFD AI Acceleration Pipeline")
    print("=" * 60)

    # Output paths
    output_dir = Path("outputs")
    case_dir = output_dir / "openfoam_ground_truth"
    ckpt_dir = output_dir / "checkpoints"
    plots_dir = output_dir / "plots"
    ai_foam_dir = output_dir / "openfoam_ai_predictions"

    plots_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Step 1: Generate High-Fidelity OpenFOAM Simulation Trajectory
    print("\n[Step 1/5] Generating OpenFOAM CFD Simulation Trajectory...")
    t0 = time.time()
    generate_and_export_openfoam_benchmark(
        output_case_dir=case_dir,
        nx=64,
        ny=64,
        n_steps=120,
        save_interval=2,
        nu=0.01,
        dt=0.001,
        u_lid=1.0,
    )
    cfd_gen_time = time.time() - t0
    print(f"✓ OpenFOAM case saved to '{case_dir}' (Generation time: {cfd_gen_time:.2f}s)")

    # Step 2: Ingest OpenFOAM Native Format into PyTorch
    print("\n[Step 2/5] Ingesting OpenFOAM field directories into PyTorch tensors...")
    dataset, normalizer = load_dataset_from_openfoam_case(
        case_dir=case_dir,
        grid_shape=(64, 64),
    )
    print(f"✓ Total simulation snapshots loaded: {len(dataset) + 1}")
    print(f"  Field shape per step: (channels=3 [u_x, u_y, p], height=64, width=64)")

    # Train / Val Split (80% / 20%)
    n_total = len(dataset)
    n_train = int(0.85 * n_total)
    train_ds = CFDTrajectoryDataset(dataset.normalized_data[:n_train], normalizer=normalizer)
    val_ds = CFDTrajectoryDataset(dataset.normalized_data[n_train:], normalizer=normalizer)

    # Step 3: Train Fourier Neural Operator (FNO-2D)
    print("\n[Step 3/5] Training Fourier Neural Operator (FNO-2D) surrogate...")
    model = FNO2d(
        in_channels=3,
        out_channels=3,
        modes1=16,
        modes2=16,
        width=48,
        num_layers=4,
    )

    history = train_surrogate_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=35,
        batch_size=8,
        lr=1e-3,
        device=device,
        checkpoint_dir=ckpt_dir,
        lambda_div=0.05,
        lambda_vort=0.02,
        log_interval=5,
    )

    # Step 4: Multi-Step Autoregressive Rollout & Benchmarking
    print("\n[Step 4/5] Running Autoregressive Rollout & Benchmarking Speedup...")
    engine = RolloutEngine(model=model, normalizer=normalizer, device=device)

    gt_trajectory = dataset.data  # unnormalized ground truth (T, 3, 64, 64)
    init_state = gt_trajectory[0]
    num_rollout_steps = 40

    bench_results = benchmark_acceleration(
        rollout_engine=engine,
        ground_truth_trajectory=gt_trajectory,
        cfd_solver_time_sec=cfd_gen_time * (num_rollout_steps / 60),
        num_steps=num_rollout_steps,
    )

    print("\n" + "=" * 60)
    print("         AI CFD ACCELERATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"• Rollout Steps:             {bench_results['num_steps']}")
    print(f"• CFD Numerical Solver Time: {bench_results['cfd_solver_time_sec']:.4f} s ({bench_results['time_per_step_cfd_ms']:.2f} ms/step)")
    print(f"• AI Surrogate Model Time:   {bench_results['ai_rollout_time_sec']:.4f} s ({bench_results['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Acceleration Factor:       \033[1;32m{bench_results['speedup_factor']:.1f}x SPEEDUP\033[0m")
    print(f"• Mean Relative L2 Error:    {bench_results['mean_relative_l2']:.4e} ({bench_results['mean_relative_l2']*100:.2f}%)")
    print(f"• Mean RMSE:                 {bench_results['mean_rmse']:.4e}")
    print("=" * 60)

    # Step 5: Visualizations & OpenFOAM Export
    print("\n[Step 5/5] Generating Visualizations and Exporting to OpenFOAM...")
    pred_traj, _ = engine.rollout(init_state, num_steps=num_rollout_steps)

    # Comparison plot at step 30
    step_eval = min(30, num_rollout_steps)
    t_eval = step_eval * 0.002
    comp_plot_path = plots_dir / "cfd_comparison.png"
    plot_field_comparison(
        gt_state=gt_trajectory[step_eval],
        pred_state=pred_traj[step_eval],
        time_val=t_eval,
        save_path=comp_plot_path,
    )
    print(f"✓ Saved field comparison figure to: {comp_plot_path}")

    # Rollout error trajectory plot
    metric_plot_path = plots_dir / "rollout_speedup_metrics.png"
    plot_rollout_metrics(bench_results, save_path=metric_plot_path)
    print(f"✓ Saved error trajectory plot to: {metric_plot_path}")

    # Export AI predictions directly to OpenFOAM format
    times = [i * 0.002 for i in range(num_rollout_steps + 1)]
    engine.export_rollout_to_openfoam(pred_traj, times, ai_foam_dir)
    print(f"✓ Exported {num_rollout_steps} AI time steps into native OpenFOAM case: '{ai_foam_dir}'")
    print("\n✨ Done! OpenFOAM AI acceleration workflow completed successfully.")


if __name__ == "__main__":
    main()
