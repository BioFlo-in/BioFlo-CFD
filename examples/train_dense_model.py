#!/usr/bin/env python3
"""
Retrain DenseCFDNet with properly conditioned transition loss and multi-step pushforward unrolling.
Produces a robust, high-capacity CFD surrogate model that remains stable over long rollouts.
"""

from pathlib import Path
import time
import numpy as np
import torch

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, CFDTrajectoryDataset, FieldNormalizer
from nlss.dataset.generator import NavierStokes2DSimulator
from nlss.models.dense_cfd_net import DenseCFDNet
from nlss.train import train_surrogate_model
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison, plot_rollout_metrics


def generate_complementary_simulations(nx=64, ny=64, n_steps=60, dt=0.005):
    """
    Generates diverse physical flow trajectories with varying Reynolds numbers and lid speeds
    to provide the dense model with sufficient training data for Navier-Stokes dynamics.
    """
    print("🌊 Generating multi-condition CFD training trajectories (varying Re and u_lid)...")
    trajectories = []
    configs = [
        {"nu": 0.01, "u_lid": 1.0},   # Re = 100
        {"nu": 0.005, "u_lid": 1.0},  # Re = 200
        {"nu": 0.01, "u_lid": 0.8},   # Re = 80
        {"nu": 0.008, "u_lid": 1.2},  # Re = 150
    ]

    for cfg in configs:
        sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=cfg["nu"], dt=dt)
        sim.set_lid_driven_cavity(u_lid=cfg["u_lid"])
        traj = sim.run_simulation(n_steps=n_steps, save_interval=1, is_cavity=True, u_lid=cfg["u_lid"])
        # Shape: (T, 3, H, W)
        ux = traj["u"]
        uy = traj["v"]
        p = traj["p"]
        stacked = np.stack([ux, uy, p], axis=1)  # (T, 3, ny, nx)
        trajectories.append(stacked)

    print(f"✓ Generated {len(trajectories)} trajectories with {n_steps} snapshots each ({len(trajectories)*n_steps} total physical states)")
    return trajectories


def main():
    print("=" * 65)
    print("  RETRAINING DENSE CFD SURROGATE MODEL (DenseCFDNet)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    ckpt_dir = Path("checkpoints_dense")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest authentic OpenFOAM case data
    print("\n[Step 1/4] Ingesting OpenFOAM simulation data...")
    openfoam_ds, normalizer = load_dataset_from_openfoam_case(
        case_dir="outputs/sample_run/openfoam_solver_case",
        grid_shape=(64, 64),
    )
    foam_traj = openfoam_ds.traj_list[0].numpy()

    # 2. Add multi-condition CFD training data
    comp_trajs = generate_complementary_simulations(nx=64, ny=64, n_steps=50, dt=0.005)
    all_trajs = [foam_traj] + comp_trajs

    # Build dataset with multi-step pushforward pairs (rollout_steps=2)
    # Fitting normalizer across all trajectories
    all_stacked = np.concatenate(all_trajs, axis=0)
    normalizer = FieldNormalizer().fit(torch.from_numpy(all_stacked).float())

    # Train / Val split (85% / 15%)
    total_samples = sum(len(t) - 2 for t in all_trajs)
    dataset = CFDTrajectoryDataset(all_trajs, rollout_steps=2, normalizer=normalizer)
    print(f"✓ Built Pushforward Training Dataset: {len(dataset)} transition sequences (rollout_steps=2)")

    n_train = int(0.85 * len(dataset))
    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    train_ds.normalizer = normalizer

    # 3. Instantiate High-Capacity DenseCFDNet
    print("\n[Step 2/4] Initializing DenseCFDNet architecture...")
    model = DenseCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=96,
        modes1=20,
        modes2=20,
        num_blocks=4,
        enforce_boundary=True,
        lid_velocity=1.0,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ DenseCFDNet trainable parameters: {total_params:,}")

    # 4. Train Model with Properly Scaled Loss
    print("\n[Step 3/4] Retraining with properly conditioned transition loss...")
    history = train_surrogate_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=40,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        checkpoint_dir=ckpt_dir,
        lambda_div=0.1,
        lambda_vort=0.05,
        log_interval=5,
    )

    # 5. Benchmark Rollout on the Authentic OpenFOAM Simulation
    print("\n[Step 4/4] Evaluating Rollout on the Authentic OpenFOAM Simulation...")
    # Load best checkpoint
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    engine = RolloutEngine(model=model, normalizer=normalizer, device=device)
    init_state = torch.from_numpy(foam_traj[0]).float()
    gt_trajectory = torch.from_numpy(foam_traj).float()

    num_steps = 40
    pred_trajectory, ai_time = engine.rollout(init_state, num_steps=num_steps)

    bench_results = benchmark_acceleration(
        rollout_engine=engine,
        ground_truth_trajectory=gt_trajectory,
        cfd_solver_time_sec=3.345,
        num_steps=num_steps,
    )

    print("\n" + "=" * 65)
    print("      DENSE MODEL ACCELERATION & ACCURACY BENCHMARK")
    print("=" * 65)
    print(f"• Rollout Steps:             {bench_results['num_steps']}")
    print(f"• OpenFOAM Solver Time:      {bench_results['cfd_solver_time_sec']:.4f} s ({bench_results['time_per_step_cfd_ms']:.2f} ms/step)")
    print(f"• Dense AI Surrogate Time:   {bench_results['ai_rollout_time_sec']:.4f} s ({bench_results['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Acceleration Factor:       \033[1;32m{bench_results['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Relative L2 Error:    {bench_results['mean_relative_l2']:.4e} ({bench_results['mean_relative_l2']*100:.2f}%)")
    print(f"• Mean Velocity/p RMSE:      {bench_results['mean_rmse']:.4e}")
    print("=" * 65 + "\n")

    # Plot comparisons
    out_plots = Path("outputs/dense_run")
    out_plots.mkdir(parents=True, exist_ok=True)
    plot_field_comparison(
        gt_state=gt_trajectory[30],
        pred_state=pred_trajectory[30],
        time_val=30 * 0.005,
        save_path=out_plots / "dense_model_field_comparison.png",
    )
    plot_rollout_metrics(bench_results, save_path=out_plots / "dense_model_speedup_metrics.png")
    print(f"📊 Saved comparison plots to: {out_plots}")


if __name__ == "__main__":
    main()
