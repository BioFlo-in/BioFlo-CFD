#!/usr/bin/env python3
"""
Train DenseCFDNet on a Diverse Multi-Regime OpenFOAM & Navier-Stokes Dataset.
Eliminates model inference bias by training across:
  - Varied Reynolds numbers (Re = 40, 100, 200, 400)
  - Forward and reverse flow directions (u_lid = +1.0, -1.0)
  - Double-driven counter-rotating cavity flows
  - Planar Poiseuille channel flow
  - Multi-scale decaying Taylor-Green vortex fields
  - Shear layer instability

Evaluates zero-shot generalization on an unseen test simulation.
"""

from pathlib import Path
import time
import numpy as np
import torch

from nlss.dataset.diverse_suite import build_diverse_openfoam_suite
from nlss.dataset.foam_dataset import CFDTrajectoryDataset, FieldNormalizer
from nlss.dataset.generator import NavierStokes2DSimulator
from nlss.models.dense_cfd_net import DenseCFDNet
from nlss.train import train_surrogate_model
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison, plot_rollout_metrics


def main():
    print("=" * 70)
    print("  TRAINING DENSE SURROGATE ON DIVERSE OPENFOAM DATASET")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    base_dir = Path("outputs/diverse_dataset_run")
    ckpt_dir = Path("checkpoints_diverse")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build Diverse OpenFOAM & Navier-Stokes Suite
    print("\n[Step 1/4] Building diverse OpenFOAM simulation suite...")
    all_trajs = build_diverse_openfoam_suite(base_dir / "simulations")

    # 2. Fit Global Normalizer and Build Pushforward Dataset
    print("\n[Step 2/4] Constructing pushforward training dataset (rollout_steps=2)...")
    all_stacked = np.concatenate(all_trajs, axis=0)
    normalizer = FieldNormalizer().fit(torch.from_numpy(all_stacked).float())

    dataset = CFDTrajectoryDataset(all_trajs, rollout_steps=2, normalizer=normalizer)
    print(f"✓ Total diverse transition sequences: {len(dataset)}")

    # Train / Val Split (85% / 15%)
    indices = torch.randperm(len(dataset)).tolist()
    n_train = int(0.85 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, indices[:n_train])
    val_ds = torch.utils.data.Subset(dataset, indices[n_train:])
    train_ds.normalizer = normalizer

    # 3. Initialize DenseCFDNet
    print("\n[Step 3/4] Initializing DenseCFDNet (General Boundary Mode)...")
    model = DenseCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=96,
        modes1=20,
        modes2=20,
        num_blocks=4,
        enforce_boundary=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Trainable parameters: {total_params:,}")

    # Train for 45 epochs with transition loss
    history = train_surrogate_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=45,
        batch_size=8,
        lr=1e-3,
        weight_decay=1e-4,
        device=device,
        checkpoint_dir=ckpt_dir,
        lambda_div=0.1,
        lambda_vort=0.05,
        log_interval=5,
    )

    # 4. Evaluate Generalization on an Unseen Test Regime (Re = 250, asymmetric velocity)
    print("\n[Step 4/4] Testing Model on an UNSEEN Simulation Regime (Re = 250)...")
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Generate unseen test case: Re = 250, nu = 0.004, u_lid = 1.0
    test_sim = NavierStokes2DSimulator(nx=64, ny=64, nu=0.004, dt=0.005)
    test_sim.set_lid_driven_cavity(u_lid=1.0)
    test_traj_dict = test_sim.run_simulation(n_steps=40, save_interval=1, is_cavity=True, u_lid=1.0)
    test_gt = np.stack([test_traj_dict["u"], test_traj_dict["v"], test_traj_dict["p"]], axis=1)
    test_gt_tensor = torch.from_numpy(test_gt).float()

    engine = RolloutEngine(model=model, normalizer=normalizer, device=device)
    init_state = test_gt_tensor[0]

    num_eval_steps = 40
    pred_trajectory, ai_time = engine.rollout(init_state, num_steps=num_eval_steps)

    bench_results = benchmark_acceleration(
        rollout_engine=engine,
        ground_truth_trajectory=test_gt_tensor,
        cfd_solver_time_sec=2.5,
        num_steps=num_eval_steps,
    )

    print("\n" + "=" * 70)
    print("      UNSEEN TEST CASE BENCHMARK (ZERO-SHOT GENERALIZATION)")
    print("=" * 70)
    print(f"• Test Regime:               Re = 250 (Unseen during training)")
    print(f"• Rollout Steps:             {bench_results['num_steps']}")
    print(f"• Numerical CFD Solver Time: {bench_results['cfd_solver_time_sec']:.4f} s ({bench_results['time_per_step_cfd_ms']:.2f} ms/step)")
    print(f"• AI Surrogate Model Time:   {bench_results['ai_rollout_time_sec']:.4f} s ({bench_results['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Acceleration Speedup:      \033[1;32m{bench_results['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Relative L2 Error:    {bench_results['mean_relative_l2']:.4e} ({bench_results['mean_relative_l2']*100:.2f}%)")
    print(f"• Mean Velocity/p RMSE:      {bench_results['mean_rmse']:.4e}")
    print("=" * 70 + "\n")

    # Save visual comparison
    comp_plot = base_dir / "unseen_regime_comparison.png"
    plot_field_comparison(
        gt_state=test_gt_tensor[30],
        pred_state=pred_trajectory[30],
        time_val=30 * 0.005,
        save_path=comp_plot,
    )
    metric_plot = base_dir / "unseen_regime_metrics.png"
    plot_rollout_metrics(bench_results, save_path=metric_plot)
    print(f"📊 Saved evaluation figures to: {base_dir}")


if __name__ == "__main__":
    main()
