"""
Command Line Interface (CLI) for the NLSS OpenFOAM AI Simulation Accelerator.
Commands:
  - generate: Generate CFD simulation dataset and OpenFOAM benchmark case
  - train: Train FNO-2D or ResUNet surrogate on simulation data
  - rollout: Autoregressively roll out AI predictions and benchmark speedup
  - compare: Plot field comparisons and error trajectories
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import torch

from .dataset.generator import NavierStokes2DSimulator, generate_and_export_openfoam_benchmark
from .dataset.foam_dataset import CFDTrajectoryDataset, FieldNormalizer, load_dataset_from_openfoam_case
from .dataset.online_dataset import print_datasets_table, fetch_openfoam_github_case
from .models.fno import FNO2d
from .models.unet import ResUNetCFD
from .train import create_model, train_surrogate_model
from .rollout.engine import RolloutEngine, benchmark_acceleration
from .visualization.plotter import plot_field_comparison, plot_rollout_metrics


def cmd_generate(args):
    print(f"🌊 Generating OpenFOAM simulation benchmark in: {args.out_dir}")
    case_path = generate_and_export_openfoam_benchmark(
        output_case_dir=args.out_dir,
        nx=args.nx,
        ny=args.ny,
        n_steps=args.n_steps,
        save_interval=args.save_interval,
        nu=args.nu,
        dt=args.dt,
        u_lid=args.u_lid,
    )
    print(f"✨ Successfully generated OpenFOAM case at: {case_path}")


def cmd_train(args):
    print(f"🧠 Training AI CFD Surrogate model: {args.model.upper()}")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    case_path = Path(args.case_dir)
    if not case_path.exists():
        print(f"Error: case directory '{args.case_dir}' not found.")
        sys.exit(1)

    dataset, normalizer = load_dataset_from_openfoam_case(
        case_dir=case_path,
        grid_shape=(args.ny, args.nx),
        rollout_steps=args.rollout_steps,
    )

    # Train / Val split (85/15)
    n_total = len(dataset)
    n_train = max(1, int(0.85 * n_total))
    train_data = dataset.traj_list[0][:n_train + args.rollout_steps]
    val_data = dataset.traj_list[0][n_train:]

    train_ds = CFDTrajectoryDataset(train_data, rollout_steps=args.rollout_steps, normalizer=normalizer)
    val_ds = (
        CFDTrajectoryDataset(val_data, rollout_steps=args.rollout_steps, normalizer=normalizer)
        if len(val_data) > args.rollout_steps + 1
        else None
    )

    model = create_model(model_type=args.model, in_channels=3, out_channels=3)
    train_surrogate_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
    )


def cmd_benchmark(args):
    print("⚡ Benchmarking AI Acceleration vs OpenFOAM CFD Simulation...")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt_file = Path(args.checkpoint)
    if not ckpt_file.exists():
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        sys.exit(1)

    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    normalizer = FieldNormalizer.from_dict(ckpt["normalizer"])

    model = create_model(model_type=args.model, in_channels=3, out_channels=3)
    model.load_state_dict(ckpt["model_state_dict"])

    # Load dataset
    dataset, _ = load_dataset_from_openfoam_case(
        case_dir=args.case_dir,
        grid_shape=(args.ny, args.nx),
    )

    # Create Rollout Engine
    engine = RolloutEngine(model=model, normalizer=normalizer, device=device)

    # Benchmark against Ground Truth
    gt_traj = dataset.data  # unnormalized (T, 3, H, W)
    cfd_sim_time = args.cfd_time_sec

    results = benchmark_acceleration(
        rollout_engine=engine,
        ground_truth_trajectory=gt_traj,
        cfd_solver_time_sec=cfd_sim_time,
        num_steps=args.steps,
    )

    print("\n" + "=" * 55)
    print("         NLSS CFD ACCELERATION BENCHMARK RESULTS")
    print("=" * 55)
    print(f"• Rollout Steps:             {results['num_steps']}")
    print(f"• CFD Numerical Solver Time: {results['cfd_solver_time_sec']:.4f} s ({results['time_per_step_cfd_ms']:.2f} ms/step)")
    print(f"• AI Surrogate Model Time:   {results['ai_rollout_time_sec']:.4f} s ({results['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• Acceleration Speedup:      \033[1;32m{results['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Relative L2 Error:    {results['mean_relative_l2']:.4e}")
    print(f"• Mean Velocity/p RMSE:      {results['mean_rmse']:.4e}")
    print("=" * 55 + "\n")

    if args.plot_out:
        plot_rollout_metrics(results, save_path=args.plot_out)
        print(f"📊 Saved metric plot to: {args.plot_out}")

    if args.export_foam_dir:
        pred_traj, _ = engine.rollout(gt_traj[0], num_steps=results["num_steps"])
        times = [i * 0.005 for i in range(results["num_steps"] + 1)]
        out_foam = engine.export_rollout_to_openfoam(pred_traj, times, args.export_foam_dir)
        print(f"💾 Exported AI predictions to OpenFOAM case at: {out_foam}")


def cmd_list_datasets(args):
    print_datasets_table()


def cmd_fetch_tutorial(args):
    fetch_openfoam_github_case(
        solver=args.solver,
        case_name=args.case_name,
        output_dir=args.out_dir,
        version=args.version,
    )


def main():
    parser = argparse.ArgumentParser(
        description="NLSS: Neural Navier-Stokes Surrogate for Accelerating OpenFOAM CFD"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List datasets
    p_list = subparsers.add_parser("list-datasets", help="List online OpenFOAM & CFD datasets")
    p_list.set_defaults(func=cmd_list_datasets)

    # Fetch tutorial
    p_fetch = subparsers.add_parser("fetch-case", help="Fetch OpenFOAM tutorial case from GitHub")
    p_fetch.add_argument("--solver", type=str, default="incompressible/icoFoam")
    p_fetch.add_argument("--case-name", type=str, default="cavity/cavity")
    p_fetch.add_argument("--out-dir", type=str, default="data/openfoam_cavity_tutorial")
    p_fetch.add_argument("--version", type=str, default="10")
    p_fetch.set_defaults(func=cmd_fetch_tutorial)

    # Generate
    p_gen = subparsers.add_parser("generate", help="Generate CFD simulation dataset")
    p_gen.add_argument("--out-dir", type=str, default="data/openfoam_lid_cavity")
    p_gen.add_argument("--nx", type=int, default=64)
    p_gen.add_argument("--ny", type=int, default=64)
    p_gen.add_argument("--n-steps", type=int, default=200)
    p_gen.add_argument("--save-interval", type=int, default=2)
    p_gen.add_argument("--nu", type=float, default=0.01)
    p_gen.add_argument("--dt", type=float, default=0.001)
    p_gen.add_argument("--u-lid", type=float, default=1.0)
    p_gen.set_defaults(func=cmd_generate)

    # Train
    p_train = subparsers.add_parser("train", help="Train CFD surrogate model")
    p_train.add_argument("--case-dir", type=str, default="data/openfoam_lid_cavity")
    p_train.add_argument("--model", type=str, choices=["fno", "unet", "dense"], default="dense")
    p_train.add_argument("--nx", type=int, default=64)
    p_train.add_argument("--ny", type=int, default=64)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--rollout-steps", type=int, default=2, help="Multi-step pushforward unrolling during training")
    p_train.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p_train.add_argument("--cpu", action="store_true")
    p_train.set_defaults(func=cmd_train)

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Benchmark speedup and accuracy")
    p_bench.add_argument("--case-dir", type=str, default="data/openfoam_lid_cavity")
    p_bench.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt")
    p_bench.add_argument("--model", type=str, choices=["fno", "unet", "dense"], default="dense")
    p_bench.add_argument("--nx", type=int, default=64)
    p_bench.add_argument("--ny", type=int, default=64)
    p_bench.add_argument("--steps", type=int, default=50)
    p_bench.add_argument("--cfd-time-sec", type=float, default=2.5)
    p_bench.add_argument("--plot-out", type=str, default="reports/speedup_curve.png")
    p_bench.add_argument("--export-foam-dir", type=str, default="data/ai_accelerated_case")
    p_bench.add_argument("--cpu", action="store_true")
    p_bench.set_defaults(func=cmd_benchmark)

    parsed = parser.parse_args()
    parsed.func(parsed)


if __name__ == "__main__":
    main()
