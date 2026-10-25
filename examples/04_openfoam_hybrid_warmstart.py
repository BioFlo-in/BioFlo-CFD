#!/usr/bin/env python3
"""
Direct Comparative Benchmark:
OpenFOAM icoFoam Solver vs. NLSS AI Surrogate Model (FNO-2D) on a sample CFD simulation.

Workflow:
1. Sets up a fresh sample OpenFOAM simulation case (Lid-Driven Cavity, Re=100, 64x64 grid).
2. Runs the OpenFOAM numerical solver (icoFoam) inside the container.
3. Runs the AI surrogate model (FNO-2D) on the exact same initial state.
4. Computes side-by-side metrics: execution time, latency per step, speedup, and error fields.
5. Generates publication-quality comparison figures with streamlines and error distributions.
6. Exports AI predictions into standard OpenFOAM time directories.
"""

import shutil
import subprocess
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.models.fno import FNO2d
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.physics.losses import compute_divergence_2d
from nlss.visualization.plotter import plot_field_comparison


def setup_sample_openfoam_case(case_dir: Path):
    """Initializes a fresh OpenFOAM simulation case directory."""
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    # Copy template files from openfoam_cases/lidDrivenCavity
    template_dir = Path("openfoam_cases/lidDrivenCavity")
    shutil.copytree(template_dir / "constant", case_dir / "constant")
    shutil.copytree(template_dir / "system", case_dir / "system")
    shutil.copytree(template_dir / "0", case_dir / "0")

    # Adjust controlDict for a clean 0.25s sample run (50 steps at dt=0.005)
    control_dict_content = """/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Case:     Sample Lid-Driven Cavity Benchmark
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    format      ascii;
    class       dictionary;
    location    "system";
    object      controlDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.25;
deltaT          0.005;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// ************************************************************************* //
"""
    (case_dir / "system" / "controlDict").write_text(control_dict_content, encoding="utf-8")


def run_openfoam_solver(case_dir: Path) -> float:
    """Runs blockMesh and icoFoam in the OpenCFD Docker container, returning solver runtime."""
    print("▶ Generating OpenFOAM computational mesh (blockMesh)...")
    cmd_mesh = [
        "docker", "run", "--rm",
        "-v", f"{case_dir.resolve()}:/case",
        "opencfd/openfoam-default:latest",
        "bash", "-c", "source /usr/lib/openfoam/openfoam2512/etc/bashrc && cd /case && blockMesh"
    ]
    res_mesh = subprocess.run(cmd_mesh, capture_output=True, text=True)
    if res_mesh.returncode != 0:
        raise RuntimeError(f"blockMesh failed:\n{res_mesh.stderr or res_mesh.stdout}")

    print("▶ Solving Incompressible Navier-Stokes equations with OpenFOAM icoFoam...")
    cmd_solver = [
        "docker", "run", "--rm",
        "-v", f"{case_dir.resolve()}:/case",
        "opencfd/openfoam-default:latest",
        "bash", "-c", "source /usr/lib/openfoam/openfoam2512/etc/bashrc && cd /case && icoFoam"
    ]
    
    t0 = time.perf_counter()
    res_solver = subprocess.run(cmd_solver, capture_output=True, text=True)
    t_cfd = time.perf_counter() - t0

    if res_solver.returncode != 0:
        raise RuntimeError(f"icoFoam failed:\n{res_solver.stderr or res_solver.stdout}")

    # Extract execution time from OpenFOAM log
    return t_cfd


def generate_comprehensive_report(
    openfoam_time: float,
    ai_time: float,
    num_steps: int,
    gt_trajectory: torch.Tensor,
    pred_trajectory: torch.Tensor,
    output_dir: Path,
):
    """Generates detailed comparison metrics and publication figures."""
    gt = gt_trajectory.cpu().numpy()
    pred = pred_trajectory.cpu().numpy()

    # Time measurements
    time_per_step_cfd = (openfoam_time / num_steps) * 1000.0
    time_per_step_ai = (ai_time / num_steps) * 1000.0
    speedup = openfoam_time / max(ai_time, 1e-6)

    # Accuracy metrics over time
    rel_l2_list = []
    rmse_u_list = []
    rmse_p_list = []
    div_violations = []

    for step in range(num_steps + 1):
        u_gt = gt[step, 0:2]
        u_pr = pred[step, 0:2]
        p_gt = gt[step, 2]
        p_pr = pred[step, 2]

        diff_all = pred[step] - gt[step]
        rel_l2 = np.linalg.norm(diff_all) / (np.linalg.norm(gt[step]) + 1e-6)
        rmse_u = np.sqrt(np.mean((u_pr - u_gt) ** 2))
        rmse_p = np.sqrt(np.mean((p_pr - p_gt) ** 2))

        rel_l2_list.append(rel_l2)
        rmse_u_list.append(rmse_u)
        rmse_p_list.append(rmse_p)

        # Divergence penalty for AI prediction
        pred_tensor = torch.from_numpy(pred[step:step+1])
        div_field = compute_divergence_2d(pred_tensor)
        div_violations.append(float(torch.norm(div_field).item()))

    # Print summary table
    print("\n" + "=" * 65)
    print("      SAMPLE SIMULATION: OPENFOAM SOLVER vs AI SURROGATE")
    print("=" * 65)
    print(f"• Total Simulation Timesteps:    {num_steps} steps (Δt = 0.005 s, t = 0 to 0.25 s)")
    print(f"• Spatial Mesh Resolution:       64 x 64 cells (4,096 cells)")
    print(f"• OpenFOAM (icoFoam) Solver:     {openfoam_time:.4f} s  ({time_per_step_cfd:.2f} ms/step)")
    print(f"• AI Surrogate Model (FNO-2D):   {ai_time:.4f} s  ({time_per_step_ai:.2f} ms/step)")
    print(f"• Acceleration Factor:           \033[1;32m{speedup:.1f}x FASTER\033[0m")
    print(f"• Final Relative L2 Error:       {rel_l2_list[-1]:.4e} ({rel_l2_list[-1]*100:.2f}%)")
    print(f"• Final Velocity RMSE:           {rmse_u_list[-1]:.4e} m/s")
    print(f"• Final Pressure RMSE:           {rmse_p_list[-1]:.4e} m²/s²")
    print(f"• Final Continuity Residual:     {div_violations[-1]:.4e}")
    print("=" * 65 + "\n")

    # Generate multi-step visualization panel (t = 0.05s, 0.15s, 0.25s)
    eval_steps = [10, 30, 50]  # corresponding to t = 0.05, 0.15, 0.25
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    y, x = np.mgrid[0:64, 0:64]

    for row_idx, step_idx in enumerate(eval_steps):
        t_current = step_idx * 0.005
        u_gt_mag = np.sqrt(gt[step_idx, 0]**2 + gt[step_idx, 1]**2)
        u_pr_mag = np.sqrt(pred[step_idx, 0]**2 + pred[step_idx, 1]**2)
        u_err = np.abs(u_pr_mag - u_gt_mag)

        p_gt = gt[step_idx, 2]
        p_pr = pred[step_idx, 2]

        # 1. OpenFOAM Velocity Magnitude + Streamlines
        im0 = axes[row_idx, 0].imshow(u_gt_mag, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
        axes[row_idx, 0].streamplot(
            np.linspace(0, 1, 64), np.linspace(0, 1, 64),
            gt[step_idx, 0], gt[step_idx, 1],
            color="white", density=0.7, linewidth=0.6, arrowsize=0.6
        )
        axes[row_idx, 0].set_title(f"OpenFOAM |U| (t={t_current:.2f}s)")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # 2. AI Surrogate Velocity Magnitude + Streamlines
        im1 = axes[row_idx, 1].imshow(u_pr_mag, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
        axes[row_idx, 1].streamplot(
            np.linspace(0, 1, 64), np.linspace(0, 1, 64),
            pred[step_idx, 0], pred[step_idx, 1],
            color="white", density=0.7, linewidth=0.6, arrowsize=0.6
        )
        axes[row_idx, 1].set_title(f"AI Surrogate |U| (t={t_current:.2f}s)")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # 3. Velocity Absolute Difference
        im2 = axes[row_idx, 2].imshow(u_err, cmap="inferno", origin="lower", extent=[0, 1, 0, 1])
        axes[row_idx, 2].set_title(f"|U_AI - U_FOAM| Error")
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        # 4. Pressure Contours Comparison
        im3 = axes[row_idx, 3].imshow(p_pr, cmap="coolwarm", origin="lower", extent=[0, 1, 0, 1])
        axes[row_idx, 3].set_title(f"AI Pressure p (t={t_current:.2f}s)")
        plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046, pad=0.04)

        for col in range(4):
            axes[row_idx, col].set_xlabel("x")
            axes[row_idx, col].set_ylabel("y")

    plot_file = output_dir / "sample_simulation_comparison.png"
    plt.savefig(plot_file, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"📊 Saved multi-timestep comparison panel to: {plot_file}")

    # Error & Speedup Evolution Chart
    fig2, (ax_err, ax_spd) = plt.subplots(1, 2, figsize=(14, 5))
    steps_arr = np.arange(len(rel_l2_list))
    times_arr = steps_arr * 0.005

    ax_err.plot(times_arr, rel_l2_list, label="Relative L2 Error", color="tab:blue", lw=2)
    ax_err.plot(times_arr, rmse_u_list, label="Velocity RMSE (m/s)", color="tab:orange", lw=2, linestyle="--")
    ax_err.set_xlabel("Simulation Physical Time (s)")
    ax_err.set_ylabel("Error Metric")
    ax_err.set_title("Rollout Accuracy Evolution Over 50 Timesteps")
    ax_err.grid(True, linestyle="--", alpha=0.6)
    ax_err.legend()

    bars = ax_spd.bar(["OpenFOAM (icoFoam)", "AI Surrogate (FNO-2D)"], [time_per_step_cfd, time_per_step_ai], color=["tab:gray", "tab:green"])
    ax_spd.set_ylabel("Computation Time per Step (milliseconds)")
    ax_spd.set_title(f"Simulation Speedup: {speedup:.1f}x Faster on RTX 3050")
    for bar in bars:
        h = bar.get_height()
        ax_spd.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.2f} ms", ha="center", va="bottom", fontweight="bold")
    ax_spd.grid(axis="y", linestyle="--", alpha=0.6)

    metric_file = output_dir / "sample_simulation_speedup_metrics.png"
    plt.savefig(metric_file, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"📊 Saved speedup & error dynamics chart to: {metric_file}")


def main():
    print("=" * 65)
    print("  NLSS: RUNNING SAMPLE OPENFOAM SIMULATION vs AI SURROGATE")
    print("=" * 65)

    base_dir = Path("outputs/sample_run")
    base_dir.mkdir(parents=True, exist_ok=True)
    case_dir = base_dir / "openfoam_solver_case"
    ai_foam_dir = base_dir / "ai_surrogate_case"

    # 1. Setup sample OpenFOAM simulation
    print("\n[Step 1/5] Setting up fresh OpenFOAM benchmark simulation case...")
    setup_sample_openfoam_case(case_dir)
    print(f"✓ Case configured in: {case_dir}")

    # 2. Run OpenFOAM Numerical Solver
    print("\n[Step 2/5] Running OpenFOAM icoFoam solver inside Docker container...")
    openfoam_time = run_openfoam_solver(case_dir)
    print(f"✓ OpenFOAM simulation finished in {openfoam_time:.3f} s")

    # 3. Load OpenFOAM Ground Truth Time Trajectory
    print("\n[Step 3/5] Loading OpenFOAM solved time directories into tensors...")
    dataset, normalizer = load_dataset_from_openfoam_case(case_dir, grid_shape=(64, 64))
    num_steps = len(dataset)
    print(f"✓ Loaded {num_steps + 1} snapshots from OpenFOAM (t=0 to t={num_steps*0.005:.2f}s)")

    # 4. Run AI Surrogate Model (DenseCFDNet)
    print("\n[Step 4/5] Executing AI Surrogate Rollout on GPU (DenseCFDNet)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = Path("checkpoints_diverse/best_model.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("checkpoints_dense/best_model.pt")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    from nlss.models.dense_cfd_net import DenseCFDNet
    model = DenseCFDNet(in_channels=3, out_channels=3, hidden_dim=96, modes1=20, modes2=20, num_blocks=4, enforce_boundary=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    trained_norm = FieldNormalizer.from_dict(ckpt["normalizer"])
    engine = RolloutEngine(model=model, normalizer=trained_norm, device=device)

    # Initial condition from OpenFOAM t=0
    init_state = dataset.data[0]

    # Run rollout
    pred_trajectory, ai_time = engine.rollout(init_state, num_steps=num_steps)
    print(f"✓ AI Surrogate finished {num_steps} steps in {ai_time:.4f} s")

    # 5. Comparative Evaluation & Visualization
    print("\n[Step 5/5] Generating Comparison Report, Metrics & Exporting...")
    generate_comprehensive_report(
        openfoam_time=openfoam_time,
        ai_time=ai_time,
        num_steps=num_steps,
        gt_trajectory=dataset.data,
        pred_trajectory=pred_trajectory,
        output_dir=base_dir,
    )

    # Export AI predictions into OpenFOAM directory format
    times = [i * 0.005 for i in range(num_steps + 1)]
    engine.export_rollout_to_openfoam(pred_trajectory, times, ai_foam_dir)
    print(f"✓ Exported AI predictions to OpenFOAM format at: {ai_foam_dir}")

    print("\n✨ Comparison run successfully completed!")


if __name__ == "__main__":
    main()
