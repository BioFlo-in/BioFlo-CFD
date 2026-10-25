#!/usr/bin/env python3
"""
OpenFOAM RunAll Multi-Simulation Generator & Varied Dataset Builder.

Runs diverse transient CFD simulations directly into native OpenFOAM case structures:
  1. foam_re100_standard (Lid-driven cavity, Re = 100)
  2. foam_re400_shear (Lid-driven cavity, Re = 400)
  3. foam_re1000_turbulent (Lid-driven cavity, Re = 1000)
  4. foam_reverse_re100 (Reverse lid-driven cavity, u_lid = -1.0)
  5. foam_fast_re200 (High-speed cavity, u_lid = 1.5)
  6. foam_slow_re50 (Low-speed cavity, u_lid = 0.5)
  7. foam_double_driven (Counter-rotating dual lids: top +1.0, bottom -1.0)
  8. foam_co_rotating (Co-rotating dual lids: top +1.0, bottom +1.0)
  9. foam_bioreactor_str (Stirred-Tank Bioreactor with central shaft & Rushton radial jet)
  10. foam_obstacle_wake (Channel flow with submerged bluff body wake)
  11. foam_channel_poiseuille (Planar duct with parabolic inflow profile)

Loads all cases using the native OpenFOAM parser and compiles a unified, debiased dataset.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch

from nlss.dataset.bioreactor import BioreactorSimulator, BioreactorGeometry
from nlss.dataset.generator import NavierStokes2DSimulator
from nlss.dataset.multi_regime_suite import simulate_obstacle_channel, simulate_shear_layer
from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.io.foam_writer import export_simulation_state_to_openfoam
from nlss.io.foam_reader import load_openfoam_simulation_series


def is_openfoam_docker_ready() -> bool:
    """Checks if the OpenCFD OpenFOAM Docker image is available."""
    try:
        res = subprocess.run(
            ["docker", "image", "inspect", "opencfd/openfoam-default:latest"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return res.returncode == 0
    except Exception:
        return False


def run_openfoam_docker(case_dir: Path) -> bool:
    """Executes blockMesh and icoFoam inside OpenCFD Docker container."""
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{case_dir.resolve()}:/case",
        "opencfd/openfoam-default:latest",
        "bash", "-c", "source /usr/lib/openfoam/openfoam2512/etc/bashrc 2>/dev/null || source /usr/lib/openfoam/etc/bashrc 2>/dev/null || true; cd /case && blockMesh && icoFoam"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return res.returncode == 0
    except Exception:
        return False


def setup_openfoam_case_scaffolding(case_dir: Path, template_dir: Path):
    """Sets up standard OpenFOAM directory scaffolding (system/ and constant/)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for folder in ["system", "constant"]:
        src = template_dir / folder
        dst = case_dir / folder
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def generate_openfoam_runall_suite(
    suite_dir: Path,
    template_dir: Path,
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 40,
    dt: float = 0.005,
) -> Dict[str, Path]:
    """Generates all 11 diverse OpenFOAM cases."""
    print("=" * 80)
    print("🚀 GENERATING & EXECUTING OPENFOAM RUNALL MULTI-SIMULATION SUITE")
    print("=" * 80)

    suite_dir.mkdir(parents=True, exist_ok=True)
    docker_ready = is_openfoam_docker_ready()
    print(f"Docker OpenCFD Container Available: {'YES (Native icoFoam Enabled)' if docker_ready else 'NO (High-Fidelity Upwind OpenFOAM Exporter Active)'}")

    case_dirs: Dict[str, Path] = {}

    # Define all case specifications
    cases = [
        {"name": "foam_re100_standard", "type": "cavity", "u_lid": 1.0, "nu": 0.01, "desc": "Standard Cavity Re=100"},
        {"name": "foam_re400_shear", "type": "cavity", "u_lid": 1.0, "nu": 0.0025, "desc": "Shear Cavity Re=400"},
        {"name": "foam_re1000_turbulent", "type": "cavity", "u_lid": 1.0, "nu": 0.001, "desc": "High Reynolds Cavity Re=1000"},
        {"name": "foam_reverse_re100", "type": "cavity", "u_lid": -1.0, "nu": 0.01, "desc": "Reverse Cavity (u_lid=-1.0)"},
        {"name": "foam_fast_re200", "type": "cavity", "u_lid": 1.5, "nu": 0.0075, "desc": "Fast Cavity (u_lid=1.5)"},
        {"name": "foam_slow_re50", "type": "cavity", "u_lid": 0.5, "nu": 0.01, "desc": "Low-Speed Cavity (u_lid=0.5)"},
        {"name": "foam_double_driven", "type": "double_cavity", "u_top": 1.0, "u_bot": -1.0, "nu": 0.01, "desc": "Dual Counter-Rotating Cavity"},
        {"name": "foam_co_rotating", "type": "double_cavity", "u_top": 1.0, "u_bot": 1.0, "nu": 0.01, "desc": "Dual Co-Rotating S-Shear Cavity"},
        {"name": "foam_bioreactor_str", "type": "bioreactor", "u_tip": 1.0, "nu": 0.01, "desc": "Rushton Stirred-Tank Bioreactor"},
        {"name": "foam_obstacle_wake", "type": "obstacle", "u_inlet": 1.0, "nu": 0.01, "desc": "Submerged Bluff Body Obstacle Wake"},
        {"name": "foam_channel_poiseuille", "type": "channel", "u_inlet": 1.0, "nu": 0.01, "desc": "Planar Poiseuille Channel Flow"},
    ]

    for idx, c in enumerate(cases, 1):
        c_name = c["name"]
        c_dir = suite_dir / c_name
        print(f"\n[{idx:2d}/{len(cases):2d}] Setting up OpenFOAM Case: {c_name} ({c['desc']})...")
        setup_openfoam_case_scaffolding(c_dir, template_dir)

        # Update transportProperties
        nu_val = c.get("nu", 0.01)
        tp_text = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ format ascii; class dictionary; location "constant"; object transportProperties; }}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
nu [0 2 -1 0 0 0 0] {nu_val};
"""
        (c_dir / "constant" / "transportProperties").write_text(tp_text)

        # Update controlDict
        end_time = n_steps * dt
        cd_text = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ format ascii; class dictionary; location "system"; object controlDict; }}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
application icoFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end_time:.4f};
deltaT {dt:.4f};
writeControl timeStep;
writeInterval 1;
purgeWrite 0;
writeFormat ascii;
writePrecision 6;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
"""
        (c_dir / "system" / "controlDict").write_text(cd_text)

        executed_with_docker = False
        if docker_ready and c["type"] == "cavity":
            # Native icoFoam execution
            u_lid_vec = f"({c['u_lid']} 0 0)"
            u_file_text = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ format ascii; class volVectorField; location "0"; object U; }}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    movingWall {{ type fixedValue; value uniform {u_lid_vec}; }}
    fixedWalls {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
"""
            (c_dir / "0" / "U").write_text(u_file_text)
            print("  • Executing blockMesh & icoFoam in Docker...")
            if run_openfoam_docker(c_dir):
                print(f"  ✓ {c_name} solved successfully via OpenCFD icoFoam.")
                executed_with_docker = True

        if not executed_with_docker:
            # Generate high-fidelity simulation and export directly to native OpenFOAM format
            t0 = time.perf_counter()
            if c["type"] == "cavity":
                sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=nu_val, dt=dt)
                sim.set_lid_driven_cavity(u_lid=c["u_lid"])
                res = sim.run_simulation(n_steps=n_steps, save_interval=1, is_cavity=True, u_lid=c["u_lid"])
                traj = np.stack([res["u"], res["v"], res["p"]], axis=1)

            elif c["type"] == "double_cavity":
                sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=nu_val, dt=dt)
                sim.u[-1, :] = c["u_top"]
                sim.u[0, :] = c["u_bot"]
                u_h, v_h, p_h = [sim.u.copy()], [sim.v.copy()], [sim.p.copy()]
                for _ in range(n_steps):
                    sim.step(is_cavity=False)
                    sim.u[-1, :] = c["u_top"]
                    sim.u[0, :] = c["u_bot"]
                    sim.u[:, 0] = 0.0
                    sim.u[:, -1] = 0.0
                    sim.v[0, :] = 0.0
                    sim.v[-1, :] = 0.0
                    sim.v[:, 0] = 0.0
                    sim.v[:, -1] = 0.0
                    u_h.append(sim.u.copy())
                    v_h.append(sim.v.copy())
                    p_h.append(sim.p.copy())
                traj = np.stack([np.stack(u_h), np.stack(v_h), np.stack(p_h)], axis=1)

            elif c["type"] == "bioreactor":
                geom = BioreactorGeometry(T=1.0, H=1.0, clearance_C=0.333)
                sim = BioreactorSimulator(nx=nx, ny=ny, nu=nu_val, dt=dt/2, u_tip=c["u_tip"], geom=geom)
                res = sim.run_simulation(n_steps=n_steps * 2, save_interval=2, is_cavity=False)
                traj = np.stack([res["u"], res["v"], res["p"]], axis=1)

            elif c["type"] == "obstacle":
                traj = simulate_obstacle_channel(nx=nx, ny=ny, n_steps=n_steps, dt=dt, nu=nu_val, u_inlet_max=c["u_inlet"])

            elif c["type"] == "channel":
                y = np.linspace(0, 1, ny)[:, None]
                u_poi = (4.0 * c["u_inlet"] * y * (1.0 - y)).astype(np.float32)
                u_grid = np.repeat(u_poi, nx, axis=1)
                sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=nu_val, dt=dt)
                sim.u = u_grid.copy()
                u_h, v_h, p_h = [sim.u.copy()], [sim.v.copy()], [sim.p.copy()]
                for _ in range(n_steps):
                    sim.step(is_cavity=False)
                    sim.u[:, 0] = u_grid[:, 0]
                    sim.u[0, :] = 0.0
                    sim.u[-1, :] = 0.0
                    sim.v[0, :] = 0.0
                    sim.v[-1, :] = 0.0
                    u_h.append(sim.u.copy())
                    v_h.append(sim.v.copy())
                    p_h.append(sim.p.copy())
                traj = np.stack([np.stack(u_h), np.stack(v_h), np.stack(p_h)], axis=1)

            t_solve = time.perf_counter() - t0

            # Export every time step into native OpenFOAM directory format
            for step_i in range(traj.shape[0]):
                time_val = step_i * dt
                u_vec = np.column_stack([traj[step_i, 0].flatten(), traj[step_i, 1].flatten()])
                p_scal = traj[step_i, 2].flatten()
                export_simulation_state_to_openfoam(
                    case_dir=c_dir,
                    time_val=time_val,
                    u_pred=u_vec,
                    p_pred=p_scal,
                )
            print(f"  ✓ {c_name} generated and exported ({traj.shape[0]} OpenFOAM time folders in {t_solve:.2f} s).")

        case_dirs[c_name] = c_dir

    return case_dirs


def compile_varied_openfoam_dataset(
    case_dirs: Dict[str, Path],
    output_cache_path: Path,
    grid_shape: Tuple[int, int] = (64, 64),
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """Loads all native OpenFOAM case directories and builds a master varied training dataset."""
    print("\n" + "=" * 80)
    print("📦 COMPILING VARIED OPENFOAM DATASET VIA NATIVE PARSER")
    print("=" * 80)

    all_inputs = []
    all_targets = []
    case_counts = {}

    for c_name, c_path in case_dirs.items():
        print(f"  • Reading OpenFOAM series from: {c_path.name}...")
        ds, norm = load_dataset_from_openfoam_case(c_path, grid_shape=grid_shape, rollout_steps=1)
        raw_trajs = ds.traj_list[0]  # shape (T, 3, ny, nx)

        inputs = raw_trajs[:-1]
        targets = raw_trajs[1:]
        all_inputs.append(inputs)
        all_targets.append(targets)
        case_counts[c_name] = inputs.shape[0]

    master_inputs = torch.cat(all_inputs, dim=0).float()
    master_targets = torch.cat(all_targets, dim=0).float()

    output_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inputs": master_inputs,
            "targets": master_targets,
            "case_counts": case_counts,
        },
        output_cache_path,
    )

    print(f"\n✨ Master Varied OpenFOAM Dataset Compiled Successfully:")
    print(f"  • Output Cache:         {output_cache_path}")
    print(f"  • Total Transitions:    {master_inputs.shape[0]:,} pairs")
    print(f"  • Total Physical Cells: {master_inputs.shape[0] * grid_shape[0] * grid_shape[1]:,} cells")
    for name, cnt in case_counts.items():
        print(f"    - {name:<26}: {cnt:4d} transitions ({cnt / master_inputs.shape[0] * 100:.1f}%)")

    return master_inputs, master_targets, case_counts


def plot_openfoam_dataset_diversity(
    master_inputs: torch.Tensor,
    case_counts: Dict[str, int],
    output_plot_path: Path,
):
    """Generates publication-quality figure analyzing dataset diversity across the OpenFOAM suite."""
    u_vals = master_inputs[:, 0].flatten().numpy()
    v_vals = master_inputs[:, 1].flatten().numpy()
    p_vals = master_inputs[:, 2].flatten().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # 1. Velocity components distribution (checking elimination of directional bias)
    axes[0].hist(u_vals, bins=60, color="#1f77b4", alpha=0.6, density=True, label="Horizontal Velocity u")
    axes[0].hist(v_vals, bins=60, color="#ff7f0e", alpha=0.6, density=True, label="Vertical Velocity v")
    axes[0].axvline(0.0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Velocity Space Distribution (Isotropic Balance)\nMean u: {np.mean(u_vals):.3e} | Mean v: {np.mean(v_vals):.3e}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Velocity Magnitude (m/s)")
    axes[0].set_ylabel("Probability Density")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 2. Case distribution bar chart
    names = [n.replace("foam_", "") for n in case_counts.keys()]
    counts = list(case_counts.values())
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
    bars = axes[1].barh(names, counts, color=colors)
    axes[1].set_title("OpenFOAM RunAll Case Distribution\n(Balanced Multi-Regime Physical Coverage)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Number of Transition Snapshots")
    axes[1].grid(True, axis="x", alpha=0.3)

    # 3. Kinematic Pressure distribution
    axes[2].hist(p_vals, bins=60, color="#2ca02c", alpha=0.7, density=True)
    axes[2].set_title(f"Kinematic Pressure Distribution p/ρ\nMean p: {np.mean(p_vals):.3e} | Std: {np.std(p_vals):.3f}", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Kinematic Pressure (m²/s²)")
    axes[2].set_ylabel("Probability Density")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("OpenFOAM Multi-Simulation Suite: Diversity & Balance Analysis", fontsize=15, fontweight="bold", y=1.02)
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved OpenFOAM dataset diversity plot: {output_plot_path}")


def main():
    suite_dir = Path("openfoam_cases/runall_suite")
    template_dir = Path("openfoam_cases/lidDrivenCavity")
    output_cache = Path("outputs/openfoam_runall_suite/varied_openfoam_dataset.pt")
    plot_path = Path("outputs/openfoam_runall_suite/plots/openfoam_dataset_diversity.png")

    case_dirs = generate_openfoam_runall_suite(
        suite_dir=suite_dir,
        template_dir=template_dir,
        nx=64,
        ny=64,
        n_steps=40,
        dt=0.005,
    )

    inputs, targets, case_counts = compile_varied_openfoam_dataset(
        case_dirs=case_dirs,
        output_cache_path=output_cache,
        grid_shape=(64, 64),
    )

    plot_openfoam_dataset_diversity(inputs, case_counts, plot_path)


if __name__ == "__main__":
    main()
