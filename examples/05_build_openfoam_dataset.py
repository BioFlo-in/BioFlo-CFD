#!/usr/bin/env python3
"""
Comprehensive Builder for Large-Scale Diverse OpenFOAM Multi-Source Dataset.

Discovers, resamples, standardizes, and packages CFD simulations across:
  1. Official OpenFOAM 2512 Tutorials (16 cases):
     - cavity, cavityClipped, cavityGrade (icoFoam)
     - pitzDaily, backwardFacingStep2D (simpleFoam separated shear layers)
     - airFoil2D (simpleFoam aerodynamic profile & wake)
     - elbow (icoFoam 90-degree turning duct)
     - mixerVessel2D (simpleFoam stirred tank)
     - pipeCyclic, rotatingCylinders, T3A (simpleFoam)
     - planarContraction, TJunction, planarPoiseuille/results/Stokes (pimpleFoam)
  2. GitHub Repositories (20+ cases each):
     - OpenFOAM_Tutorials_Plus (21 cases: abruptContraction, flatPlate, magnusEffect, OFtutorial00-12, OF301)
     - BasicOpenFOAMProgrammingTutorials (34 cases: helloWorld, IO, mesh, fields, parallel, custom classes across OF versions)
     - NimaSam_OpenFOAM_Tutorials (22 cases: cavityInPipe & heatedCavity across OF3.0.1, 4, 5, 7, 8, 9, 10 + parametric sweeps)
     - openfoam-cases (cubic_cavity 3D recirculation)
  3. Wolf Dynamics & Holzmann CFD Multi-Regime Suites (15 cases):
     - Reynolds sweeps (Re = 20, 50, 150, 300, 500, 750, 1200, 2000, 3000)
     - Geometrical aspect ratios (shallow H/L=0.5, tall H/L=1.5, deep H/L=2.0)
     - Multi-lid shear cavities (opposing, co-rotating, asymmetric)
  4. RunAll Multi-Regime Suite (11 cases):
     - Re=100, Re=400, Re=1000, obstacle wake, Poiseuille channel, etc.
  5. Stirred-Tank Bioreactor (STR) Parameter Suite (5 cases):
     - Rushton turbine stirred tank across 5 distinct impeller tip speeds (0.5 to 2.5 m/s) and viscosities.

Strictly compiles, validates, and packages the dataset without running model training.
"""

import csv
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.dataset.multi_regime_suite import apply_so2_rotation_augmentation
from nlss.dataset.bioreactor import (
    BioreactorSimulator,
    BioreactorGeometry,
    compute_bioprocess_shear_metrics,
    export_simulation_state_to_openfoam,
)

WORKSPACE = Path.cwd()
OUTPUT_DIR = WORKSPACE / "outputs/massive_openfoam_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_bioreactor_suite():
    """Generates the 5-case parametric stirred-tank bioreactor suite if missing."""
    bio_dir = WORKSPACE / "openfoam_cases/bioreactor_suite"
    bio_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("str_slow_u05", 0.5, 0.001, 1000.0),
        ("str_medium_u10", 1.0, 0.001, 1000.0),
        ("str_fast_u175", 1.75, 0.001, 1000.0),
        ("str_intense_u25", 2.5, 0.001, 1000.0),
        ("str_viscous_u10", 1.0, 0.005, 1050.0),
    ]

    for name, tip_u, visc, dens in cases:
        case_p = bio_dir / name
        if (case_p / "0.25").exists():
            continue

        geom = BioreactorGeometry(diameter=0.2, height=0.3, impeller_diameter=0.07, liquid_volume_liters=7.5)
        sim = BioreactorSimulator(geom=geom, fluid_density=dens, dynamic_viscosity=visc, grid_res=(64, 64))
        u_arr, v_arr, p_arr = sim.simulate(impeller_tip_speed=tip_u, num_steps=50, dt=0.005)

        for t_step in range(u_arr.shape[0]):
            time_val = round(t_step * 0.005, 4)
            u_vec = np.column_stack([u_arr[t_step].flatten(), v_arr[t_step].flatten()])
            p_scalar = p_arr[t_step].flatten()
            export_simulation_state_to_openfoam(
                case_dir=case_p,
                time_val=time_val,
                u_pred=u_vec,
                p_pred=p_scalar,
            )


def has_solved_snapshots(case_dir: Path) -> int:
    """Returns number of numerical time step directories containing U or U.gz."""
    if not case_dir.is_dir() or not (case_dir / "0").exists():
        return 0
    snaps = 0
    for d in case_dir.iterdir():
        if d.is_dir() and re.match(r"^\d+(\.\d+)?$", d.name):
            if (d / "U").exists() or (d / "U.gz").exists():
                snaps += 1
    return snaps


def catalog_all_sources() -> List[Tuple[str, str, str, Path]]:
    """
    Discovers all verified and solved OpenFOAM cases across all repositories and suites.
    Returns: List of (source_category, case_unique_id, description, path)
    """
    cases = []

    # 1. Official OpenFOAM Tutorials
    off_dir = WORKSPACE / "external_openfoam_sources/official_tutorials"
    if off_dir.exists():
        for cd in sorted(off_dir.glob("**/system/controlDict")):
            c = cd.parent.parent
            if has_solved_snapshots(c) >= 2:
                name = str(c.relative_to(off_dir)).replace("/", "_")
                desc = f"Official {c.name}"
                cases.append(("Official OpenFOAM Tutorials", name, desc, c))

    # 2. OpenFOAM_Tutorials_Plus (>= 20 cases)
    plus_dir = WORKSPACE / "external_openfoam_sources/OpenFOAM_Tutorials_Plus"
    if plus_dir.exists():
        for cd in sorted(plus_dir.glob("**/system/controlDict")):
            c = cd.parent.parent
            if has_solved_snapshots(c) >= 2:
                rel = c.relative_to(plus_dir)
                name = str(rel).replace("/testCase", "").replace("/", "_")
                desc = f"Tutorials Plus: {name}"
                cases.append(("GitHub OpenFOAM_Tutorials_Plus", name, desc, c))

    # 3. BasicOpenFOAMProgrammingTutorials (>= 20 cases)
    basic_dir = WORKSPACE / "external_openfoam_sources/BasicOpenFOAMProgrammingTutorials"
    if basic_dir.exists():
        for cd in sorted(basic_dir.glob("**/system/controlDict")):
            c = cd.parent.parent
            if has_solved_snapshots(c) >= 2:
                rel = c.relative_to(basic_dir)
                name = str(rel).replace("/testCase", "").replace("/", "_")
                desc = f"Programming Tutorial: {name}"
                cases.append(("GitHub BasicOpenFOAMProgrammingTutorials", name, desc, c))

    # 4. NimaSam_OpenFOAM_Tutorials (>= 20 cases)
    nima_dir = WORKSPACE / "external_openfoam_sources/NimaSam_OpenFOAM_Tutorials"
    if nima_dir.exists():
        for cd in sorted(nima_dir.glob("**/system/controlDict")):
            c = cd.parent.parent
            if has_solved_snapshots(c) >= 2:
                rel = c.relative_to(nima_dir)
                name = str(rel).replace("/", "_")
                desc = f"NimaSam Tutorial: {name}"
                cases.append(("GitHub NimaSam_OpenFOAM_Tutorials", name, desc, c))

    # 5. openfoam-cases
    cases_dir = WORKSPACE / "external_openfoam_sources/openfoam-cases"
    if cases_dir.exists():
        for cd in sorted(cases_dir.glob("**/system/controlDict")):
            c = cd.parent.parent
            if has_solved_snapshots(c) >= 2:
                name = c.name
                desc = f"openfoam-cases {name}"
                cases.append(("GitHub openfoam-cases", name, desc, c))

    # 6. Wolf Dynamics & Holzmann Suite
    wolf_dir = WORKSPACE / "openfoam_cases/wolf_holzmann_suite"
    if wolf_dir.exists():
        for c in sorted(wolf_dir.iterdir()):
            if has_solved_snapshots(c) >= 2:
                cases.append(("Wolf Dynamics / Holzmann Suite", c.name, c.name.replace("_", " ").title(), c))

    # 7. RunAll Multi-Regime Suite
    runall_dir = WORKSPACE / "openfoam_cases/runall_suite"
    if runall_dir.exists():
        for c in sorted(runall_dir.iterdir()):
            if has_solved_snapshots(c) >= 2:
                cases.append(("RunAll Multi-Regime Suite", c.name, c.name.replace("_", " ").title(), c))

    # 8. Stirred-Tank Bioreactor Parameter Suite
    bio_dir = WORKSPACE / "openfoam_cases/bioreactor_suite"
    if bio_dir.exists():
        for c in sorted(bio_dir.iterdir()):
            if has_solved_snapshots(c) >= 2:
                cases.append(("Bioreactor STR Suite", c.name, c.name.replace("_", " ").title(), c))

    # 9. Speed Variations Suite (Systematic Velocity & Reynolds Range: 0.05 - 25.0 m/s)
    speed_dir = WORKSPACE / "openfoam_cases/speed_variations_suite"
    if speed_dir.exists():
        for c in sorted(speed_dir.iterdir()):
            if has_solved_snapshots(c) >= 2:
                cases.append(("Speed Variations Suite", c.name, c.name.replace("_", " ").title(), c))

    # 10. Surface Variables Suite (Slip, Partial-Slip, Multi-Wall Shear & Viscous Drag)
    surface_dir = WORKSPACE / "openfoam_cases/surface_variations_suite"
    if surface_dir.exists():
        for c in sorted(surface_dir.iterdir()):
            if has_solved_snapshots(c) >= 2:
                cases.append(("Surface Variables Suite", c.name, c.name.replace("_", " ").title(), c))

    return cases


def build_and_package_dataset(
    cases: List[Tuple[str, str, str, Path]],
    grid_shape: Tuple[int, int] = (64, 64),
    apply_symmetry: bool = True,
):
    """Ingests, standardizes, augmentations, and packages the complete dataset."""
    print("=" * 80)
    print("📦 INGESTING AND STANDARDIZING MULTI-SIMULATION DATASET")
    print(f"Target Grid Resolution: {grid_shape[0]}x{grid_shape[1]} cells")
    print(f"Symmetry Augmentations: SO(2) 4-Fold Discrete Rotations ({apply_symmetry})")
    print(f"Total Discovered Regime Candidates: {len(cases)}")
    print("=" * 80)

    all_inputs = []
    all_targets = []
    case_manifest = []
    raw_trajectories = {}

    total_raw_snapshots = 0

    for source_cat, case_id, desc, case_path in cases:
        try:
            ds, normalizer = load_dataset_from_openfoam_case(case_path, grid_shape=grid_shape, rollout_steps=1)
            raw_traj = ds.traj_list[0].numpy()  # (T, 3, H, W)
            T = raw_traj.shape[0]

            if T >= 2:
                total_raw_snapshots += T
                raw_trajectories[case_id] = raw_traj

                # Build transition pairs with SO(2) rotation symmetry
                if apply_symmetry:
                    aug_trajs = apply_so2_rotation_augmentation(raw_traj)
                else:
                    aug_trajs = [raw_traj]

                case_pairs = 0
                for a_traj in aug_trajs:
                    all_inputs.append(a_traj[:-1])
                    all_targets.append(a_traj[1:])
                    case_pairs += a_traj.shape[0] - 1

                # Calculate velocity magnitude and pressure stats
                u_mag = np.hypot(raw_traj[:, 0], raw_traj[:, 1])
                p_field = raw_traj[:, 2]

                case_info = {
                    "source": source_cat,
                    "case_id": case_id,
                    "case_name": case_path.name,
                    "description": desc,
                    "path": str(case_path.resolve().relative_to(WORKSPACE.resolve())),
                    "snapshots": int(T),
                    "augmented_pairs": int(case_pairs),
                    "grid_h": int(grid_shape[0]),
                    "grid_w": int(grid_shape[1]),
                    "u_mean": float(np.mean(u_mag)),
                    "u_max": float(np.max(u_mag)),
                    "p_min": float(np.min(p_field)),
                    "p_max": float(np.max(p_field)),
                }
                case_manifest.append(case_info)
                print(f"  ✓ [{source_cat:<38}] {case_id:<32}: {T:3d} snaps -> {case_pairs:4d} pairs | |U|_max={np.max(u_mag):.2f}")
        except Exception as e:
            print(f"  ⚠ [{source_cat:<38}] {case_id:<32}: SKIPPED ({e})")

    master_inputs = torch.from_numpy(np.concatenate(all_inputs, axis=0)).float()
    master_targets = torch.from_numpy(np.concatenate(all_targets, axis=0)).float()

    # Fit field normalizer
    field_normalizer = FieldNormalizer()
    field_normalizer.fit(master_inputs)

    dataset_package = {
        "inputs": master_inputs,
        "targets": master_targets,
        "normalizer_mean": field_normalizer.mean,
        "normalizer_std": field_normalizer.std,
        "raw_trajectories": raw_trajectories,
        "summary": {
            "total_regimes": len(case_manifest),
            "total_raw_snapshots": total_raw_snapshots,
            "total_augmented_pairs": master_inputs.shape[0],
            "total_physical_cell_evaluations": master_inputs.shape[0] * grid_shape[0] * grid_shape[1],
            "channels": ["u_x", "u_y", "p"],
        }
    }

    # Save package
    pkg_path = OUTPUT_DIR / "massive_openfoam_dataset.pt"
    torch.save(dataset_package, pkg_path)
    print(f"\n  ✓ Successfully saved PyTorch dataset package to {pkg_path}")

    # Export Manifest to JSON and CSV
    manifest_json_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_json_path, "w") as f:
        json.dump(dataset_package["summary"] | {"cases": case_manifest}, f, indent=2)
    print(f"  ✓ Saved JSON manifest to {manifest_json_path}")

    manifest_csv_path = OUTPUT_DIR / "dataset_catalog.csv"
    with open(manifest_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_manifest[0].keys()))
        writer.writeheader()
        writer.writerows(case_manifest)
    print(f"  ✓ Saved CSV catalog to {manifest_csv_path}")

    return dataset_package, case_manifest, raw_trajectories


def plot_comprehensive_gallery(
    manifest: List[Dict],
    raw_trajectories: Dict[str, np.ndarray],
    output_path: Path,
):
    """Plots a 24-panel gallery representing cases from all 8 sources."""
    # Group manifest by source
    by_source: Dict[str, List[Dict]] = {}
    for m in manifest:
        by_source.setdefault(m["source"], []).append(m)

    selected_ids = []
    # Pick 3 representative cases from each source category (up to 24 total)
    for src, items in by_source.items():
        for item in items[:3]:
            if len(selected_ids) < 24:
                selected_ids.append(item["case_id"])

    # If fewer than 24, fill up from remaining
    for m in manifest:
        if len(selected_ids) >= 24:
            break
        if m["case_id"] not in selected_ids:
            selected_ids.append(m["case_id"])

    n_cols = 4
    n_rows = 6

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24))
    axes = axes.flatten()

    for idx, cid in enumerate(selected_ids):
        ax = axes[idx]
        traj = raw_trajectories[cid]
        state = traj[-1]  # final state
        u_mag = np.hypot(state[0], state[1])
        H, W = u_mag.shape
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)

        im = ax.imshow(u_mag, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
        try:
            ax.streamplot(x, y, state[0], state[1], color="white", density=0.7, linewidth=0.6, arrowsize=0.6)
        except Exception:
            pass

        meta = next(m for m in manifest if m["case_id"] == cid)
        short_src = meta["source"].replace("GitHub ", "").replace(" Tutorials", "")
        ax.set_title(f"[{short_src[:18]}]\n{meta['description'][:28]}", fontsize=8.0, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(selected_ids), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved 24-panel streamline gallery to {output_path}")


def plot_distribution_analysis(
    inputs: torch.Tensor,
    manifest: List[Dict],
    output_path: Path,
):
    """Plots statistical distributions of the dataset (velocity, pressure, regime breakdowns)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # 1. Regimes per source breakdown
    source_counts = {}
    for m in manifest:
        src = m["source"].replace("GitHub ", "GH: ")
        source_counts[src] = source_counts.get(src, 0) + 1

    axes[0, 0].bar(range(len(source_counts)), list(source_counts.values()), color="#1f77b4", edgecolor="black")
    axes[0, 0].set_xticks(range(len(source_counts)))
    axes[0, 0].set_xticklabels([s.replace(" ", "\n") for s in source_counts.keys()], fontsize=6.8, rotation=25)
    axes[0, 0].set_ylabel("Number of Simulations", fontweight="bold")
    axes[0, 0].set_title(f"Simulation Source Distribution ({sum(source_counts.values())} Total Regimes)", fontweight="bold")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Augmented pairs per source
    pair_counts = {}
    for m in manifest:
        src = m["source"].replace("GitHub ", "GH: ")
        pair_counts[src] = pair_counts.get(src, 0) + m["augmented_pairs"]

    axes[0, 1].bar(range(len(pair_counts)), list(pair_counts.values()), color="#2ca02c", edgecolor="black")
    axes[0, 1].set_xticks(range(len(pair_counts)))
    axes[0, 1].set_xticklabels([s.replace(" ", "\n") for s in pair_counts.keys()], fontsize=6.8, rotation=25)
    axes[0, 1].set_ylabel("Transition Pairs (S_t, S_{t+1})", fontweight="bold")
    axes[0, 1].set_title(f"Augmented Transition Pairs ({sum(pair_counts.values()):,} Total Pairs)", fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.5)

    # 3. Velocity Magnitude Distribution
    n_sample = min(inputs.shape[0], 4000)
    u_mag_sample = torch.hypot(inputs[:n_sample, 0], inputs[:n_sample, 1]).flatten().numpy()
    axes[1, 0].hist(u_mag_sample, bins=50, color="#ff7f0e", edgecolor="black", alpha=0.8, density=True)
    axes[1, 0].set_xlabel("Velocity Magnitude |U| (m/s)", fontweight="bold")
    axes[1, 0].set_ylabel("Probability Density", fontweight="bold")
    axes[1, 0].set_title("Dataset Velocity Magnitude Distribution", fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 4. Pressure Field Distribution
    p_sample = inputs[:n_sample, 2].flatten().numpy()
    p_filtered = p_sample[np.abs(p_sample) < 10.0]  # filter extreme outliers for clear visualization
    axes[1, 1].hist(p_filtered, bins=50, color="#9467bd", edgecolor="black", alpha=0.8, density=True)
    axes[1, 1].set_xlabel("Kinematic Pressure p (m^2/s^2)", fontweight="bold")
    axes[1, 1].set_ylabel("Probability Density", fontweight="bold")
    axes[1, 1].set_title("Dataset Kinematic Pressure Distribution", fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved dataset distribution analysis to {output_path}")


def main():
    print("=" * 80)
    print("🚀 BUILDING MASSIVE OPENFOAM SIMULATION DATASET (NO TRAINING)")
    print("=" * 80)

    # Step 1: Ensure Bioreactor operating suite is ready
    print("\n[1/4] Ensuring Stirred-Tank Bioreactor Parameter Suite is generated...")
    ensure_bioreactor_suite()

    # Step 2: Catalog all sources
    print("\n[2/4] Cataloging all simulation cases across official, GitHub, Wolf, Holzmann, and bioreactor suites...")
    cases = catalog_all_sources()
    print(f"  • Discovered {len(cases)} potential candidate simulation cases.")

    # Step 3: Ingest, standardize, and package dataset
    print("\n[3/4] Compiling, interpolating to 64x64, applying SO(2) symmetry, and packaging...")
    dataset_pkg, manifest, raw_trajs = build_and_package_dataset(cases, grid_shape=(64, 64), apply_symmetry=True)

    # Step 4: Visual gallery & statistical distribution plots
    print("\n[4/4] Generating visual artifacts and statistical audit plots...")
    gallery_path = OUTPUT_DIR / "comprehensive_multi_source_gallery.png"
    plot_comprehensive_gallery(manifest, raw_trajs, gallery_path)

    dist_path = OUTPUT_DIR / "dataset_distribution_analysis.png"
    plot_distribution_analysis(dataset_pkg["inputs"], manifest, dist_path)

    summary = dataset_pkg["summary"]
    print("\n" + "=" * 80)
    print("🎉 DATASET BUILD COMPLETE (NO MODEL TRAINING PERFORMED)")
    print(f"  • Total Distinct Simulation Regimes: {summary['total_regimes']}")
    print(f"  • Total Raw Simulation Snapshots:    {summary['total_raw_snapshots']:,}")
    print(f"  • Total Augmented Transition Pairs:  {summary['total_augmented_pairs']:,}")
    print(f"  • Total Physical Cell Ingestion:     {summary['total_physical_cell_evaluations']:,} cells")
    print(f"  • Artifact Directory:                {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
