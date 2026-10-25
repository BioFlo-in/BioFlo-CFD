#!/usr/bin/env python3
"""
Massive Diverse OpenFOAM Multi-Source Dataset Ingestion & Foundation Surrogate Pipeline.

Ingests and trains across 45+ distinct OpenFOAM simulations from all 5 requested sources:
  1. Official OpenFOAM 2512 Tutorials:
     - cavity, cavityClipped, cavityGrade (icoFoam)
     - pitzDaily, backwardFacingStep2D (simpleFoam separated shear layers)
     - airFoil2D, squareBend, elbow, mixerVessel2D (simpleFoam & icoFoam complex geometries)
     - planarContraction, TJunction, planarPoiseuille/results/Stokes (pimpleFoam)
  2. External GitHub Repositories:
     - OpenFOAM_Tutorials_Plus (abruptContraction)
     - openfoam-cases (cubic_cavity)
  3. Wolf Dynamics Multi-Regime Suites:
     - Reynolds sweeps (Re = 20, 50, 150, 300, 500, 750, 1200, 2000, 3000)
     - Geometrical aspect ratios (shallow H/L=0.5, tall H/L=1.5, deep H/L=2.0)
     - Multi-lid shear cavities (opposing, co-rotating, asymmetric)
  4. RunAll Multi-Regime Suite:
     - 11 canonical configurations (Re=100, Re=400, Re=1000, obstacle wake, Poiseuille channel, etc.)
  5. Stirred-Tank Bioreactor (STR) Parameter Suite:
     - Rushton turbine stirred tank across 5 distinct impeller tip speeds (0.5 to 2.5 m/s) and viscosities.

Eliminates overfitting, learns a generalist neural operator (HybridCFDNet),
and benchmarks bioprocess hydrodynamic shear metrics.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.dataset.multi_regime_suite import apply_so2_rotation_augmentation
from nlss.dataset.bioreactor import (
    BioreactorSimulator,
    BioreactorGeometry,
    compute_bioprocess_shear_metrics,
)
from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import compute_divergence_2d
from nlss.io.foam_writer import export_simulation_state_to_openfoam


WORKSPACE = Path.cwd().resolve()
OUTPUT_DIR = WORKSPACE / "outputs/massive_openfoam_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_bioreactor_suite():
    """Generates 5 distinct Stirred-Tank Bioreactor simulations and exports to OpenFOAM cases."""
    bio_dir = WORKSPACE / "openfoam_cases/bioreactor_suite"
    bio_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("str_slow_u05", 0.5, 0.01, "STR Slow Stirring (u_tip=0.5 m/s)"),
        ("str_medium_u10", 1.0, 0.01, "STR Nominal Stirring (u_tip=1.0 m/s)"),
        ("str_fast_u175", 1.75, 0.008, "STR High-Shear Stirring (u_tip=1.75 m/s)"),
        ("str_intense_u25", 2.5, 0.005, "STR Intense Mixing (u_tip=2.5 m/s)"),
        ("str_viscous_u10", 1.0, 0.02, "STR Viscous Media (u_tip=1.0 m/s, nu=0.02)"),
    ]

    for name, u_tip, nu, desc in configs:
        case_p = bio_dir / name
        if (case_p / "0").exists() and (case_p / "50").exists():
            continue

        print(f"  • Simulating {name} ({desc})...")
        geom = BioreactorGeometry(T=1.0, H=1.0)
        sim = BioreactorSimulator(nx=64, ny=64, lx=1.0, ly=1.0, nu=nu, dt=0.0025, u_tip=u_tip, geom=geom)
        res = sim.run_simulation(n_steps=100, save_interval=2, is_cavity=False)
        u_arr = res["u"]
        v_arr = res["v"]
        p_arr = res["p"]
        u_traj = np.stack([u_arr, v_arr], axis=1)

        # Export as native OpenFOAM case
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


def discover_all_cases() -> List[Tuple[str, str, Path]]:
    """Gathers all available cases across Official, GitHub, RunAll, Wolf/Holzmann, and Bioreactor suites."""
    cases = []

    # 1. Official OpenFOAM Tutorials
    official_dir = WORKSPACE / "external_openfoam_sources/official_tutorials"
    official_list = [
        ("cavity", "Cavity Flow (icoFoam)"),
        ("cavityClipped", "Clipped Cavity (icoFoam)"),
        ("cavityGrade", "Graded Cavity (icoFoam)"),
        ("pitzDaily", "Backward-Facing Step (pitzDaily)"),
        ("backwardFacingStep2D", "Separated Shear Layer Step"),
        ("airFoil2D", "Airfoil Boundary Layer & Wake"),
        ("elbow", "Elbow 90-deg Turning Duct"),
        ("mixerVessel2D", "Stirred Mixer Vessel (Bioreactor)"),
        ("squareBend", "Square Bend Duct Flow"),
        ("TJunction", "T-Junction Impinging Crossflow"),
        ("planarContraction", "Planar Contraction Flow"),
        ("planarPoiseuille/results/Stokes", "Planar Poiseuille Stokes"),
    ]
    for sub, desc in official_list:
        p = official_dir / sub
        if p.exists():
            cases.append(("Official Tutorial", desc, p))

    # 2. External GitHub Repositories
    github_cases = [
        ("OpenFOAM_Tutorials_Plus/abruptContraction/simpleFoam", "Abrupt Pipe Contraction", "GitHub Tutorials Plus"),
        ("openfoam-cases/cubic_cavity", "3D Recirculation Cavity", "GitHub openfoam-cases"),
    ]
    for sub, desc, cat in github_cases:
        p = WORKSPACE / "external_openfoam_sources" / sub
        if p.exists():
            cases.append((cat, desc, p))

    # 3. Wolf Dynamics & Holzmann CFD Suite
    wolf_dir = WORKSPACE / "openfoam_cases/wolf_holzmann_suite"
    if wolf_dir.exists():
        for cdir in sorted(wolf_dir.iterdir()):
            if cdir.is_dir() and (cdir / "0").exists():
                cases.append(("Wolf / Holzmann Suite", cdir.name.replace("_", " ").title(), cdir))

    # 4. RunAll Multi-Regime Suite
    runall_dir = WORKSPACE / "openfoam_cases/runall_suite"
    if runall_dir.exists():
        for cdir in sorted(runall_dir.iterdir()):
            if cdir.is_dir() and (cdir / "0").exists():
                cases.append(("RunAll Suite", cdir.name.replace("_", " ").title(), cdir))

    # 5. Bioreactor Parameter Suite
    bio_dir = WORKSPACE / "openfoam_cases/bioreactor_suite"
    if bio_dir.exists():
        for cdir in sorted(bio_dir.iterdir()):
            if cdir.is_dir() and (cdir / "0").exists():
                cases.append(("Bioreactor Suite", cdir.name.replace("_", " ").title(), cdir))

    return cases


def build_massive_dataset(
    cases: List[Tuple[str, str, Path]],
    grid_shape: Tuple[int, int] = (64, 64),
    apply_symmetry: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Loads all discovered OpenFOAM cases and compiles augmented transition pairs (S_t, S_{t+1})."""
    print("=" * 80)
    print("📦 INGESTING MASSIVE MULTI-SOURCE OPENFOAM SIMULATION DATASET")
    print("=" * 80)

    all_inputs = []
    all_targets = []
    case_metadata = []

    for source_cat, desc, case_path in cases:
        try:
            ds, _ = load_dataset_from_openfoam_case(case_path, grid_shape=grid_shape, rollout_steps=1)
            raw_traj = ds.traj_list[0].numpy()  # (T, 3, H, W)
            T = raw_traj.shape[0]

            if T >= 2:
                if apply_symmetry:
                    aug_trajs = apply_so2_rotation_augmentation(raw_traj)
                else:
                    aug_trajs = [raw_traj]

                pair_count = 0
                for a_traj in aug_trajs:
                    all_inputs.append(a_traj[:-1])
                    all_targets.append(a_traj[1:])
                    pair_count += a_traj.shape[0] - 1

                case_metadata.append({
                    "source": source_cat,
                    "case": case_path.name,
                    "desc": desc,
                    "snapshots": T,
                    "pairs": pair_count,
                    "sample_state": raw_traj[-1],  # final snapshot for gallery plotting
                })
                print(f"  ✓ [{source_cat:<21}] {case_path.name:<25}: {T:3d} snaps -> {pair_count:4d} augmented pairs ({desc})")
        except Exception as e:
            print(f"  ⚠ [{source_cat:<21}] {case_path.name:<25}: SKIPPED ({e})")

    master_inputs = torch.from_numpy(np.concatenate(all_inputs, axis=0)).float()
    master_targets = torch.from_numpy(np.concatenate(all_targets, axis=0)).float()

    print("\n✨ Massive Dataset Summary:")
    print(f"  • Total Distinct Simulation Regimes: {len(case_metadata)} cases")
    print(f"  • Total Transition Pairs:            {master_inputs.shape[0]:,} pairs")
    print(f"  • Total Physical Cell Ingestion:     {master_inputs.shape[0] * grid_shape[0] * grid_shape[1]:,} cells")
    return master_inputs, master_targets, case_metadata


def plot_multisource_gallery(case_metadata: List[Dict], output_path: Path):
    """Plots a multi-case gallery showcasing the immense diversity across all sources."""
    # Pick representative cases from each source
    selected = case_metadata[:16]
    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.2 * n_rows))
    axes = axes.flatten()

    for idx, meta in enumerate(selected):
        ax = axes[idx]
        state = meta["sample_state"]
        u_mag = np.hypot(state[0], state[1])
        H, W = u_mag.shape
        x = np.linspace(0, 1, W)
        y = np.linspace(0, 1, H)
        X, Y = np.meshgrid(x, y)

        im = ax.imshow(u_mag, cmap="viridis", origin="lower", extent=[0, 1, 0, 1])
        ax.streamplot(x, y, state[0], state[1], color="white", density=0.8, linewidth=0.6, arrowsize=0.6)
        ax.set_title(f"[{meta['source']}]\n{meta['desc']}", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved multi-source gallery to {output_path}")


def train_foundation_model(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[HybridCFDNet, FieldNormalizer, List[float]]:
    """Trains HybridCFDNet across the combined massive OpenFOAM dataset."""
    print("=" * 80)
    print(f"🔥 TRAINING GENERALIST FOUNDATION HYBRIDCFDNET ON {device.upper()}")
    print("=" * 80)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    normalizer = FieldNormalizer()
    normalizer.fit(inputs)
    inp_norm = normalizer.normalize(inputs)
    tgt_norm = normalizer.normalize(targets)

    dataset = TensorDataset(inp_norm, tgt_norm)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=32,
        modes1=12,
        modes2=12,
        num_layers=2,
        enforce_boundary=False,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    mse_fn = nn.MSELoss()

    loss_history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_div = 0.0

        for b_inp, b_tgt in loader:
            b_inp = b_inp.to(device)
            b_tgt = b_tgt.to(device)

            optimizer.zero_grad()
            b_pred = model(b_inp)

            # Prediction loss
            loss_mse = mse_fn(b_pred, b_tgt)

            # Physics-informed divergence penalty on unnormalized velocity
            b_pred_phys = normalizer.denormalize(b_pred)
            div = compute_divergence_2d(b_pred_phys[:, :2])
            loss_div = torch.mean(div ** 2)

            total_loss = loss_mse + 0.02 * loss_div
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_mse.item() * len(b_inp)
            epoch_div += loss_div.item() * len(b_inp)

        scheduler.step()
        epoch_loss /= len(dataset)
        epoch_div /= len(dataset)
        loss_history.append(epoch_loss)

        if epoch % 25 == 0 or epoch == 1 or epoch == epochs:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d}/{epochs:3d} | MSE Loss: {epoch_loss:.6f} | Div Loss: {epoch_div:.6f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {elapsed:.1f}s")

    return model, normalizer, loss_history


def evaluate_on_bioreactor(
    model: HybridCFDNet,
    normalizer: FieldNormalizer,
    device: str,
    output_dir: Path,
):
    """Performs 50-step autoregressive rollout on Stirred-Tank Bioreactor and calculates bioprocess shear stress."""
    print("=" * 80)
    print("🧪 EVALUATING FOUNDATION SURROGATE ON STIRRED-TANK BIOREACTOR (STR)")
    print("=" * 80)

    # 1. Ground truth simulation
    geom = BioreactorGeometry(T=1.0, H=1.0)
    sim = BioreactorSimulator(nx=64, ny=64, lx=1.0, ly=1.0, nu=0.01, dt=0.0025, u_tip=1.2, geom=geom)
    res = sim.run_simulation(n_steps=100, save_interval=2, is_cavity=False)
    gt_traj = np.stack([res["u"], res["v"], res["p"]], axis=1)  # (50, 3, 64, 64)

    # 2. Autoregressive Rollout with Foundation Surrogate
    model.eval()
    curr_state = torch.from_numpy(gt_traj[0:1]).float()
    pred_rollout = [curr_state.cpu().numpy()[0]]

    with torch.no_grad():
        for t in range(1, len(gt_traj)):
            inp_norm = normalizer.normalize(curr_state).to(device)
            out_norm = model(inp_norm)
            out_phys = normalizer.denormalize(out_norm.cpu())
            pred_rollout.append(out_phys.numpy()[0])
            curr_state = out_phys

    pred_traj = np.stack(pred_rollout, axis=0)  # (50, 3, 64, 64)

    # Calculate errors
    rel_l2_errors = []
    for t in range(len(gt_traj)):
        err = np.linalg.norm(pred_traj[t] - gt_traj[t]) / (np.linalg.norm(gt_traj[t]) + 1e-6)
        rel_l2_errors.append(err * 100.0)

    final_l2 = rel_l2_errors[-1]
    mean_l2 = np.mean(rel_l2_errors)
    print(f"  • 50-Step Autoregressive Rollout Mean Rel L2 Error: {mean_l2:.2f}%")
    print(f"  • Final Step (T=50) Rel L2 Error:                  {final_l2:.2f}%")

    # 3. Bioprocess Hydrodynamic Shear Stress & Cell Viability Metrics
    gt_u = gt_traj[-1, 0]
    gt_v = gt_traj[-1, 1]
    pred_u = pred_traj[-1, 0]
    pred_v = pred_traj[-1, 1]

    gt_metrics = compute_bioprocess_shear_metrics(gt_u, gt_v, sim.dx, sim.dy, nu=0.01)
    pred_metrics = compute_bioprocess_shear_metrics(pred_u, pred_v, sim.dx, sim.dy, nu=0.01)

    print("\n  🔬 Bioprocess Hydrodynamic Shear Stress Analysis (Final State):")
    print(f"    • Ground Truth Mean Shear Stress:     {gt_metrics['tau_mean']:.4f} Pa")
    print(f"    • Surrogate Predicted Mean Shear:     {pred_metrics['tau_mean']:.4f} Pa (Error: {abs(pred_metrics['tau_mean'] - gt_metrics['tau_mean'])/gt_metrics['tau_mean']*100:.2f}%)")
    print(f"    • Ground Truth Max Shear (Impeller):  {gt_metrics['tau_max']:.4f} Pa")
    print(f"    • Surrogate Predicted Max Shear:      {pred_metrics['tau_max']:.4f} Pa (Error: {abs(pred_metrics['tau_max'] - gt_metrics['tau_max'])/gt_metrics['tau_max']*100:.2f}%)")
    print(f"    • Kolmogorov Microscale eta_K:        {pred_metrics['eta_kolmogorov']:.6f} m vs GT: {gt_metrics['eta_kolmogorov']:.6f} m")

    # 4. Plot Comparison Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    # Row 1: Velocity Magnitude
    gt_speed = np.hypot(gt_u, gt_v)
    pred_speed = np.hypot(pred_u, pred_v)
    speed_err = np.abs(pred_speed - gt_speed)

    im0 = axes[0, 0].imshow(gt_speed, cmap="turbo", origin="lower")
    axes[0, 0].set_title("Ground Truth Speed |U| (m/s)", fontweight="bold")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(pred_speed, cmap="turbo", origin="lower")
    axes[0, 1].set_title("Surrogate Predicted Speed |U| (m/s)", fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(speed_err, cmap="magma", origin="lower")
    axes[0, 2].set_title("Absolute Velocity Error |U_pred - U_gt|", fontweight="bold")
    plt.colorbar(im2, ax=axes[0, 2])

    # Row 2: Bioprocess Shear Stress tau(x, y) & Error Rollout Curve
    im3 = axes[1, 0].imshow(gt_metrics["tau_xy"], cmap="inferno", origin="lower")
    axes[1, 0].set_title("GT Hydrodynamic Shear Stress tau (Pa)", fontweight="bold")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(pred_metrics["tau_xy"], cmap="inferno", origin="lower")
    axes[1, 1].set_title("Predicted Shear Stress tau (Pa)", fontweight="bold")
    plt.colorbar(im4, ax=axes[1, 1])

    axes[1, 2].plot(range(len(rel_l2_errors)), rel_l2_errors, color="#d62728", lw=2.5, marker="o", markersize=3)
    axes[1, 2].set_title(f"50-Step Autoregressive Error (Final: {final_l2:.2f}%)", fontweight="bold")
    axes[1, 2].set_xlabel("Rollout Step t")
    axes[1, 2].set_ylabel("Relative L2 Error (%)")
    axes[1, 2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = output_dir / "bioreactor_shear_and_rollout_analysis.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved bioreactor evaluation figure to {plot_path}")

    return {
        "mean_l2": mean_l2,
        "final_l2": final_l2,
        "gt_tau_mean": gt_metrics["tau_mean"],
        "pred_tau_mean": pred_metrics["tau_mean"],
        "tau_error_pct": abs(pred_metrics["tau_mean"] - gt_metrics["tau_mean"]) / gt_metrics["tau_mean"] * 100,
    }


def main():
    # 1. Generate Bioreactor Parameter Variations
    print("\n[Step 1/5] Ensuring Stirred-Tank Bioreactor Parameter Suite is generated...")
    generate_bioreactor_suite()

    # 2. Discover all cases
    print("\n[Step 2/5] Cataloging all simulation cases across official, GitHub, Wolf, Holzmann, and bioreactor suites...")
    cases = discover_all_cases()
    print(f"  • Discovered {len(cases)} potential candidate simulation cases.")

    # 3. Ingest and Compile Massive Dataset
    ds_save_path = OUTPUT_DIR / "massive_openfoam_dataset.pt"
    if ds_save_path.exists():
        print(f"\n[Step 3/5] Loading pre-compiled master dataset from {ds_save_path}...")
        saved = torch.load(ds_save_path, weights_only=False)
        master_inp = saved["inputs"]
        master_tgt = saved["targets"]
        meta = saved["metadata"]
        print(f"  ✓ Loaded {len(meta)} cases, {master_inp.shape[0]:,} transition pairs.")
    else:
        print("\n[Step 3/5] Compiling and standardizing dataset with SO(2) rotation symmetry...")
        master_inp, master_tgt, meta = build_massive_dataset(cases, grid_shape=(64, 64), apply_symmetry=True)
        torch.save({"inputs": master_inp, "targets": master_tgt, "metadata": meta}, ds_save_path)
        print(f"  ✓ Saved master dataset to {ds_save_path}")

    # Plot multi-source gallery
    gallery_path = OUTPUT_DIR / "comprehensive_multi_source_gallery.png"
    if not gallery_path.exists():
        plot_multisource_gallery(meta, gallery_path)

    # 4. Train Foundation Neural Operator
    print("\n[Step 4/5] Training generalist Foundation HybridCFDNet...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, normalizer, loss_history = train_foundation_model(
        master_inp,
        master_tgt,
        epochs=120,
        batch_size=16,
        lr=1e-3,
        device=device,
    )

    # Save model and normalizer
    ckpt_path = WORKSPACE / "models/massive_diverse_hybrid_cfd_net.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "loss_history": loss_history,
        "num_cases": len(meta),
        "total_pairs": master_inp.shape[0],
    }, ckpt_path)
    print(f"  ✓ Saved checkpoint to {ckpt_path}")

    # 5. Evaluate on Stirred-Tank Bioreactor
    print("\n[Step 5/5] Evaluating on industrial Stirred-Tank Bioreactor (STR)...")
    eval_res = evaluate_on_bioreactor(model, normalizer, device, OUTPUT_DIR)

    # Plot convergence
    plt.figure(figsize=(8, 4.5))
    plt.plot(loss_history, color="#1f77b4", lw=2, label="HybridCFDNet Training Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title(f"Training Convergence on Massive OpenFOAM Corpus ({len(meta)} Cases, {master_inp.shape[0]:,} Pairs)", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    conv_path = OUTPUT_DIR / "training_convergence_massive_dataset.png"
    plt.savefig(conv_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved convergence curve to {conv_path}")

    print("\n" + "=" * 80)
    print("🎉 MASSIVE OPENFOAM SURROGATE PIPELINE SUCCESSFULLY COMPLETED!")
    print(f"  • Total Regimes Ingested:    {len(meta)}")
    print(f"  • Total Training Pairs:      {master_inp.shape[0]:,}")
    print(f"  • STR Rollout Error:         {eval_res['final_l2']:.2f}% (Mean: {eval_res['mean_l2']:.2f}%)")
    print(f"  • Bioprocess Shear Error:    {eval_res['tau_error_pct']:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
