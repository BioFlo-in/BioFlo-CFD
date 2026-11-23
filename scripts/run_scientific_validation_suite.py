#!/usr/bin/env python3
"""
BioFlo-CFD: Scientific Validation Suite Runner (NLSS v1.0).
Executes the comprehensive OpenFOAM validation benchmark on the 894,950-cell 3D CSTR dataset:
  - Loads trained Neural POD Operator
  - Tests held-out OpenFOAM CFD snapshots
  - Evaluates Three-Tier Accuracy (Tiers A, B, C)
  - Evaluates Multi-Norm Velocity Errors (L2, L1, Linf, RMSE, MAE, Components, 3D Vector Angle)
  - Evaluates Regional Hydrodynamic ROIs (Impeller, Baffles, Bottom, Surface, Bulk)
  - Evaluates Physical Conservation Residuals (Divergence, Vorticity, Momentum)
  - Profiles Inference Latency & Speedup
  - Exports reports/validation_report.json and reports/validation_report.html
"""

import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from nlss.benchmark.openfoam_benchmark import OpenFOAMBenchmarkRunner


def main():
    print("=" * 80)
    print("🧪 BioFlo-CFD: Automated Scientific Validation & Audit Suite (NLSS v1.0)")
    print("=" * 80)

    data_path = BASE_DIR / "data" / "cstr_3d_dataset_64.pt"
    ckpt_path = BASE_DIR / "models" / "ultra_precision_cstr_pod_net_best.pt"
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/home/ajit/.gemini/antigravity/brain/2a56aadc-c404-4828-a476-c7f419137646")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Execution Device: {device}")

    # 1. Load Ground Truth CFD Dataset
    print(f"• Loading OpenFOAM 3D CSTR dataset from {data_path}...")
    d = torch.load(data_path, map_location="cpu", weights_only=False)
    states = d["states"]  # (81, 5, 64, 64, 64) -> [ux, uy, uz, p, k]
    times = np.array(d["times"], dtype=np.float64)  # 81 points
    fluid_mask = d["fluid_mask"].numpy().astype(bool)  # (64, 64, 64)
    N_snaps = len(times)
    print(f"  ✓ Loaded {N_snaps} transient snapshots across 64^3 grid ({int(fluid_mask.sum()):,} fluid cells)")

    # Extract 3-component velocity snapshots on active fluid cells
    mask_flat = fluid_mask.reshape(-1)
    N_active = int(mask_flat.sum())
    N_dof = 3 * N_active

    flat_vel_snaps = np.zeros((N_snaps, N_dof), dtype=np.float32)
    for i in range(N_snaps):
        u_3ch = states[i, :3].numpy()  # (3, 64, 64, 64)
        for c in range(3):
            flat_vel_snaps[i, c * N_active:(c + 1) * N_active] = u_3ch[c].reshape(-1)[mask_flat]

    pressure_snaps = states[:, 3].numpy()  # (81, 64, 64, 64)

    # 2. Load or Calibrate Canonical Neural POD Operator
    if ckpt_path.exists():
        print(f"• Loading calibrated checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        n_modes = ckpt.get("n_modes", 55)
        train_indices = ckpt.get("train_indices", list(range(0, 65)))
        test_indices = ckpt.get("test_indices", list(range(65, N_snaps)))

        model = NeuralPODOperator3D(n_modes=n_modes, param_dim=1, hidden_dim=256, num_layers=4)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.set_parameter_bounds([ckpt.get("t_min", float(times.min()))], [ckpt.get("t_max", float(times.max()))])
        train_params = times[train_indices, None]
        model.set_training_parameters(train_params, strict=False)
    else:
        print("• Checkpoint not found; performing fast calibration...")
        test_indices = [i for i in range(N_snaps) if i % 5 == 0]
        train_indices = [i for i in range(N_snaps) if i not in test_indices]
        model = NeuralPODOperator3D(n_modes=45, param_dim=1, hidden_dim=256, num_layers=4)
        model.calibrate_basis(flat_vel_snaps[train_indices], n_modes=45)
        model.set_training_parameters(times[train_indices, None])

    model = model.to(device)
    model.eval()
    print(f"  ✓ Model calibrated with {model.n_modes} singular modes ({model.get_model_summary()['spatial_dof']:,} spatial DOFs)")

    # 3. Setup Benchmark Runner
    test_vel = flat_vel_snaps[test_indices]
    test_t = times[test_indices, None]
    test_p = pressure_snaps[test_indices]

    print(f"• Executing benchmark over {len(test_indices)} held-out verification snapshots...")
    runner = OpenFOAMBenchmarkRunner(
        model=model,
        grid_shape=(64, 64, 64),
        spacings=(0.50 / 64, 0.50 / 64, 1.10 / 64),
        fluid_mask=fluid_mask,
        tank_radius=0.25,
        impeller_radius=0.083,
        impeller_elevation=0.35,
        liquid_height=1.00,
    )

    report_data = runner.run_benchmark(
        test_snapshots=test_vel,
        test_parameters=test_t,
        pressure_snapshots=test_p,
        is_unseen_condition=False,
        operating_condition_label="3D CSTR Held-Out Verification (Re=28,400, 300 RPM)",
    )

    # 4. Export Reports
    json_path, html_path = runner.export_reports(
        report_data,
        output_dir=reports_dir,
        json_filename="validation_report.json",
        html_filename="validation_report.html",
    )
    print(f"✓ Saved JSON Report: {json_path}")
    print(f"✓ Saved HTML Report: {html_path}")

    if artifact_dir.exists():
        shutil.copy2(json_path, artifact_dir / "validation_report.json")
        shutil.copy2(html_path, artifact_dir / "validation_report.html")
        print(f"✓ Copied Reports to Conversation Artifacts Directory")

    # 5. Display Console Summary
    tier = report_data["three_tier_hierarchy"]
    rois = report_data["regional_roi_metrics"]
    phys = report_data["physics_residuals"]
    inf = report_data["inference_performance"]

    print("\n" + "=" * 80)
    print("📊 BENCHMARK EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"• Tier A (POD Truncation Limit) Accuracy: {tier['tier_a_pod_reconstruction']['reconstruction_accuracy_pct']:.3f}% (Rel L2: {tier['tier_a_pod_reconstruction']['rel_l2_error']*100:.3f}%, R²: {tier['tier_a_pod_reconstruction']['r2_score']:.6f})")
    print(f"• Tier B (Neural Prediction) Accuracy:   {tier['tier_b_neural_prediction']['prediction_accuracy_pct']:.3f}% (Rel L2: {tier['tier_b_neural_prediction']['rel_l2_error']*100:.3f}%)")
    print(f"• Single-Step Inference Latency:         {inf['single_snapshot_latency_ms']:.2f} ms ({inf['cfd_speedup_factor']:,.0f}× faster than OpenFOAM)")
    print(f"• Regional Impeller Discharge ROI Error: {rois['impeller_discharge']['rel_l2_error']*100:.2f}% (Peak Linf: {rois['impeller_discharge']['rel_linf_error']*100:.2f}%)")
    print(f"• Regional Wall Baffles ROI Error:       {rois['wall_baffles']['rel_l2_error']*100:.2f}% (Jet Impingement)")
    print(f"• Regional Dished Bottom ROI Error:      {rois['dished_bottom']['rel_l2_error']*100:.2f}% (Solids Suspension)")
    print(f"• Discrete Incompressibility (div(u)):   {phys['mean_abs_divergence']:.2e} 1/s")
    print(f"• 3D Vector Angular Alignment:           {report_data['global_velocity_metrics']['angular_alignment']['mean_angle_deg']:.2f}° (cos θ = {report_data['global_velocity_metrics']['angular_alignment']['mean_cos_theta']:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
