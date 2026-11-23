#!/usr/bin/env python3
"""
BioFlo-CFD: Tier C Parametric Generalization Benchmark (NLSS v1.0).
Tests parameter-conditioned reduced-order modeling on unobserved CFD physics:
  - Aligned Transient Step Response: tau = t - t_step in [0.00, 0.50]s
  - Balanced Multi-Condition Training: 120 RPM (11 snaps), 300 RPM (11 snaps), 600 RPM (11 snaps)
  - HELD-OUT TEST CASE (TIER C): 450 RPM (11 snaps) -> ZERO exposure during training
  - Evaluates Tier A, Tier B, and Tier C metrics
  - Evaluates Regional Hydrodynamic ROIs on the unseen 450 RPM flow field
  - Exports reports/tier_c_generalization_report.json and .html
"""

import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from nlss.benchmark.openfoam_benchmark import OpenFOAMBenchmarkRunner
from nlss.validation.velocity_metrics import relative_l2_error, r2_score_field


def main():
    print("=" * 80)
    print("🔬 BioFlo-CFD: Tier C Parametric CFD Generalization Benchmark (NLSS v1.0)")
    print("=" * 80)

    data_path = BASE_DIR / "data" / "cstr_massive_multiphase_dataset.pt"
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/home/ajit/.gemini/antigravity/brain/2a56aadc-c404-4828-a476-c7f419137646")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"• Hardware Device: {device}")

    # 1. Load Multi-Condition Dataset
    print(f"• Loading multi-condition OpenFOAM dataset from {data_path}...")
    d = torch.load(data_path, map_location="cpu", weights_only=False)
    fluid_mask = d["fluid_mask"].numpy().astype(bool)  # (32, 32, 32)
    mask_flat = fluid_mask.reshape(-1)
    N_active = int(mask_flat.sum())
    N_dof = 3 * N_active
    grid_dim = d["grid_dim"]
    print(f"  ✓ Active fluid cells: {N_active:,} / {mask_flat.size:,} ({N_dof:,} velocity DOFs)")

    cases = d["cases"]

    # Extract synchronous transient step response data (tau = t - 3.90s in [0.00, 0.50]s)
    def get_transient_data(case_dict, t0=3.90, max_tau=0.50):
        st = case_dict["states"]
        ts = np.array(case_dict["times"])
        tau = ts - t0
        idx = (tau >= -1e-5) & (tau <= max_tau + 1e-5)
        st_sub = st[idx]
        n_snaps = st_sub.shape[0]
        out = np.zeros((n_snaps, N_dof), dtype=np.float32)
        for i in range(n_snaps):
            for ch in range(3):
                out[i, ch * N_active:(ch + 1) * N_active] = st_sub[i, ch].numpy().reshape(-1)[mask_flat]
        params = np.column_stack([tau[idx], np.full(n_snaps, float(case_dict["rpm"]))]).astype(np.float32)
        return out, params, tau[idx]

    X_120, P_120, tau_120 = get_transient_data(cases[2])
    X_300, P_300, tau_300 = get_transient_data(cases[1])
    X_600, P_600, tau_600 = get_transient_data(cases[4])
    X_450, P_450, tau_450 = get_transient_data(cases[3])

    print(f"\n• Dataset Partitioning (Aligned Transient Acceleration tau in [0.0, 0.50]s):")
    print(f"  - 120 RPM (Case 2): {len(tau_120)} snapshots")
    print(f"  - 300 RPM (Case 1): {len(tau_300)} snapshots")
    print(f"  - 600 RPM (Case 4): {len(tau_600)} snapshots")
    print(f"  - HELD-OUT TEST (TIER C): 450 RPM (Case 3, {len(tau_450)} snapshots) -> ZERO EXPOSURE DURING TRAINING")

    # Combine balanced training set
    train_X = np.concatenate([X_120, X_300, X_600], axis=0)  # (33, N_dof)
    train_P = np.concatenate([P_120, P_300, P_600], axis=0)  # (33, 2)

    # 2. Fit Quadratic Parametric Mean Field m(x; N) = c2*N^2 + c1*N + c0
    m_120 = np.mean(X_120, axis=0)
    m_300 = np.mean(X_300, axis=0)
    m_600 = np.mean(X_600, axis=0)

    N_vec = np.array([120.0, 300.0, 600.0])
    A_mat = np.column_stack([N_vec**2, N_vec, np.ones(3)])
    poly_coeffs = np.linalg.solve(A_mat, np.vstack([m_120, m_300, m_600]))  # (3, N_dof)

    # Condition-decoupled fluctuations
    fluct_120 = X_120 - m_120
    fluct_300 = X_300 - m_300
    fluct_600 = X_600 - m_600
    fluct_train = np.concatenate([fluct_120, fluct_300, fluct_600], axis=0)

    # 3. Calibrate Orthonormal Fluctuation Basis
    n_modes = 20
    print(f"\n• Calibrating Condition-Decoupled Fluctuation Basis ({n_modes} modes)...")
    U, S, Vt = np.linalg.svd(fluct_train, full_matrices=False)
    K = min(n_modes, len(S))
    retained_basis = Vt[:K]
    cum_e = np.cumsum(S**2) / np.sum(S**2)

    model = NeuralPODOperator3D(
        n_modes=K,
        param_dim=2,
        hidden_dim=128,
        num_layers=4,
        num_frequencies=16,
    )
    model.register_buffer("mean_field", torch.from_numpy(np.mean(train_X, axis=0)).float())
    model.register_buffer("basis_matrix", torch.from_numpy(retained_basis).float())
    model.register_buffer("singular_values", torch.from_numpy(S[:K]).float())
    model.set_parameter_bounds([0.0, 100.0], [0.55, 700.0])
    model.set_parametric_mean_polynomial(poly_coeffs)
    model.set_training_parameters(train_P, strict=False)
    model.is_calibrated = True

    print(f"  ✓ SVD complete: Top-{K} modes capture {cum_e[K-1]*100:.2f}% of fluctuation energy")

    # 4. Train Neural Parameter Stepper
    print(f"\n• Training Neural Parameter Stepper f_theta(tau, RPM)...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1500, eta_min=1e-5)

    target_a = fluct_train @ retained_basis.T  # (33, K)
    P_tensor = torch.from_numpy(train_P).float().to(device)
    Y_tensor = torch.from_numpy(target_a).float().to(device)

    # Singular value energy weighting
    S_weights = model.singular_values / (torch.sum(model.singular_values) + 1e-12)
    S_weights = (S_weights / S_weights.mean()).unsqueeze(0)

    model.train()
    t0_train = time.time()
    for epoch in range(1, 1501):
        optimizer.zero_grad()
        pred_a = model.predict_coefficients(P_tensor)
        loss = torch.mean(S_weights * (pred_a - Y_tensor)**2)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 300 == 0 or epoch == 1500:
            rel_a = relative_l2_error(pred_a.detach().cpu().numpy(), target_a)
            print(f"  Epoch {epoch:4d}/1500: Loss = {loss.item():.6e}, Mode Rel L2 = {rel_a*100:.2f}%")

    t_train = time.time() - t0_train
    print(f"  ✓ Training completed in {t_train:.2f}s")
    model.eval()

    # 5. Execute OpenFOAM Validation Benchmark on Unseen 450 RPM
    print(f"\n• Running Benchmark on UNSEEN 450 RPM Flow Field...")
    bounds = d["grid_bounds"]
    dx = float(bounds["x"][1] - bounds["x"][0]) / grid_dim
    dy = float(bounds["y"][1] - bounds["y"][0]) / grid_dim
    dz = float(bounds["z"][1] - bounds["z"][0]) / grid_dim

    runner = OpenFOAMBenchmarkRunner(
        model=model,
        grid_shape=(grid_dim, grid_dim, grid_dim),
        spacings=(dx, dy, dz),
        fluid_mask=fluid_mask,
        tank_radius=0.25,
        impeller_radius=0.083,
        impeller_elevation=0.35,
        liquid_height=1.00,
    )

    report_data = runner.run_benchmark(
        test_snapshots=X_450,
        test_parameters=P_450,
        is_unseen_condition=True,
        operating_condition_label="Tier C Unseen Operating Condition: 450 RPM (Balanced Step-Response)",
    )

    # 6. Export Refreshed Reports
    json_path, html_path = runner.export_reports(
        report_data,
        output_dir=reports_dir,
        json_filename="tier_c_generalization_report.json",
        html_filename="tier_c_generalization_report.html",
    )
    print(f"\n✓ Saved JSON Report: {json_path}")
    print(f"✓ Saved HTML Report: {html_path}")

    if artifact_dir.exists():
        shutil.copy2(json_path, artifact_dir / "tier_c_generalization_report.json")
        shutil.copy2(html_path, artifact_dir / "tier_c_generalization_report.html")
        print(f"✓ Copied Reports to Conversation Artifacts Directory")

    # 7. Executive Console Summary
    tier = report_data["three_tier_hierarchy"]
    rois = report_data["regional_roi_metrics"]
    glob = report_data["global_velocity_metrics"]
    inf = report_data["inference_performance"]
    dom = report_data["parameter_domain"]

    print("\n" + "=" * 80)
    print("📊 TIER C GENERALIZATION BENCHMARK SUMMARY (450 RPM REFINED)")
    print("=" * 80)
    print(f"• Domain Status:                  {dom['domain_status'].upper()} (is_in_domain={dom['is_in_domain']}, confidence={dom['confidence']})")
    print(f"• Tier A (POD Basis Bound):       {tier['tier_a_pod_reconstruction']['reconstruction_accuracy_pct']:.2f}% (Rel L2 = {tier['tier_a_pod_reconstruction']['rel_l2_error']*100:.2f}%, R² = {tier['tier_a_pod_reconstruction']['r2_score']:.6f})")
    print(f"• Tier B (In-Sample Prediction):  {tier['tier_b_neural_prediction']['prediction_accuracy_pct']:.2f}% (Rel L2 = {tier['tier_b_neural_prediction']['rel_l2_error']*100:.2f}%)")
    print(f"• TIER C (CFD GENERALIZATION):    {tier['tier_c_cfd_generalization']['generalization_accuracy_pct']:.2f}% (Rel L2 = {tier['tier_c_cfd_generalization']['rel_l2_error']*100:.2f}%, R² = {tier['tier_c_cfd_generalization']['r2_score']:.6f})")
    print(f"• Global Velocity R² Score:       {glob['global_r2']:.6f}")
    print(f"• 3D Vector Alignment:            {glob['angular_alignment']['mean_angle_deg']:.2f}° (cos θ = {glob['angular_alignment']['mean_cos_theta']:.4f})")
    print(f"• Regional Impeller Zone Error:   {rois['impeller_discharge']['rel_l2_error']*100:.2f}%")
    print(f"• Regional Wall Baffles Error:    {rois['wall_baffles']['rel_l2_error']*100:.2f}%")
    print(f"• Regional Dished Bottom Error:   {rois['dished_bottom']['rel_l2_error']*100:.2f}%")
    print(f"• Single-Snapshot Latency:        {inf['single_snapshot_latency_ms']:.2f} ms ({inf['cfd_speedup_factor']:,.0f}× faster than OpenFOAM)")
    print("=" * 80)


if __name__ == "__main__":
    main()
