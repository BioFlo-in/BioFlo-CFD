"""
BioFlo-CFD & BiRD: Multiphase Neural POD Operator Training & QoI Benchmark (Phase 2).
Demonstrates end-to-end continuous Eulerian-Eulerian multiphase surrogate modeling:
  1. Couples liquid velocity field u_l(x) with continuous dispersed gas volume fraction alpha_g(x).
  2. Applies scale-invariant dimensionless coordinate normalization via BiRDMeshAdapter.
  3. Calibrates multi-channel Neural POD Operator (4-channel: u_x, u_y, u_z, alpha_gas) with
     positivity-preserving bounded decoding for alpha_g in [0, 1].
  4. Evaluates first-principles two-phase QoIs ported from BiRD:
       - Higbie penetration mass transfer (k_L, a, k_L a [h^-1])
       - Integrated gas holdup <alpha_g>
       - Dimensionless scaling numbers (Re, Fr, We, Sc, Sh, Mo)
  5. Exports publication-grade validation report.
"""

import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from nlss.dataset.bird_mesh_adapter import StirredTankSpec, BiRDMeshAdapter
from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from nlss.validation.velocity_metrics import compute_comprehensive_velocity_metrics
from nlss.validation.bioprocess_metrics import (
    compute_higbie_mass_transfer,
    compute_dimensionless_bioprocess_numbers,
    compute_specific_power_input,
)


def generate_synthetic_bird_multiphase_snapshots(
    spec: StirredTankSpec,
    grid_shape: tuple = (16, 16, 16),
    n_snapshots_per_rpm: int = 15,
    rpm_list: list = [300.0, 450.0, 600.0],
    seed: int = 42,
) -> tuple:
    """
    Synthesizes analytically divergence-free two-phase flow fields for BiRD reactor geometry:
      - Liquid velocity: derived from an axisymmetric Stokes streamfunction psi(r, z)
        so div(u_r e_r + u_z e_z) == 0 analytically, combined with tangential swirl u_theta(r, z).
      - Dispersed gas holdup: sparger plume rising along center with impeller radial dispersion.
    Note: Analytical proxy benchmark (distinct from OpenFOAM Eulerian-Eulerian CFD runs).
    """
    np.random.seed(seed)
    D, H, W = grid_shape

    # Grid coordinates in [-1, 1] x [-1, 1] x [0, 1]
    z = np.linspace(0.0, 1.0, D).reshape(D, 1, 1)
    y = np.linspace(-1.0, 1.0, H).reshape(1, H, 1)
    x = np.linspace(-1.0, 1.0, W).reshape(1, 1, W)
    r = np.sqrt(x**2 + y**2)
    r_clip = np.minimum(r, 1.0)
    theta = np.arctan2(y, x)

    # Tank fluid mask: cylinder r <= 0.98
    fluid_mask = (r <= 0.98) & (z >= 0.02) & (z <= 0.98)

    snapshots_u = []      # (N, 3, D, H, W)
    snapshots_alpha = []  # (N, 1, D, H, W)
    params_list = []      # (N, 1) [RPM]

    for rpm in rpm_list:
        n_rev_s = rpm / 60.0
        v_tip = np.pi * n_rev_s * spec.Da
        z_imp = spec.C[0] / spec.H

        for snap_idx in range(n_snapshots_per_rpm):
            phase = 2.0 * np.pi * snap_idx / n_snapshots_per_rpm

            # 1. Axisymmetric divergence-free poloidal flow via streamfunction psi(r, z) = f(r) * g(z)
            # f(r) = r^2 * (1 - r^2)^2 => f(r)/r = r*(1-r^2)^2, f'(r)/r = 2*(1-r^2)*(1-3*r^2)
            g_z = np.sin(np.pi * z) * np.exp(-((z - z_imp) ** 2) / 0.16)
            dg_dz = (
                np.pi * np.cos(np.pi * z)
                - 2.0 * (z - z_imp) / 0.16 * np.sin(np.pi * z)
            ) * np.exp(-((z - z_imp) ** 2) / 0.16)

            amp = 0.25 * v_tip * (1.0 + 0.08 * np.cos(phase))
            # u_r = -(1/r) dpsi/dz, u_z = (1/r) dpsi/dr => (1/r)d(r u_r)/dr + du_z/dz == 0 identically
            u_rad = -amp * r_clip * ((1.0 - r_clip**2) ** 2) * dg_dz
            u_axial = amp * 2.0 * (1.0 - r_clip**2) * (1.0 - 3.0 * (r_clip**2)) * g_z

            # Axisymmetric tangential swirl around impeller (d u_theta / d theta = 0 => div = 0)
            u_theta = v_tip * np.exp(-((r_clip - 0.35) ** 2) / 0.08) * np.exp(-((z - z_imp) ** 2) / 0.05) * (1.0 + 0.04 * np.sin(phase))

            ux = (-u_theta * np.sin(theta) + u_rad * np.cos(theta)) * fluid_mask
            uy = (u_theta * np.cos(theta) + u_rad * np.sin(theta)) * fluid_mask
            uz = u_axial * fluid_mask

            u_liquid = np.stack([ux, uy, uz], axis=0)

            # 2. Dispersed gas volume fraction field alpha_g
            plume_base = np.exp(-(r**2) / 0.08) * (1.0 - z)
            dispersion = 0.4 * np.exp(-((r - 0.5)**2) / 0.15) * np.exp(-((z - z_imp)**2) / 0.05)
            alpha_raw = 0.02 + 0.12 * plume_base + 0.08 * dispersion
            alpha_fluct = 0.01 * np.sin(2.0 * theta + phase) * fluid_mask
            alpha_gas = np.clip((alpha_raw + alpha_fluct) * fluid_mask, 0.0, 0.40)
            alpha_gas = alpha_gas[np.newaxis, ...]  # (1, D, H, W)

            snapshots_u.append(u_liquid)
            snapshots_alpha.append(alpha_gas)
            params_list.append([rpm])

    U_all = np.stack(snapshots_u, axis=0)         # (N, 3, D, H, W)
    Alpha_all = np.stack(snapshots_alpha, axis=0) # (N, 1, D, H, W)
    params_all = np.array(params_list)            # (N, 1)

    return U_all, Alpha_all, params_all, fluid_mask


def main():
    print("=" * 80)
    print("BioFlo-CFD & BiRD: Multiphase Neural POD Operator Benchmark (Phase 2)")
    print("  [Data Source: Analytical Divergence-Free Two-Phase Proxy]")
    print("=" * 80)

    spec = StirredTankSpec(Dt=0.30, Da=0.10, H=0.45, C=[0.15], nbaffles=4)
    print(f"Reactor Geometry: Tank D={spec.Dt*1000:.0f}mm, Liquid H={spec.H*1000:.0f}mm, Impeller D={spec.Da*1000:.0f}mm")

    # 1. Synthesize two-phase snapshots across operating speeds
    grid_shape = (16, 16, 16)
    train_rpms = [300.0, 600.0]
    test_rpm = 450.0  # Unseen intermediate condition for Tier C generalization

    print(f"\n[1/5] Synthesizing continuous divergence-free two-phase snapshots...")
    U_train, Alpha_train, p_train, mask = generate_synthetic_bird_multiphase_snapshots(
        spec, grid_shape=grid_shape, n_snapshots_per_rpm=15, rpm_list=train_rpms, seed=42
    )
    U_test, Alpha_test, p_test, _ = generate_synthetic_bird_multiphase_snapshots(
        spec, grid_shape=grid_shape, n_snapshots_per_rpm=15, rpm_list=[test_rpm], seed=99
    )

    N_train, _, D, H, W = U_train.shape
    N_test = U_test.shape[0]
    print(f"  Training snapshots: {N_train} ({train_rpms} RPM)")
    print(f"  Held-out generalization snapshots: {N_test} ({test_rpm} RPM)")

    # 2. Assemble 4-channel multi-field tensors and normalize per channel so alpha_g is weighted fairly
    X_train = np.concatenate([U_train, Alpha_train], axis=1) # (N_train, 4, D, H, W)
    X_test = np.concatenate([U_test, Alpha_test], axis=1)   # (N_test, 4, D, H, W)

    channel_scales = np.maximum(np.std(X_train, axis=(0, 2, 3, 4)), 1e-4)  # (4,)
    X_train_scaled = X_train / channel_scales.reshape(1, 4, 1, 1, 1)

    N_dof = 4 * D * H * W
    X_train_flat = X_train_scaled.reshape(N_train, N_dof)

    # 3. Fit Multi-Channel Neural POD Operator on normalized channels
    n_modes = 16
    print(f"\n[2/5] Calibrating channel-normalized 4-channel Neural POD Operator (n_modes={n_modes})...")
    operator = NeuralPODOperator3D(n_modes=n_modes, param_dim=1, hidden_dim=64, num_layers=3)
    calib_stats = operator.calibrate_basis(X_train_flat, n_modes=n_modes, solver="exact")
    print(f"  Cumulative POD Kinetic Energy Captured: {calib_stats['cumulative_energy_pct']:.2f}%")
    print(f"  Theoretical Truncation Limit: {calib_stats['theoretical_reconstruction_accuracy_pct']:.2f}%")

    operator.set_training_parameters(p_train, strict=False)

    # 4. Train Neural Parameter Stepper
    print(f"\n[3/5] Training deep parameter stepper to predict mode coefficients...")
    optimizer = torch.optim.AdamW(operator.mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    p_train_t = torch.tensor(p_train, dtype=torch.float32)

    X_train_t = torch.tensor(X_train_flat, dtype=torch.float32)
    a_opt_train = torch.matmul(X_train_t - operator.mean_field, operator.basis_matrix.T)

    for epoch in range(250):
        optimizer.zero_grad()
        a_pred = operator.predict_coefficients(p_train_t)
        loss = torch.nn.functional.mse_loss(a_pred, a_opt_train)
        loss.backward()
        optimizer.step()

    print(f"  Final training mode MSE loss: {loss.item():.6f}")

    # 5. Evaluate held-out Tier C Generalization across ALL N_test snapshots on unseen 450 RPM
    print(f"\n[4/5] Evaluating Tier C Generalization on unobserved {test_rpm} RPM ({N_test} snapshots)...")
    p_test_t = torch.tensor(p_test, dtype=torch.float32)
    with torch.no_grad():
        X_pred_scaled = operator(p_test_t).view(N_test, 4, D, H, W).cpu().numpy()

    X_pred = X_pred_scaled * channel_scales.reshape(1, 4, 1, 1, 1)
    U_pred = X_pred[:, :3]
    Alpha_pred = np.clip(X_pred[:, 3:4], 0.0, 1.0)

    # Compute hydrodynamic velocity metrics across ALL test snapshots
    vel_metrics = compute_comprehensive_velocity_metrics(
        U_pred, U_test, channel_dim=1
    )
    print(f"  Tier C Velocity Relative L2 Error: {vel_metrics['global_rel_l2']*100:.2f}%")
    print(f"  Tier C Velocity R2 Score: {vel_metrics['global_r2']:.4f}")
    print(f"  Vector Cosine Alignment: {vel_metrics['angular_alignment']['mean_cos_theta']:.4f}")

    # Compute gas holdup error across ALL test snapshots
    alpha_diff = (Alpha_pred[:, 0] - Alpha_test[:, 0]) * mask[None, ...]
    alpha_true_m = Alpha_test[:, 0] * mask[None, ...]
    alpha_l2_err = float(np.linalg.norm(alpha_diff) / (np.linalg.norm(alpha_true_m) + 1e-12))
    print(f"  Tier C Gas Holdup (alpha_g) Relative Error: {alpha_l2_err*100:.2f}%")

    # 6. First-Principles Two-Phase BiRD QoI Evaluation (averaged over test snapshots)
    print(f"\n[5/5] Computing First-Principles BiRD Multiphase QoIs (Higbie Mass Transfer)...")
    ug_rise = np.zeros_like(U_test[0])
    ug_rise[2] = 0.20

    kla_true_list, kla_pred_list = [], []
    holdup_true_list, holdup_pred_list = [], []
    area_true_list, area_pred_list = [], []
    for i in range(N_test):
        qt = compute_higbie_mass_transfer(
            alpha_gas_grid=Alpha_test[i, 0],
            u_gas_grid=ug_rise,
            u_liquid_grid=U_test[i],
            d_bubble=0.003,
            diffusivity=2.1e-9,
            fluid_mask=mask,
        )
        qp = compute_higbie_mass_transfer(
            alpha_gas_grid=Alpha_pred[i, 0],
            u_gas_grid=ug_rise,
            u_liquid_grid=U_pred[i],
            d_bubble=0.003,
            diffusivity=2.1e-9,
            fluid_mask=mask,
        )
        kla_true_list.append(qt["kla_volume_averaged_h_inv"])
        kla_pred_list.append(qp["kla_volume_averaged_h_inv"])
        holdup_true_list.append(qt["mean_gas_holdup"])
        holdup_pred_list.append(qp["mean_gas_holdup"])
        area_true_list.append(qt["mean_interfacial_area_m2_m3"])
        area_pred_list.append(qp["mean_interfacial_area_m2_m3"])

    kla_true_h = float(np.mean(kla_true_list))
    kla_pred_h = float(np.mean(kla_pred_list))
    kla_err = abs(kla_pred_h - kla_true_h) / (kla_true_h + 1e-12)
    holdup_true_m = float(np.mean(holdup_true_list))
    holdup_pred_m = float(np.mean(holdup_pred_list))
    area_true_m = float(np.mean(area_true_list))
    area_pred_m = float(np.mean(area_pred_list))

    dim_nums = compute_dimensionless_bioprocess_numbers(
        N_rpm=test_rpm, D_impeller=spec.Da, d_bubble=0.003, u_slip=0.20
    )

    print(f"  Ground Truth Volume-Averaged kLa: {kla_true_h:.2f} h^-1")
    print(f"  Predicted Volume-Averaged kLa:    {kla_pred_h:.2f} h^-1")
    print(f"  Relative kLa Error:               {kla_err*100:.2f}%")
    print(f"  Mean Gas Holdup <alpha_g>:        {holdup_pred_m*100:.2f}% (True: {holdup_true_m*100:.2f}%)")
    print(f"  Impeller Reynolds Number (Re):    {dim_nums['reynolds_impeller']:.1f}")
    print(f"  Sherwood Number (Sh):             {dim_nums['sherwood_number']:.1f}")

    # 7. Export JSON Report
    report_data = {
        "benchmark_name": "BiRD_Multiphase_Neural_POD_Benchmark",
        "data_source": "analytical_divergence_free_proxy",
        "reactor_spec": {
            "tank_diameter_m": spec.Dt,
            "impeller_diameter_m": spec.Da,
            "liquid_height_m": spec.H,
            "nbaffles": spec.nbaffles,
        },
        "operating_conditions": {
            "train_rpms": train_rpms,
            "test_rpm": test_rpm,
            "n_test_snapshots": N_test,
        },
        "neural_pod_operator": {
            "n_modes": n_modes,
            "channel_normalized": True,
            "cumulative_energy_pct": calib_stats["cumulative_energy_pct"],
            "theoretical_reconstruction_accuracy_pct": calib_stats["theoretical_reconstruction_accuracy_pct"],
        },
        "tier_c_generalization_results": {
            "velocity_relative_l2_error": vel_metrics["global_rel_l2"],
            "velocity_r2_score": vel_metrics["global_r2"],
            "vector_cosine_similarity": vel_metrics["angular_alignment"]["mean_cos_theta"],
            "gas_holdup_relative_l2_error": alpha_l2_err,
        },
        "first_principles_qois": {
            "kla_true_h_inv": kla_true_h,
            "kla_predicted_h_inv": kla_pred_h,
            "kla_relative_error": kla_err,
            "mean_gas_holdup_true": holdup_true_m,
            "mean_gas_holdup_pred": holdup_pred_m,
            "mean_interfacial_area_true_m2_m3": area_true_m,
            "mean_interfacial_area_pred_m2_m3": area_pred_m,
            "dimensionless_numbers": dim_nums,
        },
    }

    report_path = BASE_DIR / "reports" / "multiphase_bird_operator_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[+] Successfully exported validation report to {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
