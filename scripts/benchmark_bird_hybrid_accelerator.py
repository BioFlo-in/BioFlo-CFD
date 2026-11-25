"""
BioFlo-CFD & BiRD: Phase 4 Closed-Loop Hybrid Solver Acceleration Benchmark.
Benchmarks actual matrix-free 3D discrete Poisson Conjugate Gradient (CG) convergence
(Cold Start x_0=0 vs. trained NLSS Warm Start) and exports valid OpenFOAM warm-start fields.
Note: Uses an actual 3D discrete Poisson CG solver proxy when OpenFOAM binaries are absent.
"""

import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from nlss.dataset.bird_mesh_adapter import StirredTankSpec
from nlss.io.foam_writer import export_multiphase_simulation_state_to_openfoam
from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from scripts.train_multiphase_bird_operator import generate_synthetic_bird_multiphase_snapshots


def _apply_discrete_neg_laplacian_3d(
    p: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    fluid_mask: np.ndarray,
) -> np.ndarray:
    """
    Applies the symmetric positive-semidefinite discrete operator A*p = -div(grad(p))
    on the active 3D fluid domain with homogeneous Neumann BCs at solid walls.
    """
    out = np.zeros_like(p)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    # X-direction flux across active fluid faces
    fx = fluid_mask[:, :, :-1] & fluid_mask[:, :, 1:]
    diff_x = (p[:, :, 1:] - p[:, :, :-1]) * fx * inv_dx2
    out[:, :, :-1] -= diff_x
    out[:, :, 1:] += diff_x

    # Y-direction flux
    fy = fluid_mask[:, :-1, :] & fluid_mask[:, 1:, :]
    diff_y = (p[:, 1:, :] - p[:, :-1, :]) * fy * inv_dy2
    out[:, :-1, :] -= diff_y
    out[:, 1:, :] += diff_y

    # Z-direction flux
    fz = fluid_mask[:-1, :, :] & fluid_mask[1:, :, :]
    diff_z = (p[1:, :, :] - p[:-1, :, :]) * fz * inv_dz2
    out[:-1, :, :] -= diff_z
    out[1:, :, :] += diff_z

    return out * fluid_mask


def solve_3d_poisson_cg(
    rhs: np.ndarray,
    x0: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    fluid_mask: np.ndarray,
    tol: float = 1e-6,
    max_iters: int = 500,
) -> dict:
    """
    Actual matrix-free Conjugate Gradient solver for 3D discrete pressure Poisson equation:
        -div(grad(p)) = rhs
    """
    t0 = time.perf_counter()
    x = (x0 * fluid_mask).astype(np.float64)
    b = (rhs * fluid_mask).astype(np.float64)
    b_norm = float(np.linalg.norm(b)) + 1e-12

    r = b - _apply_discrete_neg_laplacian_3d(x, dx, dy, dz, fluid_mask)
    initial_res_norm = float(np.linalg.norm(r))
    initial_rel_res = initial_res_norm / b_norm

    p_dir = r.copy()
    rz_old = float(np.sum(r * r))
    iters = 0

    while (np.sqrt(rz_old) / b_norm) > tol and iters < max_iters:
        Ap = _apply_discrete_neg_laplacian_3d(p_dir, dx, dy, dz, fluid_mask)
        denom = float(np.sum(p_dir * Ap))
        if denom < 1e-15:
            break
        alpha = rz_old / denom
        x = x + alpha * p_dir
        r = r - alpha * Ap
        rz_new = float(np.sum(r * r))
        beta = rz_new / (rz_old + 1e-30)
        p_dir = r + beta * p_dir
        rz_old = rz_new
        iters += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    final_rel_res = float(np.sqrt(rz_old) / b_norm)
    return {
        "iterations": iters,
        "initial_rel_residual": initial_rel_res,
        "final_rel_residual": final_rel_res,
        "solver_time_ms": elapsed_ms,
        "solution": x,
    }


def main():
    print("=" * 80)
    print("BioFlo-CFD & BiRD: Phase 4 Closed-Loop Hybrid Accelerator Benchmark")
    print("  [Solver Mode: Actual 3D Matrix-Free Discrete Poisson CG Benchmark]")
    print("=" * 80)

    spec = StirredTankSpec(Dt=0.30, Da=0.10, H=0.45, C=[0.15], nbaffles=4)
    grid_shape = (16, 16, 16)
    D, H, W = grid_shape
    dx, dy, dz = spec.Dt / W, spec.Dt / H, spec.H / D
    case_dir = BASE_DIR / "openfoam_cases" / "bird_stirred_tank"

    # 1. Train 4-channel Neural POD Operator on [300, 600] RPM
    train_rpms = [300.0, 600.0]
    test_rpm = 450.0

    print(f"\n[1/4] Training 4-channel Neural POD Operator on BiRD multiphase fields...")
    U_train, Alpha_train, p_train, mask = generate_synthetic_bird_multiphase_snapshots(
        spec, grid_shape=grid_shape, n_snapshots_per_rpm=15, rpm_list=train_rpms, seed=42
    )
    U_test, Alpha_test, p_test, _ = generate_synthetic_bird_multiphase_snapshots(
        spec, grid_shape=grid_shape, n_snapshots_per_rpm=15, rpm_list=[test_rpm], seed=99
    )

    X_train = np.concatenate([U_train, Alpha_train], axis=1)
    N_train, channels, D, H, W = X_train.shape
    channel_scales = np.maximum(np.std(X_train, axis=(0, 2, 3, 4)), 1e-4)
    X_train_scaled = X_train / channel_scales.reshape(1, 4, 1, 1, 1)
    spatial_dim = channels * D * H * W

    operator = NeuralPODOperator3D(n_modes=12, param_dim=1, hidden_dim=64, num_layers=3)
    operator.calibrate_basis(X_train_scaled.reshape(N_train, spatial_dim), n_modes=12, solver="exact")
    operator.set_training_parameters(p_train, strict=False)

    # Actually train the MLP before benchmarking
    optimizer = torch.optim.AdamW(operator.mlp.parameters(), lr=2e-3, weight_decay=1e-5)
    p_train_t = torch.tensor(p_train, dtype=torch.float32)
    X_train_t = torch.tensor(X_train_scaled.reshape(N_train, spatial_dim), dtype=torch.float32)
    a_opt_train = torch.matmul(X_train_t - operator.mean_field, operator.basis_matrix.T)
    for _ in range(250):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(operator.predict_coefficients(p_train_t), a_opt_train)
        loss.backward()
        optimizer.step()
    operator.eval()

    # 2. Run fast AI inference for unobserved 450 RPM
    print(f"\n[2/4] Executing sub-2ms AI surrogate prediction for held-out {test_rpm} RPM...")
    p_query = torch.tensor([[test_rpm]], dtype=torch.float32)

    t_start = time.perf_counter()
    for _ in range(50):
        with torch.no_grad():
            X_pred_flat = operator(p_query)
    t_infer_ms = (time.perf_counter() - t_start) / 50.0 * 1000.0

    X_pred_scaled = X_pred_flat.view(1, channels, D, H, W).cpu().numpy()[0]
    X_pred = X_pred_scaled * channel_scales.reshape(4, 1, 1, 1)
    u_liquid_pred = X_pred[:3].reshape(3, -1).T
    alpha_gas_pred = np.clip(X_pred[3].flatten(), 0.0, 1.0)
    print(f"  AI Inference Latency: {t_infer_ms:.3f} ms (Target: < 2.0 ms)")

    # 3. Export predicted fields to BiRD case directory as warm-start t=0
    print(f"\n[3/4] Injecting AI warm-start fields into {case_dir.name}...")
    warm_dir = export_multiphase_simulation_state_to_openfoam(
        case_dir=case_dir,
        time_val=0.0,
        u_liquid_pred=u_liquid_pred,
        alpha_gas_pred=alpha_gas_pred,
    )
    assert warm_dir.exists()
    assert (warm_dir / "U.liquid").exists()
    assert (warm_dir / "alpha.gas").exists()
    print(f"  Successfully wrote warm-start state to {warm_dir}")

    # 4. Benchmark Real 3D Discrete Poisson CG Convergence: Cold Start vs. AI Warm Start
    print(f"\n[4/4] Benchmarking Actual 3D Poisson CG Convergence (Cold vs. AI Warm Start)...")
    # Construct pressure potential proxy proportional to kinetic energy head 0.5*||u||^2
    p_true_3d = 0.5 * np.sum(U_test[0] ** 2, axis=0) * mask
    p_true_3d -= np.mean(p_true_3d[mask])
    p_pred_3d = 0.5 * np.sum(X_pred[:3] ** 2, axis=0) * mask
    p_pred_3d -= np.mean(p_pred_3d[mask])

    rhs = _apply_discrete_neg_laplacian_3d(p_true_3d, dx, dy, dz, mask)

    cold_solve = solve_3d_poisson_cg(
        rhs=rhs,
        x0=np.zeros_like(p_true_3d),
        dx=dx, dy=dy, dz=dz,
        fluid_mask=mask,
        tol=1e-6,
        max_iters=500,
    )
    warm_solve = solve_3d_poisson_cg(
        rhs=rhs,
        x0=p_pred_3d,
        dx=dx, dy=dy, dz=dz,
        fluid_mask=mask,
        tol=1e-6,
        max_iters=500,
    )

    t_total_cold = cold_solve["solver_time_ms"]
    t_total_warm = warm_solve["solver_time_ms"] + t_infer_ms
    speedup = t_total_cold / max(t_total_warm, 1e-6)
    res_ratio = cold_solve["initial_rel_residual"] / max(warm_solve["initial_rel_residual"], 1e-12)

    print(f"\n  --- Solver Acceleration Benchmark Results (Measured 3D CG) ---")
    print(f"  Cold Start Initial Residual:        {cold_solve['initial_rel_residual']:.4f}")
    print(f"  Warm Start Initial Residual:        {warm_solve['initial_rel_residual']:.4f} ({res_ratio:.2f}x lower initial residual)")
    print(f"  Cold Start Poisson CG Iterations:   {cold_solve['iterations']} sweeps")
    print(f"  Warm Start Poisson CG Iterations:   {warm_solve['iterations']} sweeps ({cold_solve['iterations'] - warm_solve['iterations']} fewer sweeps)")
    print(f"  Cold Start Solver Time:             {t_total_cold:.2f} ms")
    print(f"  Warm Start Total Time (AI + Solver):{t_total_warm:.2f} ms (Inference: {t_infer_ms:.2f} ms)")
    print(f"  Net Solver Acceleration Speedup:    {speedup:.2f}x")

    # Save benchmark report
    report = {
        "benchmark": "Phase4_Hybrid_Accelerator_Benchmark",
        "solver_mode": "3d_discrete_poisson_cg_proxy",
        "case": case_dir.name,
        "operating_condition_rpm": test_rpm,
        "ai_inference_latency_ms": t_infer_ms,
        "cold_start": {
            "initial_residual": cold_solve["initial_rel_residual"],
            "iterations": cold_solve["iterations"],
            "time_ms": cold_solve["solver_time_ms"],
        },
        "warm_start": {
            "initial_residual": warm_solve["initial_rel_residual"],
            "iterations": warm_solve["iterations"],
            "solver_time_ms": warm_solve["solver_time_ms"],
            "total_time_ms": t_total_warm,
        },
        "performance_gain": {
            "residual_reduction_factor": res_ratio,
            "iteration_reduction_pct": ((cold_solve["iterations"] - warm_solve["iterations"]) / max(cold_solve["iterations"], 1)) * 100.0,
            "net_speedup_factor": speedup,
        },
    }

    report_path = BASE_DIR / "reports" / "phase4_hybrid_accelerator_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Saved Phase 4 acceleration report to {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
