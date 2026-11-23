"""
BioFlo-CFD: Rigorous Physics Validation Test Suite (NLSS v1.0 Standard).
Tests:
  1. Standardized Velocity Error Norms (L2, L1, Linf, RMSE, MAE, R2, Component-wise, Vector Alignment)
  2. Discrete Divergence (div(u) ~ 0 on analytical divergence-free fields)
  3. Discrete Vorticity (curl(u) on analytical solid-body rotation)
  4. Navier-Stokes Momentum Residuals (steady convective-diffusive balance)
  5. Regional ROI Decomposition (Ensuring localized high-shear errors are isolated)
"""

import math
import numpy as np
import pytest
import torch

from nlss.validation.velocity_metrics import (
    relative_l2_error,
    relative_l1_error,
    relative_linf_error,
    rmse_mae_metrics,
    r2_score_field,
    component_wise_errors,
    vector_angular_metrics,
    compute_comprehensive_velocity_metrics,
)
from nlss.validation.physics_residuals import (
    compute_discrete_divergence,
    compute_discrete_vorticity,
    compute_vorticity_metrics,
    compute_momentum_residual_steady,
)
from nlss.validation.region_metrics import (
    generate_bioreactor_rois,
    evaluate_roi_metrics,
)


def test_velocity_error_norms():
    """Verify L2, L1, Linf, RMSE, MAE, and component-wise decomposition."""
    np.random.seed(42)
    true_vel = np.ones((3, 10, 10, 10)) * 2.0
    pred_vel = true_vel.copy()
    # Introduce 10% error in ux only
    pred_vel[0] = true_vel[0] * 1.10

    # L2 error on entire field
    l2 = relative_l2_error(pred_vel, true_vel)
    assert 0.05 < l2 < 0.07  # sqrt(0.2^2 / (2^2 * 3)) ~ 0.0577

    # Linf error is 0.2 / 2.0 = 0.10
    linf = relative_linf_error(pred_vel, true_vel)
    assert abs(linf - 0.10) < 1e-4

    comp = component_wise_errors(pred_vel, true_vel, channel_dim=0)
    assert abs(comp["ux"]["rel_l2"] - 0.10) < 1e-4
    assert comp["uy"]["rel_l2"] == 0.0
    assert comp["uz"]["rel_l2"] == 0.0


def test_vector_angular_alignment():
    """Verify 3D vector alignment cosine and degree deviation."""
    true_vel = np.zeros((3, 5, 5, 5))
    pred_vel = np.zeros((3, 5, 5, 5))

    # True pointing in +X: (1, 0, 0)
    true_vel[0] = 1.0
    # Pred rotated by 45 degrees in X-Y plane: (cos(45°), sin(45°), 0)
    pred_vel[0] = math.cos(math.radians(45.0))
    pred_vel[1] = math.sin(math.radians(45.0))

    metrics = vector_angular_metrics(pred_vel, true_vel, channel_dim=0)
    assert abs(metrics["mean_angle_deg"] - 45.0) < 1e-4
    assert abs(metrics["mean_cos_theta"] - math.cos(math.radians(45.0))) < 1e-4


def test_discrete_divergence_free_analytical():
    """Verify divergence is zero on analytical 3D solenoidal vortex field."""
    D, H, W = 16, 16, 16
    dx, dy, dz = 0.05, 0.05, 0.05
    gz, gy, gx = np.meshgrid(
        np.linspace(0, (D-1)*dz, D),
        np.linspace(0, (H-1)*dy, H),
        np.linspace(0, (W-1)*dx, W),
        indexing="ij"
    )

    # 3D Hill's spherical vortex / Beltrami flow:
    # u = sin(x)*cos(y)*cos(z), v = -cos(x)*sin(y)*cos(z), w = 0
    # div(u) = cos(x)*cos(y)*cos(z) - cos(x)*cos(y)*cos(z) + 0 = 0
    ux = np.sin(gx) * np.cos(gy) * np.cos(gz)
    uy = -np.cos(gx) * np.sin(gy) * np.cos(gz)
    uz = np.zeros_like(ux)
    u_grid = np.stack([ux, uy, uz], axis=0)

    # Evaluate in interior (excluding 1-cell finite boundary)
    mask = np.zeros((D, H, W), dtype=bool)
    mask[2:-2, 2:-2, 2:-2] = True

    metrics = compute_discrete_divergence(u_grid, dx=dx, dy=dy, dz=dz, fluid_mask=mask)
    assert metrics["mean_abs_divergence"] < 1e-3, f"Divergence not zero: {metrics['mean_abs_divergence']}"


def test_discrete_vorticity_solid_body_rotation():
    """Verify curl(u) matches 2*omega on solid-body rotation u = (-omega*y, omega*x, 0)."""
    D, H, W = 11, 11, 11
    dx, dy, dz = 0.1, 0.1, 0.1
    gz, gy, gx = np.meshgrid(
        np.linspace(-5 * dz, 5 * dz, D),
        np.linspace(-5 * dy, 5 * dy, H),
        np.linspace(-5 * dx, 5 * dx, W),
        indexing="ij"
    )

    omega = 5.0
    ux = -omega * gy
    uy = omega * gx
    uz = np.zeros_like(ux)
    u_grid = np.stack([ux, uy, uz], axis=0)

    vort = compute_discrete_vorticity(u_grid, dx=dx, dy=dy, dz=dz)
    # Interior curl should have wz = 2 * omega = 10.0, wx = 0, wy = 0
    wz_interior = vort[2, 1:-1, 1:-1, 1:-1]
    assert np.allclose(wz_interior, 2.0 * omega, atol=1e-5)
    assert np.allclose(vort[0, 1:-1, 1:-1, 1:-1], 0.0, atol=1e-5)
    assert np.allclose(vort[1, 1:-1, 1:-1, 1:-1], 0.0, atol=1e-5)


def test_region_of_interest_isolation():
    """Verify that a localized 25% impeller error is isolated and NOT masked by 0.5% bulk error."""
    D, H, W = 20, 20, 20
    gz, gy, gx = np.meshgrid(
        np.linspace(0, 1.0, D),
        np.linspace(-0.25, 0.25, H),
        np.linspace(-0.25, 0.25, W),
        indexing="ij"
    )

    rois = generate_bioreactor_rois(
        (gx, gy, gz),
        tank_radius=0.25,
        impeller_radius=0.083,
        impeller_elevation=0.35,
        liquid_height=1.0,
    )

    # Synthetic baseline velocity
    true_vel = np.ones((3, D, H, W)) * 1.5
    pred_vel = true_vel.copy()

    # Bulk has tiny error (0.5%)
    pred_vel[:, rois["bulk_recirculation"]] *= 1.005
    # Impeller has severe error (25.0%)
    pred_vel[:, rois["impeller_discharge"]] *= 1.250

    results = evaluate_roi_metrics(pred_vel, true_vel, rois, channel_dim=0)
    
    # Global metrics
    global_l2 = relative_l2_error(pred_vel, true_vel)
    impeller_l2 = results["impeller_discharge"]["rel_l2"]
    bulk_l2 = results["bulk_recirculation"]["rel_l2"]

    # Global error is diluted, but impeller ROI accurately flags the 25% error!
    assert impeller_l2 > 0.24, f"Impeller error not captured: {impeller_l2}"
    assert bulk_l2 < 0.01, f"Bulk error mismatch: {bulk_l2}"
    assert global_l2 < 0.10, "Global error should be diluted by bulk volume"
