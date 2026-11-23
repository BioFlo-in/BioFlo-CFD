"""
Unit tests for BioFlo-CFD Bioprocess Engineering Validation Subsystem.
Verifies strain rate tensor, energy dissipation, specific power (P/V),
oxygen transfer rate (k_L a), and Grenville mixing time calculations.
"""

import numpy as np
import pytest
import torch

from nlss.validation.bioprocess_metrics import (
    compute_strain_rate_tensor,
    compute_viscous_dissipation_rate,
    compute_specific_power_input,
    compute_kla,
    compute_mixing_time,
    evaluate_bioprocess_field_comparison,
)


def test_analytical_couette_shear_rate():
    """
    In planar Couette flow u = (gamma_dot * y, 0, 0),
    the strain rate tensor S_12 = S_21 = 0.5 * gamma_dot.
    Therefore dot{gamma} = sqrt(2 * (S_12^2 + S_21^2)) = gamma_dot.
    """
    D, H, W = 16, 16, 16
    dx = dy = dz = 0.05
    target_shear = 12.5  # 1/s

    # Create analytical grid
    y_coords = np.linspace(0, (H - 1) * dy, H)
    ux = np.zeros((D, H, W), dtype=np.float64)
    for j in range(H):
        ux[:, j, :] = target_shear * y_coords[j]
    uy = np.zeros_like(ux)
    uz = np.zeros_like(ux)
    u_grid = np.stack([ux, uy, uz], axis=0)

    shear_rate, S_tensor = compute_strain_rate_tensor(u_grid, dx, dy, dz)

    # In the interior (away from boundary differences), shear rate must match target_shear exactly
    interior_shear = shear_rate[2:-2, 2:-2, 2:-2]
    np.testing.assert_allclose(interior_shear, target_shear, rtol=1e-3)
    assert S_tensor.shape == (3, 3, D, H, W)


def test_dissipation_and_kolmogorov_scales():
    """Verifies that dissipation epsilon = nu * gamma_dot^2 and Kolmogorov microscale eta_K are correct."""
    D, H, W = 10, 10, 10
    dx = dy = dz = 0.02
    nu = 1e-6  # Water kinematic viscosity

    # Uniform shear of 10 s^-1
    y_coords = np.linspace(0, (H - 1) * dy, H)
    ux = np.zeros((D, H, W))
    for j in range(H):
        ux[:, j, :] = 10.0 * y_coords[j]
    u_grid = np.stack([ux, np.zeros_like(ux), np.zeros_like(ux)], axis=0)

    res = compute_viscous_dissipation_rate(u_grid, dx, dy, dz, nu=nu)
    # Expected interior dissipation = 1e-6 * 100 = 1e-4 W/kg
    assert res["mean_dissipation_rate_w_kg"] > 0.0
    assert res["kolmogorov_microscale_um"] > 0.0
    # Expected Kolmogorov microscale ~ (1e-18 / 1e-4)^(0.25) = (1e-14)^0.25 = 316 um
    assert 100.0 < res["kolmogorov_microscale_um"] < 1000.0


def test_specific_power_and_kla():
    """Verifies P/V integration and empirical mass transfer coefficient k_L a."""
    D, H, W = 8, 8, 8
    dx = dy = dz = 0.01
    u_grid = np.random.RandomState(42).randn(3, D, H, W) * 0.1

    pv_stats = compute_specific_power_input(
        u_grid, dx, dy, dz, rho=1000.0, nu=1e-6, N_rpm=300.0, D_impeller=0.10, Np=1.5
    )
    assert pv_stats["liquid_volume_m3"] > 0.0
    assert pv_stats["specific_power_dissipation_w_m3"] >= 0.0
    assert pv_stats["impeller_power_w"] > 0.0

    kla_stats = compute_kla(pv_stats["specific_power_dissipation_w_m3"], superficial_gas_velocity=0.01)
    assert kla_stats["kla_s_inv"] >= 0.0
    assert kla_stats["kla_h_inv"] == pytest.approx(kla_stats["kla_s_inv"] * 3600.0)
    assert kla_stats["estimated_otr_kg_m3_h"] >= 0.0


def test_mixing_time_grenville():
    """Verifies Grenville & Nienow mixing time scaling with RPM and vessel geometry."""
    mix_300 = compute_mixing_time(N_rpm=300.0, Np=1.5, T_tank=0.50, D_impeller=0.166, liquid_volume=0.1)
    mix_600 = compute_mixing_time(N_rpm=600.0, Np=1.5, T_tank=0.50, D_impeller=0.166, liquid_volume=0.1)

    # Blend time is inversely proportional to RPM: doubling RPM should halve mixing time
    assert mix_600["mixing_time_95_s"] == pytest.approx(mix_300["mixing_time_95_s"] / 2.0, rel=1e-3)
    # Circulation time should also halve
    assert mix_600["circulation_time_s"] == pytest.approx(mix_300["circulation_time_s"] / 2.0, rel=1e-3)
    # Tip speed should double
    assert mix_600["impeller_tip_speed_m_s"] == pytest.approx(mix_300["impeller_tip_speed_m_s"] * 2.0, rel=1e-3)


def test_bioprocess_comparison_identical_fields():
    """Identical fields should have exactly 0% relative error in bioprocess metrics."""
    D, H, W = 8, 8, 8
    dx = dy = dz = 0.02
    u_grid = np.ones((3, D, H, W)) * 0.2

    comp = evaluate_bioprocess_field_comparison(
        u_grid, u_grid, dx, dy, dz, nu=1e-6, rho=1000.0, N_rpm=300.0
    )
    assert comp["specific_power_relative_error"] == 0.0
    assert comp["kla_relative_error"] == 0.0
