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
    compute_higbie_mass_transfer,
    compute_tracer_mixing_time_t95,
    compute_dimensionless_bioprocess_numbers,
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


def test_higbie_mass_transfer():
    """Verify first-principles Higbie mass transfer formulation ported from BiRD."""
    D, H, W = 8, 8, 8
    # Constant gas holdup alpha_g = 0.05 (5%)
    alpha_g = np.full((D, H, W), 0.05, dtype=np.float64)

    # Relative velocity: gas rising at 0.25 m/s, liquid at 0.05 m/s -> u_slip = 0.20 m/s
    ug = np.zeros((3, D, H, W), dtype=np.float64)
    ug[2] = 0.25
    ul = np.zeros((3, D, H, W), dtype=np.float64)
    ul[2] = 0.05

    d_b = 0.003  # 3 mm Sauter diameter
    D_L = 2.1e-9  # O2 in water

    res = compute_higbie_mass_transfer(
        alpha_gas_grid=alpha_g,
        u_gas_grid=ug,
        u_liquid_grid=ul,
        d_bubble=d_b,
        diffusivity=D_L,
    )

    # Analytical values:
    # a = 6 * alpha_g / d_b = 6 * 0.05 / 0.003 = 100 m^2/m^3
    assert res["mean_interfacial_area_m2_m3"] == pytest.approx(100.0, rel=1e-3)
    # k_L = (2 / sqrt(pi)) * sqrt(D_L * u_rel / d_b)
    expected_kl = (2.0 / np.sqrt(np.pi)) * np.sqrt(D_L * 0.20 / d_b)
    assert res["mean_kl_m_s"] == pytest.approx(expected_kl, rel=1e-3)

    # Local kla = k_L * a = 4.2219e-4 * 100 = 0.0422 s^-1
    expected_kla_s = expected_kl * 100.0
    # Integrated over liquid volume (1 - alpha_g = 0.95):
    expected_kla_integrated = expected_kla_s / 0.95
    assert res["kla_volume_averaged_s_inv"] == pytest.approx(expected_kla_integrated, rel=1e-2)
    assert res["kla_volume_averaged_h_inv"] == pytest.approx(expected_kla_integrated * 3600.0, rel=1e-2)
    assert res["mean_gas_holdup"] == pytest.approx(0.05, rel=1e-3)


def test_tracer_mixing_time_t95():
    """Verify empirical 95% tracer homogenization curve and t95 detection."""
    T, D, H, W = 20, 4, 4, 4
    time_points = np.linspace(0.0, 10.0, T)
    tau_decay = 2.0  # characteristic mixing decay time

    # Generate synthetic decaying concentration field: c(x, t) = 1.0 + perturbation * exp(-t / tau)
    np.random.seed(42)
    base_pert = np.random.randn(D, H, W)
    base_pert -= np.mean(base_pert)
    # Scale initial perturbation so initial CoV is ~ 0.50
    base_pert = 0.50 * base_pert / np.std(base_pert)

    tracer_series = np.zeros((T, D, H, W))
    for t_idx, t_val in enumerate(time_points):
        decay = np.exp(-t_val / tau_decay)
        tracer_series[t_idx] = 1.0 + base_pert * decay

    res = compute_tracer_mixing_time_t95(tracer_series, time_points, target_band=0.05)

    assert res["is_well_mixed"] is True
    # Initial CoV ~ 0.50, target is 0.05 -> t_95 ~ tau * ln(10) ~ 4.6s
    assert 4.0 <= res["mixing_time_95_s"] <= 5.5
    assert res["final_cov"] < 0.05
    assert len(res["cov_history"]) == T


def test_dimensionless_bioprocess_numbers():
    """Verify dimensionless numbers matching BiRD standard definitions."""
    nums = compute_dimensionless_bioprocess_numbers(
        N_rpm=300.0,
        D_impeller=0.10,
        rho_liquid=1000.0,
        mu_liquid=1e-3,
        surface_tension=0.0728,
        d_bubble=0.003,
        u_slip=0.20,
        diffusivity=2.1e-9,
    )

    # Re_imp = 1000 * 5 * 0.01 / 1e-3 = 50,000 (turbulent)
    assert nums["reynolds_impeller"] == pytest.approx(50000.0, rel=1e-3)
    # Fr = 25 * 0.1 / 9.81 = 0.2548
    assert nums["froude_impeller"] == pytest.approx(2.5 / 9.81, rel=1e-3)
    # Re_bubble = 1000 * 0.2 * 0.003 / 1e-3 = 600
    assert nums["reynolds_bubble"] == pytest.approx(600.0, rel=1e-3)
    # Sc = (1e-3 / 1000) / 2.1e-9 = 476.19
    assert nums["schmidt_number"] == pytest.approx(476.19, rel=1e-2)
    # Sh = (2 / sqrt(pi)) * sqrt(600 * 476.19) > 500
    assert nums["sherwood_number"] > 500.0

