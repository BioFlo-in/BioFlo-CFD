"""
Unit tests for the Stirred-Tank Bioreactor (STR) CFD simulator and bioprocess metrics.
"""

import numpy as np
import pytest

from nlss.dataset.bioreactor import (
    BioreactorSimulator,
    BioreactorGeometry,
    compute_bioprocess_shear_metrics,
)


def test_bioreactor_initialization_and_geometry():
    geom = BioreactorGeometry(T=1.0, H=1.0)
    sim = BioreactorSimulator(nx=32, ny=32, geom=geom)
    
    assert sim.shaft_mask.shape == (32, 32)
    assert sim.impeller_mask.shape == (32, 32)
    assert sim.shaft_mask.sum() > 0
    assert sim.impeller_mask.sum() > 0
    # Impeller and shaft must be mutually exclusive in active forcing
    overlap = sim.shaft_mask & sim.impeller_mask
    assert overlap.sum() == 0


def test_bioreactor_step_and_conservation():
    sim = BioreactorSimulator(nx=32, ny=32, dt=0.005, u_tip=1.0)
    
    # Run 5 steps
    for _ in range(5):
        u, v, p = sim.step()

    assert not np.isnan(u).any()
    assert not np.isnan(v).any()
    assert not np.isnan(p).any()

    # Free surface slip: v[-1, :] == 0
    np.testing.assert_allclose(v[-1, :], 0.0, atol=1e-5)
    # Bottom wall: u[0, :] == 0, v[0, :] == 0
    np.testing.assert_allclose(u[0, :], 0.0, atol=1e-5)
    np.testing.assert_allclose(v[0, :], 0.0, atol=1e-5)


def test_bioprocess_shear_metrics():
    nx, ny = 32, 32
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)

    # Linear shear flow: u = y, v = 0 -> du/dy = 1
    y = np.linspace(0, 1, ny)[:, None]
    u[:] = y

    metrics = compute_bioprocess_shear_metrics(u, v, nu=1e-3, rho=1000.0)
    assert "shear_rate" in metrics
    assert "shear_stress" in metrics
    assert "dissipation_rate" in metrics
    assert "kolmogorov_length" in metrics
    assert metrics["tau_max_pa"] > 0
    assert metrics["epsilon_max_w_kg"] > 0
    assert "cell_damage_risk" in metrics
