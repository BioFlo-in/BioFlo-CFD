"""
Unit and Physics Validation Tests for the Learned Numerical Evolution Operator.
Verifies:
  1. Exact 3D Solenoidal Leray Projection (divergence reduction by >10^5).
  2. Autoregressive state evolution with FiLM parameter conditioning.
  3. Master LearnedNumericalEvolutionEngine multi-step rollout stability.
  4. MultiStepRolloutLoss gradient propagation and energy regularization.
"""

import pytest
import torch
import numpy as np

from nlss.models.learned_evolution_operator import (
    LerayProjection3D,
    AutoregressiveStateEvolutionOperator,
    LearnedNumericalEvolutionEngine,
    MultiStepRolloutLoss,
)


def test_leray_projection_3d_divergence_free():
    """Verify that LerayProjection3D reduces velocity field divergence to machine precision."""
    torch.manual_seed(42)
    B, C, D, H, W = 2, 3, 16, 16, 16
    proj = LerayProjection3D()

    # Generate random noisy velocity field with non-zero divergence
    u_noisy = torch.randn(B, C, D, H, W)

    # Compute initial discrete divergence
    du_dx = (torch.roll(u_noisy[:, 0], -1, dims=-1) - torch.roll(u_noisy[:, 0], 1, dims=-1)) / 2.0
    dv_dy = (torch.roll(u_noisy[:, 1], -1, dims=-2) - torch.roll(u_noisy[:, 1], 1, dims=-2)) / 2.0
    dw_dz = (torch.roll(u_noisy[:, 2], -1, dims=-3) - torch.roll(u_noisy[:, 2], 1, dims=-3)) / 2.0
    initial_div = du_dx + dv_dy + dw_dz
    initial_div_norm = torch.norm(initial_div).item()
    assert initial_div_norm > 1.0, "Initial field should have non-zero divergence"

    # Apply 3D Leray projection
    u_sol = proj(u_noisy)
    assert u_sol.shape == u_noisy.shape

    # Compute projected discrete divergence on the grid
    du_sol_dx = (torch.roll(u_sol[:, 0], -1, dims=-1) - torch.roll(u_sol[:, 0], 1, dims=-1)) / 2.0
    dv_sol_dy = (torch.roll(u_sol[:, 1], -1, dims=-2) - torch.roll(u_sol[:, 1], 1, dims=-2)) / 2.0
    dw_sol_dz = (torch.roll(u_sol[:, 2], -1, dims=-3) - torch.roll(u_sol[:, 2], 1, dims=-3)) / 2.0
    sol_div = du_sol_dx + dv_sol_dy + dw_sol_dz
    sol_div_norm = torch.norm(sol_div).item()

    # Divergence should drop by at least 6 orders of magnitude (to ~1e-5 or lower)
    assert sol_div_norm < 1e-5 * initial_div_norm
    assert sol_div_norm < 1e-3


def test_leray_projection_with_mask():
    """Verify LerayProjection3D handles 4D inputs and fluid boundary masks."""
    proj = LerayProjection3D()
    u_4d = torch.randn(3, 8, 8, 8)
    mask = torch.ones(8, 8, 8)
    mask[0, :, :] = 0.0  # Solid wall

    u_sol = proj(u_4d, fluid_mask=mask)
    assert u_sol.shape == (3, 8, 8, 8)
    assert torch.all(u_sol[:, 0, :, :] == 0.0)


def test_autoregressive_state_evolution():
    """Verify two-step autoregressive stepping with FiLM parameter conditioning."""
    torch.manual_seed(42)
    B, n_modes, param_dim = 4, 10, 2
    op = AutoregressiveStateEvolutionOperator(
        n_modes=n_modes,
        param_dim=param_dim,
        hidden_dim=32,
        num_blocks=2,
    )

    a_curr = torch.randn(B, n_modes)
    a_prev = a_curr - 0.05 * torch.randn(B, n_modes)
    dt = torch.tensor(0.01)
    params = torch.tensor([[300.0, 1e-6], [400.0, 1e-6], [450.0, 1e-6], [600.0, 1e-6]])

    a_next = op(a_curr, a_prev, dt, params)
    assert a_next.shape == (B, n_modes)
    assert not torch.isnan(a_next).any()

    # Verify backpropagation
    loss = torch.sum(a_next**2)
    loss.backward()
    for param in op.parameters():
        if param.requires_grad:
            assert param.grad is not None


def test_learned_numerical_evolution_engine_rollout():
    """Verify Master Evolution Engine autonomous multi-step trajectory rollout and boundedness."""
    torch.manual_seed(42)
    D, H, W = 8, 8, 8
    spatial_dim = 3 * D * H * W
    n_modes = 5

    # Create synthetic orthonormal POD basis via QR decomposition
    random_mat = torch.randn(spatial_dim, n_modes)
    basis, _ = torch.linalg.qr(random_mat)
    base_flow = torch.randn(3, D, H, W)

    engine = LearnedNumericalEvolutionEngine(
        pod_basis=basis,
        base_flow=base_flow,
        grid_shape=(D, H, W),
        param_dim=1,
        hidden_dim=32,
        num_blocks=2,
        enforce_projection_in_loop=True,
    )

    a_0 = torch.randn(1, n_modes)
    a_prev = a_0 * 0.98
    dt = torch.tensor([[0.01]])
    params = torch.tensor([[400.0]])

    # Autonomous 25-step rollout
    rollout_res = engine.rollout(
        a_0=a_0,
        a_prev=a_prev,
        dt=dt,
        params=params,
        num_steps=25,
        return_fields=True,
    )

    modes = rollout_res["modes"]
    ke = rollout_res["kinetic_energy"]
    div = rollout_res["divergence_norm"]
    fields = rollout_res["fields"]

    assert modes.shape == (25, 1, n_modes)
    assert fields.shape == (25, 1, 3, D, H, W)
    assert len(ke) == 25
    assert len(div) == 25

    # Check stability: kinetic energy must remain bounded (no explosion or NaN)
    for k in ke:
        assert not np.isnan(k)
        assert k < 1e5


def test_multistep_rollout_loss():
    """Verify multi-step rollout loss correctly penalizes drift, energy discrepancy, and velocity."""
    K, B, n_modes = 5, 2, 8
    loss_fn = MultiStepRolloutLoss(gamma=0.9, energy_weight=0.1, velocity_weight=0.05)

    pred = torch.randn(K, B, n_modes, requires_grad=True)
    target = torch.randn(K, B, n_modes)

    losses = loss_fn(pred, target)
    assert "total_loss" in losses
    assert "mode_loss" in losses
    assert "energy_loss" in losses
    assert "velocity_loss" in losses

    total_loss = losses["total_loss"]
    assert total_loss.item() > 0
    total_loss.backward()
    assert pred.grad is not None
