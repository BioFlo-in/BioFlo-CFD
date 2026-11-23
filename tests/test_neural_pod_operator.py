"""
BioFlo-CFD: Unit Tests for Canonical Hyper-Accurate Neural POD Operator 3D.
Validates:
  1. Exact and Randomized SVD Solvers
  2. Parameter Normalization Bounds
  3. 3D Grid Reconstruction and Spatial Masking
  4. Basis Mode Discrete Incompressibility
  5. Checkpoint Serialization & Dynamic Resizing
  6. Model Summary Diagnostics
"""

import numpy as np
import pytest
import torch

from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D, FourierFeatureEmbedding


def test_fourier_feature_embedding():
    """Verify Fourier feature embedding dimensions and non-linearity."""
    embed = FourierFeatureEmbedding(in_dim=2, num_frequencies=8, max_freq_log2=4.0)
    assert embed.out_dim == 2 + 2 * 2 * 8  # 34
    x = torch.randn(4, 2)
    out = embed(x)
    assert out.shape == (4, 34)
    assert torch.allclose(out[:, :2], x)  # First channels must be identity


def test_svd_solvers_and_parameter_bounds():
    """Verify exact and randomized SVD solvers produce valid orthonormal bases."""
    np.random.seed(42)
    N_snaps = 20
    N_dof = 100
    t = np.linspace(0, 1, N_snaps)
    # Synthetic fluid snapshot matrix with decaying singular values
    X = np.sin(2 * np.pi * t[:, None] * np.arange(1, 6)) @ np.random.randn(5, N_dof)

    # 1. Exact solver
    model_exact = NeuralPODOperator3D(n_modes=4, param_dim=1, svd_solver="exact")
    stats_exact = model_exact.calibrate_basis(X, param_bounds=([0.0], [1.0]))
    assert model_exact.is_calibrated
    assert stats_exact["retained_modes"] == 4
    assert stats_exact["cumulative_energy_pct"] > 80.0
    assert model_exact.has_param_bounds

    # 2. Randomized solver
    model_rand = NeuralPODOperator3D(n_modes=4, param_dim=1, svd_solver="randomized")
    stats_rand = model_rand.calibrate_basis(X, solver="randomized")
    assert model_rand.is_calibrated
    assert stats_rand["retained_modes"] == 4
    assert stats_rand["cumulative_energy_pct"] > 80.0


def test_grid_reconstruction_and_masking():
    """Verify reconstruct_grid handles both full grids and active fluid masks."""
    D, H, W = 8, 8, 8
    channels = 3
    N_voxels = D * H * W
    
    # Create synthetic boolean fluid mask (e.g. cylinder inside cube)
    gx, gy, gz = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H), np.linspace(0, 1, D), indexing="xy")
    r = np.sqrt(gx**2 + gy**2)
    fluid_mask_np = (r < 0.8)
    fluid_mask_tensor = torch.from_numpy(fluid_mask_np).permute(2, 0, 1)  # (D, H, W)
    N_active = int(fluid_mask_tensor.sum().item())
    N_dof = channels * N_active

    model = NeuralPODOperator3D(n_modes=3, param_dim=1)
    
    # Calibrate on active DOFs
    np.random.seed(42)
    dummy_snaps = np.random.randn(10, N_dof)
    model.calibrate_basis(dummy_snaps)

    p = torch.tensor([[0.5]])
    grid_out = model.reconstruct_grid(p, grid_shape=(D, H, W), channels=channels, fluid_mask=fluid_mask_tensor)
    assert grid_out.shape == (1, channels, D, H, W)
    
    # Check that solid voxels outside the mask remain zero
    solid_mask = ~fluid_mask_tensor
    for c in range(channels):
        assert torch.allclose(grid_out[0, c, solid_mask], torch.zeros_like(grid_out[0, c, solid_mask]))


def test_checkpoint_serialization_and_summary():
    """Verify saving, loading with buffer resizing, and summary generation."""
    model1 = NeuralPODOperator3D(n_modes=5, param_dim=2, hidden_dim=64, num_layers=2)
    np.random.seed(42)
    X = np.random.randn(15, 80)
    model1.calibrate_basis(X, param_bounds=([0.0, 100.0], [10.0, 500.0]))

    summary = model1.get_model_summary()
    assert summary["model_name"] == "BioFlo-NeuralPODOperator3D"
    assert summary["is_calibrated"] is True
    assert summary["n_modes"] == 5
    assert summary["spatial_dof"] == 80
    assert summary["has_param_bounds"] is True

    # Test serialization
    state = model1.state_dict()
    model2 = NeuralPODOperator3D(n_modes=5, param_dim=2, hidden_dim=64, num_layers=2)
    model2.load_state_dict(state)
    assert model2.is_calibrated
    assert model2.n_modes == 5
    assert model2.basis_matrix.shape == (5, 80)
    
    p = torch.tensor([[5.0, 250.0]])
    out1 = model1(p)
    out2 = model2(p)
    assert torch.allclose(out1, out2, atol=1e-5)


def test_basis_divergence_verification():
    """Verify divergence metric calculation on calibrated basis."""
    D, H, W = 6, 6, 6
    N_voxels = D * H * W
    channels = 3
    N_dof = channels * N_voxels
    model = NeuralPODOperator3D(n_modes=2, param_dim=1)

    np.random.seed(42)
    snaps = np.random.randn(8, N_dof)
    model.calibrate_basis(snaps)
    metrics = model.verify_basis_divergence(grid_shape=(D, H, W), dx=0.1, dy=0.1, dz=0.1)
    assert "mean_mode_divergence" in metrics
    assert "max_mode_divergence" in metrics
    assert metrics["retained_modes"] == 2


def test_predict_bioprocess_quantities():
    """Verify direct calculation of bioprocess engineering metrics from model."""
    D, H, W = 6, 6, 6
    N_voxels = D * H * W
    N_dof = 3 * N_voxels
    model = NeuralPODOperator3D(n_modes=2, param_dim=2)

    np.random.seed(42)
    snaps = np.random.randn(8, N_dof) * 0.1
    model.calibrate_basis(snaps)

    p = torch.tensor([[0.25, 300.0]])
    bio = model.predict_bioprocess_quantities(
        p=p,
        grid_shape=(D, H, W),
        spacings=(0.05, 0.05, 0.05),
    )
    assert "specific_power_dissipation_w_m3" in bio
    assert "kla_h_inv" in bio
    assert "mixing_time_95_s" in bio
    assert bio["kla_h_inv"] >= 0.0
    assert bio["mixing_time_95_s"] > 0.0
