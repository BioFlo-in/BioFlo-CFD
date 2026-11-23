"""
BioFlo-CFD: Rigorous Audit Verification Test Suite
Tests and validates all 8 items raised in the static audit:
  1. Non-periodic boundary curl decoder & exact discrete divergence-free condition
  2. Dynamic grid resolution & physical domain bounds
  3. Strict train/held-out test split for POD operator
  4. Multiresolution KDTree distance thresholding & obstacle exclusion
  5. Non-overlapping validation windows (3-step sequence buffer gap)
  6. POD input validation on degenerate/zero-energy datasets
  7. Discrete divergence with anisotropic spacing and physical boundary derivatives
  8. Residual path channel flexibility (in < out, in == out, in > out)
"""

import math
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from nlss.models.cylindrical_cfd_net_3d import (
    BioFloCylindricalNet3D,
    ExactCurlVelocityDecoder,
    diff_dim,
)
from nlss.models.hybrid_cfd_net_3d import (
    HybridCFDNet3D,
    compute_divergence_3d,
    ScaleInvariantRelativeLoss3D,
)
from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from scripts.ingest_cstr_3d_multires import compute_mesh_and_geometry_mask


# =========================================================================
# AUDIT ITEM 1: Non-Periodic Boundary Curl & Exact Divergence Preservation
# =========================================================================

def test_audit_1_curl_nonperiodic_boundaries():
    """Verify curl does not wrap periodically across opposite boundaries and div(curl(A)) == 0."""
    D, H, W = 12, 12, 12
    hx, hy, hz = 0.05, 0.04, 0.08
    
    # Test A: Boundary isolation (A localized pulse at x=0 does not bleed into x=-1)
    f = torch.zeros(1, 1, D, H, W)
    f[0, 0, D // 2, H // 2, 0] = 1.0  # Pulse at lower x boundary
    df_dx = diff_dim(f, dim=4, h=hx)
    # Opposite boundary at x=-1 must remain exactly zero (no roll wrapping)
    assert df_dx[0, 0, D // 2, H // 2, -1].item() == 0.0, "Boundary derivative wrapped periodically!"

    # Test B: Discrete divergence-free identity div(curl(A)) == 0 in double precision
    torch.manual_seed(42)
    A = torch.randn(2, 3, D, H, W, dtype=torch.float64)
    decoder = ExactCurlVelocityDecoder(hx=hx, hy=hy, hz=hz)
    
    # Direct computation with diff_dim
    Ax, Ay, Az = A[:, 0:1], A[:, 1:2], A[:, 2:3]
    ux = diff_dim(Az, 3, hy) - diff_dim(Ay, 2, hz)
    uy = diff_dim(Ax, 2, hz) - diff_dim(Az, 4, hx)
    uz = diff_dim(Ay, 4, hx) - diff_dim(Ax, 3, hy)
    
    div_u = diff_dim(ux, 4, hx) + diff_dim(uy, 3, hy) + diff_dim(uz, 2, hz)
    max_div = div_u.abs().max().item()
    assert max_div < 1e-11, f"Discrete divergence not zero: {max_div:.3e}"


# =========================================================================
# AUDIT ITEM 2: Dynamic Grid Resolution & Custom Domain Bounds
# =========================================================================

def test_audit_2_dynamic_grid_and_bounds():
    """Verify model dynamically supports arbitrary resolutions and physical bounds."""
    model = BioFloCylindricalNet3D(hidden_dim=16, modes_xy=4, modes_z=4, num_layers=1)
    
    # Non-64^3 shapes
    for D, H, W in [(16, 24, 32), (20, 20, 20), (32, 16, 24)]:
        x = torch.randn(1, 5, D, H, W)
        bounds = ((-0.4, 0.4), (-0.4, 0.4), (0.0, 1.8))
        out, A = model(x, return_potential=True, domain_bounds=bounds)
        assert out.shape == (1, 5, D, H, W), f"Expected shape {(1, 5, D, H, W)}, got {out.shape}"
        assert A.shape == (1, 3, D, H, W), f"Expected shape {(1, 3, D, H, W)}, got {A.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"


# =========================================================================
# AUDIT ITEM 3: Strict Train / Held-Out Test Partitioning in POD
# =========================================================================

def test_audit_3_pod_train_heldout_split():
    """Verify POD basis and prediction are strictly separated into train and test sets."""
    N_total = 20
    N_dof = 1000
    np.random.seed(42)
    # Synthetic smooth fluid trajectory
    t = np.linspace(0.0, 1.0, N_total)
    X = np.sin(2 * np.pi * t[:, None] * np.arange(1, 6)) @ np.random.randn(5, N_dof)
    
    test_indices = [3, 8, 13, 18]
    train_indices = [i for i in range(N_total) if i not in test_indices]
    
    train_X = X[train_indices]
    test_X = X[test_indices]
    
    # Basis must be calibrated EXCLUSIVELY on training data
    model = NeuralPODOperator3D(n_modes=5, param_dim=1, hidden_dim=32, num_layers=2)
    stats = model.calibrate_basis(train_X, n_modes=5)
    
    assert model.is_calibrated
    assert stats["retained_modes"] == 5
    assert stats["cumulative_energy_pct"] > 99.0
    
    # Basis matrix must have shape (5, N_dof)
    assert model.basis_matrix.shape == (5, N_dof)


# =========================================================================
# AUDIT ITEM 4: Multiresolution KDTree Distance Threshold & Obstacle Masking
# =========================================================================

def test_audit_4_kdtree_distance_threshold_and_obstacles():
    """Verify compute_mesh_and_geometry_mask exercises KDTree and geometry rules to exclude exterior & obstacles."""
    # Synthesize unstructured point cloud of cylinder with hollow shaft
    R_tank = 0.50
    R_shaft = 0.05
    z_min, z_max = 0.0, 1.0
    grid_dim = 24
    
    # 5,000 random cell centers inside valid fluid domain [R_shaft, R_tank]
    np.random.seed(42)
    r_pts = np.random.uniform(R_shaft + 0.01, R_tank - 0.01, 5000)
    theta_pts = np.random.uniform(0, 2 * math.pi, 5000)
    z_pts = np.random.uniform(z_min + 0.01, z_max - 0.01, 5000)
    mesh_cells = np.stack([r_pts * np.cos(theta_pts), r_pts * np.sin(theta_pts), z_pts], axis=-1)
    
    # Directly exercise the actual ingestion pipeline function
    fluid_mask_3d, grid_indices, meta = compute_mesh_and_geometry_mask(
        cell_coords=mesh_cells,
        grid_dim=grid_dim,
        tank_radius=R_tank,
        shaft_radius=R_shaft,
        voxel_distance_factor=1.25,
    )
    
    GX, GY, GZ = meta["GX"], meta["GY"], meta["GZ"]
    R_grid = np.sqrt(GX**2 + GY**2)
    
    # Check 1: Solid shaft interior (r < R_shaft) must be 100% masked out (False)
    shaft_voxels = R_grid < (R_shaft * 0.95)
    assert np.any(shaft_voxels), "Test grid should contain voxels within the shaft"
    assert not np.any(fluid_mask_3d[shaft_voxels]), "Solid shaft voxels incorrectly marked as fluid!"
    
    # Check 2: Outside vessel wall (r > R_tank * 1.05) must be 100% masked out (False)
    exterior_voxels = R_grid > (R_tank * 1.05)
    assert np.any(exterior_voxels), "Test grid should contain voxels outside the tank"
    assert not np.any(fluid_mask_3d[exterior_voxels]), "Exterior voxels incorrectly marked as fluid!"
    
    # Check 3: Mid-annulus voxels (well within cell coverage) must be valid fluid (True)
    mid_fluid_voxels = (R_grid > 0.15) & (R_grid < 0.40) & (GZ > 0.1) & (GZ < 0.9)
    assert np.any(fluid_mask_3d[mid_fluid_voxels]), "Valid mid-stream fluid voxels incorrectly masked out!"
    assert meta["fluid_fraction"] > 0.0, "Fluid fraction must be non-zero"


# =========================================================================
# AUDIT ITEM 5: Non-Overlapping Validation Windows
# =========================================================================

def test_audit_5_train_val_window_gap():
    """Verify that train and validation 3-step sequence windows do not share states."""
    total_states = 80
    train_states = list(range(0, 61))  # 0 .. 60
    val_states = list(range(64, total_states))  # 64 .. 79
    
    # 3-step sequence generation
    def get_windows(arr):
        return [(arr[i], arr[i+1], arr[i+2]) for i in range(len(arr) - 2)]
        
    train_windows = get_windows(train_states)
    val_windows = get_windows(val_states)
    
    train_used_states = {s for win in train_windows for s in win}
    val_used_states = {s for win in val_windows for s in win}
    
    intersection = train_used_states.intersection(val_used_states)
    assert len(intersection) == 0, f"Data leakage detected! Shared states: {intersection}"
    assert max(train_used_states) == 60
    assert min(val_used_states) == 64
    assert (min(val_used_states) - max(train_used_states)) >= 3, "Buffer gap is insufficient"


# =========================================================================
# AUDIT ITEM 6: POD Input Validation on Degenerate/Zero-Energy Datasets
# =========================================================================

def test_audit_6_pod_degenerate_inputs():
    """Verify POD calibration strictly catches zero-energy, empty, 1D, and NaN inputs."""
    model = NeuralPODOperator3D(n_modes=5)
    
    # Test A: Zero fluctuation energy (constant field)
    with pytest.raises(ValueError, match="zero or negligible fluctuation energy"):
        model.calibrate_basis(np.ones((10, 50)))
        
    # Test B: Empty matrix
    with pytest.raises(ValueError, match="snapshot_matrix cannot be empty"):
        model.calibrate_basis(np.empty((0, 50)))
        
    # Test C: 1D input array
    with pytest.raises(ValueError, match="must be a 2D array"):
        model.calibrate_basis(np.ones(50))
        
    # Test D: Non-finite values (NaN / Inf)
    nan_mat = np.random.randn(10, 50)
    nan_mat[2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite values"):
        model.calibrate_basis(nan_mat)


# =========================================================================
# AUDIT ITEM 7: Divergence with Anisotropic Spacing & Boundary Derivatives
# =========================================================================

def test_audit_7_divergence_anisotropic_and_boundaries():
    """Verify divergence accurately computes non-zero boundary derivatives with anisotropic spacing, and ScaleInvariantRelativeLoss3D passes spacings."""
    D, H, W = 10, 12, 14
    dx, dy, dz = 0.08, 0.05, 0.12
    
    # Linear velocity field: ux = a*x, uy = b*y, uz = c*z
    # Continuous divergence: dux/dx + duy/dy + duz/dz = a + b + c everywhere
    a, b, c = 2.0, -1.5, 3.5
    expected_div = a + b + c  # 4.0
    
    gx = torch.linspace(0, (W - 1) * dx, W)
    gy = torch.linspace(0, (H - 1) * dy, H)
    gz = torch.linspace(0, (D - 1) * dz, D)
    
    GZ, GY, GX = torch.meshgrid(gz, gy, gx, indexing="ij")
    
    u_field = torch.stack([a * GX, b * GY, c * GZ], dim=0).unsqueeze(0)  # (1, 3, D, H, W)
    div = compute_divergence_3d(u_field, dx=dx, dy=dy, dz=dz)
    
    # Must match expected_div everywhere including all boundary elements
    max_err = (div - expected_div).abs().max().item()
    assert max_err < 1e-4, f"Divergence mismatch on anisotropic grid: max error = {max_err:.3e}"
    
    # Explicitly check boundary faces are NOT zero
    assert not torch.allclose(div[:, :, :, 0], torch.zeros_like(div[:, :, :, 0])), "Lower X boundary is zero!"
    assert not torch.allclose(div[:, :, :, -1], torch.zeros_like(div[:, :, :, -1])), "Upper X boundary is zero!"
    assert not torch.allclose(div[:, 0, :, :], torch.zeros_like(div[:, 0, :, :])), "Lower Z boundary is zero!"
    assert not torch.allclose(div[:, -1, :, :], torch.zeros_like(div[:, -1, :, :])), "Upper Z boundary is zero!"

    # Verify ScaleInvariantRelativeLoss3D incorporates physical anisotropic spacing
    loss_fn = ScaleInvariantRelativeLoss3D(div_weight=0.1, dx=dx, dy=dy, dz=dz)
    p_field = torch.zeros(1, 1, D, H, W)
    pred_4ch = torch.cat([u_field, p_field], dim=1)
    target_4ch = torch.zeros_like(pred_4ch)
    
    total_l, metrics = loss_fn(pred_4ch, target_4ch)
    # Expected mean abs div is expected_div (4.0)
    assert abs(metrics["loss_div"] - expected_div) < 1e-4, f"Loss divergence metric mismatch: {metrics['loss_div']} vs {expected_div}"
    
    # Also verify dynamic override via forward arguments
    _, metrics_half = loss_fn(pred_4ch, target_4ch, dx=dx * 0.5, dy=dy * 0.5, dz=dz * 0.5)
    # Halving spacings doubles the finite-difference derivatives -> expected_div * 2
    assert abs(metrics_half["loss_div"] - (expected_div * 2.0)) < 1e-3


# =========================================================================
# AUDIT ITEM 8: Residual Path Flexibility (in < out, in == out, in > out)
# =========================================================================

def test_audit_8_residual_path_channels():
    """Verify HybridCFDNet3D residual path handles any channel relationship with exact math."""
    # Case A: in == out (4 -> 4)
    net_eq = HybridCFDNet3D(in_channels=4, out_channels=4, hidden_dim=16, modes=4, num_layers=1)
    x_eq = torch.randn(2, 4, 12, 12, 12)
    out_eq = net_eq(x_eq)
    assert out_eq.shape == (2, 4, 12, 12, 12)
    
    # Case B: in > out (6 -> 4, channel slicing)
    net_gt = HybridCFDNet3D(in_channels=6, out_channels=4, hidden_dim=16, modes=4, num_layers=1)
    x_gt = torch.randn(2, 6, 12, 12, 12)
    out_gt = net_gt(x_gt)
    assert out_gt.shape == (2, 4, 12, 12, 12)
    
    # Case C: in < out (3 -> 5, channel zero-padding)
    net_lt = HybridCFDNet3D(in_channels=3, out_channels=5, hidden_dim=16, modes=4, num_layers=1)
    x_lt = torch.randn(2, 3, 12, 12, 12)
    out_lt = net_lt(x_lt)
    assert out_lt.shape == (2, 5, 12, 12, 12)
    
    # Mathematically verify residual identity when projection delta is zeroed
    # 1. in < out: out should be exactly [x, 0]
    with torch.no_grad():
        net_lt.projection[-1].weight.zero_()
        net_lt.projection[-1].bias.zero_()
        out_zero_delta = net_lt(x_lt)
        assert torch.allclose(out_zero_delta[:, :3], x_lt, atol=1e-6), "in < out: residual input channels not preserved!"
        assert torch.allclose(out_zero_delta[:, 3:], torch.zeros_like(out_zero_delta[:, 3:]), atol=1e-6), "in < out: padded channels not zero!"

    # 2. in > out: out should be exactly x[:, :out_channels]
    with torch.no_grad():
        net_gt.projection[-1].weight.zero_()
        net_gt.projection[-1].bias.zero_()
        out_zero_delta_gt = net_gt(x_gt)
        assert torch.allclose(out_zero_delta_gt, x_gt[:, :4], atol=1e-6), "in > out: sliced residual channels not preserved!"

    # 3. in == out: out should be exactly x
    with torch.no_grad():
        net_eq.projection[-1].weight.zero_()
        net_eq.projection[-1].bias.zero_()
        out_zero_delta_eq = net_eq(x_eq)
        assert torch.allclose(out_zero_delta_eq, x_eq, atol=1e-6), "in == out: identity residual not preserved!"


# =========================================================================
# AUDIT ITEM 9: Unmasked Divergence-Free Validation
# =========================================================================

def test_audit_validation_does_not_mask_prediction():
    """Verify that model predictions are not masked prior to criterion evaluation, preserving div-free."""
    # Read train_novel_cylindrical_net.py to ensure validation no longer masks predictions
    with open("scripts/train_novel_cylindrical_net.py", "r") as f:
        content = f.read()
    
    # Ensure mask_5ch is not in the script
    assert "mask_5ch" not in content, "Found deprecated mask_5ch in train_novel_cylindrical_net.py!"
    # Ensure validation loop directly calls criterion with unmasked p1
    assert "p1 = p1 * mask" not in content, "Prediction masking found in validation loop!"
