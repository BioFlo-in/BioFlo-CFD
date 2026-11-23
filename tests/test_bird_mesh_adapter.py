"""
Unit and Integration Tests for BiRD Meshing & Case Adapter.
Verifies:
  1. StirredTankSpec geometry configuration and BiRD YAML layout.
  2. Automated blockMeshDict generation via BiRD meshing engine.
  3. Dimensionless coordinate normalization (LIM-02 resolution) and exact invertibility.
  4. Scanning and inspection of imported BiRD multiphase OpenFOAM cases.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from nlss.dataset.bird_mesh_adapter import (
    StirredTankSpec,
    BiRDMeshAdapter,
)


def test_stirred_tank_spec_defaults():
    spec = StirredTankSpec()
    assert spec.Dt == 0.30
    assert spec.Da == 0.10
    assert spec.H == 0.45
    assert spec.nbaffles == 4

    bird_dict = spec.to_bird_dict()
    assert "geometry" in bird_dict
    assert "mesh" in bird_dict
    assert bird_dict["geometry"]["Dt"] == 0.30
    assert bird_dict["geometry"]["C"] == [0.10]
    assert bird_dict["mesh"]["nr"] == 80


def test_bird_mesh_adapter_generate_blockmesh():
    adapter = BiRDMeshAdapter()
    spec = StirredTankSpec(
        Dt=0.20,
        Da=0.07,
        H=0.30,
        C=[0.08],
        nbaffles=4,
        nr=40,
        nz=80,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dict = Path(tmp_dir) / "system" / "blockMeshDict"
        out_path = adapter.generate_blockmesh_dict(spec, tmp_dict)

        assert out_path.exists()
        assert out_path.stat().st_size > 500

        content = out_path.read_text(encoding="utf-8")
        assert "FoamFile" in content
        assert "blockMeshDict" in content
        assert "vertices" in content
        assert "blocks" in content
        assert "patches" in content or "boundary" in content


def test_coordinate_normalization_roundtrip():
    adapter = BiRDMeshAdapter()
    spec = StirredTankSpec(Dt=0.50, H=1.00, Z0=0.0)

    # Generate synthetic physical coordinates inside the vessel
    np.random.seed(42)
    N = 100
    r = np.random.uniform(0.0, spec.Dt / 2.0, N)
    theta = np.random.uniform(0.0, 2 * np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(0.0, spec.H, N)

    coords = np.column_stack([x, y, z])

    # Normalize
    norm_coords = adapter.normalize_coordinates(coords, spec)
    assert norm_coords.shape == (N, 3)

    # Check bounds: xi_x^2 + xi_y^2 <= 1.0, zeta in [0, 1]
    norm_radii = np.sqrt(norm_coords[:, 0]**2 + norm_coords[:, 1]**2)
    assert np.all(norm_radii <= 1.0 + 1e-6)
    assert np.all(norm_coords[:, 2] >= 0.0 - 1e-6)
    assert np.all(norm_coords[:, 2] <= 1.0 + 1e-6)

    # Invert back to physical coordinates
    recovered_coords = adapter.denormalize_coordinates(norm_coords, spec)
    np.testing.assert_allclose(coords, recovered_coords, atol=1e-6)


def test_scan_imported_bird_cases():
    adapter = BiRDMeshAdapter()
    base_cases_dir = Path(__file__).resolve().parent.parent / "openfoam_cases"

    test_cases = [
        base_cases_dir / "bird_stirred_tank",
        base_cases_dir / "bird_bubble_column_20l",
        base_cases_dir / "bird_uloop_valadbeigy_exp1",
    ]

    for c_dir in test_cases:
        assert c_dir.exists(), f"Expected case dir {c_dir} to exist"
        info = adapter.scan_multiphase_case(c_dir)
        assert "case_dir" in info
        assert "times" in info
        assert "num_timesteps" in info
        assert isinstance(info["times"], list)
