import tempfile
from pathlib import Path
import numpy as np
import pytest

from nlss.io.foam_writer import (
    write_foam_scalar_field,
    write_foam_vector_field,
    export_simulation_state_to_openfoam,
    export_multiphase_simulation_state_to_openfoam,
)
from nlss.io.foam_reader import read_foam_field_file, load_openfoam_simulation_series


def test_foam_writer_and_reader():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        case_dir = tmp_path / "test_case"

        # Generate sample velocity and pressure fields
        n_cells = 100
        u_orig = np.random.randn(n_cells, 2).astype(np.float32)
        p_orig = np.random.randn(n_cells, 1).astype(np.float32)

        # Export to OpenFOAM format at time 0.05
        time_dir = export_simulation_state_to_openfoam(
            case_dir=case_dir,
            time_val=0.05,
            u_pred=u_orig,
            p_pred=p_orig,
        )

        assert time_dir.exists()
        assert (time_dir / "U").exists()
        assert (time_dir / "p").exists()

        # Read back with foam_reader
        u_field = read_foam_field_file(time_dir / "U", time_val=0.05)
        p_field = read_foam_field_file(time_dir / "p", time_val=0.05)

        assert u_field.name == "U"
        assert u_field.field_type == "volVectorField"
        assert u_field.internal_field.shape == (n_cells, 3)  # padded with z=0
        np.testing.assert_allclose(u_field.internal_field[:, :2], u_orig, atol=1e-5)

        assert p_field.name == "p"
        assert p_field.field_type == "volScalarField"
        assert p_field.internal_field.shape == (n_cells,)
        np.testing.assert_allclose(p_field.internal_field, p_orig.flatten(), atol=1e-5)


def test_multiphase_foam_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        case_dir = tmp_path / "bird_case"

        n_cells = 50
        u_liq = np.random.randn(n_cells, 3).astype(np.float32)
        alpha_g = np.random.uniform(0.01, 0.20, n_cells).astype(np.float32)

        time_dir = export_multiphase_simulation_state_to_openfoam(
            case_dir=case_dir,
            time_val=0.0,
            u_liquid_pred=u_liq,
            alpha_gas_pred=alpha_g,
        )

        assert time_dir.exists()
        assert (time_dir / "U.liquid").exists()
        assert (time_dir / "U.gas").exists()
        assert (time_dir / "alpha.gas").exists()
        assert (time_dir / "alpha.liquid").exists()

        alpha_g_read = read_foam_field_file(time_dir / "alpha.gas", time_val=0.0)
        alpha_l_read = read_foam_field_file(time_dir / "alpha.liquid", time_val=0.0)

        assert alpha_g_read.name == "alpha.gas"
        assert alpha_l_read.name == "alpha.liquid"
        # Phase sum: alpha_gas + alpha_liquid = 1.0 everywhere
        np.testing.assert_allclose(
            alpha_g_read.internal_field + alpha_l_read.internal_field,
            np.ones(n_cells),
            atol=1e-5,
        )


if __name__ == "__main__":
    test_foam_writer_and_reader()
    print("test_foam_writer_and_reader passed!")
