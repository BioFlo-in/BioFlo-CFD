"""
Writer for OpenFOAM native format field files.
Writes AI predictions (velocity U and pressure p) back into valid OpenFOAM
time directories so OpenFOAM solvers and post-processing tools can directly use them.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np


def _format_foam_header(field_name: str, field_class: str, location: str) -> str:
    """Generates standard OpenFOAM header."""
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
  =========                 |
  \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\\\    /   O peration     | Website:  https://openfoam.org
    \\\\  /    A nd           | Model:    NLSS AI Accelerated Surrogate
     \\\\/     M anipulation  |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {field_class};
    location    "{location}";
    object      {field_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def write_foam_scalar_field(
    filepath: Union[str, Path],
    values: np.ndarray,
    field_name: str = "p",
    dimensions: str = "[0 2 -2 0 0 0 0]",
    boundary_specs: Optional[Dict[str, str]] = None,
) -> None:
    """
    Writes a 1D scalar field (such as pressure p) to OpenFOAM format.
    values: 1D array of shape (N,) or (N, 1)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    location = filepath.parent.name
    flat_vals = values.flatten()
    n_cells = len(flat_vals)

    lines = [_format_foam_header(field_name, "volScalarField", location)]
    lines.append(f"dimensions      {dimensions};\n")
    lines.append(f"internalField   nonuniform List<scalar>\n{n_cells}\n(\n")

    # Format values in blocks for speed
    chunk_size = 1000
    for i in range(0, n_cells, chunk_size):
        chunk = flat_vals[i : i + chunk_size]
        lines.append("\n".join(f"{v:.6e}" for v in chunk) + "\n")

    lines.append(");\n\n")

    # Boundary field
    lines.append("boundaryField\n{\n")
    if boundary_specs:
        for patch_name, patch_body in boundary_specs.items():
            lines.append(f"    {patch_name}\n    {{\n{patch_body}\n    }}\n")
    else:
        # Default fallback boundaries
        lines.append("""    ".*"
    {
        type            zeroGradient;
    }
""")
    lines.append("}\n\n")
    lines.append("// ************************************************************************* //\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


def write_foam_vector_field(
    filepath: Union[str, Path],
    values: np.ndarray,
    field_name: str = "U",
    dimensions: str = "[0 1 -1 0 0 0 0]",
    boundary_specs: Optional[Dict[str, str]] = None,
) -> None:
    """
    Writes a 2D/3D vector field (such as velocity U) to OpenFOAM format.
    values: array of shape (N, 2) or (N, 3)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    location = filepath.parent.name

    if values.ndim == 1:
        raise ValueError("Vector field must have at least 2 dimensions")
    if values.ndim == 3:
        values = values.reshape(-1, values.shape[-1])
    n_cells = values.shape[0]

    # If 2D (u_x, u_y), pad with 0 for u_z
    if values.shape[1] == 2:
        z_col = np.zeros((n_cells, 1), dtype=values.dtype)
        vecs_3d = np.hstack([values, z_col])
    else:
        vecs_3d = values

    lines = [_format_foam_header(field_name, "volVectorField", location)]
    lines.append(f"dimensions      {dimensions};\n")
    lines.append(f"internalField   nonuniform List<vector>\n{n_cells}\n(\n")

    chunk_size = 1000
    for i in range(0, n_cells, chunk_size):
        chunk = vecs_3d[i : i + chunk_size]
        formatted = ["({:.6e} {:.6e} {:.6e})".format(v[0], v[1], v[2]) for v in chunk]
        lines.append("\n".join(formatted) + "\n")

    lines.append(");\n\n")

    # Boundary field
    lines.append("boundaryField\n{\n")
    if boundary_specs:
        for patch_name, patch_body in boundary_specs.items():
            lines.append(f"    {patch_name}\n    {{\n{patch_body}\n    }}\n")
    else:
        lines.append("""    ".*"
    {
        type            noSlip;
    }
""")
    lines.append("}\n\n")
    lines.append("// ************************************************************************* //\n")

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)


def export_simulation_state_to_openfoam(
    case_dir: Union[str, Path],
    time_val: float,
    u_pred: np.ndarray,
    p_pred: Optional[np.ndarray] = None,
    boundary_templates: Optional[Dict[str, Dict[str, str]]] = None,
) -> Path:
    """
    Exports a predicted simulation time step state directly into an OpenFOAM case directory.
    Creates folder <case_dir>/<time_val>/ containing 'U' and 'p'.
    """
    case_dir = Path(case_dir)
    # Format time directory string (e.g. 0.05 or 0.1)
    if time_val.is_integer():
        time_str = f"{int(time_val)}"
    else:
        time_str = f"{time_val:.4f}".rstrip("0").rstrip(".")

    target_time_dir = case_dir / time_str
    target_time_dir.mkdir(parents=True, exist_ok=True)

    u_bounds = boundary_templates.get("U") if boundary_templates else None
    p_bounds = boundary_templates.get("p") if boundary_templates else None

    write_foam_vector_field(
        filepath=target_time_dir / "U",
        values=u_pred,
        field_name="U",
        boundary_specs=u_bounds,
    )

    if p_pred is not None:
        write_foam_scalar_field(
            filepath=target_time_dir / "p",
            values=p_pred,
            field_name="p",
            boundary_specs=p_bounds,
        )

    return target_time_dir
