"""
High-performance parser for native OpenFOAM field files and meshes.
Supports reading time-series directories (0/, 0.01/, 0.1/, etc.) containing
volScalarField (e.g., p, k, omega, T) and volVectorField (e.g., U).
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class OpenFOAMField:
    """Represents a parsed OpenFOAM field (scalar or vector)."""

    def __init__(
        self,
        name: str,
        field_type: str,
        dimensions: str,
        internal_field: np.ndarray,
        boundary_field: Optional[Dict[str, Dict]] = None,
        time_val: float = 0.0,
    ):
        self.name = name  # e.g., 'U' or 'p'
        self.field_type = field_type  # 'volScalarField' or 'volVectorField'
        self.dimensions = dimensions  # e.g., '[0 1 -1 0 0 0 0]'
        self.internal_field = internal_field  # shape (N,) for scalar, (N, 3) for vector
        self.boundary_field = boundary_field or {}
        self.time_val = time_val

    def __repr__(self) -> str:
        return (
            f"OpenFOAMField(name='{self.name}', type='{self.field_type}', "
            f"shape={self.internal_field.shape}, time={self.time_val})"
        )


def _strip_comments(text: str) -> str:
    """Strips C/C++ style single-line and multi-line comments."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*", "", text)
    return text


def parse_foam_header(text: str) -> Dict[str, str]:
    """Extracts FoamFile header dictionary."""
    match = re.search(r"FoamFile\s*\{(.*?)\}", text, re.DOTALL)
    if not match:
        return {}
    header_content = match.group(1)
    props = {}
    for line in header_content.split(";"):
        parts = line.strip().split()
        if len(parts) >= 2:
            props[parts[0]] = parts[1].strip('"')
    return props


def parse_internal_field(text: str, field_class: str) -> np.ndarray:
    """Parses internalField section (both uniform and nonuniform List)."""
    # Check for nonuniform List
    match_list = re.search(
        r"internalField\s+nonuniform\s+List<(\w+)>\s+(\d+)\s*\((.*?)\)\s*;",
        text,
        re.DOTALL,
    )
    if match_list:
        data_type = match_list.group(1).lower()
        n_items = int(match_list.group(2))
        raw_body = match_list.group(3).strip()

        if "vector" in data_type or field_class == "volVectorField":
            # Extract tuples of format (x y z)
            vecs = re.findall(
                r"\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
                raw_body,
            )
            if len(vecs) > 0:
                arr = np.array(vecs, dtype=np.float32)
                return arr
            else:
                # Fast whitespace splitting fallback
                tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", raw_body)
                arr = np.array(tokens, dtype=np.float32)
                if arr.size == n_items * 3:
                    return arr.reshape((n_items, 3))
                return arr
        else:
            # Scalar
            tokens = re.findall(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", raw_body)
            arr = np.array(tokens, dtype=np.float32)
            return arr

    # Check for uniform scalar or vector
    match_uniform = re.search(r"internalField\s+uniform\s+([^;]+);", text)
    if match_uniform:
        val_str = match_uniform.group(1).strip()
        vec_match = re.search(
            r"\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
            val_str,
        )
        if vec_match:
            vec = [float(vec_match.group(i)) for i in (1, 2, 3)]
            return np.array([vec], dtype=np.float32)
        else:
            try:
                scalar = float(val_str)
                return np.array([scalar], dtype=np.float32)
            except ValueError:
                pass

    return np.zeros((0,), dtype=np.float32)


def read_foam_field_file(filepath: Union[str, Path], time_val: float = 0.0) -> OpenFOAMField:
    """Reads a single OpenFOAM field file (e.g., 0.1/U or 0.1/p)."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    clean_content = _strip_comments(content)
    header = parse_foam_header(clean_content)

    field_class = header.get("class", "volScalarField")
    obj_name = header.get("object", filepath.name)

    dim_match = re.search(r"dimensions\s*\[([^\]]+)\];", clean_content)
    dimensions = f"[{dim_match.group(1)}]" if dim_match else "[0 0 0 0 0 0 0]"

    internal_field = parse_internal_field(clean_content, field_class)

    return OpenFOAMField(
        name=obj_name,
        field_type=field_class,
        dimensions=dimensions,
        internal_field=internal_field,
        time_val=time_val,
    )


def get_case_time_directories(case_dir: Union[str, Path]) -> List[Tuple[float, Path]]:
    """
    Finds all valid simulation time step directories in an OpenFOAM case,
    sorted numerically by simulation time.
    """
    case_path = Path(case_dir)
    time_dirs = []

    for entry in case_path.iterdir():
        if entry.is_dir() and entry.name not in {"constant", "system", "dynamicCode", "VTK", "postProcessing"}:
            try:
                t_val = float(entry.name)
                time_dirs.append((t_val, entry))
            except ValueError:
                continue

    time_dirs.sort(key=lambda x: x[0])
    return time_dirs


def load_openfoam_simulation_series(
    case_dir: Union[str, Path],
    fields: Optional[List[str]] = None,
) -> Dict[str, Union[np.ndarray, List[float]]]:
    """
    Loads complete simulation trajectory from an OpenFOAM case.
    Returns:
        {
            'times': [0.0, 0.01, 0.02, ...],
            'U': ndarray shape (T, N_cells, 3) or (T, N_cells, 2),
            'p': ndarray shape (T, N_cells, 1) or (T, N_cells),
            ...
        }
    """
    if fields is None:
        fields = ["U", "p"]

    time_dirs = get_case_time_directories(case_dir)
    if not time_dirs:
        raise FileNotFoundError(f"No valid time directories found in OpenFOAM case: {case_dir}")

    times = [t for t, _ in time_dirs]
    data_dict: Dict[str, List[np.ndarray]] = {f: [] for f in fields}

    for t_val, t_dir in time_dirs:
        for f_name in fields:
            field_file = t_dir / f_name
            if field_file.exists():
                field_obj = read_foam_field_file(field_file, time_val=t_val)
                data_dict[f_name].append(field_obj.internal_field)
            else:
                data_dict[f_name].append(np.array([]))

    result: Dict[str, Union[np.ndarray, List[float]]] = {"times": times}
    for f_name in fields:
        raw_items = data_dict[f_name]
        valid_items = [arr for arr in raw_items if arr.size > 0]
        if valid_items and len(valid_items) == len(times):
            # Find the max cell count across timesteps (nonuniform field shape)
            max_len = max(arr.shape[0] for arr in valid_items)
            normalized_items = []
            for arr in valid_items:
                if arr.shape[0] == 1 and max_len > 1:
                    # Broadcast uniform field across all cells
                    if arr.ndim == 2:
                        expanded = np.repeat(arr, max_len, axis=0)
                    else:
                        expanded = np.repeat(arr, max_len)
                    normalized_items.append(expanded)
                else:
                    normalized_items.append(arr)

            ref_shape = normalized_items[0].shape
            if all(item.shape == ref_shape for item in normalized_items):
                result[f_name] = np.stack(normalized_items, axis=0)
            else:
                result[f_name] = normalized_items
        else:
            result[f_name] = valid_items

    return result
