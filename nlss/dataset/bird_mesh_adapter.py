"""
BioFlo-CFD: BiRD Meshing & Multiphase Case Adapter
Bridges the upstream NatLabRockies/BioReactorDesign (BiRD) toolbox with NLSS.

Key Capabilities:
  1. Parametric Stirred-Tank & Bubble-Column Mesh Generation:
     Transforms YAML / dataclass geometry specifications into OpenFOAM blockMeshDict.
  2. Dimensionless Coordinate Normalization (Resolving LIM-02):
     Maps dimensional coordinates (x, y, z) into scale-invariant domain (xi, eta, zeta) in [0, 1]^3
     allowing the Neural POD basis to generalize across vessel sizes (2L to 20,000L).
  3. BiRD Multiphase Case Ingestion:
     Parses Eulerian two-phase flow fields (alpha.air, U.air, U.water, k, epsilon, kLa).
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import os
import sys
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

# Add BiRD to python path if available in external_openfoam_sources
BIRD_DIR = Path(__file__).resolve().parent.parent.parent / "external_openfoam_sources" / "BioReactorDesign"
if BIRD_DIR.exists() and str(BIRD_DIR) not in sys.path:
    sys.path.insert(0, str(BIRD_DIR))


@dataclass
class StirredTankSpec:
    """Parametric specification of a stirred tank reactor."""
    Dt: float = 0.30            # Tank diameter [m]
    Da: float = 0.10            # Impeller diameter [m]
    H: float = 0.45             # Liquid height [m]
    nimpellers: int = 1         # Number of impellers
    C: List[float] = field(default_factory=lambda: [0.10]) # Clearance height of impeller center [m]
    W: float = 0.025            # Impeller blade width [m]
    L: float = 0.020            # Impeller blade length beyond hub [m]
    Lin: float = 0.015          # Impeller blade length inside hub [m]
    J: float = 0.025            # Baffle width [m]
    Wh: float = 0.005           # Hub thickness [m]
    polyrad: float = 0.012      # Shaft radius [m]
    Z0: float = 0.0             # Tank bottom level [m]
    nbaffles: int = 4           # Number of baffles
    # Mesh discretization density
    nr: int = 80                # Radial mesh cells per unit length
    nz: int = 160               # Axial mesh cells per unit length
    Npoly: int = 4              # Polygon cells at axis
    Na: int = 6                 # Azimuthal cells per split

    def to_bird_dict(self) -> Dict:
        """Converts to BiRD-compatible YAML dictionary layout."""
        return {
            "geometry": {
                "Dt": float(self.Dt),
                "Da": float(self.Da),
                "H": float(self.H),
                "nimpellers": int(self.nimpellers),
                "C": [float(c) for c in self.C],
                "W": float(self.W),
                "L": float(self.L),
                "Lin": float(self.Lin),
                "J": float(self.J),
                "Wh": float(self.Wh),
                "polyrad": float(self.polyrad),
                "Z0": float(self.Z0),
                "nbaffles": int(self.nbaffles),
            },
            "mesh": {
                "nr": int(self.nr),
                "nz": int(self.nz),
                "Npoly": int(self.Npoly),
                "Na": int(self.Na),
            },
        }


class BiRDMeshAdapter:
    """
    Adapter interfacing BiRD meshing generators with NLSS neural operator models.
    """

    def __init__(self, bird_root: Optional[Union[str, Path]] = None):
        if bird_root is None:
            self.bird_root = BIRD_DIR
        else:
            self.bird_root = Path(bird_root)

    def write_tank_yaml(self, spec: StirredTankSpec, output_yaml_path: Union[str, Path]) -> Path:
        """Writes a StirredTankSpec to BiRD-compliant YAML format."""
        out_p = Path(output_yaml_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        data = spec.to_bird_dict()
        with open(out_p, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return out_p

    def generate_blockmesh_dict(
        self,
        spec: StirredTankSpec,
        output_dict_path: Union[str, Path],
    ) -> Path:
        """
        Generates an OpenFOAM blockMeshDict using BiRD's stirred_tank_mesh engine.
        """
        out_dict = Path(output_dict_path)
        out_dict.parent.mkdir(parents=True, exist_ok=True)

        temp_yaml = out_dict.parent / f"_temp_bird_{out_dict.stem}.yaml"
        self.write_tank_yaml(spec, temp_yaml)

        try:
            # Try importing BiRD module directly
            from bird.meshing.stirred_tank_mesh import (
                get_reactor_geom,
                write_ofoam_preamble,
                write_vertices,
                write_edges,
                write_blocks,
                write_patches,
            )

            react = get_reactor_geom(str(temp_yaml))
            with open(out_dict, "w", encoding="utf-8") as f:
                write_ofoam_preamble(f, react)
                write_vertices(f, react)
                write_edges(f, react)
                write_blocks(f, react)
                write_patches(f, react)

        except Exception as e:
            # Fallback to invoking write_stirred_tank_mesh.py via shell if direct import fails
            script_path = self.bird_root / "applications" / "write_stirred_tank_mesh.py"
            if not script_path.exists():
                raise RuntimeError(
                    f"BiRD meshing engine not available at {self.bird_root}. Import error: {e}"
                )
            import subprocess
            cmd = [
                sys.executable,
                str(script_path),
                "-i", str(temp_yaml),
                "-o", str(out_dict),
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = str(self.bird_root)
            subprocess.run(cmd, env=env, check=True)

        finally:
            if temp_yaml.exists():
                temp_yaml.unlink()

        return out_dict

    def normalize_coordinates(
        self,
        coords: np.ndarray,
        spec: StirredTankSpec,
    ) -> np.ndarray:
        """
        Maps physical (x, y, z) coordinates into scale-invariant dimensionless space:
            r = sqrt(x^2 + y^2)
            xi = r / (Dt / 2)       in [0, 1]
            theta = atan2(y, x)      in [0, 2*pi]
            zeta = (z - Z0) / H      in [0, 1]

        Args:
            coords: Array of shape (N, 3) with columns [x, y, z].
            spec: StirredTankSpec containing Dt, H, Z0.
        Returns:
            Normalized coordinates array of shape (N, 3) with [xi_x, xi_y, zeta] in [-1, 1]x[-1, 1]x[0, 1]
        """
        R_tank = spec.Dt / 2.0
        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        xi_x = x / R_tank
        xi_y = y / R_tank
        zeta = (z - spec.Z0) / spec.H

        return np.column_stack([xi_x, xi_y, zeta])

    def denormalize_coordinates(
        self,
        norm_coords: np.ndarray,
        spec: StirredTankSpec,
    ) -> np.ndarray:
        """Inverts dimensionless coordinates back to physical (x, y, z) in meters."""
        R_tank = spec.Dt / 2.0
        x = norm_coords[:, 0] * R_tank
        y = norm_coords[:, 1] * R_tank
        z = norm_coords[:, 2] * spec.H + spec.Z0
        return np.column_stack([x, y, z])

    def scan_multiphase_case(self, case_dir: Union[str, Path]) -> Dict[str, Union[List[float], Dict]]:
        """
        Inspects an OpenFOAM BiRD multiphase case directory for available time directories
        and multiphase fields (alpha.air, U.air, U.water, etc.).
        """
        case_p = Path(case_dir)
        if not case_p.is_dir():
            raise FileNotFoundError(f"Case directory {case_dir} does not exist")

        time_dirs = []
        for d in case_p.iterdir():
            if d.is_dir():
                try:
                    val = float(d.name)
                    time_dirs.append((val, d))
                except ValueError:
                    continue

        time_dirs.sort(key=lambda x: x[0])
        times = [t[0] for t in time_dirs]

        fields_by_time = {}
        for t_val, t_dir in time_dirs:
            present_fields = [f.name for f in t_dir.iterdir() if f.is_file()]
            fields_by_time[t_val] = present_fields

        has_pbm = any("d32" in fld or "alpha.air" in fld for fld in fields_by_time.get(times[-1], [])) if times else False

        return {
            "case_dir": str(case_p),
            "times": times,
            "num_timesteps": len(times),
            "fields_by_time": fields_by_time,
            "has_multiphase_euler": any("alpha.air" in fld for fld in fields_by_time.get(times[-1], [])) if times else False,
            "has_pbm": has_pbm,
        }
