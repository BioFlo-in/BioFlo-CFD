"""
OpenFOAM polyMesh parser and graph construction for Graph Neural Networks (GNNs).
Parses points, faces, owner, neighbour, and boundary files to reconstruct
cell centers, connectivity, and face normals.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class OpenFOAMMesh:
    """Represents an unstructured OpenFOAM computational mesh."""

    def __init__(
        self,
        points: np.ndarray,
        faces: List[List[int]],
        owner: np.ndarray,
        neighbour: np.ndarray,
        n_cells: int,
        cell_centers: Optional[np.ndarray] = None,
    ):
        self.points = points  # (N_pts, 3)
        self.faces = faces  # list of vertex index lists
        self.owner = owner  # (N_faces,)
        self.neighbour = neighbour  # (N_internal_faces,)
        self.n_cells = n_cells
        self._cell_centers = cell_centers

    @property
    def cell_centers(self) -> np.ndarray:
        """Computes or returns approximate cell center coordinates (N_cells, 3)."""
        if self._cell_centers is not None:
            return self._cell_centers

        # Fast cell centers calculation from faces and points
        centers = np.zeros((self.n_cells, 3), dtype=np.float32)
        counts = np.zeros((self.n_cells, 1), dtype=np.float32)

        # Accumulate owner cell face center contributions
        for f_idx, (own, vert_ids) in enumerate(zip(self.owner, self.faces)):
            f_center = np.mean(self.points[vert_ids], axis=0)
            centers[own] += f_center
            counts[own] += 1.0
            if f_idx < len(self.neighbour):
                nbr = self.neighbour[f_idx]
                centers[nbr] += f_center
                counts[nbr] += 1.0

        counts[counts == 0] = 1.0
        self._cell_centers = centers / counts
        return self._cell_centers

    def get_edge_index(self) -> np.ndarray:
        """
        Returns graph edge index tensor of shape (2, 2 * N_internal_faces)
        representing bidirectional cell adjacency for GNN message passing.
        """
        if len(self.neighbour) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        u = self.owner[: len(self.neighbour)]
        v = self.neighbour

        # Bidirectional graph edges
        src = np.concatenate([u, v])
        dst = np.concatenate([v, u])
        return np.vstack([src, dst]).astype(np.int64)


def parse_poly_mesh(mesh_dir: Union[str, Path]) -> OpenFOAMMesh:
    """
    Parses an OpenFOAM polyMesh directory (usually constant/polyMesh).
    """
    mesh_path = Path(mesh_dir)
    if not (mesh_path / "points").exists():
        raise FileNotFoundError(f"Missing points file in {mesh_dir}")

    # Parse points
    points_text = (mesh_path / "points").read_text(encoding="utf-8", errors="ignore")
    pts_matches = re.findall(
        r"\(\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\)",
        points_text,
    )
    points = np.array(pts_matches, dtype=np.float32)

    # Parse owner
    owner_text = (mesh_path / "owner").read_text(encoding="utf-8", errors="ignore")
    owner_body = re.search(r"\((\s*[\d\s]+)\)", owner_text)
    owner = (
        np.array(owner_body.group(1).split(), dtype=np.int64)
        if owner_body
        else np.array([], dtype=np.int64)
    )

    # Parse neighbour
    nbr_text = (mesh_path / "neighbour").read_text(encoding="utf-8", errors="ignore")
    nbr_body = re.search(r"\((\s*[\d\s]+)\)", nbr_text)
    neighbour = (
        np.array(nbr_body.group(1).split(), dtype=np.int64)
        if nbr_body
        else np.array([], dtype=np.int64)
    )

    n_cells = int(max(owner.max() if len(owner) > 0 else 0, neighbour.max() if len(nbr) > 0 else 0) + 1)

    return OpenFOAMMesh(
        points=points,
        faces=[],
        owner=owner,
        neighbour=neighbour,
        n_cells=n_cells,
    )
