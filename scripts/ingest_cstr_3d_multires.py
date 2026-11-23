#!/usr/bin/env python3
"""
Multi-Resolution 3D CSTR Multiphase Dataset Ingestion Pipeline:
Maps 894,950 unstructured OpenFOAM polyhedral cells onto uniform 3D Cartesian grids.
Supports grid dimensions: 64^3 (262,144 voxels) and above.
Includes exact cylindrical fluid domain masking (Omega_fluid).
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch

from nlss.io.foam_reader import read_foam_field_file
from typing import Optional, Tuple, Dict, Any


def compute_mesh_and_geometry_mask(
    cell_coords: np.ndarray,
    grid_dim: int,
    tank_radius: float = 0.25,
    shaft_radius: float = 0.020,
    voxel_distance_factor: float = 1.25,
    domain_bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes regular Cartesian grid mapping and fluid domain mask.
    Combines:
      1. cKDTree spatial proximity: Checks that regular voxels are within
         (voxel_distance_factor * voxel_diag) of actual unstructured cell centroids.
      2. Analytical vessel boundary checks: r <= tank_radius and z_min <= z <= z_max.
      3. Solid obstacle rejection: r >= shaft_radius (impeller shaft).

    Geometric & Approximation Note:
      A nearest-neighbor centroid distance cutoff is a sphere-covering approximation
      to the polyhedral cell complex. For unstructured meshes with high-aspect-ratio boundary
      layers and internal obstacles, combining centroid distance cutoff with analytical
      geometric boundaries (vessel envelope, shaft, hub) prevents labelling solid or exterior
      points as fluid.

    Returns:
      fluid_mask_3d: (grid_dim, grid_dim, grid_dim) boolean array
      grid_indices: (grid_dim^3,) integer array of nearest mesh cell indices
      metadata: dictionary with grid spacings, bounds, and diagnostics
    """
    if domain_bounds is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = domain_bounds
    else:
        x_min, y_min, z_min = cell_coords.min(axis=0)
        x_max, y_max, z_max = cell_coords.max(axis=0)

    gx = np.linspace(x_min, x_max, grid_dim, dtype=np.float32)
    gy = np.linspace(y_min, y_max, grid_dim, dtype=np.float32)
    gz = np.linspace(z_min, z_max, grid_dim, dtype=np.float32)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1)

    dx = float((x_max - x_min) / (grid_dim - 1)) if grid_dim > 1 else 1.0
    dy = float((y_max - y_min) / (grid_dim - 1)) if grid_dim > 1 else 1.0
    dz = float((z_max - z_min) / (grid_dim - 1)) if grid_dim > 1 else 1.0
    voxel_diag = float(np.sqrt(dx**2 + dy**2 + dz**2))
    max_mesh_dist = voxel_distance_factor * voxel_diag

    tree = cKDTree(cell_coords)
    dists, grid_indices = tree.query(grid_pts, k=1)

    dists_3d = dists.reshape(grid_dim, grid_dim, grid_dim)
    r_sq = GX**2 + GY**2
    is_within_mesh = (dists_3d <= max_mesh_dist)
    is_outside_shaft = (r_sq >= (shaft_radius**2))
    is_within_tank = (r_sq <= (tank_radius * 1.005)**2) & (GZ >= (z_min - 1e-4)) & (GZ <= (z_max + 1e-4))

    fluid_mask_3d = is_within_mesh & is_outside_shaft & is_within_tank
    metadata = {
        "x_min": float(x_min), "x_max": float(x_max),
        "y_min": float(y_min), "y_max": float(y_max),
        "z_min": float(z_min), "z_max": float(z_max),
        "dx": dx, "dy": dy, "dz": dz,
        "voxel_diag": voxel_diag,
        "max_mesh_dist": max_mesh_dist,
        "fluid_fraction": float(np.mean(fluid_mask_3d) * 100.0),
        "GX": GX, "GY": GY, "GZ": GZ,
    }
    return fluid_mask_3d, grid_indices, metadata


def ingest_cstr_3d_multires(
    case_dir: str = "openfoam_cases/professional_cstr_3d_gpu_mrf",
    output_path: str = "data/cstr_3d_dataset_64.pt",
    grid_dim: int = 64,
    tank_radius: float = 0.25,
):
    print("=" * 80)
    print(f"🚀 BioFlo-CFD: Generating {grid_dim}³ High-Resolution 3D CSTR Dataset")
    print(f"   Target Voxels: {grid_dim**3:,} | Tank Radius: {tank_radius:.3f} m")
    print("=" * 80)
    case_path = Path(case_dir)
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # 1. Discover all time directories
    time_dirs = [d for d in os.listdir(case_path) if re.match(r"^[0-9]+(\.[0-9]+)?$", d)]
    time_dirs = sorted(time_dirs, key=lambda x: float(x))
    times = [float(d) for d in time_dirs]
    print(f"Found {len(times)} time directories spanning t = {times[0]:.2f}s to {times[-1]:.2f}s")

    # 2. Load cell centers
    c_file = case_path / "1.95" / "C"
    if not c_file.exists():
        raise FileNotFoundError(f"Cell coordinates file not found at {c_file}!")

    print("Loading 3D mesh cell centers (894,950 cells)...")
    t0 = time.time()
    c_field = read_foam_field_file(c_file)
    cell_coords = c_field.internal_field
    print(f"✓ Loaded {cell_coords.shape[0]:,} coordinates in {time.time() - t0:.2f}s")

    # 3. Compute regular Cartesian grid mapping and verified fluid mask
    fluid_mask_3d, grid_indices, meta = compute_mesh_and_geometry_mask(
        cell_coords=cell_coords,
        grid_dim=grid_dim,
        tank_radius=tank_radius,
        shaft_radius=0.020,
        voxel_distance_factor=1.25,
    )
    dx, dy, dz = meta["dx"], meta["dy"], meta["dz"]
    GX, GY, GZ = meta["GX"], meta["GY"], meta["GZ"]
    print(f"Domain Bounds: X=[{meta['x_min']:.3f}, {meta['x_max']:.3f}], Y=[{meta['y_min']:.3f}, {meta['y_max']:.3f}], Z=[{meta['z_min']:.3f}, {meta['z_max']:.3f}]")
    print(f"✓ Verified True Fluid Mask: {meta['fluid_fraction']:.1f}% fluid volume, {100-meta['fluid_fraction']:.1f}% exterior/solid obstacles masked")

    # 6. Ingest each time snapshot
    all_states = []
    print(f"\nProcessing {len(time_dirs)} snapshots onto ({grid_dim} x {grid_dim} x {grid_dim}) grid...")

    t_ingest_start = time.time()
    for idx, (t_str, t_val) in enumerate(zip(time_dirs, times)):
        t_dir = case_path / t_str
        t_snap_start = time.time()

        u_obj = read_foam_field_file(t_dir / "U", time_val=t_val)
        p_obj = read_foam_field_file(t_dir / "p", time_val=t_val)
        alpha_obj = read_foam_field_file(t_dir / "alpha.water", time_val=t_val)

        u_raw = u_obj.internal_field  # (N_cells, 3)
        p_raw = p_obj.internal_field  # (N_cells,)
        alpha_raw = alpha_obj.internal_field  # (N_cells,)

        # Map to 3D grid via precomputed indices
        u_grid = u_raw[grid_indices].reshape(grid_dim, grid_dim, grid_dim, 3)
        p_grid = p_raw[grid_indices].reshape(grid_dim, grid_dim, grid_dim, 1)
        alpha_grid = alpha_raw[grid_indices].reshape(grid_dim, grid_dim, grid_dim, 1)

        # Apply geometric fluid mask: zero-out exterior velocity and clamp
        u_grid[~fluid_mask_3d] = 0.0
        p_grid[~fluid_mask_3d] = 0.0
        alpha_grid[~fluid_mask_3d] = 0.0
        alpha_grid = np.clip(alpha_grid, 0.0, 1.0)

        # Combine into (5, D, H, W) -> [ux, uy, uz, p, alpha]
        combined = np.concatenate([u_grid, p_grid, alpha_grid], axis=-1)  # (D, H, W, 5)
        combined = np.transpose(combined, (3, 0, 1, 2))  # (5, D, H, W)
        all_states.append(combined)

        elapsed = time.time() - t_snap_start
        if (idx + 1) % 10 == 0 or idx == len(time_dirs) - 1:
            u_mag_max = np.max(np.linalg.norm(u_grid, axis=-1))
            alpha_mean = np.mean(alpha_grid[fluid_mask_3d])
            total_elapsed = time.time() - t_ingest_start
            print(f"[{idx+1:02d}/{len(time_dirs)}] t = {t_val:.2f}s ({elapsed:.2f}s/step) | Max |U|: {u_mag_max:.2f} m/s | Mean alpha: {alpha_mean:.3f} | Total: {total_elapsed:.1f}s")

    states_np = np.stack(all_states, axis=0)  # (T, 5, D, H, W)
    states_tensor = torch.from_numpy(states_np).float()

    input_pairs = states_tensor[:-1]
    target_pairs = states_tensor[1:]

    dataset = {
        "states": states_tensor,
        "input_pairs": input_pairs,
        "target_pairs": target_pairs,
        "times": times,
        "channels": ["ux", "uy", "uz", "p", "alpha.water"],
        "grid_dim": grid_dim,
        "grid_bounds": {"x": [float(x_min), float(x_max)], "y": [float(y_min), float(y_max)], "z": [float(z_min), float(z_max)]},
        "fluid_mask": torch.from_numpy(fluid_mask_3d).bool(),
    }

    torch.save(dataset, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 80)
    print(f"✅ {grid_dim}³ High-Resolution Dataset saved successfully to: {output_path}")
    print(f"   Tensor Shape: {states_tensor.shape} (T, C, D, H, W)")
    print(f"   Saved File Size: {file_size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dim", type=int, default=64)
    parser.add_argument("--output", type=str, default="data/cstr_3d_dataset_64.pt")
    args = parser.parse_args()

    ingest_cstr_3d_multires(grid_dim=args.grid_dim, output_path=args.output)
