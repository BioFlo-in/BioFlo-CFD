#!/usr/bin/env python3
"""
Ingests 3D OpenFOAM CSTR multiphase CFD snapshots into standardized PyTorch volumetric dataset.
Maps 894,950 unstructured mesh cell values onto a uniform (32 x 32 x 32) 3D Cartesian tensor grid.
Channels: [u_x, u_y, u_z, p, alpha.water].
"""

import os
import re
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch

from nlss.io.foam_reader import read_foam_field_file


def ingest_cstr_3d(
    case_dir: str = "openfoam_cases/professional_cstr_3d_gpu_mrf",
    output_path: str = "data/cstr_3d_real_openfoam_dataset.pt",
    grid_dim: int = 32,
):
    print("=" * 70)
    print("🚀 BioFlo-CFD: Ingesting Real 3D OpenFOAM CSTR Multiphase Dataset")
    print("=" * 70)
    case_path = Path(case_dir)
    os.makedirs(Path(output_path).parent, exist_ok=True)

    # 1. Discover all time directories
    time_dirs = [d for d in os.listdir(case_path) if re.match(r"^[0-9]+(\.[0-9]+)?$", d)]
    time_dirs = sorted(time_dirs, key=lambda x: float(x))
    times = [float(d) for d in time_dirs]
    print(f"Found {len(times)} time directories spanning t = {times[0]:.2f}s to {times[-1]:.2f}s (dt = {times[1]-times[0]:.3f}s)")

    # 2. Load cell centers from 1.95/C to construct spatial KD-Tree
    c_file = case_path / "1.95" / "C"
    if not c_file.exists():
        raise FileNotFoundError(f"Cell coordinates file not found at {c_file}!")

    print("Loading 3D mesh cell centers...")
    t0 = time.time()
    c_field = read_foam_field_file(c_file)
    cell_coords = c_field.internal_field
    print(f"Loaded {cell_coords.shape[0]:,} cell coordinates in {time.time() - t0:.2f}s")

    # 3. Create regular 3D bounding-box grid
    x_min, y_min, z_min = cell_coords.min(axis=0)
    x_max, y_max, z_max = cell_coords.max(axis=0)
    print(f"Domain Bounds: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}], Z=[{z_min:.3f}, {z_max:.3f}]")

    gx = np.linspace(x_min, x_max, grid_dim, dtype=np.float32)
    gy = np.linspace(y_min, y_max, grid_dim, dtype=np.float32)
    gz = np.linspace(z_min, z_max, grid_dim, dtype=np.float32)
    GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
    grid_pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], axis=-1)

    print(f"Building spatial cKDTree for {grid_dim}³ = {grid_pts.shape[0]:,} voxels...")
    t0 = time.time()
    tree = cKDTree(cell_coords)
    _, grid_indices = tree.query(grid_pts, k=1)
    print(f"KDTree mapping completed in {time.time() - t0:.2f}s")

    # 4. Ingest each time snapshot
    all_states = []
    print(f"\nProcessing {len(time_dirs)} snapshots onto ({grid_dim} x {grid_dim} x {grid_dim}) grid...")

    for idx, (t_str, t_val) in enumerate(zip(time_dirs, times)):
        t_dir = case_path / t_str
        t_snap_start = time.time()

        # Read U, p, alpha.water
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

        # Combine into (5, D, H, W) -> [ux, uy, uz, p, alpha]
        combined = np.concatenate([u_grid, p_grid, alpha_grid], axis=-1)  # (D, H, W, 5)
        combined = np.transpose(combined, (3, 0, 1, 2))  # (5, D, H, W)
        all_states.append(combined)

        elapsed = time.time() - t_snap_start
        if (idx + 1) % 10 == 0 or idx == len(time_dirs) - 1:
            u_mag_max = np.max(np.linalg.norm(u_grid, axis=-1))
            alpha_mean = np.mean(alpha_grid)
            print(f"[{idx+1:02d}/{len(time_dirs)}] t = {t_val:.2f}s ({elapsed:.2f}s/step) | Max |U|: {u_mag_max:.2f} m/s | Mean alpha: {alpha_mean:.3f}")

    states_np = np.stack(all_states, axis=0)  # (T, 5, D, H, W)
    states_tensor = torch.from_numpy(states_np).float()

    # Form (S_t -> S_{t+dt}) state-transition pairs
    input_pairs = states_tensor[:-1]   # (T-1, 5, D, H, W)
    target_pairs = states_tensor[1:]   # (T-1, 5, D, H, W)

    dataset_dict = {
        "states": states_tensor,
        "times": times,
        "channels": ["ux", "uy", "uz", "p", "alpha.water"],
        "input_pairs": input_pairs,
        "target_pairs": target_pairs,
        "grid_dim": grid_dim,
        "grid_bounds": {"x": (x_min, x_max), "y": (y_min, y_max), "z": (z_min, z_max)},
        "grid_coords": {"x": gx, "y": gy, "z": gz},
    }

    torch.save(dataset_dict, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Dataset successfully generated and saved to: {output_path}")
    print(f"   Tensor Shape: {states_tensor.shape} (T, C, D, H, W)")
    print(f"   Input-Target Pairs: {input_pairs.shape[0]} transitions")
    print(f"   Saved File Size: {file_size_mb:.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    ingest_cstr_3d()
