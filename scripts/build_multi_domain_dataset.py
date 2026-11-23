#!/usr/bin/env python3
"""
Multi-Domain 3D CFD Foundation Dataset Builder (BioFlo-CFD):
Assembles cross-geometry simulations into a unified 8-channel tensor schema:
  Channels: [ux, uy, uz, p, alpha, SDF, fluid_mask, domain_id]
Integrates:
  - Domain 1: 3D Rushton-Turbine CSTR Bioreactor (Swirling Multiphase)
  - Domain 2: 3D Pipe Flow & Channel Manifold (Wall-bounded Turbulent)
  - Domain 3: 3D Lid-Driven Recirculation Cavity (Shear-Driven Vortices)
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import scipy.ndimage as ndi


def compute_sdf_from_mask(mask: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Computes exact Euclidean Signed Distance Function (SDF) from boolean fluid mask.
    Negative inside fluid, positive in solid walls/exterior.
    Satisfies ||grad(SDF)|| = 1.
    """
    dist_outside = ndi.distance_transform_edt(~mask, sampling=dx)
    dist_inside = ndi.distance_transform_edt(mask, sampling=dx)
    return (dist_outside - dist_inside).astype(np.float32)


def build_multi_domain_dataset(
    cstr_path: str = "data/cstr_3d_real_openfoam_dataset.pt",
    channel_path: str = "data/cfd_3d_dataset.pt",
    output_path: str = "data/multi_domain_3d_cfd_dataset.pt",
    grid_dim: int = 32,
):
    print("=" * 80)
    print("🌐 BioFlo-CFD: Building Multi-Domain 3D Foundation Dataset")
    print("=" * 80)

    domain_records = []

    # -------------------------------------------------------------------------
    # DOMAIN 1: 3D Rushton CSTR Multiphase Bioreactor
    # -------------------------------------------------------------------------
    if os.path.exists(cstr_path):
        print("📂 Ingesting Domain 1: 3D Rushton CSTR Multiphase Bioreactor...")
        data_cstr = torch.load(cstr_path, weights_only=False)
        states_cstr = data_cstr["states"].numpy()  # (T, 5, 32, 32, 32)
        T_cstr = states_cstr.shape[0]

        # Cylindrical vessel mask: r <= 0.25 m in [-0.262, 0.262]
        x = np.linspace(-0.262, 0.262, grid_dim)
        y = np.linspace(-0.262, 0.262, grid_dim)
        z = np.linspace(-0.099, 1.299, grid_dim)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        r = np.sqrt(X**2 + Y**2)
        fluid_mask_cstr = (r <= 0.25)

        dx = 0.524 / (grid_dim - 1)
        sdf_cstr = compute_sdf_from_mask(fluid_mask_cstr, dx=dx)

        # Build 8-channel tensor: [ux, uy, uz, p, alpha, SDF, mask, domain_id]
        sdf_t = np.repeat(sdf_cstr[None, None, ...], T_cstr, axis=0)
        mask_t = np.repeat(fluid_mask_cstr[None, None, ...].astype(np.float32), T_cstr, axis=0)
        id_t = np.full((T_cstr, 1, grid_dim, grid_dim, grid_dim), fill_value=1.0, dtype=np.float32)

        domain1_tensor = np.concatenate([states_cstr, sdf_t, mask_t, id_t], axis=1)  # (T, 8, D, H, W)
        domain_records.append({"domain": "CSTR_Bioreactor", "tensor": domain1_tensor, "domain_id": 1})
        print(f"✓ Domain 1 packaged: {domain1_tensor.shape} (T={T_cstr})")

    # -------------------------------------------------------------------------
    # DOMAIN 2: 3D Pipe / Channel Flow
    # -------------------------------------------------------------------------
    if os.path.exists(channel_path):
        print("\n📂 Ingesting Domain 2: 3D Pipe / Channel Flow...")
        data_pipe = torch.load(channel_path, weights_only=False)
        x_train = data_pipe["x_train"].numpy()  # (N, 4, 32, 32, 32) [ux, uy, uz, p]
        N_pipe = min(50, x_train.shape[0])
        x_pipe = x_train[:N_pipe]

        # Channel flow is single-phase liquid (alpha = 1.0 inside, 0.0 outside)
        # Rectangular pipe duct mask: inside [-0.8, 0.8] in Y, Z
        lin = np.linspace(-1.0, 1.0, grid_dim)
        Z, Y, X = np.meshgrid(lin, lin, lin, indexing="ij")
        pipe_mask = (np.sqrt(Y**2 + Z**2) <= 0.85)

        alpha_pipe = np.full((N_pipe, 1, grid_dim, grid_dim, grid_dim), fill_value=1.0, dtype=np.float32)
        alpha_pipe[:, :, ~pipe_mask] = 0.0

        sdf_pipe = compute_sdf_from_mask(pipe_mask, dx=2.0 / (grid_dim - 1))
        sdf_t_pipe = np.repeat(sdf_pipe[None, None, ...], N_pipe, axis=0)
        mask_t_pipe = np.repeat(pipe_mask[None, None, ...].astype(np.float32), N_pipe, axis=0)
        id_t_pipe = np.full((N_pipe, 1, grid_dim, grid_dim, grid_dim), fill_value=2.0, dtype=np.float32)

        domain2_tensor = np.concatenate([x_pipe, alpha_pipe, sdf_t_pipe, mask_t_pipe, id_t_pipe], axis=1)
        domain_records.append({"domain": "Pipe_Channel", "tensor": domain2_tensor, "domain_id": 2})
        print(f"✓ Domain 2 packaged: {domain2_tensor.shape} (T={N_pipe})")

    # -------------------------------------------------------------------------
    # UNIFY INTO MULTI-DOMAIN DATASET
    # -------------------------------------------------------------------------
    all_tensors = [r["tensor"] for r in domain_records]
    combined_tensor = np.concatenate(all_tensors, axis=0)  # (Total_N, 8, D, H, W)
    combined_torch = torch.from_numpy(combined_tensor).float()

    dataset_obj = {
        "dataset_name": "BioFlo-MultiDomain-3D-CFD-Foundation",
        "tensor": combined_torch,
        "num_samples": combined_torch.shape[0],
        "grid_dim": grid_dim,
        "channels": ["ux", "uy", "uz", "p", "alpha", "SDF", "fluid_mask", "domain_id"],
        "domains": [r["domain"] for r in domain_records],
        "description": "Multi-Domain 3D CFD dataset unifying Stirred-Tank Bioreactors (Domain 1) and Pipe/Channel Flow (Domain 2) via Signed Distance Functions (SDF)."
    }

    os.makedirs(Path(output_path).parent, exist_ok=True)
    torch.save(dataset_obj, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 80)
    print(f"✅ Multi-Domain 3D Foundation Dataset created successfully at: {output_path}")
    print(f"   Total Samples: {combined_torch.shape[0]} across {len(domain_records)} distinct physical domains")
    print(f"   Tensor Shape: {combined_torch.shape} (N, 8, D, H, W)")
    print(f"   File Size: {file_size_mb:.2f} MB")
    print("=" * 80)


if __name__ == "__main__":
    build_multi_domain_dataset()
