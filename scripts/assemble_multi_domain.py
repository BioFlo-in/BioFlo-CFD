#!/usr/bin/env python3
"""
Assembles the Unified Multi-Domain 3D CFD Foundation Dataset (8 Channels):
Channels: [ux, uy, uz, p, alpha, SDF, fluid_mask, domain_id]
Integrates:
  - Domain 1: All 210 CSTR Bioreactor Snapshots across 5 Operational Regimes
  - Domain 2: 50 Wall-Bounded 3D Pipe Flow & Channel Manifold Snapshots
"""

import os
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

GRID_DIM = 32


def compute_sdf_from_mask(mask: np.ndarray, dx: float = 1.0) -> np.ndarray:
    dist_outside = ndi.distance_transform_edt(~mask, sampling=dx)
    dist_inside = ndi.distance_transform_edt(mask, sampling=dx)
    return (dist_outside - dist_inside).astype(np.float32)


def main():
    print("=" * 85)
    print("🌐 BioFlo-CFD: Assembling Unified Multi-Domain 3D Foundation Dataset")
    print("=" * 85)

    cstr_path = DATA_DIR / "cstr_massive_multiphase_dataset.pt"
    pipe_path = DATA_DIR / "cfd_3d_dataset.pt"
    out_path = DATA_DIR / "multi_domain_3d_cfd_dataset.pt"

    assert cstr_path.exists(), f"CSTR dataset missing at {cstr_path}"
    assert pipe_path.exists(), f"Pipe dataset missing at {pipe_path}"

    cstr_data = torch.load(cstr_path, weights_only=False)
    pipe_data = torch.load(pipe_path, weights_only=False)

    multi_domain_samples = []

    # 1. Domain 1: CSTR Bioreactor (Domain ID = 1.0)
    sdf_cstr = cstr_data["sdf"].unsqueeze(0)  # (1, D, H, W)
    mask_cstr = cstr_data["fluid_mask"].float().unsqueeze(0)  # (1, D, H, W)
    domain_id_cstr = torch.ones((1, GRID_DIM, GRID_DIM, GRID_DIM), dtype=torch.float32)

    total_cstr = 0
    for case in cstr_data["cases"]:
        states = case["states"]  # (T, 5, D, H, W)
        for s in states:
            sample_8ch = torch.cat([s, sdf_cstr, mask_cstr, domain_id_cstr], dim=0)  # (8, D, H, W)
            multi_domain_samples.append(sample_8ch)
            total_cstr += 1

    print(f"✓ Ingested {total_cstr} CSTR samples across {len(cstr_data['cases'])} cases")

    # 2. Domain 2: Pipe Flow (Domain ID = 2.0)
    x_train_pipe = pipe_data["x_train"].float()  # (106, 4, D, H, W) -> [ux, uy, uz, p]
    N_pipe = 50
    pipe_subset = x_train_pipe[:N_pipe]

    lin = np.linspace(-1.0, 1.0, GRID_DIM)
    Z_p, Y_p, X_p = np.meshgrid(lin, lin, lin, indexing="ij")
    pipe_mask = (np.sqrt(Y_p**2 + Z_p**2) <= 0.85)
    sdf_pipe = compute_sdf_from_mask(pipe_mask, dx=2.0 / (GRID_DIM - 1))

    sdf_tensor_pipe = torch.from_numpy(sdf_pipe).unsqueeze(0)
    mask_tensor_pipe = torch.from_numpy(pipe_mask.astype(np.float32)).unsqueeze(0)
    alpha_pipe = torch.ones((1, GRID_DIM, GRID_DIM, GRID_DIM), dtype=torch.float32)
    alpha_pipe[:, ~pipe_mask] = 0.0
    domain_id_pipe = torch.full((1, GRID_DIM, GRID_DIM, GRID_DIM), 2.0, dtype=torch.float32)

    for s in pipe_subset:
        sample_8ch = torch.cat([s, alpha_pipe, sdf_tensor_pipe, mask_tensor_pipe, domain_id_pipe], dim=0)
        multi_domain_samples.append(sample_8ch)

    print(f"✓ Ingested {N_pipe} Pipe Flow samples (Domain 2)")

    foundation_tensor = torch.stack(multi_domain_samples, dim=0)  # (N_total, 8, D, H, W)

    torch.save({
        "dataset_name": "BioFlo-MultiDomain-3D-CFD-Foundation",
        "tensor": foundation_tensor,
        "num_samples": foundation_tensor.shape[0],
        "grid_dim": GRID_DIM,
        "channels": ["ux", "uy", "uz", "p", "alpha", "SDF", "fluid_mask", "domain_id"],
        "domains": ["CSTR_Bioreactor", "Pipe_Channel"],
        "cases_included": [c["name"] for c in cstr_data["cases"]] + ["3D Cylindrical Pipe Flow"],
        "description": "Massive Multi-Domain 3D CFD Foundation Dataset spanning 5 CSTR operational regimes (120, 300, 450, 600 RPM + Viscous Broth) and Wall-Bounded Pipe Flow.",
    }, out_path)

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("\n" + "=" * 85)
    print("✅ MULTI-DOMAIN FOUNDATION DATASET ASSEMBLY COMPLETE!")
    print(f"   Final Tensor Shape: {foundation_tensor.shape} (N, C, D, H, W)")
    print(f"   Total Multi-Regime Snapshots: {foundation_tensor.shape[0]}")
    print(f"   Output File: {out_path} ({file_size_mb:.1f} MB)")
    print("=" * 85)


if __name__ == "__main__":
    main()
