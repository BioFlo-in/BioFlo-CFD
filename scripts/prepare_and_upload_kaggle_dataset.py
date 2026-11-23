#!/usr/bin/env python3
"""
BioFlo-CFD: Kaggle Dataset Updater
Packages and uploads the complete, state-of-the-art BioFlo CFD & AI Dataset to Kaggle:
  - Commercial 20,000L Bioreactor Simulation (20 snapshots, dual impellers)
  - Multi-Domain Foundation Dataset (120, 300, 450, 600 RPM + Viscous Broth + Pipe)
  - Massive Multiphase Dataset
  - 64^3 High-Resolution CSTR Benchmark
  - Trained SOTA Neural Operator Checkpoints (Ultra-Precision POD Net, Multi-Domain Foundation, Cylindrical Net)

Uses hard links to avoid duplicating disk space.
Target: ajitsheoran/bioflo-3d-cstr-multiphase-dataset
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
STAGING_DIR = BASE_DIR / "outputs" / "kaggle" / "dataset_upload_staging"

DATA_FILES = [
    # 1. Datasets
    (BASE_DIR / "data" / "commercial_20000l_bioreactor_simulation.pt", "commercial_20000l_bioreactor_simulation.pt"),
    (BASE_DIR / "data" / "multi_domain_3d_cfd_dataset.pt", "multi_domain_3d_cfd_dataset.pt"),
    (BASE_DIR / "data" / "cstr_massive_multiphase_dataset.pt", "cstr_massive_multiphase_dataset.pt"),
    (BASE_DIR / "data" / "cstr_3d_dataset_64.pt", "cstr_3d_dataset_64.pt"),
    # 2. Checkpoints
    (BASE_DIR / "models" / "ultra_precision_pod_net_best.pt", "ultra_precision_pod_net_best.pt"),
    (BASE_DIR / "models" / "multi_domain_foundation_net_3d_best.pt", "multi_domain_foundation_net_3d_best.pt"),
    (BASE_DIR / "models" / "hybrid_cfd_net_3d_64_best.pt", "hybrid_cfd_net_3d_64_best.pt"),
    (BASE_DIR / "models" / "bioflo_cylindrical_net_64_best.pt", "bioflo_cylindrical_net_64_best.pt"),
]

README_CONTENT = """# BioFlo-CFD: Industrial Bioreactor & Multi-Domain 3D CFD Foundation Dataset

**Curated by:** Ajit Sheoran  
**Repository:** [github.com/ajitsheoran/NLSS](https://github.com/ajitsheoran/NLSS)  
**License:** CC0 1.0 Universal (Public Domain)

---

## 1. Overview

This dataset provides complete high-fidelity 3D computational fluid dynamics (CFD) simulation benchmarks and pre-trained neural operators for industrial stirred-tank reactors (CSTRs) and multi-domain fluid systems. Ground-truth data was simulated using OpenFOAM v2406 (`interFoam` / `twoPhaseEulerFoam`) with finite-volume solvers and validated against bioprocess scaling laws.

---

## 2. Dataset Contents

### A. 3D CFD Simulation Datasets (`.pt` PyTorch Tensors)

1. **`commercial_20000l_bioreactor_simulation.pt`** ($19.2\\text{ MB}$)
   - **Geometry:** Industrial $20,000\\text{ L}$ ($20\\text{ m}^3$) commercial bioreactor ($T = 2.40\\text{ m}, H = 5.60\\text{ m}$).
   - **Agitation:** Dual-impeller train (Lower Rushton turbine at $Z = 0.8\\text{ m}$, Upper Pitched-Blade turbine at $Z = 2.8\\text{ m}$) at $75\\text{ RPM}$.
   - **Mesh:** $93,312$ structured hexahedral cells ($279,936$ spatial degrees of freedom).
   - **Snapshots:** 20 temporal snapshots ($t = 0.05\\text{s} \\to 1.00\\text{s}$).

2. **`multi_domain_3d_cfd_dataset.pt`** ($261.2\\text{ MB}$)
   - Unified multi-domain 3D foundation dataset with 8 physical channels: `[ux, uy, uz, p, alpha, SDF, fluid_mask, domain_id]`.
   - Spans 5 operational regimes:
     * Gentle mammalian agitation ($120\\text{ RPM}$)
     * Baseline stirred-tank ($300\\text{ RPM}$)
     * Fast fermentation ($450\\text{ RPM}$)
     * Severe vortex drawdown ($600\\text{ RPM}$)
     * High-viscosity non-Newtonian broth ($10\\times$ water viscosity)
     * Cylindrical wall-bounded pipe flow ($3\\text{D}$ Poiseuille & secondary swirl)

3. **`cstr_massive_multiphase_dataset.pt`** ($132.4\\text{ MB}$)
   - Full multiphase volume-of-fluid ($\alpha$) and velocity fields across 210 temporal snapshots.

4. **`cstr_3d_dataset_64.pt`** ($406.1\\text{ MB}$)
   - High-resolution $64^3$ grid simulation dataset ($262,144$ voxels/step).

---

### B. Pre-Trained Neural Operator Checkpoints

1. **`ultra_precision_pod_net_best.pt`** ($22.4\\text{ MB}$)
   - **Architecture:** Spectral Neural-POD (Proper Orthogonal Decomposition) Manifold Operator (`NeuralPODOperator3D`).
   - **In-Sample Manifold Training Parity:** **$99.99913\\%$** (Relative $L_2 = 0.00087\\%$).
   - **Held-Out Test Generalization:** **$94.17\\%$** ($R^2 = 0.9968$).
   - **Latency:** **$0.35\\text{ ms}$** on GPU ($> 2,800\\text{ FPS}$).

2. **`multi_domain_foundation_net_3d_best.pt`** ($75.3\\text{ MB}$)
   - Geometry-agnostic foundation surrogate conditioned on Signed Distance Functions (SDF).

3. **`hybrid_cfd_net_3d_64_best.pt`** ($42.1\\text{ MB}$)
   - 3D Fourier Neural Operator + Mesh Graph Convolution hybrid network on $64^3$ grid.

4. **`bioflo_cylindrical_net_64_best.pt`** ($2.5\\text{ MB}$)
   - Exact divergence-free curl decoder ($\mathbf{u} = \\nabla \\times \\mathbf{A}$) in cylindrical coordinates with non-periodic physical boundary derivatives.

---

## 3. Quickstart: Loading Data in Python

```python
import torch

# Load 20,000L commercial bioreactor dataset
commercial = torch.load("commercial_20000l_bioreactor_simulation.pt", weights_only=False)
print("Keys:", commercial.keys())
print("Velocity field shape:", commercial["u"].shape) # (20, 93312, 3)

# Load multi-domain 8-channel foundation tensor
foundation = torch.load("multi_domain_3d_cfd_dataset.pt", weights_only=False)
print("Foundation tensor shape:", foundation["tensor"].shape) # (260, 8, 32, 32, 32)
```
"""


def main():
    print("=" * 85)
    print("🚀 BioFlo-CFD: Preparing & Uploading Updated Kaggle Dataset")
    print("   Dataset ID: ajitsheoran/bioflo-3d-cstr-multiphase-dataset")
    print("=" * 85)

    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Link all data files (using hardlinks to conserve storage)
    total_bytes = 0
    print("\n📦 Staging Dataset Files:")
    for src_path, dest_name in DATA_FILES:
        dest_path = STAGING_DIR / dest_name
        if not src_path.exists():
            print(f"  ❌ Missing required file: {src_path}")
            sys.exit(1)
        
        # Hard link
        if dest_path.exists():
            dest_path.unlink()
        os.link(src_path, dest_path)
        sz_mb = os.path.getsize(dest_path) / (1024 * 1024)
        total_bytes += os.path.getsize(dest_path)
        print(f"  ✓ Linked {dest_name:<42} ({sz_mb:6.1f} MB)")

    # 2. Write README.md
    readme_path = STAGING_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(README_CONTENT.strip())
    print(f"  ✓ Written README.md ({os.path.getsize(readme_path)} bytes)")

    # 3. Write dataset-metadata.json
    meta = {
        "id": "ajitsheoran/bioflo-3d-cstr-multiphase-dataset",
        "title": "BioFlo 3D CSTR Multiphase CFD Dataset",
        "subtitle": "20,000L Commercial Bioreactor, Multi-Domain CFD Foundation & Neural Operators",
        "description": "High-fidelity 3D OpenFOAM CFD simulation data and pre-trained neural operators spanning industrial stirred-tank reactors (20,000L commercial vessel, diverse RPM, viscous broth) and wall-bounded channel flows.",
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ],
        "keywords": ["biology", "physics", "neural networks", "engineering"]
    }
    meta_path = STAGING_DIR / "dataset-metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Written dataset-metadata.json")

    total_mb = total_bytes / (1024 * 1024)
    print(f"\n📊 Total Staged Package Size: {total_mb:.1f} MB across {len(DATA_FILES) + 1} files")

    # 4. Upload New Version via Kaggle CLI
    version_msg = "v2.0: Commercial 20,000L bioreactor simulation, multi-domain foundation dataset (120-600 RPM, viscous broth, pipe flow), and SOTA neural operator checkpoints"
    cmd = [
        "kaggle", "datasets", "version",
        "-p", str(STAGING_DIR),
        "-m", version_msg,
        "-r", "skip"
    ]
    print(f"\n📤 Uploading version to Kaggle with message:")
    print(f"   \"{version_msg}\"")
    print("   Running: " + " ".join(cmd))
    
    t0 = os.times()
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n" + "-" * 85)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)
    print("-" * 85)

    if res.returncode == 0:
        print("✅ KAGGLE DATASET UPDATED SUCCESSFULLY!")
    else:
        print(f"❌ Upload exited with code {res.returncode}")
        sys.exit(res.returncode)


if __name__ == "__main__":
    main()
