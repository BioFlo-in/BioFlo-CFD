"""
Online OpenFOAM & CFD Datasets Manager.
Provides catalogs, downloaders, and converters for prominent public CFD datasets:
  1. AirfRANS (NeurIPS 2022) - 1,000 OpenFOAM RANS simulations around NACA airfoils.
  2. PDEBench (ICLR 2023) - 2D/3D Navier-Stokes & turbulence benchmarks.
  3. DeepMind CylinderFlow - Flow past a cylinder (Karman vortex street).
  4. DrivAerNet - 4,000 3D car aerodynamic OpenFOAM simulations.
  5. OpenFOAM Official Tutorials Repository - Standard verified cases (pitzDaily, cavity, etc.).
"""

import json
import os
from pathlib import Path
import shutil
import urllib.request
import zipfile
import tarfile
from typing import Dict, List, Optional, Union
import numpy as np
import torch

from .foam_dataset import CFDTrajectoryDataset, FieldNormalizer


ONLINE_DATASETS_CATALOG = {
    "airfrans": {
        "name": "AirfRANS (Inria & IRT SystemX)",
        "conference": "NeurIPS 2022 (Datasets & Benchmarks)",
        "description": "1,000 high-fidelity OpenFOAM RANS simulations (Spalart-Allmaras) over 2D NACA airfoils. Targets: velocity U, pressure p, and turbulent viscosity nut.",
        "size": "~2.4 GB",
        "mesh_type": "Unstructured OpenFOAM polyhedral mesh & point cloud",
        "homepage": "https://github.com/Extrality/airfrans_lib",
        "download_url": "https://huggingface.co/datasets/Extrality/AirfRANS",
        "install_cmd": "uv pip install airfrans",
    },
    "pdebench": {
        "name": "PDEBench (2D Navier-Stokes & Turbulence)",
        "conference": "ICLR 2023",
        "description": "Benchmark suite of 2D/3D Navier-Stokes equations with varied Reynolds numbers, decaying and forced turbulence.",
        "size": "Variable (subsets ~500 MB to 10 GB)",
        "mesh_type": "Regular Cartesian Grid (128x128 / 512x512)",
        "homepage": "https://github.com/pdebench/PDEBench",
        "download_url": "https://darus.uni-stuttgart.de/dataverse/sciml_benchmark",
    },
    "cylinder_flow": {
        "name": "DeepMind CylinderFlow (Karman Vortex Street)",
        "description": "Unsteady incompressible 2D flow past a circular cylinder showing vortex shedding at Re=100-300.",
        "size": "~300 MB",
        "mesh_type": "Unstructured triangular mesh & regular grid",
        "homepage": "https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets",
        "download_url": "https://storage.googleapis.com/learning-to-simulate-complex-physics/datasets/cylinder_flow.zip",
    },
    "drivaernet": {
        "name": "DrivAerNet (3D Vehicle Aerodynamics)",
        "conference": "ASME / NeurIPS",
        "description": "4,000 full 3D OpenFOAM simulations of vehicle aerodynamics (simpleFoam) for surface pressure and aerodynamic drag prediction.",
        "size": "~100 GB+",
        "mesh_type": "3D Unstructured polyhedral OpenFOAM mesh",
        "homepage": "https://drivaernet.github.io/",
        "download_url": "https://drivaernet.github.io/download",
    },
    "openfoam_tutorials": {
        "name": "OpenFOAM Official Verified Cases",
        "description": "Collection of verified standard OpenFOAM cases: pitzDaily (backward-facing step), cavity (lid-driven), motorBike, damBreak.",
        "size": "~50 MB",
        "mesh_type": "Native OpenFOAM case directories",
        "homepage": "https://github.com/OpenFOAM/OpenFOAM-10/tree/master/tutorials",
        "download_url": "https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-10/master/tutorials/",
    },
}


def list_available_online_datasets() -> Dict[str, dict]:
    """Returns catalog of online OpenFOAM and CFD datasets."""
    return ONLINE_DATASETS_CATALOG


def print_datasets_table():
    """Prints a formatted summary of available online CFD datasets."""
    print("=" * 80)
    print("           ONLINE OPENFOAM & CFD MACHINE LEARNING DATASETS")
    print("=" * 80)
    for key, info in ONLINE_DATASETS_CATALOG.items():
        print(f"\n📂 [{key.upper()}] {info['name']}")
        if "conference" in info:
            print(f"   • Venue:        {info['conference']}")
        print(f"   • Description:  {info['description']}")
        print(f"   • Mesh Type:    {info['mesh_type']}")
        print(f"   • Size:         {info['size']}")
        print(f"   • Homepage:     {info['homepage']}")
        if "install_cmd" in info:
            print(f"   • Quick Install:{info['install_cmd']}")
    print("\n" + "=" * 80)


def download_file_with_progress(url: str, output_path: Union[str, Path]) -> Path:
    """Downloads a file from a URL with basic progress logging."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"📥 Downloading: {url} -> {out}")

    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100.0, (downloaded / total_size) * 100.0)
            if block_num % 500 == 0:
                print(f"   Progress: {pct:.1f}% ({downloaded / 1e6:.1f} MB / {total_size / 1e6:.1f} MB)")

    urllib.request.urlretrieve(url, out, reporthook=reporthook)
    print(f"✓ Download complete: {out}")
    return out


def fetch_openfoam_github_case(
    solver: str,
    case_name: str,
    output_dir: Union[str, Path],
    version: str = "10",
) -> Path:
    """
    Fetches a standard tutorial case directly from the official OpenFOAM GitHub repository.
    Example: solver='incompressible/icoFoam', case_name='cavity/cavity'
    """
    base_url = f"https://raw.githubusercontent.com/OpenFOAM/OpenFOAM-{version}/master/tutorials/{solver}/{case_name}"
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"📥 Fetching OpenFOAM official tutorial: {case_name} -> {target}")
    # Standard files
    files_to_fetch = [
        "0/U",
        "0/p",
        "constant/transportProperties",
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "system/blockMeshDict",
    ]

    fetched = 0
    for rel_path in files_to_fetch:
        file_url = f"{base_url}/{rel_path}"
        dest_file = target / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(
                file_url,
                headers={"User-Agent": "Mozilla/5.0 (NLSS-OpenFOAM-Agent)"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode("utf-8")
                dest_file.write_text(content, encoding="utf-8")
                fetched += 1
        except Exception as e:
            # Not all cases have all files (e.g. blockMeshDict may vary)
            pass

    print(f"✓ Retrieved {fetched} OpenFOAM case files in: {target}")
    return target
