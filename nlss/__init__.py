"""
BioFlo-CFD: AI Surrogate & OpenFOAM Hybrid Accelerator
Physics-informed neural operators for fluid dynamics & bioprocess digital twins.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import torch

from .models.fno import FNO2d
from .models.unet import ResUNetCFD
from .models.gnn import MeshGNN
from .models.hybrid_cfd_net import HybridCFDNet
from .physics.losses import PhysicsInformedCFDLoss, compute_divergence_2d, compute_vorticity_2d
from .dataset.generator import NavierStokes2DSimulator, generate_and_export_openfoam_benchmark
from .dataset.bioreactor import BioreactorSimulator, BioreactorGeometry, compute_bioprocess_shear_metrics
from .dataset.foam_dataset import CFDTrajectoryDataset, FieldNormalizer, load_dataset_from_openfoam_case
from .rollout.engine import RolloutEngine, benchmark_acceleration
from .hybrid.warm_starter import OpenFOAMHybridAccelerator
from .visualization.plotter import plot_field_comparison, plot_rollout_metrics

__version__ = "0.1.0"


def load_pretrained_model(
    device: Optional[Union[str, torch.device]] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[HybridCFDNet, FieldNormalizer]:
    """
    Loads the official pre-trained BioFlo-CFD HybridCFDNet surrogate model.
    Returns:
        (model, normalizer): Ready-to-use evaluation model and field normalizer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if weights_path is None:
        root = Path(__file__).resolve().parent.parent
        weights_path = root / "pretrained/bioflo_cfd_hybrid_net.pt"

    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at: {weights_path}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"]
    preset = ckpt.get("preset", "base")

    model = HybridCFDNet.from_preset(preset, in_channels=3, out_channels=3, enforce_boundary=False).to(device)
    model.load_state_dict(sd)
    model.eval()

    if "normalizer" in ckpt and isinstance(ckpt["normalizer"], dict):
        normalizer = FieldNormalizer.from_dict(ckpt["normalizer"])
    elif "normalizer_mean" in ckpt:
        normalizer = FieldNormalizer()
        normalizer.mean = ckpt["normalizer_mean"].to(device)
        normalizer.std = ckpt["normalizer_std"].to(device)
    else:
        normalizer = FieldNormalizer()

    return model, normalizer


__all__ = [
    "HybridCFDNet",
    "FNO2d",
    "ResUNetCFD",
    "MeshGNN",
    "BioreactorSimulator",
    "BioreactorGeometry",
    "compute_bioprocess_shear_metrics",
    "PhysicsInformedCFDLoss",
    "compute_divergence_2d",
    "compute_vorticity_2d",
    "NavierStokes2DSimulator",
    "generate_and_export_openfoam_benchmark",
    "CFDTrajectoryDataset",
    "FieldNormalizer",
    "load_dataset_from_openfoam_case",
    "load_pretrained_model",
    "RolloutEngine",
    "benchmark_acceleration",
    "OpenFOAMHybridAccelerator",
    "plot_field_comparison",
    "plot_rollout_metrics",
]
