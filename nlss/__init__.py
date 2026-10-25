"""
NLSS: Neural Navier-Stokes Learned Surrogate
AI model framework for accelerating OpenFOAM CFD simulations.
"""

from .models.fno import FNO2d
from .models.unet import ResUNetCFD
from .models.gnn import MeshGNN
from .physics.losses import PhysicsInformedCFDLoss, compute_divergence_2d, compute_vorticity_2d
from .dataset.generator import NavierStokes2DSimulator, generate_and_export_openfoam_benchmark
from .dataset.foam_dataset import CFDTrajectoryDataset, FieldNormalizer, load_dataset_from_openfoam_case
from .rollout.engine import RolloutEngine, benchmark_acceleration
from .hybrid.warm_starter import OpenFOAMHybridAccelerator
from .visualization.plotter import plot_field_comparison, plot_rollout_metrics

__version__ = "0.1.0"
__all__ = [
    "FNO2d",
    "ResUNetCFD",
    "MeshGNN",
    "PhysicsInformedCFDLoss",
    "compute_divergence_2d",
    "compute_vorticity_2d",
    "NavierStokes2DSimulator",
    "generate_and_export_openfoam_benchmark",
    "CFDTrajectoryDataset",
    "FieldNormalizer",
    "load_dataset_from_openfoam_case",
    "RolloutEngine",
    "benchmark_acceleration",
    "OpenFOAMHybridAccelerator",
    "plot_field_comparison",
    "plot_rollout_metrics",
]
