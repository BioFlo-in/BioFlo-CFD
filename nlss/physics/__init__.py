from .losses import (
    compute_divergence_2d,
    compute_vorticity_2d,
    DeltaTransitionLoss,
    NormalizedMSELoss,
    PhysicsInformedCFDLoss,
)

__all__ = [
    "compute_divergence_2d",
    "compute_vorticity_2d",
    "DeltaTransitionLoss",
    "NormalizedMSELoss",
    "PhysicsInformedCFDLoss",
]
