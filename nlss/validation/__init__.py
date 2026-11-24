"""
BioFlo-CFD: Scientific Validation & Verification Subsystem (NLSS v1.0 Standard).
"""

from .velocity_metrics import (
    relative_l2_error,
    relative_l1_error,
    relative_linf_error,
    rmse_mae_metrics,
    r2_score_field,
    component_wise_errors,
    vector_angular_metrics,
    compute_comprehensive_velocity_metrics,
)
from .physics_residuals import (
    compute_discrete_divergence,
    compute_discrete_vorticity,
    compute_vorticity_metrics,
    compute_momentum_residual_steady,
)
from .region_metrics import (
    generate_bioreactor_rois,
    evaluate_roi_metrics,
)
from .generalization_tests import (
    OutOfDomainError,
    ParameterDomainChecker,
    evaluate_three_tier_accuracy,
)
from .bioprocess_metrics import (
    compute_strain_rate_tensor,
    compute_viscous_dissipation_rate,
    compute_specific_power_input,
    compute_kla,
    compute_mixing_time,
    compute_higbie_mass_transfer,
    compute_tracer_mixing_time_t95,
    compute_dimensionless_bioprocess_numbers,
    evaluate_bioprocess_field_comparison,
)

__all__ = [
    "relative_l2_error",
    "relative_l1_error",
    "relative_linf_error",
    "rmse_mae_metrics",
    "r2_score_field",
    "component_wise_errors",
    "vector_angular_metrics",
    "compute_comprehensive_velocity_metrics",
    "compute_discrete_divergence",
    "compute_discrete_vorticity",
    "compute_vorticity_metrics",
    "compute_momentum_residual_steady",
    "generate_bioreactor_rois",
    "evaluate_roi_metrics",
    "OutOfDomainError",
    "ParameterDomainChecker",
    "evaluate_three_tier_accuracy",
    "compute_strain_rate_tensor",
    "compute_viscous_dissipation_rate",
    "compute_specific_power_input",
    "compute_kla",
    "compute_mixing_time",
    "compute_higbie_mass_transfer",
    "compute_tracer_mixing_time_t95",
    "compute_dimensionless_bioprocess_numbers",
    "evaluate_bioprocess_field_comparison",
]
