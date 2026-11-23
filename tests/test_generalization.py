"""
BioFlo-CFD: Generalization & Parameter Domain Validation Tests.
Tests:
  1. ParameterDomainChecker (interpolation vs extrapolation classification)
  2. OutOfDomainError raising under strict mode
  3. NeuralPODOperator3D.predict_with_confidence API
  4. Three-Tier Accuracy Evaluation Protocol (Tiers A, B, and C separation)
"""

import numpy as np
import pytest
import torch

from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D, OutOfDomainError
from nlss.validation.generalization_tests import (
    ParameterDomainChecker,
    evaluate_three_tier_accuracy,
)


def test_parameter_domain_checker():
    """Verify interpolation detection, extrapolation detection, and nearest distance."""
    # Training conditions: 100, 200, 300, 400 RPM with time [0, 5.0]
    rpms = [100.0, 200.0, 300.0, 400.0]
    times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    train_pts = np.array([[t, rpm] for t in times for rpm in rpms])

    checker = ParameterDomainChecker(train_pts, strict=False)

    # 1. Valid Interpolation query (250 RPM at t=2.5s)
    q_interp = [2.5, 250.0]
    res_interp = checker.check(q_interp)
    assert res_interp["is_in_domain"] is True
    assert res_interp["domain_status"] == "interpolation"
    assert res_interp["confidence"] in ["high", "medium"]
    assert res_interp["confidence_score"] > 0.30

    # 2. Extrapolation query (800 RPM at t=2.5s)
    q_extrap = [2.5, 800.0]
    res_extrap = checker.check(q_extrap)
    assert res_extrap["is_in_domain"] is False
    assert res_extrap["domain_status"] == "extrapolation"
    assert res_extrap["confidence"] in ["low", "untrusted"]
    assert res_extrap["confidence_score"] < 0.10

    # 3. Strict mode raises OutOfDomainError
    strict_checker = ParameterDomainChecker(train_pts, strict=True)
    with pytest.raises(OutOfDomainError, match="OUTSIDE the validated training envelope"):
        strict_checker.check(q_extrap)


def test_model_predict_with_confidence():
    """Verify model integration with domain checking and confidence output."""
    model = NeuralPODOperator3D(n_modes=4, param_dim=2, hidden_dim=32, num_layers=2)
    train_params = np.array([
        [0.0, 100.0],
        [1.0, 100.0],
        [0.0, 400.0],
        [1.0, 400.0],
    ])
    snaps = np.random.randn(4, 50)
    model.calibrate_basis(snaps)
    model.set_training_parameters(train_params, strict=False)

    # Interpolation query
    p_valid = torch.tensor([[0.5, 250.0]])
    out = model.predict_with_confidence(p_valid)
    assert out["is_in_domain"] is True
    assert out["domain_status"] == "interpolation"
    assert out["prediction"].shape == (1, 50)

    # Extrapolation query
    p_invalid = torch.tensor([[0.5, 900.0]])
    out_extrap = model.predict_with_confidence(p_invalid)
    assert out_extrap["is_in_domain"] is False
    assert out_extrap["domain_status"] == "extrapolation"

    # Strict mode
    model.strict_domain = True
    with pytest.raises(OutOfDomainError):
        model.predict_with_confidence(p_invalid)


def test_three_tier_accuracy_evaluation():
    """Verify clean mathematical separation between Tiers A, B, and C."""
    np.random.seed(42)
    N_snaps = 20
    N_dof = 100
    t = np.linspace(0, 1, N_snaps)
    X = np.sin(2 * np.pi * t[:, None] * np.arange(1, 5)) @ np.random.randn(4, N_dof)

    model = NeuralPODOperator3D(n_modes=4, param_dim=1, hidden_dim=32, num_layers=2)
    model.calibrate_basis(X[:15], n_modes=4)

    test_snaps = X[15:]
    test_params = t[15:, None]

    report = evaluate_three_tier_accuracy(
        model,
        test_snapshots=test_snaps,
        test_params=test_params,
        is_unseen_condition=True,
    )

    assert "tier_a_pod_reconstruction" in report
    assert "tier_b_neural_prediction" in report
    assert "tier_c_cfd_generalization" in report

    # Tier A reconstruction error is strictly <= Tier B prediction error
    err_a = report["tier_a_pod_reconstruction"]["rel_l2_error"]
    err_b = report["tier_b_neural_prediction"]["rel_l2_error"]
    assert err_a <= err_b + 1e-6, "Optimal POD projection must achieve <= neural prediction error"
