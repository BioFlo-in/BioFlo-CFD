"""
BioFlo-CFD: Unit Tests for Benchmark & Reporting Engine (NLSS v1.0).
Tests:
  1. Latency & Throughput Benchmark profiling
  2. OpenFOAM Benchmark Runner end-to-end execution
  3. Automatic JSON and HTML report generation
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from nlss.benchmark.inference_benchmark import benchmark_model_inference
from nlss.benchmark.openfoam_benchmark import OpenFOAMBenchmarkRunner


def test_inference_benchmark_profiling():
    """Verify latency percentiles, speedup factor, and throughput calculations."""
    model = NeuralPODOperator3D(n_modes=5, param_dim=1, hidden_dim=32, num_layers=2)
    dummy_snaps = np.random.randn(8, 100)
    model.calibrate_basis(dummy_snaps)

    profile = benchmark_model_inference(
        model,
        param_dim=1,
        batch_sizes=[1, 4],
        n_warmup=5,
        n_iterations=15,
        device="cpu",
        openfoam_step_latency_sec=10.0,
    )

    assert "device" in profile
    assert profile["single_snapshot_latency_ms"] > 0.0
    assert profile["cfd_speedup_factor"] > 10.0
    assert "batch_1" in profile["batches"]
    assert "batch_4" in profile["batches"]
    assert profile["batches"]["batch_4"]["throughput_snapshots_per_sec"] > 0.0


def test_openfoam_benchmark_runner_and_reports(tmp_path: Path):
    """Verify complete benchmark execution and JSON/HTML report generation."""
    D, H, W = 10, 10, 10
    channels = 3
    N_voxels = D * H * W
    N_dof = channels * N_voxels
    N_snaps = 12

    # Synthesize data
    np.random.seed(42)
    t = np.linspace(0, 1, N_snaps)
    X = np.sin(2 * np.pi * t[:, None] * np.arange(1, 4)) @ np.random.randn(3, N_dof)

    # Train model
    model = NeuralPODOperator3D(n_modes=3, param_dim=1, hidden_dim=32, num_layers=2)
    model.calibrate_basis(X[:8], n_modes=3)
    model.set_training_parameters(t[:8, None])

    test_snaps = X[8:]
    test_params = t[8:, None]

    runner = OpenFOAMBenchmarkRunner(
        model=model,
        grid_shape=(D, H, W),
        spacings=(0.05, 0.05, 0.10),
        tank_radius=0.25,
        impeller_radius=0.083,
        liquid_height=1.0,
    )

    report_data = runner.run_benchmark(
        test_snapshots=test_snaps,
        test_parameters=test_params,
        is_unseen_condition=False,
        operating_condition_label="Unit Test Held-Out Verification",
    )

    assert "three_tier_hierarchy" in report_data
    assert "global_velocity_metrics" in report_data
    assert "regional_roi_metrics" in report_data
    assert "impeller_discharge" in report_data["regional_roi_metrics"]
    assert "physics_residuals" in report_data
    assert "inference_performance" in report_data

    # Export reports
    json_path, html_path = runner.export_reports(
        report_data,
        output_dir=tmp_path / "reports",
        json_filename="test_report.json",
        html_filename="test_report.html",
    )

    assert json_path.exists()
    assert html_path.exists()

    with open(json_path, "r") as f:
        loaded_json = json.load(f)
    assert loaded_json["condition_label"] == "Unit Test Held-Out Verification"

    with open(html_path, "r") as f:
        html_text = f.read()
    assert "BioFlo-CFD" in html_text
    assert "Impeller Discharge Zone" in html_text
