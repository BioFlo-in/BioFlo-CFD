import tempfile
from pathlib import Path
import pytest
import torch

from nlss.dataset.generator import generate_and_export_openfoam_benchmark
from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, CFDTrajectoryDataset
from nlss.train import create_model, train_surrogate_model
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.io.foam_reader import get_case_time_directories


def test_complete_simulation_acceleration_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        case_dir = tmp_path / "lid_cavity_foam"
        ckpt_dir = tmp_path / "checkpoints"
        ai_foam_dir = tmp_path / "ai_foam_out"

        # 1. Generate OpenFOAM CFD simulation data (32x32 grid, 10 snapshots)
        generate_and_export_openfoam_benchmark(
            output_case_dir=case_dir,
            nx=32,
            ny=32,
            n_steps=20,
            save_interval=2,
            nu=0.01,
            dt=0.001,
            u_lid=1.0,
        )

        time_dirs = get_case_time_directories(case_dir)
        assert len(time_dirs) >= 10, f"Expected at least 10 time directories, got {len(time_dirs)}"

        # 2. Load dataset from OpenFOAM directory structure
        dataset, normalizer = load_dataset_from_openfoam_case(
            case_dir=case_dir,
            grid_shape=(32, 32),
        )
        assert len(dataset) >= 9
        assert dataset.data.shape[1] == 3  # (ux, uy, p)

        # 3. Instantiate FNO model & train for 3 epochs
        model = create_model(model_type="fno", in_channels=3, out_channels=3, modes1=8, modes2=8, width=24)
        history = train_surrogate_model(
            model=model,
            train_dataset=dataset,
            epochs=3,
            batch_size=4,
            lr=1e-3,
            checkpoint_dir=ckpt_dir,
            log_interval=1,
        )

        assert len(history["train_loss"]) == 3
        assert (ckpt_dir / "final_model.pt").exists()

        # 4. Rollout & Acceleration Benchmark
        engine = RolloutEngine(model=model, normalizer=normalizer)
        init_state = dataset.data[0]  # (3, 32, 32)

        pred_traj, ai_time = engine.rollout(init_state, num_steps=5)
        assert pred_traj.shape == (6, 3, 32, 32)
        assert ai_time > 0

        # Benchmark speedup
        bench_metrics = benchmark_acceleration(
            rollout_engine=engine,
            ground_truth_trajectory=dataset.data[:6],
            cfd_solver_time_sec=0.5,
            num_steps=5,
        )
        assert "speedup_factor" in bench_metrics
        assert bench_metrics["speedup_factor"] > 0

        # 5. Export predicted trajectory back to OpenFOAM format
        times = [i * 0.002 for i in range(6)]
        out_p = engine.export_rollout_to_openfoam(pred_traj, times, ai_foam_dir)
        ai_time_dirs = get_case_time_directories(ai_foam_dir)
        assert len(ai_time_dirs) == 6
        assert (ai_foam_dir / "0" / "U").exists()
        assert (ai_foam_dir / "0" / "p").exists()


if __name__ == "__main__":
    test_complete_simulation_acceleration_pipeline()
    print("End-to-end simulation acceleration pipeline test passed!")
