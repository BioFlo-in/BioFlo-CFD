"""
BioFlo-CFD: Rigorous Inference Benchmark & Latency Profiling Suite
Measures single-step latency, batch throughput, memory footprint, and CFD speedup factors.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def benchmark_model_inference(
    model: nn.Module,
    param_dim: int = 1,
    batch_sizes: List[int] = [1, 4, 16, 64],
    n_warmup: int = 25,
    n_iterations: int = 100,
    device: Optional[str] = None,
    openfoam_step_latency_sec: float = 16.5,
) -> Dict[str, Union[Dict, str, float]]:
    """
    Measures latency percentiles, throughput, and memory consumption for model inference.
    
    Args:
        model: Calibrated PyTorch model (e.g., NeuralPODOperator3D)
        param_dim: Dimension of parameter input tensor p
        batch_sizes: List of batch sizes to benchmark
        n_warmup: Warmup iterations to trigger JIT / cache warming
        n_iterations: Timing iterations per batch size
        device: Device to run on ('cpu' or 'cuda'); defaults to model parameter device
        openfoam_step_latency_sec: Baseline CFD solve time per time step in seconds
    """
    if device is None:
        target_device = next(model.parameters()).device
    else:
        target_device = torch.device(device)
        model = model.to(target_device)

    model.eval()
    is_cuda = target_device.type == "cuda"
    results_by_batch = {}

    for b in batch_sizes:
        dummy_p = torch.randn(b, param_dim, device=target_device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = model(dummy_p)
            if is_cuda:
                torch.cuda.synchronize()

        # Benchmarking
        latencies = []
        with torch.no_grad():
            for _ in range(n_iterations):
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(dummy_p)
                if is_cuda:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

        lat_arr = np.array(latencies)
        mean_lat = float(np.mean(lat_arr))
        median_lat = float(np.median(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))
        p99_lat = float(np.percentile(lat_arr, 99))
        min_lat = float(np.min(lat_arr))
        max_lat = float(np.max(lat_arr))
        throughput = float((b / (mean_lat / 1000.0))) if mean_lat > 0 else 0.0

        # Memory profile
        mem_info = {}
        if is_cuda:
            mem_info["gpu_allocated_mb"] = float(torch.cuda.memory_allocated(target_device) / (1024**2))
            mem_info["gpu_reserved_mb"] = float(torch.cuda.memory_reserved(target_device) / (1024**2))

        # Speedup vs CFD
        per_snapshot_latency_sec = (mean_lat / 1000.0) / b
        speedup = float(openfoam_step_latency_sec / (per_snapshot_latency_sec + 1e-12))

        results_by_batch[f"batch_{b}"] = {
            "batch_size": b,
            "mean_latency_ms": mean_lat,
            "latency_per_snapshot_ms": float(mean_lat / b),
            "median_latency_ms": median_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "min_latency_ms": min_lat,
            "max_latency_ms": max_lat,
            "throughput_snapshots_per_sec": throughput,
            "openfoam_baseline_sec": openfoam_step_latency_sec,
            "cfd_speedup_factor": speedup,
            **mem_info,
        }

    # Extract single-sample baseline (b=1)
    single_res = results_by_batch.get("batch_1", list(results_by_batch.values())[0])

    return {
        "device": str(target_device),
        "single_snapshot_latency_ms": single_res["mean_latency_ms"],
        "single_snapshot_p95_ms": single_res["p95_latency_ms"],
        "cfd_speedup_factor": single_res["cfd_speedup_factor"],
        "batches": results_by_batch,
    }
