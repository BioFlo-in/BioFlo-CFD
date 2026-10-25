#!/usr/bin/env python3
"""
Demonstration & Verification of HybridCFDNet.
Integrates:
  1. Multi-Scale Neural Operator (Fourier Spectral Conv + Dense Multi-Dilation Conv)
  2. Mesh GraphNet (GNN cell-face flux message passing)
  3. DNN Inter-Layer Linkers with gating highways
  4. Spatiotemporal Transformer Head (Causal temporal + spatial multi-head self-attention)
  5. Spatially-Adaptive Importance-Weighted Physics Loss
"""

from pathlib import Path
import time
import torch
import numpy as np

from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import compute_cfd_importance_weight, PhysicsInformedCFDLoss
from nlss.rollout.engine import RolloutEngine, benchmark_acceleration
from nlss.visualization.plotter import plot_field_comparison


def main():
    print("=" * 80)
    print("  HYBRID CFD-NET ARCHITECTURE VERIFICATION & BENCHMARK")
    print("  [Multi-Scale Operator + MeshGraphNet + DNN Linkers + Spatiotemporal Transformer]")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = Path("outputs/hybrid_cfd_net")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Hybrid Architecture
    print("\n[Step 1/4] Constructing HybridCFDNet...")
    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=48,
        modes1=12,
        modes2=12,
        num_layers=2,
        num_heads=4,
        enforce_boundary=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ HybridCFDNet initialized with {total_params:,} parameters:")
    print("  • Fourier Spectral Operator:   12x12 frequency modes")
    print("  • Multi-Dilation Dense Conv:   Receptive field 3x3 (dilations 1 & 2)")
    print("  • Mesh GraphNet:               4-face flux message passing across grid cells")
    print("  • Inter-Layer DNN Linkers:     MLP with LayerNorm & learnable gating highways")
    print("  • Transformer Head:            4-head spatial + causal temporal self-attention")

    # 2. Load dataset
    print("\n[Step 2/4] Loading cached CFD simulation trajectory...")
    cache_path = Path("outputs/million_run_1000epochs/dataset_cache/million_datapoints_master.pt")
    if not cache_path.exists():
        print("Dataset cache not found, skipping benchmark.")
        return

    data = torch.load(cache_path, weights_only=False).to(device)
    train_x = data[:60]
    train_y = data[1:61]
    print(f"✓ Loaded {train_x.shape[0]} test transitions ({train_x.shape[0]*64*64:,} physical cells)")

    # 3. Compute spatial importance weights and optimize for 10 calibration steps
    print("\n[Step 3/4] Running importance-weighted calibration steps...")
    weights = compute_cfd_importance_weight(train_y, train_x)
    criterion = PhysicsInformedCFDLoss(use_importance_weighting=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    t0 = time.perf_counter()
    model.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        pred = model(train_x[:16])
        loss, metrics = criterion(pred, train_y[:16], input_state=train_x[:16], weight=weights[:16])
        loss.backward()
        optimizer.step()
        if step % 5 == 0 or step == 1:
            print(f"  Step [{step:02d}/10] Active Loss: {metrics['loss_total']:.4e} | Rel L2: {metrics['loss_rel_l2']:.4e}")

    dt_calib = time.perf_counter() - t0
    print(f"✓ 10 Hybrid calibration steps finished in {dt_calib:.2f} s")

    # 4. Multi-Step Rollout Benchmark
    print("\n[Step 4/4] Executing Multi-Step Rollout Benchmark...")
    from nlss.dataset.foam_dataset import FieldNormalizer
    normalizer = FieldNormalizer().fit(train_x.cpu())
    model.eval()
    engine = RolloutEngine(model=model, normalizer=normalizer, device=device)
    
    num_steps = 30
    init_state = train_x[0]
    pred_traj, ai_time = engine.rollout(init_state, num_steps=num_steps)

    bench = benchmark_acceleration(
        rollout_engine=engine,
        ground_truth_trajectory=train_x[:num_steps+1].cpu(),
        cfd_solver_time_sec=2.1,
        num_steps=num_steps,
    )

    print("\n" + "=" * 80)
    print("        HYBRID CFD-NET BENCHMARK RESULTS")
    print("=" * 80)
    print(f"• Rollout Steps:                 {bench['num_steps']}")
    print(f"• AI Rollout Execution Time:     {bench['ai_rollout_time_sec']:.4f} s ({bench['time_per_step_ai_ms']:.2f} ms/step)")
    print(f"• OpenFOAM Numerical Time:       {bench['cfd_solver_time_sec']:.4f} s ({bench['time_per_step_cfd_ms']:.2f} ms/step)")
    print(f"• Speedup Factor:                \033[1;32m{bench['speedup_factor']:.1f}x FASTER\033[0m")
    print(f"• Mean Velocity/Pressure RMSE:   {bench['mean_rmse']:.4e}")
    print("=" * 80 + "\n")

    # Plot field comparison
    plot_field_comparison(
        gt_state=train_x[25].cpu(),
        pred_state=pred_traj[25].cpu(),
        time_val=25 * 0.005,
        save_path=out_dir / "hybrid_cfd_net_rollout.png",
    )
    print(f"📊 Saved field comparison to: {out_dir / 'hybrid_cfd_net_rollout.png'}")


if __name__ == "__main__":
    main()
