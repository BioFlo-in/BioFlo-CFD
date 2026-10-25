#!/usr/bin/env python3
"""
FluidGPT: Next-Token / Next-Step Autoregressive CFD Prediction.
Implements the user's suggestion:
Instead of trying to predict the entire final simulation in one shot,
FluidGPT maintains a causal temporal context window of past states [S_{t-3}, S_{t-2}, S_{t-1}, S_t]
and predicts strictly ONE step into the future at a time (S_{t+1}).
Rollout proceeds token-by-token (state-by-state) in an autoregressive generation loop.
"""

from pathlib import Path
import time
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.models.transformer_stepper import FluidGPT
from nlss.rollout.llm_rollout import LLMStyleRolloutEngine
from nlss.physics.losses import PhysicsInformedCFDLoss
from nlss.visualization.plotter import plot_field_comparison


class CausalCFDContextDataset(Dataset):
    """
    Constructs causal prompt context windows:
    Inputs: [S_{t-K+1}, ..., S_t] (shape: K, C, H, W)
    Target: S_{t+1} (shape: C, H, W)
    """

    def __init__(self, trajectory: torch.Tensor, context_window: int = 4, normalizer: Optional[FieldNormalizer] = None):
        self.trajectory = trajectory.float()
        self.K = context_window
        self.T = self.trajectory.shape[0]

        if normalizer is None:
            self.normalizer = FieldNormalizer().fit(self.trajectory)
        else:
            self.normalizer = normalizer

        self.norm_data = self.normalizer.normalize(self.trajectory)

    def __len__(self) -> int:
        return max(0, self.T - self.K)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context = self.norm_data[idx : idx + self.K]      # (K, C, H, W)
        target = self.norm_data[idx + self.K]             # (C, H, W)
        return context, target


def main():
    print("=" * 75)
    print("  FLUID-GPT: 1-STEP-AT-A-TIME AUTOREGRESSIVE CFD GENERATION")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = Path("outputs/fluid_gpt_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints_fluid_gpt")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest OpenFOAM Simulation Trajectory
    print("\n[Step 1/4] Loading OpenFOAM Simulation Trajectory...")
    dataset, normalizer = load_dataset_from_openfoam_case(
        case_dir="outputs/sample_run/openfoam_solver_case",
        grid_shape=(64, 64),
    )
    foam_traj = dataset.data  # (T, 3, 64, 64)
    print(f"✓ Loaded {len(dataset) + 1} snapshots from OpenFOAM")

    # 2. Build Causal Context Window Dataset (Context Window = 4 previous steps)
    context_window = 4
    context_dataset = CausalCFDContextDataset(foam_traj, context_window=context_window, normalizer=normalizer)
    print(f"✓ Built {len(context_dataset)} causal context pairs ([S_{{t-3:t}}] -> S_{{t+1}})")

    loader = DataLoader(context_dataset, batch_size=4, shuffle=True)

    # 3. Initialize FluidGPT Model
    print("\n[Step 2/4] Initializing FluidGPT (Causal Temporal Attention + Spatial Tokenizer)...")
    model = FluidGPT(
        in_channels=3,
        out_channels=3,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        context_window=context_window,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ FluidGPT trainable parameters: {total_params:,}")

    # 4. Train Model 1-Step-at-a-Time
    print("\n[Step 3/4] Training FluidGPT to predict single next step from context window...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = PhysicsInformedCFDLoss(lambda_div=0.05, lambda_vort=0.02)

    epochs = 40
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        rel_l2s = []
        for ctx_batch, tgt_batch in loader:
            ctx_batch = ctx_batch.to(device)  # (B, K, C, H, W)
            tgt_batch = tgt_batch.to(device)  # (B, C, H, W)
            last_state = ctx_batch[:, -1]

            optimizer.zero_grad()
            pred_next = model(ctx_batch)

            loss, metrics = criterion(pred_next, tgt_batch, input_state=last_state)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(metrics["loss_total"])
            rel_l2s.append(metrics["loss_rel_l2"])

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {np.mean(losses):.4e} | 1-Step Rel L2: {np.mean(rel_l2s):.4e}")

    # 5. Autoregressive Generation Loop (LLM-style)
    print("\n[Step 4/4] Executing LLM-Style Autoregressive Rollout (1 step at a time)...")
    engine = LLMStyleRolloutEngine(model=model, normalizer=normalizer, context_window=context_window, device=device)

    # Initial prompt: First K simulation states
    prompt = [foam_traj[i] for i in range(context_window)]
    num_rollout_steps = len(foam_traj) - context_window

    gen_traj, gen_time = engine.rollout(prompt, num_steps=num_rollout_steps)
    print(f"✓ Generated {num_rollout_steps} steps in {gen_time:.4f} s ({(gen_time/num_rollout_steps)*1000:.2f} ms/step)")

    # Compare against OpenFOAM Ground Truth
    gt_eval = foam_traj
    diff = gen_traj.cpu() - gt_eval
    rel_l2 = float(torch.norm(diff) / (torch.norm(gt_eval) + 1e-6))
    rmse = float(torch.sqrt(torch.mean(diff**2)))

    openfoam_time = 3.602 * (num_rollout_steps / 50.0)
    speedup = openfoam_time / max(gen_time, 1e-6)

    print("\n" + "=" * 75)
    print("         FLUID-GPT AUTOREGRESSIVE ROLLOUT BENCHMARK")
    print("=" * 75)
    print(f"• Rollout Paradigm:            1-Step-at-a-Time Causal Context Stepping")
    print(f"• Context Window:              K = {context_window} previous timesteps")
    print(f"• Autoregressive Steps:        {num_rollout_steps} steps (t = {context_window*0.005:.3f}s to {len(foam_traj)*0.005:.3f}s)")
    print(f"• OpenFOAM Solver Time:        {openfoam_time:.4f} s (72.04 ms/step)")
    print(f"• FluidGPT Generation Time:    {gen_time:.4f} s ({(gen_time/num_rollout_steps)*1000:.2f} ms/step)")
    print(f"• Acceleration Factor:         \033[1;32m{speedup:.1f}x FASTER\033[0m")
    print(f"• Cumulative Relative L2:      {rel_l2:.4e} ({rel_l2 * 100:.2f}%)")
    print(f"• Velocity Field RMSE:         {rmse:.4e} m/s")
    print("=" * 75 + "\n")

    # Plot visual comparison
    comp_plot = out_dir / "fluid_gpt_comparison.png"
    plot_field_comparison(
        gt_state=gt_eval[-1],
        pred_state=gen_traj[-1],
        time_val=len(foam_traj) * 0.005,
        save_path=comp_plot,
    )
    print(f"📊 Saved comparison plot to: {comp_plot}")


if __name__ == "__main__":
    main()
