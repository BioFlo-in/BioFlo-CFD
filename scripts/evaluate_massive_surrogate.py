#!/usr/bin/env python3
"""
Evaluates the trained massive HybridCFDNet surrogate against ground-truth OpenFOAM CFD
across 4 distinct regimes: Canonical Cavity, High Speed, Surface Variation, and Bioreactor.
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from nlss.dataset.foam_dataset import load_dataset_from_openfoam_case, FieldNormalizer
from nlss.models.hybrid_cfd_net import HybridCFDNet
from nlss.physics.losses import compute_divergence_2d


WORKSPACE = Path(__file__).resolve().parent.parent
CKPT_PATH = WORKSPACE / "checkpoints_massive_openfoam/hybrid_cfd_net_best.pt"
OUTPUT_DIR = WORKSPACE / "checkpoints_massive_openfoam"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating model on {device}")

    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location=device)
    model = HybridCFDNet(
        in_channels=3,
        out_channels=3,
        hidden_dim=32,
        modes1=12,
        modes2=12,
        num_layers=2,
        enforce_boundary=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = ckpt["normalizer"]

    test_cases = [
        ("Canonical Cavity (Re=100)", WORKSPACE / "external_openfoam_sources/official_tutorials/cavity"),
        ("High Speed Flow (Re=2500)", WORKSPACE / "openfoam_cases/speed_variations_suite/speed_high_re2500"),
        ("Dual Moving Walls (Surface)", WORKSPACE / "openfoam_cases/surface_variations_suite/surface_counter_moving_horizontal"),
        ("Stirred Bioreactor (Tip 2.0)", WORKSPACE / "openfoam_cases/bioreactor_suite/bioreactor_stirred_tank_tip_2.00"),
    ]

    fig, axes = plt.subplots(len(test_cases), 3, figsize=(15, 4 * len(test_cases)))
    results = []

    for row_idx, (case_name, case_dir) in enumerate(test_cases):
        if not case_dir.exists():
            continue
        ds, _ = load_dataset_from_openfoam_case(case_dir, grid_shape=(64, 64), rollout_steps=1)
        raw_traj = ds.traj_list[0]  # (T, 3, 64, 64)
        if len(raw_traj) < 2:
            continue

        s0 = raw_traj[0:1].to(device)
        s_true = raw_traj[-1].numpy()  # (3, 64, 64)

        # Autoregressive rollout to final state
        s_curr = s0
        t0 = time.time()
        with torch.no_grad():
            for _ in range(len(raw_traj) - 1):
                s_norm = normalizer.normalize(s_curr)
                pred_norm = model(s_norm)
                s_next = normalizer.denormalize(pred_norm)
                s_curr = s_next
        dt_ms = (time.time() - t0) * 1000.0 / (len(raw_traj) - 1)

        s_pred = s_curr[0].cpu().numpy()

        u_true_mag = np.hypot(s_true[0], s_true[1])
        u_pred_mag = np.hypot(s_pred[0], s_pred[1])
        u_err = np.abs(u_pred_mag - u_true_mag)

        rel_l2 = np.linalg.norm(u_pred_mag - u_true_mag) / (np.linalg.norm(u_true_mag) + 1e-6)

        results.append({
            "case": case_name,
            "rel_l2": rel_l2,
            "latency_ms": dt_ms,
        })

        # Plot Ground Truth
        im0 = axes[row_idx, 0].imshow(u_true_mag, cmap="viridis", origin="lower")
        axes[row_idx, 0].set_title(f"{case_name}\nOpenFOAM Ground Truth", fontweight="bold", fontsize=10)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Plot Surrogate Prediction
        im1 = axes[row_idx, 1].imshow(u_pred_mag, cmap="viridis", origin="lower")
        axes[row_idx, 1].set_title(f"HybridCFDNet Surrogate Prediction\n(Latency: {dt_ms:.2f} ms/step)", fontweight="bold", fontsize=10)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Plot Absolute Error
        im2 = axes[row_idx, 2].imshow(u_err, cmap="magma", origin="lower")
        axes[row_idx, 2].set_title(f"Absolute Error |U_pred - U_true|\nRelative L2: {rel_l2*100:.2f}%", fontweight="bold", fontsize=10)
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    gallery_path = OUTPUT_DIR / "generalist_surrogate_evaluation_gallery.png"
    plt.savefig(gallery_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved evaluation gallery to {gallery_path}")

    print("\n" + "=" * 60)
    print("EVALUATION BENCHMARK SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"• {r['case']:<32} : Rel L2 Error = {r['rel_l2']*100:5.2f}% | Latency = {r['latency_ms']:.2f} ms/step")
    print("=" * 60)


if __name__ == "__main__":
    main()
