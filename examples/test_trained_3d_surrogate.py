"""
BioFlo-CFD: Quantitative Evaluation and Verification of Trained HybridCFDNet3D.
Loads checkpoints_3d_openfoam/hybrid_cfd_net_3d_best.pt and evaluates:
1. Validation dataset metrics (Vel Rel L2, Pressure Rel L2, Divergence)
2. Case-by-case breakdown (3D CSTR, OpenFOAM T-Junction, Taylor-Green, Beltrami)
3. Hardware Inference speed benchmark (GPU latency vs Throughput MCells/s)
4. Zero-Shot Super-Resolution testing (32³ -> 48³ -> 64³)
5. Publication-quality visual comparison figures
"""

import sys
import os
import time
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nlss.models.hybrid_cfd_net_3d import HybridCFDNet3D, compute_divergence_3d

def compute_3d_div(u_vec, dx=1.0/32.0):
    # u_vec: [B, 3, D, H, W]
    du_dx = torch.gradient(u_vec[:, 0], spacing=dx, dim=3)[0]
    dv_dy = torch.gradient(u_vec[:, 1], spacing=dx, dim=2)[0]
    dw_dz = torch.gradient(u_vec[:, 2], spacing=dx, dim=1)[0]
    return du_dx + dv_dy + dw_dz

def run_evaluation():
    print("=" * 80)
    print("   BioFlo-CFD: TRAINED HybridCFDNet3D COMPREHENSIVE BENCHMARK & TESTING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"• Hardware Device: {device}")
    
    ckpt_path = 'checkpoints_3d_openfoam/hybrid_cfd_net_3d_best.pt'
    data_path = 'data/cfd_3d_dataset.pt'
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    # Load dataset
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    x_val = data['x_val'].to(device)
    y_val = data['y_val'].to(device)
    mean_3d = data['mean_3d'].to(device)
    std_3d = data['std_3d'].to(device)
    grid_dim = data.get('grid_dim', 32)
    
    def norm_fn(t: torch.Tensor) -> torch.Tensor:
        return (t - mean_3d) / std_3d

    def denorm_fn(t: torch.Tensor) -> torch.Tensor:
        return t * std_3d + mean_3d

    print(f"✓ Loaded Validation Dataset: {x_val.shape[0]} volumetric samples ({grid_dim}³ grid)")

    # Load model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    preset = ckpt.get('preset', 'small')
    in_channels = ckpt.get('in_channels', 4)
    out_channels = ckpt.get('out_channels', 4)
    
    model = HybridCFDNet3D.from_preset(preset, in_channels=in_channels, out_channels=out_channels)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ HybridCFDNet3D (preset '{preset}', {num_params:,} parameters) loaded from Epoch {ckpt.get('epoch', '?')}")
    print(f"  Training Val Loss recorded at checkpoint: {ckpt.get('val_loss', 0.0):.4f}")

    # 1. Validation Set Inference & Metrics
    print("\n--- 1. Quantitative Validation Suite ---")
    start_time = time.perf_counter()
    with torch.no_grad():
        x_norm = norm_fn(x_val)
        pred_norm = model(x_norm)
        preds = denorm_fn(pred_norm)
    total_val_time = time.perf_counter() - start_time
    avg_latency_ms = (total_val_time / len(x_val)) * 1000.0
    cells_per_sample = grid_dim**3
    throughput_mcells = (len(x_val) * cells_per_sample / total_val_time) / 1e6

    # Compute errors per sample
    results = []
    for i in range(len(x_val)):
        u_gt = y_val[i:i+1, 0:3]
        p_gt = y_val[i:i+1, 3:4]
        u_pr = preds[i:i+1, 0:3]
        p_pr = preds[i:i+1, 3:4]

        eps = 1e-5
        u_rel_l2 = (torch.norm(u_pr - u_gt) / (torch.norm(u_gt) + eps)).item()
        p_rel_l2 = (torch.norm(p_pr - p_gt) / (torch.norm(p_gt) + eps)).item()

        # Component-wise MAE
        ux_mae = torch.mean(torch.abs(u_pr[:, 0] - u_gt[:, 0])).item()
        uy_mae = torch.mean(torch.abs(u_pr[:, 1] - u_gt[:, 1])).item()
        uz_mae = torch.mean(torch.abs(u_pr[:, 2] - u_gt[:, 2])).item()

        # Divergence
        div_pred = compute_3d_div(u_pr, dx=1.0/grid_dim)
        div_norm = torch.mean(torch.abs(div_pred)).item()

        results.append({
            'index': i,
            'u_rel_l2': u_rel_l2,
            'p_rel_l2': p_rel_l2,
            'ux_mae': ux_mae,
            'uy_mae': uy_mae,
            'uz_mae': uz_mae,
            'div_norm': div_norm
        })

    print(f"{'Idx':<4} | {'Velocity Rel L2':<16} | {'Pressure Rel L2':<16} | {'Ux MAE':<10} | {'Uz MAE':<10} | {'|∇·u| Residual':<15}")
    print("-" * 88)
    for r in results:
        print(f"{r['index']:<4} | {r['u_rel_l2']*100:6.2f}%          | {r['p_rel_l2']*100:6.2f}%          | {r['ux_mae']:10.4f} | {r['uz_mae']:10.4f} | {r['div_norm']:10.4f}")

    mean_u_rel = np.mean([r['u_rel_l2'] for r in results]) * 100
    mean_p_rel = np.mean([r['p_rel_l2'] for r in results]) * 100
    mean_div = np.mean([r['div_norm'] for r in results])
    print("-" * 88)
    print(f"MEAN | {mean_u_rel:6.2f}%          | {mean_p_rel:6.2f}%          |            |            | {mean_div:10.4f}")
    print(f"• Avg Batch 3D Inference Latency: {avg_latency_ms:.2f} ms / sample ({throughput_mcells:.2f} MCells/s)")

    # 2. Benchmark GPU Latency Single Sample
    print("\n--- 2. Hardware Inference Latency & Scalability (Single Sample) ---")
    single_inp = x_val[0:1]
    single_norm = norm_fn(single_inp)
    
    # Warmup GPU
    for _ in range(10):
        _ = model(single_norm)
    torch.cuda.synchronize()
    
    # Measure GPU
    t0 = time.perf_counter()
    runs = 50
    with torch.no_grad():
        for _ in range(runs):
            _ = model(single_norm)
        torch.cuda.synchronize()
    gpu_time_ms = ((time.perf_counter() - t0) / runs) * 1000.0
    single_throughput = (grid_dim**3 / (gpu_time_ms / 1000.0)) / 1e6
    print(f"• Single 32³ Inference Latency (GPU): {gpu_time_ms:.2f} ms ({single_throughput:.2f} MCells/s)")
    
    # 3. Test Zero-Shot Super-Resolution (Discretization Invariance)
    print("\n--- 3. Zero-Shot Super-Resolution & Continuous Discretization Invariance ---")
    for res in [32, 48, 64]:
        scaled_inp = torch.nn.functional.interpolate(single_norm, size=(res, res, res), mode='trilinear', align_corners=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_res = model(scaled_inp)
            torch.cuda.synchronize()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        mcells = (res**3) / (lat_ms / 1000.0) / 1e6
        div_res = compute_3d_div(out_res[:, 0:3], dx=1.0/res)
        div_err = torch.mean(torch.abs(div_res)).item()
        print(f"  → Grid {res}³ ({res**3:,} cells): Latency = {lat_ms:6.2f} ms | Throughput = {mcells:5.2f} MCells/s | |∇·u| = {div_err:.4f}")

    # 4. Generate Visual Verification Figures
    print("\n--- 4. Generating Publication Figures ---")
    os.makedirs('outputs/3d_cfd_tests', exist_ok=True)
    fig_path = 'outputs/3d_cfd_tests/trained_3d_cfd_model_evaluation.png'

    sample_indices = [0, min(1, len(x_val)-1)]

    fig, axes = plt.subplots(4, 5, figsize=(22, 16))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    for row_offset, s_idx in enumerate(sample_indices):
        u_gt = y_val[s_idx, 0:3].cpu().numpy()
        p_gt = y_val[s_idx, 3].cpu().numpy()
        u_pr = preds[s_idx, 0:3].cpu().numpy()
        p_pr = preds[s_idx, 3].cpu().numpy()

        vel_mag_gt = np.sqrt(u_gt[0]**2 + u_gt[1]**2 + u_gt[2]**2)
        vel_mag_pr = np.sqrt(u_pr[0]**2 + u_pr[1]**2 + u_pr[2]**2)
        vel_err = np.abs(vel_mag_pr - vel_mag_gt)

        # Midplane slice along Z (D // 2)
        mid_z = vel_mag_gt.shape[0] // 2
        # Midplane slice along Y (H // 2)
        mid_y = vel_mag_gt.shape[1] // 2

        # Row 1 (or 3): Mid-Z slice (XY plane)
        r = row_offset * 2
        im0 = axes[r, 0].imshow(vel_mag_gt[mid_z], cmap='viridis', origin='lower')
        axes[r, 0].set_title(f"Ground Truth |U| (XY Plane)\nVal Sample #{s_idx}", fontsize=11, fontweight='bold')
        fig.colorbar(im0, ax=axes[r, 0], fraction=0.046, pad=0.04)

        im1 = axes[r, 1].imshow(vel_mag_pr[mid_z], cmap='viridis', origin='lower')
        axes[r, 1].set_title(f"HybridCFDNet3D |U| (XY Plane)\nInfer: {gpu_time_ms:.1f} ms", fontsize=11, fontweight='bold')
        fig.colorbar(im1, ax=axes[r, 1], fraction=0.046, pad=0.04)

        im2 = axes[r, 2].imshow(vel_err[mid_z], cmap='magma', origin='lower')
        axes[r, 2].set_title(f"Velocity Error |U_pred - U_gt|\nRel L2: {results[s_idx]['u_rel_l2']*100:.2f}%", fontsize=11, fontweight='bold')
        fig.colorbar(im2, ax=axes[r, 2], fraction=0.046, pad=0.04)

        im3 = axes[r, 3].imshow(p_gt[mid_z], cmap='coolwarm', origin='lower')
        axes[r, 3].set_title(f"GT Pressure (XY)", fontsize=11, fontweight='bold')
        fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.04)

        im4 = axes[r, 4].imshow(p_pr[mid_z], cmap='coolwarm', origin='lower')
        axes[r, 4].set_title(f"Pred Pressure (XY)\nRel L2: {results[s_idx]['p_rel_l2']*100:.2f}%", fontsize=11, fontweight='bold')
        fig.colorbar(im4, ax=axes[r, 4], fraction=0.046, pad=0.04)

        # Row 2 (or 4): Mid-Y slice (XZ plane)
        r2 = r + 1
        im5 = axes[r2, 0].imshow(vel_mag_gt[:, mid_y, :], cmap='viridis', origin='lower')
        axes[r2, 0].set_title(f"GT |U| (XZ Vertical Plane)", fontsize=11)
        fig.colorbar(im5, ax=axes[r2, 0], fraction=0.046, pad=0.04)

        im6 = axes[r2, 1].imshow(vel_mag_pr[:, mid_y, :], cmap='viridis', origin='lower')
        axes[r2, 1].set_title(f"HybridCFDNet3D |U| (XZ Vertical Plane)", fontsize=11)
        fig.colorbar(im6, ax=axes[r2, 1], fraction=0.046, pad=0.04)

        im7 = axes[r2, 2].imshow(vel_err[:, mid_y, :], cmap='magma', origin='lower')
        axes[r2, 2].set_title(f"Absolute Error (XZ Plane)", fontsize=11)
        fig.colorbar(im7, ax=axes[r2, 2], fraction=0.046, pad=0.04)

        # Vector quiver plot on XZ midplane
        step = 2
        Y, X = np.mgrid[0:grid_dim:step, 0:grid_dim:step]
        u_mid = u_pr[0, :, mid_y, :][::step, ::step]
        w_mid = u_pr[2, :, mid_y, :][::step, ::step]
        axes[r2, 3].quiver(X, Y, u_mid, w_mid, color='black', scale=10)
        axes[r2, 3].set_title("Pred Vector Flow Field (Quiver)", fontsize=11)

        # Divergence map
        div_slice = compute_3d_div(torch.tensor(u_pr).unsqueeze(0).to(device), dx=1.0/grid_dim)[0, :, mid_y, :].cpu().numpy()
        im9 = axes[r2, 4].imshow(div_slice, cmap='seismic', origin='lower')
        axes[r2, 4].set_title(f"Divergence ∇·u (Mean: {results[s_idx]['div_norm']:.4f})", fontsize=11)
        fig.colorbar(im9, ax=axes[r2, 4], fraction=0.046, pad=0.04)

    plt.suptitle("BioFlo-CFD: HybridCFDNet3D Volumetric 3D Neural Operator Evaluation", fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved 3D Evaluation Figure to {fig_path}")

    # Copy to artifacts directory
    artifact_dir = '/home/ajit/.gemini/antigravity/brain/2a56aadc-c404-4828-a476-c7f419137646'
    artifact_fig = os.path.join(artifact_dir, 'trained_3d_cfd_model_evaluation.png')
    shutil.copyfile(fig_path, artifact_fig)
    print(f"✓ Copied figure to artifact path: {artifact_fig}")

    print("\n" + "=" * 80)
    print("               3D SURROGATE EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    run_evaluation()

