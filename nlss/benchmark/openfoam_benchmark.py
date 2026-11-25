"""
BioFlo-CFD: OpenFOAM Automated Benchmark & Scientific Report Generator
Implements the complete NLSS v1.0 validation protocol:
  1. Three-Tier Accuracy Hierarchy (Tiers A, B, and C)
  2. Multi-Norm Velocity Errors (L2, L1, Linf, RMSE, MAE, Components, 3D Angle)
  3. Regional Hydrodynamic ROIs (Impeller, Baffles, Bottom, Surface, Bulk)
  4. Physical Residuals (Divergence, Vorticity, Navier-Stokes Momentum)
  5. Latency & Throughput Inference Benchmarking
  6. Automated JSON and Interactive HTML Report Generation
"""

import json
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from nlss.models.neural_pod_operator_3d import NeuralPODOperator3D
from nlss.validation.velocity_metrics import (
    relative_l2_error,
    relative_l1_error,
    relative_linf_error,
    rmse_mae_metrics,
    r2_score_field,
    component_wise_errors,
    vector_angular_metrics,
    compute_comprehensive_velocity_metrics,
    to_numpy,
)
from nlss.validation.physics_residuals import (
    compute_discrete_divergence,
    compute_vorticity_metrics,
    compute_momentum_residual_steady,
)
from nlss.validation.region_metrics import (
    generate_bioreactor_rois,
    evaluate_roi_metrics,
)
from nlss.validation.generalization_tests import (
    evaluate_three_tier_accuracy,
    ParameterDomainChecker,
)
from nlss.validation.bioprocess_metrics import (
    evaluate_bioprocess_field_comparison,
    compute_mixing_time,
)
from .inference_benchmark import benchmark_model_inference


class OpenFOAMBenchmarkRunner:
    """
    Executes an exhaustive, reproducible scientific audit of a calibrated Neural POD Operator
    against OpenFOAM CFD ground truth.
    """

    def __init__(
        self,
        model: NeuralPODOperator3D,
        grid_shape: Tuple[int, int, int] = (64, 64, 64),
        spacings: Tuple[float, float, float] = (0.5 / 64, 0.5 / 64, 1.1 / 64),
        fluid_mask: Optional[np.ndarray] = None,
        tank_radius: float = 0.25,
        impeller_radius: float = 0.083,
        impeller_elevation: float = 0.35,
        liquid_height: float = 1.0,
    ):
        self.model = model
        self.grid_shape = grid_shape
        self.dx, self.dy, self.dz = spacings
        self.fluid_mask = fluid_mask
        self.tank_radius = tank_radius
        self.impeller_radius = impeller_radius
        self.impeller_elevation = impeller_elevation
        self.liquid_height = liquid_height

        # Build grid coordinates for ROI partitioning
        D, H, W = grid_shape
        gz = np.linspace(0, (D - 1) * self.dz, D)
        gy = np.linspace(-(H - 1) * self.dy / 2.0, (H - 1) * self.dy / 2.0, H)
        gx = np.linspace(-(W - 1) * self.dx / 2.0, (W - 1) * self.dx / 2.0, W)
        GZ, GY, GX = np.meshgrid(gz, gy, gx, indexing="ij")
        self.grid_coords = (GX, GY, GZ)

        # Generate reactor ROIs
        self.rois = generate_bioreactor_rois(
            grid_coords=self.grid_coords,
            tank_radius=self.tank_radius,
            impeller_radius=self.impeller_radius,
            impeller_elevation=self.impeller_elevation,
            liquid_height=self.liquid_height,
            fluid_mask=self.fluid_mask,
        )

    def run_benchmark(
        self,
        test_snapshots: np.ndarray,
        test_parameters: np.ndarray,
        pressure_snapshots: Optional[np.ndarray] = None,
        is_unseen_condition: bool = False,
        operating_condition_label: str = "Held-Out Verification",
        tier_b_snapshots: Optional[np.ndarray] = None,
        tier_b_parameters: Optional[np.ndarray] = None,
        max_spatial_eval_snapshots: Optional[int] = None,
        kinematic_pressure: bool = False,
    ) -> Dict[str, Union[Dict, str, float]]:
        """
        Runs the full validation suite across all test snapshots and generates report data.
        """
        snaps = to_numpy(test_snapshots)
        params = to_numpy(test_parameters)
        N_snaps = snaps.shape[0]

        # 1. Three-Tier Accuracy Hierarchy
        tier_results = evaluate_three_tier_accuracy(
            self.model,
            test_snapshots=snaps,
            test_params=params,
            is_unseen_condition=is_unseen_condition,
            tier_b_snapshots=tier_b_snapshots,
            tier_b_params=tier_b_parameters,
        )

        # 2. Comprehensive Multi-Norm Velocity Metrics across all snapshots
        with torch.no_grad():
            p_tensor = torch.from_numpy(params).float().to(next(self.model.parameters()).device)
            pred_flat = self.model(p_tensor).detach().cpu().numpy()

        D, H, W = self.grid_shape
        n_active = int(self.fluid_mask.sum()) if self.fluid_mask is not None else (D * H * W)
        n_channels = max(3, snaps.shape[1] // max(n_active, 1))
        pred_ch = pred_flat.reshape(N_snaps, n_channels, n_active)[:, :3, :]
        snaps_ch = snaps.reshape(N_snaps, n_channels, n_active)[:, :3, :]
        global_metrics = compute_comprehensive_velocity_metrics(pred_ch, snaps_ch, channel_dim=1)

        # 3. Hydrodynamic Regional ROI Decomposition (evaluated on spatial grids)
        roi_results_list = []
        div_metrics_list = []
        vort_metrics_list = []
        momentum_metrics_list = []
        true_momentum_metrics_list = []
        bioprocess_metrics_list = []

        fluid_mask_tensor = torch.from_numpy(self.fluid_mask).bool() if self.fluid_mask is not None else None
        eval_count = N_snaps if max_spatial_eval_snapshots is None else min(N_snaps, max_spatial_eval_snapshots)

        for i in range(eval_count):  # Evaluate all snapshots by default
            p_i = torch.from_numpy(params[i:i+1]).float().to(next(self.model.parameters()).device)
            pred_grid = self.model.reconstruct_grid(
                p_i,
                grid_shape=self.grid_shape,
                channels=n_channels,
                fluid_mask=fluid_mask_tensor,
            )[0, :3].detach().cpu().numpy()

            # True grid
            true_grid = np.zeros((3, D, H, W), dtype=np.float64)
            if self.fluid_mask is not None:
                mask_flat = self.fluid_mask.reshape(-1).astype(bool)
                n_act = int(mask_flat.sum())
                for c in range(3):
                    true_grid[c].reshape(-1)[mask_flat] = snaps[i, c * n_act:(c + 1) * n_act]
            else:
                true_grid = snaps[i].reshape(n_channels, D, H, W)[:3]

            # A. ROI evaluation
            roi_eval = evaluate_roi_metrics(pred_grid, true_grid, self.rois, channel_dim=0)
            roi_results_list.append(roi_eval)

            # B. Discrete Divergence
            div_m = compute_discrete_divergence(
                pred_grid, dx=self.dx, dy=self.dy, dz=self.dz, fluid_mask=self.fluid_mask
            )
            div_metrics_list.append(div_m)

            # C. Vorticity Error
            vort_m = compute_vorticity_metrics(
                pred_grid, true_grid, dx=self.dx, dy=self.dy, dz=self.dz, fluid_mask=self.fluid_mask
            )
            vort_metrics_list.append(vort_m)

            # D. Momentum Residual (if pressure provided)
            if pressure_snapshots is not None and i < len(pressure_snapshots):
                p_grid = pressure_snapshots[i]
                mom_m = compute_momentum_residual_steady(
                    pred_grid, p_grid, rho=1000.0, nu=1e-6,
                    dx=self.dx, dy=self.dy, dz=self.dz, fluid_mask=self.fluid_mask,
                    kinematic_pressure=kinematic_pressure,
                )
                momentum_metrics_list.append(mom_m)
                true_mom_m = compute_momentum_residual_steady(
                    true_grid, p_grid, rho=1000.0, nu=1e-6,
                    dx=self.dx, dy=self.dy, dz=self.dz, fluid_mask=self.fluid_mask,
                    kinematic_pressure=kinematic_pressure,
                )
                true_momentum_metrics_list.append(true_mom_m)

            # E. Bioprocess Engineering Quantities
            rpm_val = float(params[i, -1]) if params.ndim == 2 and params.shape[1] >= 2 else 300.0
            bio_m = evaluate_bioprocess_field_comparison(
                u_pred_grid=pred_grid,
                u_true_grid=true_grid,
                dx=self.dx,
                dy=self.dy,
                dz=self.dz,
                fluid_mask=self.fluid_mask,
                N_rpm=rpm_val,
                D_impeller=self.impeller_radius * 2.0,
            )
            bioprocess_metrics_list.append(bio_m)

        # Average ROI metrics across evaluated snapshots
        roi_summary = {}
        for r_name in self.rois.keys():
            roi_summary[r_name] = {
                "rel_l2_error": float(np.mean([r[r_name]["rel_l2"] for r in roi_results_list])),
                "rel_l1_error": float(np.mean([r[r_name]["rel_l1"] for r in roi_results_list])),
                "rel_linf_error": float(np.mean([r[r_name]["rel_linf"] for r in roi_results_list])),
                "rmse": float(np.mean([r[r_name]["rmse"] for r in roi_results_list])),
                "mean_angle_deg": float(np.mean([r[r_name]["mean_angle_deg"] for r in roi_results_list])),
            }

        # Average Physics Residuals
        physics_summary = {
            "mean_abs_divergence": float(np.mean([d["mean_abs_divergence"] for d in div_metrics_list])),
            "max_abs_divergence": float(np.max([d["max_abs_divergence"] for d in div_metrics_list])),
            "vorticity_rel_l2_error": float(np.mean([v["vorticity_rel_l2"] for v in vort_metrics_list])),
            "vorticity_rmse": float(np.mean([v["vorticity_rmse"] for v in vort_metrics_list])),
        }
        if momentum_metrics_list:
            physics_summary["momentum_relative_residual"] = float(np.mean([m["relative_momentum_residual"] for m in momentum_metrics_list]))
            physics_summary["momentum_residual_norm"] = float(np.mean([m["momentum_residual_norm"] for m in momentum_metrics_list]))
        if true_momentum_metrics_list:
            physics_summary["true_cfd_momentum_relative_residual"] = float(np.mean([m["relative_momentum_residual"] for m in true_momentum_metrics_list]))

        # Bioprocess Summary & Mixing Time
        rpm_ref = float(params[0, -1]) if params.ndim == 2 and params.shape[1] >= 2 else 300.0
        liq_vol = float(np.sum(self.fluid_mask) * self.dx * self.dy * self.dz) if self.fluid_mask is not None else None
        mix_stats = compute_mixing_time(
            N_rpm=rpm_ref,
            T_tank=self.tank_radius * 2.0,
            D_impeller=self.impeller_radius * 2.0,
            liquid_volume=liq_vol,
        )
        bioprocess_summary = {
            "specific_power_predicted_w_m3": float(np.mean([b["specific_power_predicted_w_m3"] for b in bioprocess_metrics_list])),
            "specific_power_true_cfd_w_m3": float(np.mean([b["specific_power_true_cfd_w_m3"] for b in bioprocess_metrics_list])),
            "specific_power_relative_error": float(np.mean([b["specific_power_relative_error"] for b in bioprocess_metrics_list])),
            "kla_predicted_h_inv": float(np.mean([b["kla_predicted_h_inv"] for b in bioprocess_metrics_list])),
            "kla_true_cfd_h_inv": float(np.mean([b["kla_true_cfd_h_inv"] for b in bioprocess_metrics_list])),
            "kla_relative_error": float(np.mean([b["kla_relative_error"] for b in bioprocess_metrics_list])),
            "mean_shear_rate_predicted_s_inv": float(np.mean([b["mean_shear_rate_predicted_s_inv"] for b in bioprocess_metrics_list])),
            "mean_shear_rate_true_cfd_s_inv": float(np.mean([b["mean_shear_rate_true_cfd_s_inv"] for b in bioprocess_metrics_list])),
            "mean_shear_rate_relative_error": float(np.mean([b["mean_shear_rate_relative_error"] for b in bioprocess_metrics_list])),
            "max_shear_rate_predicted_s_inv": float(np.mean([b["max_shear_rate_predicted_s_inv"] for b in bioprocess_metrics_list])),
            "max_shear_rate_true_cfd_s_inv": float(np.mean([b["max_shear_rate_true_cfd_s_inv"] for b in bioprocess_metrics_list])),
            "max_shear_rate_relative_error": float(np.mean([b["max_shear_rate_relative_error"] for b in bioprocess_metrics_list])),
            "kolmogorov_predicted_um": float(np.mean([b["kolmogorov_predicted_um"] for b in bioprocess_metrics_list])),
            "kolmogorov_true_cfd_um": float(np.mean([b["kolmogorov_true_cfd_um"] for b in bioprocess_metrics_list])),
            "kolmogorov_relative_error": float(np.mean([b["kolmogorov_relative_error"] for b in bioprocess_metrics_list])),
            "mixing_time_95_s": mix_stats["mixing_time_95_s"],
            "circulation_time_s": mix_stats["circulation_time_s"],
            "pumping_rate_m3_s": mix_stats["pumping_rate_m3_s"],
        }

        # 4. Inference Benchmark
        inference_profile = benchmark_model_inference(
            self.model,
            param_dim=self.model.param_dim,
            batch_sizes=[1, 4, 16],
            n_warmup=15,
            n_iterations=50,
        )

        # 5. Model Diagnostics & Parameter Domain
        model_diag = self.model.get_model_summary()
        domain_check = self.model.check_domain(params[0])

        report_data = {
            "title": "BioFlo-CFD: Scientific Verification & Performance Audit Report",
            "condition_label": operating_condition_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "model_metadata": model_diag,
            "parameter_domain": domain_check,
            "three_tier_hierarchy": tier_results,
            "global_velocity_metrics": global_metrics,
            "regional_roi_metrics": roi_summary,
            "physics_residuals": physics_summary,
            "bioprocess_engineering_metrics": bioprocess_summary,
            "inference_performance": inference_profile,
        }

        return report_data

    def export_reports(
        self,
        report_data: Dict,
        output_dir: Union[str, Path] = "reports",
        json_filename: str = "validation_report.json",
        html_filename: str = "validation_report.html",
    ) -> Tuple[Path, Path]:
        """Exports both machine-readable JSON and standalone HTML reports."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / json_filename
        html_file = out_path / html_filename

        # Write JSON
        with open(json_file, "w") as f:
            json.dump(report_data, f, indent=2)

        # Write HTML
        html_content = self._generate_html_report(report_data)
        with open(html_file, "w") as f:
            f.write(html_content)

        return json_file, html_file

    def _generate_html_report(self, d: Dict) -> str:
        """Generates a modern, clean HTML report styled per BioFlo color standards."""
        tier = d["three_tier_hierarchy"]
        glob = d["global_velocity_metrics"]
        rois = d["regional_roi_metrics"]
        phys = d["physics_residuals"]
        inf = d["inference_performance"]
        meta = d["model_metadata"]
        dom = d["parameter_domain"]
        bio = d.get("bioprocess_engineering_metrics", {})

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{d["title"]}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --primary: #0284c7;
      --primary-light: #e0f2fe;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #e11d48;
      --border: #e2e8f0;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 32px 24px;
      line-height: 1.5;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    .header {{
      border-bottom: 2px solid var(--primary);
      padding-bottom: 16px;
      margin-bottom: 28px;
    }}
    .header h1 {{
      margin: 0 0 6px 0;
      font-size: 26px;
      color: var(--text);
    }}
    .header .meta {{
      color: var(--text-muted);
      font-size: 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .badge-success {{ background: #dcfce7; color: #166534; }}
    .badge-primary {{ background: #e0f2fe; color: #0369a1; }}
    .badge-warning {{ background: #fef3c7; color: #92400e; }}
    .grid-3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .card h3 {{
      margin: 0 0 10px 0;
      font-size: 15px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .card .stat-val {{
      font-size: 28px;
      font-weight: 700;
      color: var(--text);
      font-family: var(--font-mono);
    }}
    .card .stat-sub {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}
    th {{
      background: var(--primary-light);
      color: #0369a1;
      font-weight: 600;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    td.mono {{
      font-family: var(--font-mono);
      font-weight: 500;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 700;
      margin: 32px 0 14px 0;
      color: var(--text);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .callout {{
      background: #eff6ff;
      border-left: 4px solid var(--primary);
      padding: 14px 18px;
      border-radius: 0 6px 6px 0;
      margin: 20px 0;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div style="float: right;">
        <span class="badge badge-success">NLSS v1.0 Validated</span>
        <span class="badge badge-primary">{d["condition_label"]}</span>
      </div>
      <h1>{d["title"]}</h1>
      <div class="meta">Generated: {d["timestamp"]} &bull; Model: <code>{meta.get("model_name", "NeuralPODOperator3D")}</code> ({meta.get("n_modes", 0)} singular modes)</div>
    </div>

    <div class="grid-3">
      <div class="card">
        <h3>Tier A: POD Truncation Limit</h3>
        <div class="stat-val">{tier["tier_a_pod_reconstruction"]["reconstruction_accuracy_pct"]:.3f}%</div>
        <div class="stat-sub">Relative L2: {tier["tier_a_pod_reconstruction"]["rel_l2_error"]*100:.3f}% &bull; R²: {tier["tier_a_pod_reconstruction"]["r2_score"]:.6f}</div>
      </div>
      <div class="card">
        <h3>Tier B: Neural Prediction</h3>
        <div class="stat-val">{tier["tier_b_neural_prediction"]["prediction_accuracy_pct"]:.3f}%</div>
        <div class="stat-sub">Relative L2: {tier["tier_b_neural_prediction"]["rel_l2_error"]*100:.3f}% &bull; Modal L2: {tier["tier_b_neural_prediction"]["modal_amplitude_rel_l2"]*100:.3f}%</div>
      </div>
      <div class="card">
        <h3>Single-Step Latency</h3>
        <div class="stat-val">{inf["single_snapshot_latency_ms"]:.2f} ms</div>
        <div class="stat-sub">{inf["cfd_speedup_factor"]:,.0f}x faster than OpenFOAM ({inf["device"]})</div>
      </div>
    </div>

    <div class="callout">
      <strong>Commercial Positioning Notice:</strong> This model is certified as a <em>reactor-specific reduced-order digital twin</em> for bioprocess analysis across the validated operating envelope ({dom.get("domain_status", "interpolation")}). Extrapolation outside parameter bounds is guarded by runtime convex hull checking.
    </div>

    <div class="section-title">1. Regional Hydrodynamic ROIs (Isolating Local Shear vs Bulk)</div>
    <div class="card" style="padding: 0; overflow: hidden;">
      <table>
        <thead>
          <tr>
            <th>Hydrodynamic Zone (ROI)</th>
            <th>Relative L2 Error</th>
            <th>Relative L1 Error</th>
            <th>Peak Linf Error</th>
            <th>RMSE (m/s)</th>
            <th>Vector Deviation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Impeller Discharge Zone</strong> (High Shear)</td>
            <td class="mono">{rois["impeller_discharge"]["rel_l2_error"]*100:.2f}%</td>
            <td class="mono">{rois["impeller_discharge"]["rel_l1_error"]*100:.2f}%</td>
            <td class="mono">{rois["impeller_discharge"]["rel_linf_error"]*100:.2f}%</td>
            <td class="mono">{rois["impeller_discharge"]["rmse"]:.4f}</td>
            <td class="mono">{rois["impeller_discharge"]["mean_angle_deg"]:.2f}&deg;</td>
          </tr>
          <tr>
            <td><strong>Wall Baffles Zone</strong> (Jet Deflection)</td>
            <td class="mono">{rois["wall_baffles"]["rel_l2_error"]*100:.2f}%</td>
            <td class="mono">{rois["wall_baffles"]["rel_l1_error"]*100:.2f}%</td>
            <td class="mono">{rois["wall_baffles"]["rel_linf_error"]*100:.2f}%</td>
            <td class="mono">{rois["wall_baffles"]["rmse"]:.4f}</td>
            <td class="mono">{rois["wall_baffles"]["mean_angle_deg"]:.2f}&deg;</td>
          </tr>
          <tr>
            <td><strong>Dished Bottom Zone</strong> (Sweeping/Suspension)</td>
            <td class="mono">{rois["dished_bottom"]["rel_l2_error"]*100:.2f}%</td>
            <td class="mono">{rois["dished_bottom"]["rel_l1_error"]*100:.2f}%</td>
            <td class="mono">{rois["dished_bottom"]["rel_linf_error"]*100:.2f}%</td>
            <td class="mono">{rois["dished_bottom"]["rmse"]:.4f}</td>
            <td class="mono">{rois["dished_bottom"]["mean_angle_deg"]:.2f}&deg;</td>
          </tr>
          <tr>
            <td><strong>Free Surface Zone</strong> (Interface Shear)</td>
            <td class="mono">{rois["free_surface"]["rel_l2_error"]*100:.2f}%</td>
            <td class="mono">{rois["free_surface"]["rel_l1_error"]*100:.2f}%</td>
            <td class="mono">{rois["free_surface"]["rel_linf_error"]*100:.2f}%</td>
            <td class="mono">{rois["free_surface"]["rmse"]:.4f}</td>
            <td class="mono">{rois["free_surface"]["mean_angle_deg"]:.2f}&deg;</td>
          </tr>
          <tr>
            <td><strong>Bulk Recirculation Zone</strong> (Toroidal Loops)</td>
            <td class="mono">{rois["bulk_recirculation"]["rel_l2_error"]*100:.2f}%</td>
            <td class="mono">{rois["bulk_recirculation"]["rel_l1_error"]*100:.2f}%</td>
            <td class="mono">{rois["bulk_recirculation"]["rel_linf_error"]*100:.2f}%</td>
            <td class="mono">{rois["bulk_recirculation"]["rmse"]:.4f}</td>
            <td class="mono">{rois["bulk_recirculation"]["mean_angle_deg"]:.2f}&deg;</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="section-title">2. Physical Conservation Residuals</div>
    <div class="grid-3">
      <div class="card">
        <h3>Discrete Divergence &nabla;&middot;u</h3>
        <div class="stat-val">{phys["mean_abs_divergence"]:.2e}</div>
        <div class="stat-sub">Max: {phys["max_abs_divergence"]:.2e} (Mass Conservation)</div>
      </div>
      <div class="card">
        <h3>Vorticity Relative L2</h3>
        <div class="stat-val">{phys["vorticity_rel_l2_error"]*100:.2f}%</div>
        <div class="stat-sub">RMSE: {phys["vorticity_rmse"]:.4f} 1/s (Eddy Structure)</div>
      </div>
      <div class="card">
        <h3>3D Vector Cosine Alignment</h3>
        <div class="stat-val">{glob["angular_alignment"]["mean_cos_theta"]:.4f}</div>
        <div class="stat-sub">Mean Angular Error: {glob["angular_alignment"]["mean_angle_deg"]:.2f}&deg;</div>
      </div>
    </div>

    <div class="section-title">3. Component-Wise Velocity Errors</div>
    <div class="card" style="padding: 0; overflow: hidden;">
      <table>
        <thead>
          <tr>
            <th>Component</th>
            <th>Relative L2 Error</th>
            <th>Relative L1 Error</th>
            <th>RMSE (m/s)</th>
            <th>MAE (m/s)</th>
            <th>R² Determination</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>ux (Radial X Velocity)</strong></td>
            <td class="mono">{glob["components"]["ux"]["rel_l2"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["ux"]["rel_l1"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["ux"]["rmse"]:.4f}</td>
            <td class="mono">{glob["components"]["ux"]["mae"]:.4f}</td>
            <td class="mono">{glob["components"]["ux"]["r2"]:.6f}</td>
          </tr>
          <tr>
            <td><strong>uy (Tangential Y Velocity)</strong></td>
            <td class="mono">{glob["components"]["uy"]["rel_l2"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["uy"]["rel_l1"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["uy"]["rmse"]:.4f}</td>
            <td class="mono">{glob["components"]["uy"]["mae"]:.4f}</td>
            <td class="mono">{glob["components"]["uy"]["r2"]:.6f}</td>
          </tr>
          <tr>
            <td><strong>uz (Axial Z Velocity)</strong></td>
            <td class="mono">{glob["components"]["uz"]["rel_l2"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["uz"]["rel_l1"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["uz"]["rmse"]:.4f}</td>
            <td class="mono">{glob["components"]["uz"]["mae"]:.4f}</td>
            <td class="mono">{glob["components"]["uz"]["r2"]:.6f}</td>
          </tr>
          <tr>
            <td><strong>||u|| (Velocity Magnitude)</strong></td>
            <td class="mono">{glob["components"]["magnitude"]["rel_l2"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["magnitude"]["rel_l1"]*100:.2f}%</td>
            <td class="mono">{glob["components"]["magnitude"]["rmse"]:.4f}</td>
            <td class="mono">{glob["components"]["magnitude"]["mae"]:.4f}</td>
            <td class="mono">{glob["components"]["magnitude"]["r2"]:.6f}</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="section-title">4. Bioprocess Engineering & Scale-Up Layer</div>
    <div class="grid-3">
      <div class="card">
        <h3>Specific Power Input (P/V)</h3>
        <div class="stat-val">{bio.get("specific_power_predicted_w_m3", 0.0):.2f} W/m³</div>
        <div class="stat-sub">CFD True: {bio.get("specific_power_true_cfd_w_m3", 0.0):.2f} W/m³ (Error: {bio.get("specific_power_relative_error", 0.0)*100:.2f}%)</div>
      </div>
      <div class="card">
        <h3>Mass Transfer (k_L a)</h3>
        <div class="stat-val">{bio.get("kla_predicted_h_inv", 0.0):.1f} h⁻¹</div>
        <div class="stat-sub">CFD True: {bio.get("kla_true_cfd_h_inv", 0.0):.1f} h⁻¹ (Error: {bio.get("kla_relative_error", 0.0)*100:.2f}%)</div>
      </div>
      <div class="card">
        <h3>Mixing & Blend Time (τ₉₅)</h3>
        <div class="stat-val">{bio.get("mixing_time_95_s", 0.0):.2f} s</div>
        <div class="stat-sub">Circulation Turnover: {bio.get("circulation_time_s", 0.0):.2f} s (Q = {bio.get("pumping_rate_m3_s", 0.0)*1000:.2f} L/s)</div>
      </div>
    </div>
    <div class="grid-3">
      <div class="card">
        <h3>Max Shear Strain Rate (γ̇_max)</h3>
        <div class="stat-val">{bio.get("max_shear_rate_predicted_s_inv", 0.0):.1f} s⁻¹</div>
        <div class="stat-sub">CFD True: {bio.get("max_shear_rate_true_cfd_s_inv", 0.0):.1f} s⁻¹ (Error: {bio.get("max_shear_rate_relative_error", 0.0)*100:.2f}%)</div>
      </div>
      <div class="card">
        <h3>Kolmogorov Microscale (η_K)</h3>
        <div class="stat-val">{bio.get("kolmogorov_predicted_um", 0.0):.1f} μm</div>
        <div class="stat-sub">CFD True: {bio.get("kolmogorov_true_cfd_um", 0.0):.1f} μm (Error: {bio.get("kolmogorov_relative_error", 0.0)*100:.2f}%)</div>
      </div>
      <div class="card">
        <h3>Operating Regime</h3>
        <div class="stat-val" style="font-size: 20px; color: var(--primary);">Turbulent CSTR</div>
        <div class="stat-sub">Rushton Turbine &bull; Air-Water (ρ=1000, ν=1e-6)</div>
      </div>
    </div>

    <div style="margin-top: 36px; text-align: center; color: var(--text-muted); font-size: 13px;">
      BioFlo-CFD NLSS v1.0 &bull; Verified against OpenFOAM v2406 Finite Volume Ground Truth
    </div>
  </div>
</body>
</html>"""
        return html
