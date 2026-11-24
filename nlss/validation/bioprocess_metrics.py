"""
BioFlo-CFD: Derived Bioprocess Engineering Variables Layer (NLSS v1.0).

Quantifies derived bioprocess hydrodynamic parameters from 3D CFD velocity fields:
  1. Strain Rate Tensor S_ij and Shear Rate dot{gamma} = sqrt(2 * S : S)
  2. Local and Volumetric Energy Dissipation Rate epsilon = 2 * nu * S : S
  3. Specific Power Input P/V [W/m^3] (Dissipation Integral vs. Impeller Power)
  4. Volumetric Mass Transfer Coefficient k_L a [1/s, 1/h] (Van 't Riet / Garcia-Ochoa correlation)
  5. Turbulent Mixing Time tau_mix, 95% and Circulation Time tau_circ [s] (Grenville / Nienow)
  6. Kolmogorov Microscale length eta_K [um] (Shear stress exposure on cells)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch


def compute_strain_rate_tensor(
    u_grid: Union[np.ndarray, torch.Tensor],
    dx: float,
    dy: float,
    dz: float,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the symmetric strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    and the scalar shear strain rate dot{gamma} = sqrt(2 * S : S).

    Args:
        u_grid: 3D velocity array of shape (3, D, H, W) or (B, 3, D, H, W)
        dx, dy, dz: Spatial mesh resolutions [m]
        fluid_mask: Optional boolean mask of shape (D, H, W)

    Returns:
        shear_rate: Array of shape (D, H, W) with dot{gamma} [1/s]
        S_tensor: Array of shape (3, 3, D, H, W) with strain rate tensor components
    """
    if isinstance(u_grid, torch.Tensor):
        u_np = u_grid.detach().cpu().numpy()
    else:
        u_np = np.asarray(u_grid, dtype=np.float64)

    if u_np.ndim == 5:
        # Batch dimension: take the first item if B=1, or average
        u_np = u_np[0]

    if u_np.shape[0] != 3 or u_np.ndim != 4:
        raise ValueError(f"u_grid must have shape (3, D, H, W), got {u_np.shape}")

    ux = u_np[0]
    uy = u_np[1]
    uz = u_np[2]
    D, H, W = ux.shape

    # Gradients using centered finite differences with edge-replicated boundaries
    du_dx = np.gradient(ux, dx, axis=2)
    du_dy = np.gradient(ux, dy, axis=1)
    du_dz = np.gradient(ux, dz, axis=0)

    dv_dx = np.gradient(uy, dx, axis=2)
    dv_dy = np.gradient(uy, dy, axis=1)
    dv_dz = np.gradient(uy, dz, axis=0)

    dw_dx = np.gradient(uz, dx, axis=2)
    dw_dy = np.gradient(uz, dy, axis=1)
    dw_dz = np.gradient(uz, dz, axis=0)

    # Symmetric strain rate tensor S_ij = 0.5 * (grad u + grad u^T)
    S11 = du_dx
    S22 = dv_dy
    S33 = dw_dz
    S12 = 0.5 * (du_dy + dv_dx)
    S13 = 0.5 * (du_dz + dw_dx)
    S23 = 0.5 * (dv_dz + dw_dy)

    # 2 * S : S = 2 * (S11^2 + S22^2 + S33^2 + 2*S12^2 + 2*S13^2 + 2*S23^2)
    two_S_contract = 2.0 * (
        S11**2 + S22**2 + S33**2 + 2.0 * (S12**2 + S13**2 + S23**2)
    )
    two_S_contract = np.maximum(two_S_contract, 0.0)
    shear_rate = np.sqrt(two_S_contract)

    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask_np = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(fluid_mask, dtype=bool)
        shear_rate[~mask_np] = 0.0

    S_tensor = np.stack([
        np.stack([S11, S12, S13], axis=0),
        np.stack([S12, S22, S23], axis=0),
        np.stack([S13, S23, S33], axis=0),
    ], axis=0)

    return shear_rate, S_tensor


def compute_viscous_dissipation_rate(
    u_grid: Union[np.ndarray, torch.Tensor],
    dx: float,
    dy: float,
    dz: float,
    nu: float = 1e-6,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes local and volume-averaged turbulent/viscous energy dissipation rate:
        epsilon(x) = 2 * nu * (S : S) = nu * dot{gamma}^2  [m^2 / s^3 = W / kg]
    and the corresponding Kolmogorov microscale length:
        eta_K = (nu^3 / epsilon_mean)^(1/4)  [m, um]
    """
    shear_rate, _ = compute_strain_rate_tensor(u_grid, dx, dy, dz, fluid_mask=fluid_mask)
    epsilon_field = nu * (shear_rate**2)

    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask_np = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(fluid_mask, dtype=bool)
        active_eps = epsilon_field[mask_np]
        active_shear = shear_rate[mask_np]
    else:
        active_eps = epsilon_field.ravel()
        active_shear = shear_rate.ravel()

    mean_eps = float(np.mean(active_eps)) if len(active_eps) > 0 else 0.0
    median_eps = float(np.median(active_eps)) if len(active_eps) > 0 else 0.0
    max_eps = float(np.max(active_eps)) if len(active_eps) > 0 else 0.0
    p90_eps = float(np.percentile(active_eps, 90)) if len(active_eps) > 0 else 0.0
    p99_eps = float(np.percentile(active_eps, 99)) if len(active_eps) > 0 else 0.0

    mean_shear = float(np.mean(active_shear)) if len(active_shear) > 0 else 0.0
    max_shear = float(np.max(active_shear)) if len(active_shear) > 0 else 0.0
    p90_shear = float(np.percentile(active_shear, 90)) if len(active_shear) > 0 else 0.0

    # Multi-scale Kolmogorov microscale distribution eta_K = (nu^3 / epsilon)^(1/4)
    def calc_eta(eps_val: float) -> Tuple[float, float]:
        if eps_val > 1e-12:
            eta_m = float((nu**3 / eps_val) ** 0.25)
            return eta_m, eta_m * 1e6
        return float("inf"), float("inf")

    _, kolm_mean_um = calc_eta(mean_eps)
    _, kolm_median_um = calc_eta(median_eps)
    _, kolm_min_um = calc_eta(max_eps)       # Critical smallest eddy size at peak shear
    _, kolm_p10_um = calc_eta(p90_eps)       # Smallest 10% eddy threshold

    # Biological scale ratios (relative to 15 um mammalian CHO cell and 150 um microcarrier bead)
    cho_cell_ratio = float(15.0 / kolm_min_um) if np.isfinite(kolm_min_um) and kolm_min_um > 0 else 0.0
    microcarrier_ratio = float(150.0 / kolm_min_um) if np.isfinite(kolm_min_um) and kolm_min_um > 0 else 0.0

    return {
        "mean_dissipation_rate_w_kg": mean_eps,
        "median_dissipation_rate_w_kg": median_eps,
        "max_dissipation_rate_w_kg": max_eps,
        "p90_dissipation_rate_w_kg": p90_eps,
        "p99_dissipation_rate_w_kg": p99_eps,
        "mean_shear_rate_s_inv": mean_shear,
        "max_shear_rate_s_inv": max_shear,
        "p90_shear_rate_s_inv": p90_shear,
        "kolmogorov_microscale_mean_um": kolm_mean_um,
        "kolmogorov_microscale_median_um": kolm_median_um,
        "kolmogorov_microscale_min_um": kolm_min_um,
        "kolmogorov_microscale_p10_um": kolm_p10_um,
        "kolmogorov_microscale_um": kolm_mean_um,  # Backward compatibility
        "cell_diameter_to_min_kolmogorov_ratio": cho_cell_ratio,
        "microcarrier_to_min_kolmogorov_ratio": microcarrier_ratio,
        "dissipation_field": epsilon_field,
        "shear_rate_field": shear_rate,
    }


def compute_specific_power_input(
    u_grid: Union[np.ndarray, torch.Tensor],
    dx: float,
    dy: float,
    dz: float,
    rho: float = 1000.0,
    nu: float = 1e-6,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    N_rpm: Optional[float] = None,
    D_impeller: float = 0.166,
    Np: float = 1.5,
) -> Dict[str, float]:
    """
    Computes specific power input P/V [W/m^3] via:
      1. Direct volume integration of turbulent dissipation rate:
         P_diss = rho * int_{Omega} epsilon dOmega = rho * sum(epsilon) * dV
         (P/V)_diss = P_diss / V_liquid
      2. Empirical impeller shaft power correlation:
         P_imp = Np * rho * N^3 * D^5
         (P/V)_imp = P_imp / V_liquid
    """
    diss_stats = compute_viscous_dissipation_rate(u_grid, dx, dy, dz, nu=nu, fluid_mask=fluid_mask)
    eps_field = diss_stats["dissipation_field"]
    voxel_vol = dx * dy * dz

    if fluid_mask is not None:
        if isinstance(fluid_mask, torch.Tensor):
            mask_np = fluid_mask.detach().cpu().numpy().astype(bool)
        else:
            mask_np = np.asarray(fluid_mask, dtype=bool)
        n_cells = int(np.sum(mask_np))
        v_liquid = n_cells * voxel_vol
        total_p_diss = float(rho * np.sum(eps_field[mask_np]) * voxel_vol)
    else:
        n_cells = eps_field.size
        v_liquid = n_cells * voxel_vol
        total_p_diss = float(rho * np.sum(eps_field) * voxel_vol)

    specific_p_diss = total_p_diss / (v_liquid + 1e-12)

    # Empirical correlation if N_rpm is provided
    if N_rpm is not None and N_rpm > 0.0:
        n_rev_s = N_rpm / 60.0
        p_imp = float(Np * rho * (n_rev_s**3) * (D_impeller**5))
        specific_p_imp = p_imp / (v_liquid + 1e-12)
    else:
        p_imp = 0.0
        specific_p_imp = 0.0

    return {
        "liquid_volume_m3": float(v_liquid),
        "total_power_dissipation_w": total_p_diss,
        "specific_power_dissipation_w_m3": specific_p_diss,
        "impeller_power_w": p_imp,
        "specific_power_impeller_w_m3": specific_p_imp,
        "mean_dissipation_rate_w_kg": diss_stats["mean_dissipation_rate_w_kg"],
        "max_shear_rate_s_inv": diss_stats["max_shear_rate_s_inv"],
        "kolmogorov_microscale_um": diss_stats["kolmogorov_microscale_um"],
    }


def compute_kla(
    specific_power_w_m3: float,
    superficial_gas_velocity: float = 0.01,  # v_s in m/s (typical: 0.005 - 0.03 m/s)
    C: float = 0.026,
    alpha: float = 0.45,
    beta: float = 0.50,
) -> Dict[str, float]:
    """
    Computes volumetric mass transfer coefficient k_L a using the Van 't Riet (1979) /
    Garcia-Ochoa & Gomez (2009) bioprocess correlation:
        k_L a = C * (P / V)^alpha * (v_s)^beta  [s^-1]
    """
    p_v = max(specific_power_w_m3, 0.0)
    v_s = max(superficial_gas_velocity, 1e-6)
    kla_s_inv = float(C * (p_v**alpha) * (v_s**beta))
    kla_h_inv = float(kla_s_inv * 3600.0)

    # Typical oxygen transfer rate (OTR) driving force: delta_C = C*_O2 - C_L ~ 7.5 mg/L = 0.0075 kg/m^3
    otr_kg_m3_h = float(kla_h_inv * 0.0075)

    return {
        "kla_s_inv": kla_s_inv,
        "kla_h_inv": kla_h_inv,
        "estimated_otr_kg_m3_h": otr_kg_m3_h,
        "specific_power_input_w_m3": p_v,
        "superficial_gas_velocity_m_s": v_s,
    }


def compute_mixing_time(
    N_rpm: float,
    Np: float = 1.5,
    T_tank: float = 0.50,
    D_impeller: float = 0.166,
    liquid_volume: Optional[float] = None,
    Nq: float = 0.75,
) -> Dict[str, float]:
    """
    Computes 95% turbulent blend time and circulation turnover time:
        tau_mix, 95% = 5.4 / (N * Np^(1/3)) * (T / D)^2  (Grenville & Nienow 2004)
        tau_circ = V_liquid / Q_pump = V_liquid / (Nq * N * D^3)
    """
    if N_rpm <= 0.0:
        return {
            "mixing_time_95_s": float("inf"),
            "circulation_time_s": float("inf"),
            "pumping_rate_m3_s": 0.0,
        }

    n_rev_s = N_rpm / 60.0
    # Grenville turbulent blend time
    tau_mix = float((5.4 / (n_rev_s * (Np ** (1.0 / 3.0)))) * ((T_tank / D_impeller) ** 2))

    # Impeller pumping capacity Q = Nq * N * D^3
    q_pump = float(Nq * n_rev_s * (D_impeller**3))

    if liquid_volume is not None and liquid_volume > 0.0:
        tau_circ = float(liquid_volume / (q_pump + 1e-12))
    else:
        tau_circ = 0.0

    return {
        "mixing_time_95_s": tau_mix,
        "circulation_time_s": tau_circ,
        "pumping_rate_m3_s": q_pump,
        "impeller_tip_speed_m_s": float(np.pi * n_rev_s * D_impeller),
    }


def compute_higbie_mass_transfer(
    alpha_gas_grid: Union[np.ndarray, torch.Tensor],
    u_gas_grid: Union[np.ndarray, torch.Tensor],
    u_liquid_grid: Union[np.ndarray, torch.Tensor],
    d_bubble: Union[float, np.ndarray, torch.Tensor] = 0.003,  # Sauter d32 in meters (default 3mm)
    diffusivity: float = 2.1e-9,  # D_O2 in water at 25°C [m^2/s]
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    rho_liquid: float = 1000.0,
    mu_liquid: float = 1e-3,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    First-Principles Multiphase Interfacial Mass Transfer via Higbie Penetration Theory
    (Ported from BiRD / NREL BioReactorDesign toolbox).

    Governing Equations:
      1. Slip Velocity: u_rel(x) = ||U_gas(x) - U_liquid(x)||_2
      2. Interfacial Area Density: a(x) = 6 * alpha_gas(x) / d_bubble(x)   [m^2 / m^3]
      3. Liquid-Film Transfer Coefficient:
             k_L(x) = (2 / sqrt(pi)) * sqrt(D_L * u_rel(x) / d_bubble(x))   [m / s]
      4. Volumetric Mass Transfer Coefficient Field:
             k_L a(x) = k_L(x) * a(x)                                       [1 / s]
      5. Volume-Averaged Vessel k_L a:
             <k_L a> = integral(k_L a * dOmega) / integral((1 - alpha_gas) * dOmega)

    Args:
        alpha_gas_grid: Gas volume fraction array of shape (D, H, W)
        u_gas_grid: Gas velocity vector of shape (3, D, H, W)
        u_liquid_grid: Liquid velocity vector of shape (3, D, H, W)
        d_bubble: Bubble Sauter diameter [m], scalar or array of shape (D, H, W)
        diffusivity: Molecular diffusion coefficient D_L [m^2/s] (default 2.1e-9 for O2 in water)
        fluid_mask: Optional boolean mask of shape (D, H, W)
        rho_liquid: Liquid density [kg/m^3]
        mu_liquid: Dynamic liquid viscosity [Pa.s]

    Returns:
        Dictionary containing mean gas holdup, volume-averaged kla [s^-1, h^-1],
        mean interfacial area [m^2/m^3], mean kL [m/s], and 3D field arrays.
    """
    def to_np(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().astype(np.float64)
        return np.asarray(arr, dtype=np.float64)

    alpha_g = np.clip(to_np(alpha_gas_grid), 0.0, 1.0)
    ug = to_np(u_gas_grid)
    ul = to_np(u_liquid_grid)

    if ug.ndim == 5:
        ug = ug[0]
    if ul.ndim == 5:
        ul = ul[0]

    # Slip velocity magnitude: ||u_gas - u_liq||
    u_diff = ug - ul
    u_rel = np.sqrt(np.sum(u_diff**2, axis=0))

    if isinstance(d_bubble, (int, float)):
        db = float(max(d_bubble, 1e-6))
    else:
        db = np.maximum(to_np(d_bubble), 1e-6)

    # Interfacial area a = 6 * alpha_g / d_b
    a_field = 6.0 * alpha_g / db

    # Higbie penetration theory: k_L = (2 / sqrt(pi)) * sqrt(D_L * u_rel / d_b)
    # Avoid negative or singular values under sqrt
    safe_ratio = np.maximum(diffusivity * u_rel / db, 0.0)
    kl_field = (2.0 / np.sqrt(np.pi)) * np.sqrt(safe_ratio)

    # Local volumetric mass transfer coefficient k_L a [1/s]
    kla_field = kl_field * a_field

    if fluid_mask is not None:
        mask = to_np(fluid_mask).astype(bool)
        alpha_g[~mask] = 0.0
        a_field[~mask] = 0.0
        kl_field[~mask] = 0.0
        kla_field[~mask] = 0.0
    else:
        mask = np.ones_like(alpha_g, dtype=bool)

    # Liquid volume fraction
    alpha_l = np.maximum(1.0 - alpha_g, 1e-6)

    # Volume-weighted averages over the fluid domain
    n_fluid = np.sum(mask)
    if n_fluid > 0:
        mean_gas_holdup = float(np.mean(alpha_g[mask]))
        mean_interfacial_area = float(np.mean(a_field[mask]))
        mean_kl = float(np.mean(kl_field[mask]))
        # Averaged over liquid volume
        integrated_kla_s = float(np.sum(kla_field[mask]) / np.sum(alpha_l[mask]))
    else:
        mean_gas_holdup = 0.0
        mean_interfacial_area = 0.0
        mean_kl = 0.0
        integrated_kla_s = 0.0

    integrated_kla_h = integrated_kla_s * 3600.0

    return {
        "mean_gas_holdup": mean_gas_holdup,
        "mean_interfacial_area_m2_m3": mean_interfacial_area,
        "mean_kl_m_s": mean_kl,
        "kla_volume_averaged_s_inv": integrated_kla_s,
        "kla_volume_averaged_h_inv": integrated_kla_h,
        "kla_field": kla_field,
        "interfacial_area_field": a_field,
        "slip_velocity_field": u_rel,
    }


def compute_tracer_mixing_time_t95(
    tracer_history: Union[np.ndarray, torch.Tensor],
    time_points: Union[np.ndarray, List[float]],
    target_band: float = 0.05,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Dict[str, Union[float, List[float]]]:
    """
    Computes empirical 95% homogenization mixing time t_95 from transient tracer concentration series
    (Matching BiRD / NREL experimental tracer washout standard).

    Criterion:
      Evaluates the spatial Coefficient of Variation: CoV(t) = sigma_c(t) / c_bar(t).
      t_95 is the earliest time t* such that CoV(t) <= target_band (default 0.05)
      for all remaining time steps in the trajectory.

    Args:
        tracer_history: Concentration tensor of shape (T, D, H, W) or (T, N_cells)
        time_points: Array/list of timestamps [s] of length T
        target_band: Relative homogenization threshold (0.05 = +/-5% well-mixed band)
        fluid_mask: Optional boolean mask of active fluid cells

    Returns:
        Dictionary with t95_s, is_well_mixed boolean, CoV trajectory, and mean concentration curve.
    """
    def to_np(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy().astype(np.float64)
        return np.asarray(arr, dtype=np.float64)

    c_series = to_np(tracer_history)
    t_points = np.asarray(time_points, dtype=np.float64)
    T = len(t_points)

    if c_series.shape[0] != T:
        raise ValueError(f"Time dimension mismatch: c_series {c_series.shape[0]} vs time_points {T}")

    if fluid_mask is not None:
        mask = to_np(fluid_mask).astype(bool)
    else:
        mask = None

    mean_conc = []
    cov_list = []

    for k in range(T):
        c_k = c_series[k]
        if mask is not None:
            c_vals = c_k[mask]
        else:
            c_vals = c_k.flatten()

        c_bar = float(np.mean(c_vals))
        c_std = float(np.std(c_vals))
        cov = float(c_std / (abs(c_bar) + 1e-12))

        mean_conc.append(c_bar)
        cov_list.append(cov)

    cov_arr = np.array(cov_list)

    # Find earliest time t* where all subsequent steps stay within target_band
    is_mixed = False
    t95_val = float("inf")
    for idx in range(T):
        if np.all(cov_arr[idx:] <= target_band):
            t95_val = float(t_points[idx])
            is_mixed = True
            break

    return {
        "mixing_time_95_s": t95_val,
        "is_well_mixed": is_mixed,
        "final_cov": float(cov_arr[-1]) if T > 0 else 1.0,
        "cov_history": cov_list,
        "mean_concentration_history": mean_conc,
        "time_points": t_points.tolist(),
    }


def compute_dimensionless_bioprocess_numbers(
    N_rpm: float,
    D_impeller: float,
    rho_liquid: float = 1000.0,
    mu_liquid: float = 1e-3,
    surface_tension: float = 0.0728,  # water-air at 20°C [N/m]
    d_bubble: float = 0.003,          # [m]
    u_slip: float = 0.20,             # [m/s]
    diffusivity: float = 2.1e-9,      # [m^2/s]
    g: float = 9.81,
) -> Dict[str, float]:
    """
    Computes standard bioprocess dimensionless hydrodynamic numbers
    (Ported from BiRD dimensionless.py).
    """
    n_rev_s = N_rpm / 60.0
    nu_liquid = mu_liquid / rho_liquid

    # Impeller Reynolds number: Re_imp = rho * N * D^2 / mu
    Re_imp = float((rho_liquid * n_rev_s * (D_impeller**2)) / mu_liquid)

    # Impeller Froude number: Fr = N^2 * D / g
    Fr_imp = float(((n_rev_s**2) * D_impeller) / g)

    # Impeller Weber number: We = rho * N^2 * D^3 / sigma
    We_imp = float((rho_liquid * (n_rev_s**2) * (D_impeller**3)) / surface_tension)

    # Bubble Reynolds number: Re_b = rho * u_slip * d_b / mu
    Re_bubble = float((rho_liquid * u_slip * d_bubble) / mu_liquid)

    # Bubble Weber number: We_b = rho * u_slip^2 * d_b / sigma
    We_bubble = float((rho_liquid * (u_slip**2) * d_bubble) / surface_tension)

    # Schmidt number: Sc = nu / D_L
    Sc = float(nu_liquid / diffusivity)

    # Sherwood number via Higbie: Sh = (2 / sqrt(pi)) * sqrt(Re_b * Sc)
    Sh = float((2.0 / np.sqrt(np.pi)) * np.sqrt(Re_bubble * Sc))

    # Morton number: Mo = g * mu^4 * Delta_rho / (rho^2 * sigma^3)
    # For water-air, Delta_rho ~ rho_liquid
    Mo = float((g * (mu_liquid**4)) / ((rho_liquid**2) * (surface_tension**3)))

    return {
        "reynolds_impeller": Re_imp,
        "froude_impeller": Fr_imp,
        "weber_impeller": We_imp,
        "reynolds_bubble": Re_bubble,
        "weber_bubble": We_bubble,
        "schmidt_number": Sc,
        "sherwood_number": Sh,
        "morton_number": Mo,
    }


def evaluate_bioprocess_field_comparison(
    u_pred_grid: Union[np.ndarray, torch.Tensor],
    u_true_grid: Union[np.ndarray, torch.Tensor],
    dx: float,
    dy: float,
    dz: float,
    fluid_mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
    nu: float = 1e-6,
    rho: float = 1000.0,
    N_rpm: Optional[float] = None,
    D_impeller: float = 0.166,
    Np: float = 1.5,
    v_s: float = 0.01,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Performs comprehensive bioprocess engineering validation comparing predicted vs. CFD ground truth.
    Computes absolute values and relative errors across:
      - Specific power input (P/V)
      - Mean and maximum shear strain rate (dot{gamma})
      - Turbulent dissipation rate (epsilon)
      - Kolmogorov microscale length (eta_K)
      - Mass transfer coefficient (k_L a)
    """
    p_pred = compute_specific_power_input(
        u_pred_grid, dx, dy, dz, rho=rho, nu=nu, fluid_mask=fluid_mask, N_rpm=N_rpm, D_impeller=D_impeller, Np=Np
    )
    p_true = compute_specific_power_input(
        u_true_grid, dx, dy, dz, rho=rho, nu=nu, fluid_mask=fluid_mask, N_rpm=N_rpm, D_impeller=D_impeller, Np=Np
    )

    kla_pred = compute_kla(p_pred["specific_power_dissipation_w_m3"], superficial_gas_velocity=v_s)
    kla_true = compute_kla(p_true["specific_power_dissipation_w_m3"], superficial_gas_velocity=v_s)

    def rel_err(pred: float, true: float) -> float:
        if abs(true) < 1e-12:
            return 0.0 if abs(pred) < 1e-12 else 1.0
        return float(abs(pred - true) / abs(true))

    rel_pv_err = rel_err(p_pred["specific_power_dissipation_w_m3"], p_true["specific_power_dissipation_w_m3"])
    rel_shear_mean_err = rel_err(p_pred["mean_dissipation_rate_w_kg"], p_true["mean_dissipation_rate_w_kg"])
    rel_shear_max_err = rel_err(p_pred["max_shear_rate_s_inv"], p_true["max_shear_rate_s_inv"])
    rel_kla_err = rel_err(kla_pred["kla_h_inv"], kla_true["kla_h_inv"])
    rel_kolmogorov_err = rel_err(p_pred["kolmogorov_microscale_um"], p_true["kolmogorov_microscale_um"])

    return {
        "specific_power_predicted_w_m3": p_pred["specific_power_dissipation_w_m3"],
        "specific_power_true_cfd_w_m3": p_true["specific_power_dissipation_w_m3"],
        "specific_power_relative_error": rel_pv_err,
        "mean_shear_rate_predicted_s_inv": p_pred["mean_dissipation_rate_w_kg"],
        "mean_shear_rate_true_cfd_s_inv": p_true["mean_dissipation_rate_w_kg"],
        "max_shear_rate_predicted_s_inv": p_pred["max_shear_rate_s_inv"],
        "max_shear_rate_true_cfd_s_inv": p_true["max_shear_rate_s_inv"],
        "max_shear_rate_relative_error": rel_shear_max_err,
        "kla_predicted_h_inv": kla_pred["kla_h_inv"],
        "kla_true_cfd_h_inv": kla_true["kla_h_inv"],
        "kla_relative_error": rel_kla_err,
        "kolmogorov_predicted_um": p_pred["kolmogorov_microscale_um"],
        "kolmogorov_true_cfd_um": p_true["kolmogorov_microscale_um"],
        "kolmogorov_relative_error": rel_kolmogorov_err,
        "kolmogorov_min_predicted_um": p_pred.get("kolmogorov_microscale_min_um", p_pred["kolmogorov_microscale_um"]),
        "kolmogorov_min_true_um": p_true.get("kolmogorov_microscale_min_um", p_true["kolmogorov_microscale_um"]),
        "cell_to_kolmogorov_ratio": p_pred.get("cell_diameter_to_min_kolmogorov_ratio", 0.0),
    }
