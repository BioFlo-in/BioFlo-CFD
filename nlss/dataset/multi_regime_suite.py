"""
Comprehensive Multi-Regime CFD Simulation Suite.

Generates a wide variety of transient Navier-Stokes flows to eliminate overfitting:
  1. Stirred-Tank Bioreactor (STR) flows: Varied tip speeds (0.6, 1.0, 1.4 m/s) and impeller clearances (0.22, 0.33, 0.50).
  2. Multi-Driven Cavities: Standard forward, reverse, high/low speed, and double-driven cavities.
  3. Planar Channel & Duct Flows: Horizontal and vertical Poiseuille flows with parabolic profiles.
  4. Obstacle & Bluff Body Flows: Channel flow around central and offset solid obstacles with separation wakes.
  5. Taylor-Green Multi-Scale Vortex Decays: Modes k=1, 2, 3 with viscous dissipation.
  6. Perturbed Shear Layer Flows: Hyperbolic tangent profiles with Kelvin-Helmholtz roll-up.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from .bioreactor import BioreactorSimulator, BioreactorGeometry
from .generator import NavierStokes2DSimulator


def simulate_obstacle_channel(
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 40,
    dt: float = 0.005,
    nu: float = 0.01,
    u_inlet_max: float = 1.0,
    obstacle_x: Tuple[float, float] = (0.30, 0.45),
    obstacle_y: Tuple[float, float] = (0.40, 0.60),
) -> np.ndarray:
    """Simulates transient channel flow past a submerged solid obstacle."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    rho = 1.0

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = np.zeros((ny, nx), dtype=np.float32)

    obstacle_mask = (
        (X >= obstacle_x[0]) & (X <= obstacle_x[1])
        & (Y >= obstacle_y[0]) & (Y <= obstacle_y[1])
    )
    y_col = y[:, None]
    u_inlet = (4.0 * u_inlet_max * y_col * (1.0 - y_col)).astype(np.float32)

    dx2 = dx**2
    dy2 = dy**2
    denom = 2.0 * (dx2 + dy2)

    u_history, v_history, p_history = [u.copy()], [v.copy()], [p.copy()]

    for _ in range(n_steps):
        u[:, 0] = u_inlet[:, 0]
        v[:, 0] = 0.0
        u[obstacle_mask] = 0.0
        v[obstacle_mask] = 0.0

        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)

        b = np.zeros_like(p)
        b[1:-1, 1:-1] = rho * (
            (1.0 / dt) * (du_dx + dv_dy)
            - (du_dx**2 + 2 * (du_dy * dv_dx) + dv_dy**2)
        )

        for _ in range(35):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (
                    (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
                    + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
                    - b[1:-1, 1:-1] * dx2 * dy2
                )
                / denom
            )
            p[:, 0] = p[:, 1]
            p[:, -1] = 0.0  # Atmospheric outlet
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

        un = u.copy()
        vn = v.copy()
        u_c = un[1:-1, 1:-1]
        v_c = vn[1:-1, 1:-1]

        u_adv_x = np.where(u_c > 0, u_c * (u_c - un[1:-1, :-2]) / dx, u_c * (un[1:-1, 2:] - u_c) / dx)
        u_adv_y = np.where(v_c > 0, v_c * (u_c - un[:-2, 1:-1]) / dy, v_c * (un[2:, 1:-1] - u_c) / dy)
        v_adv_x = np.where(u_c > 0, u_c * (v_c - vn[1:-1, :-2]) / dx, u_c * (vn[1:-1, 2:] - v_c) / dx)
        v_adv_y = np.where(v_c > 0, v_c * (v_c - vn[:-2, 1:-1]) / dy, v_c * (vn[2:, 1:-1] - v_c) / dy)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - dt * (u_adv_x + u_adv_y)
            - (dt / (2 * rho * dx)) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu * (dt / dx2) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            + nu * (dt / dy2) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - dt * (v_adv_x + v_adv_y)
            - (dt / (2 * rho * dy)) * (p[2:, 1:-1] - p[:-2, 1:-1])
            + nu * (dt / dx2) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
            + nu * (dt / dy2) * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
        )

        # Boundary conditions
        u[:, 0] = u_inlet[:, 0]
        v[:, 0] = 0.0
        u[0, :] = 0.0
        v[0, :] = 0.0
        u[-1, :] = 0.0
        v[-1, :] = 0.0
        u[:, -1] = u[:, -2]  # Convective zero-gradient outlet
        v[:, -1] = v[:, -2]

        u[obstacle_mask] = 0.0
        v[obstacle_mask] = 0.0

        u_history.append(u.copy())
        v_history.append(v.copy())
        p_history.append(p.copy())

    return np.stack([np.stack(u_history), np.stack(v_history), np.stack(p_history)], axis=1)


def simulate_shear_layer(
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 40,
    dt: float = 0.005,
    nu: float = 0.01,
    pert_amp: float = 0.05,
) -> np.ndarray:
    """Simulates perturbed shear layer using stable upwind momentum advection."""
    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    rho = 1.0

    y_sh = np.linspace(-1, 1, ny)[:, None]
    u = np.repeat(np.tanh(y_sh / 0.20), nx, axis=1).astype(np.float32)
    v_row = (pert_amp * np.sin(2 * np.pi * np.linspace(0, 1, nx)[None, :])).astype(np.float32)
    v = np.repeat(v_row, ny, axis=0)
    p = np.zeros((ny, nx), dtype=np.float32)

    dx2 = dx**2
    dy2 = dy**2
    denom = 2.0 * (dx2 + dy2)
    dt_sub = dt / 2
    total_substeps = n_steps * 2

    u_history, v_history, p_history = [u.copy()], [v.copy()], [p.copy()]

    for step_i in range(1, total_substeps + 1):
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)

        b = np.zeros_like(p)
        b[1:-1, 1:-1] = rho * (
            (1.0 / dt_sub) * (du_dx + dv_dy)
            - (du_dx**2 + 2 * (du_dy * dv_dx) + dv_dy**2)
        )

        for _ in range(30):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (
                    (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
                    + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
                    - b[1:-1, 1:-1] * dx2 * dy2
                )
                / denom
            )
            p[:, -1] = p[:, -2]
            p[:, 0] = p[:, 1]
            p[0, :] = p[1, :]
            p[-1, :] = 0.0

        un = u.copy()
        vn = v.copy()
        u_c = un[1:-1, 1:-1]
        v_c = vn[1:-1, 1:-1]

        u_adv_x = np.where(u_c > 0, u_c * (u_c - un[1:-1, :-2]) / dx, u_c * (un[1:-1, 2:] - u_c) / dx)
        u_adv_y = np.where(v_c > 0, v_c * (u_c - un[:-2, 1:-1]) / dy, v_c * (un[2:, 1:-1] - u_c) / dy)
        v_adv_x = np.where(u_c > 0, u_c * (v_c - vn[1:-1, :-2]) / dx, u_c * (vn[1:-1, 2:] - v_c) / dx)
        v_adv_y = np.where(v_c > 0, v_c * (v_c - vn[:-2, 1:-1]) / dy, v_c * (vn[2:, 1:-1] - v_c) / dy)

        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - dt_sub * (u_adv_x + u_adv_y)
            - (dt_sub / (2 * rho * dx)) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu * (dt_sub / dx2) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            + nu * (dt_sub / dy2) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - dt_sub * (v_adv_x + v_adv_y)
            - (dt_sub / (2 * rho * dy)) * (p[2:, 1:-1] - p[:-2, 1:-1])
            + nu * (dt_sub / dx2) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
            + nu * (dt_sub / dy2) * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
        )

        u[0, :] = -1.0
        u[-1, :] = 1.0
        v[0, :] = 0.0
        v[-1, :] = 0.0
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

        if step_i % 2 == 0:
            u_history.append(u.copy())
            v_history.append(v.copy())
            p_history.append(p.copy())

    return np.stack([np.stack(u_history), np.stack(v_history), np.stack(p_history)], axis=1)


def generate_all_multi_regime_trajectories(
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 40,
    dt: float = 0.005,
) -> Dict[str, List[np.ndarray]]:
    """
    Synthesizes diverse CFD regimes across 6 major fluid dynamic categories:
      A. Stirred-Tank Bioreactors (STR)
      B. Multi-Speed & Reverse Driven Cavities
      C. Channel & Duct Flows (Horizontal & Vertical)
      D. Submerged Obstacle Wake Flows
      E. Taylor-Green Multi-Mode Vortex Decays
      F. Perturbed Kelvin-Helmholtz Shear Layers
    """
    regimes: Dict[str, List[np.ndarray]] = {
        "bioreactor": [],
        "cavity": [],
        "channel": [],
        "obstacle": [],
        "taylor_green": [],
        "shear_layer": [],
    }

    # =========================================================================
    # A. STIRRED-TANK BIOREACTORS (Varied tip speeds and impeller clearance)
    # =========================================================================
    print("  [1/6] Simulating Stirred-Tank Bioreactor regimes (STR)...")
    bio_configs = [
        {"u_tip": 1.0, "clearance": 0.333, "desc": "Standard Rushton (1.0 m/s, C=T/3)"},
        {"u_tip": 0.6, "clearance": 0.333, "desc": "Low-Shear Culture (0.6 m/s, C=T/3)"},
        {"u_tip": 1.4, "clearance": 0.333, "desc": "High-Shear Dispersion (1.4 m/s, C=T/3)"},
        {"u_tip": 1.0, "clearance": 0.220, "desc": "Deep Impeller Clearance (C=0.22 T)"},
        {"u_tip": 1.0, "clearance": 0.500, "desc": "Mid-Depth Impeller Clearance (C=0.50 T)"},
    ]
    for cfg in bio_configs:
        geom = BioreactorGeometry(T=1.0, H=1.0, clearance_C=cfg["clearance"])
        # Substep for CFL stability across high tip speeds
        substeps = 2
        dt_sub = dt / substeps
        sim = BioreactorSimulator(nx=nx, ny=ny, nu=0.01, dt=dt_sub, u_tip=cfg["u_tip"], geom=geom)
        res = sim.run_simulation(n_steps=n_steps * substeps, save_interval=substeps, is_cavity=False)
        traj = np.stack([res["u"], res["v"], res["p"]], axis=1)  # (T, 3, ny, nx)
        regimes["bioreactor"].append(traj)

    # =========================================================================
    # B. MULTI-DRIVEN & REVERSE CAVITIES (Forward, reverse, low/high speed, double)
    # =========================================================================
    print("  [2/6] Simulating Multi-Driven Cavities (Forward, Reverse, Double)...")
    cav_speeds = [1.0, 1.5, 0.5, -1.0, -1.5]
    for spd in cav_speeds:
        sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=0.01, dt=dt)
        sim.set_lid_driven_cavity(u_lid=spd)
        res = sim.run_simulation(n_steps=n_steps, save_interval=1, is_cavity=True, u_lid=spd)
        traj = np.stack([res["u"], res["v"], res["p"]], axis=1)
        regimes["cavity"].append(traj)

    # Double-Driven Cavity (top +1.0, bottom -1.0)
    sim_double = NavierStokes2DSimulator(nx=nx, ny=ny, nu=0.01, dt=dt)
    sim_double.u[-1, :] = 1.0
    sim_double.u[0, :] = -1.0
    u_h, v_h, p_h = [sim_double.u.copy()], [sim_double.v.copy()], [sim_double.p.copy()]
    for _ in range(n_steps):
        sim_double.step(is_cavity=False)
        sim_double.u[-1, :] = 1.0
        sim_double.u[0, :] = -1.0
        sim_double.u[:, 0] = 0.0
        sim_double.u[:, -1] = 0.0
        sim_double.v[0, :] = 0.0
        sim_double.v[-1, :] = 0.0
        sim_double.v[:, 0] = 0.0
        sim_double.v[:, -1] = 0.0
        u_h.append(sim_double.u.copy())
        v_h.append(sim_double.v.copy())
        p_h.append(sim_double.p.copy())
    regimes["cavity"].append(np.stack([np.stack(u_h), np.stack(v_h), np.stack(p_h)], axis=1))

    # =========================================================================
    # C. PLANAR CHANNEL & DUCT FLOWS (Poiseuille parabolic profiles)
    # =========================================================================
    print("  [3/6] Simulating Channel & Duct Flows (Horizontal & Vertical)...")
    y = np.linspace(0, 1, ny)[:, None]
    for u0 in [1.0, 1.5, -1.0]:
        u_poi = (4.0 * u0 * y * (1.0 - y)).astype(np.float32)
        u_grid = np.repeat(u_poi, nx, axis=1)
        sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=0.01, dt=dt)
        sim.u = u_grid.copy()
        u_h, v_h, p_h = [sim.u.copy()], [sim.v.copy()], [sim.p.copy()]
        for _ in range(n_steps):
            sim.step(is_cavity=False)
            sim.u[:, 0] = u_grid[:, 0]
            sim.u[0, :] = 0.0
            sim.u[-1, :] = 0.0
            sim.v[0, :] = 0.0
            sim.v[-1, :] = 0.0
            u_h.append(sim.u.copy())
            v_h.append(sim.v.copy())
            p_h.append(sim.p.copy())
        regimes["channel"].append(np.stack([np.stack(u_h), np.stack(v_h), np.stack(p_h)], axis=1))

    # Vertical Duct Flow (Upflow v0 = 1.0)
    x = np.linspace(0, 1, nx)[None, :]
    v_vert = (4.0 * 1.0 * x * (1.0 - x)).astype(np.float32)
    v_grid = np.repeat(v_vert, ny, axis=0)
    sim_v = NavierStokes2DSimulator(nx=nx, ny=ny, nu=0.01, dt=dt)
    sim_v.v = v_grid.copy()
    u_h, v_h, p_h = [sim_v.u.copy()], [sim_v.v.copy()], [sim_v.p.copy()]
    for _ in range(n_steps):
        sim_v.step(is_cavity=False)
        sim_v.v[0, :] = v_grid[0, :]
        sim_v.v[:, 0] = 0.0
        sim_v.v[:, -1] = 0.0
        sim_v.u[:, 0] = 0.0
        sim_v.u[:, -1] = 0.0
        u_h.append(sim_v.u.copy())
        v_h.append(sim_v.v.copy())
        p_h.append(sim_v.p.copy())
    regimes["channel"].append(np.stack([np.stack(u_h), np.stack(v_h), np.stack(p_h)], axis=1))

    # =========================================================================
    # D. OBSTACLE & BLUFF BODY FLOWS (Central & offset solid blocks)
    # =========================================================================
    print("  [4/6] Simulating Bluff Body & Obstacle Wake Flows...")
    # Center square obstacle
    regimes["obstacle"].append(
        simulate_obstacle_channel(
            nx=nx, ny=ny, n_steps=n_steps, dt=dt,
            obstacle_x=(0.32, 0.46), obstacle_y=(0.40, 0.60),
        )
    )
    # Offset obstacle (causing asymmetric vortex shedding)
    regimes["obstacle"].append(
        simulate_obstacle_channel(
            nx=nx, ny=ny, n_steps=n_steps, dt=dt,
            obstacle_x=(0.30, 0.44), obstacle_y=(0.25, 0.45),
        )
    )
    # Orifice / Constriction Flow (top and bottom baffles)
    regimes["obstacle"].append(
        simulate_obstacle_channel(
            nx=nx, ny=ny, n_steps=n_steps, dt=dt,
            obstacle_x=(0.45, 0.55), obstacle_y=(0.65, 0.95),
        )
    )

    # =========================================================================
    # E. TAYLOR-GREEN MULTI-SCALE VORTEX DECAYS (Modes k=1, 2, 3)
    # =========================================================================
    print("  [5/6] Simulating Taylor-Green Multi-Mode Decays (k=1, 2, 3)...")
    x_tg = np.linspace(0, 2 * np.pi, nx)
    y_tg = np.linspace(0, 2 * np.pi, ny)
    X_tg, Y_tg = np.meshgrid(x_tg, y_tg)
    for k_m in [1, 2, 3]:
        u_tg, v_tg, p_tg = [], [], []
        nu_tg = 0.01
        for s in range(n_steps + 1):
            t_val = s * dt
            decay = np.exp(-2.0 * (k_m**2) * nu_tg * t_val)
            decay_p = np.exp(-4.0 * (k_m**2) * nu_tg * t_val)
            u_t = (np.sin(k_m * X_tg) * np.cos(k_m * Y_tg) * decay).astype(np.float32)
            v_t = (-np.cos(k_m * X_tg) * np.sin(k_m * Y_tg) * decay).astype(np.float32)
            p_t = (-0.25 * (np.cos(2 * k_m * X_tg) + np.cos(2 * k_m * Y_tg)) * decay_p).astype(np.float32)
            u_tg.append(u_t)
            v_tg.append(v_t)
            p_tg.append(p_t)
        regimes["taylor_green"].append(np.stack([np.stack(u_tg), np.stack(v_tg), np.stack(p_tg)], axis=1))

    # =========================================================================
    # F. PERTURBED SHEAR LAYERS (Kelvin-Helmholtz vortex pairings)
    # =========================================================================
    print("  [6/6] Simulating Perturbed Kelvin-Helmholtz Shear Layers...")
    for pert_amp in [0.05, 0.10]:
        traj_sh = simulate_shear_layer(nx=nx, ny=ny, n_steps=n_steps, dt=dt, nu=0.01, pert_amp=pert_amp)
        regimes["shear_layer"].append(traj_sh)

    total_trajs = sum(len(v) for v in regimes.values())
    total_snaps = sum(sum(t.shape[0] for t in v) for v in regimes.values())
    print(f"✓ All Multi-Regime Trajectories Generated: {total_trajs} distinct flow simulations, {total_snaps} physical snapshots.")
    return regimes


def apply_so2_rotation_augmentation(trajectory: np.ndarray) -> List[np.ndarray]:
    """
    Applies exact SO(2) physical Navier-Stokes rotational symmetries (0, 90, 180, 270 deg).
    Under 90-degree counterclockwise rotation:
      (x', y') = (-y, x)
      (u', v') = (-v, u)
      p' = p
    """
    # trajectory shape: (T, 3, ny, nx) where channel 0=u, 1=v, 2=p
    augmented = [trajectory]

    # 90 deg CCW
    u90 = -np.rot90(trajectory[:, 1], k=1, axes=(1, 2))
    v90 = np.rot90(trajectory[:, 0], k=1, axes=(1, 2))
    p90 = np.rot90(trajectory[:, 2], k=1, axes=(1, 2))
    augmented.append(np.stack([u90, v90, p90], axis=1))

    # 180 deg
    u180 = -np.rot90(trajectory[:, 0], k=2, axes=(1, 2))
    v180 = -np.rot90(trajectory[:, 1], k=2, axes=(1, 2))
    p180 = np.rot90(trajectory[:, 2], k=2, axes=(1, 2))
    augmented.append(np.stack([u180, v180, p180], axis=1))

    # 270 deg
    u270 = np.rot90(trajectory[:, 1], k=3, axes=(1, 2))
    v270 = -np.rot90(trajectory[:, 0], k=3, axes=(1, 2))
    p270 = np.rot90(trajectory[:, 2], k=3, axes=(1, 2))
    augmented.append(np.stack([u270, v270, p270], axis=1))

    return augmented


def build_augmented_master_dataset(
    regimes: Dict[str, List[np.ndarray]],
    apply_symmetry: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    """
    Combines all simulation regimes, applies SO(2) symmetry augmentation,
    and constructs valid within-trajectory single-step transition pairs:
      inputs:  S_t
      targets: S_{t+1}
    Guarantees no boundary crossing between distinct trajectories.
    """
    all_inputs = []
    all_targets = []
    regime_counts = {}

    for name, traj_list in regimes.items():
        count = 0
        for traj in traj_list:
            # traj shape: (T, 3, ny, nx)
            if apply_symmetry:
                aug_trajs = apply_so2_rotation_augmentation(traj)
            else:
                aug_trajs = [traj]

            for a_traj in aug_trajs:
                all_inputs.append(a_traj[:-1])
                all_targets.append(a_traj[1:])
                count += a_traj.shape[0] - 1

        regime_counts[name] = count

    inputs_tensor = torch.from_numpy(np.concatenate(all_inputs, axis=0)).float()
    targets_tensor = torch.from_numpy(np.concatenate(all_targets, axis=0)).float()

    return inputs_tensor, targets_tensor, regime_counts
