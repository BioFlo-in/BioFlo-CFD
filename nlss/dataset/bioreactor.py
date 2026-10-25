"""
Stirred-Tank Bioreactor (STR) Fluid Dynamics & Bioprocess Shear Engine.

Simulates 2D transient flow in an industrial stirred-tank bioreactor equipped with:
  - Central agitator drive shaft
  - Rushton turbine impeller (radial discharge jet)
  - Wall baffles preventing vortex funneling
  - Top liquid free-surface (slip boundary)
  - No-slip bottom and vessel sidewalls

Computes bioprocess shear stress and turbulent dissipation metrics critical for
cell viability assessment (mammalian CHO, stem cells, microbial fermentation).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

from .generator import NavierStokes2DSimulator
from ..io.foam_writer import export_simulation_state_to_openfoam


@dataclass
class BioreactorGeometry:
    """
    Standard geometrical proportions of a Rushton-stirred tank reactor (STR):
      - Tank diameter T = 1.0 m, liquid height H = 1.0 m (H/T = 1.0)
      - Impeller diameter D = T / 3 = 0.33 m
      - Off-bottom clearance C = T / 3 = 0.33 m
      - Impeller blade height w = D / 5 = 0.07 m
      - Shaft diameter d_s = 0.08 m
      - Baffle width B = T / 10 = 0.10 m (modeled along sidewalls)
    """
    T: float = 1.0
    H: float = 1.0
    clearance_C: float = 0.333
    impeller_D: float = 0.333
    impeller_height: float = 0.070
    shaft_diameter: float = 0.080
    baffle_width: float = 0.050


class BioreactorSimulator(NavierStokes2DSimulator):
    """
    Transient Incompressible Navier-Stokes Solver for Stirred-Tank Bioreactors.
    
    Generates the characteristic dual-loop toroidal recirculation flow:
      1. Radial discharge jet directed toward vessel walls at impeller clearance.
      2. Upper recirculation loop returning along top free-surface and shaft.
      3. Lower recirculation loop sweeping the tank floor.
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        lx: float = 1.0,
        ly: float = 1.0,
        rho: float = 1.0,       # Kinematic formulation (rho=1.0, p is kinematic pressure p/rho like OpenFOAM icoFoam)
        nu: float = 0.01,       # Kinematic viscosity (m^2/s)
        dt: float = 0.005,
        u_tip: float = 1.0,     # Impeller tip velocity (m/s)
        geom: Optional[BioreactorGeometry] = None,
    ):
        super().__init__(nx=nx, ny=ny, lx=lx, ly=ly, rho=rho, nu=nu, dt=dt)
        self.u_tip = u_tip
        self.geom = geom or BioreactorGeometry(T=lx, H=ly)

        # Coordinate grids
        x = np.linspace(0, self.lx, self.nx)
        y = np.linspace(0, self.ly, self.ny)
        self.X, self.Y = np.meshgrid(x, y)

        # Create geometric masks
        xc = self.lx / 2.0
        yc = self.geom.clearance_C
        r_shaft = self.geom.shaft_diameter / 2.0
        r_imp = self.geom.impeller_D / 2.0
        h_half = self.geom.impeller_height / 2.0

        # Shaft mask: vertical cylinder along x = xc from clearance down or through the reactor
        self.shaft_mask = (np.abs(self.X - xc) <= r_shaft) & (self.Y >= (yc - h_half * 0.5))

        # Impeller swept zone: radial region excluding shaft
        self.impeller_mask = (
            (np.abs(self.Y - yc) <= h_half)
            & (np.abs(self.X - xc) <= r_imp)
            & (np.abs(self.X - xc) > r_shaft)
        )

        # Baffle masks along sidewalls (preventing tangential solid-body swirl)
        self.left_baffle_mask = (self.X <= self.geom.baffle_width) & (self.Y >= 0.10) & (self.Y <= 0.90)
        self.right_baffle_mask = (self.X >= (self.lx - self.geom.baffle_width)) & (self.Y >= 0.10) & (self.Y <= 0.90)

    def apply_bioreactor_boundary_conditions(self):
        """Enforces vessel walls, free-surface, shaft, and baffle boundary conditions."""
        # Bottom wall: no-slip
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0

        # Left and right walls: no-slip
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0

        # Top boundary: liquid free surface (slip: zero vertical velocity, zero shear du/dy = 0)
        self.v[-1, :] = 0.0
        self.u[-1, :] = self.u[-2, :]

        # Central agitator shaft: no-slip solid boundary
        self.u[self.shaft_mask] = 0.0
        self.v[self.shaft_mask] = 0.0

    def apply_impeller_radial_jet(self):
        """
        Applies radial discharge momentum from the rotating Rushton turbine blades.
        Fluid is pumped radially outwards from the shaft toward both vessel walls.
        """
        xc = self.lx / 2.0
        # Right blade sector: jet velocity toward right wall (+u_tip)
        right_blade = self.impeller_mask & (self.X > xc)
        self.u[right_blade] = self.u_tip

        # Left blade sector: jet velocity toward left wall (-u_tip)
        left_blade = self.impeller_mask & (self.X < xc)
        self.u[left_blade] = -self.u_tip

        # Axial suction into the impeller eye from above and below
        yc = self.geom.clearance_C
        above_eye = self.impeller_mask & (self.Y > yc)
        below_eye = self.impeller_mask & (self.Y < yc)
        self.v[above_eye] = -0.25 * self.u_tip
        self.v[below_eye] = 0.25 * self.u_tip

    def step(self, is_cavity: bool = False, u_lid: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advances the bioreactor flow by one time step dt using Chorin's projection method
        with immersed impeller forcing and vessel boundary conditions.
        """
        u, v, p = self.u, self.v, self.p
        dt, dx, dy = self.dt, self.dx, self.dy
        rho, nu = self.rho, self.nu

        # 1. Impeller active momentum forcing
        self.apply_impeller_radial_jet()

        # 2. Compute divergence source term for pressure Poisson equation
        b = np.zeros_like(p)
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)

        b[1:-1, 1:-1] = rho * (
            (1.0 / dt) * (du_dx + dv_dy)
            - (du_dx**2 + 2 * (du_dy * dv_dx) + dv_dy**2)
        )

        # 3. Solve Pressure Poisson Equation
        dx2 = dx**2
        dy2 = dy**2
        denom = 2.0 * (dx2 + dy2)
        nit = 40

        for _ in range(nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                (
                    (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
                    + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
                    - b[1:-1, 1:-1] * dx2 * dy2
                )
                / denom
            )
            # Pressure boundary conditions: Neumann dp/dn = 0 on solid walls
            p[:, -1] = p[:, -2]  # Right
            p[:, 0] = p[:, 1]    # Left
            p[0, :] = p[1, :]    # Bottom
            p[-1, :] = 0.0       # Atmospheric pressure at liquid free surface

        # 4. Momentum advection (stable upwind differencing) and diffusion
        un = u.copy()
        vn = v.copy()

        u_c = un[1:-1, 1:-1]
        v_c = vn[1:-1, 1:-1]

        # Upwind advection in x and y directions
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

        # 5. Enforce vessel boundary conditions
        self.apply_bioreactor_boundary_conditions()

        return self.u, self.v, self.p


def compute_bioprocess_shear_metrics(
    u: np.ndarray,
    v: np.ndarray,
    nu: float = 1.0e-3,
    rho: float = 1000.0,
    mu: Optional[float] = None,
    dx: float = 1.0 / 63,
    dy: float = 1.0 / 63,
) -> Dict[str, Union[np.ndarray, float, str]]:
    """
    Computes key hydrodynamic shear stress and energy dissipation metrics:
      - Shear strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
      - Shear strain rate magnitude gamma_dot = sqrt(2 * S_ij * S_ij) (1/s)
      - Hydrodynamic shear stress tau = mu * gamma_dot = rho * nu * gamma_dot (Pa)
      - Turbulent energy dissipation rate epsilon = nu * gamma_dot^2 (m^2/s^3 = W/kg)
      - Kolmogorov microscale length eta_K = (nu^3 / epsilon)^(1/4) (m)

    Crucial for cell culture:
      Mammalian cells (CHO, HEK293) lyse when shear stress tau exceeds 5-10 Pa,
      or when Kolmogorov eddies eta_K become smaller than cell diameter (~15 um).
    """
    if mu is None:
        mu = rho * nu  # Dynamic viscosity (Pa*s)

    # Calculate velocity gradients (handling single snapshot or trajectory)
    if u.ndim == 2:
        dudx = np.gradient(u, dx, axis=1)
        dudy = np.gradient(u, dy, axis=0)
        dvdx = np.gradient(v, dx, axis=1)
        dvdy = np.gradient(v, dy, axis=0)
    elif u.ndim == 3:  # (T, ny, nx)
        dudx = np.gradient(u, dx, axis=2)
        dudy = np.gradient(u, dy, axis=1)
        dvdx = np.gradient(v, dx, axis=2)
        dvdy = np.gradient(v, dy, axis=1)
    else:
        raise ValueError(f"Expected 2D or 3D velocity array, got shape {u.shape}")

    # Rate-of-strain tensor components
    S_xx = dudx
    S_yy = dvdy
    S_xy = 0.5 * (dudy + dvdx)

    # Magnitude of strain rate: gamma_dot = sqrt(2 * S_ij * S_ij)
    gamma_dot = np.sqrt(np.maximum(2.0 * (S_xx**2 + S_yy**2 + 2.0 * S_xy**2), 0.0))

    # Hydrodynamic shear stress (Pa)
    shear_stress = mu * gamma_dot

    # Energy dissipation rate epsilon (W/kg)
    epsilon = nu * (gamma_dot**2)

    # Kolmogorov microscale length (m)
    eta_kolmogorov = (nu**3 / (epsilon + 1.0e-12)) ** 0.25

    # Quantitative summary statistics
    tau_max = float(np.max(shear_stress))
    tau_mean = float(np.mean(shear_stress))
    eps_max = float(np.max(epsilon))
    eps_mean = float(np.mean(epsilon))
    eta_min_um = float(np.min(eta_kolmogorov)) * 1.0e6  # convert to micrometers

    # Cell damage risk assessment (typical mammalian threshold: 5.0 - 10.0 Pa)
    if tau_max > 10.0 or eta_min_um < 15.0:
        risk_level = "HIGH (Potential cell lysis in impeller tip zone)"
    elif tau_max > 3.0 or eta_min_um < 30.0:
        risk_level = "MODERATE (Sub-lethal physiological shear, potential metabolic shift)"
    else:
        risk_level = "LOW / SAFE (Optimal growth regime for mammalian & stem cell culture)"

    return {
        "shear_rate": gamma_dot,
        "shear_stress": shear_stress,
        "dissipation_rate": epsilon,
        "kolmogorov_length": eta_kolmogorov,
        "tau_max_pa": tau_max,
        "tau_mean_pa": tau_mean,
        "epsilon_max_w_kg": eps_max,
        "epsilon_mean_w_kg": eps_mean,
        "kolmogorov_min_um": eta_min_um,
        "cell_damage_risk": risk_level,
    }
