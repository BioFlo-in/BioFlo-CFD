"""
Incompressible Navier-Stokes CFD simulation generator.
Simulates 2D transient fluid flows (Lid-Driven Cavity, Taylor-Green vortex, Shear layer)
using the Chorin projection method, and exports results directly into native OpenFOAM
case directories or numpy/PyTorch format.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

from ..io.foam_writer import export_simulation_state_to_openfoam


class NavierStokes2DSimulator:
    """
    Finite Difference 2D Incompressible Navier-Stokes Solver.
    Governing equations:
        du/dt + (u.grad)u = -grad(p)/rho + nu * laplacian(u)
        div(u) = 0
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        lx: float = 1.0,
        ly: float = 1.0,
        rho: float = 1.0,
        nu: float = 0.01,
        dt: float = 0.001,
    ):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        self.rho = rho
        self.nu = nu
        self.dt = dt

        # State fields: u (horizontal velocity), v (vertical velocity), p (pressure)
        self.u = np.zeros((ny, nx), dtype=np.float32)
        self.v = np.zeros((ny, nx), dtype=np.float32)
        self.p = np.zeros((ny, nx), dtype=np.float32)

    def set_lid_driven_cavity(self, u_lid: float = 1.0):
        """Initializes Lid-Driven Cavity boundary conditions (standard CFD benchmark)."""
        self.u.fill(0.0)
        self.v.fill(0.0)
        self.p.fill(0.0)
        self.u[-1, :] = u_lid  # Top lid moving at u_lid

    def set_taylor_green(self, u0: float = 1.0):
        """Initializes Taylor-Green vortex flow."""
        x = np.linspace(0, 2 * np.pi, self.nx)
        y = np.linspace(0, 2 * np.pi, self.ny)
        X, Y = np.meshgrid(x, y)
        self.u = (u0 * np.sin(X) * np.cos(Y)).astype(np.float32)
        self.v = (-u0 * np.cos(X) * np.sin(Y)).astype(np.float32)
        self.p = (self.rho * u0**2 / 4.0 * (np.cos(2 * X) + np.cos(2 * Y))).astype(np.float32)

    def _solve_pressure_poisson(self, b: np.ndarray, nit: int = 50) -> np.ndarray:
        """Solves Poisson equation for pressure: laplacian(p) = b."""
        pn = np.empty_like(self.p)
        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2.0 * (dx2 + dy2)

        for _ in range(nit):
            pn = self.p.copy()
            self.p[1:-1, 1:-1] = (
                (
                    (pn[1:-1, 2:] + pn[1:-1, :-2]) * dy2
                    + (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx2
                    - b[1:-1, 1:-1] * dx2 * dy2
                )
                / denom
            )

            # Neumann boundary conditions: dp/dn = 0
            self.p[:, -1] = self.p[:, -2]  # Right
            self.p[0, :] = self.p[1, :]   # Bottom
            self.p[:, 0] = self.p[:, 1]   # Left
            self.p[-1, :] = 0.0           # Top pressure reference

        return self.p

    def step(self, is_cavity: bool = True, u_lid: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advances the CFD state by one time step dt using Chorin's projection method.
        """
        u, v, p = self.u, self.v, self.p
        dt, dx, dy = self.dt, self.dx, self.dy
        rho, nu = self.rho, self.nu

        # 1. Compute divergence source term for pressure Poisson equation
        b = np.zeros_like(p)
        du_dx = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        dv_dy = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)

        b[1:-1, 1:-1] = rho * (
            (1.0 / dt) * (du_dx + dv_dy)
            - (du_dx**2 + 2 * (du_dy * dv_dx) + dv_dy**2)
        )

        # 2. Solve Pressure Poisson Equation
        self._solve_pressure_poisson(b)

        # 3. Momentum predictor & projection step
        un = u.copy()
        vn = v.copy()

        # Internal cells update
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * (dt / dx) * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - vn[1:-1, 1:-1] * (dt / dy) * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            - (dt / (2 * rho * dx)) * (self.p[1:-1, 2:] - self.p[1:-1, :-2])
            + nu * (dt / dx**2) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2])
            + nu * (dt / dy**2) * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * (dt / dx) * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            - vn[1:-1, 1:-1] * (dt / dy) * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            - (dt / (2 * rho * dy)) * (self.p[2:, 1:-1] - self.p[:-2, 1:-1])
            + nu * (dt / dx**2) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2])
            + nu * (dt / dy**2) * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])
        )

        # 4. Enforce boundary conditions
        if is_cavity:
            u[0, :] = 0.0        # Bottom wall
            u[:, 0] = 0.0        # Left wall
            u[:, -1] = 0.0       # Right wall
            u[-1, :] = u_lid     # Top moving lid

            v[0, :] = 0.0
            v[-1, :] = 0.0
            v[:, 0] = 0.0
            v[:, -1] = 0.0

        return self.u, self.v, self.p

    def run_simulation(
        self,
        n_steps: int = 100,
        save_interval: int = 1,
        is_cavity: bool = True,
        u_lid: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Runs transient simulation and records time series trajectory.
        Returns:
            {
                'u': (T, ny, nx),
                'v': (T, ny, nx),
                'p': (T, ny, nx),
                'times': (T,)
            }
        """
        u_history = [self.u.copy()]
        v_history = [self.v.copy()]
        p_history = [self.p.copy()]
        times = [0.0]

        for step_i in range(1, n_steps + 1):
            self.step(is_cavity=is_cavity, u_lid=u_lid)
            if step_i % save_interval == 0:
                u_history.append(self.u.copy())
                v_history.append(self.v.copy())
                p_history.append(self.p.copy())
                times.append(step_i * self.dt)

        return {
            "u": np.stack(u_history, axis=0),
            "v": np.stack(v_history, axis=0),
            "p": np.stack(p_history, axis=0),
            "times": np.array(times, dtype=np.float32),
        }


def generate_and_export_openfoam_benchmark(
    output_case_dir: Union[str, Path],
    nx: int = 64,
    ny: int = 64,
    n_steps: int = 100,
    save_interval: int = 5,
    nu: float = 0.01,
    dt: float = 0.001,
    u_lid: float = 1.0,
) -> Path:
    """
    Generates a full CFD trajectory and exports each saved time step directly
    into an OpenFOAM case directory structure (0/, 0.005/, 0.01/, etc.).
    """
    case_path = Path(output_case_dir)
    case_path.mkdir(parents=True, exist_ok=True)

    sim = NavierStokes2DSimulator(nx=nx, ny=ny, nu=nu, dt=dt)
    sim.set_lid_driven_cavity(u_lid=u_lid)

    traj = sim.run_simulation(
        n_steps=n_steps,
        save_interval=save_interval,
        is_cavity=True,
        u_lid=u_lid,
    )

    times = traj["times"]
    n_snapshots = len(times)

    for i in range(n_snapshots):
        t_val = float(times[i])
        u_2d = traj["u"][i]
        v_2d = traj["v"][i]
        p_2d = traj["p"][i]

        # Flatten into OpenFOAM (N_cells, 2) and (N_cells,)
        u_vec = np.column_stack([u_2d.flatten(), v_2d.flatten()])
        p_scalar = p_2d.flatten()

        export_simulation_state_to_openfoam(
            case_dir=case_path,
            time_val=t_val,
            u_pred=u_vec,
            p_pred=p_scalar,
        )

    return case_path
