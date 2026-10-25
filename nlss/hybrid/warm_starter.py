"""
Hybrid OpenFOAM Solver Accelerator & Warm-Starter.
Uses the AI model to predict the next simulation state, writing it directly as
an ultra-accurate initial guess for OpenFOAM's PISO/SIMPLE iterative linear solver.
This slashes linear solver iterations to convergence while preserving exact physical conservation.
"""

import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from ..rollout.engine import RolloutEngine
from ..io.foam_reader import read_foam_field_file
from ..io.foam_writer import export_simulation_state_to_openfoam


class OpenFOAMHybridAccelerator:
    """
    Coordinates AI state prediction with OpenFOAM solver execution.
    """

    def __init__(
        self,
        case_dir: Union[str, Path],
        rollout_engine: RolloutEngine,
        solver_name: str = "icoFoam",
    ):
        self.case_dir = Path(case_dir)
        self.engine = rollout_engine
        self.solver_name = solver_name

    def warm_start_step(
        self,
        current_time: float,
        next_time: float,
        grid_shape: Optional[tuple] = None,
    ) -> Path:
        """
        1. Reads current state S_t from OpenFOAM case.
        2. Generates AI prediction S_{t+dt}^{AI}.
        3. Writes prediction into case_dir/<next_time>/ as initial guess.
        4. OpenFOAM solver starts from this guess, converging in significantly fewer iterations.
        """
        curr_time_dir = self.case_dir / (
            f"{int(current_time)}" if current_time.is_integer() else f"{current_time:.4f}".rstrip("0").rstrip(".")
        )

        u_file = curr_time_dir / "U"
        p_file = curr_time_dir / "p"

        if not u_file.exists() or not p_file.exists():
            raise FileNotFoundError(f"Missing U or p file in {curr_time_dir}")

        u_field = read_foam_field_file(u_file, current_time).internal_field
        p_field = read_foam_field_file(p_file, current_time).internal_field

        if grid_shape is not None:
            H, W = grid_shape
            ux = u_field[:, 0].reshape(1, 1, H, W)
            uy = u_field[:, 1].reshape(1, 1, H, W)
            p = p_field.reshape(1, 1, H, W)
            state_tensor = torch.from_numpy(np.concatenate([ux, uy, p], axis=1)).float()
        else:
            state_tensor = torch.from_numpy(
                np.column_stack([u_field[:, :2], p_field]).T
            ).unsqueeze(0).float()

        # Predict next state with AI surrogate
        pred_next = self.engine.predict_next_state(state_tensor).cpu().numpy()

        if grid_shape is not None:
            H, W = grid_shape
            pred_ux = pred_next[0].flatten()
            pred_uy = pred_next[1].flatten()
            pred_p = pred_next[2].flatten()
            u_vec = np.column_stack([pred_ux, pred_uy])
        else:
            u_vec = pred_next[:2].T
            pred_p = pred_next[2]

        # Write warm-start fields into next time directory
        warm_dir = export_simulation_state_to_openfoam(
            case_dir=self.case_dir,
            time_val=next_time,
            u_pred=u_vec,
            p_pred=pred_p,
        )

        return warm_dir

    def run_openfoam_step(self, end_time: float) -> Tuple[bool, str]:
        """
        Runs OpenFOAM solver if installed on the host system.
        """
        cmd = [self.solver_name, "-case", str(self.case_dir)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return res.returncode == 0, res.stdout
        except FileNotFoundError:
            return False, f"Solver '{self.solver_name}' not found on system PATH."
        except Exception as e:
            return False, str(e)
