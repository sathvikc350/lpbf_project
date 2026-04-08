from __future__ import annotations

from typing import Optional, List, Any

import torch
from dl4to.criteria import Criterion
from dl4to.problem import Problem

from lpbf_to.losses.manufacturing_loss import ManufacturingLoss


class ManufacturingCriterion(Criterion):
    """
    DL4TO Criterion wrapper around ManufacturingLoss.

    Returns: tensor shape (batch_size,) where each element is scalar manufacturing loss.

    Notes:
    - This criterion does NOT optimize P_raw/v_raw; it only evaluates the loss.
    - You MUST call set_process_params(P_raw, v_raw) before using it.
    """

    def __init__(
        self,
        problem: Problem,
        mfg_loss: ManufacturingLoss,
        tile_indices: torch.Tensor,
        material: str = "IN718",
        h_mm: float = 0.11,
        t_mm: float = 0.04,
        d_req_um: float = 80.0,
        d_max_um: float = 200.0,
        name: str = "manufacturing_loss",
    ):
        super().__init__(name=name, supervised=False, compute_only_on_design_space=True)

        self.problem = problem
        self.mfg_loss = mfg_loss

        self.tile_indices = tile_indices
        self.material = material
        self.h_mm = float(h_mm)
        self.t_mm = float(t_mm)
        self.d_req_um = float(d_req_um)
        self.d_max_um = float(d_max_um)

        self.P_raw: Optional[torch.nn.Parameter] = None
        self.v_raw: Optional[torch.nn.Parameter] = None

    @property
    def differentiable(self) -> bool:
        return True

    @property
    def lower_is_better(self) -> bool:
        return True

    def set_process_params(self, P_raw: torch.nn.Parameter, v_raw: torch.nn.Parameter) -> None:
        self.P_raw = P_raw
        self.v_raw = v_raw

    def __call__(
        self,
        solutions: List[Any],
        gt_solutions: Optional[List[Any]] = None,
        binary: bool = False,
    ) -> torch.Tensor:
        if self.P_raw is None or self.v_raw is None:
            raise RuntimeError(
                "ManufacturingCriterion: P_raw/v_raw not set. "
                "Call set_process_params(P_raw, v_raw) before evaluating."
            )

        outs: List[torch.Tensor] = []

        for sol in solutions:
            rho = sol.get_θ(binary=binary) if hasattr(sol, "get_θ") else sol.θ

            # Convert (1, nx, ny, nz) -> (nx, ny, nz) if needed
            if isinstance(rho, torch.Tensor) and rho.dim() == 4 and rho.shape[0] == 1:
                rho = rho[0]

            out = self.mfg_loss(
                rho=rho,
                P_raw=self.P_raw,
                v_raw=self.v_raw,
                tile_indices=self.tile_indices,
                material=self.material,
                h_mm=self.h_mm,
                t_mm=self.t_mm,
                d_req_um=self.d_req_um,
                d_max_um=self.d_max_um,
            )
            outs.append(out["loss_total_mfg"])

        return torch.stack(outs)
