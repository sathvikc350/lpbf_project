# --- SciPy guard: prevent UMFPACK thread-local crash on import ---
import scipy.sparse.linalg._dsolve.linsolve as _linsolve
_linsolve.useUmfpack.u = False

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Any, List

from dl4to.pde import FDM
from dl4to.problem import Problem
from dl4to.solution import Solution
from dl4to.topo_solvers import SIMP
from dl4to.criteria import (
    Compliance,
    WeightedCriterion,
    CombinedCriterion,
)

from lpbf_to.losses.manufacturing_loss import ManufacturingLoss, Bounds


# Masked VF/VC )

class MaskedVolumeFraction(nn.Module):
    """
    Volume fraction computed ONLY on designable voxels (Ω_design == -1).
    DL4TO expects some Criterion-like attributes; we provide them.
    """
    def __init__(self, compute_only_on_design_space: bool = True, eps: float = 1e-12):
        super().__init__()
        self.compute_only_on_design_space = bool(compute_only_on_design_space)
        self.eps = eps

        self.name = "masked_volume_fraction_design"
        self.lower_is_better = False
        self.supervised = False
        self.differentiable = True

    def __call__(
        self,
        solutions: List[Any],
        gt_solutions: Optional[List[Any]] = None,
        binary: bool = False,
    ) -> torch.Tensor:
        vals: List[torch.Tensor] = []
        for sol in solutions:
            theta = sol.get_θ(binary=binary) if hasattr(sol, "get_θ") else sol.θ
            Omega = sol.problem.Ω_design.to(theta.device)  # (1,X,Y,Z)

            mask = (Omega == -1.0)
            if not mask.any():
                vals.append(theta.new_tensor(0.0))
            else:
                vals.append(theta[mask].mean())

        return torch.stack(vals)



class MaskedVolumeConstraint(nn.Module):
    """
    Penalty on excess VF over target, computed ONLY on design voxels.
    penalty = softplus(vf - target)  (or relu / square_relu)
    """
    def __init__(
        self,
        max_volume_fraction: float = 0.25,
        threshold_fct: str = "softplus",
        compute_only_on_design_space: bool = True,
    ):
        super().__init__()
        self.max_volume_fraction = float(max_volume_fraction)
        self.threshold_fct = threshold_fct
        self.compute_only_on_design_space = bool(compute_only_on_design_space)

        self.name = f"masked_volume_constraint_design_{self.max_volume_fraction}"
        self.lower_is_better = True
        self.supervised = False
        self.differentiable = True

        self.vf = MaskedVolumeFraction(compute_only_on_design_space=True)

    def _threshold(self, x: torch.Tensor) -> torch.Tensor:
        if self.threshold_fct == "softplus":
            return F.softplus(x)
        if self.threshold_fct == "relu":
            return F.relu(x)
        if self.threshold_fct == "square_relu":
            return F.relu(x) ** 2
        raise ValueError(f"Unknown threshold_fct='{self.threshold_fct}'")

    def __call__(
        self,
        solutions: List[Any],
        gt_solutions: Optional[List[Any]] = None,
        binary: bool = False,
    ) -> torch.Tensor:
    # gt_solutions is unused here; keep signature for compatibility
        vf = self.vf(solutions, gt_solutions, binary)
        return self._threshold(vf - self.max_volume_fraction)



def main():
    device = "cpu"
    dtype = torch.float32

    # ManufacturingLoss 
    
    asset_dir = "/workspaces/lpbf_project/lpbf_to/surrogates/meltpool_v1"
    tile_shape = (2, 2, 2)
    K = tile_shape[0] * tile_shape[1] * tile_shape[2]

    mfg = ManufacturingLoss(
        asset_dir=asset_dir,
        tile_shape=tile_shape,
        bounds_by_material={"IN718": Bounds(P_min=100, P_max=600, v_min=400, v_max=1100)},
        tv_weight=0.05,
        rho_power=4.0,
    )

    # Grid
    nx, ny, nz = 32, 32, 16

    # tile_indices for 2x2x2 tiling
    tile_indices = torch.empty((nx, ny, nz), dtype=torch.int64, device=device)
    for ix in range(nx):
        tx = 0 if ix < nx // 2 else 1
        for iy in range(ny):
            ty = 0 if iy < ny // 2 else 1
            for iz in range(nz):
                tz = 0 if iz < nz // 2 else 1
                tile_indices[ix, iy, iz] = tx + 2 * ty + 4 * tz

    rho_uniform = torch.full((nx, ny, nz), 0.5, dtype=dtype, device=device)
    P_raw = torch.nn.Parameter(torch.zeros(K, dtype=dtype, device=device))
    v_raw = torch.nn.Parameter(torch.zeros(K, dtype=dtype, device=device))

    out = mfg(
        rho=rho_uniform,
        P_raw=P_raw,
        v_raw=v_raw,
        tile_indices=tile_indices,
        material="IN718",
        h_mm=0.11,
        t_mm=0.04,
        d_req_um=80.0,
        d_max_um=200.0,
    )
    print("[MFG] loss_total_mfg:", float(out["loss_total_mfg"]))
    print("[MFG] P_map_stats:", out["P_map_stats"].tolist())
    print("[MFG] v_map_stats:", out["v_map_stats"].tolist())
    print("[MFG] depth_stats:", out["depth_stats"].tolist())

    
    Ω_dirichlet = torch.zeros((3, nx, ny, nz), dtype=torch.bool, device=device)
    Ω_dirichlet[:, 0, :, :] = True

    Ω_design = -torch.ones((1, nx, ny, nz), dtype=dtype, device=device)
    Ω_design[:, 0, :, :] = 1.0  # force solid on x=0 face

    Fext = torch.zeros((3, nx, ny, nz), dtype=dtype, device=device)
    Fext[1, -1, ny // 2 - 1 : ny // 2 + 1, nz // 2 - 1 : nz // 2 + 1] = -1.0

    pde_solver = FDM()

    problem = Problem(
        E=1.0,
        ν=0.3,
        σ_ys=1.0,
        h=1.0,
        Ω_dirichlet=Ω_dirichlet,
        Ω_design=Ω_design,
        F=Fext,
        pde_solver=pde_solver,
        device=device,
        dtype=dtype,
        restrict_density_for_voxels_with_applied_forces=False,
    )

    theta_init = torch.full((1, *problem.shape), 0.25, device=device, dtype=dtype)
    sol_init = Solution(problem=problem, θ=theta_init, enforce_θ_on_Ω_design=True)

  
    # Criteria
    
    C = Compliance(problem)  # type: ignore[arg-type]


    alpha_attr = "\u03b1"  # 'α'
    if hasattr(C, alpha_attr):
        try:
            val = getattr(C, alpha_attr)
            if not isinstance(val, (int, float)) and not torch.is_tensor(val):
                setattr(C, alpha_attr, 1.0)
        except Exception:
            setattr(C, alpha_attr, 1.0)

    VF = MaskedVolumeFraction(compute_only_on_design_space=True)
    VC = MaskedVolumeConstraint(
        max_volume_fraction=0.25,
        threshold_fct="square_relu",
        compute_only_on_design_space=True,
    )


  
    # Runtime checks (pre-SIMP)
   
    Omega = problem.Ω_design
    mask = (Omega == -1.0)
    print("[VF CHECK] designable voxels:", int(mask.sum()), "/", int(Omega.numel()))
    print("[VF CHECK] manual VF_design:", float(sol_init.θ[mask].mean()) if mask.any() else None)
    print("[VF CHECK] masked VF_design :", float(VF([sol_init]).mean()))
    print("[VF CHECK] masked VC_design :", float(VC([sol_init]).mean()))
    if hasattr(C, alpha_attr):
        print("[CHECK] type(C.α) =", type(getattr(C, alpha_attr)), "| value =", getattr(C, alpha_attr))
    print("[CHECK] Compliance(sol_init):", float(C([sol_init]).mean()))

    
    # SIMP 
    
    for w_vol in [20.0, 50.0, 100.0, 200.0, 500.0]:
        J = CombinedCriterion(WeightedCriterion(C, 1.0), WeightedCriterion(VC, w_vol))
        print(f"\n[CRIT] Testing w_vol = {w_vol}")
        print("[CRIT] Built J:", J.name)
        print("[CRIT] J.lower_is_better:", J.lower_is_better)

        simp = SIMP(
            criterion=J,
            n_iterations=80,
            verbose=True,
            return_intermediate_solutions=False,
        )

        sol_out = simp([sol_init], eval_mode=True)
        sol_out = sol_out[0] if isinstance(sol_out, list) else sol_out

        theta_out = sol_out.θ.detach()
        vf_after = float(VF([sol_out]).mean())
        vc_after = float(VC([sol_out]).mean())
        comp_after = float(C([sol_out]).mean())

        print("[SIMP] theta_out min/mean/max:",
              float(theta_out.min()), float(theta_out.mean()), float(theta_out.max()))
        print(f"[SIMP] VF_design after: {vf_after:.4f}")
        print(f"[SIMP] VC_design after: {vc_after:.4f}")
        print(f"[SIMP] Compliance after: {comp_after:.6f}")

    gc.collect()


if __name__ == "__main__":
    main()
