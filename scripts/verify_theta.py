import scipy.sparse.linalg._dsolve.linsolve as _linsolve
_linsolve.useUmfpack.u = False  # prevent UMFPACK thread-local crash on dl4to import

# =========================
# Imports
# =========================
import argparse
import os
import torch

from typing import Optional, Any, List

from dl4to.pde import FDM
from dl4to.problem import Problem
from dl4to.solution import Solution
from dl4to.criteria import Criterion, Compliance

from lpbf_to.losses.manufacturing_loss import ManufacturingLoss, Bounds



# =========================
# Masked VF (design-space only)
# =========================
class MaskedVolumeFraction(Criterion):
    """VF over designable voxels only: Ω_design == -1."""
    def __init__(self, problem: Problem, name: str = "masked_vf_design"):
        super().__init__(name=name, supervised=False, compute_only_on_design_space=True)
        self.problem = problem

    @property
    def differentiable(self) -> bool:
        return True

    @property
    def lower_is_better(self) -> bool:
        return True

    def __call__(
        self,
        solutions: List[Any],
        gt_solutions: Optional[List[Any]] = None,
        binary: bool = False,
    ) -> torch.Tensor:
        Omega = self.problem.Ω_design.to(self.problem.device)
        design_mask = (Omega == -1.0)
        outs: List[torch.Tensor] = []
        for sol in solutions:
            theta = sol.get_θ(binary=binary) if hasattr(sol, "get_θ") else sol.θ
            outs.append(theta[design_mask].mean())

        return torch.stack(outs)


def _fix_compliance_alpha(C: Compliance) -> str:
    """Some dl4to versions store greek alpha as attribute 'α'."""
    alpha_attr = "\u03b1"  # 'α'
    if hasattr(C, alpha_attr):
        try:
            val = getattr(C, alpha_attr)
            if not isinstance(val, (int, float)) and not torch.is_tensor(val):
                setattr(C, alpha_attr, 1.0)
        except Exception:
            setattr(C, alpha_attr, 1.0)
    return alpha_attr


def _build_problem(device: str, dtype: torch.dtype, nx: int, ny: int, nz: int) -> Problem:
    # Dirichlet: fix x=0 face (all 3 displacement components)
    Ω_dirichlet = torch.zeros((3, nx, ny, nz), dtype=torch.bool, device=device)
    Ω_dirichlet[:, 0, :, :] = True

    # DL4TO convention:
    # Ω_design == -1 -> designable
    # Ω_design ==  1 -> forced solid
    # Ω_design ==  0 -> forced void
    Ω_design = -torch.ones((1, nx, ny, nz), dtype=dtype, device=device)
    Ω_design[:, 0, :, :] = 1.0  # forced solid on x=0 face

    # Force: downward on x=max face, small patch
    F = torch.zeros((3, nx, ny, nz), dtype=dtype, device=device)
    F[1, -1, ny // 2 - 1 : ny // 2 + 1, nz // 2 - 1 : nz // 2 + 1] = -1.0

    problem = Problem(
        E=1.0,
        ν=0.3,
        σ_ys=1.0,
        h=1.0,
        Ω_dirichlet=Ω_dirichlet,
        Ω_design=Ω_design,
        F=F,
        pde_solver=FDM(),
        device=device,
        dtype=dtype,
        restrict_density_for_voxels_with_applied_forces=False,
    )
    return problem


def _make_tile_indices(nx: int, ny: int, nz: int, device: str) -> torch.Tensor:
    # 2x2x2 tiling -> K=8
    tile_indices = torch.empty((nx, ny, nz), dtype=torch.int64, device=device)
    for ix in range(nx):
        tx = 0 if ix < nx // 2 else 1
        for iy in range(ny):
            ty = 0 if iy < ny // 2 else 1
            for iz in range(nz):
                tz = 0 if iz < nz // 2 else 1
                tile_indices[ix, iy, iz] = tx + 2 * ty + 4 * tz
    return tile_indices


def _print_theta_stats(tag: str, theta: torch.Tensor, Omega: torch.Tensor):
    # theta is (1,nx,ny,nz); Omega is (1,nx,ny,nz)
    design = (Omega == -1.0)
    forced_solid = (Omega == 1.0)
    forced_void = (Omega == 0.0)

    def _safe_mean(x):
        return float(x.mean()) if x.numel() > 0 else None

    print(f"\n[{tag}] θ overall min/mean/max: {float(theta.min()):.6f} {float(theta.mean()):.6f} {float(theta.max()):.6f}")
    print(f"[{tag}] counts design/solid/void: {int(design.sum())} {int(forced_solid.sum())} {int(forced_void.sum())}")
    print(f"[{tag}] θ mean in design     : {_safe_mean(theta[design])}")
    print(f"[{tag}] θ mean in forced_solid: {_safe_mean(theta[forced_solid])}")
    print(f"[{tag}] θ mean in forced_void : {_safe_mean(theta[forced_void])}")

    # Hard checks
    solid_vals = theta[forced_solid]
    if solid_vals.numel() > 0:
        print(f"[{tag}] forced_solid min/max: {float(solid_vals.min()):.6f} / {float(solid_vals.max()):.6f}")


def _compliance_value(C: Compliance, sol: Solution) -> float:
    with torch.no_grad():
        return float(C([sol]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta_path", type=str, default="/workspaces/lpbf_project/scripts/theta_latest.pt")
    ap.add_argument("--nx", type=int, default=32)
    ap.add_argument("--ny", type=int, default=32)
    ap.add_argument("--nz", type=int, default=16)
    ap.add_argument("--material", type=str, default="IN718")
    ap.add_argument("--h_mm", type=float, default=0.11)
    ap.add_argument("--t_mm", type=float, default=0.04)
    ap.add_argument("--d_req_um", type=float, default=80.0)
    ap.add_argument("--d_max_um", type=float, default=200.0)
    ap.add_argument("--binary_thresholds", type=float, nargs="+", default=[0.5])
    args = ap.parse_args()

    device = "cpu"
    dtype = torch.float32

    if not os.path.exists(args.theta_path):
        raise FileNotFoundError(f"theta_path not found: {args.theta_path}")

    theta_loaded = torch.load(args.theta_path, map_location="cpu")
    if not isinstance(theta_loaded, torch.Tensor):
        raise TypeError(f"Expected tensor in {args.theta_path}, got: {type(theta_loaded)}")

    # Build Problem
    problem = _build_problem(device=device, dtype=dtype, nx=args.nx, ny=args.ny, nz=args.nz)
    Omega = problem.Ω_design

    expected_shape = (1, *problem.shape)
    if tuple(theta_loaded.shape) != expected_shape:
        raise ValueError(f"Loaded theta shape {tuple(theta_loaded.shape)} != expected {expected_shape}")

    theta0 = theta_loaded.to(device=device, dtype=dtype)

    # Build Solution (continuous)
    sol_cont = Solution(problem=problem, θ=theta0, enforce_θ_on_Ω_design=True)

    # Criteria
    VFm = MaskedVolumeFraction(problem)
    C = Compliance(problem)  # type: ignore[arg-type]

    alpha_attr = _fix_compliance_alpha(C)
    if hasattr(C, alpha_attr):
        print("[CHECK] Compliance alpha attr present; type/value:", type(getattr(C, alpha_attr)), getattr(C, alpha_attr))

    # --- VF checks (continuous)
    design_mask = (Omega == -1.0)
    vf_manual = float(sol_cont.θ[design_mask].mean())
    vf_masked = float(VFm([sol_cont]).mean())

    print(f"\n[LOAD] theta: {args.theta_path}")
    print(f"[VF] designable voxels: {int(design_mask.sum())} / {Omega.numel()}")
    print(f"[VF] manual VF_design : {vf_manual:.6f}")
    print(f"[VF] masked VF_design : {vf_masked:.6f}")

    _print_theta_stats("CONTINUOUS", sol_cont.θ.detach(), Omega)

    # --- Compliance (continuous)
    comp_cont = _compliance_value(C, sol_cont)
    print(f"[COMP] continuous compliance: {comp_cont:.6f}")

    # =========================
    # Binary checks (threshold(s))
    # =========================
    for thr in args.binary_thresholds:
        theta_bin = (sol_cont.θ.detach() >= float(thr)).to(dtype)
        sol_bin = Solution(problem=problem, θ=theta_bin, enforce_θ_on_Ω_design=True)

        vf_bin = float(VFm([sol_bin]).mean())
        comp_bin = _compliance_value(C, sol_bin)

        _print_theta_stats(f"BINARY@{thr:g}", sol_bin.θ.detach(), Omega)
        print(f"[VF] binary@{thr:g} VF_design : {vf_bin:.6f}")
        print(f"[COMP] binary@{thr:g} compliance: {comp_bin:.6f}")

    # =========================
    # ManufacturingLoss evaluation on final θ (no optimization)
    # =========================
    asset_dir = "/workspaces/lpbf_project/lpbf_to/surrogates/meltpool_v1"
    tile_shape = (2, 2, 2)
    K = tile_shape[0] * tile_shape[1] * tile_shape[2]

    mfg = ManufacturingLoss(
        asset_dir=asset_dir,
        tile_shape=tile_shape,
        bounds_by_material={args.material: Bounds(P_min=100, P_max=600, v_min=400, v_max=1100)},
        tv_weight=0.05,
        rho_power=4.0,
    )

    tile_indices = _make_tile_indices(args.nx, args.ny, args.nz, device=device)

    # Use default raw=0 => midpoint mapping for bounded params in your implementation
    P_raw = torch.zeros(K, dtype=dtype, device=device)
    v_raw = torch.zeros(K, dtype=dtype, device=device)

    with torch.no_grad():
        mout = mfg(
            rho=sol_cont.θ.detach().squeeze(0),  # ManufacturingLoss expects (nx,ny,nz)
            P_raw=P_raw,
            v_raw=v_raw,
            tile_indices=tile_indices,
            material=args.material,
            h_mm=float(args.h_mm),
            t_mm=float(args.t_mm),
            d_req_um=float(args.d_req_um),
            d_max_um=float(args.d_max_um),
        )

    print("\n[MFG] (evaluated on CONTINUOUS θ, with default P_raw/v_raw=0)")
    print("[MFG] loss_total_mfg:", float(mout["loss_total_mfg"]))
    print("[MFG] P_map_stats   :", mout["P_map_stats"].tolist())
    print("[MFG] v_map_stats   :", mout["v_map_stats"].tolist())
    print("[MFG] depth_stats   :", mout["depth_stats"].tolist())
    if "risk_stats" in mout:
        print("[MFG] risk_stats    :", mout["risk_stats"].tolist())

    # Final basic sanity
    if torch.isnan(sol_cont.θ).any():
        raise RuntimeError("NaNs detected in theta (continuous).")
    print("\n[OK] Verification completed.")


if __name__ == "__main__":
    main()
