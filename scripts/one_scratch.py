#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def load_module_from_file(module_file: str):
    """Import a .py file by path (no 'scripts.' import issues)."""
    module_path = str(Path(module_file).resolve())
    spec = importlib.util.spec_from_file_location("m_loaded", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[attr-defined]
    return m, module_path


def show_constants(module_file: str) -> None:
    m, module_path = load_module_from_file(module_file)
    print(f"\n=== Module constants from {module_path} (ALL_CAPS) ===")
    names = sorted([n for n in dir(m) if n.isupper()])
    for n in names:
        v = getattr(m, n)
        if isinstance(v, (int, float, str, bool, tuple)):
            print(f"  {n} = {v}")


def show_theta(theta_path: str) -> None:
    theta = torch.load(theta_path, map_location="cpu")
    print(f"\n=== theta @ {theta_path} ===")
    print("type:", type(theta))
    if isinstance(theta, torch.Tensor):
        print("shape:", tuple(theta.shape))
        print("dtype:", theta.dtype)
        print("min/max:", float(theta.min()), float(theta.max()))
    else:
        try:
            print("keys:", list(theta.keys())[:30])  # type: ignore[attr-defined]
        except Exception:
            print("value:", theta)


def decode_pv_phys(
    pv: torch.Tensor, P_min: float, P_max: float, v_min: float, v_max: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pv = pv.flatten()
    K = pv.numel() // 2
    if pv.numel() != 2 * K:
        raise ValueError(f"Expected even-length pv, got {pv.numel()}")
    P_raw = pv[:K]
    v_raw = pv[K:]

    def map_tanh_to_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        u = torch.tanh(x)  # [-1, 1]
        return lo + (u + 1.0) * 0.5 * (hi - lo)

    P = map_tanh_to_range(P_raw, P_min, P_max)
    v = map_tanh_to_range(v_raw, v_min, v_max)
    return P_raw, v_raw, P, v


def show_pv_phys(
    pv_path: str,
    P_min: float,
    P_max: float,
    v_min: float,
    v_max: float,
) -> None:
    pv = torch.load(pv_path, map_location="cpu")
    if not isinstance(pv, torch.Tensor):
        raise TypeError("pv file did not load a tensor")
    pv = pv.flatten()

    K = pv.numel() // 2
    print(f"\n=== pv @ {pv_path} ===")
    print("pv numel:", pv.numel(), "K:", K)

    P_raw, v_raw, P, v = decode_pv_phys(pv, P_min, P_max, v_min, v_max)

    print("\nP_raw:", [round(float(x), 6) for x in P_raw])
    print("v_raw:", [round(float(x), 6) for x in v_raw])

    print("\nBounds:")
    print(f"  P_min={P_min}  P_max={P_max}")
    print(f"  v_min={v_min}  v_max={v_max}")

    print("\nP (W):", [round(float(x), 3) for x in P])
    print("v (mm/s):", [round(float(x), 3) for x in v])

    print("\nP stats (W) min/max:", float(P.min()), float(P.max()))
    print("v stats (mm/s) min/max:", float(v.min()), float(v.max()))


def scan_beta_vf(module_file: str, theta_path: str, betas: List[float]) -> None:
    print("\n=== beta scan: eta + theta_phys stats + VF_phys ===")

    m, module_path = load_module_from_file(module_file)

    theta = torch.load(theta_path, map_location="cpu")
    if not isinstance(theta, torch.Tensor):
        raise TypeError("theta file did not load a tensor")
    theta = theta.detach().cpu()

    target_vf = float(getattr(m, "TARGET_VF", 0.25))
    proj = getattr(m, "projection")
    solve_eta_with_beta = getattr(m, "solve_eta_with_beta")

    # IMPORTANT:
    # design_mask must be boolean for indexing. For scratch scan we use full domain mask.
    design_mask = torch.ones_like(theta, dtype=torch.bool)

    print("module:", module_path)
    print("TARGET_VF:", target_vf)
    print("beta list:", [float(b) for b in betas])
    print("design_mask: shape=", tuple(design_mask.shape), "dtype=", design_mask.dtype, "mean=", float(design_mask.float().mean()))

    for beta_req in betas:
        theta_clean = theta.clamp(0.0, 1.0)

        # CRITICAL FIX:
        # solve_eta_with_beta returns 5 values. We must UNPACK them.
        eta, beta_eff, feasible, vf_lo, vf_hi = solve_eta_with_beta(
            theta_clean, float(beta_req), target_vf, design_mask
        )

        theta_phys = proj(theta_clean, float(beta_eff), float(eta))

        vf_raw = float(theta_clean[design_mask].mean().item())
        vf_phys = float(theta_phys[design_mask].mean().item())

        tmin = float(theta_phys.min().item())
        tmax = float(theta_phys.max().item())
        frac_lo = float((theta_phys[design_mask] < 0.05).float().mean().item())
        frac_hi = float((theta_phys[design_mask] > 0.95).float().mean().item())

        print(
            f"\n[beta_req={beta_req:>6.2f}] beta_eff={beta_eff:>6.2f} feasible={feasible} "
            f"vf@lo={vf_lo:.4f} vf@hi={vf_hi:.4f} eta={eta:.6f}"
        )
        print(f"  VF raw ={vf_raw:.4f}")
        print(f"  VF phys={vf_phys:.4f}")
        print(f"  theta_phys min/max={tmin:.4f}/{tmax:.4f}  frac<0.05={frac_lo:.3f}  frac>0.95={frac_hi:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LPBF scratch inspector: constants + theta stats + pv->(P,v) decoding + beta VF scan"
    )
    ap.add_argument("--module-file", default="scripts/smoke_joint_opt_fast.py",
                    help="Path to the python file to introspect for ALL_CAPS constants and eta/projection funcs")
    ap.add_argument("--theta-path", default="scripts/theta_latest.pt",
                    help="Path to theta_latest.pt")
    ap.add_argument("--pv-path", default="scripts/pv_latest.pt",
                    help="Path to pv_latest.pt (packed P_raw then v_raw)")

    ap.add_argument("--show-constants", action="store_true")
    ap.add_argument("--show-theta", action="store_true")
    ap.add_argument("--show-pv-phys", action="store_true")
    ap.add_argument("--scan-beta-vf", action="store_true")

    # default bounds (override if you want)
    ap.add_argument("--P-min", type=float, default=100.0)
    ap.add_argument("--P-max", type=float, default=600.0)
    ap.add_argument("--v-min", type=float, default=400.0)
    ap.add_argument("--v-max", type=float, default=1100.0)

    args = ap.parse_args()

    if args.show_constants:
        show_constants(args.module_file)

    if args.show_theta:
        show_theta(args.theta_path)

    if args.show_pv_phys:
        show_pv_phys(args.pv_path, args.P_min, args.P_max, args.v_min, args.v_max)

    if args.scan_beta_vf:
        betas = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        scan_beta_vf(args.module_file, args.theta_path, betas)


if __name__ == "__main__":
    main()
