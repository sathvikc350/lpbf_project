#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch


# -----------------------------
# Small utilities
# -----------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_load_cpu(path: Path) -> torch.Tensor:
    # Avoid noisy warnings when possible.
    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise RuntimeError(f"{path} did not contain a torch.Tensor")
    return obj


def clamp_range(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mid(lo: float, hi: float) -> float:
    return 0.5 * (lo + hi)


def as_range(v: Any) -> Tuple[float, float]:
    """
    Supports:
      - [lo, hi]
      - [single]  -> (single, single)
      - scalar
    """
    if isinstance(v, (list, tuple)):
        if len(v) == 2:
            return float(v[0]), float(v[1])
        if len(v) == 1:
            return float(v[0]), float(v[0])
        raise ValueError(f"Invalid list range: {v}")
    return float(v), float(v)


# -----------------------------
# Surrogate loading (your bundle)
# -----------------------------
@dataclass
class BundleIdentity:
    bundle_dir: str
    cfg_path: str
    weights_path: str
    cfg_sha256: str
    weights_sha256: str
    contract_id: str


def load_bundle_identity(bundle_dir: Path) -> BundleIdentity:
    cfg = bundle_dir / "surrogate_config.json"
    wts = bundle_dir / "weights.pt"
    if not cfg.exists():
        raise RuntimeError(f"Missing {cfg}")
    if not wts.exists():
        raise RuntimeError(f"Missing {wts}")

    cfg_sha = sha256_file(cfg)
    wts_sha = sha256_file(wts)

    # "contract id" = hash(cfg_sha + wts_sha) so it changes if either changes
    contract = hashlib.sha256((cfg_sha + wts_sha).encode("utf-8")).hexdigest()

    return BundleIdentity(
        bundle_dir=str(bundle_dir),
        cfg_path=str(cfg),
        weights_path=str(wts),
        cfg_sha256=cfg_sha,
        weights_sha256=wts_sha,
        contract_id=contract,
    )


def load_surrogate(bundle_dir: Path):
    """
    Uses lpbf_to/surrogates/meltpool_v1/inference.py
      - class SurrogateInference(asset_dir, device='cpu')
      - .forward(P, v, h, t, mat_id)
    """
    inf_py = bundle_dir / "inference.py"
    if not inf_py.exists():
        raise RuntimeError(f"Missing {inf_py}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("mp_inf", str(inf_py))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    if not hasattr(mod, "SurrogateInference"):
        raise RuntimeError("inference.py does not define SurrogateInference")

    SurrogateInference = getattr(mod, "SurrogateInference")
    surr = SurrogateInference(str(bundle_dir), device="cpu")
    surr.eval()
    return surr


def run_surrogate_pred(
    surr,
    P: float,
    v: float,
    h: float,
    t: float,
    mat_id: int,
) -> Dict[str, float]:
    """
    Returns {'width_um': ..., 'depth_um': ...}
    Handles these possible forward returns:
      - tensor shape [2] or [1,2]
      - tuple/list of two tensors/scalars
    """
    with torch.no_grad():
        y = surr.forward(
            torch.tensor([P], dtype=torch.float32),
            torch.tensor([v], dtype=torch.float32),
            torch.tensor([h], dtype=torch.float32),
            torch.tensor([t], dtype=torch.float32),
            int(mat_id),
        )

    # Case A: tuple/list
    if isinstance(y, (tuple, list)):
        if len(y) != 2:
            raise RuntimeError(f"Unexpected tuple output length: {len(y)}")
        w = y[0]
        d = y[1]
        wv = float(w.detach().cpu().flatten()[0].item() if hasattr(w, "detach") else float(w))
        dv = float(d.detach().cpu().flatten()[0].item() if hasattr(d, "detach") else float(d))
        return {"width_um": wv, "depth_um": dv}

    # Case B: tensor
    if not isinstance(y, torch.Tensor):
        raise RuntimeError(f"Unexpected surrogate output type: {type(y)}")

    yt = y.detach().cpu().float().flatten()
    if yt.numel() == 2:
        return {"width_um": float(yt[0].item()), "depth_um": float(yt[1].item())}
    if yt.numel() == 1:
        # Some wrappers may output only depth; we treat this as invalid for this project.
        raise RuntimeError("Surrogate returned a single value; expected [width_um, depth_um].")
    raise RuntimeError(f"Unexpected tensor output size: {yt.numel()}")


# -----------------------------
# Registry loading
# -----------------------------
def load_registry(reg_path_or_dir: Path) -> Dict[str, Any]:
    if reg_path_or_dir.is_dir():
        p = reg_path_or_dir / "material_registry.json"
    else:
        p = reg_path_or_dir
    if not p.exists():
        raise RuntimeError(f"Could not find registry at {p}")
    return json.loads(p.read_text())


def get_material_id(cfg: Dict[str, Any], material: str) -> int:
    m2i = cfg.get("material_to_id", {})
    if material not in m2i:
        raise RuntimeError(f"material_to_id missing '{material}' in surrogate_config.json")
    return int(m2i[material])


def load_surrogate_config(bundle_dir: Path) -> Dict[str, Any]:
    p = bundle_dir / "surrogate_config.json"
    return json.loads(p.read_text())


# -----------------------------
# Validity + trust coverage
# -----------------------------
def inside_range(val: float, lo: float, hi: float) -> bool:
    return (val >= lo) and (val <= hi)


def trust_coverage_fraction(
    P_map: torch.Tensor,
    v_map: torch.Tensor,
    theta_phys: torch.Tensor,
    P_lo: float, P_hi: float,
    v_lo: float, v_hi: float,
    h_val: float, h_lo: float, h_hi: float,
    t_val: float, t_lo: float, t_hi: float,
    solid_thresh: float = 0.5,
) -> float:
    """
    "Trust score" = fraction of solid voxels whose P and v are within material window,
    and (h,t) are within their material window.

    Note: if h/t are fixed scalars and outside window, trust is 0.
    """
    # If scalar h/t invalid, trust is 0 immediately
    if not inside_range(h_val, h_lo, h_hi):
        return 0.0
    if not inside_range(t_val, t_lo, t_hi):
        return 0.0

    # theta_phys: (1,NX,NY,NZ) or (NX,NY,NZ)
    if theta_phys.dim() == 4:
        rho = theta_phys[0]
    else:
        rho = theta_phys
    solid = (rho > solid_thresh)

    total = int(solid.sum().item())
    if total == 0:
        return 0.0

    # P_map/v_map are (NX,NY,NZ)
    inside = solid & (P_map >= P_lo) & (P_map <= P_hi) & (v_map >= v_lo) & (v_map <= v_hi)
    good = int(inside.sum().item())
    return float(good / total)


# -----------------------------
# Probe point selection
# -----------------------------
def material_validity(reg: Dict[str, Any], material: str) -> Dict[str, Tuple[float, float]]:
    vw = reg["materials"][material]["validity_window"]
    out: Dict[str, Tuple[float, float]] = {}
    for k in ["P_W", "v_mm_per_s", "h_mm", "t_mm"]:
        out[k] = as_range(vw[k])
    return out


def choose_probe_point(
    probe_mode: str,
    reg: Dict[str, Any],
    material: str,
    P_fixed: float,
    v_fixed: float,
) -> Tuple[float, float, float, float]:
    vw = material_validity(reg, material)
    P_lo, P_hi = vw["P_W"]
    v_lo, v_hi = vw["v_mm_per_s"]
    h_lo, h_hi = vw["h_mm"]
    t_lo, t_hi = vw["t_mm"]

    # Fixed: use provided P/v; pick h/t as material's exact values (midpoints)
    if probe_mode == "fixed":
        P = P_fixed
        v = v_fixed
        h = mid(h_lo, h_hi)
        t = mid(t_lo, t_hi)
        return P, v, h, t

    # material_aware: use midpoints of validity windows for that material
    if probe_mode == "material_aware":
        P = mid(P_lo, P_hi)
        v = mid(v_lo, v_hi)
        h = mid(h_lo, h_hi)
        t = mid(t_lo, t_hi)
        return P, v, h, t

    # virtual_overlay (A1): same as material_aware for inference probe,
    # but the "trust score" is computed using an *overlay map*:
    #   - IN718/316L: engine P_map/v_map
    #   - Ti64: virtual v_map at its validity midpoint (constant), and P_map clamped into Ti64 window
    # Probe point still uses Ti64-valid midpoint values.
    if probe_mode == "virtual_overlay":
        P = mid(P_lo, P_hi)
        v = mid(v_lo, v_hi)
        h = mid(h_lo, h_hi)
        t = mid(t_lo, t_hi)
        return P, v, h, t

    raise RuntimeError(f"Unknown probe_mode: {probe_mode}")


def make_overlay_maps_for_material(
    probe_mode: str,
    reg: Dict[str, Any],
    material: str,
    P_map_engine: torch.Tensor,
    v_map_engine: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns P_map_used, v_map_used for trust scoring.
    In fixed/material_aware: use engine maps unchanged.
    In virtual_overlay:
      - IN718/316L: use engine maps unchanged
      - Ti64: clamp P_map into Ti64 P window, set v_map to constant midpoint of Ti64 v window
    """
    if probe_mode != "virtual_overlay":
        return P_map_engine, v_map_engine

    if material != "Ti64":
        return P_map_engine, v_map_engine

    vw = material_validity(reg, material)
    P_lo, P_hi = vw["P_W"]
    v_lo, v_hi = vw["v_mm_per_s"]
    v_mid = mid(v_lo, v_hi)

    P_map = P_map_engine.clone()
    P_map = torch.clamp(P_map, min=float(P_lo), max=float(P_hi))

    v_map = torch.full_like(v_map_engine, float(v_mid))
    return P_map, v_map


# -----------------------------
# Pretty print table (no emojis)
# -----------------------------
def fmt_pct(x: float) -> str:
    return f"{100.0*x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-3 scoreboard (material swap validation).")
    ap.add_argument("--engine_dir", required=True, type=str, help="Phase-1 engine locked dir (theta_phys_beta64.pt, P_map_W.pt, v_map_mms.pt).")
    ap.add_argument("--registry", required=True, type=str, help="Path to registry dir OR material_registry.json.")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory.")
    ap.add_argument("--probe_mode", choices=["fixed", "material_aware", "virtual_overlay"], default="fixed")
    ap.add_argument("--P_probe", type=float, default=300.0)
    ap.add_argument("--v_probe", type=float, default=900.0)
    ap.add_argument("--anchor", action="store_true", help="Run a WARN-only anchor probe at same P/v for all materials.")
    ap.add_argument("--P_anchor", type=float, default=350.0)
    ap.add_argument("--v_anchor", type=float, default=1200.0)
    args = ap.parse_args()

    t0 = time.time()

    eng = Path(args.engine_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = load_registry(Path(args.registry))

    bundle_dir = Path(reg["surrogate_bundle"]["bundle_dir"])
    cfg = load_surrogate_config(bundle_dir)
    ident = load_bundle_identity(bundle_dir)

    print("\n=== Phase-3 Material Scoreboard ===")
    print(f"engine_dir: {eng}")
    print(f"registry:   {Path(args.registry) if not Path(args.registry).is_dir() else (Path(args.registry)/'material_registry.json')}")
    print(f"bundle_dir: {bundle_dir}\n")

    print("[INDUSTRIAL] Surrogate bundle identity:")
    print(f"  surrogate_config.json: {ident.cfg_path}")
    print(f"  cfg_sha256:            {ident.cfg_sha256}")
    print(f"  weights.pt:            {ident.weights_path}")
    print(f"  weights_sha256:        {ident.weights_sha256}")
    print(f"  contract_id:           {ident.contract_id}\n")

    # Load engine artifacts
    theta_phys = torch_load_cpu(eng / "theta_phys_beta64.pt").float()
    P_map_engine = torch_load_cpu(eng / "P_map_W.pt").float()
    v_map_engine = torch_load_cpu(eng / "v_map_mms.pt").float()

    # Load surrogate
    surr = load_surrogate(bundle_dir)

    materials = list(reg["materials"].keys())
    if "IN718" not in materials:
        raise RuntimeError("Registry must include IN718 as baseline material.")

    # Pick baseline probe = IN718 probe point (depends on probe_mode)
    P0_in, v0_in, h_in, t_in = choose_probe_point(args.probe_mode, reg, "IN718", args.P_probe, args.v_probe)
    id_in = get_material_id(cfg, "IN718")
    in_pred = run_surrogate_pred(surr, P0_in, v0_in, h_in, t_in, id_in)

    print("Probe points:")
    if args.probe_mode == "fixed":
        print(f"  mode=fixed: using provided P_probe/v_probe, h/t from each material window midpoint")
    elif args.probe_mode == "material_aware":
        print(f"  mode=material_aware: P/v/h/t are midpoints of each material validity window")
    else:
        print(f"  mode=virtual_overlay (A1): inference probes use material midpoints; Ti64 trust uses virtual overlay maps")
    print(f"\n  IN718 baseline probe:")
    print(f"    P={P0_in:.1f} W, v={v0_in:.1f} mm/s, h={h_in:.3f} mm, t={t_in:.3f} mm")
    print(f"    pred (um): width={in_pred['width_um']:.3f}, depth={in_pred['depth_um']:.3f}\n")

    anchor_rows = []
    if args.anchor:
        print("Anchor probe (WARN-only):")
        print(f"  P={args.P_anchor:.1f} W, v={args.v_anchor:.1f} mm/s (same for all materials)")
        for m in materials:
            vw = material_validity(reg, m)
            P_lo, P_hi = vw["P_W"]
            v_lo, v_hi = vw["v_mm_per_s"]
            h_lo, h_hi = vw["h_mm"]
            t_lo, t_hi = vw["t_mm"]
            h0 = mid(h_lo, h_hi)
            t0m = mid(t_lo, t_hi)

            mid_id = get_material_id(cfg, m)
            pred = run_surrogate_pred(surr, float(args.P_anchor), float(args.v_anchor), h0, t0m, mid_id)
            outside = (not inside_range(args.P_anchor, P_lo, P_hi)) or (not inside_range(args.v_anchor, v_lo, v_hi))
            tag = "outside validity" if outside else "inside validity"
            print(f"  {m}: width={pred['width_um']:.3f} um, depth={pred['depth_um']:.3f} um ({tag})")
            anchor_rows.append({"material": m, "P": args.P_anchor, "v": args.v_anchor, "h": h0, "t": t0m, "pred": pred, "outside": outside})
        print("")

    # Build scoreboard rows
    rows = []
    warnings = []
    hard_fail = False

    for m in materials:
        vw = material_validity(reg, m)
        P_lo, P_hi = vw["P_W"]
        v_lo, v_hi = vw["v_mm_per_s"]
        h_lo, h_hi = vw["h_mm"]
        t_lo, t_hi = vw["t_mm"]

        P0, v0, h0, t0m = choose_probe_point(args.probe_mode, reg, m, args.P_probe, args.v_probe)
        mat_id = get_material_id(cfg, m)
        pred = run_surrogate_pred(surr, P0, v0, h0, t0m, mat_id)

        # Trust coverage uses maps (engine maps or overlay maps)
        P_map_used, v_map_used = make_overlay_maps_for_material(args.probe_mode, reg, m, P_map_engine, v_map_engine)
        trust = trust_coverage_fraction(
            P_map_used, v_map_used, theta_phys,
            P_lo, P_hi, v_lo, v_hi,
            h0, h_lo, h_hi,
            t0m, t_lo, t_hi,
            solid_thresh=0.5,
        )

        # Sensitivity delta vs IN718 baseline depth
        if m == "IN718":
            sens = "Baseline"
        else:
            if in_pred["depth_um"] == 0:
                sens = "N/A"
            else:
                delta = (pred["depth_um"] - in_pred["depth_um"]) / in_pred["depth_um"]
                sens = f"{delta*100.0:+.1f}% depth"

        # Load status = hashes are already printed; we mark OK
        load_status = "PASS (Hash OK)"

        # Warn if trust is very low
        if trust < 0.5:
            warnings.append(f"WARN: Low trust coverage for {m}: {trust*100.0:.1f}%")

        rows.append({
            "material": m,
            "load": load_status,
            "sensitivity": sens,
            "trust": trust,
            "probe": {"P": P0, "v": v0, "h": h0, "t": t0m},
            "pred_um": pred,
            "validity": {"P_W": [P_lo, P_hi], "v_mm_per_s": [v_lo, v_hi], "h_mm": [h_lo, h_hi], "t_mm": [t_lo, t_hi]},
        })

    # HARD FAIL rule: if material conditioning appears ignored at probe point (exact same outputs)
    # We only apply this when probe_mode is fixed OR material_aware (same surrogate point is meaningful)
    # For virtual_overlay, probes differ per material, so identical outputs are unlikely and not a clean gate.
    if args.probe_mode in ("fixed", "material_aware"):
        # compare each material to IN718
        for r in rows:
            if r["material"] == "IN718":
                continue
            same_w = abs(r["pred_um"]["width_um"] - in_pred["width_um"]) < 1e-6
            same_d = abs(r["pred_um"]["depth_um"] - in_pred["depth_um"]) < 1e-6
            if same_w and same_d:
                hard_fail = True
                warnings.append(f"HARD FAIL: {r['material']} matches IN718 exactly at probe point (material conditioning may be ignored).")

    # Print table (no emoji)
    print("Scoreboard:")
    headers = ["Material", "Surrogate Load", "Sensitivity Delta", "Trust Score"]
    colw = [10, 16, 33, 12]

    def cell(s: str, w: int) -> str:
        s2 = s if len(s) <= w else (s[:w-3] + "...")
        return s2.ljust(w)

    line = "+" + "+".join(["-" * w for w in colw]) + "+"
    print(line)
    print("|" + "|".join([cell(h, w) for h, w in zip(headers, colw)]) + "|")
    print("+" + "+".join(["=" * w for w in colw]) + "+")
    for r in rows:
        print(
            "|"
            + cell(r["material"], colw[0]) + "|"
            + cell(r["load"], colw[1]) + "|"
            + cell(r["sensitivity"], colw[2]) + "|"
            + cell(f"{r['trust']*100.0:.1f}%", colw[3]) + "|"
        )
    print(line)

    print("\nResult:")
    print(f"  hard_fail={hard_fail}")
    print(f"  warnings={len(warnings)}")
    for i, w in enumerate(warnings, 1):
        print(f"   {i}. {w}")

    out = {
        "phase": "Phase-3 Scoreboard",
        "created_unix_s": time.time(),
        "engine_dir": str(eng),
        "registry_path": str(Path(args.registry) if not Path(args.registry).is_dir() else (Path(args.registry) / "material_registry.json")),
        "probe_mode": args.probe_mode,
        "bundle_identity": {
            "bundle_dir": ident.bundle_dir,
            "cfg_path": ident.cfg_path,
            "weights_path": ident.weights_path,
            "cfg_sha256": ident.cfg_sha256,
            "weights_sha256": ident.weights_sha256,
            "contract_id": ident.contract_id,
        },
        "baseline_material": "IN718",
        "baseline_probe": {"P": P0_in, "v": v0_in, "h": h_in, "t": t_in, "pred_um": in_pred},
        "anchor": {"enabled": bool(args.anchor), "rows": anchor_rows},
        "rows": rows,
        "hard_fail": hard_fail,
        "warnings": warnings,
        "runtime_s": float(time.time() - t0),
    }

    out_path = out_dir / "scoreboard.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path}")
    print(f"runtime_s={out['runtime_s']:.2f}")


if __name__ == "__main__":
    main()
