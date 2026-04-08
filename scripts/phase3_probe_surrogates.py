#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_load_cpu(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _as_range(v: Any) -> Optional[Tuple[float, float]]:
    # Accept: [lo, hi], (lo, hi), {"min":lo,"max":hi}, scalar, [x]
    if v is None:
        return None
    if isinstance(v, dict):
        if "min" in v and "max" in v:
            return float(v["min"]), float(v["max"])
        if "lo" in v and "hi" in v:
            return float(v["lo"]), float(v["hi"])
        return None
    if isinstance(v, (list, tuple)):
        if len(v) >= 2:
            return float(v[0]), float(v[1])
        if len(v) == 1:
            x = float(v[0])
            return x, x
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x, x
    return None


def _clip01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def compute_v_ed(P_W: float, v_mm_per_s: float, h_mm: float, t_mm: float) -> float:
    denom = max(v_mm_per_s * h_mm * t_mm, 1e-12)
    return float(P_W / denom)


def compute_l_ed(P_W: float, v_mm_per_s: float) -> float:
    denom = max(v_mm_per_s, 1e-12)
    return float(P_W / denom)


@dataclass
class ProbePoint:
    P_W: float
    v_mm_per_s: float
    h_mm: float
    t_mm: float

    def as_numeric(self, numeric_features: List[str]) -> List[float]:
        out: List[float] = []
        for k in numeric_features:
            if k == "P_W":
                out.append(self.P_W)
            elif k == "v_mm_per_s":
                out.append(self.v_mm_per_s)
            elif k == "h_mm":
                out.append(self.h_mm)
            elif k == "t_mm":
                out.append(self.t_mm)
            elif k == "LED_J_per_mm":
                out.append(compute_l_ed(self.P_W, self.v_mm_per_s))
            elif k == "VED_J_per_mm3":
                out.append(compute_v_ed(self.P_W, self.v_mm_per_s, self.h_mm, self.t_mm))
            else:
                # Unknown feature -> 0 (but record in report later)
                out.append(0.0)
        return out


class MeltPoolSurrogate(nn.Module):
    def __init__(
        self,
        num_materials: int,
        emb_dim: int,
        hidden_sizes: List[int],
        dropout: float,
        num_numeric: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_materials, emb_dim)
        in_dim = num_numeric + emb_dim

        layers: List[nn.Module] = []
        d = in_dim
        for hs in hidden_sizes:
            layers.append(nn.Linear(d, hs))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hs
        layers.append(nn.Linear(d, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num: torch.Tensor, material_id: torch.Tensor) -> torch.Tensor:
        e = self.emb(material_id)
        if e.ndim == 1:
            e = e.unsqueeze(0)
        if x_num.ndim == 1:
            x_num = x_num.unsqueeze(0)
        x = torch.cat([x_num, e], dim=-1)
        return self.mlp(x)


def load_bundle(bundle: Path) -> Tuple[Dict[str, Any], Path, Path]:
    cfg_path = bundle / "surrogate_config.json"
    w_path = bundle / "weights.pt"
    if not cfg_path.exists():
        raise RuntimeError(f"Missing surrogate_config.json in {bundle}")
    if not w_path.exists():
        raise RuntimeError(f"Missing weights.pt in {bundle}")

    cfg = json.loads(cfg_path.read_text())
    return cfg, cfg_path, w_path


def build_model_from_cfg(cfg: Dict[str, Any]) -> MeltPoolSurrogate:
    num_materials = int(cfg.get("num_materials", 0))
    emb_dim = int(cfg.get("emb_dim", 0))
    hidden_sizes = [int(x) for x in cfg.get("hidden_sizes", [])]
    dropout = float(cfg.get("dropout", 0.0))

    numeric_features = cfg.get("numeric_features", cfg.get("numeric_feature_cols", []))
    if not isinstance(numeric_features, list):
        raise RuntimeError("Config numeric_features is not a list.")
    num_numeric = int(len(numeric_features))

    outputs = cfg.get("outputs", cfg.get("target_cols", []))
    if not isinstance(outputs, list):
        raise RuntimeError("Config outputs is not a list.")
    out_dim = int(len(outputs))

    if num_materials <= 0 or emb_dim <= 0 or out_dim <= 0 or num_numeric <= 0:
        raise RuntimeError(
            f"Bad config: num_materials={num_materials}, emb_dim={emb_dim}, "
            f"num_numeric={num_numeric}, out_dim={out_dim}"
        )

    return MeltPoolSurrogate(
        num_materials=num_materials,
        emb_dim=emb_dim,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        num_numeric=num_numeric,
        out_dim=out_dim,
    )


def try_load_state_dict(model: nn.Module, weights_obj: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {"loaded": False, "missing_keys": [], "unexpected_keys": [], "error": None}

    # weights.pt may be:
    # - state_dict directly
    # - {"state_dict": ...}
    # - {"model": ...}
    sd = None
    if isinstance(weights_obj, dict):
        if all(isinstance(k, str) for k in weights_obj.keys()) and any(
            k.startswith("mlp.") or k.startswith("emb.") for k in weights_obj.keys()
        ):
            sd = weights_obj
        elif "state_dict" in weights_obj and isinstance(weights_obj["state_dict"], dict):
            sd = weights_obj["state_dict"]
        elif "model" in weights_obj and isinstance(weights_obj["model"], dict):
            sd = weights_obj["model"]

    if sd is None:
        info["error"] = "Could not locate a state_dict in weights file."
        return info

    try:
        r = model.load_state_dict(sd, strict=False)
        info["loaded"] = True
        info["missing_keys"] = list(getattr(r, "missing_keys", []))
        info["unexpected_keys"] = list(getattr(r, "unexpected_keys", []))
        return info
    except Exception as e:
        info["error"] = repr(e)
        return info


def standardize(x: torch.Tensor, mean: List[float], scale: List[float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=torch.float32)
    s = torch.tensor(scale, dtype=torch.float32)
    s = torch.where(s.abs() < 1e-12, torch.ones_like(s), s)
    return (x - m) / s


def check_validity_windows(
    cfg: Dict[str, Any],
    material: str,
    pt: ProbePoint,
    numeric_features: List[str],
) -> Dict[str, Any]:
    vw_g = cfg.get("validity_window_global", {}) or {}
    vw_m_all = cfg.get("validity_window_by_material", {}) or {}
    vw_m = vw_m_all.get(material, {}) if isinstance(vw_m_all, dict) else {}

    def range_for(name: str) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        rg = _as_range(vw_g.get(name)) if isinstance(vw_g, dict) else None
        rm = _as_range(vw_m.get(name)) if isinstance(vw_m, dict) else None
        return rg, rm

    vals = {
        "P_W": pt.P_W,
        "v_mm_per_s": pt.v_mm_per_s,
        "h_mm": pt.h_mm,
        "t_mm": pt.t_mm,
        "LED_J_per_mm": compute_l_ed(pt.P_W, pt.v_mm_per_s),
        "VED_J_per_mm3": compute_v_ed(pt.P_W, pt.v_mm_per_s, pt.h_mm, pt.t_mm),
    }

    report: Dict[str, Any] = {"material": material, "values": vals, "global": {}, "by_material": {}, "warn": []}

    for k in vals.keys():
        rg, rm = range_for(k)
        x = float(vals[k])

        def eval_one(r: Optional[Tuple[float, float]]) -> Dict[str, Any]:
            if r is None:
                return {"has_range": False}
            lo, hi = float(r[0]), float(r[1])
            inside = (x >= lo) and (x <= hi)
            # "10% outside" warning metric: distance beyond bounds normalized by span
            span = max(hi - lo, 1e-12)
            outside_frac = 0.0
            if x < lo:
                outside_frac = float((lo - x) / span)
            elif x > hi:
                outside_frac = float((x - hi) / span)
            return {"has_range": True, "lo": lo, "hi": hi, "inside": inside, "outside_frac": outside_frac}

        eg = eval_one(rg)
        em = eval_one(rm)
        report["global"][k] = eg
        report["by_material"][k] = em

        if eg.get("has_range") and (not eg.get("inside")) and float(eg.get("outside_frac", 0.0)) > 0.10:
            report["warn"].append(f"TrustBoundary(Global): {k} is >10% outside global validity window.")
        if em.get("has_range") and (not em.get("inside")) and float(em.get("outside_frac", 0.0)) > 0.10:
            report["warn"].append(f"TrustBoundary(Material): {k} is >10% outside {material} validity window.")

    # Extra industrial-grade line: schema checksum of numeric feature order
    schema_str = json.dumps({"numeric_features": numeric_features, "outputs": cfg.get("outputs", [])}, sort_keys=True)
    report["schema_sha256"] = hashlib.sha256(schema_str.encode("utf-8")).hexdigest()

    return report


def monotonic_probe(
    model: MeltPoolSurrogate,
    cfg: Dict[str, Any],
    material: str,
    material_id: int,
    base: ProbePoint,
    numeric_features: List[str],
    mean: List[float],
    scale: List[float],
) -> Dict[str, Any]:
    # Probe: vary P (keep v), vary v (keep P), expect depth & width increase with P, decrease with v (typical).
    # We keep this as WARN-only.
    outputs = cfg.get("outputs", [])
    out_order = list(outputs) if isinstance(outputs, list) else ["out0", "out1"]

    def run(pt: ProbePoint) -> torch.Tensor:
        x = torch.tensor(pt.as_numeric(numeric_features), dtype=torch.float32)
        xz = standardize(x, mean, scale)
        mid = torch.tensor([material_id], dtype=torch.long)
        with torch.no_grad():
            y = model(xz, mid).squeeze(0).float()
        return y

    # Choose 5 points each direction
    P_vals = [base.P_W * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]
    v_vals = [base.v_mm_per_s * f for f in [0.8, 0.9, 1.0, 1.1, 1.2]]

    ys_P = []
    for P in P_vals:
        ys_P.append(run(ProbePoint(P_W=float(P), v_mm_per_s=base.v_mm_per_s, h_mm=base.h_mm, t_mm=base.t_mm)))

    ys_v = []
    for v in v_vals:
        ys_v.append(run(ProbePoint(P_W=base.P_W, v_mm_per_s=float(v), h_mm=base.h_mm, t_mm=base.t_mm)))

    def trend(vals: List[torch.Tensor]) -> List[float]:
        # return list of slopes vs index (finite differences)
        arr = torch.stack(vals, dim=0)  # (N, out_dim)
        dif = arr[1:] - arr[:-1]
        return dif.mean(dim=0).tolist()

    slope_P = trend(ys_P)
    slope_v = trend(ys_v)

    # Output plausibility bounds (hard-ish sanity): width/depth must be finite and >= 0
    def finite_and_nonneg(y: torch.Tensor) -> bool:
        return bool(torch.isfinite(y).all().item() and (y >= -1e-6).all().item())

    ok_all = all(finite_and_nonneg(y) for y in ys_P + ys_v)

    return {
        "material": material,
        "P_vals": [float(x) for x in P_vals],
        "v_vals": [float(x) for x in v_vals],
        "outputs": out_order,
        "avg_slope_vs_P": [float(x) for x in slope_P],
        "avg_slope_vs_v": [float(x) for x in slope_v],
        "finite_nonneg_all": bool(ok_all),
        "warn": [],
    }


def sensibility_check(depths_um: Dict[str, float]) -> Dict[str, Any]:
    # WARN-only. Typical expectation: Ti64 depth >= IN718 at same (P,v) in many windows.
    out: Dict[str, Any] = {"depth_um": depths_um, "warn": []}
    if "Ti64" in depths_um and "IN718" in depths_um:
        if depths_um["Ti64"] + 1e-6 < depths_um["IN718"]:
            out["warn"].append("Sensibility: Ti64 depth < IN718 at fixed (P,v). Check labels/windows.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--P0", type=float, default=200.0)
    ap.add_argument("--v0", type=float, default=800.0)
    ap.add_argument("--h0", type=float, default=0.1)
    ap.add_argument("--t0", type=float, default=0.04)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, cfg_path, w_path = load_bundle(bundle)

    numeric_features = cfg.get("numeric_features", cfg.get("numeric_feature_cols", []))
    if not isinstance(numeric_features, list):
        raise RuntimeError("numeric_features not found or invalid in config.")

    outputs = cfg.get("outputs", cfg.get("target_cols", []))
    if not isinstance(outputs, list):
        raise RuntimeError("outputs not found or invalid in config.")

    material_to_id = cfg.get("material_to_id", {})
    if not isinstance(material_to_id, dict) or len(material_to_id) == 0:
        raise RuntimeError("material_to_id missing/invalid in config.")

    mean = cfg.get("scaler_mean", [])
    scale = cfg.get("scaler_scale", [])
    if not (isinstance(mean, list) and isinstance(scale, list) and len(mean) == len(numeric_features) == len(scale)):
        raise RuntimeError(
            f"Bad scaler: len(mean)={len(mean)}, len(scale)={len(scale)}, len(numeric_features)={len(numeric_features)}"
        )

    model = build_model_from_cfg(cfg)
    weights_obj = torch_load_cpu(w_path)
    load_info = try_load_state_dict(model, weights_obj)

    # HARD FAIL: must load weights
    if not load_info.get("loaded", False):
        raise RuntimeError(f"Failed to load weights: {load_info.get('error')}")

    model.eval()

    base_pt = ProbePoint(P_W=args.P0, v_mm_per_s=args.v0, h_mm=args.h0, t_mm=args.t0)

    results: Dict[str, Any] = {
        "phase": "Phase-3 Probe Surrogates",
        "created_unix_s": time.time(),
        "bundle": str(bundle),
        "config_path": str(cfg_path),
        "weights_path": str(w_path),
        "config_sha256": sha256_file(cfg_path),
        "weights_sha256": sha256_file(w_path),
        "numeric_features": list(numeric_features),
        "outputs": list(outputs),
        "material_to_id": dict(material_to_id),
        "weights_load": load_info,
        "probe_point": {
            "P_W": float(args.P0),
            "v_mm_per_s": float(args.v0),
            "h_mm": float(args.h0),
            "t_mm": float(args.t0),
            "LED_J_per_mm": compute_l_ed(args.P0, args.v0),
            "VED_J_per_mm3": compute_v_ed(args.P0, args.v0, args.h0, args.t0),
        },
        "per_material": {},
        "sensibility": {},
        "hard_fail": False,
        "warnings": [],
    }

    depths_at_base: Dict[str, float] = {}

    for m, mid in material_to_id.items():
        mid_i = int(mid)

        validity = check_validity_windows(cfg, m, base_pt, numeric_features)

        x = torch.tensor(base_pt.as_numeric(numeric_features), dtype=torch.float32)
        xz = standardize(x, mean, scale)
        mid_t = torch.tensor([mid_i], dtype=torch.long)

        with torch.no_grad():
            y = model(xz, mid_t).squeeze(0).float()

        y_list = [float(v) for v in y.tolist()]
        out_map = {outputs[i]: y_list[i] for i in range(len(outputs))}

        # HARD FAIL: non-finite
        if not torch.isfinite(y).all().item():
            results["hard_fail"] = True

        # HARD FAIL: absurd scale (very permissive, just catches explosions)
        # width/depth should not be thousands of mm -> in um, > 1e6 um is suspicious
        for name, val in out_map.items():
            if (not math.isfinite(val)) or (val < -1e-3) or (val > 1e6):
                results["hard_fail"] = True

        probe = monotonic_probe(model, cfg, m, mid_i, base_pt, numeric_features, mean, scale)

        if "depth_um" in out_map:
            depths_at_base[m] = float(out_map["depth_um"])

        results["per_material"][m] = {
            "material_id": mid_i,
            "validity": validity,
            "pred_at_base": out_map,
            "monotonic_probe": probe,
        }

        # Collect warnings
        for w in validity.get("warn", []):
            results["warnings"].append(w)

        # Monotonic warnings (WARN-only)
        # Typical sign: slope_vs_P for depth should be >= 0, slope_vs_v for depth <= 0
        if "depth_um" in outputs:
            idx = outputs.index("depth_um")
            sP = float(probe["avg_slope_vs_P"][idx])
            sv = float(probe["avg_slope_vs_v"][idx])
            if sP < 0:
                results["warnings"].append(f"MonotonicProbe({m}): depth decreases with P on average.")
            if sv > 0:
                results["warnings"].append(f"MonotonicProbe({m}): depth increases with v on average.")

    results["sensibility"] = sensibility_check(depths_at_base)
    results["warnings"].extend(results["sensibility"].get("warn", []))

    # Extra industrial-grade line: a single “contract id” derived from config+weights+schema
    contract_str = json.dumps(
        {
            "config_sha256": results["config_sha256"],
            "weights_sha256": results["weights_sha256"],
            "numeric_features": results["numeric_features"],
            "outputs": results["outputs"],
            "material_to_id": results["material_to_id"],
        },
        sort_keys=True,
    )
    results["SURROGATE_CONTRACT_ID"] = hashlib.sha256(contract_str.encode("utf-8")).hexdigest()

    out_json = out_dir / "probe_results.json"
    out_json.write_text(json.dumps(results, indent=2))

    print("[OK] Wrote:", out_json)
    print("hard_fail:", bool(results["hard_fail"]))
    print("warnings:", len(results["warnings"]))
    print("SURROGATE_CONTRACT_ID:", results["SURROGATE_CONTRACT_ID"][:16] + "...")
    if results["hard_fail"]:
        print("[HARD FAIL] Probe detected a hard failure. Do NOT proceed.")
        sys.exit(2)
    else:
        print("[PASS] Probe completed (warnings may exist).")


if __name__ == "__main__":
    main()
