#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_range(v: Any, key: str) -> Tuple[float, float]:
    """
    Accepts:
      - [lo, hi]
      - [x]  (fixed)
      - x    (fixed)
    Returns (lo, hi)
    """
    if isinstance(v, (int, float)):
        x = float(v)
        return x, x

    if isinstance(v, (list, tuple)):
        if len(v) == 2:
            return float(v[0]), float(v[1])
        if len(v) == 1:
            x = float(v[0])
            return x, x

    raise RuntimeError(f"Invalid or missing range for '{key}': {v!r}")


def get_range(vw: Dict[str, Any], key: str) -> Tuple[float, float]:
    if key not in vw:
        raise RuntimeError(f"Missing '{key}' in validity window.")
    return _as_range(vw[key], key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: build material_registry.json from meltpool_v1 bundle + Phase-2 diag + engine lock.")
    ap.add_argument("--bundle", type=str, required=True, help="Path to meltpool_v1 bundle folder.")
    ap.add_argument("--out", type=str, required=True, help="Output path for material_registry.json")
    ap.add_argument("--engine_lock_id", type=str, required=True, help="Engine lock SHA-256 from Phase 1.")
    ap.add_argument("--phase2_diag_dir", type=str, required=True, help="Locked Phase-2 diagnostics folder (for thresholds/constants).")
    args = ap.parse_args()

    bundle = Path(args.bundle)
    cfg_path = bundle / "surrogate_config.json"
    weights_path = bundle / "weights.pt"
    scaler_path = bundle / "scaler.joblib"

    if not cfg_path.exists():
        raise RuntimeError(f"Missing surrogate_config.json in bundle: {cfg_path}")
    if not weights_path.exists():
        raise RuntimeError(f"Missing weights.pt in bundle: {weights_path}")

    cfg = json.loads(cfg_path.read_text())

    vw_global = cfg.get("validity_window_global", {})
    vw_by_mat = cfg.get("validity_window_by_material", {})
    material_to_id = cfg.get("material_to_id", {})
    numeric_features = cfg.get("numeric_features", cfg.get("numeric_feature_cols", []))
    outputs = cfg.get("outputs", cfg.get("target_cols", []))
    units = cfg.get("units", {})

    if not vw_by_mat or not material_to_id:
        raise RuntimeError("Config missing validity_window_by_material or material_to_id.")

    # Load Phase-2 manifest so registry records the diagnostic constants we already locked
    phase2_dir = Path(args.phase2_diag_dir)
    man_path = phase2_dir / "manifest.json"
    if not man_path.exists():
        raise RuntimeError(f"Missing Phase-2 manifest.json at: {man_path}")
    man = json.loads(man_path.read_text())

    diag_thresholds = man.get("DIAGNOSTIC_THRESHOLDS", {})
    assumed_process = man.get("ASSUMED_PROCESS_CONSTANTS", {})
    conventions_locked = man.get("conventions_locked", {})

    # Build per-material registry entries
    materials: Dict[str, Any] = {}
    for m in material_to_id.keys():
        vw_m = vw_by_mat.get(m, {})

        P_lo, P_hi = get_range(vw_m if "P_W" in vw_m else vw_global, "P_W")
        v_lo, v_hi = get_range(vw_m if "v_mm_per_s" in vw_m else vw_global, "v_mm_per_s")
        h_lo, h_hi = get_range(vw_m if "h_mm" in vw_m else vw_global, "h_mm")
        t_lo, t_hi = get_range(vw_m if "t_mm" in vw_m else vw_global, "t_mm")

        # Optional windows if present
        ved_lo, ved_hi = (None, None)
        if "VED_J_per_mm3" in (vw_m or vw_global):
            ved_lo, ved_hi = get_range(vw_m if "VED_J_per_mm3" in vw_m else vw_global, "VED_J_per_mm3")

        led_lo, led_hi = (None, None)
        if "LED_J_per_mm" in (vw_m or vw_global):
            led_lo, led_hi = get_range(vw_m if "LED_J_per_mm" in vw_m else vw_global, "LED_J_per_mm")

        materials[m] = {
            "material_id": int(material_to_id[m]),
            "validity_window": {
                "P_W": [P_lo, P_hi],
                "v_mm_per_s": [v_lo, v_hi],
                "h_mm": [h_lo, h_hi],
                "t_mm": [t_lo, t_hi],
                "VED_J_per_mm3": None if ved_lo is None else [ved_lo, ved_hi],
                "LED_J_per_mm": None if led_lo is None else [led_lo, led_hi],
            },
            "process_health": {
                "target_ved_range_J_per_mm3": None,  # placeholder: fill later per material
                "target_led_range_J_per_mm": None,   # placeholder: fill later per material
            },
            "thermal_history": {
                "enabled": False,
                "note": "Placeholder for Phase-5/6: track local re-melts / revisit counts along scan strategy.",
            },
        }

    # Industrial line: compute a stable “bundle fingerprint” (config + weights + scaler if present)
    bundle_fingerprint = {
        "surrogate_config.json": sha256_file(cfg_path),
        "weights.pt": sha256_file(weights_path),
        "scaler.joblib": sha256_file(scaler_path) if scaler_path.exists() else "MISSING",
    }
    surrogate_contract_id = hashlib.sha256(
        (bundle_fingerprint["surrogate_config.json"] + bundle_fingerprint["weights.pt"]).encode("utf-8")
    ).hexdigest()

    registry: Dict[str, Any] = {
        "phase": "Phase-3 Material Registry",
        "created_unix_s": time.time(),
        "engine_lock_id": args.engine_lock_id,
        "phase2_diag_dir": str(phase2_dir),
        "surrogate_bundle": {
            "bundle_dir": str(bundle),
            "surrogate_contract_id": surrogate_contract_id,
            "fingerprint_sha256": bundle_fingerprint,
            "numeric_features": numeric_features,
            "outputs": outputs,
            "units": units,
        },
        "conventions_locked": conventions_locked,
        "diagnostic_thresholds_from_phase2": diag_thresholds,
        "assumed_process_constants_from_phase2": assumed_process,
        "materials": materials,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(registry, indent=2))

    print("[OK] Wrote registry:", str(out_path))
    print("SURROGATE_CONTRACT_ID:", surrogate_contract_id[:16] + "...")
    print("materials:", list(materials.keys()))


if __name__ == "__main__":
    main()
