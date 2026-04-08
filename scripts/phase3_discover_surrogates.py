"""
PHASE 3 - STEP 1: SURROGATE DISCOVERY / INVENTORY

What this script does:
- Scans the repo for candidate weight files (*.pt, *.pth) and related config artifacts.
- Writes an inventory JSON listing what was found.

Why it exists:
- Prevents "guessing" where models live.
- Creates traceability: we can prove which files existed at discovery time.

Important note:
- Most *.pt files in this repo are NOT meltpool surrogates (many are theta checkpoints / artifacts),
  so material guessing may return "unknown" for many entries.
- Our actual production meltpool surrogate is the packaged bundle:
  lpbf_to/surrogates/meltpool_v1/{weights.pt, surrogate_config.json, scaler.joblib}
"""






#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path("/workspaces/lpbf_project")

WEIGHT_EXTS = {".pt", ".pth"}
SIDE_EXTS = {".json", ".yml", ".yaml", ".txt"}

MATERIAL_HINTS = {
    "in718": ["in718", "718", "inconel"],
    "ti64": ["ti64", "titan", "titanium", "ti-6", "ti6", "ti_64"],
    "316l": ["316", "316l", "ss316", "stainless"],
}

def guess_material(name: str) -> str:
    n = name.lower()
    for mat, keys in MATERIAL_HINTS.items():
        if any(k in n for k in keys):
            return mat
    return "unknown"

def nearby_sidecars(weight_path: Path, max_depth: int = 2) -> List[str]:
    """
    Look for normalization/schema/config files near the weights:
    - same folder
    - parent folder
    - one level below
    """
    out: List[str] = []
    roots = {weight_path.parent, weight_path.parent.parent}
    for r in list(roots):
        if not r.exists():
            continue
        # same folder + one level below
        cand = list(r.glob("*")) + list(r.glob("*/*"))
        for fp in cand:
            if fp.is_file() and fp.suffix.lower() in SIDE_EXTS:
                out.append(str(fp))
    # also files with same stem next to weight
    for fp in weight_path.parent.glob(weight_path.stem + ".*"):
        if fp.is_file() and fp.suffix.lower() in SIDE_EXTS:
            out.append(str(fp))
    # unique + sorted
    return sorted(list(dict.fromkeys(out)))

def main() -> None:
    ap = argparse.ArgumentParser(description="Discover surrogate weight files and nearby sidecar configs.")
    ap.add_argument("--root", type=str, default=str(REPO), help="Root folder to scan.")
    ap.add_argument("--out", type=str, required=True, help="Output JSON path (inventory).")
    ap.add_argument("--max", type=int, default=200, help="Max number of weight files to record.")
    args = ap.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights: List[Path] = []
    for ext in WEIGHT_EXTS:
        weights.extend(root.rglob(f"*{ext}"))

    # Filter obvious non-model tensors if desired (keep permissive for now)
    # If you want stricter filtering later, we can add rules.
    weights = sorted(weights)[: args.max]

    inventory: Dict[str, Dict] = {}
    for w in weights:
        mat = guess_material(w.name)
        inventory[str(w)] = {
            "material_guess": mat,
            "filename": w.name,
            "size_bytes": w.stat().st_size if w.exists() else None,
            "sidecars_nearby": nearby_sidecars(w),
        }

    payload = {
        "root": str(root),
        "num_weights_found": len(weights),
        "weights": inventory,
        "notes": "This is a discovery inventory only. Probe script will determine which weights are valid surrogates.",
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Wrote inventory: {out_path}")
    print(f"num_weights_found={payload['num_weights_found']}")
    # Quick summary by material guess
    counts: Dict[str, int] = {}
    for v in inventory.values():
        counts[v["material_guess"]] = counts.get(v["material_guess"], 0) + 1
    print("material_guess_counts:", counts)

if __name__ == "__main__":
    main()
