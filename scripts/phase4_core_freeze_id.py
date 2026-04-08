#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

EXCLUDE_SUFFIXES = (".png", ".log")
EXCLUDE_BASENAMES = {
    "PHASE4_FREEZE_MANIFEST.json",
    "PHASE4_LOCK_ID.txt",
    "PHASE4_CORE_FREEZE_MANIFEST.json",
    "PHASE4_CORE_FREEZE_ID.txt",
}

# JSON fields we must ignore for deterministic identity (timestamps/paths/etc.)
JSON_DROP_KEYS = {
    "created_unix_s",
    "created_at",
    "timestamp",
    "time",
    "runtime_s",
    "out_dir",
    "baseline_dir",
    "bundle_dir",
    "engine_dir",
    "registry",
    "path",
    "paths",
}

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonicalize_json(obj):
    """
    Recursively remove nondeterministic keys and normalize structure.
    """
    if isinstance(obj, dict):
        out = {}
        for k in sorted(obj.keys()):
            if k in JSON_DROP_KEYS:
                continue
            out[k] = canonicalize_json(obj[k])
        return out
    if isinstance(obj, list):
        return [canonicalize_json(x) for x in obj]
    return obj

def canonical_json_bytes(path: Path) -> bytes:
    obj = json.loads(path.read_text())
    obj = canonicalize_json(obj)
    # Canonical dump: sorted keys, minimal whitespace, stable floats as Python prints them
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")

def is_excluded(path: Path) -> bool:
    if path.name in EXCLUDE_BASENAMES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return False

def collect_core_files(ph4: Path) -> List[Path]:
    """
    Collect only the CORE set:
      - selected subtrees/files
      - exclude PNG/log
      - keep deterministic list
    """
    must_roots = [
        ph4 / "phase1_engine_locked_in718_20260210_204852",
        ph4 / "phase2_diag_locked_in718_20260211_204203",
        ph4 / "phase3_registry_locked_in718_20260212_212457",
        ph4 / "phase3_scoreboard_locked_in718_20260216_201333",
        ph4 / "surrogate_bundle" / "meltpool_v1",
    ]

    files: List[Path] = []
    for root in must_roots:
        if not root.exists():
            raise RuntimeError(f"Missing expected CORE root: {root}")

        for fp in root.rglob("*"):
            if fp.is_file() and not is_excluded(fp):
                files.append(fp)

    # Additional filtering: exclude png/log already handled; keep only key types for core
    # We include: .pt, .py, .json, .joblib (joblib optional but fine), plus any small text if present.
    allowed_suffixes = {".pt", ".py", ".json", ".joblib", ".txt"}
    files = [f for f in files if f.suffix.lower() in allowed_suffixes]

    # Sort for stable ordering
    files.sort(key=lambda p: str(p.relative_to(ph4)))
    return files

def hash_core_file(ph4: Path, fp: Path) -> Tuple[str, str]:
    rel = str(fp.relative_to(ph4))
    if fp.suffix.lower() == ".json":
        h = sha256_bytes(canonical_json_bytes(fp))
        return rel, h
    else:
        return rel, sha256_file(fp)

def main() -> None:
    ap = argparse.ArgumentParser(description="Compute deterministic CORE_FREEZE_ID for Phase-4 bundle (exclude PNG/log; canonicalize JSON).")
    ap.add_argument("--ph4_dir", required=True, type=str, help="Path to Phase-4 freeze directory.")
    args = ap.parse_args()

    ph4 = Path(args.ph4_dir)
    if not ph4.exists():
        raise RuntimeError(f"PH4 dir does not exist: {ph4}")

    core_files = collect_core_files(ph4)

    per_file: Dict[str, str] = {}
    for fp in core_files:
        rel, h = hash_core_file(ph4, fp)
        per_file[rel] = h

    # CORE_FREEZE_ID is hash of the concatenation of "rel\0hash\n" in sorted order
    concat = b""
    for rel in sorted(per_file.keys()):
        concat += rel.encode("utf-8") + b"\0" + per_file[rel].encode("utf-8") + b"\n"
    core_id = sha256_bytes(concat)

    out_manifest = ph4 / "PHASE4_CORE_FREEZE_MANIFEST.json"
    out_id = ph4 / "PHASE4_CORE_FREEZE_ID.txt"

    manifest = {
        "core_freeze_id_sha256": core_id,
        "num_core_files": len(core_files),
        "excluded_suffixes": list(EXCLUDE_SUFFIXES),
        "excluded_basenames": sorted(list(EXCLUDE_BASENAMES)),
        "json_drop_keys": sorted(list(JSON_DROP_KEYS)),
        "core_files_sha256": per_file,
    }

    out_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    out_id.write_text(core_id + "\n")

    print("[OK] Wrote:")
    print(f"  {out_manifest}")
    print(f"  {out_id}")
    print(f"CORE_FREEZE_ID_SHA256: {core_id}")

if __name__ == "__main__":
    main()
