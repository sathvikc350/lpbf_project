#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    if not (repo / ".git").exists():
        return "UNKNOWN"
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "UNKNOWN"


def env_snapshot() -> Dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }


def list_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            files.append(p)
    files.sort(key=lambda x: str(x))
    return files


def safe_read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def fingerprint_scripts(repo: Path) -> Dict[str, str]:
    """
    Fingerprint the exact scripts that define the engine behavior.
    (If these change, your "kernel" changed.)
    """
    scripts = [
        "scripts/smoke_joint_opt_fast.py",
        "scripts/phase2_diagnostics.py",
        "scripts/phase3_probe_surrogates.py",
        "scripts/phase3_build_registry.py",
        "scripts/phase3_scoreboard.py",
        "lpbf_to/surrogates/meltpool_v1/inference.py",
        "lpbf_to/surrogates/meltpool_v1/surrogate_config.json",
        "lpbf_to/surrogates/meltpool_v1/weights.pt",
    ]
    out: Dict[str, str] = {}
    for rel in scripts:
        p = repo / rel
        out[rel] = sha256_file(p) if p.exists() and p.is_file() else "MISSING"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-4 freeze manifest: sha256 every frozen file + script fingerprints.")
    ap.add_argument("--ph4_dir", type=str, required=True, help="Phase-4 freeze directory path.")
    ap.add_argument("--repo", type=str, default="/workspaces/lpbf_project", help="Repo root.")
    args = ap.parse_args()

    ph4 = Path(args.ph4_dir).resolve()
    repo = Path(args.repo).resolve()

    if not ph4.exists():
        raise RuntimeError(f"PH4 dir not found: {ph4}")

    # Hash every file under PH4
    files = list_files(ph4)
    artifacts: List[Tuple[str, str, int]] = []
    for fp in files:
        rel = str(fp.relative_to(ph4))
        h = sha256_file(fp)
        sz = fp.stat().st_size
        artifacts.append((rel, h, int(sz)))

    # Pull a few “identity” fields if present
    reg_path = ph4 / "phase3_registry_locked_in718_20260212_212457" / "material_registry.json"
    registry = safe_read_json(reg_path)
    bundle_dir = registry.get("surrogate_bundle", {}).get("bundle_dir", "")

    surr_cfg_path = ph4 / "surrogate_bundle" / "meltpool_v1" / "surrogate_config.json"
    surr_cfg = safe_read_json(surr_cfg_path)

    manifest = {
        "phase": "Phase-4 Engine Freeze",
        # NOTE: timestamps are useful metadata, but NOT used for determinism checks.
        "created_unix_s": time.time(),
        "ph4_dir": str(ph4),
        "git_commit": git_commit(repo),
        "env": env_snapshot(),

        "identity": {
            "engine_locked_dir": str(ph4 / "phase1_engine_locked_in718_20260210_204852"),
            "phase2_diag_locked_dir": str(ph4 / "phase2_diag_locked_in718_20260211_204203"),
            "phase3_registry_locked_dir": str(ph4 / "phase3_registry_locked_in718_20260212_212457"),
            "phase3_scoreboard_locked_dir": str(ph4 / "phase3_scoreboard_locked_in718_20260216_201333"),
            "surrogate_bundle_dir_in_registry": bundle_dir,
        },

        "surrogate": {
            "material_to_id": surr_cfg.get("material_to_id", {}),
            "outputs": surr_cfg.get("outputs", []),
            "validity_window_global": surr_cfg.get("validity_window_global", {}),
            "validity_window_by_material": surr_cfg.get("validity_window_by_material", {}),
        },

        "script_fingerprints_sha256": fingerprint_scripts(repo),

        # Sorted, deterministic list of every frozen file
        "frozen_files": [
            {"path": rel, "sha256": h, "bytes": sz}
            for (rel, h, sz) in artifacts
        ],
    }

    out_path = ph4 / "PHASE4_FREEZE_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # Also write a tiny one-line “lock id” that can be compared easily
    # (hash of the manifest itself)
    manifest_sha = sha256_file(out_path)
    (ph4 / "PHASE4_LOCK_ID.txt").write_text(manifest_sha + "\n")

    print("[OK] Wrote:")
    print(f"  {out_path}")
    print(f"  {ph4 / 'PHASE4_LOCK_ID.txt'}")
    print("LOCK_ID_SHA256:", manifest_sha)


if __name__ == "__main__":
    main()
