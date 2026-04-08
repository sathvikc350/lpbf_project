# backend/app/runs_store.py
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from backend.app.db import connect


@dataclass
class RunPaths:
    run_dir: Path
    inputs_dir: Path
    checkpoints_dir: Path
    exports_dir: Path
    reports_dir: Path
    logs_dir: Path


def runs_root() -> Path:
    root = Path("/workspaces/lpbf_project/artifacts/runs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def make_run_paths(run_id: str) -> RunPaths:
    rd = runs_root() / run_id
    p = RunPaths(
        run_dir=rd,
        inputs_dir=rd / "inputs",
        checkpoints_dir=rd / "checkpoints",
        exports_dir=rd / "exports",
        reports_dir=rd / "reports",
        logs_dir=rd / "logs",
    )
    for d in [p.run_dir, p.inputs_dir, p.checkpoints_dir, p.exports_dir, p.reports_dir, p.logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return p


def new_run_id() -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2))


def write_checkpoint(
    *,
    run_id: str,
    run_dir: Path,
    it: int,
    theta_phys: torch.Tensor,
) -> Dict[str, Any]:
    """
    Writes BOTH:
      - checkpoints/checkpoint_iter_XXXXXX.json  (metadata)
      - checkpoints/theta_phys_iter_XXXXXX.pt    (actual tensor snapshot)

    This is the minimum needed for a true "resume mid-run" later.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta_path = ckpt_dir / f"checkpoint_iter_{it:06d}.json"
    theta_path = ckpt_dir / f"theta_phys_iter_{it:06d}.pt"

    torch.save(theta_phys.detach().cpu(), str(theta_path))

    meta = {
        "run_id": run_id,
        "iter": int(it),
        "created_unix_s": time.time(),
        "theta_path": str(theta_path),
    }
    write_json(meta_path, meta)
    return meta


def create_run_record(
    *,
    name: str,
    ph4_dir: str,
    lock_id: str,
    core_freeze_id: str,
    capability_snapshot: Dict[str, Any],
    domain: Dict[str, Any],
    process: Dict[str, Any],
    opt_config: Dict[str, Any],
) -> Dict[str, Any]:
    run_id = new_run_id()
    now = time.time()

    paths = make_run_paths(run_id)

    # Write inputs (filesystem contract)
    write_json(paths.inputs_dir / "capability_snapshot.json", capability_snapshot)
    write_json(paths.inputs_dir / "domain.json", domain)
    write_json(paths.inputs_dir / "process.json", process)
    write_json(paths.inputs_dir / "opt_config.json", opt_config)

    # Extract fields for DB
    material = process.get("material")
    voxel_size_mm = domain.get("voxel_size_mm")

    # Prefer your newer naming ("total_iters") if present
    total_iters = opt_config.get("total_iters", opt_config.get("iterations", 0))
    total_iters = int(total_iters or 0)

    probe_mode = opt_config.get("probe_mode")

    # Insert row in SQLite
    con = connect()
    try:
        con.execute(
            """
            INSERT INTO runs(
                run_id, name, status, created_unix_s, updated_unix_s,
                material, voxel_size_mm, probe_mode,
                ph4_dir, lock_id, core_freeze_id,
                run_dir, last_iter, total_iters,
                warnings_json, error_text
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                name,
                "CREATED",
                now,
                now,
                material,
                voxel_size_mm,
                probe_mode,
                ph4_dir,
                lock_id,
                core_freeze_id,
                str(paths.run_dir),
                0,
                total_iters,
                "[]",
                None,
            ),
        )
        con.commit()
    finally:
        con.close()

    return {
        "run_id": run_id,
        "status": "CREATED",
        "run_dir": str(paths.run_dir),
    }


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    """
    Safe DB fetch: uses cursor.description to build dict.
    """
    con = connect()
    try:
        cur = con.execute("SELECT * FROM runs WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        if row is None:
            return None

        # sqlite3.Row supports mapping, but we normalize to plain dict
        names = [d[0] for d in cur.description]
        d = {names[i]: row[i] for i in range(len(names))}

        # Parse warnings_json
        try:
            d["warnings"] = json.loads(d.get("warnings_json", "[]") or "[]")
        except Exception:
            d["warnings"] = []
        d.pop("warnings_json", None)

        return d
    finally:
        con.close()
