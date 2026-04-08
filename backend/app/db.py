# backend/app/db.py
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

# Default DB location (override with LPBF_DB_PATH env var)
DEFAULT_DB_PATH = Path("/workspaces/lpbf_project/artifacts/runs.db")


def get_db_path() -> Path:
    p = Path(os.environ.get("LPBF_DB_PATH", str(DEFAULT_DB_PATH)))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    dbp = db_path or get_db_path()
    con = sqlite3.connect(str(dbp))
    con.execute("PRAGMA journal_mode=WAL;")          # better crash safety + concurrency
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA synchronous=NORMAL;")        # good durability/perf tradeoff
    return con


def init_db(db_path: Optional[Path] = None) -> Path:
    """
    Creates tables if they don't exist.
    """
    con = connect(db_path)
    cur = con.cursor()

    # Runs table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            name TEXT,
            status TEXT NOT NULL,                     -- CREATED/RUNNING/PAUSED/COMPLETED/FAILED
            created_unix_s REAL NOT NULL,
            updated_unix_s REAL NOT NULL,

            material TEXT,
            voxel_size_mm REAL,
            probe_mode TEXT,

            ph4_dir TEXT NOT NULL,
            lock_id TEXT NOT NULL,
            core_freeze_id TEXT NOT NULL,

            run_dir TEXT NOT NULL,                    -- artifacts/runs/<run_id>
            last_iter INTEGER DEFAULT 0,
            total_iters INTEGER DEFAULT 0,

            warnings_json TEXT DEFAULT "[]",
            error_text TEXT
        );
        """
    )

    # Checkpoints table (resume mid-run)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            iter INTEGER NOT NULL,
            checkpoint_path TEXT NOT NULL,
            created_unix_s REAL NOT NULL,

            UNIQUE(run_id, iter),
            FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        );
        """
    )

    # Simple key/value table for app state (optional but handy)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS kv (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );
        """
    )

    con.commit()
    con.close()
    return get_db_path()
