from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import json
import hashlib
import sqlite3
import time

app = FastAPI(title="LPBF Digital Twin (Frozen Kernel API)", version="0.1.0")

# -----------------------------
# CORS (required for UI @5173)
# -----------------------------
# This fixes: "blocked by CORS policy: No 'Access-Control-Allow-Origin' header"
# so the browser can fetch VTP/STL downloads from :8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from .runs_api import router as runs_router
from .ws_api import attach_ws_routes

app.include_router(runs_router)
attach_ws_routes(app)

PH4_DIR = Path("/workspaces/lpbf_project/artifacts/phase4_engine_freeze_locked_in718_20260216_205826")
DB_PATH = Path("/workspaces/lpbf_project/artifacts/runs.db")


@app.on_event("startup")
def _startup_sweep_zombie_runs() -> None:
    """
    If the backend restarts, runner threads are gone,
    but DB rows might still say RUNNING / PAUSED / QUEUED.

    This prevents "zombie runs" by making them terminal + consistent.
    """
    if not DB_PATH.exists():
        return

    now = time.time()
    con = sqlite3.connect(str(DB_PATH))
    try:
        con.execute(
            """
            UPDATE runs
            SET status='STOPPED',
                control_state='IDLE',
                paused_at_unix_s=NULL,
                updated_unix_s=?,
                stop_reason=COALESCE(stop_reason, 'server restarted')
            WHERE UPPER(status) IN ('RUNNING','PAUSED','QUEUED')
            """,
            (now,),
        )

        # Normalize CREATED rows that somehow carried a non-IDLE control_state
        con.execute(
            """
            UPDATE runs
            SET control_state='IDLE',
                updated_unix_s=?
            WHERE UPPER(status)='CREATED'
              AND UPPER(COALESCE(control_state,'')) <> 'IDLE'
            """,
            (now,),
        )

        con.commit()
    finally:
        con.close()


@app.get("/health")
def health():
    return {"ok": True, "ph4_dir": str(PH4_DIR)}


REG_PATH = PH4_DIR / "phase3_registry_locked_in718_20260212_212457" / "material_registry.json"
FROZEN_BUNDLE_DIR = PH4_DIR / "surrogate_bundle" / "meltpool_v1"
SURROGATE_CFG = FROZEN_BUNDLE_DIR / "surrogate_config.json"
SURROGATE_WEIGHTS = FROZEN_BUNDLE_DIR / "weights.pt"


def _sha256_file(p: Path) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(p: Path) -> str | None:
    return p.read_text().strip() if p.exists() else None


@app.get("/capabilities")
def capabilities():
    if not REG_PATH.exists():
        return {"ok": False, "error": f"missing registry: {REG_PATH}", "ph4_dir": str(PH4_DIR)}

    reg = json.loads(REG_PATH.read_text())
    materials = list(reg["materials"].keys())

    validity_windows = {}
    for m in materials:
        vw = reg["materials"][m]["validity_window"]
        validity_windows[m] = {
            "P_W": vw.get("P_W"),
            "v_mm_per_s": vw.get("v_mm_per_s"),
            "h_mm": vw.get("h_mm"),
            "t_mm": vw.get("t_mm"),
            "VED_J_per_mm3": vw.get("VED_J_per_mm3"),
            "LED_J_per_mm": vw.get("LED_J_per_mm"),
        }

    return {
        "ok": True,
        "ph4_dir": str(PH4_DIR),
        "engine_ids": {
            "LOCK_ID": _read_text(PH4_DIR / "PHASE4_LOCK_ID.txt"),
            "CORE_FREEZE_ID": _read_text(PH4_DIR / "PHASE4_CORE_FREEZE_ID.txt"),
        },
        "materials": materials,
        "validity_windows": validity_windows,
        "surrogate_bundle": {
            "bundle_dir": str(FROZEN_BUNDLE_DIR),
            "surrogate_contract_id": reg.get("surrogate_bundle", {}).get("surrogate_contract_id"),
            "cfg_sha256": _sha256_file(SURROGATE_CFG),
            "weights_sha256": _sha256_file(SURROGATE_WEIGHTS),
        },
        "build_direction": "+Z",
        "domain_cap": [128, 128, 128],
        "voxel_size_mm_options": [0.25, 0.5, 1.0],
        "stl_import_mode": "seed_within_domain",
        "expert_override_supported": True,
        "progress_transport": "websocket",
        "notes": "v1 uses Phase-4 frozen kernel; UI must not hardcode any ranges/options.",
    }
