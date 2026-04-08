# backend/app/ws_api.py
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

REPO = Path("/workspaces/lpbf_project")
DB_PATH = REPO / "artifacts" / "runs.db"

WS_SCHEMA_VERSION = "ws_schema_v1"


# -----------------------------
# DB helpers
# -----------------------------
def _connect() -> sqlite3.Connection:
    # short timeout reduces "database is locked" surprises during rapid ticks
    return sqlite3.connect(str(DB_PATH), timeout=5.0)


def _fetchone_dict(sql: str, params: tuple) -> Optional[Dict[str, Any]]:
    con = _connect()
    try:
        cur = con.execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))
    finally:
        con.close()


def _exec(sql: str, params: tuple) -> None:
    con = _connect()
    try:
        con.execute(sql, params)
        con.commit()
    finally:
        con.close()


def _apply_control_action(run_id: str, action: str, reason: Optional[str]) -> Dict[str, Any]:
    """
    Applies control to DB in a way compatible with your runner:
      - pause  -> control_state=PAUSE_REQUESTED (runner will set status=PAUSED)
      - resume -> control_state=RUNNING + clear paused_at_unix_s + clear stop_reason
      - stop   -> control_state=STOP_REQUESTED (runner will set status=STOPPED)
    """
    a = (action or "").strip().lower()
    now = time.time()

    if a == "pause":
        _exec(
            """
            UPDATE runs
            SET control_state=?,
                updated_unix_s=?,
                stop_reason=COALESCE(?, stop_reason)
            WHERE run_id=?
            """,
            ("PAUSE_REQUESTED", now, reason, run_id),
        )
        return {"control_state": "PAUSE_REQUESTED", "stop_reason": reason}

    if a == "resume":
        _exec(
            """
            UPDATE runs
            SET control_state=?,
                updated_unix_s=?,
                paused_at_unix_s=NULL,
                stop_reason=NULL,
                status=CASE WHEN status='PAUSED' THEN 'RUNNING' ELSE status END
            WHERE run_id=?
            """,
            ("RUNNING", now, run_id),
        )
        return {"control_state": "RUNNING"}

    if a == "stop":
        _exec(
            """
            UPDATE runs
            SET control_state=?,
                updated_unix_s=?,
                stop_reason=COALESCE(?, stop_reason)
            WHERE run_id=?
            """,
            ("STOP_REQUESTED", now, reason, run_id),
        )
        return {"control_state": "STOP_REQUESTED", "stop_reason": reason}

    raise ValueError("Unknown control action (expected: pause|resume|stop)")


# -----------------------------
# checkpoint / preview / latest helpers
# -----------------------------
def _checkpoint_iters(run_dir: Path) -> list[int]:
    """
    Returns all checkpoint iters found, preferring theta_phys_iter_*.pt.
    """
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []

    iters: list[int] = []

    for p in ckpt_dir.glob("theta_phys_iter_*.pt"):
        try:
            it = int(p.name.split("_")[-1].split(".")[0])
            iters.append(it)
        except Exception:
            pass

    if iters:
        return sorted(set(iters))

    for p in ckpt_dir.glob("checkpoint_iter_*.json"):
        try:
            it = int(p.name.split("_")[-1].split(".")[0])
            iters.append(it)
        except Exception:
            pass

    return sorted(set(iters))


def _health_status(run: Dict[str, Any]) -> str:
    """
    Simple v1 health:
      - ERROR if error_text is set OR logs/run_error.txt exists and has content
      - OK otherwise
    """
    if run.get("error_text"):
        return "ERROR"

    run_dir_str = run.get("run_dir")
    if not run_dir_str:
        return "OK"

    p = Path(str(run_dir_str)) / "logs" / "run_error.txt"
    try:
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            return "ERROR"
    except Exception:
        pass

    return "OK"


def _progress_pct(run: Dict[str, Any]) -> float:
    """
    v1 progress percent (0..100). Safe for total_iters=0.
    """
    try:
        last_iter = int(run.get("last_iter") or 0)
        total_iters = int(run.get("total_iters") or 0)
        if total_iters <= 0:
            return 0.0
        pct = 100.0 * (float(last_iter) / float(total_iters))
        return round(min(max(pct, 0.0), 100.0), 1)
    except Exception:
        return 0.0


def _enrich_run_for_ui(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds UI fields (same idea as /latest):
      - latest_checkpoint_iter
      - num_checkpoints
      - progress_pct
      - health_status
      - preview_stl_url
      - preview_watertight_stl_url
    Uses relative URLs so it works behind proxies.
    """
    out = dict(run)

    run_dir_str = out.get("run_dir")
    if not run_dir_str:
        out["latest_checkpoint_iter"] = None
        out["num_checkpoints"] = 0
        out["progress_pct"] = _progress_pct(out)
        out["health_status"] = _health_status(out)
        out["preview_stl_url"] = None
        out["preview_watertight_stl_url"] = None
        return out

    run_dir = Path(str(run_dir_str))
    iters = _checkpoint_iters(run_dir)
    latest = iters[-1] if iters else None

    out["latest_checkpoint_iter"] = latest
    out["num_checkpoints"] = len(iters)
    out["progress_pct"] = _progress_pct(out)
    out["health_status"] = _health_status(out)

    if latest is None:
        out["preview_stl_url"] = None
        out["preview_watertight_stl_url"] = None
        return out

    run_id = out.get("run_id")
    out["preview_stl_url"] = f"/runs/{run_id}/checkpoints/{latest}/preview_stl?level=0.525"
    out["preview_watertight_stl_url"] = (
        f"/runs/{run_id}/checkpoints/{latest}/preview_stl_watertight?level=0.525&pitch_mm=0.25"
    )
    return out


# -----------------------------
# WS routes
# -----------------------------
def attach_ws_routes(app: FastAPI) -> None:
    @app.websocket("/ws/runs/{run_id}")
    async def ws_runs(websocket: WebSocket, run_id: str):
        await websocket.accept()

        tick_period_s = 0.5
        poll_period_s = 0.05
        next_tick = time.time() + tick_period_s

        async def _send(payload: Dict[str, Any]) -> None:
            # enforce schema_version everywhere
            if "schema_version" not in payload:
                payload["schema_version"] = WS_SCHEMA_VERSION
            await websocket.send_text(json.dumps(payload))

        try:
            # initial snapshot
            run = _fetchone_dict("SELECT * FROM runs WHERE run_id=?", (run_id,))
            if run is None:
                await _send({"ok": False, "type": "error", "error": "Run not found", "run_id": run_id, "ts": time.time()})
                await websocket.close(code=1008)
                return

            await _send({"ok": True, "type": "init", "run": _enrich_run_for_ui(run), "ts": time.time()})

            while True:
                # 1) Try receive client message (non-blocking via timeout)
                try:
                    msg = await asyncio.wait_for(websocket.receive_text(), timeout=poll_period_s)
                    try:
                        payload = json.loads(msg)
                    except Exception:
                        await _send({"ok": False, "type": "error", "error": "Expected JSON message", "ts": time.time()})
                        continue

                    mtype = str(payload.get("type", "")).strip().lower()
                    if mtype != "control":
                        await _send({"ok": False, "type": "error", "error": "Unsupported message type", "ts": time.time()})
                        continue

                    action = str(payload.get("action", "")).strip().lower()
                    reason = payload.get("reason", None)
                    reason = str(reason) if reason is not None else None

                    try:
                        applied = _apply_control_action(run_id, action, reason)
                        await _send({"ok": True, "type": "ack", "action": action, "ts": time.time(), **applied})
                    except Exception as e:
                        await _send({"ok": False, "type": "error", "error": str(e), "ts": time.time()})

                except asyncio.TimeoutError:
                    pass

                # 2) Tick on schedule
                now = time.time()
                if now >= next_tick:
                    run = _fetchone_dict("SELECT * FROM runs WHERE run_id=?", (run_id,))
                    if run is None:
                        await _send({"ok": False, "type": "gone", "run_id": run_id, "ts": now})
                        await websocket.close(code=1008)
                        return

                    await _send({"ok": True, "type": "tick", "ts": now, "run": _enrich_run_for_ui(run)})
                    next_tick = now + tick_period_s

        except WebSocketDisconnect:
            return
        except Exception as e:
            # Never crash server on WS
            try:
                await _send({"ok": False, "type": "error", "error": str(e), "ts": time.time()})
            except Exception:
                pass
            try:
                await websocket.close(code=1011)
            except Exception:
                pass
