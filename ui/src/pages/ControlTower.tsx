import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

type Capabilities = {
  ok: boolean;
  ph4_dir: string;
  engine_ids?: { LOCK_ID?: string; CORE_FREEZE_ID?: string };
  materials?: string[];
  build_direction?: string;
  domain_cap?: number[];
  voxel_size_mm_options?: number[];
  notes?: string;
};

type RunRow = {
  run_id: string;
  name: string;
  status: string;
  last_iter: number;
  total_iters: number;
  material?: string | null;
  control_state?: string | null;
  updated_unix_s?: number | null;
  stop_reason?: string | null;
  error_text?: string | null;
};

type RunsResponse = { ok: boolean; runs: RunRow[] };

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function progressPct(lastIter: number, totalIters: number) {
  if (!totalIters || totalIters <= 0) return 0;
  return clamp((lastIter / totalIters) * 100, 0, 100);
}

function fmtAgo(updatedUnixS?: number | null) {
  if (!updatedUnixS) return "—";
  const diff = Math.max(0, Date.now() / 1000 - updatedUnixS);
  if (diff < 10) return "just now";
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function statusPill(status: string) {
  const s = (status || "").toUpperCase();
  const base =
    "inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold";

  if (s === "RUNNING") return `${base} bg-sky-500/10 text-sky-200 border-sky-400/20`;
  if (s === "PAUSED") return `${base} bg-amber-500/10 text-amber-200 border-amber-400/20`;
  if (s === "DONE") return `${base} bg-emerald-500/10 text-emerald-200 border-emerald-400/20`;
  if (s === "STOPPED") return `${base} bg-white/5 text-slate-200 border-white/10`;
  if (s === "FAILED") return `${base} bg-rose-500/10 text-rose-200 border-rose-400/20`;
  if (s === "QUEUED") return `${base} bg-violet-500/10 text-violet-200 border-violet-400/20`;
  return `${base} bg-white/5 text-slate-200 border-white/10`;
}

function isAbortError(e: unknown) {
  return e instanceof DOMException && e.name === "AbortError";
}

async function apiPost(path: string, body?: unknown) {
  const res = await fetch(path, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const j = (await res.json()) as { detail?: string };
      if (j?.detail) detail = j.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  try {
    return await res.json();
  } catch {
    return {};
  }
}

export default function ControlTower() {
  const nav = useNavigate();

  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busyRun, setBusyRun] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const capsCtl = new AbortController();
    const runsCtl = new AbortController();

    async function loadCaps() {
      try {
        const res = await fetch("/capabilities", { signal: capsCtl.signal });
        if (!res.ok) throw new Error(`GET /capabilities failed: ${res.status}`);
        const data = (await res.json()) as Capabilities;
        if (!cancelled) setCaps(data);
      } catch (e: unknown) {
        if (isAbortError(e)) return;
        const msg = e instanceof Error ? e.message : String(e);
        if (!cancelled) setErr(msg);
      }
    }

    async function loadRuns() {
      try {
        const res = await fetch("/runs", { signal: runsCtl.signal });
        if (!res.ok) throw new Error(`GET /runs failed: ${res.status}`);
        const data = (await res.json()) as RunsResponse;
        if (!cancelled) setRuns(data.runs || []);
      } catch (e: unknown) {
        if (isAbortError(e)) return;
        const msg = e instanceof Error ? e.message : String(e);
        if (!cancelled) setErr(msg);
      }
    }

    loadCaps();
    loadRuns();

    const t = setInterval(loadRuns, 1500);

    return () => {
      cancelled = true;
      clearInterval(t);
      capsCtl.abort();
      runsCtl.abort();
    };
  }, []);

  const summary = useMemo(() => {
    const materials = caps?.materials?.length ?? 0;
    const vox = caps?.voxel_size_mm_options?.length ?? 0;
    const domain = caps?.domain_cap ? `${caps.domain_cap.join("×")}` : "—";
    return { materials, vox, domain };
  }, [caps]);

  function openRun(r: RunRow) {
    const s = (r.status || "").toUpperCase();
    if (s === "RUNNING" || s === "PAUSED" || s === "QUEUED") {
      nav(`/optimization-run/${r.run_id}`);
    } else {
      nav(`/results/${r.run_id}`);
    }
  }

  async function doPause(r: RunRow) {
    setErr(null);
    setBusyRun(r.run_id);
    try {
      await apiPost(`/runs/${r.run_id}/pause`, { reason: "pause requested (ui)" });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
    } finally {
      setBusyRun(null);
    }
  }

  async function doResume(r: RunRow) {
    setErr(null);
    setBusyRun(r.run_id);
    try {
      await apiPost(`/runs/${r.run_id}/resume`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
    } finally {
      setBusyRun(null);
    }
  }

  async function doStop(r: RunRow) {
    setErr(null);
    setBusyRun(r.run_id);
    try {
      await apiPost(`/runs/${r.run_id}/stop`, { reason: "stop requested (ui)" });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(msg);
    } finally {
      setBusyRun(null);
    }
  }

  return (
    <div className="space-y-4">
      {err && (
        <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-100">
          {err}
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-4">
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Kernel
          </div>
          <div className="mt-2 text-lg font-semibold">{caps?.ok ? "OK" : "…"}</div>
          <div className="mt-2 break-all text-xs text-slate-400">
            {caps?.ph4_dir ?? "Loading…"}
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Materials
          </div>
          <div className="mt-2 text-lg font-semibold">{summary.materials}</div>
          <div className="mt-2 text-xs text-slate-400">
            {(caps?.materials ?? []).join(", ") || "—"}
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Domain cap
          </div>
          <div className="mt-2 text-lg font-semibold">{summary.domain}</div>
          <div className="mt-2 text-xs text-slate-400">
            Build dir: {caps?.build_direction ?? "—"}
          </div>
        </div>

        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
            Voxel sizes
          </div>
          <div className="mt-2 text-lg font-semibold">{summary.vox}</div>
          <div className="mt-2 text-xs text-slate-400">
            {(caps?.voxel_size_mm_options ?? []).join(", ") || "—"} mm
          </div>
        </div>
      </div>

      <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5">
        <div className="flex items-center justify-between border-b border-white/10 px-4 py-3">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Runs
            </div>
            <div className="mt-1 text-lg font-semibold">{runs.length}</div>
          </div>
          <div className="text-xs text-slate-400">Auto-refresh: 1.5s</div>
        </div>

        <div className="w-full overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-white/5 text-xs uppercase tracking-wide text-slate-400">
                <th className="px-4 py-3 text-left">Run</th>
                <th className="px-4 py-3 text-left">Status</th>
                <th className="px-4 py-3 text-left">Progress</th>
                <th className="px-4 py-3 text-left">Material</th>
                <th className="px-4 py-3 text-left">Control</th>
                <th className="px-4 py-3 text-left">Updated</th>
                <th className="px-4 py-3 text-left">Actions</th>
              </tr>
            </thead>

            <tbody>
              {runs.map((r) => {
                const pct = progressPct(r.last_iter, r.total_iters);
                const s = (r.status || "").toUpperCase();
                const busy = busyRun === r.run_id;

                return (
                  <tr key={r.run_id} className="border-t border-white/5">
                    <td className="px-4 py-3">
                      <div className="font-semibold text-slate-100">{r.name || "—"}</div>
                      <div className="mt-1 font-mono text-xs text-slate-400">{r.run_id}</div>

                      {(r.stop_reason || r.error_text) && (
                        <div className="mt-2 text-xs text-slate-400">
                          {r.error_text ? (
                            <span className="text-rose-200">error: {r.error_text}</span>
                          ) : (
                            <span>reason: {r.stop_reason}</span>
                          )}
                        </div>
                      )}
                    </td>

                    <td className="px-4 py-3">
                      <span className={statusPill(r.status)}>{r.status}</span>
                    </td>

                    <td className="px-4 py-3 min-w-[260px]">
                      <div className="flex justify-between text-xs text-slate-400">
                        <span>
                          {r.last_iter}/{r.total_iters || 0}
                        </span>
                        <span>{pct.toFixed(1)}%</span>
                      </div>
                      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10">
                        <div className="h-full bg-sky-500" style={{ width: `${pct}%` }} />
                      </div>
                    </td>

                    <td className="px-4 py-3 text-sm text-slate-200">{r.material ?? "—"}</td>
                    <td className="px-4 py-3 text-sm text-slate-200">{r.control_state ?? "—"}</td>
                    <td className="px-4 py-3 text-sm text-slate-400">{fmtAgo(r.updated_unix_s)}</td>

                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-2">
                        <button
                          className="rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-semibold text-slate-100 hover:bg-white/10"
                          onClick={() => openRun(r)}
                        >
                          Open
                        </button>

                        <button
                          disabled={busy || s !== "RUNNING"}
                          className="rounded-lg border border-amber-400/20 bg-amber-500/10 px-3 py-1.5 text-xs font-semibold text-amber-100 hover:bg-amber-500/20 disabled:opacity-40"
                          onClick={() => doPause(r)}
                        >
                          Pause
                        </button>

                        <button
                          disabled={busy || s !== "PAUSED"}
                          className="rounded-lg border border-sky-400/20 bg-sky-500/10 px-3 py-1.5 text-xs font-semibold text-sky-100 hover:bg-sky-500/20 disabled:opacity-40"
                          onClick={() => doResume(r)}
                        >
                          Resume
                        </button>

                        <button
                          disabled={busy || !(s === "RUNNING" || s === "PAUSED")}
                          className="rounded-lg border border-rose-400/20 bg-rose-500/10 px-3 py-1.5 text-xs font-semibold text-rose-100 hover:bg-rose-500/20 disabled:opacity-40"
                          onClick={() => doStop(r)}
                        >
                          Stop
                        </button>
                      </div>

                      {busy && <div className="mt-2 text-xs text-slate-400">sending…</div>}
                    </td>
                  </tr>
                );
              })}

              {runs.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-sm text-slate-400">
                    No runs found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {caps?.notes && <div className="text-xs text-slate-400">Note: {caps.notes}</div>}
    </div>
  );
}
