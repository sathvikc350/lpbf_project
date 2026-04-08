import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

type RunRow = {
  run_id: string;
  name?: string;
  status?: string;
  updated_unix_s?: number;
  material?: string;
  last_iter?: number;
  total_iters?: number;
  stop_reason?: string | null;
};

type RunsResp = { ok: boolean; runs: RunRow[] };

type ArtifactsResp = {
  ok: boolean;
  exports: Array<{ rel: string }>;
};

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  const text = await res.text();
  if (!res.ok) {
    try {
      const j = JSON.parse(text) as { detail?: string };
      throw new Error(j.detail || `GET ${url} failed: ${res.status}`);
    } catch {
      throw new Error(text || `GET ${url} failed: ${res.status}`);
    }
  }
  return JSON.parse(text) as T;
}

function fmtTime(unix?: number) {
  if (!unix) return "—";
  const d = new Date(unix * 1000);
  return d.toLocaleString();
}

function isDone(r: RunRow) {
  return (r.status || "").toUpperCase() === "DONE";
}

function hasVtp(a: ArtifactsResp | null) {
  const ex = a?.exports || [];
  return ex.some((x) => x.rel.startsWith("physics/") && /physics_manifest_iter_\d{6}\.vtp$/.test(x.rel));
}

export default function ResultsHome() {
  const nav = useNavigate();
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [runs, setRuns] = useState<RunRow[]>([]);
  const [artMap, setArtMap] = useState<Record<string, ArtifactsResp | null>>({});

  async function refresh() {
    setBusy(true);
    setErr(null);

    try {
      const r = await getJson<RunsResp>("/runs");
      const allRuns = (r.runs || []).slice();
      setRuns(allRuns);

      // Only check artifacts for DONE runs (keeps it cheap)
      const doneRuns = allRuns.filter(isDone);

      const entries = await Promise.all(
        doneRuns.map(async (row) => {
          try {
            const a = await getJson<ArtifactsResp>(`/runs/${row.run_id}/artifacts`);
            return [row.run_id, a] as const;
          } catch {
            return [row.run_id, null] as const;
          }
        })
      );

      const next: Record<string, ArtifactsResp | null> = {};
      for (const [rid, a] of entries) next[rid] = a;
      setArtMap(next);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  const shown = useMemo(() => {
    // Only DONE runs that actually have VTP exports
    const done = runs.filter(isDone);
    const withVtp = done.filter((r) => hasVtp(artMap[r.run_id] ?? null));

    return withVtp.sort((a, b) => (b.updated_unix_s ?? 0) - (a.updated_unix_s ?? 0));
  }, [runs, artMap]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Results View</div>
          <div className="text-lg font-semibold text-slate-100">Done runs with exports</div>
          <div className="text-xs text-slate-400">
            Only shows DONE runs that have <span className="font-mono">physics_manifest_iter_XXXXXX.vtp</span>
          </div>
        </div>

        <button
          type="button"
          onClick={() => void refresh()}
          className={cx(
            "rounded-lg px-3 py-2 text-sm font-semibold border transition",
            busy
              ? "bg-white/5 text-slate-400 border-white/10"
              : "bg-sky-500/15 text-sky-100 border-sky-400/30 hover:bg-sky-500/25"
          )}
          disabled={busy}
        >
          {busy ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      {err && (
        <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-100 break-words">
          {err}
        </div>
      )}

      <div className="rounded-2xl border border-white/10 bg-white/5 overflow-hidden">
        <div className="overflow-auto">
          <table className="w-full text-sm">
            <thead className="bg-white/5 text-slate-300">
              <tr>
                <th className="text-left px-4 py-3">Run</th>
                <th className="text-left px-4 py-3">Material</th>
                <th className="text-left px-4 py-3">Iters</th>
                <th className="text-left px-4 py-3">Updated</th>
                <th className="text-right px-4 py-3">Action</th>
              </tr>
            </thead>
            <tbody className="text-slate-200">
              {shown.length === 0 ? (
                <tr>
                  <td className="px-4 py-4 text-slate-400" colSpan={5}>
                    No DONE runs with VTP exports yet. (Run the exporter step or finish a run that writes exports.)
                  </td>
                </tr>
              ) : (
                shown.map((r) => {
                  const it = `${r.last_iter ?? 0}/${r.total_iters ?? 0}`;
                  return (
                    <tr key={r.run_id} className="border-t border-white/10">
                      <td className="px-4 py-3">
                        <div className="font-mono text-xs">{r.run_id}</div>
                        <div className="text-xs text-slate-400">{r.name || "—"}</div>
                      </td>
                      <td className="px-4 py-3">{r.material || "—"}</td>
                      <td className="px-4 py-3">{it}</td>
                      <td className="px-4 py-3 text-slate-300">{fmtTime(r.updated_unix_s)}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          onClick={() => nav(`/results/${r.run_id}`)}
                          className="rounded-lg px-3 py-2 text-sm font-semibold bg-emerald-600 text-white hover:bg-emerald-500 transition"
                        >
                          Open
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div className="text-xs text-slate-400">
        Note: “DONE” ≠ “has exports”. This page only shows “DONE + has VTP”.
      </div>
    </div>
  );
}
