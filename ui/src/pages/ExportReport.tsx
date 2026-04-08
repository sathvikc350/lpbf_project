import { useEffect, useMemo, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

type RunRow = {
  run_id: string;
  name?: string | null;
  status: string;
  material?: string | null;
  last_iter?: number;
  total_iters?: number;
  created_unix_s?: number;
};

type RunsResp = {
  ok: boolean;
  runs: RunRow[];
};

type ArtItem = {
  rel: string;
  download_url: string;
  bytes?: number;
  mtime_unix_s?: number;
};

type ArtifactsResp = {
  ok: boolean;
  run_id: string;
  exports: ArtItem[];
  reports: ArtItem[];
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

async function postNoBody(url: string): Promise<void> {
  const res = await fetch(url, { method: "POST" });
  const text = await res.text();
  if (!res.ok) {
    try {
      const j = JSON.parse(text) as { detail?: string };
      throw new Error(j.detail || `POST ${url} failed: ${res.status}`);
    } catch {
      throw new Error(text || `POST ${url} failed: ${res.status}`);
    }
  }
}

function fmtBytes(n?: number) {
  if (!n || !Number.isFinite(n)) return "—";
  if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

function pickLatestByPrefix(items: ArtItem[], prefix: string, suffix: string) {
  const candidates = items.filter((x) => x.rel.startsWith(prefix) && x.rel.endsWith(suffix));
  candidates.sort((a, b) => (b.mtime_unix_s ?? 0) - (a.mtime_unix_s ?? 0));
  return candidates[0] ?? null;
}

export default function ExportReport() {
  const { runId: runIdFromPath } = useParams();
  const [searchParams] = useSearchParams();
  const runId = runIdFromPath || searchParams.get("runId") || "";

  const [runs, setRuns] = useState<RunRow[]>([]);
  const [arts, setArts] = useState<ArtifactsResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [summaryBusy, setSummaryBusy] = useState(false);

  useEffect(() => {
    let alive = true;

    (async () => {
      try {
        const data = await getJson<RunsResp>("/runs");
        if (alive) {
          setRuns(data.runs || []);
          setErr(null);
        }
      } catch (e: unknown) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!runId) {
      setArts(null);
      return;
    }

    let alive = true;

    (async () => {
      try {
        const data = await getJson<ArtifactsResp>(`/runs/${runId}/artifacts`);
        if (alive) {
          setArts(data);
          setErr(null);
        }
      } catch (e: unknown) {
        if (alive) setErr(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      alive = false;
    };
  }, [runId]);

  const latestPreview = useMemo(
    () => (arts ? pickLatestByPrefix(arts.exports || [], "preview/", ".stl") : null),
    [arts]
  );

  const latestWatertight = useMemo(
    () =>
      arts
        ? pickLatestByPrefix(
            (arts.exports || []).filter((x) => !x.rel.includes("_tmp_surface_")),
            "preview_watertight/",
            ".stl"
          )
        : null,
    [arts]
  );

  const latestVtp = useMemo(
    () => (arts ? pickLatestByPrefix(arts.exports || [], "physics/", ".vtp") : null),
    [arts]
  );

  const summaryFile = useMemo(
    () => (arts ? pickLatestByPrefix(arts.reports || [], "", ".json") : null),
    [arts]
  );

  const createSummary = async () => {
    if (!runId) return;
    setSummaryBusy(true);
    try {
      await postNoBody(`/runs/${runId}/reports/summary`);
      const data = await getJson<ArtifactsResp>(`/runs/${runId}/artifacts`);
      setArts(data);
      setErr(null);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setSummaryBusy(false);
    }
  };

  return (
    <section className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-lg font-semibold text-slate-800">Export & Report</div>
          <div className="mt-1 text-xs text-slate-500">
            Select a run and download the finished deliverables.
          </div>
        </div>

        <Link
          to="/results"
          className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300"
        >
          Open Results Catalog
        </Link>
      </div>

      {err ? (
        <div className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
          {err}
        </div>
      ) : null}

      <div className="border border-slate-200 rounded-xl bg-slate-50 p-4">
        <div className="text-sm font-semibold text-slate-700">Available Runs</div>
        <div className="mt-3 flex flex-wrap gap-2">
          {runs.map((r) => (
            <Link
              key={r.run_id}
              to={`/export/${r.run_id}`}
              className={cx(
                "rounded-lg border px-3 py-2 text-sm transition",
                runId === r.run_id
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-white text-slate-700 border-slate-200 hover:bg-blue-50 hover:border-blue-300"
              )}
            >
              {r.run_id} · {r.status}
            </Link>
          ))}
        </div>
      </div>

      {!runId ? (
        <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-600">
          Choose a run above to load exports and reports.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="border border-slate-200 rounded-xl bg-slate-50 p-4">
            <div className="text-sm font-semibold text-slate-700">Downloads</div>

            <div className="mt-3 space-y-3">
              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="font-medium text-slate-700">Physics VTP</div>
                <div className="mt-1 text-xs text-slate-500 font-mono break-all">
                  {latestVtp?.rel ?? "Not available"}
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-slate-500">{fmtBytes(latestVtp?.bytes)}</span>
                  {latestVtp ? (
                    <a
                      href={latestVtp.download_url}
                      className="rounded-lg bg-blue-600 px-3 py-2 text-white text-sm hover:bg-blue-700"
                    >
                      Download
                    </a>
                  ) : null}
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="font-medium text-slate-700">Preview STL</div>
                <div className="mt-1 text-xs text-slate-500 font-mono break-all">
                  {latestPreview?.rel ?? "Not available"}
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-slate-500">{fmtBytes(latestPreview?.bytes)}</span>
                  {latestPreview ? (
                    <a
                      href={latestPreview.download_url}
                      className="rounded-lg bg-blue-600 px-3 py-2 text-white text-sm hover:bg-blue-700"
                    >
                      Download
                    </a>
                  ) : null}
                </div>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="font-medium text-slate-700">Watertight STL</div>
                <div className="mt-1 text-xs text-slate-500 font-mono break-all">
                  {latestWatertight?.rel ?? "Not available"}
                </div>
                <div className="mt-2 flex items-center justify-between">
                  <span className="text-xs text-slate-500">{fmtBytes(latestWatertight?.bytes)}</span>
                  {latestWatertight ? (
                    <a
                      href={latestWatertight.download_url}
                      className="rounded-lg bg-blue-600 px-3 py-2 text-white text-sm hover:bg-blue-700"
                    >
                      Download
                    </a>
                  ) : null}
                </div>
              </div>
            </div>
          </div>

          <div className="border border-slate-200 rounded-xl bg-slate-50 p-4">
            <div className="text-sm font-semibold text-slate-700">Summary Report</div>
            <div className="mt-2 text-sm text-slate-600">
              Generate or download the JSON summary for this run.
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => void createSummary()}
                disabled={summaryBusy}
                className={cx(
                  "rounded-lg px-3 py-2 text-sm font-medium border transition",
                  summaryBusy
                    ? "bg-slate-200 text-slate-500 border-slate-200"
                    : "bg-white text-slate-700 border-slate-200 hover:bg-blue-50 hover:border-blue-300"
                )}
              >
                {summaryBusy ? "Generating..." : "Generate Summary"}
              </button>

              {summaryFile ? (
                <a
                  href={summaryFile.download_url}
                  className="rounded-lg bg-blue-600 px-3 py-2 text-sm font-medium text-white hover:bg-blue-700"
                >
                  Download Summary JSON
                </a>
              ) : null}
            </div>

            <div className="mt-4 rounded-lg border border-slate-200 bg-white p-3">
              <div className="text-xs text-slate-500 font-mono break-all">
                {summaryFile?.rel ?? "No summary file yet."}
              </div>
              <div className="mt-2 text-xs text-slate-500">
                {summaryFile ? `Size: ${fmtBytes(summaryFile.bytes)}` : "Generate one above."}
              </div>
            </div>

            <div className="mt-4 border-t border-slate-200 pt-4">
              <div className="text-sm font-semibold text-slate-700">Navigation</div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Link
                  to={`/results/${runId}`}
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300"
                >
                  Open Results View
                </Link>
                <Link
                  to="/control-tower"
                  className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-blue-50 hover:border-blue-300"
                >
                  Open Control Tower
                </Link>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
