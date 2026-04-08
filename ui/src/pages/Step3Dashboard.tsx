import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, Link } from "react-router-dom";
import VTPViewer from "./VTPViewer";
import type { ClipAxis, ClipState, HistogramPayload, ScalarKey } from "./VTPViewer";

type ScalarMode = "geometry" | "overhang" | "thermal";
type ViewMode = "physics" | "preview" | "watertight";

type ArtifactsResp = {
  ok: boolean;
  run_id: string;
  exports: Array<{
    rel: string;
    download_url: string;
    path: string;
    mtime_unix_s?: number;
    bytes?: number;
  }>;
};

type LiveSolverInfo = {
  stage?: string | null;
  stage_iter?: number | null;
  stage_total?: number | null;
  stage_pct?: number | null;
  detail?: string | null;
  log_mtime_unix_s?: number | null;
};

type LatestResp = {
  ok: boolean;
  run: {
    run_id: string;
    status: string;
    last_iter: number;
    total_iters: number;
    material: string;
    voxel_size_mm: number;
  };
  progress_pct?: number;
  live_solver?: LiveSolverInfo;
};

type ArtItem = ArtifactsResp["exports"][number];

function extractPhysicsIter(rel: string): number | null {
  const m = rel.match(/physics_manifest_iter_(\d{6})\.vtp$/);
  if (!m) return null;
  return Number(m[1]);
}

function extractPreviewIter(rel: string): number | null {
  const m = rel.match(/preview_iter_(\d{6})_lvl_[\d.]+\.stl$/);
  if (!m) return null;
  return Number(m[1]);
}

function extractWatertightIter(rel: string): number | null {
  const m = rel.match(/preview_watertight_iter_(\d{6})_lvl_[\d.]+_pitch_[\d.]+\.stl$/);
  if (!m) return null;
  return Number(m[1]);
}

function pickLatestVtp(exportsArr: ArtifactsResp["exports"]) {
  const candidates = exportsArr
    .filter((x) => x.rel.startsWith("physics/") && x.rel.endsWith(".vtp"))
    .map((x) => ({ ...x, it: extractPhysicsIter(x.rel) }))
    .filter((x) => x.it != null) as Array<ArtItem & { it: number }>;
  candidates.sort((a, b) => b.it - a.it);
  return candidates[0] ?? null;
}

function pickLatestPreviewStl(exportsArr: ArtifactsResp["exports"]) {
  const candidates = exportsArr
    .filter((x) => x.rel.startsWith("preview/") && x.rel.endsWith(".stl"))
    .map((x) => ({ ...x, it: extractPreviewIter(x.rel) }))
    .filter((x) => x.it != null) as Array<ArtItem & { it: number }>;
  candidates.sort((a, b) => b.it - a.it);
  return candidates[0] ?? null;
}

function pickLatestWatertightStl(exportsArr: ArtifactsResp["exports"]) {
  const candidates = exportsArr
    .filter(
      (x) =>
        x.rel.startsWith("preview_watertight/") &&
        x.rel.endsWith(".stl") &&
        !x.rel.includes("/_tmp_surface_")
    )
    .map((x) => ({ ...x, it: extractWatertightIter(x.rel) }))
    .filter((x) => x.it != null) as Array<ArtItem & { it: number }>;
  candidates.sort((a, b) => b.it - a.it);
  return candidates[0] ?? null;
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal });
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

function fmt(n: number) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(2);
  return n.toFixed(4);
}

function bytesFmt(n?: number) {
  if (!n || !Number.isFinite(n)) return "—";
  const kb = 1024;
  const mb = kb * 1024;
  if (n >= mb) return `${(n / mb).toFixed(2)} MB`;
  if (n >= kb) return `${(n / kb).toFixed(1)} KB`;
  return `${n} B`;
}

function legendTitleForMode(mode: ScalarMode) {
  if (mode === "geometry") return "Geometry / Density";
  if (mode === "overhang") return "Overhang Risk";
  return "VED Thermal Proxy";
}

function legendBodyForMode(mode: ScalarMode, hist: HistogramPayload | null) {
  if (mode === "geometry") {
    return (
      <>
        <div>Blue = lower density / lower theta</div>
        <div>Light gray = mid density</div>
        <div>Orange = higher density / higher theta</div>
        {hist ? (
          <div className="mt-1 text-slate-500">
            Current scalar source: <span className="font-mono">{hist.scalar}</span>
          </div>
        ) : null}
      </>
    );
  }

  if (mode === "overhang") {
    return (
      <>
        <div>Blue = safer / lower overhang flag</div>
        <div>Red = overhang-risk region</div>
        <div className="mt-1 text-slate-500">Scalar source: overhang_mask</div>
      </>
    );
  }

  return (
    <>
      <div>Blue = lower VED</div>
      <div>Light gray = mid VED</div>
      <div>Orange = higher VED</div>
      <div className="mt-1 text-slate-500">Scalar source: VED_J_per_mm3</div>
    </>
  );
}

export default function Step3Dashboard({
  pollMs = 1500,
  runId: runIdProp,
  embedded = false,
}: {
  pollMs?: number;
  runId?: string | null;
  embedded?: boolean;
}) {
  const params = useParams();
  const runId = runIdProp ?? (params.runId as string | undefined);

  const [err, setErr] = useState<string | null>(null);
  const [statusText, setStatusText] = useState<string | null>(null);
  const [latest, setLatest] = useState<LatestResp | null>(null);

  const [viewMode, setViewMode] = useState<ViewMode>("physics");
  const [scalarMode, setScalarMode] = useState<ScalarMode>("geometry");
  const [clip, setClip] = useState<ClipState>({ enabled: false, axis: "Z", t: 0.5 });
  const [hist, setHist] = useState<HistogramPayload | null>(null);

  const [vtpUrl, setVtpUrl] = useState<string | null>(null);
  const [vtpRel, setVtpRel] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewRel, setPreviewRel] = useState<string | null>(null);
  const [previewBytes, setPreviewBytes] = useState<number | undefined>(undefined);
  const [watertightUrl, setWatertightUrl] = useState<string | null>(null);
  const [watertightRel, setWatertightRel] = useState<string | null>(null);
  const [watertightBytes, setWatertightBytes] = useState<number | undefined>(undefined);

  const lastVtpRelRef = useRef<string | null>(null);

  const scalarName = useMemo<ScalarKey>(() => {
    if (scalarMode === "overhang") return "overhang_mask";
    if (scalarMode === "thermal") return "VED_J_per_mm3";
    return "geometry";
  }, [scalarMode]);

  const tick = useCallback(
    async (signal?: AbortSignal) => {
      if (!runId) return;

      const l = await getJson<LatestResp>(`/runs/${runId}/latest`, signal);
      setLatest(l);

      const a = await getJson<ArtifactsResp>(`/runs/${runId}/artifacts`, signal);
      const bestVtp = pickLatestVtp(a.exports || []);
      const bestPreview = pickLatestPreviewStl(a.exports || []);
      const bestWatertight = pickLatestWatertightStl(a.exports || []);

      setPreviewRel(bestPreview?.rel ?? null);
      setPreviewUrl(bestPreview?.download_url ?? null);
      setPreviewBytes(bestPreview?.bytes);

      setWatertightRel(bestWatertight?.rel ?? null);
      setWatertightUrl(bestWatertight?.download_url ?? null);
      setWatertightBytes(bestWatertight?.bytes);

      if (!bestVtp) return;

      if (lastVtpRelRef.current !== bestVtp.rel) {
        lastVtpRelRef.current = bestVtp.rel;
        setVtpRel(bestVtp.rel);
        setVtpUrl(bestVtp.download_url);
        setHist(null);
      }
    },
    [runId]
  );

  useEffect(() => {
    if (!runId) return;

    let alive = true;
    const ctl = new AbortController();

    (async () => {
      setErr(null);
      try {
        await tick(ctl.signal);
        while (alive) {
          await new Promise((r) => setTimeout(r, pollMs));
          await tick(ctl.signal);
        }
      } catch (e: unknown) {
        if (!alive) return;
        const msg = e instanceof Error ? e.message : String(e);
        setErr(msg);
      }
    })();

    return () => {
      alive = false;
      ctl.abort();
    };
  }, [runId, tick, pollMs]);

  const onHistogram = useCallback((h: HistogramPayload | null) => {
    setHist(h);
  }, []);

  if (!runId) return <div className="text-slate-500 text-sm">Start a run to see live results.</div>;

  const status = latest?.run?.status ?? "—";
  const prog = Math.max(0, Math.min(100, latest?.progress_pct ?? 0));
  const liveSolver = latest?.live_solver ?? null;
  const liveStage = liveSolver?.stage ?? "—";
  const liveDetail = liveSolver?.detail ?? null;
  const liveStagePct =
    typeof liveSolver?.stage_pct === "number" && Number.isFinite(liveSolver.stage_pct)
      ? Math.max(0, Math.min(100, liveSolver.stage_pct))
      : null;

  const previewAvailable = !!previewUrl;
  const watertightAvailable = !!watertightUrl;
  const isDone = (status || "").toUpperCase() === "DONE";

  return (
    <section className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-lg font-semibold text-slate-800">Results & Analytics</div>
          <div className="mt-1 text-xs text-slate-500">
            run_id: <span className="font-mono">{runId}</span>
          </div>
          {!embedded && (
            <div className="mt-1 text-xs text-slate-500">
              <Link to="/results" className="text-blue-600 hover:underline">
                ← Back to Results list
              </Link>
            </div>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setViewMode("physics")}
            className={[
              "px-3 py-1.5 text-sm rounded-md border transition",
              viewMode === "physics"
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
            ].join(" ")}
          >
            Physics
          </button>
          <button
            type="button"
            onClick={() => setViewMode("preview")}
            className={[
              "px-3 py-1.5 text-sm rounded-md border transition",
              viewMode === "preview"
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
            ].join(" ")}
          >
            Preview STL
          </button>
          <button
            type="button"
            onClick={() => setViewMode("watertight")}
            className={[
              "px-3 py-1.5 text-sm rounded-md border transition",
              viewMode === "watertight"
                ? "bg-blue-600 text-white border-blue-600"
                : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
            ].join(" ")}
          >
            Watertight STL
          </button>
        </div>
      </div>

      <div className="mt-2 text-xs text-slate-500 space-y-1">
        <div>physics: <span className="font-mono">{vtpRel ?? "none yet"}</span></div>
        <div>preview: <span className="font-mono">{previewRel ?? "none yet"}</span></div>
        <div>watertight: <span className="font-mono">{watertightRel ?? "none yet"}</span></div>
      </div>

      {err ? (
        <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800">
          {err}
        </div>
      ) : null}

      {statusText && viewMode === "physics" ? (
        <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-700">
          {statusText}
        </div>
      ) : null}

      <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1 border border-slate-200 rounded-xl p-4 bg-slate-50">
          <div className="text-sm font-semibold text-slate-700">Run Stats</div>

          <div className="mt-3 space-y-2 text-sm text-slate-700">
            <div className="flex justify-between">
              <span>Status</span>
              <span className="font-semibold">{status}</span>
            </div>

            <div className="flex justify-between">
              <span>Progress</span>
              <span className="font-semibold">{prog.toFixed(1)}%</span>
            </div>

            <div className="mt-1">
              <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                <div
                  className="h-full rounded-full bg-blue-600 transition-all duration-500"
                  style={{ width: `${prog}%` }}
                />
              </div>
            </div>

            <div className="flex justify-between">
              <span>Live solver phase</span>
              <span className="font-semibold">{liveStage}</span>
            </div>

            {liveDetail ? (
              <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
                {liveDetail}
              </div>
            ) : null}

            {liveStagePct != null ? (
              <div>
                <div className="mb-1 flex justify-between text-xs text-slate-600">
                  <span>Current phase progress</span>
                  <span>{liveStagePct.toFixed(0)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-slate-200 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                    style={{ width: `${liveStagePct}%` }}
                  />
                </div>
              </div>
            ) : null}

            <div className="flex justify-between">
              <span>Material</span>
              <span className="font-semibold">{latest?.run?.material ?? "—"}</span>
            </div>
            <div className="flex justify-between">
              <span>Iteration</span>
              <span className="font-semibold">
                {latest?.run?.last_iter ?? 0}/{latest?.run?.total_iters ?? 0}
              </span>
            </div>
          </div>

          {viewMode === "physics" ? (
            <>
              <div className="mt-4 border-t border-slate-200 pt-4">
                <div className="text-sm font-semibold text-slate-700">View Layer</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  <button
                    type="button"
                    onClick={() => setScalarMode("geometry")}
                    className={[
                      "px-3 py-1.5 text-sm rounded-md border transition",
                      scalarMode === "geometry"
                        ? "bg-blue-600 text-white border-blue-600"
                        : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
                    ].join(" ")}
                  >
                    Geometry
                  </button>
                  <button
                    type="button"
                    onClick={() => setScalarMode("overhang")}
                    className={[
                      "px-3 py-1.5 text-sm rounded-md border transition",
                      scalarMode === "overhang"
                        ? "bg-blue-600 text-white border-blue-600"
                        : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
                    ].join(" ")}
                  >
                    Overhang
                  </button>
                  <button
                    type="button"
                    onClick={() => setScalarMode("thermal")}
                    className={[
                      "px-3 py-1.5 text-sm rounded-md border transition",
                      scalarMode === "thermal"
                        ? "bg-blue-600 text-white border-blue-600"
                        : "bg-white text-slate-700 border-slate-200 hover:bg-slate-50",
                    ].join(" ")}
                  >
                    VED
                  </button>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-slate-200 bg-white p-4">
                <div className="text-sm font-semibold text-slate-700">Legend</div>
                <div className="mt-2 text-sm text-slate-700">
                  Active view: <span className="font-semibold">{legendTitleForMode(scalarMode)}</span>
                </div>
                <div className="mt-2 space-y-1 text-sm text-slate-600">
                  {legendBodyForMode(scalarMode, hist)}
                </div>
              </div>

              <div className="mt-4 border-t border-slate-200 pt-4">
                <div className="text-sm font-semibold text-slate-700">Clipping Plane</div>
                <label className="mt-2 flex items-center gap-2 text-sm text-slate-700">
                  <input
                    type="checkbox"
                    checked={clip.enabled}
                    onChange={(e) => setClip((c) => ({ ...c, enabled: e.target.checked }))}
                  />
                  Enable clipping plane
                </label>
                <div className="mt-2 flex items-center gap-2">
                  <select
                    className="rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm"
                    value={clip.axis}
                    onChange={(e) => setClip((c) => ({ ...c, axis: e.target.value as ClipAxis }))}
                    disabled={!clip.enabled}
                  >
                    <option value="X">X</option>
                    <option value="Y">Y</option>
                    <option value="Z">Z</option>
                  </select>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.01}
                    value={clip.t}
                    onChange={(e) => setClip((c) => ({ ...c, t: Number(e.target.value) }))}
                    className="w-full"
                    disabled={!clip.enabled}
                  />
                </div>
                <div className="mt-1 text-xs text-slate-500">Slice position: {(clip.t * 100).toFixed(0)}%</div>
              </div>

              <div className="mt-4 border-t border-slate-200 pt-4">
                <div className="text-sm font-semibold text-slate-700">Histogram</div>
                <div className="mt-1 text-xs text-slate-500">
                  Live scalar histogram for the active view.
                </div>

                {hist ? (
                  <>
                    <div className="mt-1 text-xs text-slate-500">
                      Scalar: <span className="font-mono">{hist.scalar}</span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">
                      Range: <span className="font-mono">{fmt(hist.min)}</span> →{" "}
                      <span className="font-mono">{fmt(hist.max)}</span>
                    </div>
                    <div className="mt-2 h-28 rounded-lg bg-white border border-slate-200 flex items-end gap-[2px] p-2">
                      {hist.counts.map((c, i) => {
                        const maxC = Math.max(1, ...hist.counts);
                        const hPct = (c / maxC) * 100;
                        return (
                          <div
                            key={i}
                            title={`bin ${i + 1}/${hist.bins}: ${c}`}
                            className="flex-1 rounded-sm bg-blue-600"
                            style={{ height: `${hPct}%` }}
                          />
                        );
                      })}
                    </div>
                    <div className="mt-1 flex justify-between text-[10px] text-slate-500">
                      <span>{fmt(hist.min)}</span>
                      <span>{fmt((hist.min + hist.max) / 2)}</span>
                      <span>{fmt(hist.max)}</span>
                    </div>
                  </>
                ) : (
                  <div className="mt-2 h-28 rounded-lg bg-white border border-slate-200 flex items-center justify-center text-xs text-slate-500">
                    Waiting for scalar data…
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="mt-4 border-t border-slate-200 pt-4">
              <div className="text-sm font-semibold text-slate-700">Manufacturing Export</div>
              {viewMode === "preview" ? (
                previewAvailable ? (
                  <div className="mt-3 space-y-2 text-sm text-slate-700">
                    <div>File: <span className="font-mono break-all">{previewRel}</span></div>
                    <div>Size: {bytesFmt(previewBytes)}</div>
                    <a
                      href={previewUrl ?? "#"}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                    >
                      Open / Download Preview STL
                    </a>
                  </div>
                ) : (
                  <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                    Preview STL not available for this run yet.
                  </div>
                )
              ) : watertightAvailable ? (
                <div className="mt-3 space-y-2 text-sm text-slate-700">
                  <div>File: <span className="font-mono break-all">{watertightRel}</span></div>
                  <div>Size: {bytesFmt(watertightBytes)}</div>
                  <a
                    href={watertightUrl ?? "#"}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                  >
                    Open / Download Watertight STL
                  </a>
                </div>
              ) : (
                <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                  Watertight STL not available for this run yet.
                </div>
              )}

              <div className="mt-4 border-t border-slate-200 pt-4 flex flex-wrap gap-2">
                <a
                  href={watertightUrl ?? previewUrl ?? "#"}
                  target="_blank"
                  rel="noreferrer"
                  className={[
                    "inline-flex rounded-lg px-3 py-2 text-sm font-medium border",
                    isDone && (watertightUrl || previewUrl)
                      ? "bg-blue-600 text-white border-blue-600 hover:bg-blue-700"
                      : "bg-slate-200 text-slate-500 border-slate-200 pointer-events-none",
                  ].join(" ")}
                >
                  Download STL
                </a>
                <button
                  type="button"
                  disabled={!isDone}
                  className={[
                    "inline-flex rounded-lg px-3 py-2 text-sm font-medium border",
                    isDone
                      ? "bg-white text-slate-700 border-slate-200 hover:bg-slate-50"
                      : "bg-slate-200 text-slate-500 border-slate-200",
                  ].join(" ")}
                  onClick={() => alert("Export Summary: next step")}
                >
                  Export Summary
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 border border-slate-200 rounded-xl p-4 bg-slate-50">
          {viewMode === "physics" ? (
            <>
              <div className="text-sm font-semibold text-slate-700">3D Viewer</div>
              <div className="mt-1 text-sm text-slate-500">
                Modes available: Geometry, Overhang, and VED thermal proxy.
              </div>
              <div className="mt-3">
                <VTPViewer
                  vtpUrl={vtpUrl}
                  scalarName={scalarName}
                  clip={clip}
                  onHistogram={onHistogram}
                  onStatusText={setStatusText}
                />
              </div>
              <div className="mt-3 text-xs text-slate-500">
                Note: “VED” is a proxy thermal field and overhang is a rule-based printability diagnostic.
              </div>
            </>
          ) : viewMode === "preview" ? (
            <>
              <div className="text-sm font-semibold text-slate-700">Preview STL</div>
              <div className="mt-3 rounded-xl border border-slate-200 bg-white p-6">
                {previewAvailable ? (
                  <div className="space-y-3">
                    <div className="text-sm text-slate-700">Preview STL is available for this run.</div>
                    <div className="text-xs text-slate-500 font-mono break-all">{previewRel}</div>
                    <a
                      href={previewUrl ?? "#"}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                    >
                      Open / Download Preview STL
                    </a>
                  </div>
                ) : (
                  <div className="text-sm text-slate-500">Preview STL not available.</div>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="text-sm font-semibold text-slate-700">Watertight STL</div>
              <div className="mt-3 rounded-xl border border-slate-200 bg-white p-6">
                {watertightAvailable ? (
                  <div className="space-y-3">
                    <div className="text-sm text-slate-700">Watertight STL is available for this run.</div>
                    <div className="text-xs text-slate-500 font-mono break-all">{watertightRel}</div>
                    <a
                      href={watertightUrl ?? "#"}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700"
                    >
                      Open / Download Watertight STL
                    </a>
                  </div>
                ) : (
                  <div className="text-sm text-slate-500">Watertight STL not available.</div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
