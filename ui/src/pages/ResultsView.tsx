// src/pages/ResultsView.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";

// vtk.js imports (manual pipeline – avoids GenericRenderWindow init crash)
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";
import vtkRenderWindow from "@kitware/vtk.js/Rendering/Core/RenderWindow";
import vtkOpenGLRenderWindow from "@kitware/vtk.js/Rendering/OpenGL/RenderWindow";
import vtkRenderWindowInteractor from "@kitware/vtk.js/Rendering/Core/RenderWindowInteractor";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import vtkXMLPolyDataReader from "@kitware/vtk.js/IO/XML/XMLPolyDataReader";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";

type ArtItem = { rel: string; download_url: string };
type ArtifactsResp = { ok: boolean; exports: ArtItem[] };

type ScalarKey = "theta" | "overhang_mask" | "VED_J_per_mm3";

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

function extractIter(rel: string): number | null {
  const m = rel.match(/physics_manifest_iter_(\d{6})\.vtp$/);
  if (!m) return null;
  return Number(m[1]);
}

function pickLatestPhysicsVtp(exportsArr: ArtItem[]): ArtItem | null {
  const candidates = exportsArr
    .filter((x) => x.rel.startsWith("physics/") && x.rel.endsWith(".vtp"))
    .map((x) => ({ x, it: extractIter(x.rel) }))
    .filter((o) => o.it != null) as Array<{ x: ArtItem; it: number }>;

  candidates.sort((a, b) => b.it - a.it);
  return candidates[0]?.x ?? null;
}

function safeDelete(x: unknown) {
  const obj = x as { delete?: () => void };
  try {
    obj?.delete?.();
  } catch {
    // ignore
  }
}

// ---- duck types (no any) ----
type VtkDataArray = {
  getName?: () => string;
  getRange?: () => [number, number];
};
type VtkPointData = {
  getArrayByName?: (n: string) => VtkDataArray | null;
  setActiveScalars?: (n: string) => void;
  getNumberOfArrays?: () => number;
  getArrayByIndex?: (i: number) => VtkDataArray | null;
};
type VtkPolyData = {
  getPointData?: () => VtkPointData | undefined;
};

type VtkCtx = {
  rw: vtkRenderWindow;
  gl: vtkOpenGLRenderWindow;
  ren: vtkRenderer;
  iren: vtkRenderWindowInteractor;
  actor: vtkActor;
  mapper: vtkMapper;
  ctf: vtkColorTransferFunction;
  polydata: VtkPolyData | null;
};

export default function ResultsView() {
  const { runId } = useParams();
  const containerRef = useRef<HTMLDivElement | null>(null);

  const [err, setErr] = useState<string | null>(null);
  const [scalar, setScalar] = useState<ScalarKey>("theta");
  const [available, setAvailable] = useState<string[]>([]);
  const [vtpUrl, setVtpUrl] = useState<string | null>(null);
  const [vtpRel, setVtpRel] = useState<string | null>(null);

  const lastRelRef = useRef<string | null>(null);
  const vtkRef = useRef<VtkCtx | null>(null);
  const initedRef = useRef(false);

  const tabs: Array<{ key: ScalarKey; label: string }> = useMemo(
    () => [
      { key: "theta", label: "Geometry (theta)" },
      { key: "overhang_mask", label: "Overhang Risk" },
      { key: "VED_J_per_mm3", label: "Thermal (VED Proxy)" },
    ],
    []
  );

  // 1) Init VTK safely (wait for container to be sized)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    if (initedRef.current) return; // strictmode/double-mount protection

    let cancelled = false;

    async function waitForSize() {
      for (let i = 0; i < 60; i++) {
        if (cancelled) return false;
        const w = el.clientWidth;
        const h = el.clientHeight;
        if (w > 10 && h > 10) return true;
        await new Promise((r) => requestAnimationFrame(() => r(null)));
      }
      return false;
    }

    (async () => {
      const ok = await waitForSize();
      if (!ok || cancelled) return;

      // cleanup old if exists
      if (vtkRef.current) {
        safeDelete(vtkRef.current.iren);
        safeDelete(vtkRef.current.gl);
        safeDelete(vtkRef.current.rw);
        vtkRef.current = null;
      }

      const rw = vtkRenderWindow.newInstance();
      const ren = vtkRenderer.newInstance({ background: [0.95, 0.96, 0.98] });
      rw.addRenderer(ren);

      const gl = vtkOpenGLRenderWindow.newInstance();
      gl.setContainer(el);
      rw.addView(gl);

      // IMPORTANT: set size before first render
      gl.setSize(Math.max(10, el.clientWidth), Math.max(10, el.clientHeight));

      const iren = vtkRenderWindowInteractor.newInstance();
      iren.setView(gl);
      iren.initialize();
      iren.bindEvents(el);

      const style = vtkInteractorStyleTrackballCamera.newInstance();
      iren.setInteractorStyle(style);

      const mapper = vtkMapper.newInstance();
      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      ren.addActor(actor);

      const ctf = vtkColorTransferFunction.newInstance();

      vtkRef.current = { rw, gl, ren, iren, actor, mapper, ctf, polydata: null };
      initedRef.current = true;

      rw.render();

      const onResize = () => {
        if (!vtkRef.current) return;
        const w = Math.max(10, el.clientWidth);
        const h = Math.max(10, el.clientHeight);
        vtkRef.current.gl.setSize(w, h);
        vtkRef.current.rw.render();
      };

      window.addEventListener("resize", onResize);

      // cleanup
      return () => {
        window.removeEventListener("resize", onResize);
      };
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  // 2) Poll artifacts (no-flicker)
  useEffect(() => {
    if (!runId) {
      setErr("Missing runId in URL.");
      return;
    }

    let alive = true;
    const pollMs = 1500;

    async function tick() {
      try {
        const res = await fetch(`/runs/${runId}/artifacts`);
        const j = (await res.json()) as ArtifactsResp;
        if (!res.ok || !j.ok) throw new Error("Failed to load artifacts.");

        const best = pickLatestPhysicsVtp(j.exports || []);
        if (!best) return;

        if (lastRelRef.current !== best.rel) {
          lastRelRef.current = best.rel;
          setVtpRel(best.rel);
          setVtpUrl(best.download_url);
        }
      } catch (e: unknown) {
        if (!alive) return;
        setErr(e instanceof Error ? e.message : String(e));
      }
    }

    setErr(null);
    void tick();

    const t = setInterval(() => void tick(), pollMs);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [runId]);

  function applyScalar(key: ScalarKey) {
    const st = vtkRef.current;
    if (!st?.polydata) return;

    const poly = st.polydata;
    const pd = poly.getPointData?.();
    if (!pd) return;

    const arr = pd.getArrayByName?.(key);
    if (!arr) {
      setErr(`Scalar "${key}" not found. Available: ${available.join(", ") || "—"}`);
      return;
    }

    pd.setActiveScalars?.(key);

    st.mapper.setScalarVisibility(true);
    st.mapper.setColorModeToMapScalars();
    st.ctf.removeAllPoints();

    if (key === "overhang_mask") {
      st.ctf.addRGBPoint(0, 0.70, 0.80, 0.95);
      st.ctf.addRGBPoint(1, 0.90, 0.20, 0.20);
      st.mapper.setLookupTable(st.ctf);
      st.mapper.setUseLookupTableScalarRange(true);
      st.mapper.setScalarRange(0, 1);
    } else {
      const r = arr.getRange?.() ?? [0, 1];
      const lo = r[0];
      const hi = r[1];
      const mid = (lo + hi) / 2;

      st.ctf.addRGBPoint(lo, 0.10, 0.20, 0.80);
      st.ctf.addRGBPoint(mid, 0.85, 0.85, 0.85);
      st.ctf.addRGBPoint(hi, 0.80, 0.10, 0.10);

      st.mapper.setLookupTable(st.ctf);
      st.mapper.setUseLookupTableScalarRange(true);
      st.mapper.setScalarRange(lo, hi);
    }

    st.rw.render();
  }

  // 3) Load VTP when URL changes
  useEffect(() => {
    let cancelled = false;

    async function load() {
      setErr(null);
      setAvailable([]);

      const st = vtkRef.current;
      if (!st) return;
      if (!vtpUrl) return;

      try {
        const res = await fetch(vtpUrl);
        if (!res.ok) throw new Error(`Failed to download VTP: ${res.status}`);
        const buf = await res.arrayBuffer();
        if (cancelled) return;

        const reader = vtkXMLPolyDataReader.newInstance();
        reader.parseAsArrayBuffer(buf);

        const out = reader.getOutputData(0);
        const poly = out as unknown as VtkPolyData;

        st.polydata = poly;
        st.mapper.setInputData(out);

        const pd = poly.getPointData?.();
        const names: string[] = [];
        const n = pd?.getNumberOfArrays?.() ?? 0;
        for (let i = 0; i < n; i++) {
          const a = pd?.getArrayByIndex?.(i);
          const nm = a?.getName?.();
          if (nm) names.push(nm);
        }
        setAvailable(names);

        st.ren.resetCamera();
        applyScalar(scalar);
        st.rw.render();
      } catch (e: unknown) {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [vtpUrl]);

  // 4) Apply scalar when user changes tab
  useEffect(() => {
    if (!vtkRef.current?.polydata) return;
    applyScalar(scalar);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scalar]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-400 font-semibold">
            Results View
          </div>
          <div className="text-lg font-semibold text-slate-100">
            Run: <span className="font-mono">{runId ?? "—"}</span>
          </div>
          <div className="text-xs text-slate-400">
            VTP: <span className="font-mono break-all">{vtpRel ?? "waiting…"}</span>
          </div>
        </div>

        <div className="flex gap-2">
          {tabs.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setScalar(t.key)}
              className={cx(
                "rounded-lg px-3 py-2 text-sm border transition",
                scalar === t.key
                  ? "bg-sky-500/15 text-sky-100 border-sky-400/30"
                  : "bg-white/5 text-slate-200 border-white/10 hover:bg-white/10"
              )}
              title={
                available.length
                  ? available.includes(t.key)
                    ? "available"
                    : "not in file"
                  : "loading"
              }
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {err && (
        <div className="rounded-xl border border-rose-400/30 bg-rose-500/10 p-3 text-sm text-rose-100 break-words">
          {err}
        </div>
      )}

      <div className="rounded-2xl border border-white/10 bg-white/5 p-3">
        <div className="text-xs text-slate-400 mb-2">
          Scalars in file:{" "}
          <span className="font-mono">{available.join(", ") || "loading…"}</span>
        </div>

        <div
          ref={containerRef}
          className="h-[520px] w-full rounded-xl bg-slate-100 overflow-hidden"
        />
      </div>
    </div>
  );
}
