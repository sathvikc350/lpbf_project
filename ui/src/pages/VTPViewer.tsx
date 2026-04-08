import { useEffect, useMemo, useRef, useState } from "react";
import "@kitware/vtk.js/Rendering/Profiles/Geometry";

import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import vtkXMLPolyDataReader from "@kitware/vtk.js/IO/XML/XMLPolyDataReader";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkInteractorStyleTrackballCamera from "@kitware/vtk.js/Interaction/Style/InteractorStyleTrackballCamera";
import vtkPlane from "@kitware/vtk.js/Common/DataModel/Plane";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";

export type ClipAxis = "X" | "Y" | "Z";
export type ClipState = { enabled: boolean; axis: ClipAxis; t: number };

export type ScalarKey = "theta" | "overhang_mask" | "VED_J_per_mm3" | "geometry";

export type HistogramPayload = {
  scalar: "theta" | "overhang_mask" | "VED_J_per_mm3";
  min: number;
  max: number;
  counts: number[];
  bins: number;
};

type Props = {
  vtpUrl: string | null;
  scalarName: ScalarKey;
  clip?: ClipState;
  showGhostBox?: boolean;
  onHistogram?: (h: HistogramPayload | null) => void;
  onStatusText?: (s: string | null) => void;
};

type VtkArrayLike = {
  getName?: () => string;
  getRange?: () => [number, number];
  getNumberOfTuples?: () => number;
  getData?: () => Float32Array | Float64Array | number[];
} | null;

type PointDataLike = {
  getArrayByName?: (n: string) => VtkArrayLike;
  getArrays?: () => Array<{ getName?: () => string }>;
  setActiveScalars?: (n: string) => void;
};

type PolyDataLike = {
  getPointData?: () => PointDataLike;
  getBounds?: () => [number, number, number, number, number, number];
};

type VtkState = {
  grw: vtkGenericRenderWindow;
  reader: vtkXMLPolyDataReader;
  mapper: vtkMapper;
  actor: vtkActor;
  ctf: vtkColorTransferFunction;
  clipPlane: vtkPlane;
  polydata: PolyDataLike | null;
  ro: ResizeObserver | null;
};

function safeDelete(x: unknown) {
  const obj = x as { delete?: () => void };
  try {
    obj?.delete?.();
  } catch {
    // ignore
  }
}

function clamp01(x: number) {
  if (Number.isNaN(x)) return 0.5;
  return Math.max(0, Math.min(1, x));
}

function computeHistogram(values: number[], bins = 24) {
  if (!values.length) return null;

  const sorted = values.slice().sort((a, b) => a - b);
  const p = (q: number) => sorted[Math.floor(q * (sorted.length - 1))];
  const lo = p(0.02);
  const hi = p(0.98);

  const min = Number.isFinite(lo) ? lo : sorted[0];
  const max = Number.isFinite(hi) ? hi : sorted[sorted.length - 1];

  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;

  if (min === max) {
    return { min, max, counts: Array(bins).fill(0), bins };
  }

  const counts = Array(bins).fill(0);
  const span = max - min;

  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const t = (v - min) / span;
    const idx = Math.max(0, Math.min(bins - 1, Math.floor(t * bins)));
    counts[idx] += 1;
  }

  return { min, max, counts, bins };
}

function listPointArrayNames(pd: PointDataLike | undefined): string[] {
  const arrays = pd?.getArrays?.() ?? [];
  return arrays
    .map((a) => a?.getName?.())
    .filter((x): x is string => typeof x === "string" && x.trim().length > 0);
}

function resolveScalarArray(
  pd: PointDataLike | undefined,
  requested: ScalarKey
): {
  scalarKey: "theta" | "overhang_mask" | "VED_J_per_mm3";
  array: VtkArrayLike;
  resolvedName: string;
} | null {
  if (!pd) return null;

  const names = listPointArrayNames(pd);

  const tryNames = (candidateNames: string[]) => {
    for (const n of candidateNames) {
      const arr = pd.getArrayByName?.(n);
      if (arr) return { arr, name: n };
    }
    return null;
  };

  if (requested === "geometry") {
    const hit = tryNames([
      "theta",
      "theta_phys",
      "density",
      "rho",
      "design_density",
      "material_fraction",
    ]);
    if (hit) {
      return {
        scalarKey: "theta",
        array: hit.arr,
        resolvedName: hit.name,
      };
    }
    return null;
  }

  if (requested === "overhang_mask") {
    const hit = tryNames(["overhang_mask"]);
    if (hit) {
      return {
        scalarKey: "overhang_mask",
        array: hit.arr,
        resolvedName: hit.name,
      };
    }
    return null;
  }

  const vedHit = tryNames(["VED_J_per_mm3", "ved", "VED"]);
  if (vedHit) {
    return {
      scalarKey: "VED_J_per_mm3",
      array: vedHit.arr,
      resolvedName: vedHit.name,
    };
  }

  if (names.length > 0) {
    console.warn("Requested scalar not found. Available arrays:", names);
  }
  return null;
}

export default function VTPViewer({
  vtpUrl,
  scalarName,
  clip,
  onHistogram,
  onStatusText,
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const vtkRef = useRef<VtkState | null>(null);

  const hasInputRef = useRef(false);
  const [loadedTick, setLoadedTick] = useState(0);
  const lastHistKeyRef = useRef<string>("");

  const requestedScalar = useMemo<ScalarKey>(() => scalarName, [scalarName]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    if (vtkRef.current) {
      vtkRef.current.ro?.disconnect();
      safeDelete(vtkRef.current.grw);
      vtkRef.current = null;
    }
    hasInputRef.current = false;

    const grw = vtkGenericRenderWindow.newInstance({
      background: [0.15, 0.16, 0.18],
    });
    grw.setContainer(el);

    const iren = grw.getInteractor();
    iren.setInteractorStyle(vtkInteractorStyleTrackballCamera.newInstance());

    const reader = vtkXMLPolyDataReader.newInstance();
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(vtkPolyData.newInstance());

    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    actor.getProperty().setColor(0.9, 0.9, 0.9);
    actor.getProperty().setAmbient(0.2);
    actor.getProperty().setDiffuse(0.8);

    const renderer = grw.getRenderer();
    renderer.addActor(actor);

    const ctf = vtkColorTransferFunction.newInstance();
    const clipPlane = vtkPlane.newInstance();

    const ro = new ResizeObserver(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w > 10 && h > 10) {
        grw.resize();
        if (hasInputRef.current) grw.getRenderWindow().render();
      }
    });
    ro.observe(el);

    vtkRef.current = { grw, reader, mapper, actor, ctf, clipPlane, polydata: null, ro };

    requestAnimationFrame(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      if (w > 10 && h > 10) grw.resize();
    });

    return () => {
      ro.disconnect();
      safeDelete(grw);
      vtkRef.current = null;
      hasInputRef.current = false;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const st = vtkRef.current;
      if (!st || !vtpUrl) return;

      onStatusText?.("Loading VTP…");

      const res = await fetch(vtpUrl);
      if (!res.ok) throw new Error(`Failed to download VTP: ${res.status}`);

      const buf = await res.arrayBuffer();
      if (cancelled) return;

      st.reader.parseAsArrayBuffer(buf);

      const poly = st.reader.getOutputData(0) as PolyDataLike;
      st.polydata = poly;

      st.mapper.setInputData(poly as unknown as vtkPolyData);
      hasInputRef.current = true;

      st.grw.getRenderer().resetCamera();
      st.grw.getRenderWindow().render();

      onStatusText?.(null);
      setLoadedTick((x) => x + 1);
    }

    load().catch((e: unknown) => {
      const msg = e instanceof Error ? e.message : String(e);
      console.warn("VTP load failed:", msg);
      onStatusText?.(`VTP load failed: ${msg}`);
      onHistogram?.(null);
    });

    return () => {
      cancelled = true;
    };
  }, [vtpUrl, onHistogram, onStatusText]);

  useEffect(() => {
    const st = vtkRef.current;
    if (!st?.polydata) return;

    const mapperDuck = st.mapper as unknown as {
      addClippingPlane?: (p: unknown) => void;
      removeAllClippingPlanes?: () => void;
    };

    const b = st.polydata.getBounds?.();
    if (!b) return;

    const [xmin, xmax, ymin, ymax, zmin, zmax] = b;
    const t = clamp01(clip?.t ?? 0.5);
    const enabled = !!clip?.enabled;
    const axis = clip?.axis ?? "Z";

    mapperDuck.removeAllClippingPlanes?.();

    if (!enabled) {
      st.grw.getRenderWindow().render();
      return;
    }

    if (axis === "X") {
      const x = xmin + t * (xmax - xmin);
      st.clipPlane.setNormal(1, 0, 0);
      st.clipPlane.setOrigin(x, 0, 0);
    } else if (axis === "Y") {
      const y = ymin + t * (ymax - ymin);
      st.clipPlane.setNormal(0, 1, 0);
      st.clipPlane.setOrigin(0, y, 0);
    } else {
      const z = zmin + t * (zmax - zmin);
      st.clipPlane.setNormal(0, 0, 1);
      st.clipPlane.setOrigin(0, 0, z);
    }

    mapperDuck.addClippingPlane?.(st.clipPlane);
    st.grw.getRenderWindow().render();
  }, [clip?.enabled, clip?.axis, clip?.t, loadedTick]);

  useEffect(() => {
    const st = vtkRef.current;
    if (!st?.polydata) return;

    const pd = st.polydata.getPointData?.();
    const resolved = resolveScalarArray(pd, requestedScalar);

    if (!resolved) {
      const available = listPointArrayNames(pd);
      st.mapper.setScalarVisibility(false);
      if (hasInputRef.current) st.grw.getRenderWindow().render();
      onHistogram?.(null);

      if (requestedScalar === "geometry") {
        onStatusText?.(
          available.length
            ? `Geometry scalar coloring not found. Available point arrays: ${available.join(", ")}`
            : "Geometry scalar coloring not found in this VTP."
        );
      } else {
        onStatusText?.(
          available.length
            ? `Scalar "${requestedScalar}" not found. Available point arrays: ${available.join(", ")}`
            : `Scalar "${requestedScalar}" not found in this VTP.`
        );
      }
      return;
    }

    onStatusText?.(null);

    pd?.setActiveScalars?.(resolved.resolvedName);

    st.mapper.setScalarVisibility(true);
    st.mapper.setColorModeToMapScalars();
    st.mapper.setLookupTable(st.ctf);
    st.mapper.setUseLookupTableScalarRange(true);

    st.ctf.removeAllPoints();

    if (resolved.scalarKey === "overhang_mask") {
      st.ctf.addRGBPoint(0, 0.1, 0.6, 1.0);
      st.ctf.addRGBPoint(1, 1.0, 0.2, 0.2);
      st.mapper.setScalarRange(0, 1);
    } else if (resolved.scalarKey === "theta") {
      const range = resolved.array?.getRange?.() ?? [0, 1];
      const lo = range[0];
      const hi = range[1];
      const mid = (lo + hi) / 2;

      st.ctf.addRGBPoint(lo, 0.12, 0.30, 0.95);
      st.ctf.addRGBPoint(mid, 0.88, 0.90, 0.93);
      st.ctf.addRGBPoint(hi, 0.98, 0.45, 0.05);
      st.mapper.setScalarRange(lo, hi);
    } else {
      const range = resolved.array?.getRange?.() ?? [0, 1];
      const lo = range[0];
      const hi = range[1];
      const mid = (lo + hi) / 2;

      st.ctf.addRGBPoint(lo, 0.0, 0.4, 1.0);
      st.ctf.addRGBPoint(mid, 0.9, 0.9, 0.9);
      st.ctf.addRGBPoint(hi, 1.0, 0.3, 0.0);
      st.mapper.setScalarRange(lo, hi);
    }

    if (hasInputRef.current) st.grw.getRenderWindow().render();

    const raw = resolved.array?.getData?.();
    const values = raw ? Array.from(raw as ArrayLike<number>) : [];
    const h = computeHistogram(values, 24);

    if (!h) {
      onHistogram?.(null);
      return;
    }

    const payload: HistogramPayload = {
      scalar: resolved.scalarKey,
      min: h.min,
      max: h.max,
      counts: h.counts,
      bins: h.bins,
    };

    const key = `${payload.scalar}|${payload.min.toFixed(6)}|${payload.max.toFixed(6)}|${payload.counts.join(",")}`;
    if (key !== lastHistKeyRef.current) {
      lastHistKeyRef.current = key;
      onHistogram?.(payload);
    }
  }, [requestedScalar, loadedTick, onHistogram, onStatusText]);

  return (
    <div className="w-full relative">
      <div ref={containerRef} className="h-[520px] w-full rounded-xl bg-slate-900 border border-slate-200" />
      {!vtpUrl && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="rounded-lg bg-black/60 px-4 py-2 text-sm text-slate-200 border border-white/10">
            No VTP export found for this run yet.
          </div>
        </div>
      )}
    </div>
  );
}
