// src/pages/DesignSetup.tsx
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction, HTMLAttributes } from "react";
import Step3Dashboard from "./Step3Dashboard";

type DomainCubeOk = {
  ok: true;
  path: string;
  grid: [number, number, number];
  voxel_size_mm: number;
};

type RunCreateResp = { run_id: string };

type RunStatusResp = {
  ok: boolean;
  run: {
    run_id: string;
    status: string;
  };
};

type GeometryUploadResp = {
  ok: true;
  filename: string;
  stored_path: string;
  bbox_mm: { min: [number, number, number]; max: [number, number, number] };
  dims_mm: [number, number, number];
  voxel_size_mm_used: number;
  grid_estimate: [number, number, number];
  fits_128_cubed: boolean;
  suggested_voxel_size_mm: number;
};

type DomainWriteResp = { ok: true; path: string };

type Capabilities = {
  ok: boolean;
  materials: string[];
  validity_windows: Record<
    string,
    {
      P_W?: [number, number] | null;
      v_mm_per_s?: [number, number] | null;
      h_mm?: [number, number] | null;
      t_mm?: [number, number] | null;
      VED_J_per_mm3?: [number, number] | null;
      LED_J_per_mm?: [number, number] | null;
    }
  >;
};

type ProcessWriteResp = { ok: true; path: string };
type OptWriteResp = { ok: true; path: string };

type TopTab = "design" | "material" | "process";

type RegionBox = {
  type: "box";
  name: string;
  bounds_ijk: {
    i: [number, number];
    j: [number, number];
    k: [number, number];
  };
};

type LoadBox = RegionBox & {
  force: {
    fx: number;
    fy: number;
    fz: number;
  };
  magnitude: number;
  units_force: "arb";
};

type RegionDraft = {
  id: string;
  name: string;
  i0: number;
  i1: number;
  j0: number;
  j1: number;
  k0: number;
  k1: number;
};

type LoadDraft = RegionDraft & {
  fx: number;
  fy: number;
  fz: number;
  magnitude: number;
};

type CommittedDomainMeta = {
  sourceLabel: string | null;
  setupLabel: string | null;
  isStlSeed: boolean;
  supportCount: number;
  loadCount: number;
  nonDesignCount: number;
};

const ACTIVE_RUN_STORAGE_KEY = "lpbf_active_run_id";

async function apiGetJson<T>(url: string): Promise<T> {
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

async function apiPostJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  if (!res.ok) {
    try {
      const j = JSON.parse(text) as { detail?: string };
      throw new Error(j.detail || `POST ${url} failed: ${res.status}`);
    } catch {
      throw new Error(text || `POST ${url} failed: ${res.status}`);
    }
  }
  return JSON.parse(text) as T;
}

async function apiPostNoBody(url: string): Promise<void> {
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

async function apiPostForm<T>(url: string, form: FormData): Promise<T> {
  const res = await fetch(url, { method: "POST", body: form });
  const text = await res.text();
  if (!res.ok) {
    try {
      const j = JSON.parse(text) as { detail?: string };
      throw new Error(j.detail || `POST ${url} failed: ${res.status}`);
    } catch {
      throw new Error(text || `POST ${url} failed: ${res.status}`);
    }
  }
  return JSON.parse(text) as T;
}

function fmt(n: number, digits = 2) {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function midpoint(r?: [number, number] | null) {
  if (!r || r.length !== 2) return null;
  return (r[0] + r[1]) / 2;
}

function scrollToId(id: string) {
  requestAnimationFrame(() => {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  });
}

function parseNumberInput(s: string): number | null {
  const trimmed = s.trim();
  if (trimmed === "") return null;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : null;
}

function makeId() {
  return Math.random().toString(36).slice(2, 10);
}

function clampSpan(lo: number, hi: number, n: number): [number, number] {
  const lo2 = Math.max(0, Math.min(Math.floor(lo), n));
  const hi2 = Math.max(0, Math.min(Math.ceil(hi), n));
  return hi2 < lo2 ? [hi2, lo2] : [lo2, hi2];
}

function toRegionBox(d: RegionDraft): RegionBox {
  return {
    type: "box",
    name: d.name || "region",
    bounds_ijk: {
      i: [Math.round(d.i0), Math.round(d.i1)],
      j: [Math.round(d.j0), Math.round(d.j1)],
      k: [Math.round(d.k0), Math.round(d.k1)],
    },
  };
}

function toLoadBox(d: LoadDraft): LoadBox {
  return {
    type: "box",
    name: d.name || "load",
    bounds_ijk: {
      i: [Math.round(d.i0), Math.round(d.i1)],
      j: [Math.round(d.j0), Math.round(d.j1)],
      k: [Math.round(d.k0), Math.round(d.k1)],
    },
    force: {
      fx: Number(d.fx),
      fy: Number(d.fy),
      fz: Number(d.fz),
    },
    magnitude: Number(d.magnitude),
    units_force: "arb",
  };
}

function makeStarterSupports(grid: [number, number, number]): RegionDraft[] {
  const [, ny, nz] = grid;
  return [
    {
      id: makeId(),
      name: "Base Support",
      i0: 0,
      i1: 1,
      j0: 0,
      j1: ny,
      k0: 0,
      k1: nz,
    },
  ];
}

function makeStarterLoads(grid: [number, number, number]): LoadDraft[] {
  const [nx, ny, nz] = grid;
  const i = clampSpan(0.88 * nx, nx, nx);
  const j = clampSpan(0.44 * ny, 0.56 * ny, ny);
  const k = clampSpan(0.44 * nz, 0.56 * nz, nz);
  return [
    {
      id: makeId(),
      name: "Applied Load",
      i0: i[0],
      i1: i[1],
      j0: j[0],
      j1: j[1],
      k0: k[0],
      k1: k[1],
      fx: 0,
      fy: 0,
      fz: -1,
      magnitude: 1.0,
    },
  ];
}

function makeStarterNonDesign(): RegionDraft[] {
  return [];
}

function NumericTextInput({
  value,
  onCommit,
  className,
  inputMode = "decimal",
}: {
  value: number;
  onCommit: (v: number) => void;
  className?: string;
  inputMode?: HTMLAttributes<HTMLInputElement>["inputMode"];
}) {
  const [text, setText] = useState(String(value));

  useEffect(() => {
    setText(String(value));
  }, [value]);

  return (
    <input
      className={className}
      type="text"
      inputMode={inputMode}
      value={text}
      onFocus={(e) => e.currentTarget.select()}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => {
        const n = Number(text);
        if (Number.isFinite(n)) {
          onCommit(n);
          setText(String(n));
        } else {
          setText(String(value));
        }
      }}
    />
  );
}

function RegionEditor({
  title,
  items,
  setItems,
  draftGrid,
  namePrefix,
}: {
  title: string;
  items: RegionDraft[];
  setItems: Dispatch<SetStateAction<RegionDraft[]>>;
  draftGrid: [number, number, number] | null;
  namePrefix: string;
}) {
  function update(id: string, patch: Partial<RegionDraft>) {
    setItems((prev) => prev.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  }

  function addOne() {
    setItems((prev) => [
      ...prev,
      {
        id: makeId(),
        name: `${namePrefix}_${prev.length + 1}`,
        i0: 0,
        i1: 1,
        j0: 0,
        j1: 1,
        k0: 0,
        k1: 1,
      },
    ]);
  }

  function removeOne(id: string) {
    setItems((prev) => prev.filter((x) => x.id !== id));
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-800">{title}</div>
          <div className="text-xs text-slate-500 mt-1">
            Box regions are authored directly by the user and written to domain.json as entered.
          </div>
        </div>
        <button
          type="button"
          className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
          onClick={addOne}
        >
          Add
        </button>
      </div>

      {draftGrid && (
        <div className="mt-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
          Active grid: ({draftGrid[0]}, {draftGrid[1]}, {draftGrid[2]})
        </div>
      )}

      <div className="mt-3 space-y-3">
        {items.length === 0 ? (
          <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-xs text-slate-500">
            No regions defined.
          </div>
        ) : (
          items.map((it, idx) => (
            <div key={it.id} className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs font-semibold text-slate-800">
                  {title} #{idx + 1}
                </div>
                <button
                  type="button"
                  className="px-2 py-1 rounded-md border border-slate-200 bg-white text-xs text-slate-600 transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                  onClick={() => removeOne(it.id)}
                >
                  Remove
                </button>
              </div>

              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                <label className="text-xs font-semibold text-slate-700">
                  Region Name
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    type="text"
                    value={it.name}
                    onChange={(e) => update(it.id, { name: e.target.value })}
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  X Start Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.i0}
                    onCommit={(v) => update(it.id, { i0: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  X End Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.i1}
                    onCommit={(v) => update(it.id, { i1: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Y Start Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.j0}
                    onCommit={(v) => update(it.id, { j0: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Y End Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.j1}
                    onCommit={(v) => update(it.id, { j1: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Z Start Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.k0}
                    onCommit={(v) => update(it.id, { k0: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Z End Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.k1}
                    onCommit={(v) => update(it.id, { k1: v })}
                    inputMode="numeric"
                  />
                </label>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function LoadEditor({
  items,
  setItems,
  draftGrid,
}: {
  items: LoadDraft[];
  setItems: Dispatch<SetStateAction<LoadDraft[]>>;
  draftGrid: [number, number, number] | null;
}) {
  function update(id: string, patch: Partial<LoadDraft>) {
    setItems((prev) => prev.map((x) => (x.id === id ? { ...x, ...patch } : x)));
  }

  function addOne() {
    setItems((prev) => [
      ...prev,
      {
        id: makeId(),
        name: `load_${prev.length + 1}`,
        i0: 0,
        i1: 1,
        j0: 0,
        j1: 1,
        k0: 0,
        k1: 1,
        fx: 0,
        fy: 0,
        fz: -1,
        magnitude: 1.0,
      },
    ]);
  }

  function removeOne(id: string) {
    setItems((prev) => prev.filter((x) => x.id !== id));
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-center justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-800">Load Regions</div>
          <div className="text-xs text-slate-500 mt-1">
            Load boxes, force direction, and magnitude are authored directly by the user.
          </div>
        </div>
        <button
          type="button"
          className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
          onClick={addOne}
        >
          Add
        </button>
      </div>

      {draftGrid && (
        <div className="mt-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
          Active grid: ({draftGrid[0]}, {draftGrid[1]}, {draftGrid[2]})
        </div>
      )}

      <div className="mt-3 space-y-3">
        {items.length === 0 ? (
          <div className="rounded-lg border border-slate-200 bg-white px-3 py-3 text-xs text-slate-500">
            No loads defined.
          </div>
        ) : (
          items.map((it, idx) => (
            <div key={it.id} className="rounded-lg border border-slate-200 bg-white p-3">
              <div className="flex items-center justify-between gap-2">
                <div className="text-xs font-semibold text-slate-800">Load #{idx + 1}</div>
                <button
                  type="button"
                  className="px-2 py-1 rounded-md border border-slate-200 bg-white text-xs text-slate-600 transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                  onClick={() => removeOne(it.id)}
                >
                  Remove
                </button>
              </div>

              <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3">
                <label className="text-xs font-semibold text-slate-700">
                  Load Name
                  <input
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    type="text"
                    value={it.name}
                    onChange={(e) => update(it.id, { name: e.target.value })}
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  X Start Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.i0}
                    onCommit={(v) => update(it.id, { i0: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  X End Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.i1}
                    onCommit={(v) => update(it.id, { i1: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Y Start Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.j0}
                    onCommit={(v) => update(it.id, { j0: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Y End Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.j1}
                    onCommit={(v) => update(it.id, { j1: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Z Start Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.k0}
                    onCommit={(v) => update(it.id, { k0: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Z End Index
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.k1}
                    onCommit={(v) => update(it.id, { k1: v })}
                    inputMode="numeric"
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Force X
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.fx}
                    onCommit={(v) => update(it.id, { fx: v })}
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Force Y
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.fy}
                    onCommit={(v) => update(it.id, { fy: v })}
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Force Z
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.fz}
                    onCommit={(v) => update(it.id, { fz: v })}
                  />
                </label>

                <label className="text-xs font-semibold text-slate-700">
                  Force Magnitude
                  <NumericTextInput
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-900"
                    value={it.magnitude}
                    onCommit={(v) => update(it.id, { magnitude: v })}
                  />
                </label>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function DomainSummaryCard({
  hasCommittedDomain,
  sourceLabel,
  dimsMm,
  voxelSizeMm,
  grid,
  utilizationPct,
  bboxText,
  isStlSeed,
  setupLabel,
  supportCount,
  loadCount,
  nonDesignCount,
}: {
  hasCommittedDomain: boolean;
  sourceLabel: string | null;
  dimsMm: [number, number, number] | null;
  voxelSizeMm: number | null;
  grid: [number, number, number] | null;
  utilizationPct: number | null;
  bboxText: string | null;
  isStlSeed: boolean;
  setupLabel: string | null;
  supportCount: number;
  loadCount: number;
  nonDesignCount: number;
}) {
  return (
    <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex items-center justify-between">
        <div className="text-[11px] uppercase tracking-wide text-slate-500">Committed Domain Summary</div>
        <div className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600">
          {sourceLabel ?? "No domain yet"}
        </div>
      </div>

      {!hasCommittedDomain ? (
        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          Step 1 has not been committed yet. Use <span className="font-semibold">Use Cube Domain</span> or <span className="font-semibold">Use Uploaded STL Domain</span> first.
        </div>
      ) : (
        <div className="mt-3 space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Envelope</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">
                {dimsMm ? `${fmt(dimsMm[0], 2)} × ${fmt(dimsMm[1], 2)} × ${fmt(dimsMm[2], 2)} mm` : "—"}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Build direction</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">+Z</div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Voxel size</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">
                {voxelSizeMm == null ? "—" : `${fmt(voxelSizeMm, 3)} mm`}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Grid</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">
                {grid ? `(${grid[0]}, ${grid[1]}, ${grid[2]})` : "—"}
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Supports</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">{supportCount}</div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Loads</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">{loadCount}</div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 col-span-2">
              <div className="text-[11px] uppercase tracking-wide text-slate-500">Utilization</div>
              <div className="mt-1 text-sm font-semibold text-slate-800">
                {utilizationPct == null ? "—" : `${utilizationPct.toFixed(2)}% of 128^3`}
              </div>
            </div>
          </div>

          {setupLabel ? (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
              Active setup: <span className="font-semibold text-slate-700">{setupLabel}</span>
            </div>
          ) : null}

          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
            non_design_regions: <span className="font-semibold text-slate-700">{nonDesignCount}</span>
          </div>

          {bboxText ? (
            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600 break-all">
              {bboxText}
            </div>
          ) : null}

          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
            {isStlSeed
              ? "STL geometry/envelope metadata came from the uploaded STL. Supports, loads, and non-design regions came from the user-authored inputs."
              : "Cube domain dimensions came from user input. Supports, loads, and non-design regions came from the user-authored inputs."}
          </div>
        </div>
      )}
    </div>
  );
}

export default function DesignSetup() {
  const [topTab, setTopTab] = useState<TopTab>("design");
  const [runId, setRunId] = useState<string | null>(null);

  const [cubeXText, setCubeXText] = useState<string>("100");
  const [cubeYText, setCubeYText] = useState<string>("100");
  const [cubeZText, setCubeZText] = useState<string>("60");
  const [voxelSize, setVoxelSize] = useState<number>(1.0);

  const cubeX = Number(cubeXText);
  const cubeY = Number(cubeYText);
  const cubeZ = Number(cubeZText);

  const [domainInfo, setDomainInfo] = useState<DomainCubeOk | null>(null);
  const [domainErr, setDomainErr] = useState<string | null>(null);
  const [busyDomain, setBusyDomain] = useState<boolean>(false);

  const [geoInfo, setGeoInfo] = useState<GeometryUploadResp | null>(null);
  const [geoErr, setGeoErr] = useState<string | null>(null);
  const [busyUpload, setBusyUpload] = useState<boolean>(false);

  const [supports, setSupports] = useState<RegionDraft[]>([]);
  const [loads, setLoads] = useState<LoadDraft[]>([]);
  const [nonDesignRegions, setNonDesignRegions] = useState<RegionDraft[]>([]);

  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [capsErr, setCapsErr] = useState<string | null>(null);
  const [capsBusy, setCapsBusy] = useState<boolean>(false);

  const [material, setMaterial] = useState<string>("IN718");

  const [pInput, setPInput] = useState<string>("350");
  const [vInput, setVInput] = useState<string>("750");
  const [hInput, setHInput] = useState<string>("0.11");
  const [tInput, setTInput] = useState<string>("0.04");

  const [expertOverride, setExpertOverride] = useState<boolean>(false);
  const [overrideReason, setOverrideReason] = useState<string>("");

  const [procBusy, setProcBusy] = useState<boolean>(false);
  const [procErr, setProcErr] = useState<string | null>(null);
  const [procOkMsg, setProcOkMsg] = useState<string | null>(null);
  const [processSaved, setProcessSaved] = useState<boolean>(false);

  const [optErr, setOptErr] = useState<string | null>(null);
  const [optOkMsg, setOptOkMsg] = useState<string | null>(null);

  const [startBusy, setStartBusy] = useState<boolean>(false);
  const [startErr, setStartErr] = useState<string | null>(null);
  const [startedRunId, setStartedRunId] = useState<string | null>(null);

  const [committedMeta, setCommittedMeta] = useState<CommittedDomainMeta | null>(null);

  const stlInputRef = useRef<HTMLInputElement | null>(null);

  const topTabs: Array<{ key: TopTab; label: string }> = [
    { key: "design", label: "Design Setup" },
    { key: "material", label: "Material & Constraints" },
    { key: "process", label: "Process Windows" },
  ];

  useEffect(() => {
    const savedRunId = sessionStorage.getItem(ACTIVE_RUN_STORAGE_KEY);
    if (!savedRunId) return;

    let cancelled = false;

    (async () => {
      try {
        const existing = await apiGetJson<RunStatusResp>(`/runs/${savedRunId}`);
        if (cancelled) return;

        const status = (existing.run?.status || "").toUpperCase();
        if (["RUNNING", "PAUSED", "DONE", "QUEUED"].includes(status)) {
          setRunId(savedRunId);
          setStartedRunId(savedRunId);
        } else {
          sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
        }
      } catch {
        sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const maxMm = 128 * voxelSize;

    const clampText = (prev: string, fallback: string) => {
      const n = Number(prev);
      if (!Number.isFinite(n)) return fallback;
      return String(Math.min(n, maxMm));
    };

    setCubeXText((prev) => clampText(prev, "100"));
    setCubeYText((prev) => clampText(prev, "100"));
    setCubeZText((prev) => clampText(prev, "60"));
  }, [voxelSize]);

  const nx_est = useMemo(
    () => Math.floor((Number.isFinite(cubeX) ? cubeX : 0) / Math.max(voxelSize, 1e-9)),
    [cubeX, voxelSize]
  );
  const ny_est = useMemo(
    () => Math.floor((Number.isFinite(cubeY) ? cubeY : 0) / Math.max(voxelSize, 1e-9)),
    [cubeY, voxelSize]
  );
  const nz_est = useMemo(
    () => Math.floor((Number.isFinite(cubeZ) ? cubeZ : 0) / Math.max(voxelSize, 1e-9)),
    [cubeZ, voxelSize]
  );

  const previewGrid = useMemo<[number, number, number] | null>(() => {
    if (nx_est <= 0 || ny_est <= 0 || nz_est <= 0) return null;
    return [nx_est, ny_est, nz_est];
  }, [nx_est, ny_est, nz_est]);

  const capOk = useMemo(() => nx_est <= 128 && ny_est <= 128 && nz_est <= 128, [nx_est, ny_est, nz_est]);
  const capText = useMemo(
    () => `Grid = (${nx_est}, ${ny_est}, ${nz_est}) cap = (128, 128, 128)`,
    [nx_est, ny_est, nz_est]
  );

  const draftGrid = useMemo<[number, number, number] | null>(() => {
    if (geoInfo?.grid_estimate) return geoInfo.grid_estimate;
    return previewGrid;
  }, [geoInfo, previewGrid]);

  useEffect(() => {
    if (!geoInfo && supports.length === 0 && loads.length === 0 && previewGrid) {
      setSupports(makeStarterSupports(previewGrid));
      setLoads(makeStarterLoads(previewGrid));
      setNonDesignRegions(makeStarterNonDesign());
    }
  }, [geoInfo, previewGrid, supports.length, loads.length]);

  const domainReady = useMemo(() => Boolean(runId && domainInfo?.ok), [runId, domainInfo]);

  const committedGrid = useMemo<[number, number, number] | null>(() => {
    if (!domainReady || !domainInfo?.grid) return null;
    return domainInfo.grid;
  }, [domainReady, domainInfo]);

  const committedVoxel = useMemo<number | null>(() => {
    if (!domainReady || !domainInfo?.voxel_size_mm) return null;
    return domainInfo.voxel_size_mm;
  }, [domainReady, domainInfo]);

  const committedDimsMm = useMemo<[number, number, number] | null>(() => {
    if (!domainReady) return null;
    if (committedMeta?.isStlSeed && geoInfo?.dims_mm) return geoInfo.dims_mm;
    if (!committedGrid || committedVoxel == null) return null;
    const [nx, ny, nz] = committedGrid;
    return [nx * committedVoxel, ny * committedVoxel, nz * committedVoxel];
  }, [domainReady, committedMeta, geoInfo, committedGrid, committedVoxel]);

  const committedUtilPct = useMemo(() => {
    if (!domainReady || !committedGrid) return null;
    const [nx, ny, nz] = committedGrid;
    const used = nx * ny * nz;
    const cap = 128 * 128 * 128;
    return (used / cap) * 100;
  }, [domainReady, committedGrid]);

  const pVal = parseNumberInput(pInput);
  const vVal = parseNumberInput(vInput);
  const hVal = parseNumberInput(hInput);
  const tVal = parseNumberInput(tInput);

  const ved = useMemo(() => {
    if (pVal == null || vVal == null || hVal == null || tVal == null) return NaN;
    const eps = 1e-12;
    return pVal / (Math.max(vVal, eps) * Math.max(hVal, eps) * Math.max(tVal, eps));
  }, [pVal, vVal, hVal, tVal]);

  const vw = useMemo(() => {
    if (!caps?.ok) return null;
    return caps.validity_windows?.[material] || null;
  }, [caps, material]);

  const withinWindow = useMemo(() => {
    if (!vw) return { ok: true, reasons: [] as string[] };

    const reasons: string[] = [];
    const inRange = (val: number | null, r?: [number, number] | null, name?: string) => {
      if (val == null) {
        reasons.push(`${name} is missing`);
        return;
      }
      if (!r || r.length !== 2) return;
      const [lo, hi] = r;
      if (val < lo || val > hi) reasons.push(`${name} outside [${lo}, ${hi}]`);
    };

    inRange(pVal, vw.P_W ?? null, "P_W");
    inRange(vVal, vw.v_mm_per_s ?? null, "v");
    inRange(hVal, vw.h_mm ?? null, "h");
    inRange(tVal, vw.t_mm ?? null, "t");

    return { ok: reasons.length === 0, reasons };
  }, [vw, pVal, vVal, hVal, tVal]);

  const overrideRequired = useMemo(() => !withinWindow.ok, [withinWindow.ok]);
  const overrideOk = useMemo(() => {
    if (!overrideRequired) return true;
    return expertOverride && overrideReason.trim().length >= 5;
  }, [overrideRequired, expertOverride, overrideReason]);

  const step2Enabled = domainReady;
  const canStart = useMemo(
    () => step2Enabled && processSaved && !!runId,
    [step2Enabled, processSaved, runId]
  );

  const authoringWarnings = useMemo(() => {
    const msgs: string[] = [];
    if (!draftGrid) return msgs;

    const [nx, ny, nz] = draftGrid;
    const outOfBounds = (b: RegionDraft) =>
      b.i0 < 0 || b.i1 > nx || b.j0 < 0 || b.j1 > ny || b.k0 < 0 || b.k1 > nz;

    if (supports.length === 0) msgs.push("No supports are currently defined.");
    if (loads.length === 0) msgs.push("No loads are currently defined.");

    supports.forEach((b) => {
      if (outOfBounds(b)) msgs.push(`Support "${b.name}" extends outside the current grid.`);
    });

    loads.forEach((b) => {
      if (outOfBounds(b)) msgs.push(`Load "${b.name}" extends outside the current grid.`);
      if (Number(b.fx) === 0 && Number(b.fy) === 0 && Number(b.fz) === 0) {
        msgs.push(`Load "${b.name}" currently has a zero force vector.`);
      }
    });

    nonDesignRegions.forEach((b) => {
      if (outOfBounds(b)) msgs.push(`Non-design region "${b.name}" extends outside the current grid.`);
    });

    return msgs;
  }, [draftGrid, supports, loads, nonDesignRegions]);

  const ensureCapsLoaded = useCallback(async () => {
    if (caps?.ok) return;
    setCapsErr(null);
    setCapsBusy(true);
    try {
      const c = await apiGetJson<Capabilities>("/capabilities");
      setCaps(c);
      if (c?.materials?.length && !c.materials.includes(material)) setMaterial(c.materials[0]);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setCapsErr(msg);
    } finally {
      setCapsBusy(false);
    }
  }, [caps, material]);

  useEffect(() => {
    if (step2Enabled) void ensureCapsLoaded();
  }, [step2Enabled, ensureCapsLoaded]);

  useEffect(() => {
    if (!caps?.ok) return;
    const w = caps.validity_windows?.[material];
    if (!w) return;

    const mpP = midpoint(w.P_W ?? null);
    const mpV = midpoint(w.v_mm_per_s ?? null);
    const mpH = midpoint(w.h_mm ?? null);
    const mpT = midpoint(w.t_mm ?? null);

    if (mpP != null) setPInput(String(mpP));
    if (mpV != null) setVInput(String(mpV));
    if (mpH != null) setHInput(String(mpH));
    if (mpT != null) setTInput(String(mpT));
  }, [caps, material]);

  function loadSuggestedRegionsForGrid(grid: [number, number, number]) {
    setSupports(makeStarterSupports(grid));
    setLoads(makeStarterLoads(grid));
    setNonDesignRegions(makeStarterNonDesign());
  }

  async function createRunIfNeeded(): Promise<string> {
    if (runId) {
      try {
        const existing = await apiGetJson<RunStatusResp>(`/runs/${runId}`);
        const status = (existing.run?.status || "").toUpperCase();

        if (!["DONE", "FAILED", "STOPPED"].includes(status)) {
          return runId;
        }
      } catch {
        // create fresh run below
      }
    }

    const created = await apiPostJson<RunCreateResp>("/runs", {
      name: "ui_design_setup",
      material: material || "IN718",
      voxel_size_mm: voxelSize,
    });

    setRunId(created.run_id);
    return created.run_id;
  }

  function buildCubeDomainPayload(nx: number, ny: number, nz: number) {
    return {
      preset: "domain_custom",
      units: "mm",
      voxel_size_mm: voxelSize,
      grid: { nx, ny, nz },
      build_direction: "+Z",
      supports: supports.map(toRegionBox),
      loads: loads.map(toLoadBox),
      non_design_regions: nonDesignRegions.map(toRegionBox),
      notes: "Cube domain authored in UI. Supports, loads, and non-design regions are user-defined.",
    };
  }

  function buildStlDomainPayload(up: GeometryUploadResp) {
    const [gx, gy, gz] = up.grid_estimate;
    return {
      preset: "domain_custom",
      units: "mm",
      voxel_size_mm: up.voxel_size_mm_used,
      grid: { nx: gx, ny: gy, nz: gz },
      build_direction: "+Z",
      source: {
        type: "stl",
        stored_path: up.stored_path,
        bbox_mm: up.bbox_mm,
        dims_mm: up.dims_mm,
      },
      supports: supports.map(toRegionBox),
      loads: loads.map(toLoadBox),
      non_design_regions: nonDesignRegions.map(toRegionBox),
      notes: "STL envelope authored in UI. Geometry came from STL; supports, loads, and non-design regions are user-defined.",
    };
  }

  async function handleUseCubeDomain() {
    setDomainErr(null);
    setDomainInfo(null);
    setGeoErr(null);
    setProcErr(null);
    setProcOkMsg(null);
    setProcessSaved(false);
    setOptErr(null);
    setOptOkMsg(null);
    setStartedRunId(null);
    sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
    setBusyDomain(true);

    try {
      const id = await createRunIfNeeded();

      const nx = Math.floor((Number.isFinite(cubeX) ? cubeX : 0) / Math.max(voxelSize, 1e-9));
      const ny = Math.floor((Number.isFinite(cubeY) ? cubeY : 0) / Math.max(voxelSize, 1e-9));
      const nz = Math.floor((Number.isFinite(cubeZ) ? cubeZ : 0) / Math.max(voxelSize, 1e-9));

      if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw new Error("Invalid dimensions/voxel size: resulted in non-positive grid.");
      }

      const domain = buildCubeDomainPayload(nx, ny, nz);
      const wrote = await apiPostJson<DomainWriteResp>(`/runs/${id}/inputs/domain`, { domain });

      setGeoInfo(null);
      setRunId(id);
      setDomainInfo({
        ok: true,
        path: wrote.path,
        grid: [nx, ny, nz],
        voxel_size_mm: voxelSize,
      });
      setCommittedMeta({
        sourceLabel: "Cube domain",
        setupLabel: "User-authored cube domain",
        isStlSeed: false,
        supportCount: supports.length,
        loadCount: loads.length,
        nonDesignCount: nonDesignRegions.length,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setDomainErr(msg);
    } finally {
      setBusyDomain(false);
    }
  }

  async function handleChooseStl(file: File) {
    setGeoErr(null);
    setGeoInfo(null);
    setDomainErr(null);
    setDomainInfo(null);
    setCommittedMeta(null);
    setProcErr(null);
    setProcOkMsg(null);
    setProcessSaved(false);
    setOptErr(null);
    setOptOkMsg(null);
    setStartedRunId(null);
    sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
    setBusyUpload(true);

    try {
      const id = await createRunIfNeeded();

      const fd = new FormData();
      fd.append("file", file);
      const up = await apiPostForm<GeometryUploadResp>(`/runs/${id}/inputs/geometry_upload`, fd);

      setRunId(id);
      setGeoInfo(up);
      loadSuggestedRegionsForGrid(up.grid_estimate);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setGeoErr(msg);
    } finally {
      setBusyUpload(false);
    }
  }

  async function handleUseUploadedStlDomain() {
    setDomainErr(null);
    setProcErr(null);
    setProcOkMsg(null);
    setProcessSaved(false);
    setOptErr(null);
    setOptOkMsg(null);
    setStartedRunId(null);
    sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);
    setBusyDomain(true);

    try {
      if (!geoInfo) throw new Error("Upload an STL first.");

      const id = await createRunIfNeeded();
      const domain = buildStlDomainPayload(geoInfo);
      const wrote = await apiPostJson<DomainWriteResp>(`/runs/${id}/inputs/domain`, { domain });

      const [gx, gy, gz] = geoInfo.grid_estimate;
      setRunId(id);
      setDomainInfo({
        ok: true,
        path: wrote.path,
        grid: [gx, gy, gz],
        voxel_size_mm: geoInfo.voxel_size_mm_used,
      });
      setCommittedMeta({
        sourceLabel: "STL envelope",
        setupLabel: "Uploaded STL envelope + user-authored regions",
        isStlSeed: true,
        supportCount: supports.length,
        loadCount: loads.length,
        nonDesignCount: nonDesignRegions.length,
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setDomainErr(msg);
    } finally {
      setBusyDomain(false);
    }
  }

  async function handleSaveProcess() {
    setProcErr(null);
    setProcOkMsg(null);
    setProcessSaved(false);
    setProcBusy(true);

    try {
      if (!domainReady) throw new Error("Step 1 not complete. Save cube or STL domain first.");
      if (pVal == null || vVal == null || hVal == null || tVal == null) {
        throw new Error("All process parameter fields must have valid numeric values.");
      }
      if (overrideRequired && !overrideOk) {
        throw new Error("Outside validated window. Enable Expert Override and provide a short reason.");
      }

      await ensureCapsLoaded();

      const id = await createRunIfNeeded();
      const payload = {
        material,
        params: {
          P_W: pVal,
          v_mm_per_s: vVal,
          h_mm: hVal,
          t_mm: tVal,
        },
        expert_override: Boolean(overrideRequired ? expertOverride : false),
        override_reason: overrideRequired ? overrideReason.trim() : null,
      };

      const wrote = await apiPostJson<ProcessWriteResp>(`/runs/${id}/inputs/process`, payload);
      setProcOkMsg(`Saved: ${wrote.path}`);
      setProcessSaved(true);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setProcErr(msg);
    } finally {
      setProcBusy(false);
    }
  }

  async function saveOptConfigNow(id: string): Promise<OptWriteResp> {
    return apiPostJson<OptWriteResp>(`/runs/${id}/inputs/opt`, {
      total_iters: 400,
      snapshot_every: 100,
      iso_threshold: 0.525,
      objective: "compliance",
      target_vf: 0.4,
    });
  }

  async function handleStartOptimization() {
    if (!runId) return;
    setStartErr(null);
    setStartBusy(true);
    setOptErr(null);
    setOptOkMsg(null);

    try {
      const wrote = await saveOptConfigNow(runId);
      setOptOkMsg(`Saved: ${wrote.path}`);

      await apiPostNoBody(`/runs/${runId}/start`);
      setStartedRunId(runId);
      sessionStorage.setItem(ACTIVE_RUN_STORAGE_KEY, runId);

      requestAnimationFrame(() => {
        document.getElementById("live-results-cockpit")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setStartErr(msg);
    } finally {
      setStartBusy(false);
    }
  }

  async function handleStopRunFromSetup() {
    if (!startedRunId) return;
    try {
      await apiPostNoBody(`/runs/${startedRunId}/stop`);
    } catch {
      // polled dashboard will surface state
    }
  }

  function handleOpenControlTower() {
    window.location.href = "/control-tower";
  }

  function handleResetAll() {
    setRunId(null);
    setStartedRunId(null);
    sessionStorage.removeItem(ACTIVE_RUN_STORAGE_KEY);

    setCubeXText("100");
    setCubeYText("100");
    setCubeZText("60");
    setVoxelSize(1.0);

    setDomainInfo(null);
    setDomainErr(null);
    setBusyDomain(false);

    setGeoInfo(null);
    setGeoErr(null);
    setBusyUpload(false);

    setSupports([]);
    setLoads([]);
    setNonDesignRegions([]);

    setCaps(null);
    setCapsErr(null);
    setCapsBusy(false);

    setMaterial("IN718");
    setPInput("350");
    setVInput("750");
    setHInput("0.11");
    setTInput("0.04");

    setExpertOverride(false);
    setOverrideReason("");

    setProcBusy(false);
    setProcErr(null);
    setProcOkMsg(null);
    setProcessSaved(false);

    setOptErr(null);
    setOptOkMsg(null);

    setStartBusy(false);
    setStartErr(null);

    setCommittedMeta(null);
    setTopTab("design");

    requestAnimationFrame(() => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-2">
        {topTabs.map((t) => {
          const active = topTab === t.key;
          return (
            <button
              key={t.key}
              type="button"
              onClick={() => {
                setTopTab(t.key);

                if (t.key === "design") {
                  scrollToId("design-setup-step-1");
                  return;
                }

                if (t.key === "material") {
                  void ensureCapsLoaded();
                  scrollToId("design-setup-step-2");
                  return;
                }

                if (t.key === "process") {
                  void ensureCapsLoaded();
                  scrollToId("design-setup-step-2");
                }
              }}
              className={[
                "px-3 py-2 rounded-lg border shadow-sm text-sm transition",
                active
                  ? "bg-white border-slate-200 font-semibold text-slate-900"
                  : "bg-white/70 border-slate-200 text-slate-600 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700",
              ].join(" ")}
            >
              {t.label}
            </button>
          );
        })}
      </div>

      <section id="design-setup-step-1" className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
        <h2 className="text-xl font-semibold text-slate-800">Step 1: Define Geometry</h2>

        <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <div className="font-semibold">Memory / runtime note</div>
          <div className="mt-1">
            Larger domains and finer voxel sizes can require much more memory and runtime. For quick local verification, start smaller; for heavier cases, use a stronger machine or cloud environment.
          </div>
        </div>

        <div className="mt-3 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
          <div className="font-semibold">Recommended quick validation</div>
          <div className="mt-1">
            A reliable quick test is <span className="font-semibold">16 × 16 × 8 mm</span> at <span className="font-semibold">1.0 mm</span> voxel size. For a more visible mechanical result, try <span className="font-semibold">32 × 16 × 32 mm</span> at <span className="font-semibold">1.0 mm</span>.
          </div>
        </div>

        <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          What you author in Step 1 is what gets written to <span className="font-mono">domain.json</span>.
        </div>

        <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
            <div className="text-sm font-semibold text-slate-800">Import STL Envelope</div>
            <div className="text-xs text-slate-500 mt-1">
              Upload STL to auto-populate geometry/envelope data. Then author supports, loads, and non-design regions yourself.
            </div>

            <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
              <div className="font-semibold">v1 STL import notes</div>
              <ul className="mt-1 list-disc pl-5 space-y-1">
                <li>Use watertight single-body STL when possible.</li>
                <li>STL auto-populates bbox, dims, grid estimate, and source metadata.</li>
                <li>Supports, loads, and non-design regions still come from your Step 1 inputs.</li>
              </ul>
            </div>

            <div className="mt-3">
              <input
                ref={stlInputRef}
                type="file"
                accept=".stl"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleChooseStl(f);
                  e.currentTarget.value = "";
                }}
              />
              <button
                type="button"
                disabled={busyUpload}
                className={[
                  "px-3 py-2 rounded-lg text-sm font-semibold transition",
                  busyUpload ? "bg-slate-300 text-slate-600" : "bg-blue-600 text-white hover:bg-blue-500",
                ].join(" ")}
                onClick={() => stlInputRef.current?.click()}
              >
                {busyUpload ? "Uploading..." : "Choose STL file..."}
              </button>
            </div>

            {geoInfo && (
              <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-900">
                <div className="font-semibold">STL uploaded</div>
                <div className="mt-1 break-words">
                  dims = ({fmt(geoInfo.dims_mm[0], 2)}, {fmt(geoInfo.dims_mm[1], 2)}, {fmt(geoInfo.dims_mm[2], 2)}) mm
                </div>
                <div className="mt-1 break-words">
                  grid_estimate = ({geoInfo.grid_estimate[0]}, {geoInfo.grid_estimate[1]}, {geoInfo.grid_estimate[2]}) @ {geoInfo.voxel_size_mm_used} mm
                </div>
                <div className="mt-1 break-all text-emerald-900/80">
                  stored_path = {geoInfo.stored_path}
                </div>
                {!geoInfo.fits_128_cubed && (
                  <div className="mt-1 text-rose-700">
                    Exceeds 128^3. Suggested voxel size: {geoInfo.suggested_voxel_size_mm.toFixed(3)} mm
                  </div>
                )}

                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    type="button"
                    className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                    onClick={() => loadSuggestedRegionsForGrid(geoInfo.grid_estimate)}
                  >
                    Load Suggested Regions
                  </button>
                </div>
              </div>
            )}

            {geoErr && (
              <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800 break-words">
                <div className="font-semibold">STL upload error</div>
                <div className="mt-1">{geoErr}</div>
              </div>
            )}
          </div>

          <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-slate-800">Cube Domain</div>
                <div className="text-xs text-slate-500 mt-1">User-authored structured domain dimensions (mm)</div>
              </div>

              <div className="text-right">
                <div className="text-[11px] font-semibold text-slate-600">Voxel size (mm)</div>
                <select
                  className="mt-1 rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  value={voxelSize}
                  onChange={(e) => {
                    const nextVoxel = parseFloat(e.target.value);
                    setVoxelSize(nextVoxel);
                  }}
                >
                  <option value={0.25}>0.25</option>
                  <option value={0.5}>0.5</option>
                  <option value={1.0}>1.0</option>
                </select>
              </div>
            </div>

            <div className="mt-3 grid grid-cols-3 gap-3">
              <label className="text-xs font-semibold text-slate-700">
                X Size (mm)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={cubeXText}
                  onFocus={(e) => e.currentTarget.select()}
                  onChange={(e) => setCubeXText(e.target.value)}
                  onBlur={() => {
                    const n = Number(cubeXText);
                    if (!Number.isFinite(n) || n <= 0) {
                      setCubeXText("100");
                    } else {
                      setCubeXText(String(n));
                    }
                  }}
                />
              </label>

              <label className="text-xs font-semibold text-slate-700">
                Y Size (mm)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={cubeYText}
                  onFocus={(e) => e.currentTarget.select()}
                  onChange={(e) => setCubeYText(e.target.value)}
                  onBlur={() => {
                    const n = Number(cubeYText);
                    if (!Number.isFinite(n) || n <= 0) {
                      setCubeYText("100");
                    } else {
                      setCubeYText(String(n));
                    }
                  }}
                />
              </label>

              <label className="text-xs font-semibold text-slate-700">
                Z Size (mm)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={cubeZText}
                  onFocus={(e) => e.currentTarget.select()}
                  onChange={(e) => setCubeZText(e.target.value)}
                  onBlur={() => {
                    const n = Number(cubeZText);
                    if (!Number.isFinite(n) || n <= 0) {
                      setCubeZText("60");
                    } else {
                      setCubeZText(String(n));
                    }
                  }}
                />
              </label>
            </div>

            <div className="mt-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-600">{capText}</span>
                <span
                  className={[
                    "inline-flex items-center rounded-full px-2 py-0.5 font-semibold",
                    capOk ? "bg-emerald-50 text-emerald-700" : "bg-rose-50 text-rose-700",
                  ].join(" ")}
                >
                  {capOk ? "Within cap" : "Exceeds cap"}
                </span>
              </div>
              {!capOk && (
                <div className="mt-1 text-rose-700">
                  Increase voxel size or reduce dimensions to fit 128^3.
                </div>
              )}
            </div>

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button
                type="button"
                className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                onClick={() => {
                  if (previewGrid) loadSuggestedRegionsForGrid(previewGrid);
                }}
              >
                Load Suggested Regions
              </button>

              {runId && (
                <div className="text-xs text-slate-500">
                  run_id: <span className="font-mono">{runId}</span>
                </div>
              )}
            </div>

            {domainInfo && !geoInfo && (
              <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-900">
                <div className="font-semibold">Cube domain saved</div>
                <div className="mt-1 break-words">
                  grid = ({domainInfo.grid[0]}, {domainInfo.grid[1]}, {domainInfo.grid[2]}) @ {domainInfo.voxel_size_mm} mm
                </div>
                <div className="mt-1 break-all text-emerald-900/80">{domainInfo.path}</div>
              </div>
            )}

            {domainErr && (
              <div className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-800 break-words">
                <div className="font-semibold">Domain error</div>
                <div className="mt-1">{domainErr}</div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-4 grid grid-cols-1 gap-4">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="text-sm font-semibold text-slate-800">Domain Regions</div>
            <div className="text-xs text-slate-500 mt-1">
              Use these region inputs for either cube mode or STL mode. These values are the structural intent of the problem.
            </div>

            {draftGrid && (
              <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                Current authoring source: <span className="font-semibold text-slate-700">{geoInfo ? "Uploaded STL envelope" : "Cube draft"}</span>
                {" • "}
                active grid = ({draftGrid[0]}, {draftGrid[1]}, {draftGrid[2]})
              </div>
            )}

            {authoringWarnings.length > 0 && (
              <div className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                <div className="font-semibold">Authoring advisories</div>
                <ul className="mt-1 list-disc pl-5 space-y-1">
                  {authoringWarnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
                <div className="mt-2 text-amber-900/80">
                  These are advisories only. Your authored inputs are still used as entered.
                </div>
              </div>
            )}

            <div className="mt-4 grid grid-cols-1 gap-4">
              <RegionEditor
                title="Support Regions"
                items={supports}
                setItems={setSupports}
                draftGrid={draftGrid}
                namePrefix="support"
              />

              <LoadEditor
                items={loads}
                setItems={setLoads}
                draftGrid={draftGrid}
              />

              <RegionEditor
                title="Preserved Regions (Non-Design)"
                items={nonDesignRegions}
                setItems={setNonDesignRegions}
                draftGrid={draftGrid}
                namePrefix="non_design"
              />
            </div>

            <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-slate-200 pt-4">
              <button
                type="button"
                disabled={busyDomain}
                className={[
                  "px-4 py-2 rounded-lg text-sm font-semibold transition",
                  busyDomain ? "bg-slate-200 text-slate-500" : "bg-blue-600 text-white hover:bg-blue-500",
                ].join(" ")}
                onClick={() => void handleUseCubeDomain()}
              >
                {busyDomain ? "Saving..." : "Use Cube Domain"}
              </button>

              <button
                type="button"
                disabled={busyDomain || !geoInfo}
                className={[
                  "px-4 py-2 rounded-lg text-sm font-semibold transition",
                  busyDomain || !geoInfo
                    ? "bg-slate-200 text-slate-500"
                    : "bg-emerald-600 text-white hover:bg-emerald-500",
                ].join(" ")}
                onClick={() => void handleUseUploadedStlDomain()}
              >
                {busyDomain ? "Saving..." : "Use Uploaded STL Domain"}
              </button>

              <div className="text-xs text-slate-500">
                Commit your authored Step 1 geometry here after setting dimensions and regions.
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="design-setup-step-2" className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-slate-800">Step 2: Configure Settings</h2>
          {!step2Enabled && (
            <span className="rounded-full bg-amber-50 text-amber-800 border border-amber-200 px-3 py-1 text-xs font-semibold">
              Complete Step 1 first
            </span>
          )}
        </div>

        <div className="mt-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
            <div className="text-sm font-semibold text-slate-800">Preview / Domain Summary</div>
            <div className="text-xs text-slate-500 mt-1">Shows only the committed Step 1 domain.</div>

            <DomainSummaryCard
              hasCommittedDomain={domainReady}
              sourceLabel={committedMeta?.sourceLabel ?? null}
              dimsMm={committedDimsMm}
              voxelSizeMm={committedVoxel}
              grid={committedGrid}
              utilizationPct={committedUtilPct}
              bboxText={
                committedMeta?.isStlSeed && geoInfo?.bbox_mm
                  ? `bbox min=${geoInfo.bbox_mm.min.join(", ")} max=${geoInfo.bbox_mm.max.join(", ")}`
                  : null
              }
              isStlSeed={Boolean(committedMeta?.isStlSeed)}
              setupLabel={committedMeta?.setupLabel ?? null}
              supportCount={committedMeta?.supportCount ?? 0}
              loadCount={committedMeta?.loadCount ?? 0}
              nonDesignCount={committedMeta?.nonDesignCount ?? 0}
            />

            <div className="mt-3 flex gap-2">
              <button
                type="button"
                className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                onClick={handleResetAll}
              >
                Reset
              </button>
            </div>
          </div>

          <div className="border border-slate-200 rounded-xl p-4 bg-slate-50">
            <div className="text-sm font-semibold text-slate-800">Process Parameters</div>
            <div className="text-xs text-slate-500 mt-1">
              Safe/default values come from capabilities. Outside tested process windows is allowed using Expert Override.
            </div>

            <div className="mt-3 flex items-center gap-2">
              <button
                type="button"
                className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                onClick={() => void ensureCapsLoaded()}
                disabled={capsBusy}
              >
                {capsBusy ? "Loading..." : "Refresh capabilities"}
              </button>
              {capsErr && <div className="text-xs text-rose-700 break-words">{capsErr}</div>}
              {caps?.ok && <div className="text-xs text-emerald-700">Capabilities OK</div>}
            </div>

            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
              <label className="text-xs font-semibold text-slate-700">
                Material
                <select
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  value={material}
                  onChange={(e) => setMaterial(e.target.value)}
                  disabled={!step2Enabled}
                >
                  {(caps?.materials?.length ? caps.materials : ["316L", "IN718", "Ti64"]).map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </label>

              <label className="text-xs font-semibold text-slate-700">
                Power P (W)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={pInput}
                  onChange={(e) => setPInput(e.target.value)}
                  disabled={!step2Enabled}
                />
              </label>

              <label className="text-xs font-semibold text-slate-700">
                Speed v (mm/s)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={vInput}
                  onChange={(e) => setVInput(e.target.value)}
                  disabled={!step2Enabled}
                />
              </label>

              <label className="text-xs font-semibold text-slate-700">
                Hatch spacing h (mm)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={hInput}
                  onChange={(e) => setHInput(e.target.value)}
                  disabled={!step2Enabled}
                />
              </label>

              <label className="text-xs font-semibold text-slate-700">
                Layer thickness t (mm)
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 px-2 py-1.5 text-sm"
                  type="text"
                  inputMode="decimal"
                  value={tInput}
                  onChange={(e) => setTInput(e.target.value)}
                  disabled={!step2Enabled}
                />
              </label>

              <div className="rounded-lg border border-slate-200 bg-white p-3">
                <div className="text-[11px] uppercase tracking-wide text-slate-500">VED (J/mm^3)</div>
                <div className="mt-1 text-sm font-semibold text-slate-800">{fmt(ved, 2)}</div>
              </div>
            </div>

            <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3 text-xs">
              <div className="flex items-center justify-between">
                <div className="font-semibold text-slate-700">Validation window</div>
                <span
                  className={[
                    "inline-flex items-center rounded-full px-2 py-0.5 font-semibold",
                    withinWindow.ok ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-800",
                  ].join(" ")}
                >
                  {withinWindow.ok ? "Within window" : "Outside window"}
                </span>
              </div>

              {vw ? (
                <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2 text-slate-600">
                  <div>P: {vw.P_W ? `[${vw.P_W[0]}, ${vw.P_W[1]}]` : "—"}</div>
                  <div>v: {vw.v_mm_per_s ? `[${vw.v_mm_per_s[0]}, ${vw.v_mm_per_s[1]}]` : "—"}</div>
                  <div>h: {vw.h_mm ? `[${vw.h_mm[0]}, ${vw.h_mm[1]}]` : "—"}</div>
                  <div>t: {vw.t_mm ? `[${vw.t_mm[0]}, ${vw.t_mm[1]}]` : "—"}</div>
                </div>
              ) : (
                <div className="mt-2 text-slate-600">Load capabilities to show material window.</div>
              )}

              {!withinWindow.ok && (
                <div className="mt-2 text-amber-900">
                  <div className="font-semibold">Warnings:</div>
                  <ul className="mt-1 list-disc pl-5">
                    {withinWindow.reasons.map((r) => (
                      <li key={r}>{r}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            <div className="mt-3 rounded-lg border border-slate-200 bg-white p-3">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={expertOverride}
                  onChange={(e) => setExpertOverride(e.target.checked)}
                  disabled={!step2Enabled || !overrideRequired}
                />
                <span className="font-semibold text-slate-700">Expert Override</span>
                {!overrideRequired && <span className="text-xs text-slate-500">(not needed)</span>}
              </label>

              <div className="mt-2">
                <div className="text-xs font-semibold text-slate-700">Override reason (required if outside window)</div>
                <input
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white text-slate-900 placeholder:text-slate-400 px-2 py-2 text-sm"
                  value={overrideReason}
                  onChange={(e) => setOverrideReason(e.target.value)}
                  placeholder="e.g., exploring out-of-window params for sensitivity study"
                  disabled={!step2Enabled || !overrideRequired || !expertOverride}
                />
                {overrideRequired && expertOverride && overrideReason.trim().length > 0 && overrideReason.trim().length < 5 && (
                  <div className="mt-1 text-xs text-rose-700">Reason too short.</div>
                )}
              </div>
            </div>

            <div className="mt-3">
              <button
                type="button"
                disabled={!step2Enabled || procBusy || (overrideRequired && !overrideOk)}
                className={[
                  "w-full px-4 py-3 rounded-xl text-sm font-semibold transition",
                  !step2Enabled || (overrideRequired && !overrideOk) || procBusy
                    ? "bg-slate-200 text-slate-500"
                    : "bg-blue-600 text-white hover:bg-blue-500",
                ].join(" ")}
                onClick={() => void handleSaveProcess()}
                title={
                  !step2Enabled
                    ? "Complete Step 1 first"
                    : overrideRequired && !overrideOk
                      ? "Enable Expert Override + reason"
                      : "Save process.json"
                }
              >
                {procBusy ? "Saving..." : "Save Process Settings"}
              </button>

              {procOkMsg && <div className="mt-2 text-xs text-emerald-700 break-all">{procOkMsg}</div>}
              {procErr && <div className="mt-2 text-xs text-rose-700 break-words">{procErr}</div>}
            </div>

            <div className="mt-4">
              <button
                type="button"
                disabled={!canStart || startBusy}
                className={[
                  "w-full px-4 py-3 rounded-xl text-sm font-semibold transition",
                  !canStart || startBusy
                    ? "bg-slate-200 text-slate-500"
                    : "bg-emerald-600 text-white hover:bg-emerald-500",
                ].join(" ")}
                onClick={() => void handleStartOptimization()}
                title={!canStart ? "Complete Step 1 + Save Process first" : "Start run"}
              >
                {startBusy ? "Starting..." : "Start Optimization"}
              </button>

              {startErr && <div className="mt-2 text-xs text-rose-700 break-words">{startErr}</div>}
              {!processSaved && step2Enabled && (
                <div className="mt-2 text-xs text-slate-500">Save Process Settings first.</div>
              )}
              {processSaved && step2Enabled && !startErr && (
                <div className="mt-2 text-xs text-slate-500">
                  Optimization config will be written automatically when you start.
                </div>
              )}
              {optOkMsg && <div className="mt-2 text-xs text-emerald-700 break-all">{optOkMsg}</div>}
              {optErr && <div className="mt-2 text-xs text-rose-700 break-words">{optErr}</div>}
            </div>
          </div>
        </div>
      </section>

      <section id="live-results-cockpit" className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
        {!startedRunId ? (
          <div className="space-y-3">
            <div className="text-lg font-semibold text-slate-800">Step 3: Results & Analytics</div>
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-5 text-sm text-slate-600">
              After you click <span className="font-semibold">Start Optimization</span>, the live cockpit will appear here and begin polling the run artifacts automatically.
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm text-slate-600">
                Active run: <span className="font-mono">{startedRunId}</span>
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => void handleStopRunFromSetup()}
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                >
                  Stop Run
                </button>

                <button
                  type="button"
                  onClick={handleResetAll}
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                >
                  Reset
                </button>

                <button
                  type="button"
                  onClick={handleOpenControlTower}
                  className="px-3 py-2 rounded-lg bg-white border border-slate-200 text-slate-700 text-sm font-semibold transition hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
                >
                  Open Control Tower
                </button>
              </div>
            </div>

            <Step3Dashboard runId={startedRunId} embedded />
          </div>
        )}
      </section>

      <div className="text-xs text-slate-500">
        Footnote: Risk maps are proxy diagnostics; no full CFD / thermo-fluid simulation is performed in v1.
      </div>
    </div>
  );
}
