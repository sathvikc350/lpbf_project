import { NavLink, Outlet } from "react-router-dom";

const DISCLOSURE =
  "* Diagnostics are proxy indicators derived from the frozen kernel, geometry heuristics, and surrogate inference outputs. They are not full thermo-fluid (CFD) or thermo-mechanical stress simulations. Use for comparative guidance and debugging, not certification.";

function cx(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export default function Layout() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="sticky top-0 z-30 border-b border-slate-800 bg-gradient-to-r from-slate-950 to-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div>
            <div className="text-xs text-slate-400">LPBF Digital Twin</div>
            <div className="text-lg font-semibold tracking-tight">
              Sathvik’s AM Topology Optimization Tool
            </div>
          </div>

          <div className="text-xs text-slate-400">
            Proxy: <span className="font-mono">5173 → 8000</span>
          </div>
        </div>
      </header>

      <div className="mx-auto flex max-w-7xl gap-6 px-6 py-6">
        <aside className="w-72 shrink-0">
          <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-4 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]">
            <NavLink
              to="/design-setup"
              className="block w-full rounded-xl bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow transition hover:bg-blue-500 text-center"
            >
              Import Geometry
            </NavLink>

            <div className="mt-5 px-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
              Workflow
            </div>

            <nav className="mt-2 space-y-1">
              <NavLink
                to="/control-tower"
                className={({ isActive }) =>
                  cx(
                    "flex items-center justify-between rounded-xl px-3 py-2 text-sm transition",
                    "hover:bg-slate-800/60",
                    isActive && "bg-slate-800/70 text-white border border-slate-700/60"
                  )
                }
              >
                <span>Control Tower</span>
                <span className="text-xs text-slate-400">Runs</span>
              </NavLink>

              <NavLink
                to="/design-setup"
                className={({ isActive }) =>
                  cx(
                    "flex items-center justify-between rounded-xl px-3 py-2 text-sm transition",
                    "hover:bg-slate-800/60",
                    isActive && "bg-slate-800/70 text-white border border-slate-700/60"
                  )
                }
              >
                <span>Design Setup</span>
                <span className="text-xs text-slate-400">Author</span>
              </NavLink>

              <NavLink
                to="/control-tower"
                className={({ isActive }) =>
                  cx(
                    "flex items-center justify-between rounded-xl px-3 py-2 text-sm transition",
                    "hover:bg-slate-800/60",
                    isActive && "bg-slate-800/70 text-white border border-slate-700/60"
                  )
                }
                title="Open a run from Control Tower"
              >
                <span>Optimization Run</span>
                <span className="text-xs text-slate-400">Pick run</span>
              </NavLink>

              <NavLink
                to="/results"
                className={({ isActive }) =>
                  cx(
                    "flex items-center justify-between rounded-xl px-3 py-2 text-sm transition",
                    "hover:bg-slate-800/60",
                    isActive && "bg-slate-800/70 text-white border border-slate-700/60"
                  )
                }
                title="Shows completed (DONE) runs"
              >
                <span>Results View</span>
                <span className="text-xs text-slate-400">DONE</span>
              </NavLink>

              <NavLink
                to="/export"
                className={({ isActive }) =>
                  cx(
                    "flex items-center justify-between rounded-xl px-3 py-2 text-sm transition",
                    "hover:bg-slate-800/60",
                    isActive && "bg-slate-800/70 text-white border border-slate-700/60"
                  )
                }
              >
                <span>Export & Report</span>
                <span className="text-xs text-slate-400">Ready</span>
              </NavLink>
            </nav>
          </div>
        </aside>

        <main className="min-w-0 flex-1">
          <Outlet />
        </main>
      </div>

      <footer className="border-t border-slate-800 bg-slate-950">
        <div className="mx-auto max-w-7xl px-6 py-4 text-xs text-slate-400">
          <span className="font-semibold text-slate-300">Disclosure (v1):</span>{" "}
          {DISCLOSURE}
        </div>
      </footer>
    </div>
  );
}
