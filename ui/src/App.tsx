import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./Layout";
import ControlTower from "./pages/ControlTower";
import DesignSetup from "./pages/DesignSetup";
import ResultsHome from "./pages/ResultsHome";
import Step3Dashboard from "./pages/Step3Dashboard";
import ExportReport from "./pages/ExportReport";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Navigate to="/control-tower" replace />} />

        <Route path="/control-tower" element={<ControlTower />} />
        <Route path="/design-setup" element={<DesignSetup />} />

        <Route
          path="/optimization-run/:runId"
          element={<div className="text-slate-700">Open the active run from Control Tower.</div>}
        />

        <Route path="/results" element={<ResultsHome />} />
        <Route path="/results/:runId" element={<Step3Dashboard />} />

        <Route path="/export" element={<ExportReport />} />
        <Route path="/export/:runId" element={<ExportReport />} />

        <Route path="*" element={<Navigate to="/control-tower" replace />} />
      </Route>
    </Routes>
  );
}

