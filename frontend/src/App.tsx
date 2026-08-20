import { FormEvent, lazy, Suspense, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { LoadingState } from "./components/StatePanel";
import { authRequired, getAccessKey, setAccessKey } from "./api";
const Dashboard = lazy(() => import("./pages/Dashboard").then(module => ({ default: module.Dashboard })));
const Investigation = lazy(() => import("./pages/Investigation").then(module => ({ default: module.Investigation })));
const ModelInfoPage = lazy(() => import("./pages/ModelInfo").then(module => ({ default: module.ModelInfoPage })));
function AccessGate({ children }: { children: React.ReactNode }) {
  const [authorized, setAuthorized] = useState(!authRequired || Boolean(getAccessKey()));
  const [key, setKey] = useState("");
  if (authorized) return children;
  function submit(event: FormEvent) {
    event.preventDefault();
    if (!key.trim()) return;
    setAccessKey(key.trim());
    setAuthorized(true);
  }
  return <main className="access-page"><form className="access-card" onSubmit={submit}><div className="brand-mark">S</div><p className="eyebrow">Protected prototype</p><h1>Synchrony access</h1><p>Enter the reader key to investigate records or the administrator key to run the demo. The key stays in this browser tab and is never bundled into the frontend.</p><label htmlFor="access-key">API access key</label><input id="access-key" type="password" autoComplete="off" value={key} onChange={event => setKey(event.target.value)} /><button className="button primary" type="submit">Open dashboard</button></form></main>;
}
export function App() { return <AccessGate><Suspense fallback={<LoadingState label="Loading workspace" />}><Routes><Route element={<Shell />}><Route index element={<Dashboard />} /><Route path="applications/:applicationId" element={<Investigation />} /><Route path="model-info" element={<ModelInfoPage />} /><Route path="*" element={<Navigate to="/" replace />} /></Route></Routes></Suspense></AccessGate>; }
