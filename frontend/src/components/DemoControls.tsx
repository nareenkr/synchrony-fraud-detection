import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import type { RandomDemoRequest } from "../types";

interface RandomForm {
  count: number;
  eventsPerSecond: number;
  seed: number;
  normal: number;
  suspicious: number;
  fraud: number;
}

const defaults: RandomForm = {
  count: 100,
  eventsPerSecond: 2,
  seed: 20260820,
  normal: 80,
  suspicious: 15,
  fraud: 5,
};

export function DemoControls() {
  const client = useQueryClient();
  const [showRandom, setShowRandom] = useState(false);
  const [form, setForm] = useState<RandomForm>(defaults);
  const refresh = () => client.invalidateQueries();
  const run = useMutation({ mutationFn: api.startDemo, onSuccess: refresh });
  const random = useMutation({ mutationFn: api.startRandom, onSuccess: refresh });
  const stop = useMutation({ mutationFn: api.stopDemo, onSuccess: refresh });
  const reset = useMutation({ mutationFn: api.resetDemo, onSuccess: refresh });
  const busy = run.isPending || random.isPending || stop.isPending || reset.isPending;
  const error = run.error || random.error || stop.error || reset.error;
  const totalPercent = form.normal + form.suspicious + form.fraud;
  const valid = totalPercent === 100 && form.count >= 1 && form.count <= 5000
    && form.eventsPerSecond >= 0.25 && form.eventsPerSecond <= 20;

  function setNumber(field: keyof RandomForm, value: string) {
    setForm(previous => ({ ...previous, [field]: Number(value) }));
  }

  function startRandom() {
    const request: RandomDemoRequest = {
      count: form.count,
      interval_ms: Math.round(1000 / form.eventsPerSecond),
      seed: form.seed,
      normal_percent: form.normal,
      suspicious_percent: form.suspicious,
      fraud_percent: form.fraud,
    };
    random.mutate(request);
  }

  return <div className="demo-wrapper">
    <div className="demo-controls" aria-label="Demo controls">
      <button className="button primary" aria-label="Start demo" onClick={() => run.mutate()} disabled={busy}>Start scripted demo</button>
      <button className="button secondary" onClick={() => setShowRandom(value => !value)} disabled={busy} aria-expanded={showRandom}>Random stream</button>
      <button className="button secondary" onClick={() => stop.mutate()} disabled={busy}>Pause</button>
      <button className="button ghost" onClick={() => reset.mutate()} disabled={busy}>Reset</button>
    </div>
    {showRandom && <div className="random-panel" aria-label="Random transaction stream settings">
      <div className="random-panel-heading">
        <div><strong>Seeded random stream</strong><span>Synthetic events through the real scoring path</span></div>
        <span className={totalPercent === 100 ? "mix-valid" : "mix-invalid"}>Mix {totalPercent}%</span>
      </div>
      <div className="random-fields">
        <NumberField label="Transactions" value={form.count} min={1} max={5000} onChange={value => setNumber("count", value)} />
        <NumberField label="Target events / sec" value={form.eventsPerSecond} min={0.25} max={20} step={0.25} onChange={value => setNumber("eventsPerSecond", value)} />
        <NumberField label="Seed" value={form.seed} min={0} max={2147483647} onChange={value => setNumber("seed", value)} />
        <NumberField label="Normal %" value={form.normal} min={0} max={100} onChange={value => setNumber("normal", value)} />
        <NumberField label="Suspicious %" value={form.suspicious} min={0} max={100} onChange={value => setNumber("suspicious", value)} />
        <NumberField label="Fraud ring %" value={form.fraud} min={0} max={100} onChange={value => setNumber("fraud", value)} />
      </div>
      <div className="random-actions">
        <small>Reset before replaying the same seed to avoid replacing identical application IDs.</small>
        <button className="button primary" onClick={startRandom} disabled={busy || !valid}>Start random stream</button>
      </div>
    </div>}
    {error && <p className="inline-error" role="alert">Demo action failed: {error.message}</p>}
  </div>;
}

function NumberField({ label, value, min, max, step = 1, onChange }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: string) => void;
}) {
  return <label><span>{label}</span><input type="number" value={value} min={min} max={max} step={step} onChange={event => onChange(event.target.value)} /></label>;
}
