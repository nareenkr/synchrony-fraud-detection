import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { DecisionBadge } from "../components/DecisionBadge";
import { RiskGauge } from "../components/RiskGauge";
import { EmptyState, ErrorState, LoadingState } from "../components/StatePanel";
import type { ComponentScores, InvestigatorOutcome } from "../types";

const labels: Record<keyof ComponentScores, string> = { supervised_probability: "Model probability", anomaly_score: "Anomaly score", behavioral_risk: "Behavioral risk", graph_risk: "Graph risk" };

export function Investigation() {
  const { applicationId = "" } = useParams();
  const queryClient = useQueryClient();
  const [reviewerId, setReviewerId] = useState("DEMO-REVIEWER");
  const query = useQuery({ queryKey: ["application", applicationId], queryFn: () => api.application(applicationId), enabled: Boolean(applicationId) });
  const review = useMutation({ mutationFn: (outcome: InvestigatorOutcome) => api.submitReview(applicationId, outcome, reviewerId), onSuccess: () => void queryClient.invalidateQueries({ queryKey: ["learning-status"] }) });
  if (query.isLoading) return <LoadingState label="Loading investigation" />;
  if (query.error || !query.data) return <ErrorState message={query.error?.message ?? "Application not found."} retry={() => void query.refetch()} />;
  const app = query.data; const reasons = app.reasons ?? app.signals.map(signal => signal.message);
  return <section><Link to="/" className="back-link">← Back to monitoring</Link><div className="page-heading investigation-heading"><div><p className="eyebrow">Application investigation</p><h1>{app.application_id}</h1><p>Assessed {new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "medium" }).format(new Date(app.assessed_at))}</p></div><DecisionBadge decision={app.decision} /></div>
    <div className="investigation-grid">
      <article className="card risk-summary"><div><p className="eyebrow">Combined fraud risk</p><RiskGauge score={app.risk_score} /></div><div className="decision-copy"><span>Recommended action</span><h2>{app.recommended_action.replaceAll("_", " ")}</h2><p>Decision: {app.decision.replaceAll("_", " ")} · Channel: {app.channel.replaceAll("_", " ")}</p></div></article>
      <article className="card component-card"><div className="card-heading"><h2>Detection components</h2><p>Normalized contribution indicators</p></div><div className="component-list">{Object.entries(app.component_scores).map(([key, value]) => <div key={key}><div><span>{labels[key as keyof ComponentScores]}</span><strong>{Math.round(value * 100)}%</strong></div><span className="component-track"><i style={{ width: `${value * 100}%` }} /></span></div>)}</div></article>
      <article className="card reasons-card"><div className="card-heading"><h2>Why this decision?</h2><p>Human-readable factors, without sensitive rule internals</p></div>{reasons.length ? <ol>{reasons.map((reason, index) => <li key={`${reason}-${index}`}><span>{index + 1}</span>{reason}</li>)}</ol> : <EmptyState title="No major reasons" body="No material risk explanation was generated for this assessment." />}</article>
      <article className="card signals-card"><div className="card-heading"><h2>Detected signals</h2><p>Evidence grouped by detection source</p></div>{app.signals.length ? <div className="signal-list">{app.signals.map(signal => <div className="signal" key={`${signal.code}-${signal.source}`}><span className={`signal-source source-${signal.source}`}>{signal.source}</span><div><strong>{signal.message}</strong><small>{signal.code.replaceAll("_", " ")} · severity {Math.round(signal.severity * 100)}%</small></div></div>)}</div> : <EmptyState title="No signals detected" body="This application did not trigger a curated fraud signal." />}</article>
      <article className="card feedback-card"><div className="card-heading"><h2>Investigator outcome</h2><p>Verified outcomes measure false positives and governed model drift.</p></div><label>Reviewer reference<input aria-label="Reviewer reference" value={reviewerId} onChange={event => setReviewerId(event.target.value)} /></label><div className="feedback-actions"><button className="button secondary" disabled={review.isPending || !reviewerId} onClick={() => review.mutate("CONFIRMED_FRAUD")}>Confirm fraud</button><button className="button secondary" disabled={review.isPending || !reviewerId} onClick={() => review.mutate("LEGITIMATE")}>Mark legitimate</button><button className="button ghost" disabled={review.isPending || !reviewerId} onClick={() => review.mutate("INCONCLUSIVE")}>Inconclusive</button></div>{review.isSuccess && <p className="feedback-success">Outcome recorded. Learning indicators have been refreshed.</p>}{review.error && <p className="inline-error">{review.error.message}</p>}</article>
      <article className="card audit-card"><div className="card-heading"><h2>Assessment record</h2><p>Versions retained for auditability</p></div><dl><div><dt>Assessed at</dt><dd>{new Date(app.assessed_at).toISOString()}</dd></div><div><dt>Lending channel</dt><dd>{app.channel.replaceAll("_", " ")}</dd></div><div><dt>Model version</dt><dd>{app.model_version}</dd></div><div><dt>Risk policy</dt><dd>{app.risk_config_version}</dd></div><div><dt>Feature schema</dt><dd>{app.feature_schema_version}</dd></div></dl></article>
    </div>
  </section>;
}
