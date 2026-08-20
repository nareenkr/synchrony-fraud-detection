import type { ApplicationAssessment } from "../types";
export const assessment: ApplicationAssessment = {
  application_id: "APP-1044", event_timestamp: "2026-08-20T10:29:00Z", assessed_at: "2026-08-20T10:30:00Z", risk_score: 91, decision: "HIGH_RISK",
  channel: "MOBILE", recommended_action: "HOLD_AND_INVESTIGATE",
  component_scores: { supervised_probability: .82, anomaly_score: .71, behavioral_risk: .88, graph_risk: .94 },
  reasons: ["Device is associated with multiple applicants", "Several applications were submitted within one hour"],
  signals: [{ code: "SHARED_DEVICE", message: "Device is associated with multiple applicants", severity: .94, source: "graph" }],
  model_version: "fraud-xgb-v1", risk_config_version: "risk-v1", feature_schema_version: "features-v1",
};
export function json(body: unknown, status = 200): Response { return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }); }
