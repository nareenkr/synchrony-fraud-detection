import type { Analytics, ApplicationAssessment, AssessmentResult, Decision, DemoStatus, InvestigatorOutcome, InvestigatorReview, LearningStatus, LoanApplicationInput, ModelInfo, RandomDemoRequest } from "./types";
const API_BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");
const ACCESS_KEY_STORAGE = "synchrony-api-access-key";
export const authRequired = import.meta.env.VITE_AUTH_REQUIRED === "true";
export const getAccessKey = (): string => sessionStorage.getItem(ACCESS_KEY_STORAGE) ?? "";
export const setAccessKey = (value: string): void => sessionStorage.setItem(ACCESS_KEY_STORAGE, value);
export const clearAccessKey = (): void => sessionStorage.removeItem(ACCESS_KEY_STORAGE);
export class ApiError extends Error { constructor(public status: number, message: string) { super(message); this.name = "ApiError"; } }
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const accessKey = getAccessKey();
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...(accessKey ? { "X-API-Key": accessKey } : {}), ...init?.headers } });
  if (!response.ok) { let message = `Request failed (${response.status})`; try { const body = await response.json() as { detail?: string }; if (body.detail) message = body.detail; } catch { /* safe fallback */ } throw new ApiError(response.status, message); }
  return response.json() as Promise<T>;
}
const numeric = (value: unknown): number => typeof value === "number" && Number.isFinite(value) ? value : 0;
const decisions: Decision[] = ["APPROVE", "MANUAL_REVIEW", "HIGH_RISK"];
export function normalizeApplications(payload: ApplicationAssessment[] | { applications?: ApplicationAssessment[]; items?: ApplicationAssessment[] }): ApplicationAssessment[] { return Array.isArray(payload) ? payload : payload.applications ?? payload.items ?? []; }
export function normalizeAnalytics(raw: Record<string, unknown>): Analytics {
  const distribution = (raw.decision_distribution ?? raw.decision_counts ?? {}) as Partial<Record<Decision, number>>;
  const approved = numeric(raw.approved_applications ?? distribution.APPROVE); const review = numeric(raw.manual_review_applications ?? raw.manual_reviews ?? distribution.MANUAL_REVIEW); const highRisk = numeric(raw.high_risk_applications ?? distribution.HIGH_RISK); const total = numeric(raw.total_applications) || approved + review + highRisk;
  return { total_applications: total, approved_applications: approved, manual_review_applications: review, high_risk_applications: highRisk, average_risk_score: numeric(raw.average_risk_score), fraud_rate: raw.fraud_rate === undefined && raw.fraud_high_risk_rate === undefined ? (total ? highRisk / total : 0) : numeric(raw.fraud_rate ?? raw.fraud_high_risk_rate), decision_distribution: Object.fromEntries(decisions.map((decision) => [decision, numeric(distribution[decision]) || ({ APPROVE: approved, MANUAL_REVIEW: review, HIGH_RISK: highRisk }[decision])])) as Record<Decision, number>, risk_trend: Array.isArray(raw.risk_trend) ? raw.risk_trend as Analytics["risk_trend"] : [] };
}
export const api = {
  assess: (payload: LoanApplicationInput) => request<AssessmentResult>("/predict", { method: "POST", body: JSON.stringify(payload) }),
  applications: async () => normalizeApplications(await request<ApplicationAssessment[] | { applications?: ApplicationAssessment[]; items?: ApplicationAssessment[] }>("/applications?limit=50&offset=0")),
  application: (id: string) => request<ApplicationAssessment>(`/applications/${encodeURIComponent(id)}`),
  analytics: async () => normalizeAnalytics(await request<Record<string, unknown>>("/analytics")),
  modelInfo: () => request<ModelInfo>("/model-info"),
  learningStatus: () => request<LearningStatus>("/learning/status"),
  submitReview: (id: string, outcome: InvestigatorOutcome, reviewerId: string) => request<InvestigatorReview>(`/applications/${encodeURIComponent(id)}/review`, { method: "POST", body: JSON.stringify({ outcome, reviewer_id: reviewerId }) }),
  startDemo: () => request<DemoStatus>("/demo/run", { method: "POST", body: "{}" }), startRandom: (config: RandomDemoRequest) => request<DemoStatus>("/demo/random/run", { method: "POST", body: JSON.stringify(config) }), stopDemo: () => request<DemoStatus>("/demo/stop", { method: "POST" }), resetDemo: () => request<DemoStatus>("/demo/reset", { method: "POST" }),
};
