export type Decision = "APPROVE" | "MANUAL_REVIEW" | "HIGH_RISK";
export type LendingChannel = "WEB" | "MOBILE" | "PARTNER_API" | "AGENT";
export type InvestigatorOutcome = "CONFIRMED_FRAUD" | "LEGITIMATE" | "INCONCLUSIVE";
export interface ComponentScores { supervised_probability: number; anomaly_score: number; behavioral_risk: number; graph_risk: number }
export interface RiskSignal { code: string; message: string; severity: number; source: "supervised" | "anomaly" | "behavioral" | "graph" }
export interface ApplicationAssessment { application_id: string; event_timestamp?: string; assessed_at: string; risk_score: number; decision: Decision; channel: LendingChannel; recommended_action: "CONTINUE_WITH_STANDARD_CHECKS" | "REQUIRE_ENHANCED_VERIFICATION" | "HOLD_AND_INVESTIGATE"; component_scores: ComponentScores; reasons?: string[]; signals: RiskSignal[]; model_version: string; risk_config_version: string; feature_schema_version: string }
export interface AnalyticsPoint { timestamp: string; average_risk_score: number; total_applications: number }
export interface Analytics { total_applications: number; approved_applications: number; manual_review_applications: number; high_risk_applications: number; average_risk_score: number; fraud_rate: number; decision_distribution: Record<Decision, number>; risk_trend: AnalyticsPoint[] }
export interface FeatureDescription { name: string; description?: string; minimum?: number; maximum?: number }
export interface ModelInfo { model_name?: string; model_version: string; bundle_id?: string; feature_schema_version: string; risk_config_version?: string; status?: string; classifier_threshold?: number; thresholds?: { manual_review: number; high_risk: number }; weights?: Record<string, number>; metrics?: Record<string, number | null>; features?: Array<string | FeatureDescription>; prototype_only?: boolean }
export interface DemoStatus { running?: boolean; status?: string; message?: string }
export interface RandomDemoRequest { count: number; interval_ms: number; seed: number; normal_percent: number; suspicious_percent: number; fraud_percent: number }
export interface LearningStatus { reviewed_applications: number; confirmed_fraud: number; legitimate: number; inconclusive: number; false_positive_reviews: number; missed_fraud_reviews: number; reviewed_alerts: number; false_positive_review_rate: number; minimum_feedback_required: number; retraining_recommended: boolean; governance_status: "COLLECTING_FEEDBACK" | "RETRAINING_REVIEW_REQUIRED" | "MONITORING" }
export interface InvestigatorReview { application_id: string; outcome: InvestigatorOutcome; reviewed_at: string }
