import type { Decision } from "../types";
const labels: Record<Decision, string> = { APPROVE: "Approved", MANUAL_REVIEW: "Manual review", HIGH_RISK: "High risk" };
export function DecisionBadge({ decision }: { decision: Decision }) { return <span className={`decision-badge decision-${decision.toLowerCase()}`}><span aria-hidden="true" className="decision-dot" />{labels[decision]}</span>; }
