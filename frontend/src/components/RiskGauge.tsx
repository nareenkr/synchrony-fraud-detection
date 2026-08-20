import type { CSSProperties } from "react";
export function RiskGauge({ score }: { score: number }) { const bounded = Math.max(0, Math.min(100, score)); return <div className="risk-gauge" style={{ "--risk": `${bounded * 3.6}deg` } as CSSProperties} role="img" aria-label={`Fraud risk ${bounded} out of 100`}><div><strong>{Math.round(bounded)}</strong><span>/ 100</span></div></div>; }
