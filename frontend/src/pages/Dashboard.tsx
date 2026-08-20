import { useQuery } from "@tanstack/react-query";
import { Area, AreaChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { api } from "../api";
import { DecisionBadge } from "../components/DecisionBadge";
import { DemoControls } from "../components/DemoControls";
import { EmptyState, ErrorState, LoadingState } from "../components/StatePanel";
import type { Decision } from "../types";

const colors: Record<Decision, string> = { APPROVE: "#25c49a", MANUAL_REVIEW: "#f4b860", HIGH_RISK: "#ef6b73" };
const decisionLabel: Record<Decision, string> = { APPROVE: "Approved", MANUAL_REVIEW: "Manual review", HIGH_RISK: "High risk" };
const formatTime = (value: string) => new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date(value));

export function Dashboard() {
  const applications = useQuery({ queryKey: ["applications"], queryFn: api.applications, refetchInterval: 1000 });
  const analytics = useQuery({ queryKey: ["analytics"], queryFn: api.analytics, refetchInterval: 1000 });
  const loading = applications.isLoading || analytics.isLoading;
  const error = applications.error || analytics.error;
  const retry = () => { void applications.refetch(); void analytics.refetch(); };
  if (loading) return <section><PageHeading /><LoadingState label="Connecting to the decision stream" /></section>;
  if (error || !analytics.data || !applications.data) return <section><PageHeading /><ErrorState message={error?.message ?? "The service returned an incomplete response."} retry={retry} /></section>;
  const stats = analytics.data;
  const riskTrend = stats.risk_trend.length ? stats.risk_trend : [...applications.data].reverse().map((application, index) => ({ timestamp: application.assessed_at, average_risk_score: application.risk_score, total_applications: index + 1 }));
  const distribution = (Object.entries(stats.decision_distribution) as [Decision, number][]).map(([name, value]) => ({ name, value }));
  return <section>
    <PageHeading />
    <div className="kpi-grid">
      <Kpi label="Total applications" value={stats.total_applications.toLocaleString()} detail="Processed this session" accent="teal" />
      <Kpi label="Approved" value={stats.approved_applications.toLocaleString()} detail={percentage(stats.approved_applications, stats.total_applications)} accent="green" />
      <Kpi label="Manual reviews" value={stats.manual_review_applications.toLocaleString()} detail={percentage(stats.manual_review_applications, stats.total_applications)} accent="amber" />
      <Kpi label="High risk" value={stats.high_risk_applications.toLocaleString()} detail={percentage(stats.high_risk_applications, stats.total_applications)} accent="red" />
      <Kpi label="Average risk" value={stats.average_risk_score.toFixed(1)} detail="Out of 100" accent="purple" />
      <Kpi label="Fraud rate" value={`${(stats.fraud_rate <= 1 ? stats.fraud_rate * 100 : stats.fraud_rate).toFixed(1)}%`} detail="High-risk decisions" accent="red" />
    </div>
    <div className="dashboard-grid">
      <article className="card chart-card wide"><CardHeading title="Risk activity" subtitle="Average risk score over the current demo session" />
        {riskTrend.length ? <div className="chart-wrap" aria-label="Risk activity chart"><ResponsiveContainer width="100%" height="100%"><AreaChart data={riskTrend} margin={{ top: 12, right: 10, left: -20, bottom: 0 }}><defs><linearGradient id="riskFill" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#36d3a9" stopOpacity={0.35}/><stop offset="95%" stopColor="#36d3a9" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="#173b35" strokeDasharray="4 4" vertical={false}/><XAxis dataKey="timestamp" tickFormatter={formatTime} tick={{ fill: "#789e96", fontSize: 11 }} axisLine={false} tickLine={false}/><YAxis domain={[0, 100]} tick={{ fill: "#789e96", fontSize: 11 }} axisLine={false} tickLine={false}/><Tooltip contentStyle={{ background: "#102c27", border: "1px solid #285149", borderRadius: 10 }}/><Area type="monotone" dataKey="average_risk_score" name="Risk score" stroke="#36d3a9" strokeWidth={2.5} fill="url(#riskFill)" /></AreaChart></ResponsiveContainer></div> : <EmptyState title="No trend yet" body="Start the demo to build a live risk timeline." />}
      </article>
      <article className="card chart-card"><CardHeading title="Decision mix" subtitle="Current session distribution" />
        {stats.total_applications ? <div className="donut-layout"><div className="donut"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={58} outerRadius={78} paddingAngle={3}>{distribution.map((item) => <Cell key={item.name} fill={colors[item.name]} />)}</Pie><Tooltip contentStyle={{ background: "#102c27", border: "1px solid #285149", borderRadius: 10 }}/></PieChart></ResponsiveContainer><div className="donut-total"><strong>{stats.total_applications}</strong><span>Total</span></div></div><ul className="legend">{distribution.map(item => <li key={item.name}><span style={{ background: colors[item.name] }} />{decisionLabel[item.name]}<strong>{item.value}</strong></li>)}</ul></div> : <EmptyState title="No decisions yet" body="Decision counts appear as applications arrive." />}
      </article>
    </div>
    <article className="card feed-card"><div className="feed-heading"><CardHeading title="Live applications" subtitle="Updates every second from the persisted decision stream" /><span className="live-indicator"><i /> Live</span></div>
      {applications.data.length ? <div className="table-scroll"><table><thead><tr><th>Application</th><th>Assessed</th><th>Risk</th><th>Decision</th><th><span className="sr-only">Open</span></th></tr></thead><tbody>{applications.data.map(app => <tr key={app.application_id}><td><Link to={`/applications/${encodeURIComponent(app.application_id)}`} className="application-link">{app.application_id}</Link></td><td>{formatTime(app.assessed_at)}</td><td><div className="risk-cell"><strong>{app.risk_score.toFixed(0)}</strong><span className="risk-track"><i style={{ width: `${app.risk_score}%` }} /></span></div></td><td><DecisionBadge decision={app.decision} /></td><td><Link className="row-action" aria-label={`Investigate ${app.application_id}`} to={`/applications/${encodeURIComponent(app.application_id)}`}>→</Link></td></tr>)}</tbody></table></div> : <EmptyState title="Waiting for applications" body="Use Start demo to stream the normal, suspicious, and fraud-ring scenarios." />}
    </article>
  </section>;
}

function PageHeading() { return <div className="page-heading"><div><p className="eyebrow">Real-time overview</p><h1>Fraud monitoring</h1><p>Track lending decisions and investigate emerging risk.</p></div><DemoControls /></div>; }
function CardHeading({ title, subtitle }: { title: string; subtitle: string }) { return <div className="card-heading"><h2>{title}</h2><p>{subtitle}</p></div>; }
function Kpi({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) { return <article className={`kpi-card accent-${accent}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>; }
function percentage(value: number, total: number) { return total ? `${((value / total) * 100).toFixed(1)}% of total` : "0% of total"; }
