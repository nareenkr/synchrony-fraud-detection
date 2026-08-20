import { FormEvent } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import type { LoanApplicationInput } from "../types";

type FieldProps = {
  label: string;
  name: string;
  type?: "text" | "number" | "datetime-local";
  required?: boolean;
  placeholder?: string;
  min?: string;
  max?: string;
  step?: string;
  pattern?: string;
  defaultValue?: string | number;
};

const identifierPattern = "[A-Za-z0-9][A-Za-z0-9_.:@-]*";
const nowLocal = () => {
  const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60_000);
  return now.toISOString().slice(0, 16);
};

function Field({ label, name, type = "text", ...props }: FieldProps) {
  return <label className="assessment-field"><span>{label}{props.required && <i>Required</i>}</span><input name={name} type={type} {...props} /></label>;
}

const numericFields = new Set([
  "requested_loan_amount", "income", "debt_to_income_ratio", "account_age_days",
  "bank_account_age_days", "device_changes_30d", "login_frequency_24h",
  "failed_login_attempts_24h", "previous_rejected_applications", "transaction_amount",
  "transaction_frequency_24h", "transaction_amount_deviation", "origin_balance_before",
  "origin_balance_after",
]);

export function NewApplication() {
  const navigate = useNavigate();
  const mutation = useMutation({ mutationFn: api.assess, onSuccess: result => navigate(`/applications/${encodeURIComponent(result.application_id)}`) });

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const values = Object.fromEntries(new FormData(event.currentTarget));
    const payload: LoanApplicationInput = {};
    for (const [key, rawValue] of Object.entries(values)) {
      const value = String(rawValue).trim();
      if (!value) continue;
      if (numericFields.has(key)) payload[key] = Number(value);
      else if (key === "unusual_login_location") payload[key] = value === "true";
      else if (key === "event_timestamp") payload[key] = new Date(value).toISOString();
      else payload[key] = value;
    }
    mutation.mutate(payload);
  }

  return <section>
    <div className="page-heading"><div><p className="eyebrow">Manual assessment</p><h1>Assess a loan application</h1><p>Enter a synthetic test application and run it through the complete fraud pipeline.</p></div></div>
    <aside className="data-warning"><strong>Synthetic data only</strong><p>Do not enter a real name, account number, address, government identifier, or customer financial information. Use opaque demo identifiers.</p></aside>
    <form className="assessment-form" onSubmit={submit}>
      <fieldset className="card"><legend>Application and loan</legend><div className="form-grid">
        <Field label="Application ID" name="application_id" required pattern={identifierPattern} placeholder="APP-MANUAL-001" />
        <Field label="Applicant ID" name="user_id" required pattern={identifierPattern} placeholder="USER-DEMO-001" />
        <Field label="Application time" name="event_timestamp" type="datetime-local" required defaultValue={nowLocal()} />
        <label className="assessment-field"><span>Lending channel<i>Required</i></span><select name="channel" defaultValue="WEB"><option value="WEB">Web</option><option value="MOBILE">Mobile</option><option value="PARTNER_API">Partner API</option><option value="AGENT">Agent assisted</option></select></label>
        <Field label="Requested loan amount" name="requested_loan_amount" type="number" required min="0.01" max="10000000" step="0.01" placeholder="5000" />
        <Field label="Annual income" name="income" type="number" min="0.01" max="100000000" step="0.01" placeholder="80000" />
        <Field label="Debt-to-income ratio" name="debt_to_income_ratio" type="number" min="0" max="10" step="0.01" placeholder="0.28" />
        <Field label="Previous rejected applications" name="previous_rejected_applications" type="number" min="0" max="100000" step="1" placeholder="0" />
      </div></fieldset>

      <fieldset className="card"><legend>Account and identity links</legend><div className="form-grid">
        <Field label="Account age (days)" name="account_age_days" type="number" min="0" max="36500" step="1" placeholder="900" />
        <Field label="Bank-account age (days)" name="bank_account_age_days" type="number" min="0" max="36500" step="1" placeholder="750" />
        <Field label="Device ID" name="device_id" pattern={identifierPattern} placeholder="DEVICE-DEMO-001" />
        <Field label="IP address" name="ip_address" placeholder="203.0.113.8" />
        <Field label="Bank-account ID" name="bank_account_id" pattern={identifierPattern} placeholder="BANK-DEMO-001" />
        <Field label="Geographic region" name="geographic_region" placeholder="Demo Region" />
        <Field label="Device changes (30 days)" name="device_changes_30d" type="number" min="0" max="1000" step="1" placeholder="0" />
        <label className="assessment-field"><span>Unusual login location</span><select name="unusual_login_location" defaultValue=""><option value="">Not observed</option><option value="false">No</option><option value="true">Yes</option></select></label>
      </div></fieldset>

      <fieldset className="card"><legend>Recent behaviour and transactions</legend><div className="form-grid">
        <Field label="Login frequency (24 hours)" name="login_frequency_24h" type="number" min="0" max="100000" step="1" placeholder="4" />
        <Field label="Failed logins (24 hours)" name="failed_login_attempts_24h" type="number" min="0" max="100000" step="1" placeholder="0" />
        <Field label="Recent transaction amount" name="transaction_amount" type="number" min="0" max="100000000" step="0.01" placeholder="320" />
        <Field label="Transaction frequency (24 hours)" name="transaction_frequency_24h" type="number" min="0" max="1000000" step="1" placeholder="3" />
        <Field label="Transaction amount deviation" name="transaction_amount_deviation" type="number" min="0" max="1000" step="0.01" placeholder="0.15" />
        <Field label="Balance before transaction" name="origin_balance_before" type="number" min="0" max="1000000000" step="0.01" placeholder="12000" />
        <Field label="Balance after transaction" name="origin_balance_after" type="number" min="0" max="1000000000" step="0.01" placeholder="11680" />
      </div></fieldset>
      <div className="assessment-actions"><p>The application is validated, feature-engineered, scored by all four components, explained, and persisted.</p><button className="button primary" type="submit" disabled={mutation.isPending}>{mutation.isPending ? "Assessing…" : "Assess fraud risk"}</button></div>
      {mutation.error && <p className="assessment-error" role="alert">{mutation.error.message}</p>}
    </form>
  </section>;
}
