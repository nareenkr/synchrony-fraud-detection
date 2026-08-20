# Recorded Demo Runbook

Target a 4-5 minute recording. Use only synthetic demo data and keep terminals free of secrets.

## Before recording

1. Run `scripts/bootstrap.ps1 -SkipTraining` once.
2. Start the backend and frontend using the local commands in `README.md`.
3. Run `scripts/smoke.ps1` and confirm readiness is `ready`.
4. Close unrelated windows and hide notifications, email, tokens, and environment-variable values.
5. Open the dashboard at `http://localhost:5173` and reset the demo.

## Suggested narration and clicks

### 0:00-0:35 — problem and value

“Loan fraud changes faster than static rules. Synchrony combines supervised probability, anomaly,
behavioral velocity, and shared-entity graph signals to prioritize investigations in real time. It
is decision support, never authority for autonomous credit denial.”

Show the monitoring screen and point to the synthetic/prototype label.

### 0:35-1:10 — architecture

Briefly show the architecture slide or `docs/ARCHITECTURE.md`. Mention React, FastAPI, PostgreSQL,
Redis, the shared feature builder, and safe explanations. State that pgvector/Bedrock are not used
because this statement has no semantic-search or generative requirement.

### 1:10-2:20 — live deterministic scenario

Click **Reset**, then **Start demo**. Explain that every fixture goes through the real `/predict`
pipeline and is persisted—there are no random client-side KPIs. Point out normal approvals,
manual-review applications, and the shared device/IP fraud-ring sequence moving to high risk.

Optionally open **Random stream** and show the bounded count, events-per-second, seed, and risk-mix
controls. Explain that the generator is random-looking but seeded for replay, and that suspicious
bursts and fraud-ring relationships are correlated rather than independently random fields.

Optionally open **Assess application**, enter an opaque synthetic application ID, requested amount,
income, channel, and behaviour observations, and submit it. Explain that manual entries and
simulated entries use the identical validation, feature, ML, explanation, and persistence path.

### 2:20-3:10 — investigation and explainability

Open a high-risk row. Show the 0-100 score, decision, four component scores, curated reasons,
signals, lending channel, recommended operational action, timestamp, and model/config/schema
versions. Mark the case **Confirm fraud**, **Mark legitimate**, or **Inconclusive** and explain that
the investigator outcome is the trusted feedback signal. Identifiers are pseudonymized and raw
financial inputs are not persisted.

### 3:10-3:45 — model evidence

Open **Model info**. Describe chronological splitting, train-only preprocessing, validation-time
threshold selection, and point-in-time prior-only features. Clearly say the displayed metrics use a
small synthetic development fixture and do not establish production or fairness readiness. Show
the governed learning loop: reviewed evidence, false-positive review rate, missed fraud, and
retraining status. State that it recommends an offline review and never auto-deploys a model.

### 3:45-4:20 — engineering and security

Show `README.md`, the tests, and the protected Compose login if time permits. Mention strict input
validation, reader/admin access, environment-only secrets, health checks, non-root containers, and
the responsible-AI segment report.

### 4:20-4:40 — close

“The next production gates are representative labeled lending data, calibration and drift
monitoring, enterprise OIDC, an investigator appeal workflow, and controlled cloud deployment.”

## Recording acceptance check

- Dashboard, one high-risk investigation, and model information are readable at 1080p.
- No real customer data, passwords, API keys, email, or personal notifications are visible.
- Audio is clear and the recording plays from beginning to end.
- Rename the final recording through `scripts/package_submission.py`; do not hand-rename only one
  artifact and risk inconsistent roll-number naming.
