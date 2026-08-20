# Implementation Plan

## Architectural choices

- Public base data: PaySim transaction data, because it supplies fraud labels, amounts, balances, entities, and time steps. It is synthetic transaction data rather than lending ground truth, so that limitation remains prominent.
- Enrichment: deterministic lending, device, IP, account, behavioral, and graph attributes generated from a fixed seed and stable entity keys. No generated column is represented as observed PaySim data.
- Split strategy: chronological train/validation/test boundaries, with entity-overlap audits. Preprocessors fit on training data only.
- Models: class-weighted logistic regression baseline, XGBoost challenger, and Isolation Forest with empirical quantile calibration. Selection emphasizes PR-AUC and recall subject to false-positive/manual-review capacity rather than accuracy.
- Explanations: SHAP for the selected XGBoost tree model and a deterministic perturbation equivalent for supported linear models, translated into curated reason codes and plain language.
- Runtime: FastAPI, SQLAlchemy with SQLite/PostgreSQL, in-memory or atomic Redis TTL state, React polling, and a non-root Compose deployment.

## Implementation status

Milestones M0-M8 are implemented. Automated backend, model, state, persistence, API, and frontend checks pass; the local HTTP smoke and deterministic double replay pass. Compose configuration validates with runtime secrets, but container startup still requires a running Docker engine. Visual browser verification remains a manual gate when an in-app browser surface is available.

## Milestones and dependencies

### M0 — Blueprint and contracts

Define the architecture, canonical schemas, enums, package boundaries, risk configuration, provenance catalog, requirement IDs, and verification evidence. This milestone blocks all others.

Exit gate: design documents agree on data order, interfaces, API contracts, privacy boundaries, and completion evidence.

### M1 — Canonical data and leakage-safe replay

Implement Pydantic event/assessment schemas, deterministic fixture/enrichment generation, chronological splits, dataset manifest, and point-in-time offline replay.

Depends on M0. Blocks honest model development.

Exit gate: deterministic hashes and row counts, provenance validation, no future/self-counting, and split/entity leakage tests pass.

### M2 — Shared features and online state

Implement the ordered feature contract, shared feature builder, in-memory TTL store, Redis-compatible interface, behavioral windows, and lightweight shared-device/IP/bank counts.

Depends on M1. Blocks training/inference parity and hybrid scoring.

Exit gate: feature formulas, expiry, missing values, new users, repeated applications, and offline/online parity pass.

### M3 — Supervised training and evaluation

Train logistic regression and XGBoost pipelines with class weighting. Tune an operating threshold on validation data, freeze it, evaluate once on test data, and persist a validated bundle.

Depends on M1-M2.

Exit gate: precision, recall, F1, ROC-AUC, PR-AUC, FPR, confusion matrix, threshold table, PR/ROC plots, feature importance, and selection rationale are retained.

### M4 — Anomaly, behavioral, and graph components

Train Isolation Forest, calibrate its score, implement behavioral rules, and implement bounded graph/cardinality signals.

Depends on M2; supervised and anomaly work can proceed independently after the shared feature contract freezes.

Exit gate: every component returns finite normalized values and deterministic reason codes; stateful scenarios change risk in the expected direction.

### M5 — Hybrid risk and explainability

Implement validated weights/thresholds, decisions, SHAP adapter, reason mapping, hybrid evaluation, and ablations.

Depends on M3-M4. Blocks production `/predict`.

Exit gate: boundary/normalization tests pass; classifier-only and hybrid metrics/ablations are reported; canonical scenario ordering is normal < suspicious < ring.

### M6 — Persistence and FastAPI vertical slice

Implement repositories, migrations/schema initialization, required endpoints, dashboard query endpoints, privacy protections, sanitized errors, and API tests.

Depends on M5.

Exit gate: a prediction traverses validation through persistence and retrieval with complete assessment data and no stored raw identifiers.

### M7 — Simulator and dashboard

Implement fixed normal, suspicious, and fraud-ring sequences; CLI and API simulation; React monitoring feed, analytics, investigation view, model info, and demo controls.

Depends on M6.

Exit gate: reset/replay is deterministic, all decisions appear, details survive refresh, responsive/error states are covered by component/build checks, and a browser smoke test is performed where a browser surface is available.

### M8 — Hardening and reproducibility

Add PostgreSQL/Redis adapters, Compose, responsible-AI segment report/model card, threat/privacy model, dependency/security checks, operational smoke/load checks, and final documentation.

Depends on the working vertical slice. Optional graph clustering and SSE are considered only here and only if they improve the product.

Exit gate: both lightweight local and Compose paths build, start, seed, demonstrate, test, and stop from documented commands.

## Deterministic demo contract

- Normal: established account, stable device, normal frequency and loan/income ratio -> score below approve threshold.
- Suspicious: young account, new device, burst, shared IP, large ratio -> manual review band.
- Fraud ring: multiple identities sharing device/IP, rapid applications, abnormal behavior -> high-risk band.

Assertions target score bands, ordering, decisions, and required signals. They do not depend on brittle exact floating-point scores.

## Completion evidence

Retain test results, coverage, dataset manifest/hash, artifact/config hashes, evaluation JSON and plots, threshold rationale, hybrid ablation, model card, segment metrics with sample sizes, privacy/threat documentation, dependency scan, API smoke evidence, and browser evidence when the runtime provides a browser surface.
