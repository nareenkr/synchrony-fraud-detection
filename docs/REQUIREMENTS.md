# Requirement Traceability

This matrix preserves the original scope. A green unit test alone is not proof for a broad product requirement; each row identifies the authoritative artifact and final evidence needed.

| ID | Requirement | Implemented artifacts | Verified evidence |
|---|---|---|---|
| DATA-01 | Public dataset plus explicit real/synthetic separation | `training.prepare_data`, source-aware manifest, provenance catalog, `-InputPath` scripts | fixture/public source kind, filename/hash, and provenance tests pass |
| DATA-02 | No leakage and training/inference parity | chronological replay and shared feature/state contracts | temporal split, prior-only state, entity-overlap, and parity tests pass |
| ML-01 | Interpretable baseline and stronger classifier | logistic and XGBoost training pipelines | held-out comparison and selection rationale retained in the artifact manifest/report |
| ML-02 | Imbalance-aware thresholding and required metrics | class weights, validation threshold selection, evaluation/plot CLIs | required metrics, confusion matrix, ROC/PR/threshold plots retained |
| ML-03 | Unsupervised anomaly detection | Isolation Forest and empirical quantile calibrator | digest/schema/load and finite normalized-output tests pass |
| RISK-01 | Behavioral and velocity detection | TTL windows and curated behavioral rules | repeated/device/IP/failed-login scenario tests pass |
| RISK-02 | Lightweight graph fraud signals | shared device/IP/bank cardinality scorer | fraud-ring and bounded-score tests pass |
| RISK-03 | Configurable 0-100 hybrid score and decisions | `config/risk.yaml` and validated risk engine | weight, threshold, boundary, normalization, and scenario tests pass |
| EXPL-01 | SHAP/equivalent and human-readable reasons | SHAP/perturbation adapters and curated reason catalog | attribution and sanitization tests pass; no raw contributions returned |
| API-01 | Required FastAPI endpoints | health, model-info, and predict routes | strict contract, malformed-input, and real-artifact HTTP smoke pass |
| API-02 | Persistence/query/analytics endpoints | SQLAlchemy repository and dashboard routes | transaction, pagination, detail, analytics, privacy, and HTTP smoke tests pass |
| RT-01 | Real-time state with simple fallback | in-memory and atomic Redis adapters | prior-only, TTL, ordering, HMAC key, namespace, and adapter tests pass |
| UI-01 | Monitoring and investigation dashboard | React/Vite dashboard and typed client | 3 test files/5 tests and production build pass; visual browser unavailable in this runtime |
| DEMO-01 | Normal, suspicious, and ring simulations | deterministic simulator through the production decision path | two HTTP reset/replays matched 11 IDs/scores/decisions: 1 approve, 4 review, 6 high risk |
| EVAL-01 | Classifier and hybrid evaluation report | retained Markdown/JSON reports and six plots | reports include threshold rationale, scenarios, and component ablations |
| RAI-01 | Responsible-AI analysis and limitations | model card and segment report | support, recall, FPR, review/selection rates, uncertainty, and limits retained |
| SEC-01 | Validation, privacy, secrets, safe storage/logging | strict schemas, limits, HMAC/coarsening, safe errors/configuration | raw-value SQL/Redis command inspection, 64 KiB guard, safe-error tests, and npm audit pass |
| TEST-01 | Required unit/API/model/edge tests | backend and frontend suites | Ruff clean; 123 backend tests pass; measured backend/training coverage 80%; frontend tests/build pass |
| DX-01 | Reproducible local and Compose setup | scripts, README, env example, non-root Dockerfiles, Compose | local HTTP smoke passes; Compose config validates; startup awaits a running Docker engine |
