# Hackathon Submission Alignment

This project chooses the real-time digital-lending fraud problem statement. The matrix separates
implemented evidence from optional technology and from submission actions that require the student.

## Exact problem-statement match

| Challenge phrase | Implemented resolution |
|---|---|
| Real-time across digital lending channels | Every validated event identifies `WEB`, `MOBILE`, `PARTNER_API`, or `AGENT`; the same synchronous scoring path returns and persists an assessment immediately, while the dashboard polls the live feed each second. |
| ML and behavioral analytics | XGBoost/logistic fraud probability is combined with Isolation Forest anomaly detection, velocity/behavioral rules, and shared-device/IP/bank graph signals. |
| Identify, prevent, and respond | The API returns `APPROVE`, `MANUAL_REVIEW`, or `HIGH_RISK` plus an operational action: continue checks, require enhanced verification, or hold and investigate. These are fraud controls, not autonomous credit decisions. |
| Sophisticated and dynamic patterns | Prior-only rolling state detects bursts, device changes, login anomalies, and coordinated identity rings instead of relying on static rules alone. |
| Proactive and adaptive | Investigators record `CONFIRMED_FRAUD`, `LEGITIMATE`, or `INCONCLUSIVE`. The governed learning status measures reviewed false positives/missed fraud and recommends retraining review only after sufficient evidence. The live model is never silently self-modified. |
| Reduce false positives and preserve trust | A manual-review band, human-readable reasons, component scores, investigator correction, audit versions, and explicit recommended actions keep a human in control. |

| Requested area | Status | Evidence / decision |
|---|---|---|
| React JS frontend | Implemented | `frontend/src` monitoring, investigation, model-info, error/empty/loading states, responsive CSS |
| Spring Boot or equivalent API | Implemented | FastAPI/Pydantic service with OpenAPI-first request/response contracts |
| PostgreSQL structured database | Implemented | SQLAlchemy repository; PostgreSQL 16 in Compose; SQLite is the lightweight local default |
| pgvector / vector database | Not applicable | No semantic-search requirement exists in this fraud statement; adding a vector store would not improve the detection path |
| Bedrock / LLM / embeddings | Not applicable | Fraud scores are deterministic, testable ML and rules. Curated explanations avoid hallucination and do not require a generative model |
| Cloud/deployment platform | Deployment-ready | Non-root Docker images, Compose, health checks, PostgreSQL/Redis, environment configuration. No claim of an actual AWS deployment |
| Secure storage/access | Implemented for prototype | Environment-only secrets, HMAC pseudonyms, persistence allowlist, no raw request logs, non-root containers |
| Authentication/authorization basics | Implemented | Production requires distinct reader/admin API keys; constant-time checks; reader GET-only, admin mutation access |
| Input validation / secure API | Implemented | Strict Pydantic schemas, finite/bounded values, 64 KiB request cap, sanitized 4xx/5xx responses, request IDs, restricted CORS |
| Monitoring/logging | Basic | Liveness/readiness checks and sanitized structured request-error logging; production telemetry backend remains future work |
| Clean, modular code and naming | Implemented | Separate API, services, features, fraud, explainability, state, persistence, and training packages |
| API-first design | Implemented | Typed OpenAPI endpoints plus frontend contract normalization and API tests |
| README/setup/run instructions | Implemented | Local, training, testing, troubleshooting, and Compose workflows in `README.md` |
| Architecture diagram | Implemented | Mermaid diagram in `docs/ARCHITECTURE.md` and a presentation architecture slide |
| Unit/basic test coverage | Implemented | Backend and training suite plus frontend component tests; exact evidence in `docs/VERIFICATION.md` |
| Responsible AI | Implemented | Model card, segment report with sample support, false-positive safeguards, human-review boundary, no autonomous denial |
| Explainability/transparency | Implemented | Curated reasons from SHAP/rule contributions; component scores, model/config/schema versions exposed safely |
| Git/GitHub | Partially prepared | Git is initialized on `main`; the student must commit with their own identity and push to their own GitHub repository before submission |
| PPT or PDF | Prepared | Generate the PDF deck with `python scripts/build_submission_deck.py --roll-number <ROLL>` |
| Recorded demo | Student action required | Record using `docs/DEMO_RECORDING.md`; a recording cannot be fabricated by the codebase |
| Roll-number-only names / ZIP | Prepared | `scripts/package_submission.py` validates the roll number and creates `<ROLL>.pdf`, `<ROLL>.mp4`, and `<ROLL>.zip` |

## Why the optional AI stack is intentionally absent

The architecture examples say a strong prototype *may* include a vector database, LLM, prompt
templates, guardrails, or an agent. Those components fit semantic search and generative workflows.
This problem is tabular, temporal fraud decisioning. XGBoost, Isolation Forest, velocity rules, and
shared-entity graph signals are more auditable, deterministic, and measurable. The absence of an
LLM is an engineering decision, not an incomplete dependency.

## Claims that must stay qualified

- The bundled fixture and PaySim are synthetic; their metrics are development evidence only.
- `HIGH_RISK` means investigation priority, never automatic credit rejection.
- Reader/admin keys demonstrate basic access control, not enterprise identity management.
- Compose demonstrates portable deployment, not proof of a live AWS production environment.
- The student must add their roll number, record the demo, create Git/GitHub evidence, and submit
  before **12:00 PM IST on 21 August**. The campus pitch is scheduled for **24-25 August**.
