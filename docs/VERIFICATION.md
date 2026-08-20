# Verification record

Final audit date: 2026-08-20 (Asia/Calcutta).

This record distinguishes executable evidence from environmental limitations. It does not claim
that synthetic-data metrics establish production fitness.

## Automated quality gates

| Gate | Command/evidence | Result |
|---|---|---|
| Python lint | `python -m ruff check .` | Pass |
| Backend/model tests | `python -m pytest --cov=backend.app --cov=training --cov-report=term -q` | 125 passed; 81% measured coverage; no warnings |
| Feedback/adaptation | persistence, API, and frontend tests | Investigator outcome upsert, pseudonymous reviewer boundary, false-positive/missed-fraud aggregation, and governed retraining status pass |
| Access control | protected FastAPI test client | Public health; unauthenticated denial; reader GET-only; admin mutation; production fail-closed |
| Python dependencies | `python -m pip check` | No broken requirements |
| Frontend tests | `npm.cmd test -- --run` | 3 files, 5 tests passed |
| Random stream | seeded generator, API, and dashboard tests | Exact weighted mix, deterministic replay, correlation preservation, bounds, and real decisioning path pass |
| Frontend production build | `npm.cmd run build` | TypeScript and Vite build passed; 682 modules transformed |
| Frontend dependencies | `npm.cmd audit --audit-level=high` | 0 vulnerabilities reported |
| Compose interpolation | runtime-only PostgreSQL/HMAC secrets plus `docker compose config --quiet` | Pass |
| PowerShell scripts | PowerShell parser over `scripts/*.ps1` | Pass |

## Training and retained evidence

`scripts/train.ps1 -Seed 20260819` completed preparation, logistic/XGBoost comparison, anomaly
training, frozen evaluation, six plots, hybrid ablation, and responsible-AI segmentation.

- Prepared source: explicitly labeled `development_fixture`, 600 rows, chronological 420/90/90
  train/validation/test split, oversampled for pipeline verification.
- Selected classifier: XGBoost; frozen threshold `0.0056898766197264194`.
- Untouched fixture test metrics: recall 1.00, FPR 0.0909, precision 0.20, F1 0.3333,
  ROC-AUC 1.00, PR-AUC 1.00.
- Hybrid review-or-higher metrics: recall 1.00, FPR 0.00, precision/F1 1.00 on the same fixture.
- The reports and model card explicitly prohibit interpreting these designed-fixture results as
  real lending performance.

Artifact integrity was recomputed and matched the frozen manifests:

- classifier: `9cab56a0be8cd46a7468fcd187f94dd0f8e6b31ac32e4665522229afefe50c36`
- feature schema: `ddd388ba04fd0325b07f4a1cbc63d9463ee5a4fbcbe27a69d20c5c23da77afe4`
- anomaly model: `97287bbe5f00550dfeacb9b8816920b71731bf35f152fcc14e1d9e8952a145df`
- prepared manifest: `80ad3b60010180b30451585bf5c471a617413d53a17bb53a87160a3904f94546`
- risk policy: `risk-v1-f6274a8c88e7`

## Runtime evidence

- The real application loaded the retained XGBoost and Isolation Forest artifacts and reported
  ready through `/health` and safe metadata through `/model-info`.
- `scripts/smoke.ps1` passed health, model info, prediction, persistence, detail, analytics, and
  the background simulator over HTTP.
- The API accepts web, mobile, partner-API, and agent channels; detail responses expose the channel
  and a decision-specific operational action. Investigator feedback updates `/learning/status`.
- Two independent HTTP reset/replays produced identical 11 application IDs, scores, and decisions:
  1 `APPROVE`, 4 `MANUAL_REVIEW`, and 6 `HIGH_RISK`.
- A separate real-artifact regression repeats the canonical sequence twice inside the suite and
  verifies normal `<40`, every suspicious case `40..69`, and every ring case `>=70`.
- The final production dashboard bundle served HTTP 200 and both referenced entry assets served
  successfully. Component tests cover monitoring KPIs/feed/demo start, empty state, investigation
  risk/components/reasons, and model metadata.

## Environment-only limitations

- Browser-client discovery returned no available in-app or connected browser surface, so a visual
  click/screenshot pass could not be performed in this environment. No unrelated standalone
  browser runner was substituted. The production bundle, HTTP entry/assets, and component tests
  passed.
- Docker Compose configuration validates, but the Docker Desktop Linux engine was not running, so
  containers could not be started here. The lightweight SQLite/in-memory stack was started and
  exercised end to end. Compose startup remains a documented operator check on a Docker-enabled
  machine.
