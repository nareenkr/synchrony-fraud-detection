# Synchrony Fraud Decisioning Prototype

Synchrony is an end-to-end real-time fraud decision-support prototype for digital lending. It
validates a loan event, computes leakage-safe point-in-time features, combines supervised fraud
probability, Isolation Forest anomaly score, behavioral velocity rules, and shared-entity graph
signals, then returns a configurable `0..100` risk score, decision, and human-readable reasons.
FastAPI persists privacy-reduced results for a React monitoring and investigation dashboard. A
deterministic simulator provides normal, suspicious, and likely fraud-ring demonstrations across
web, mobile, partner-API, and assisted-agent channels. Investigator outcomes feed a governed
learning monitor that measures false-positive reviews and missed fraud without silently changing
the live model.

This is synthetic-data portfolio software, not a production lending control. `HIGH_RISK` means
escalation to a human investigator; it is not authority for autonomous credit denial.

## Architecture and data

```text
validated event -> prior-only state snapshot -> shared feature builder
  -> supervised + anomaly + behavior + graph -> configurable risk policy
  -> curated explanation -> privacy-reduced SQL record -> API/dashboard
```

Training and inference use the same ordered feature contract. Preparation creates chronological
train/validation/test partitions; preprocessing fits on training data only, model/threshold
selection uses validation, and the test partition is evaluated once. The public base is PaySim,
which is itself synthetic mobile-money data—not lending ground truth. Lending, device, IP, bank,
login, and velocity observations are deterministic synthetic enrichment. Provenance for every
field is recorded in [data/feature_provenance.yaml](data/feature_provenance.yaml).

More detail:

- [Architecture](docs/ARCHITECTURE.md)
- [Implementation plan](docs/IMPLEMENTATION_PLAN.md)
- [Requirement traceability](docs/REQUIREMENTS.md)
- [Final verification record](docs/VERIFICATION.md)
- [Security and privacy boundary](docs/SECURITY_PRIVACY.md)
- [Hackathon checklist alignment](docs/HACKATHON_ALIGNMENT.md)
- [Recorded-demo runbook](docs/DEMO_RECORDING.md)

## Local setup (SQLite and in-memory state)

Prerequisites are Python 3.11-3.13, Node.js 20+, npm, and PowerShell 7 or Windows PowerShell 5.1.

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\Activate.ps1
```

Bootstrap installs dependencies, creates deterministic development data/artifacts/reports, and
installs the frontend lockfile. Existing artifacts can be used with a shorter setup:

```powershell
.\scripts\bootstrap.ps1 -SkipTraining
```

Start the backend in one terminal:

```powershell
$env:APP_ENV = "local"
$env:DATABASE_URL = "sqlite:///./synchrony.db"
$env:STATE_BACKEND = "memory"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the dashboard in another terminal:

```powershell
Set-Location frontend
npm.cmd run dev
```

Open `http://localhost:5173`. API docs are at `http://127.0.0.1:8000/docs`. Verify the complete
HTTP path after startup with `.\scripts\smoke.ps1`.

## Deterministic training and evaluation

The complete command is `.\scripts\train.ps1`. Its equivalent stages are:

```powershell
python -m training.prepare_data --seed 20260819 --output-dir data/processed
python -m training.train_classifier --seed 20260819 --data-dir data/processed --output-dir artifacts/supervised-v1 --model-version supervised-v1 --overwrite
python -m training.train_anomaly --seed 20260819 --data-dir data/processed --output artifacts/anomaly-v1.joblib
python -m training.evaluate --data-dir data/processed --bundle-dir artifacts/supervised-v1 --output reports/model_evaluation.md
python -m training.plot_evaluation --data-dir data/processed --bundle-dir artifacts/supervised-v1 --output-dir reports/plots
python -m training.hybrid_eval --data-dir data/processed --classifier-bundle artifacts/supervised-v1 --anomaly-artifact artifacts/anomaly-v1.joblib
python -m training.responsible_ai --data-dir data/processed --classifier-bundle artifacts/supervised-v1 --anomaly-artifact artifacts/anomaly-v1.joblib
```

Provide `--input path/to/paysim.csv` to `training.prepare_data` to use a local PaySim file. No
dataset is downloaded automatically. With no input, it uses the deterministic development fixture.

### Public PaySim dataset

PaySim is a synthetic mobile-money simulator based on aggregated transaction patterns, rather than
real lending-application ground truth. The original project and citation are available from the
[PaySim repository](https://github.com/EdgarLopezPhD/PaySim) and the
[2016 simulator paper](https://www.msc-les.org/proceedings/emss/2016/EMSS2016_249.pdf). The commonly
used labeled CSV is distributed through
[Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1). Download it under its applicable terms;
large raw datasets are intentionally not committed or downloaded by setup scripts.

Run the complete pipeline against that CSV with:

```powershell
.\scripts\train.ps1 -InputPath C:\path\to\PS_20174392719_1491204439457_log.csv
```

Or supply `-InputPath` to `bootstrap.ps1` for first-time setup and full-data training. The input must
contain the documented PaySim source columns; preparation fails closed on a missing or malformed
schema. The generated manifest records row counts, chronological boundaries, provenance classes,
seed, content hash, and entity-overlap audit.

## Example API request

```powershell
$body = @{
  application_id = "APP-EXAMPLE-001"
  user_id = "USER-EXAMPLE-001"
  event_timestamp = "2026-08-20T10:00:00Z"
  requested_loan_amount = 5000.0
  channel = "MOBILE"
  income = 80000.0
  account_age_days = 900
  device_id = "DEVICE-EXAMPLE-001"
  ip_address = "203.0.113.8"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" -Body $body
```

Dashboard endpoints include `GET /applications`, `GET /applications/{id}`, `GET /analytics`,
`POST /applications/{id}/review`, `GET /learning/status`, and
the `/demo/run`, `/demo/random/run`, `/demo/status`, `/demo/stop`, and `/demo/reset` controls.

### Manual loan assessment

Open **Assess application** in the dashboard to enter a synthetic loan request manually. The form
supports the full validated event contract: loan and income values, account tenure, lending
channel, opaque device/IP/bank links, login observations, and recent transaction behaviour. It
calls the same `POST /predict` endpoint as the simulators, then opens the persisted investigation
with the ML risk score, decision, component scores, reasons, and recommended action. Do not enter
real customer or personal data into this prototype.

An investigator can label an assessed case as `CONFIRMED_FRAUD`, `LEGITIMATE`, or `INCONCLUSIVE`.
The reviewer reference is HMAC-pseudonymized and never returned. Learning status aggregates only
verified outcomes. Before the minimum evidence threshold it remains `COLLECTING_FEEDBACK`; after
that, harmful false-positive or missed-fraud evidence can trigger `RETRAINING_REVIEW_REQUIRED`.
Retraining, frozen validation, approval, and versioned deployment remain explicit offline gates.

### Automated random transaction stream

Open **Random stream** on the monitoring dashboard to configure a bounded synthetic stream. The
defaults generate 100 events at a target of two per second with an 80% normal, 15% suspicious, and
5% fraud-ring mix. Count is capped at 5,000 and target rate at 20 events per second. Actual throughput
also includes model scoring, explanation, and persistence time. The percentages must total 100.

The seed makes the event order and values reproducible. Suspicious profiles share user/device/IP
clusters to create bursts; fraud-ring profiles share devices, IPs, and sometimes bank accounts.
Every event passes through the real decisioning, state, explanation, persistence, and analytics
path. Reset before replaying the same seed because reproducible application IDs intentionally
upsert rather than create duplicates.

The equivalent API call is:

```powershell
$random = @{
  count = 100
  interval_ms = 500
  seed = 20260820
  normal_percent = 80
  suspicious_percent = 15
  fraud_percent = 5
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/demo/random/run `
  -Method Post -ContentType "application/json" -Body $random
```

## Docker Compose

Compose runs the non-root backend and frontend plus PostgreSQL and Redis. The backend uses
PostgreSQL persistence and the atomic Redis state adapter; Redis is service-internal and is not
published to the host. Generate local-only secrets in your shell; they are passed at runtime and
never baked into an image:

```powershell
$env:POSTGRES_PASSWORD = -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
$env:PSEUDONYM_KEY = -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 48 | ForEach-Object {[char]$_})
$env:API_READ_KEY = -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 40 | ForEach-Object {[char]$_})
$env:API_ADMIN_KEY = -join ((48..57)+(65..90)+(97..122) | Get-Random -Count 40 | ForEach-Object {[char]$_})
docker compose up --build
```

Open `http://localhost:5173`; the API is exposed on `http://localhost:8000`. Inspect readiness with
`docker compose ps` and `Invoke-RestMethod http://localhost:8000/health`. Stop containers with
`docker compose down`. Adding `--volumes` deletes the local PostgreSQL demo database and is
destructive. The Compose dashboard asks for an access key: the reader key permits investigation
GETs and the administrator key also permits scoring and demo-control POSTs. The entered key is kept
in tab-scoped `sessionStorage`; it is never compiled into the frontend. Local development leaves
authentication disabled unless `AUTH_ENABLED=true` is configured with two distinct 32+ character
keys. Production configuration fails closed when authentication is disabled.

## Testing

```powershell
pytest -q
ruff check .
Set-Location frontend
npm.cmd test
npm.cmd run build
```

## Hackathon presentation and submission package

Build the 14-slide PDF with the student's real roll number:

```powershell
python scripts\build_submission_deck.py --roll-number <ROLL> --output submission\final\<ROLL>.pdf
```

After recording the MP4 with [the demo runbook](docs/DEMO_RECORDING.md), generate and validate all
roll-number-only artifacts in one step:

```powershell
python scripts\package_submission.py --roll-number <ROLL> --video C:\path\to\recorded-demo.mp4
```

This produces `submission/final/<ROLL>.pdf`, `<ROLL>.mp4`, and `<ROLL>.zip`; the ZIP contains only
the identically based PDF and MP4. Review every file before emailing it. The communicated deadline
is **12:00 PM IST on 21 August** and the campus pitch is scheduled for **24-25 August**.

## Privacy, limitations, and troubleshooting

SQL storage contains HMAC pseudonyms and coarse derived bands, never raw user/device/IP/bank IDs,
income, requested amount, request bodies, or arbitrary reason text. Application IDs must be opaque
public IDs. Secrets belong in environment variables. See
[docs/SECURITY_PRIVACY.md](docs/SECURITY_PRIVACY.md) for controls and gaps.

- **Health says models are not ready:** run `scripts/train.ps1` and confirm artifact paths match the
  environment.
- **Port already in use:** change the host side of the Compose port mapping, or stop the conflicting
  local process.
- **Frontend cannot reach the API:** local Vite expects `/api`; the Compose build targets
  `http://localhost:8000`. Rebuild after changing `VITE_API_BASE_URL`.
- **PostgreSQL authentication fails:** use URL-safe password characters, keep the same exported
  value, and recreate only the disposable demo volume if its initialized password differs.
- **Demo reset is forbidden:** `PERSISTENCE_NAMESPACE` must be `demo`, `demo-*`, or `demo_*`.
- **Multiple local backend workers show inconsistent velocity:** memory state is process-local. Set
  `STATE_BACKEND=redis` with the optional Redis dependency, or use the Compose stack.

Model metrics on synthetic/development data do not demonstrate real-world validity, fairness,
security, or regulatory compliance. False positives can harm applicants; human review, appeal,
monitoring, representative validation data, and formal governance remain mandatory.
