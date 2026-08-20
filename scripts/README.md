# Developer scripts

Run these PowerShell scripts from any directory; each resolves the repository root itself.

- `bootstrap.ps1` creates `.venv`, installs Python development/PostgreSQL/Redis extras, runs
  `npm ci`, and executes the deterministic training pipeline. Use `-SkipTraining` or
  `-SkipFrontend` for a shorter setup.
- `train.ps1` runs preparation, supervised training, anomaly training, test evaluation, plots,
  hybrid evaluation, and responsible-AI segmentation with seed `20260819`. Override the
  interpreter with `-Python`, the seed with `-Seed`, or provide a downloaded public PaySim CSV
  with `-InputPath`.
- `smoke.ps1` checks readiness, model metadata, prediction, persistence, analytics, and the full
  deterministic demo through HTTP. Start the backend first. Use `-BaseUrl` for a different address
  or `-SkipDemo` for only the synchronous endpoints.

Examples:

```powershell
.\scripts\bootstrap.ps1
.\scripts\train.ps1 -Python .\.venv\Scripts\python.exe
.\scripts\train.ps1 -InputPath C:\path\to\paysim.csv
.\scripts\smoke.ps1 -BaseUrl http://127.0.0.1:8000
```

The scripts fail on the first unsuccessful command and call shared Python modules rather than
duplicating application or scoring logic.
