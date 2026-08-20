# Synchrony fraud dashboard

React + TypeScript monitoring UI for the fraud decisioning prototype. It polls the backend once per second while live mode is enabled and exposes monitoring, investigation, model information, and deterministic demo controls.

```powershell
npm.cmd install
npm.cmd run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` to target another backend. Run `npm.cmd test` and `npm.cmd run build` before shipping.

This interface uses synthetic demonstration applications and is a prototype decision-support tool. It is not approved for autonomous lending decisions.
