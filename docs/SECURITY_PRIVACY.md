# Security and privacy boundary

Synchrony is a portfolio prototype and decision-support demonstration. It is not approved to
process real applicants, make autonomous lending decisions, or satisfy a particular regulatory
control framework. `HIGH_RISK` means investigation priority, never automatic credit denial.

## Data handling

The request schema accepts financial and linkage observations because they are needed to build
fraud features. Raw `user_id`, device ID, IP address, bank account ID, income, requested amount,
geography, and the request body are not persisted. The SQL allowlist contains:

- an explicitly public, opaque `application_id`;
- the event's bounded lending-channel category;
- namespace-separated HMAC-SHA256 pseudonyms for user, device, bank, and coarse IP network;
- coarse account-age and loan-to-income bands;
- timestamps, normalized scores, decisions, artifact/config versions, and curated reason codes.

IP addresses are reduced to an IPv4 `/24` or IPv6 `/48` before keyed pseudonymization. HMAC keys
must come from the environment and must not be committed. Pseudonyms remain linkable data rather
than anonymous data; protect the database and key separately. Rotate a compromised key by clearing
the affected prototype namespace and replaying synthetic demo data. The local in-memory velocity
store temporarily holds event linkage values for process-local windows and loses them on restart.

Application IDs shown in the dashboard must be generated opaque public IDs. Do not put a name,
email address, account number, or other personal data into `application_id`.

## Implemented prototype controls

- Strict Pydantic validation, bounded fields, timezone-aware timestamps, and rejected unknown input.
- Sanitized validation/internal errors that do not echo request values or exception messages.
- Parameterized SQLAlchemy statements and explicit transaction rollback.
- Namespace-scoped demo deletion; non-demo namespaces cannot use the reset operation.
- Curated explanation text. Free-form submitted reason messages are never written to the database.
- Environment-only database passwords and HMAC keys; neither container image embeds a secret.
- Restricted CORS, 64 KiB request-size guard, container health checks, non-root application users,
  and `no-new-privileges` in Compose.
- Investigator outcome references are HMAC-pseudonymized; raw reviewer references are not returned
  by the API or stored in SQL.
- No race, religion, or deliberately sensitive demographic attributes in the feature contract.

## Threats and remaining gaps

| Threat | Current mitigation | Remaining production work |
| --- | --- | --- |
| Identifier disclosure at rest | HMAC pseudonyms and narrow SQL allowlist | Managed KMS/HSM, key versioning, encryption at rest, retention jobs |
| Injection or malformed input | Strict request models and ORM parameters | WAF/API gateway limits and adversarial fuzzing |
| Credential disclosure | Environment injection and sanitized errors | Secret manager, automated rotation, audit alerts |
| Unauthorized dashboard/API use | Constant-time API-key checks; reader/admin authorization; production fail-closed; restricted CORS | Replace prototype keys with enterprise OIDC, short-lived tokens, tenant isolation, and immutable audit trail |
| Traffic interception | Bind locally or use an internal Compose network | TLS termination and service identity |
| Abuse and denial of service | Input bounds and body-size guard | Rate limits, quotas, capacity/load testing |
| Model manipulation or drift | Immutable artifact hashes and versioned outputs | Signed artifacts, registry ACLs, monitoring, rollback process |
| False-positive harm and proxy bias | Human-review band and segment report | Governance review, representative data, appeals and outcome monitoring |
| Redis exposure | Atomic adapter uses HMAC-only keys; Compose does not host-publish Redis | Add Redis authentication/TLS and network policy outside a trusted local environment |

PostgreSQL schema creation currently uses `create_all`; production deployment still needs reviewed,
versioned migrations, backup/restore drills, retention policy, access logging, and least-privilege
database roles. Compose credentials are suitable only for a local isolated environment.

## Operator checklist

1. Use synthetic data only and bind services to trusted interfaces.
2. Generate unique high-entropy `PSEUDONYM_KEY` and `POSTGRES_PASSWORD` values outside the repo.
3. Confirm `GET /health` is ready and inspect model/config versions before a demonstration.
4. Redis mode fails readiness when unavailable. Use authentication and TLS for any non-local Redis deployment.
5. Run tests, training/evaluation, responsible-AI reporting, and `scripts/smoke.ps1` after changes.
6. Use `POST /demo/reset` only for the `demo` namespace; deleting the Compose volume is a separate,
   destructive operator action.
