# Security Notes (RC.1)

- Sandbox & I/O
  - CLI enforces workspace sandbox (--workspace) for file ops.
  - Avoid absolute paths unless explicitly allowed.
- Tools & MCP
  - Permission profiles via LEXON_MCP_PERMISSION_PROFILE.
  - Tool quotas and auditing enabled; failures gated with timeouts.
- Limits & Quotas
  - Per-provider budgets/quotas; rate limiter primitive.
  - Multioutput limits and MIME validations.
- Data Privacy
  - Schema and PII gates configurable; optional blocking.
- Reproducibility
  - Deterministic runs default in CI (simulated providers, LEXON_SEED=0).



