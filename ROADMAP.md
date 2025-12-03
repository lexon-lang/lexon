## Roadmap Snapshot

This roadmap reflects the v1.0.0-rc.1 scope and the current, implemented reality.

### Implemented (GA scope in this bundle)
- Core & Concurrency: `typeof`, `Ok`/`Error`, async orchestration (`task.spawn/await`, `join_all`/`join_any`), channels + `select_any`, `timeout`, `retry`, `rate_limiter`.
- LLM Orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`, `ask_safe`, structured streaming (`STREAM_CHUNK`/`STREAM_EVENT`).
- Sessions & Context: start/ask/history, summarize/compress, context retrieve/merge, durable store, TTL/GC + `sessions.gc_now`, list/delete.
- RAG Advanced:
  - Tokenization & Chunking: precise per‑model BPE for `rag.tokenize`/`rag.token_count`/`rag.chunk_tokens`; `memory_index.ingest_chunks`.
  - Retrieval: hybrid search (SQLite + Qdrant) with alpha weighting, metadata filters, pagination (offset/limit_factor), auto ensure collection, retries/backoff/throttle, raw Qdrant filter passthrough, schema/index helpers.
  - Post‑processing: `rag.rerank` (LLM), `rag.rerank_cross_encoder` (batched; env `LEXON_RERANK_BATCH_SIZE`/`LEXON_RERANK_MAX_ITEMS`), `rag.fuse_passages`, `rag.fuse_passages_semantic` (+citations), `rag.summarize_chunks`, `rag.optimize_window` (token‑budget), `memory_index.hybrid_search_page`/`hybrid_search_all`, `memory_index.hybrid_search_llm_rerank`.
- Structured semantic memory:
  - Canonical `memory_space` + `remember_*` + `recall_*` primitives with multi-level summaries, pinning, policies, deterministic resets/freeze_clock, and before-action hooks.
  - Pluggable backends selectable via `LEXON_MEMORY_BACKEND=basic|patricia|raptor|hybrid` (basic tree, Patricia trie, RAPTOR clustering, GraphRAG/MemTree hybrid).
  - Deterministic sample/golden (`samples/memory/structured_semantic.lx` + `golden/memory/structured_semantic.txt`) and real-provider verification (`remember_raw` + OpenAI).
- Providers & Routing v2: OpenAI/Anthropic/Gemini + Generic (HuggingFace/Ollama/Custom), budgets/quotas, retries/backoff, health/capacity/failure‑rate, canary, decision metrics.
- Agents & Orchestration: create/run/chain/parallel/cancel, deadlines/budgets, supervisor (state/list), tool registry with scopes/quotas, `on_tool_call`/`on_tool_error`, OTEL spans.
- Anti‑hallucination & Quality: confidence scoring, validation (schema/PII) via `quality.*`, configurable gates.
- MCP 1.1: stdio/WS, tools DSL (register/list/tool_info/set_quota), quotas/audit, cancellation/timeouts, sandbox/allowlist, env/lexon.toml config, spans/metrics.
- Data Ops: CSV/JSON/Parquet load/save; fixtures; ETL minis.
- Cache & Persistence: cache TTL/GC + invalidation, distributed cache stub, durable session/vector stores.
- CLI & Linter: `lexc run`/`config`/`new`; strict linter (blocking I/O, missing await); samples + goldens.
- Metrics & Observability: per‑call logs, rollups JSON, Prometheus export, starter Grafana dashboard (latency/cost/quality overlays), OTEL hooks.
- Packaging & CI: release binaries, wheels, Homebrew, Dockerfiles; PR + nightly matrices; golden coverage.

### Next refinements (post‑GA, high signal)
- Qdrant convenience presets for typed payload indexes (keyword/number/date) and compact filter builders.
- Telemetry dashboards: enrich model/provider breakdowns, error stratification, budget/quality overlays.
- Provider expansion & rerank: optional adapters (Azure OpenAI/Bedrock) and external cross‑encoder endpoints.
- DX & Samples: VS Code LSP enrichments, more end‑to‑end samples; optional REPL.
- AOT packaging (embedder first: LexIR + VM as single binary).
- IR optimizer (const‑fold, DCE, selective inline) and robust string‑concat lowering.
- Fuzzing nightly for parser/HIR/IR with minimal corpus and crash triage.
- Web search presets and stable goldens per provider (normalized `/Results`).

### Planned enhancements (prioritized)
- Language/IR robustness: explicit coercion rules, predictable truthiness, support for safe top‑level expressions, improved lowering for concatenations (recommend `strings.join`).
- Types and data modeling: lightweight ADTs (Option/Result), better branch type unification, pattern matching, first‑class generics (scoped), JSON path contracts.
- Modules and packaging: package manager (index + lockfile), workspace support, semantic versioning, clearer module roots.
- Stdlib and I/O: regex captures/groups, richer time (parse/format/tz), JSONPath, Parquet/Arrow, streaming CSV/HTTP, DataFrame bridge (Arrow).
- HTTP client extensions (native): multipart/form‑data upload, download‑to‑file (streamed), per‑request retry/backoff and redirect policy, optional cookie jar; keep `LEXON_ALLOW_HTTP` gating.
- JSON↔Dataset bridges: direct bridge from JSON array of flat objects to Dataset and helper `json_array_to_csv` (or equivalent) to simplify CSV export without manual loops.
- Sockets (opt‑in, sandboxed): TCP client builtins (`tcp.connect/send/recv/close`) behind `LEXON_ALLOW_NET`, with quotas, size/time limits and no server‑side listeners.
- FP ergonomics: first‑class lambdas/closures for `map`/`filter`/`reduce` instead of string expressions; maintain string form as a compatibility layer.
- Observability: full OTLP exporter, spans for LLM/HTTP/RAG/MCP, ready‑to‑use Grafana dashboards.
- MCP/agents: contract versioning, backpressure and streaming tokens, reproducible multi‑agent examples.
- Python ecosystem: MCP tools for pandas/numpy a corto plazo; puente Arrow/builtins `py.*` a medio plazo.
- CI hardening: property‑based tests, fuzz gates, deterministic seeds y budgets, smoke opcionales “real‑mode” con secrets.

<!-- Auxiliary operational notes were removed to keep the roadmap strictly prospective. -->


