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

### Notes
- CI runs with deterministic seeds and normalization; real providers are opt‑in for reproducibility.
- See `golden/rag/*` for RAG coverage (tokenize, optimize_window, rerank, fusion, pagination, llm_rerank), and `samples/apps/research_analyst` for a comprehensive demo.


