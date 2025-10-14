## RC Roadmap Snapshot

This roadmap reflects the v1.0.0-rc.1 scope and is the authoritative plan for this RC bundle.

### Implemented in this RC
- Core: `typeof`, `Ok`, `Error`, `is_ok`, `is_error`, `unwrap`.
- LLM orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`, `ask_safe`.
- Providers (real): OpenAI (`gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`), Anthropic (`claude-*`), Google Gemini (`gemini-1.5-pro`, `gemini-1.5-flash`).
- Anti‑hallucination helpers: `validate_response(basic)`.
- Sessions/context: basic session state (`session_start`, `session_ask`, `session_history`).
- RAG Lite: vector store with deterministic embeddings; basic `vector_search`.
- Multioutput (Preview): `ask_multioutput`, helpers `get_multioutput_*`, `save_multioutput_file`, `load/save_binary_file`; real_binaries (opt‑in), auto‑mkdir, MIME inference, pseudo‑streaming.
- Iterators/FP: `map`, `filter`, `reduce`, `range`, `enumerate`, etc.
- Async runtime: async/await, timeouts, cancellation.
- Data ops: CSV/JSON load/filter/select/take/export; `load_csv`, `save_csv`, `load_json`, `save_json`.
- I/O (sandboxed): `read_file`, `write_file`, `save_file`, `load_file`, `execute`.
- Config: `set_default_model`, `get_provider_default`, `lexon.toml`.
- Tooling: CLI `compile`, `bench`, `lint`; VSIX; CI in bundle.

### Planned for GA (granular)
- Anti‑hallucination: `ask_with_validation`, `confidence_score`, domain validators; enrich `validate_response` strategies.
- Sessions: add `session_summarize`, `session_compress`, `context_merge/retrieve`.
- RAG: real embeddings, `auto_rag_context` end‑to‑end, hybrid search.
- CLI: `run`, `config`, `new`; improve `lint` rules (blocking I/O, missing await).
- Telemetry: default OTEL templates, spans for data/memory/LLM; sampling.
- Multioutput: incremental streaming API, non text‑like binaries, per‑file metadata, callbacks/progress, limits and validations.
- Providers: routing policies (cost/latency), health checks.

### LLM‑first completeness plan 
- Prompting & evaluation
  - Prompt templates/versioning, prompt registry per project.
  - Evals datasets, golden tests, coverage, automatic metrics (exact, BLEU/ROUGE, task metrics).
- Tool calling & permissions
  - Tool registry with scopes/permissions and policies (allowlist/denylist, quotas).
  - MCP integration (1.1) with permission profiles and auditing.
- Structured streaming
  - Streaming API for LLM (tokens/chunks + metadata) and Multioutput (per‑file progress).
- Stdlib & infra
  - http/ws client, timers/sleep, path utils, basic zip/crypto.
  - Secrets manager and sandbox/policy profiles.
- Dev UX & packaging
  - `lexc new` (project/modules/deps scaffold); `lexc run`; REPL and basic debugging.
- Testing
  - Native testing framework (assert/fixtures), snapshot tests, property/fuzz for the DSL.
- Cost, quotas and reliability
  - Per‑execution budgets, rate limiting, unified retry/backoff.
- Routing/providers
  - A/B & canary, health checks, policies by cost/latency/capacity, automatic model selection.
- Persistence
  - Durable stores for session/memory/cache with TTL/GC.
- Observability
  - Full coverage of tracing/metrics/logs, SLOs, ready‑made dashboards (OTEL).
- Publishing & distribution
  - Official binaries/containers, stable plugin/SDK.
- Processes & docs
  - Formal API Freeze, stability levels, migration guides.

### Missing or experimental (post‑GA)
- Validation: `ask_verified`, `configure_validation`, `hallucination_detect`.
- Agents: `agent_create/run`, chains/parallel, supervisor/collab, state_machine, workflows.
- Concurrency DSL: `spawn_task`, `join_all`, `select_first`, `timeout_after`, channels, rate limiter.
- Providers: `configure_provider`, advanced routing, provider health monitoring.
- Cache: invalidation and distributed cache.

### Function gaps index (planned)
- Anti‑hallucination (advanced):
  - ask_verified(prompt, domain, options)
  - ask_with_validation(prompt, config)
  - confidence_score(response)
  - hallucination_detect(response, context)
  - configure_validation(config)
- Sessions and context:
  - session_summarize(session_id, options)
  - session_compress(session_id, options)
  - extract_key_points(session_id, options)
  - context_window_manage(session_id, options)
  - context_retrieve(session_id, query)
  - context_merge([sessions], options)
  - session_set_objectives(session_id, objectives)
  - session_track_context(session_id, context)
  - session_check_alignment(session_id, options)
  - analyze_summary_quality(session_id, metrics)
  - session_configure(session_id, config)
- Global configuration:
  - set_global_mode(mode)
  - set_provider_default(provider, model)
  - set_use_case_default(use_case, model)
  - get_use_case_default(use_case)
  - configure_lexon(config)
- Iteration/FP:
  - for_each(array, callback)
- RAG / Vector:
  - memory_index.hybrid_search(query, k)
  - auto_rag_context()
  - embedding_generation()
  - semantic_similarity()
- Cache:
  - cache_invalidation()
  - distributed_cache()
- Agents (autonomous) and orchestration:
  - agent_create/run, agent_chain/parallel, agent_state_machine
  - agent_workflow/supervisor/collaboration/delegation
- Concurrency advanced:
  - spawn_task, join_all, select_first, timeout_after
  - channels, select!, retry, rate_limiter
- Providers system:
  - configure_provider(), intelligent routing, provider health monitoring
- CLI & Tooling:
  - lexc repl, lexc config, lexc new, lexc run (subcomando dedicado)

### MCP server support (target: 1.1)
- Core loop: JSON‑RPC 2.0 over stdio (first) and WebSocket (optional).
- Tool registry: expose Lexon functions as MCP tools with arity/type validation.
- Concurrency & cancellation: map `CancellationToken/TaskHandle` to cancel requests; configurable timeouts.
- Signals & lifecycle: SIGINT/SIGTERM → cancel/cleanup; orderly shutdown.
- Security: I/O sandbox and `execute()` opt‑in; size/duration limits.
- Configuration: `lexon.toml` + env; provider/model selection per tool.
- Observability: spans per request/response, error/latency counters.
- Packaging: `lexc mcp` subcommand with flags (`--stdio`, `--ws`, `--workspace`, `--allow-exec`).


