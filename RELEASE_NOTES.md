## Lexon v1.0.0-rc.1 — Release Notes

### Overview
Lexon is an LLM‑native DSL with first‑class async/await, orchestration, anti‑hallucination, RAG Lite, sessions, and multioutput. This RC includes the CLI, VS Code extension, Python binding, and CI.

### Key features
- Orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`.
- Anti‑hallucination (MVP): `ask_safe`, `confidence_score`, `validate_response`.
- RAG Lite: `memory_index.ingest`, `memory_index.vector_search`, `auto_rag_context`.
- Structured semantic memory: `memory_space.*`, `remember_*`, `recall_*` with pluggable tree/trie/graph backends (`basic`, `patricia`, `raptor`, `hybrid`).
- Sessions: `session_summarize`, `session_compress`, `extract_key_points`, `context_window_manage`.
- Multioutput: `ask_multioutput`, `get_multioutput_*`, `save_multioutput_file`, `load/save_binary_file`.
- Iterators/FP: `enumerate`, `range`, `map`, `filter`, `reduce`, `zip`, `flatten`, `unique`, `sort`, `reverse`, `chunk`, `find`, `count`.
- Async/await runtime, timeouts, cancellation.
- Tooling: `lexc-cli` (`compile`, `bench`, `lint`), VSIX, optional OpenTelemetry.

### Configuration
- Defaults via `lexon.toml` (`[system] default_provider`, provider `default_model`).
- Env overrides: `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`; API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`.
- Simulated by default; real calls enabled when provider/model + keys are set.

### Security and telemetry
- Sandbox: `execute()` disabled unless `--allow-exec`; absolute paths blocked unless `--workspace`.
- Telemetry optional: `LEXON_OTEL=1`, `OTEL_EXPORTER_OTLP_ENDPOINT`.

### Quality
- CI gates: fmt, clippy `-D warnings`, tests, linter, samples smoke, OTEL smoke.
- Docs: `README.md` + `DOCUMENTATION.md` with API index and quickstarts.

### Getting started
```bash
cargo build --workspace
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
```

Real providers quickstart is included in the README and DOCUMENTATION.

### Notes for GA
- Reduce debug verbosity by default; keep `--verbose`/env for deep traces.
- Add snapshot tests for multioutput artifacts and RAG contexts.
- Include 2–3 real‑provider smoke tests behind env keys.


