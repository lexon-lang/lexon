# Shipping Lexon v1.0.0‑rc.1: a practical LLM‑native programming language

Lexon brings orchestration, validation, sessions/RAG, async/await, and multioutput into a single coherent DSL. This RC ships the core language/runtime, CLI, VS Code extension, Python binding, and CI.

## Why a language for LLMs?
SDK calls alone don’t solve concurrency, validation, artifacts, or repeatability. Lexon elevates these as built‑ins so teams can deliver reliable AI features with less glue code.

## What’s in v1.0.0‑rc.1
- Orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`.
- Validation (MVP): `ask_safe`, `confidence_score`, `validate_response`.
- Sessions and context: summarization, compression, key points, context‑window management.
- RAG Lite: ingest, vector search, automatic context.
- Multioutput: a single call that returns text + multiple files + metadata (incl. binary).
- Iterators/FP: `range`, `map`, `filter`, `reduce`, `zip`, `flatten`, `unique`, `sort`, `reverse`, `chunk`, `find`, `count`.
- Async/await runtime with timeouts and cancellation.
- Tooling: `lexc-cli`, linter for async issues, optional OpenTelemetry, VS Code extension, Python binding.

## Configuration (real vs simulated)
Lexon runs simulated by default for determinism. To switch to real providers, set `[system].default_provider` and export API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Provider base URLs can be overridden for proxies.

## End‑to‑end examples
- Orchestration pipelines with parallel+merge and fallback.
- RAG: ingest → vector search → `auto_rag_context()` → `ask`.
- Multioutput: generate HTML/CSS/JSON artifacts in one operation.

## Safety and observability
Sandboxed by default: `execute()` requires `--allow-exec`, absolute paths require `--workspace`. Telemetry is opt‑in via `LEXON_OTEL=1` and `OTEL_EXPORTER_OTLP_ENDPOINT`.

## Value and monetization
Teams get measurable validation, deterministic artifacts, and RAG workflows out‑of‑the‑box. A viable path: open‑core + managed cloud (routing/cache/policies), enterprise features (governance/audit/SSO), team plans and support, plus a plugin marketplace.

## Code quality and GA readiness
The RC builds cleanly, CI gates are in place, and docs include an API index and quickstarts. For GA we plan to reduce default debug verbosity, add snapshot tests for multioutput/RAG, and include a few real‑provider smoke tests.

## Try it
```bash
cargo build --workspace
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
```
Real providers quickstart is in the README and DOCUMENTATION.

Feedback and contributions are welcome—especially around providers, validators, and iterator extensions.


