# Shipping Lexon v1.0.0‑rc.1: a practical LLM‑native programming language

Lexon is an AI‑native DSL that treats orchestration, validation, sessions/RAG, async/await, and multioutput as first‑class language features. This RC ships the core runtime, CLI, VS Code extension, Python binding, and a stable CI.

## Why now
Teams struggle to bolt LLMs into products with ad‑hoc glue code. Concurrency, reliability, artifacts, and cost control live outside typical SDKs. Lexon elevates these as built‑ins so you can ship reliable AI features faster.

## What’s in rc.1 (implemented)
- Orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`
- Validation (MVP): `ask_safe`, `confidence_score`, `validate_response`
- Sessions/RAG Lite: ingest, vector search, automatic context, summarization, compression, key points
- Multioutput: text + multiple files (incl. binary) in one call with metadata
- Iterators/FP: `range`, `map`, `filter`, `reduce`, `zip`, `flatten`, `unique`, `sort`, `reverse`, `chunk`, `find`, `count`
- Async runtime with cancellation and timeouts
- Tooling: `lexc-cli`, linter for async issues, optional OTEL, VS Code extension, Python binding

The single source of truth for runnable coverage is the verified demo:
`docs/reference/examples/lexon_verified_demo_working.lx`.

## Quick start
```bash
# build CLI
cargo build --release -p lexc-cli
# run the hello sample (simulated provider by default)
./target/release/lexc-cli -- compile --run samples/00-hello-lexon.lx
```
Switching to real providers is documented in README (set default provider and export API keys).

## Design principles
- Language‑level async/await across LLM, data and I/O
- Deterministic, offline‑first samples and goldens
- Safety by default (sandbox, explicit opt‑ins)
- Observability via opt‑in OpenTelemetry

## Road to GA
- Enrich validation strategies and snapshot tests for multioutput/RAG
- Add a few real‑provider smoke tests gated by env keys
- Reduce default verbosity; improve error messages and docs

## Get involved
- Try the samples, open an issue, or propose a provider/validator.
- VS Code extension lives in `vscode-lexon/` (prebuilt VSIX included).
- Python binding (`crates/lexon-py`) exposes compile/run primitives.

## License
Apache-2.0
