![Lexon](assets/lexon-wordmark.svg)

## Lexon v1.0.0-rc.1

A self-contained RC of the Lexon language and tooling: language/runtime, CLI, samples, Tree-sitter grammar, VS Code extension, and Python binding.

### Why Lexon (quick)
- Orchestrate LLM + data as first-class language features (async/await, parallel/merge/fallback/ensemble).
- Built-in reliability: validation (`ask_safe`) with confidence scoring; sessions/RAG Lite.
- Deterministic artifacts: multioutput (text + multiple files/binaries) from a single operation.

### What is Lexon?
Lexon is an LLM‑first programming language with first‑class async/await, LLM orchestration (parallelism, merge, fallback, ensemble), functional data processing, multioutput generation, session memory, and built‑in validation (anti‑hallucination). It emphasizes determinism (seeds, snapshots) and safe defaults (sandboxed I/O).

This RC includes:
- Advanced RAG helpers (hybrid search, rerank, fuse, summarize).
- MCP stdio/WS servers with quotas/schemas/heartbeats/progress.
- Web search builtin (configurable via `lexon.toml`) and HTTP data sources.
- A focused stdlib (encoding/strings/math/regex/time/number/crypto/json).
- Multi‑file modules and aliasing.

Quick taste:
```lexon
pub fn main() {
  set_default_model("simulated");
  let r = ask("Say hello from Lexon");
  print(r);
}
```
Run it with the CLI (simulated by default):
```bash
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
```

### Contents
- `lexc/`: language/runtime implementation (Rust)
- `lexc-cli/`: CLI tool
- `tree-sitter-lexon/`: parser sources
- `samples/`: deterministic, offline-first examples + goldens
- `vscode-lexon/`: VS Code extension (prebuilt VSIX)
- `crates/lexon-py/`: Python binding (maturin/pyo3)
- `lexon.toml`: default configuration
- `Makefile.toml`: convenience tasks

### Documentation
- Documentation: [DOCUMENTATION.md](DOCUMENTATION.md)
- Roadmap: [ROADMAP.md](ROADMAP.md)
- Release Notes: [RELEASE_NOTES.md](RELEASE_NOTES.md)
- Blog Draft: [blog_post.md](blog_post.md)

### 5-minute recipes (copy/paste)
- Mini‑ETL (medallion‑lite):
```bash
cargo run -q -p lexc-cli -- compile --run samples/etl/mini_medallion.lx
```
Generates `output/etl_top_high.json` (filtered CSV) and `output/etl_report.md` (LLM summary). Uses real LLM if keys are set; otherwise simulated.

- Notes organizer:
```bash
cargo run -q -p lexc-cli -- compile --run samples/notes/organizer.lx
```
Creates `output/notes_ActionItems.md` and `output/notes_Summary.json` from a small note set using `ask_merge("synthesize")` and `ask_safe`.

### Configuration (summary)
- Modes: simulated (default) vs real provider calls.
- Defaults: loaded from `lexon.toml` → `[system] default_provider`, per‑provider defaults.
- Env overrides: `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`.
- API keys: set provider envs (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`).
- Programmatic helpers: `set_default_model(model)`, `get_provider_default(provider)`.
- Telemetry (optional): `LEXON_OTEL=1` and `OTEL_EXPORTER_OTLP_ENDPOINT`.
- RAG/rerank knobs: `LEXON_RERANK_BATCH_SIZE`, `LEXON_RERANK_MAX_ITEMS`.
- Vector store: `LEXON_VECTOR_BACKEND=sqlite_local|qdrant`, `LEXON_QDRANT_URL`, `LEXON_QDRANT_COLLECTION`, `LEXON_QDRANT_THROTTLE_MS`.
- Sandbox: `--allow-exec` to enable `execute()`, `--workspace PATH` for absolute paths.

See detailed configuration in `DOCUMENTATION.md`.

### Validation helper: ask_with_validation
Use `ask_with_validation(prompt, config_json)` to get validated text directly (returns string). Example:
```lexon
let s = ask_with_validation("Name the project and one key feature", {
  "validation_types": ["basic"],
  "min_confidence": 0.6
});
```

### Web search configuration (TOML or env)
Pick and configure a web search engine via `lexon.toml`:
```toml
[web_search]
provider = "duckduckgo"
endpoint = "https://duckduckgo.com/"
query_param = "q"
count_param = "n"
format_param = "format"
format_value = "json"
auth_mode = "none"         # none|header|query
auth_name = "Authorization" # header or query param name
auth_env = "WEB_SEARCH_API_KEY"
```
Env override (quick):
```bash
export LEXON_WEB_SEARCH_ENDPOINT=https://duckduckgo.com/
```
Preset examples (uncomment and set the API key env):
```toml
# Brave Search JSON (header API key)
[web_search]
provider = "brave"
endpoint = "https://api.search.brave.com/res/v1/web/search"
query_param = "q"
count_param = "count"
auth_mode = "header"
auth_name = "X-Subscription-Token"
auth_env = "BRAVE_SEARCH_API_KEY"

# SerpAPI (query API key)
#[web_search]
#provider = "serpapi"
#endpoint = "https://serpapi.com/search.json"
#query_param = "q"
#count_param = "num"
#auth_mode = "query"
#auth_name = "api_key"
#auth_env = "SERPAPI_API_KEY"
```

<!-- CI: trigger rc build -->

### Real providers quickstart
- Export API keys (set only what you use):
```bash
export OPENAI_API_KEY=sk-...
# export ANTHROPIC_API_KEY=...
# export GOOGLE_API_KEY=...
```
- Optional: override base URLs (self‑hosted gateways):
```bash
# export OPENAI_BASE_URL=https://your-openai-proxy/v1
# export ANTHROPIC_BASE_URL=https://your-anthropic-proxy/v1
# export GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```
- In `lexon.toml`, set the default provider (already set to `simulated` by default):
```toml
[system]
default_provider = "openai"   # or "anthropic" / "google" / "simulated"
```
- Ensure the provider has a `default_model` (already defined under `[providers.<name>]`).
- Run a program that calls `ask` (uses the provider/model defaults):
```bash
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
```
- Programmatic tweak inside `.lx` (optional):
```lexon
set_default_model("gpt-4");
let r = ask("Say hello from a real provider");
```

### Key features
- Async/await across LLM, data, and file I/O
- LLM orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`, `ask_safe`
- Anti‑hallucination: confidence scoring and domain validation (schema/PII gates)
- RAG advanced:
  - Precise per‑model tokenization (BPE): `rag.tokenize` / `rag.token_count` / `rag.chunk_tokens`
  - Hybrid search (SQLite + Qdrant) with filters and pagination; auto ensure collection; retries/backoff/throttle; raw Qdrant filters; schema/index helpers
  - Rerank (LLM) and cross‑encoder rerank with batching and env limits
  - Fusion (semantic) with optional citations; `rag.optimize_window` (token‑budget); summarize chunks
- Multioutput system: structured streaming + per‑file progress
- Sessions & memory: start/ask/history, summarize/compress, context retrieve/merge, TTL/GC, durable store
- Providers & routing v2: OpenAI/Anthropic/Gemini + Generic (HuggingFace/Ollama/Custom); budgets/canary/retries/backoff; health/capacity/failure‑rate routing
- Agents & orchestration events: create/run/chain/parallel/cancel; supervisor; tool registry with scopes/quotas; `on_tool_call`/`on_tool_error`
- Telemetry & metrics: rollups JSON, per‑call logs, Prometheus export, starter Grafana dashboard
- Sandbox by default: safe execution and constrained file I/O

### State of AOT, Optimizer, Fuzzing
- AOT/native binaries: not implemented in this RC. Planned approaches:
  - “Embedder” (ship VM + LexIR) as a single binary.
  - Transpile LexIR→Rust/C then compile.
- Optimizer: basic constant-folding and string coercions are handled in the executor; a dedicated IR optimizer is planned post-GA.
- Fuzzing: **available now** via [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz). Three targets live in `fuzz/` (parser, HIR builder, HIR→LexIR). Usage:
  ```bash
  rustup toolchain install nightly
  cargo fuzz list
  cargo +nightly fuzz run parser_cst -- -max_total_time=10
  cargo +nightly fuzz run hir_builder -- -max_total_time=10
  cargo +nightly fuzz run hir_to_lexir -- -max_total_time=10
  ```
  Outputs land under `fuzz/artifacts/<target>/`; corpora are tracked (empty placeholders in git).

### Known limitations (RC)
- Prefer simple, sequential concatenations inside functions, or `strings.join(parts, "")`, to avoid complex expression lowering edge cases.
- Top‑level expressions should be assignments or calls (avoid bare literals).
- Truthiness in `if` is permissive (bools, non‑empty strings/arrays/objects, non‑zero numbers; `"true"/"false"` parsed).

---

## Quickstart

Prereqs: Rust (pinned by `rust-toolchain.toml`), optional VS Code and Python 3.9+.

Build and run a sample:
```bash
cargo build --workspace
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
```

Run all samples and verify goldens:
```bash
cargo make samples-smoke
cargo make samples-snapshot
```

Run the verified demo (uses sandbox flags for file/exec):
```bash
cargo run -q -p lexc-cli -- compile --allow-exec --workspace . --run ../docs/reference/examples/lexon_verified_demo_working.lx
```

CLI sandbox defaults:
- `execute()` is disabled unless `--allow-exec`
- Absolute paths are blocked unless `--workspace PATH`

---

## CLI usage (lexc-cli)

```bash
# Compile and run
cargo run -q -p lexc-cli -- compile --run path/to/file.lx

# Lint for async/await and blocking I/O
cargo run -q -p lexc-cli -- lint path/to/file.lx

# Build with telemetry (optional feature)
cargo build -p lexc --features otel
```

### Cancellation and timeout

```bash
# Set a default timeout (milliseconds) for program execution
cargo run -q -p lexc-cli -- compile --run --timeout 1000 samples/01-async-parallel.lx

# Internally, the async scheduler supports TaskHandle::cancel() and timeouts.
# Long-running tasks exceeding the configured timeout will be cancelled.
```

### Linting (missing_await and blocking I/O)

```bash
# Lint one or more .lx files for async issues
cargo run -q -p lexc-cli -- lint samples/00-hello-lexon.lx

# Notes:
# - The lint command runs rules including MissingAwait and Blocking I/O in async contexts.
# - `compile --check` performs syntax/semantic checks only; it does not run the linter.
```

Configuration via `lexon.toml` controls defaults (provider/model, timeouts, etc.).

---

## Samples

Included scenarios (offline-first with goldens):
- Hello Lexon (`samples/00-hello-lexon.lx`)
- Async parallel orchestration (`samples/01-async-parallel.lx`)
- Research Analyst app (end-to-end, with MCP) (`samples/apps/research_analyst/main.lx`)
- Release notes copilot (multioutput)
- Data quality report (CSV → HTML/Markdown)
- Eval harness (parallel + fallback)
- Static site generator (deterministic multioutput)
- Triage pipeline (CSV → routes)
- Sessions (history/summarize/compress)

Run them:
```bash
cargo make samples-smoke
cargo make samples-snapshot
# Or run the Research Analyst app directly
cargo run -q -p lexc-cli -- compile --run samples/apps/research_analyst/main.lx
```

---

## Development

Toolchain pinned in `rust-toolchain.toml` (1.82.0). CI gates include fmt, clippy `-D warnings`, tests, samples smoke, snapshot diff, public API diff, OTEL smoke, and security checks.

Test:
```bash
cargo test --workspace -- --nocapture
```

Telemetry (optional): build with `--features otel` and set `LEXON_OTEL=1` (point `OTEL_EXPORTER_OTLP_ENDPOINT` to your collector).

---

## VS Code extension

Prebuilt VSIX at `vscode-lexon/lexon-1.0.0.vsix`.
```bash
cd vscode-lexon
code --install-extension lexon-1.0.0.vsix --force
```
Activation: open any `.lx` file; status bar shows “Lexon”, syntax highlighting and snippets load.

---

## Python binding (lexon_py)

Requirements: Python 3.9+, pip, maturin.

Optional: use a virtual environment
```bash
# from v1.0.0-rc.1/
python3 -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip maturin
```

Install (editable):
```bash
make python-dev-install
# or: cd crates/lexon-py && python3 -m pip install maturin && python3 -m maturin develop
```
Smoke:
```bash
python3 - << 'PY'
from lexon_py import compile_lx, PyRuntime
# Minimal no-op program for smoke
lexir = compile_lx('')
PyRuntime().execute_json(lexir)
print('Python integration OK')
PY
```

---

## Security & Sandbox
- Safe by default: `execute()` disabled, absolute paths blocked without `--workspace`
- Telemetry exports only minimal metadata, not prompt payloads or secrets

## License
Apache-2.0 