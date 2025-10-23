![Lexon](assets/lexon-wordmark.svg)

## Lexon v1.0.0-rc.1

A self-contained RC of the Lexon language and tooling: language/runtime, CLI, samples, Tree-sitter grammar, VS Code extension, and Python binding.

### Why Lexon (quick)
- Orchestrate LLM + data as first-class language features (async/await, parallel/merge/fallback/ensemble).
- Built-in reliability: validation (`ask_safe`) with confidence scoring; sessions/RAG Lite.
- Deterministic artifacts: multioutput (text + multiple files/binaries) from a single operation.

### What is Lexon?
Lexon is an LLM-first programming language, built with extensive AI‑assisted development ("vibe coding"). It provides first-class async/await, LLM orchestration (parallelism, merge, fallback, ensemble), functional data processing, multioutput generation, session memory, and native validation (anti‑hallucination).

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
- RC Roadmap: [ROADMAP.md](ROADMAP.md)
- Full Project Roadmap: [../ROADMAP.md](../ROADMAP.md)
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
- Sandbox: `--allow-exec` to enable `execute()`, `--workspace PATH` for absolute paths.

See detailed configuration in `DOCUMENTATION.md`.

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
- LLM orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_with_fallback`, `ask_ensemble`
- Anti-hallucination: `ask_safe` with validation strategies and confidence thresholds
- Functional data ops (CSV/JSON): load/select/filter/take/export; JSON schema inference/validation
- Multioutput system: generate/persist multiple files from a single operation
- Sessions and memory: `session_start/ask/history`, summarization, compression, context-window
- Telemetry (optional): minimal spans for scheduling and key ops via OpenTelemetry
- Sandbox by default: safe execution and constrained file I/O

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