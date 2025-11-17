# Introducing Lexon v1.0.0‑rc.1 — a language for reliable LLM software

Lexon is a programming language and runtime for building AI-first workloads with the same rigor we expect from traditional backends. Instead of gluing prompts together, you write `.lx` programs that combine:

- **Orchestration primitives** (`ask`, `ask_parallel`, `ask_merge`, `ask_ensemble`, `ask_with_fallback`) backed by an async runtime with cancellation, retries, and deterministic seeds.
- **Built-in safety nets** (`ask_with_validation`, `ask_safe`, schema/PII gates, confidence scoring) so “what did the model say?” is auditable and testable.
- **Sessions & RAG helpers** (`session_*`, `memory_index.*`, `rag.tokenize/rerank/fuse`, `rag.summarize_chunks`) for keeping context fresh without bolting on extra services.
- **Multi-output generation** (`ask_multioutput`, `save_multioutput_file`, `save_binary_file_stream`) to produce Markdown, CSV, JSON, and line-delimited data from a single request.
- **MCP stdio/WS servers** with JSON-RPC, quotas, heartbeats, progress, and optional TLS/Bearer/JWT/mTLS, so Lexon programs become addressable coprocessors.
- **Batteries-included stdlib**: `http.get`, `http.request`, `web.search`, `encoding/strings/math/regex/time/number/crypto/json`, plus a Python binding (`crates/lexon-py/`) and VS Code extension.

This RC ships the language, compiler (`lexc/`), CLI (`lexc-cli/`), tree-sitter grammar, a curated set of samples + goldens, CI (fmt/clippy/tests/smoke/golden/OTEL), and docs (`README`, `DOCUMENTATION`, `RELEASE_NOTES`). It’s intentionally self-contained—drop the repo into a CI agent, run `cargo make samples-smoke`, and you get deterministic coverage.

---

## Case study: the Research Analyst app

To see Lexon in action, run `samples/apps/research_analyst/main.lx`. It synthesizes an executive brief by mixing RAG, ensembles, validation, a web search backend, and multi-artifact exports. Here’s how it works:

### 1. Bootstrap models, session, and RAG context
```lexon
let model = get_env_or("LEXON_MODEL", "openai:gpt-4o-mini");
set_default_model(model);
let sid = session_start(model, "research_demo");

let _ = memory_index__set_metadata("samples/00-hello-lexon.lx", "{ \"category\": \"hello\" }");
let _ing: int = memory_index.ingest(["samples/00-hello-lexon.lx"]);
let _ctx = auto_rag_context();
let _ = memory_index.vector_search("hello lexon", 2);
```
The app respects env overrides (`LEXON_MODEL`, `OPENAI_BASE_URL`, API keys, etc.). Sessions are first-class: every `ask` can append context to `sid`, and `session_summarize`/`session_compress` produce “what happened” digests automatically.

### 2. Orchestration & validation pipeline

1. `ask_ensemble` composes two prompts (“summarize the repo” vs “what are the main capabilities”) and merges them with `MajorityVote`. The output becomes the **CONSENSUS** block.
2. `model_arbitrage` runs a quick, structured debate between `openai:gpt-4o-mini` and `google:gemini-1.5-pro`, using `openai:gpt-4o` as the decider to produce the **ARBITRAGE** summary.
3. `ask_with_validation("Name the project and one key feature", { "min_confidence": 0.6, "validation_types": ["basic"], "max_attempts": 2 })` yields the **SAFE** facts—if confidence stays below 0.6, it retries with stricter prompts and ultimately returns a failure string the caller can inspect.
4. A second `ask_with_validation` produces three bullet‑proof **NEXT ACTIONS** (owner + due date). If the model deviates from the required format, Lexon detects it (`regex.match` + `validate_response`) and falls back to a seeded template so you always get three actionable lines.

### 3. Web search & deterministic citations

Lexon ships a `web.search` builtin tied to `lexon.toml`:
```toml
[web_search]
provider    = "serpapi"
endpoint    = "https://serpapi.com/search.json"
query_param = "q"
count_param = "num"
auth_mode   = "query"
auth_name   = "api_key"
auth_env    = "SERPAPI_API_KEY"
extra_params = { engine = "google" }
```

In the program:
```lexon
let topic = get_env_or("RA_QUERY", "micro agent communication protocol");
let n_search = parse_env_int("RA_CITATIONS_N", 10);
let web = web__search(topic, n_search);
let citations_md = build_citations_dynamic(web, n_search, 100);
summary_md = strings.join([summary_md, citations_md], "");
```
`build_citations_dynamic` walks `/Results/0..n`, picks `title`/`Text` fallbacks, normalizes URLs (prepends `https://` for `www.` results), truncates to 100 chars, checks for HTTP-only links if `RA_CITATIONS_HTTP_ONLY=1`, and always appends `- See WEB section for details` so readers know how to trace the sources. `RA_CITATIONS_N`, `RA_CITATIONS_MAXLEN`, and `RA_CITATIONS_HTTP_ONLY` are tunable via env.

### 4. Multi-output packaging & metadata

The program uses `ask_multioutput` to gather both a short narrative and a CSV (with key findings). Everything is persisted with deterministic file names, dual writes (repo `output/` + root `../output/`), and metadata extras:
```lexon
let mo = ask_multioutput("Create a short report and a CSV with 2 rows", ["report.md", "appendix.csv"]);
save_binary_file_stream(load_binary_file("output/ra_report.md", "report.md"), "../output/ra_report.md");
save_binary_file_stream(load_binary_file("output/ra_appendix.csv", "appendix.csv"), "../output/ra_appendix.csv");

let meta = strings.join([
    "Generated: ", time.now_iso8601(), "\n",
    "Model: ", strings.upper(model), "\n",
    "Summary SHA256: ", crypto.sha256(summary_md), "\n",
    "Title: Research Analyst — Key Conclusions\n"
], "");
save_file(meta, "output/ra_meta.txt");
save_file(meta, "../output/ra_meta.txt");
save_file(crypto.sha256(summary_md), "output/ra_snapshot.txt");
```

### 5. Running it

```bash
# simulate everything
cargo run -q -p lexc-cli -- compile --run samples/apps/research_analyst/main.lx

# real providers + search
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export LEXON_WEB_SEARCH_ENDPOINT=https://serpapi.com/search.json
export SERPAPI_API_KEY=sa-...
export LEXON_ALLOW_HTTP=1
export RA_QUERY="AI agent frameworks comparison"
export RA_CITATIONS_N=10
cargo run -q -p lexc-cli -- compile --run samples/apps/research_analyst/main.lx
```
Outputs (Markdown, HTML, CSV, JSON, snapshots) land in `v1.0.0-rc.1/output/` **and** `../output/`, so you can diff runs or ship them to downstream dashboards.

---

## What else is in the RC?

- **Orchestration & safety**: `ask`, `ask_parallel`, `ask_merge`, `ask_ensemble`, `ask_with_fallback`, `ask_with_validation`, `ask_safe`, `session_*`, `memory_index.*`, `rag.*`, `ask_multioutput`.
- **MCP**: `--mcp-stdio` / `--mcp-ws` (TLS optional) with `list_tools`, `tool_info`, `tool_call`, `set_quota`, heartbeats, progress, Bearer/JWT/mTLS, `LEXON_MCP_RATE_PER_MIN`, `LEXON_MCP_ACL_FILE`, `LEXON_MCP_PERMISSION_PROFILE`.
- **Stdlib**: `encoding.*`, `strings.*`, `math.*`, `regex.*`, `time.now_iso8601`, `number.format`, `crypto.sha256`, `json.*`, `http.get`, `http.get_json`, `http.request`, `web.search`, `save_file`, `save_json`, `load_csv`, `save_csv`, `data_filter/select/take`, `map/filter/reduce/zip/flatten/unique/sort/reverse/chunk/find/count`.
- **Tooling**: `cargo make samples-smoke`, `cargo make samples-snapshot`, `cargo +nightly fuzz run parser_cst`, `cargo make otel-smoke`, `python -m maturin develop` for the binding, VS Code extension (`./vscode-lexon`).
- **Docs & release notes**: `README` (quickstart), `DOCUMENTATION` (full API + config + fuzz guide), `RELEASE_NOTES` (RC scope and GA checklist), this blog post.
- **Fuzzing**: `fuzz/` crate with `parser_cst`, `hir_builder`, `hir_to_lexir`. Run `cargo +nightly fuzz run ... -- -max_total_time=60` to stress the compiler; corpora/artifacts folders include `.gitkeep` + `.gitignore` to stay clean in git.

---

## Roadmap toward GA

- **AOT/Embedder**: produce single binaries that package LexIR + runtime (for air-gapped or low-latency deployments).
- **Optimizer**: richer IR passes (const-fold, DCE, inline, string flattening) to remove current “simple concatenation” constraints.
- **Optional sockets & data bridges**: `tcp.connect/send/recv` (protected by `LEXON_ALLOW_NET` and quotas) and one-liner JSON↔Dataset transforms (e.g., `json_to_dataset`).
- **Telemetry**: full OTLP spans for every `ask`, `session`, `RAG`, `MCP tool_call`, plus better CLI UX for tracing.
- **Provider ergonomics**: built-in key validators, `lexc-cli providers test`, curated `lexon.toml` presets, sample prompt packs for verticals.
- **Fuzz + coverage in CI**: nightly `cargo +nightly fuzz run ...` gating, targeted corpora for parser/HIR, plus `cargo llvm-cov` to track IR/exec coverage.

---

## Get involved

1. **Build & test**
   ```bash
   cargo build --workspace
   cargo test
   cargo make samples-smoke
   cargo +nightly fuzz run parser_cst -- -max_total_time=30
   ```
2. **Run the samples** (`samples/apps`, `samples/rag`, `samples/mcp`, `samples/validation`, …) and inspect `golden/*` to understand the expected outputs.
3. **Extend the language**: add a new stdlib helper, a pipeline, or a fuzz corpus entry—everything is regular Rust + tree-sitter + cargo-fuzz.
4. **File issues/PRs**: provider configs, telemetry wishes, CLI ergonomics, doc gaps, or stubs you want to help implement (AOT, optimizer, sockets, JSON↔Dataset).

Lexon is Apache‑2.0, RC-ready, and already running inside internal prototypes. If you’re building AI systems that have to be repeatable, auditable, and observable, we’d love your feedback—and your contributions.
