# Lexon v1.0.0-rc.1 — A Full-stack Language for LLM Systems

Every time I tried to build a serious LLM workflow, I ended up juggling scripts, notebooks, glue services, and half a dozen “context” hacks. I wanted a language where async orchestration, validation, RAG, agents, and now structured memory are first-class—not bolted on after the fact. Lexon is my answer: an LLM-first programming language with a deterministic runtime, strong governance, and batteries included. This RC introduces the biggest addition so far, **Structured Project Memory**, but it sits next to peers like MCP, sessions, merge/fallback/ensemble, arbitrage, multioutput, and advanced RAG. Here’s the tour.

---

## 0. Copy-paste quickstart

Run one command, watch async orchestration + merge happen, and be done before the kettle boils:

```bash
cargo run -q -p lexc-cli -- compile --run samples/01-async-parallel.lx
```

```lexon
pub fn main() {
  set_default_model("simulated");

  let [outline, slogans] = ask_parallel([
    ask { user: "Outline the release checklist for Lexon RC.1"; temperature: 0.1; },
    ask { user: "Give me two motivating one-liners for the launch"; temperature: 0.4; }
  ]);

  let merged = ask_merge(outline, slogans, "Return two concise bullet points for kickoff");
  print(merged);
}
```

No shell scripts, no YAML pipelines: `lexc-cli` compiles and runs the IR, and the deterministic runtime simulates the LLMs until you provide real API keys.

### Toolchain assumptions

- Rust toolchain: `rustup override set 1.82.0 && rustup component add clippy rustfmt`.
- Build step: `cargo build --workspace --locked`.
- Provider keys: export `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or declare custom `[providers]` blocks in `lexon.toml`.
- Sandbox flags: `--workspace .` gates file I/O, `--allow-exec` unlocks `execute()`, `LEXON_ALLOW_HTTP=1` enables the HTTP client.

---

## 1. Lexon in three minutes

If you’re tired of stitching together Python notebooks, LangChain pipelines, or orchestration DAGs just to run prompts with context and validation, Lexon is the opposite experience: a real language with LLM-first primitives, governance, and structured memory built in. Instead of spinning up extra frameworks and services for every new feature, Lexon gives you:

- **One surface for orchestration, validation, memory, and RAG**—no need to juggle multiple frameworks or config layers.
- **Less glue code**—async runtime, sessions, RAG, structured memory, and multioutput are all built-in primitives.
- **Minimal setup**—configure `lexon.toml` once; you don’t have to wire half a dozen services just to get started.

That `lexon.toml` (shipped in `v1.0.0-rc.1/lexon.toml`) is where you declare `[system] default_provider`, `[providers.<name>]` blocks, `web_search` presets, sandbox flags, and structured-memory backends. No extra bootstrap layer required.

- **Language & control flow**: modules (`modules/` roots + aliasing), `if/while/for/match`, public/private functions, typed/inferred variables, predictable truthiness, JSON-like structs, error primitives (`Ok/Error/is_ok/unwrap`).
- **Async runtime**: `task.spawn/await`, `join_all`, `join_any`, `select_any`, channels, rate limiter, retry policies, timeouts, cooperative cancellation.
- **Iterators & FP**: `map`, `filter`, `reduce`, `range`, `zip`, `chunk`, `strings.join`, inline expressions for dataset pipelines.
- **Standard library**: encoding, regex, math, advanced time/tz, JSONPath, dataset bridges (CSV/JSON/Parquet/Arrow).
- **Sandbox & governance**: `execute()` disabled by default, absolute paths gated by `--workspace`, budgets per provider, telemetry hooks (Prometheus, OTEL).

Hello world stays simple:

```lexon
pub fn main() {
  set_default_model("simulated");
  let message = ask("Say hello from Lexon");
  print(message);
}
```

Execution goes through `lexc-cli`. Offline simulations are the default; setting `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or custom provider blocks in `lexon.toml` seamlessly switches to real models.

### How it compares

- **Orchestration surface**
  - Lexon: Language primitives (`ask*`, `task.spawn`, `before_action`).
  - LangChain/LlamaIndex: Library APIs layered over notebooks + config.
  - Plain Python/Rust async: Hand-scripted tasks, retries, fan-out.
- **Memory + RAG**
  - Lexon: Vector + structured memory built in, deterministic samples.
  - LangChain/LlamaIndex: Plugins that depend on extra stores/services.
  - Plain host languages: Manual DB + embedding wiring.
- **Governance**
  - Lexon: Sandbox flags, budgets, deterministic runtime, OTEL hooks.
  - LangChain/LlamaIndex: Depends on surrounding app.
  - Plain host languages: Logging/telemetry are DIY.
- **Developer ergonomics**
  - Lexon: One CLI + `lexon.toml`, VS Code extension, tree-sitter.
  - LangChain/LlamaIndex: Mix of Python, YAML, notebooks, env orchestration.
  - Plain host languages: Maximum control but more glue.
- **Parallelism**
  - Lexon: Scheduler understands LLM calls, `ask_parallel`, `ask_merge`.
  - LangChain/LlamaIndex: Tied to whichever executor each project wires.
  - Plain host languages: You own concurrency + cancellation.

---

---

## 2. LLM orchestration suite (beyond any single feature)

The orchestration surface is intentionally broad because real apps need more than a lone `ask()`:

- **ask-family**: `ask`, `ask_parallel`, `ask_merge` (summarize/synthesize), `ask_with_fallback`, `ask_ensemble`, `model_arbitrage`, `model_dialogue`.
- **Validation & quality (anti-hallucination)**: `ask_safe`, `ask_with_validation`, `quality.*` gates for schema/PII/confidence, configurable retries/budgets.
- **Multioutput**: `ask_multioutput` emits primary text + multiple deterministic files (JSON/CSV/Markdown/binary stubs) with helpers `get_multioutput_*`, `save_multioutput_file`.
- **Sessions**: `session_start/ask/history/summarize/compress/extract_key_points`, TTL/GC via `sessions.gc_now`, context window management.
- **RAG stack**: per-model tokenization, `memory_index.ingest_chunks`, hybrid search (SQLite + Qdrant), rerank (LLM + cross-encoder), semantic fusion with citations, `rag.optimize_window`.
- **Data & web**: CSV/JSON/Parquet load/save, dataset ops (`load_csv`, `save_json`, Parquet via Polars/Arrow), `web.search` (DuckDuckGo, Brave, SerpAPI, custom endpoints), HTTP client (opt-in via `LEXON_ALLOW_HTTP`), string/regex helpers for parsing.

Example pipeline:

```lexon
let ds = load_csv("samples/triage/tickets.csv");
let urgent = filter(ds, 'priority == "high"');
save_json(urgent, "output/high_tickets.json");

let brief = ask_safe {
  user: "Summarize the high priority tickets",
  validation: "basic",
  max_attempts: 2
};

print(brief);
```

Dual-role prompt (system + user) with guardrails:

```lexon
let summary = ask {
  system: "You are Lexon's semantic memory layer. Be precise.";
  user: "Summarize the runtime guide in two bullet points.";
  model: "openai:gpt-4o-mini";
  temperature: 0.2;
  max_tokens: 256;
};
```

Hybrid search + fusion + answer:

```lexon
let hits = memory_index.hybrid_search("before_action hook", 5);
let context = rag.fuse_passages(hits, 3);
let answer = ask {
  system: "Use only the provided context.";
  user: strings.join([
    "Context:\n", context, "\n\nQuestion: How do before_action hooks enrich agents?"
  ], "")
};
print(answer);
```

---

## 3. MCP agents, tooling, and observability

Agents aren’t useful if you can’t govern them, cancel them, or see what they’re doing. MCP support comes built-in—you can launch stdio or WebSocket MCP servers directly from Lexon, register tools with quotas, and stream progress/cancelation signals without extra glue:

- **MCP 1.1**: stdio + WebSocket servers, tool registry with quotas, cooperative cancellation (`rpc.cancel`), heartbeats, streaming progress.
- **Agents**: `agent_create/run`, parallel/chained flows, supervisors, budgets, deadlines, telemetry spans, `on_tool_call`/`on_tool_error`.
- **Context hooks**: `before_action use_context` automatically pulls structured-memory bundles before any agent step.
- **Observability & governance**: OTEL export, Prometheus metrics, per-call rollups, CLI lint/fmt/clippy/test/golden gates, provider budgets, deterministic goldens.
- **Configuration**: `lexon.toml` drives providers, web search, sandbox toggles (`LEXON_ALLOW_HTTP`, `LEXON_ALLOW_NET`), memory paths.

Everything funnels through the same governance rails: structured memory, RAG queries, HTTP calls, MCP tools, and multioutput share telemetry, budgets, and deterministic behavior.

Start servers and hook context:

```bash
# stdio server
cargo run -q -p lexc -- --mcp-stdio

# WebSocket server with custom addr
cargo run -q -p lexc -- --mcp-ws --mcp-addr 127.0.0.1:9443
```

```lexon
before_action use_context project="lexon_demo" topic="runtime";

let supervisor = agent_create("runtime_supervisor", """{"model":"openai:gpt-4o-mini","budget_usd":0.15}""");
let report = agent_run(supervisor, "Produce the deployment checklist for RC.1", """{"deadline_ms": 30000}""");
print(report);
```

### Telemetry in practice

Lexon’s OTEL hooks are baked into the runtime. Flip a single env var and every scheduler hop, `ask` call, structured-memory write, or MCP tool execution emits spans:

```bash
LEXON_OTEL=1 OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \
  cargo run -q -p lexc-cli -- compile --run samples/01-async-parallel.lx
```

The bundled OTLP smoke collector (`cargo make otel-smoke`) surfaces spans such as `lexon.scheduler.execute`, `lexon.ask.request`, and `lexon.memory.remember_raw`, each annotated with model, tokens, duration, and budget metadata—proof that this stack is observability-ready.


---

## 4. Structured Project Memory (spotlight)

This RC’s headline feature is a native “second brain” for each Lexon project. It doesn’t replace RAG; it sits above it, curating the knowledge humans actually care about before any retrieval call.

### 4.1 Architecture

1. **Semantic layer** (LLM-assisted or manual) creates canonical `MemoryObject`s with `path`, `kind`, `raw`, `summary_micro/short/long`, `tags`, `metadata`, `relevance`, `pinned`, timestamps, and policies.
2. **Tree/index backends** (pluggable):
   - `basic`: heuristic scoring (relevance, pinning, topic/kind/tag matches).
   - `patricia`: compressed trie for fast path-prefix lookups.
   - `raptor`: RAPTOR-style clustering using tags + recency.
   - `hybrid` (GraphRAG/MemTree): entity-token overlap + clustering.
   - Select via `LEXON_MEMORY_BACKEND=basic|patricia|raptor|hybrid`.

### 4.2 Language primitives

```lexon
let _ = memory_space.create("lexon_demo", """{"reset": true}""");

let obj = remember_raw(
  "lexon_demo",
  "decision",
  read_file("docs/runtime_decision.md"),
  """{"project": "runtime", "path_hint": "lexon/runtime/decisions"}"""
);

let bundle = recall_context(
  "lexon_demo",
  "runtime",
  """{"limit": 3, "include_raw": true, "freeze_clock": "2025-01-01T00:00:00Z"}"""
);

print(obj);
print(bundle);
```

`remember_raw` talks to the configured provider (OpenAI, Anthropic, Google, Ollama, HF, custom) to infer structure, tags, relevance, pinning suggestions, and metadata; `remember_structured` ingests pre-built payloads. Both obey budgets, retries, telemetry, and deterministic testing features such as `freeze_clock`.

### 4.3 Why it matters

- **Curated context before RAG**: agents see pinned/high-relevance guides, configs, and raw snippets; RAG handles the rest.
- **Backend flexibility**: swap heuristics (Patricia trie, RAPTOR clusters, GraphRAG/MemTree hybrid) without touching application code.
- **Policies & governance**: per-project auto-pin, retention, visibility flags, deterministic resets.
- **Multi-provider ready**: uses the same LLM adapter as `ask`, so it works with every supported backend.

---

## 5. Where it fits in the broader stack

- **RAG** — `recall_context` yields `global_summary + sections + raw`; chain with `memory_index.hybrid_search` for large corpora.
- **MCP/agents** — `before_action use_context` auto-injects bundles so pinned guides/configs always enter the prompt.
- **Sessions** — Session transcripts are summarized/stored via `remember_structured`, giving each project a durable storyline.
- **Multioutput** — Curated memories feed `ask_multioutput` to produce multi-file briefings.
- **Observability** — `remember_*` / `recall_*` emit spans, respect budgets, and surface in OTEL/Prometheus dashboards.



Structured memory is a peer to RAG, MCP, sessions, merge/fallback/ensemble, arbitrage, multioutput, and other orchestration features—one more first-class capability, not the only story.

---

## 6. Lexon feature catalog (expanded)

If you just need the checklist, here it is:

- **Core language**: modules, `if/while/for/match`, functions, structs, error handling, string/data utilities.
- **Async runtime**: scheduler, `task.spawn/await`, `join_all`, `select_any`, channels, rate limiter, retries, timeouts.
- **Iterators/FP**: `map`, `filter`, `reduce`, `zip`, `chunk`, `flatten`, `unique`, `find`, `count`.
- **Data & IO**: load/save CSV/JSON/Parquet, dataset bridges, ETL minis, sandboxed `execute`, deterministic goldens.
- **Networking/web**: HTTP client (opt-in), configurable `web.search`, custom endpoints via `lexon.toml`.
- **LLM orchestration**: ask family, `ask_safe`, `ask_with_validation`, multioutput, model arbitrage/dialogue.
- **Validation/quality**: `quality.*`, gates for PII/schema/confidence, `sessions.gc_now`, budgets per provider.
- **Memory systems**: vector memory (`memory_index.*`), structured project memory (`memory_space.*`, `remember_*`, `recall_*`, pin/policy APIs), `before_action use_context`.
- **RAG**: hybrid search, rerank (LLM + cross-encoder), fusion, `rag.optimize_window`, per-model tokenization.
- **Agents & MCP**: stdio/WS servers, tool registry, quotas, cancellation, streaming, agent supervisors.
- **Observability/governance**: OTEL, Prometheus, CLI lint/fmt/clippy/tests/golden gates, budgets/quotas, telemetry redaction.
- **Tooling**: VS Code extension, Tree-sitter grammar, Python binding (`lexon_py`), `cargo-make` tasks, fuzz harnesses.

---

## 7. Stability map (RC vs 1.1)

- **Language + scheduler** — RC.1: ✅ Frozen (modules, async/await, scheduler, error handling). Heading to 1.1: perf profiling + IR tweaks.
- **ask/ask_parallel/ask_merge + validation** — RC.1: ✅ APIs frozen. Next: richer ensembles + arbitration knobs.
- **Structured Project Memory** — RC.1: ✅ GA-level primitives + pluggable backends. Next: memory inspector CLI, backend hints.
- **Vector RAG + multioutput** — RC.1: 🔄 Stable core, polishing APIs. Next: RAG Lite presets + better multioutput helpers.
- **MCP/agents** — RC.1: ✅ CLI switches, quotas, cancellation in place. Next: sample supervisors + tool packs.
- **Observability** — RC.1: ✅ OTEL/Prometheus spans wired. Next: dashboard templates + provider budget reports.
- **Iterators/data transforms** — RC.1: ✅ map/filter/reduce/ETL minis. Next: windowed ops + join helpers.

**GA exit criteria:** p95 runtime <1.2× baseline (`samples/apps/research_analyst`), <1% token-budget regression on `samples/memory/structured_semantic`, and OTEL spans present for every tool/ask call in CI smoke tests.

---

---

## 8. Samples & how to actually use Lexon

These are the programs I run to prove Lexon still “gets it right” end-to-end:

- `samples/00-hello-lexon.lx`: syntax and CLI workflow.
- `samples/01-async-parallel.lx`: scheduler, `task.spawn`, `select_any`.
- [`samples/apps/research_analyst/main.lx`](../samples/apps/research_analyst/main.lx): full MCP + web search + RAG + sessions demo.
- [`samples/memory/structured_semantic.lx`](../samples/memory/structured_semantic.lx): deterministic structured-memory smoke test + [`golden/memory/structured_semantic.txt`](../golden/memory/structured_semantic.txt).
- Plus ETL mini, notes organizer, triage pipeline, release-notes copilot, static site generator, eval harness.

Commands:

```bash
cargo build --workspace
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
cargo make samples-smoke
cargo make samples-snapshot
```

Switch structured memory backend:

```bash
LEXON_MEMORY_BACKEND=hybrid cargo run --bin lexc -- samples/memory/structured_semantic.lx
```

Use real providers by exporting API keys and editing `lexon.toml` (`default_provider`, per-provider defaults). Without keys, everything is deterministic and runs offline.

---

## 9. Roadmap snapshot

- More structured-memory backends (TemporalTree decay, external GraphRAG adapters).
- Per-call backend overrides (`{"backend": "patricia"}` hints).
- Cross-project memory links and role-based visibility.
- Memory inspector tooling (`lexc memory browse`) and better introspection.
- RAG Lite bundles plus richer multioutput helpers (e.g., deterministic static-site generator sample).
- Bigger MCP demos that mix structured memory, agents, RAG, multioutput.
- Plus the broader items already listed in [`ROADMAP.md`](../ROADMAP.md): Qdrant presets, telemetry dashboards, provider expansion, IR optimizations, richer stdlib/networking, sockets, CI hardening, and DX tooling.

We gate GA on three metrics: median `cargo make samples-smoke`, p95 runtime for `samples/apps/research_analyst`, and structured-memory recall accuracy on the golden sample. When those stay inside budget for two consecutive runs, RC graduates to 1.0.

---

## 10. Getting started

1. Clone `github.com/lexon-lang/lexon` (RC pinned to Rust 1.82).
2. `cargo build --workspace`.
3. Run samples in simulated mode or export provider keys for real runs.
4. Play with `samples/memory/structured_semantic.lx` using different backends (`basic`, `patricia`, `raptor`, `hybrid`).
5. Read `README.md`, `DOCUMENTATION.md`, `communication/lexon_memory_features.md` for the deep details.

Lexon already drives MCP agents, ETL pipelines, copilots, RAG flows, and now high-signal project memory—all from one language. If you try it, let me know what you build; I’m still the only person maintaining this thing, and your feedback shapes the backlog.

*Repository*: [github.com/lexon-lang/lexon](https://github.com/lexon-lang/lexon)  
*Docs*: `README.md`, `DOCUMENTATION.md`, `communication/lexon_memory_features.md`  
*Contact*: open an issue/PR or share your demo referencing Lexon + Structured Project Memory.

---

## 11. What to try next

- **Ship a memory-first agent**: run `samples/memory/structured_semantic.lx`, pin the insights you care about, then wire `before_action use_context` into your go-to agent template so every step sees the curated context.
- **Launch MCP in under a minute**: `lexc --mcp-ws --mcp-addr 127.0.0.1:9443` and hand that endpoint to Cursor, Claude Desktop, or your own supervisor—Lexon handles streaming, quotas, and cancellation.
- **Blend RAG + structured memory**: call `recall_context` first, reuse its summaries as a system prompt, then run `memory_index.hybrid_search` for long-tail retrieval. Less hallucination, more grounded answers.
- **Automate governance checks**: enable OTEL/Prometheus, run `cargo make samples-smoke`, and watch structured memory, agents, and MCP emit spans with budgets so you can spot regressions early.
- **Prototype in Python**: install `lexon_py`, compile `.lx` snippets inline, and keep your existing notebooks while you migrate orchestrations into Lexon.
- **Star + grab an issue**: [github.com/lexon-lang/lexon](https://github.com/lexon-lang/lexon) and the [`good first issue` queue](https://github.com/lexon-lang/lexon/issues?q=is%3Aopen+label%3A%22good+first+issue%22) are the fastest way to contribute feedback.
- **Run a full workflow**: `cargo run -q -p lexc-cli -- compile --run samples/apps/research_analyst/main.lx` hits MCP, web search, RAG, sessions, and structured memory end-to-end—perfect demo fodder.

