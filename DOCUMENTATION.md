## Lexon Documentation

This document reflects what is implemented. Examples use only supported syntax and functions. Lexon is an LLM‑first programming language, built with extensive AI‑assisted development ("vibe coding").

### 1. Syntax (implemented)
- Files: `.lx`. Statements end with `;`. Blocks use `{ ... }`.
- Comments: `// line` and `/* block */`.
- Identifiers: variables/functions `snake_case`; types/traits `CamelCase`.
- Literals: integers (`42`), floats (`3.14`), booleans (`true/false`), strings (`"text"`).
- Collections: arrays (`[1, 2]`), JSON objects (`{ key: "value" }`).
- Control flow: `if/else`, `while`, `for in`, `match`.
- Functions: `pub fn name(args) -> type { ... }` and `fn name(...) -> type { ... }`.
- Errors: `Ok(value)`, `Error(msg)`, `is_ok(x)`, `is_error(x)`, `unwrap(x)`.
- Async: `async fn` and `await` on supported expressions (see 8, 16).

Minimal example:
```lexon
pub fn main() {
  let x = 2;
  if x > 1 { print("ok"); }
}
```

Supported statements and expressions (non‑exhaustive):
- Declarations: `let`, `const`, `pub fn`, `fn`.
- Expressions: literals, arrays, objects, identifiers, function calls, method calls, `await`.
- Control: `if/else`, `while`, `for in`, `match`.
- Type query: `typeof(expr)`.
- Results: `Ok(x)`, `Error(msg)`, `is_ok(x)`, `is_error(x)`, `unwrap(x)`.

#### 1.1 Variables and typing
- Declaration:
```lexon
let count = 0;           // inferred int
let name: string = "A"; // explicit type
const PI = 3.14159;      // constant
```
- Mutation and assignment:
```lexon
count = count + 1;
```

#### 1.2 Type inference
- Primitives and arithmetic are inferred: `int`, `float`, `bool`, `string`.
- Collections infer homogenous types when possible; mixed values are represented as JSON.
- `typeof(x)` returns a runtime string: `"int"`, `"float"`, `"bool"`, `"string"`, `"array"`, `"object"`.

#### 1.3 Scope and visibility
- Function scope: variables declared inside a function are not visible outside.
- Block scope: variables declared inside `{ ... }` are confined to that block.
- Public vs private functions:
```lexon
pub fn public_api() { }
fn internal_helper() { }
```

#### 1.4 Modules and imports
- Project modules live under `modules/` by default (see `lexon.toml`).
- Basic module pattern:
```lexon
// modules/core/llm.lx
pub fn util() { }

// in your program
// import declarations are supported by the grammar; keep module paths under modules_dir
util();
```
Search and resolution:
- Imports resolve by trying `<path>.lx`, then `<path>/index.lx`, then `<path>/mod.lx` in the current directory and in any extra roots from `LEXON_MODULE_ROOTS` (colon‑separated).
- On failure, the CLI will print all candidate paths it tried and a hint to set `LEXON_MODULE_ROOTS`.
- Aliases and selective imports are supported. Module aliasing (`import lib.math as math;`) and item aliasing (`import lib.math::{double as dbl};`) are resolved during lowering; calls are flattened (e.g., `math::double(3)` → `lib__math__double(3)`). Keep module roots under `modules/` (or set `LEXON_MODULE_ROOTS`) and avoid deep alias chains.

#### 1.5 Operators
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical: `&&`, `||`
- Indexing and dot access on arrays/objects when applicable.

### 8. LLM primitives (implemented)

Block `ask`:
```lexon
let r = ask {
  system: "You are helpful";
  user: "Explain Rust in one sentence";
  model: "simulated";
  temperature: 0.2;
  max_tokens: 128;
};
```

Function `ask`:
```lexon
let r = ask("Explain Rust in one sentence");
let r2 = ask("Explain Rust in one sentence", "simulated");
```

Block `ask_safe` (validation keys supported):
```lexon
let s = ask_safe {
  user: "Capital of France?";
  validation: "basic";                // basic|ensemble|fact_check|comprehensive
  confidence_threshold: 0.8;
  max_attempts: 2;
  cross_reference_models: ["simulated"]; // identifiers/strings accepted
  use_fact_checking: true;
};
```

Function `ask_safe` with named parameters:
```lexon
let s2 = ask_safe("Capital of France?", validation: "basic", max_attempts: 2);
```

Helper `ask_with_validation` (string result):
```lexon
let validated = ask_with_validation("Name the project and one key feature", {
  "validation_types": ["basic"],
  "min_confidence": 0.6
});
```

Also available: `ask_parallel(...)`, `ask_merge(responses, "synthesize"|"summarize")`, `ask_with_fallback(models, prompt)`, `ask_ensemble(prompts, strategy, model)`.

Await support: `await ask { ... }`, `await ask_safe { ... }`, and the orchestration calls above.

Parameters supported in block forms (per grammar):
- `ask { system|user|schema|model|temperature|max_tokens }`
- `ask_safe { system|user|schema|model|temperature|max_tokens|validation|confidence_threshold|max_attempts|cross_reference_models|use_fact_checking }`

### 16. Multioutput

Types:
- `BinaryFile { name: string, mime_type: string, size: int, content: bytes }`
- `MultiOutput { primary_text: string, binary_files: [BinaryFile], metadata: {string:string} }`

Create multioutput from a single LLM call:
```lexon
let mo = ask_multioutput(
  "From these tickets, produce triage.csv and playbook.md",
  ["triage.csv", "playbook.md"]
);

let text = get_multioutput_text(mo);
let files = get_multioutput_files(mo);
let meta = get_multioutput_metadata(mo);
save_multioutput_file(mo, 0, "output/triage.csv");
```

Runtime behavior:
- `ask_multioutput(prompt: string, output_files: [string]) -> MultiOutput`.
- Primary text is produced via the LLM adapter.
- Binary files are generated deterministically by extension:
  - `.json`: pretty JSON stub including prompt and filename.
  - `.csv`: sample CSV with header and a couple rows.
  - `.txt`/other: simple text payload.
- Helpers:
  - `get_multioutput_text(MultiOutput) -> string`
  - `get_multioutput_files(MultiOutput) -> array`
  - `get_multioutput_metadata(MultiOutput) -> json`
  - `save_multioutput_file(MultiOutput, index: int, path: string) -> void`
  - `load_binary_file(path: string, name?) -> BinaryFile`
  - `save_binary_file(BinaryFile, path: string) -> void`

Notes:
- Arguments are validated (types/arity). Errors are surfaced as runtime errors.
- All file I/O honors CLI sandbox flags.
- Real binaries (opt‑in): set environment variable `LEXON_REAL_BINARIES=1` to use the LLM response as payload for text-like outputs (JSON wraps `{ text: ... }`, CSV emits a small summary row). Default is deterministic stubs.
- Directory creation: `save_multioutput_file(...)` will auto-create parent directories of the provided path.
- Pseudo‑streaming (opt‑in): set `LEXON_STREAM_TEXT=1` to print primary_text lines as they are available (stdout only; no incremental API yet).
- Not GA yet: incremental streaming API; non text‑like binary generation; per‑file advanced metadata; progress callbacks; strict per‑file size limits.

Error cases:
- Wrong arity or types for `ask_multioutput` → `ArgumentError`.
- Non‑array `output_files` or non‑string elements → `ArgumentError`.
- `save_multioutput_file` wrong index/path types → `ArgumentError`.
- I/O failures when saving/loading binary files → `DataError`.

Implemented helpers summary:
- `get_multioutput_text(mo) -> string`
- `get_multioutput_files(mo) -> array`
- `get_multioutput_metadata(mo) -> json`
- `save_multioutput_file(mo, index, path) -> void`
- `load_binary_file(path, name?) -> BinaryFile`
- `save_binary_file(BinaryFile, path) -> void`

Full API index (implemented in this RC):
- Core: `typeof`, `Ok`, `Error`, `is_ok`, `is_error`, `unwrap`
- LLM orchestration: `ask`, `ask_parallel`, `ask_merge`, `ask_ensemble`, `ask_with_fallback`, `ask_safe`
- Sessions: `session_start`, `session_ask`, `session_history`, `session_summarize`, `session_compress`, `extract_key_points`, `context_window_manage`
- RAG: `memory_index.ingest`, `memory_index.vector_search`, `auto_rag_context`
- Iterators/FP: `enumerate`, `range`, `map`, `filter`, `reduce`, `zip`, `flatten`, `unique`, `sort`, `reverse`, `chunk`, `find`, `count`
- I/O: `read_file`, `write_file`, `save_file`, `load_file`, `execute`, `load_csv`, `save_csv`, `load_json`, `save_json`
- Config: `set_default_model`, `get_provider_default`
- Multioutput: `ask_multioutput`, `get_multioutput_text`, `get_multioutput_files`, `get_multioutput_metadata`, `save_multioutput_file`, `load_binary_file`, `save_binary_file`

---

### 22. Advanced orchestration (debate & dialogue)

- `model_arbitrage(topic: string, models: string | [string], decider?: string, rounds?: int) -> string`
  - Runs a lightweight multi‑model debate and returns a consensus string.
  ```lexon
  let consensus = model_arbitrage(
    "Should we invest in data quality this sprint?",
    "gpt-4,claude-3-5-sonnet-20241022",
    "gpt-4",
    1
  );
  ```

- `model_dialogue(participants: string, topic: string, rounds?: int) -> string`
  - Structured dialogue among named participants producing a transcript.
  ```lexon
  let transcript = model_dialogue(
    "AI Assistant 1, AI Assistant 2",
    "Tradeoffs of validation vs recall",
    2
  );
  ```

---

### 23. CLI and tooling (implemented)

Commands in this RC:
- `compile` — compile `.lx` and optionally run.
- `bench` — run compile/exec benchmarks with warmup/iterations.
- `lint` — lint `.lx` sources for async issues.

Examples:
```bash
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx
cargo run -q -p lexc-cli -- lint -v samples/01-async-parallel.lx
cargo run -q -p lexc-cli -- bench --file samples/00-hello-lexon.lx --iterations 5
```

Sandbox flags (enforced by runtime):
- `--allow-exec` to enable `execute()`.
- `--workspace PATH` to allow absolute paths under PATH.

VS Code extension:
- Prebuilt VSIX in `vscode-lexon/`. Install with `code --install-extension lexon-1.0.0.vsix`.
- Features: syntax highlighting, snippets for `async fn`/`await`, basic completions.

Python binding (preview):
- Crate at `crates/lexon-py/`. Build with `maturin` and import as `lexon_py`.
- Current scope: compile `.lx` to LexIR and run simple programs. Not all opcodes are executed yet.

### 24. Linter rules (implemented)

Run:
```bash
cargo run -q -p lexc-cli -- lint path/to/file.lx
```
Rules:
- `MissingAwait` — async calls not awaited.
- `BlockingIoInAsync` — blocking file/network in async context.
- `AsyncFunctionNotAwaited`, `SyncCallInAsyncContext` — additional async misuse.

### 25. Telemetry (optional)

Build and enable:
```bash
cargo build -p lexc --features otel
export LEXON_OTEL=1
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```
Emitted spans: key scheduler operations and LLM calls.

### 26. Configuration (`lexon.toml`)

Defaults are loaded from `lexon.toml` in this RC. Common keys:
- `[system]` (mode/runtime): `default_provider`, `fallback_provider`, `fallback_model`, `cache_enabled`, `retry_attempts`, `timeout_default`
- `[providers.<name>]` blocks: `base_url`, `headers`, `endpoints`, `default_model`, `models` (with `max_tokens`, `cost_per_1k`)
Programmatic helpers:
Environment variables (overrides):
- Base URLs: `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `GOOGLE_BASE_URL`
- API keys used for auth headers: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`

Simulated vs real:
- `default_provider = "simulated"` uses the simulated client (safe for offline demos)
- To use real providers, set `default_provider` to the provider, ensure its `default_model` in `lexon.toml`, and export the API key env vars

Real providers quickstart (detailed):
```bash
# 1) Export API keys (set only what you need)
export OPENAI_API_KEY=sk-...
# export ANTHROPIC_API_KEY=...
# export GOOGLE_API_KEY=...

# 2) Optional: override base URLs (proxies or gateways)
# export OPENAI_BASE_URL=https://your-openai-proxy/v1
# export ANTHROPIC_BASE_URL=https://your-anthropic-proxy/v1
# export GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta

# 3) Edit lexon.toml
cat <<'TOML' >> lexon.toml
[system]
default_provider = "openai"

[providers.openai]
default_model = "gpt-4"
TOML

# 4) Run a sample
cargo run -q -p lexc-cli -- compile --run samples/00-hello-lexon.lx

# 5) Optional in-code override
# set_default_model("gpt-4");
```
- `set_default_model(model_name: string) -> bool`
- `get_provider_default(provider: string) -> string`

Real providers: usage examples
```lexon
// OpenAI (requires OPENAI_API_KEY)
set_default_model("gpt-4");
let r1 = ask("Say hello from OpenAI");
print(r1);

// Anthropic (requires ANTHROPIC_API_KEY)
let r2 = ask("Say hello from Anthropic", "claude-3-5-sonnet-20241022");
print(r2);

// Google Gemini (requires GOOGLE_API_KEY)
let r3 = ask("Say hello from Gemini", "gemini-1.5-pro");
print(r3);
```

#### 26.1 Web search configuration
Lexon can perform web searches via `web.search`. Configure it in `lexon.toml` or via environment variables:
```toml
[web_search]
provider = "duckduckgo"
endpoint = "https://duckduckgo.com/"
query_param = "q"
count_param = "n"
format_param = "format"
format_value = "json"
auth_mode = "none"         # none|header|query
auth_name = "Authorization" # header or param name
auth_env = "WEB_SEARCH_API_KEY"
```
Quick override via env:
```bash
export LEXON_WEB_SEARCH_ENDPOINT=https://duckduckgo.com/
```
Example presets with API key:
```toml
# Brave Search (header key)
[web_search]
provider = "brave"
endpoint = "https://api.search.brave.com/res/v1/web/search"
query_param = "q"
count_param = "count"
auth_mode = "header"
auth_name = "X-Subscription-Token"
auth_env = "BRAVE_SEARCH_API_KEY"

# SerpAPI (query key)
#[web_search]
#provider = "serpapi"
#endpoint = "https://serpapi.com/search.json"
#query_param = "q"
#count_param = "num"
#auth_mode = "query"
#auth_name = "api_key"
#auth_env = "SERPAPI_API_KEY"
```
The runtime signs the request according to `auth_mode` and appends any `extra_params` defined in TOML.

### 27. Sessions API details

Signatures:
- `session_start(provider?: string, name?: string) -> string`
- `session_ask(session_id: string, prompt: string) -> string`
- `session_history(session_id: string) -> string`
- `session_summarize(session_id: string, options: { max_len?: int }) -> string`
- `session_compress(session_id: string, options: { ratio?: float }) -> string`
- `extract_key_points(session_id: string, options: { top_k?: int }) -> [string]`
- `context_window_manage(session_id: string, options: { target_tokens?: int }) -> bool`

Example:
```lexon
let sid = session_start("simulated", "onboarding");
let r = session_ask(sid, "Hello");
let s = session_summarize(sid, { max_len: 400 });
```

### 28. RAG / Vector memory details

Signatures:
- `memory_index.ingest(paths: string | [string]) -> int`
- `memory_index.vector_search(query: string, k: int) -> array`
- `auto_rag_context() -> string`

Backends:
- Default local backend uses SQLite for metadata/embeddings persistence with in-process cosine similarity.
- You can select an external vector store via environment: `LEXON_VECTOR_BACKEND=sqlite_local|qdrant`.
  - Qdrant settings: `LEXON_QDRANT_URL`, `LEXON_QDRANT_COLLECTION`, `LEXON_QDRANT_API_KEY`.
  - When Qdrant is selected, ingest (including `ingest_chunks`) and `vector_search` use Qdrant HTTP API.

Chunked ingest:
- `memory_index.ingest_chunks(paths: string | [string], options_json?) -> int`
  - Options: `{ by: "tokens"|"chars"|"paragraphs", size: int, overlap: int }` (defaults: tokens/200/40)

Notes:
- Ingest accepts CSV/JSON/NDJSON/text; builds a lightweight vector index.
- `vector_search` returns top‑k matches with simple cosine similarity.

### Structured semantic memory

Signatures:
- `memory_space.create(name: string, metadata_json?) -> summary_json`
  - Metadata doubles as options; pass `{"reset": true}` to drop every object in that space.
- `memory_space.list() -> [summary_json]`
- `remember_structured(space: string, payload_json: string | json, options_json?) -> memory_json`
- `remember_raw(space: string, kind: string, raw_text: string, options_json?) -> memory_json`
- `pin_memory(space: string, path_or_id: string)`
- `unpin_memory(space: string, path_or_id: string)`
- `set_memory_policy(space: string, policy_json)`
- `recall_context(space: string, topic: string, options_json?) -> bundle_json`
- `recall_kind(space: string, kind: string, options_json?) -> [memory_json]`
- Backends: select with `LEXON_MEMORY_BACKEND=basic|patricia|raptor|hybrid` (default `basic`). `patricia` favors path-prefix tries; `raptor` applies lightweight clustering; `hybrid` mixes GraphRAG-style entity overlap with clustering.

Memory object schema:

```
{
  "path": "project/module/topic",
  "kind": "guide|config|decision|log|custom",
  "raw": "... original text ...",
  "summary_micro": "1-2 sentences",
  "summary_short": "short paragraph / bullets",
  "summary_long": "long-form abstract",
  "tags": ["lowercase", "slugs"],
  "metadata": {"project": "...", "importance": "high", "space": "..."},
  "relevance": "high|medium|low",
  "pinned": true|false,
  "created_at": "RFC3339",
  "updated_at": "RFC3339",
  "id": "mem_* (optional override)"
}
```

`remember_raw` calls the active LLM provider (configure `LEXON_MEMORY_SEMANTIC_MODEL` or set `options.model`) to fill this schema based on hints (`path_hint`, `project`, `tags`, `metadata`, `auto_pin`, etc.). `remember_structured` accepts a pre-built payload and auto-completes missing summaries from the `raw` text.

Recall options (`options_json`) support:
- `limit` (default 5), `raw_limit` (default 2), `include_raw`, `include_metadata`
- `prefer_kinds`, `prefer_tags`
- `require_high_relevance` (only return relevance = high)
- `freeze_clock` (RFC3339 string) to override the `generated_at` timestamp for deterministic tests
- `backend` (optional) to hint the service when per-call overrides ship in future releases (for now use `LEXON_MEMORY_BACKEND` env).

Bundled context responses include `global_summary`, individual sections, and optional raw snippets ordered by relevance (pinned + high relevance first).

### 29. Data operations

Signatures:
- `load_csv(path) -> dataset`, `save_csv(dataset, path) -> bool`
- `load_json(path) -> dataset`, `save_json(object, path) -> bool`
- `data_load(path) -> dataset`, `data_filter(dataset, expr) -> dataset`
- `data_select(dataset, fields) -> dataset`, `data_take(dataset, n) -> dataset`

Example:
```lexon
let ds = data_load("samples/triage/tickets.csv");
let f = data_filter(ds, "priority == 'high'");
data_export(f, "output/high.json");
```

### 30. Iterators and FP utilities

Signatures:
- `enumerate(array) -> [[index,value]]`
- `range(end)`, `range(start,end)`, `range(start,end,step)`
- `map(array, fn(x){...})`, `filter(array, fn(x){...})`, `reduce(array, init, fn(a,x){...})`
- `zip(a,b)`, `flatten(nested)`, `unique(array)`, `sort(array, order?)`, `reverse(array)`, `chunk(array, size)`, `find(array, fn)`, `count(array, fn?)`

Example:
```lexon
let arr = [1,2,3];
// String predicate/transform expressions are supported
let doubled = map(arr, 'x * 2');
let evens = filter(arr, 'x % 2 == 0');
let total = reduce(arr, 0, 'acc + x');
```

Notes:
- Predicates and callbacks use string expressions (e.g., `'x % 2 == 0'`).
- Advanced pipelines can be built by chaining `map`, `filter`, `reduce`, etc.

### 31. Error taxonomy
### 32. RAG utilities

Functions:
- `rag.tokenize(text: string, model?: string) -> [string]`
- `rag.token_count(text: string, model?: string) -> int`
- `rag.chunk_tokens(text: string, size: int, overlap?: int, model?: string) -> [string]`
- `rag.rerank(results_json: array, query: string) -> array` (sort by score or textual match)
- `rag.fuse_passages(results_json: array, max?: int) -> string` (concatenate top passages)
- `rag.summarize_chunks(chunks_json: array, model?: string) -> string` (LLM-based synthesis)

Notes:
- Tokenization is whitespace-based by default; model-aware tokenization hooks are planned.
- `ingest_chunks` chunker supports `by: tokens|chars|paragraphs` for flexible pipelines.


The executor returns structured errors surfaced as runtime failures:
- `ArgumentError` — wrong arity/type or invalid argument values.
- `DataError` — file/serialization issues.
- `RuntimeError` — command execution or unexpected runtime faults.

All errors propagate to the CLI with clear messages; multioutput helpers validate inputs strictly.


### 33. MCP (Micro-Agent Communication Protocol) - Minimal Contract

Lexon exposes tools over JSON-RPC 2.0 via stdio and WebSocket servers.

- Servers:
  - `lexc --mcp-stdio` reads requests from stdin and writes responses to stdout.
  - `lexc --mcp-ws` starts a WebSocket server (addr from `LEXON_MCP_WS_ADDR`, default `127.0.0.1:8080`).

- Methods (requests):
  - `list_tools` → `{ "result": { "tools": [ { "name": string, "description"?: string, "meta"?: json } ] } }`
  - `tool_info` → `{ "params": { "name": string } }` returns `{ "result": { "name": string, "meta": json } }`
  - `tool_call` → `{ "params": { "name": string, "args"?: json } }`
    - Validates `args` against optional JSON schema (per-tool).
    - Enforces per-tool quotas and concurrency limits.
    - Returns either `{ "result": { "name": string, "output": json } }` or `{ "error": { "code": int, "message": string } }`.
  - `set_quota` → `{ "params": { "name": string, "allowed_calls": int } }` to set tool quotas (in‑process; persisted if memory_path configured).
  - `rpc.cancel` → cooperative cancellation (stdio: flips an internal cancel flag; WS: acknowledges).
  - `rpc.discover` → lists supported methods and protocol version.
  - `ping` → `{ "result": { "pong": true } }`

- Notifications (server → client):
  - `heartbeat` every `LEXON_MCP_HEARTBEAT_MS` ms if set, both stdio and WS.
  - `progress` (stdio and WS, opt‑in) when `LEXON_MCP_STREAM=1` during tool execution.
  - `rpc.cancel` (WS): best‑effort cooperative — the server listens for `rpc.cancel` with the same `id` during execution and sets an internal cancel flag; if a tool completes too quickly, the cancel may arrive too late.

- Security:
  - Bearer auth (optional): set `LEXON_MCP_AUTH_BEARER=token`. In stdio, the client must send `{"method":"auth","params":{"bearer":"token"}}` before any other call (except `ping`/`rpc.discover`); in WS, the handshake requires header `Authorization: Bearer token`.
  - TLS for WS (optional): set `LEXON_MCP_TLS_CERT` and `LEXON_MCP_TLS_KEY` (PEM). The WS server listens with TLS and validates Bearer during the handshake if configured.

#### 33.1. TLS/WS smoke script

Local test script (requires `websocat` if you want to see messages):

```bash
zsh scripts/mcp_ws_tls_smoke.sh
```

It does:
- Generates a self‑signed cert in `tmp/certs/`.
- Starts `--mcp-ws` with TLS and `LEXON_MCP_AUTH_BEARER=secret`.
- Attempts the WS handshake with `websocat` sending `Authorization: Bearer secret`.
- Shuts the server down.

Useful environment variables:
- `LEXON_MCP_WS_ADDR` (default `127.0.0.1:9443`)
- `LEXON_MCP_AUTH_BEARER` (default `secret`)
- `CERT_DIR` to change the certificates directory

- JWT authentication (optional):
  - `LEXON_MCP_AUTH_JWT_HS256=secret` enables HS256 JWT verification. For WS send `Authorization: Bearer <jwt>`. For stdio, send `{"method":"auth","params":{"jwt":"<jwt>"}}`.
  - HS256 signature and `exp` are verified. Identity derives from `sub` (if present).

- Per‑identity rate limits (stdio):
  - `LEXON_MCP_RATE_PER_MIN=N` limits `list_tools`/`tool_info`/`tool_call` per identity per minute. Errors use code `-32015`.

- Per‑identity ACL (stdio):
  - `LEXON_MCP_ACL_FILE=acl.json` (JSON: `{"identity":["tool1","tool2"]}`) restricts `tool_call`. Errors use code `-32016`.

- Client mTLS (WS, optional):
  - `LEXON_MCP_TLS_CLIENT_CA=ca.pem` requires a client cert signed by that CA during the TLS handshake.

- Security and governance:
  - Permission profiles via `LEXON_MCP_PERMISSION_PROFILE`; unknown tools denied by default if allowlist set (`LEXON_MCP_ALLOWED_TOOLS=tool1,tool2`).
  - Quotas: per‑tool `allowed_calls`/`used_calls` tracked in memory and optionally persisted to `${LEXON_MEMORY_PATH}/mcp_tools.json`.
  - Concurrency: per‑tool `max_concurrency` in tool meta; enforced in stdio/WS paths.
  - Timeouts: `LEXON_MCP_TOOL_TIMEOUT_MS` for individual tool execution; `LEXON_MCP_MSG_TIMEOUT_MS` for message send timeouts.

### 34. Standard library (stdlib)

Basic utility functions available in the runtime. Names accept dotted and flattened notation (e.g. `encoding.base64_encode` ≡ `encoding__base64_encode`). The dotted form is recommended.

#### 34.1 encoding
- `encoding.base64_encode(s: string) -> string`
- `encoding.base64_decode(s: string) -> string`
- `encoding.hex_encode(s: string) -> string`
- `encoding.hex_decode(s: string) -> string`
- `encoding.url_encode(s: string) -> string`
- `encoding.url_decode(s: string) -> string`

Example:
```lexon
let s = "Hello Lexon!";
let b64 = encoding.base64_encode(s);
let back = encoding.base64_decode(b64);  // "Hello Lexon!"
```

#### 34.2 strings
-- `strings.length(s: string) -> int`            (counts characters, not bytes)
- `strings.lower(s: string) -> string`
- `strings.upper(s: string) -> string`
- `strings.replace(s: string, from: string, to: string) -> string`
- `strings.split(s: string, sep: string) -> [string]`
- `strings.join(parts: [string], sep: string) -> string`
- `strings.starts_with(s: string, prefix: string) -> bool`
- `strings.substring(s: string, start: int, len?: int) -> string`

Example:
```lexon
let parts = strings.split("a,b,c", ",");
let out = strings.join(parts, "|");  // "a|b|c"
```

#### 34.3 math
- `math.sqrt(x: number) -> float`
- `math.pow(x: number, y: number) -> float`
- `math.min(a: number, b: number) -> float`
- `math.max(a: number, b: number) -> float`
- `math.clamp(x: number, lo: number, hi: number) -> float`

Notes:
- Parameters accept `int | float | string` (parsed to `float` when applicable).
- `clamp` returns `x` if `lo > hi`.

Example:
```lexon
print(math.sqrt(9));         // 3
print(math.pow(2, 3));       // 8
print(math.clamp(10, 0, 7)); // 7
```

#### 34.4 regex
- `regex.match(s: string, pattern: string) -> bool` (Regex Rust)
- `regex.replace(s: string, pattern: string, repl: string) -> string`

Example:
```lexon
print(regex.match("abc-123", "[0-9]+"));                 // true
print(regex.replace("abc-123-xyz", "[0-9]+", "#"));      // "abc-#-xyz"
```

#### 34.5 time
- `time.now_iso8601() -> string` (UTC, ISO‑8601)

Example:
```lexon
let ts = time.now_iso8601(); // "2025-11-08T12:34:56Z"
```

#### 34.6 number
- `number.format(n: number, decimals?: int=2) -> string`

Example:
```lexon
print(number.format(3.14159, 3)); // "3.142"
```

#### 34.7 crypto
- `crypto.sha256(s: string) -> string` (lowercase hex)

Example:
```lexon
print(crypto.sha256("lexon")); // "2f7bdf33..."
```

- 34.8 json
- `json.parse(text: string) -> json`
- `json.to_string(value: json|any) -> string`
- `json.get(obj: json|text, key: string) -> json|null`
- `json.path(obj: json|text, pointer: string) -> json|null` (JSON Pointer `/a/b/0`)
- `json.keys(obj: json|text) -> json` (array of strings)
- `json.length(value: json|text|string) -> int` (arrays/objects/strings)
- `json.index(arr: json|text, idx: int) -> json|null`
- `json.as_string(value: json|any) -> string` (if value is a JSON string, returns raw string; otherwise serializes)

Example:
```lexon
let j = json.parse("{\"a\":1,\"b\":[{\"t\":\"T1\"},{\"t\":\"T2\"}]}");
print(json.length(json.get(j, "b"))); // 2
print(json.to_string(json.path(j, "/b/1/t"))); // "T2"
```

#### 34.9 http
- `http.get(url: string) -> string`
- `http.get_json(url: string) -> json`
- `http.request(method: string, url: string, body?: string|json, headers?: json) -> json`

Behavior:
- `http.request` returns an object:
  - `status: int`
  - `headers: json` (flat map)
  - `body: string`
  - `body_json?: json` (present when body parses as JSON)

Configuration:
- Requires `LEXON_ALLOW_HTTP=1`
- Timeouts/retries: `LEXON_HTTP_TIMEOUT_MS` (default 10000), `LEXON_HTTP_RETRIES` (default 0)

Example:
```lexon
// GET JSON
let r = http.request("GET", "https://jsonplaceholder.typicode.com/todos/1");
print(json.to_string(json.get(r, "status")));       // 200
let data = json.get(r, "body_json");
print(json.to_string(json.path(data, "/title")));

// POST with headers and JSON body
let res = http.request("POST",
                       "https://postman-echo.com/post",
                       { "hello": "lexon" },
                       { "Content-Type": "application/json" });
print(json.to_string(json.get(res, "status")));     // 200
```

### 35. Known limitations (RC)
- Deeply nested string concatenations may hit lowering limits; prefer `strings.join([...], "")` or sequences of `let` + simple `+`.
- Avoid stray top‑level literals: use assignments or calls.
- String coercion for `+` exists at runtime; for strict control use `json.to_string`/`json.as_string`.

### 36. Fuzzing (parser / HIR / LexIR)

Lexon ships `cargo-fuzz` harnesses under `v1.0.0-rc.1/fuzz/`. Targets:

| Target        | Description                                                      |
|---------------|------------------------------------------------------------------|
| `parser_cst`  | Feeds random UTF-8 into the tree-sitter parser (CST only).       |
| `hir_builder` | Parses input and runs `build_hir_from_cst`.                      |
| `hir_to_lexir`| Full CST → HIR → LexIR conversion (semantic/execution skipped).  |

Prereqs:
```bash
rustup toolchain install nightly
cargo install cargo-fuzz
```

Run from repo root:
```bash
cargo fuzz list
cargo +nightly fuzz run parser_cst -- -max_total_time=60
cargo +nightly fuzz run hir_builder -- -max_total_time=60
cargo +nightly fuzz run hir_to_lexir -- -max_total_time=60
```

Artifacts/minimized cases land in `fuzz/artifacts/<target>/`; corpora live in `fuzz/corpus/<target>/`. When a crash is found, copy the minimized input into the corpus and consider adding a regression test or golden.

- Configuration:
  - Heartbeats: `LEXON_MCP_HEARTBEAT_MS=500` (ms).
  - Streaming progress (stdio): `LEXON_MCP_STREAM=1`.
  - WS addr: `LEXON_MCP_WS_ADDR=127.0.0.1:8080`.
  - One‑shot WS (auto exit on idle): `LEXON_MCP_SERVER_ONESHOT=1`.
  - Built‑in tools available: `emit_file`, `read_file`, `write_file`, `web.search` (plus any dynamically registered tools).

- Error codes (indicative):
  - `-32001` tool not allowed
  - `-32002` quota exceeded
  - `-32003` tool execution error
  - `-32004` tool timeout
  - `-32000` unauthorized
  - `-32010` not allowed under permission profile
  - `-32011` stdio tool execution error
  - `-32012` schema validation error
  - `-32013` concurrency limit exceeded

Argument schema:
- Full JSON Schema is supported if the tool `schema` is a JSON Schema object (compiled with `jsonschema`); detailed validation messages included.
- Simplified support (basic-typed properties) as a fallback.
Notes:
- Stdio cooperative cancellation is best‑effort: the server sets a cancel flag and tools periodically check it; long pure‑CPU operations may complete before cancellation.
- WS: heartbeats implemented; streaming progress for WS may be added in a future increment.
