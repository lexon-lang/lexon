## Lexon Documentation (RC Bundle)

This document reflects what is implemented in this RC. Examples use only supported syntax and functions. Lexon is an LLM‑first programming language, built with extensive AI‑assisted development ("vibe coding").

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

Also available: `ask_parallel(...)`, `ask_merge(responses, "synthesize"|"summarize")`, `ask_with_fallback(models, prompt)`, `ask_ensemble(prompts, strategy, model)`.

Await support: `await ask { ... }`, `await ask_safe { ... }`, and the orchestration calls above.

Parameters supported in block forms (per grammar):
- `ask { system|user|schema|model|temperature|max_tokens }`
- `ask_safe { system|user|schema|model|temperature|max_tokens|validation|confidence_threshold|max_attempts|cross_reference_models|use_fact_checking }`

### 16. Multioutput (Preview in RC)

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

Runtime behavior (current RC):
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

Notes (RC limitations):
- Arguments are validated (types/arity). Errors are surfaced as runtime errors.
- All file I/O honors CLI sandbox flags.
- Real binaries (opt‑in): set environment variable `LEXON_REAL_BINARIES=1` to use the LLM response as payload for text-like outputs (JSON wraps `{ text: ... }`, CSV emits a small summary row). Default is deterministic stubs.
- Directory creation: `save_multioutput_file(...)` will auto-create parent directories of the provided path.
- Pseudo‑streaming (opt‑in): set `LEXON_STREAM_TEXT=1` to print primary_text lines as they are available (stdout only; no incremental API yet).
- Not GA yet: incremental streaming API; non text‑like binary generation; per‑file advanced metadata; progress callbacks; strict per‑file size limits.

Error cases (current RC):
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

Notes:
- Ingest accepts CSV/JSON/NDJSON/text; builds a lightweight vector index.
- `vector_search` returns top‑k matches with simple cosine similarity.

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
// String predicate/transform expressions are supported in this RC
let doubled = map(arr, 'x * 2');
let evens = filter(arr, 'x % 2 == 0');
let total = reduce(arr, 0, 'acc + x');
```

Notes:
- Predicates and callbacks use string expressions in this RC (e.g., `'x % 2 == 0'`).
- Advanced pipelines can be built by chaining `map`, `filter`, `reduce`, etc.

### 31. Error taxonomy

The executor returns structured errors surfaced as runtime failures:
- `ArgumentError` — wrong arity/type or invalid argument values.
- `DataError` — file/serialization issues.
- `RuntimeError` — command execution or unexpected runtime faults.

All errors propagate to the CLI with clear messages; multioutput helpers validate inputs strictly.

