# Structured Project Memory — Technical Specification (v1.0.0-rc.1)

## 1. Purpose

Structured Project Memory (SPM) is Lexon’s native “second brain.” It ingests raw artifacts (code, guides, decisions, logs), normalizes them into canonical `MemoryObject`s, stores them in pluggable tree/index backends, and exposes retrieval primitives (`recall_context`, `recall_kind`, `before_action use_context`) so every Lexon program can access curated context before or alongside traditional RAG. The goal is to provide durable knowledge per project with deterministic behavior, governance, and zero extra frameworks.

## 2. System overview

```
Raw assets ──> Semantic layer ──> MemoryObject ──> Backend (basic/patricia/raptor/hybrid)
     ^             |                                 |               |
     |             |                                 |               └─> Policy evaluation, summaries
     |             └─> remember_raw / remember_structured           ↓
  Lexon runtime <────────────────────── recall_context / recall_kind / before_action use_context
```

### 2.1 Components

1. **Semantic Layer**
   - Input: raw text + hints (`path_hint`, `kind`, `project`, `tags`, `metadata`).
   - Output: normalized `MemoryObject`.
   - Implementations: `remember_raw` (LLM-assisted) or `remember_structured` (pre-built JSON).
2. **Storage Layer (Pluggable Backend)**
   - Persists `MemorySpaceFile` under `.lexon/structured_memory/<space>.json` (see `space_path` helper). Each file contains the entire space, sorted deterministically by `updated_at`.
   - Provides ordering/scoring for topics (`order_for_topic`) and kinds (`order_for_kind`).
   - Backends: `basic`, `patricia`, `raptor`, `hybrid (GraphRAG/MemTree)`.
3. **Runtime Layer**
   - `memory_space.create/list`, `remember_*`, `recall_*`, `pin_memory`, `set_memory_policy`.
   - `before_action use_context` hook injects bundles into MCP agents/flows.

## 3. Memory object schema

```json
{
  "id": "mem_* (optional override)",
  "path": "project/module/topic",
  "kind": "guide|config|decision|log|custom",
  "raw": "... original text ...",
  "summary_micro": "1-2 sentences",
  "summary_short": "short paragraph / bullets",
  "summary_long": "long-form abstract",
  "tags": ["lowercase", "slugs"],
  "metadata": {"project": "...", "space": "...", "importance": "high"},
  "relevance": "high|medium|low",
  "pinned": true|false,
  "created_at": "RFC3339",
  "updated_at": "RFC3339"
}
```

- `MemorySpaceFile` aggregates `objects`, `metadata`, `policies`, `updated_at`.
```json
{
  "id": "mem_runtime_decision_001",
  "path": "lexon/runtime/release_checklist",
  "kind": "decision",
  "raw": "Ship RC once structured memory recalls <200ms and MCP supervisor is green.",
  "summary_micro": "RC go/no-go checklist",
  "summary_short": "Decisions and blockers for promoting Lexon v1.0.0-rc.1 to GA.",
  "summary_long": "Context, telemetry targets, and required samples-smoke pass/fail report for the release.",
  "tags": ["runtime", "release", "checklist"],
  "metadata": {"project": "lexon", "space": "runtime", "source": "docs/runtime_decision.md", "hash": "sha256:6be0...", "pii_flags": []},
  "relevance": "high",
  "pinned": true,
  "chunks": [
    {"chunk_id": "runtime_decision_001#0", "text": "Context..."},
    {"chunk_id": "runtime_decision_001#1", "text": "Checklist..."}
  ],
  "created_at": "2025-12-01T11:32:00Z",
  "updated_at": "2025-12-01T12:04:00Z"
}
```

- `MemorySpaceFile` aggregates `objects`, `metadata`, `policies`, `updated_at`.
- Policies can include auto-pin rules, retention hints, visibility flags (future).

### 3.1 Retention + PII

| Policy field | Meaning | Default |
|--------------|---------|---------|
| `ttl_days` | Max age before objects are archived/pruned. `memory_space.gc()` enforces it. | `null` (never expires) |
| `pii_redaction` | List of regexes or detectors (emails, keys) to scrub before persisting `raw`. | `[]` |
| `export_path` | Absolute or workspace-relative directory for `memory_space.export`. Produces JSON bundle per space. | `.lexon/exports/<space>.json` |
| `allow_delete` | Boolean guard for `memory_space.delete(space)` operations. | `false` |

Retention enforcement flow:
1. `remember_*` stamps `created_at`/`updated_at` using deterministic clock when `freeze_clock` is set.
2. `set_memory_policy` with `ttl_days` + `allow_delete=true` enables `memory_space.gc(space)` to purge expired objects (and write audit logs).
3. Redaction occurs before serialization: detectors strip emails/API keys from `raw` and stash hashes inside `metadata.pii_flags`.
4. Exports are JSON bundles signed with `space` + `hash` so teams can hand them to audit/compliance or restore elsewhere.

Deletion/export commands are CLI-level operations so teams can satisfy GDPR/CCPA requests without editing Lexon files manually.

## 4. API surface (Lexon primitives)

| Primitive | Summary |
|-----------|---------|
| `memory_space.create(name, metadata_json?)` | Initializes/resets a space. Pass `{"reset": true}` to wipe contents (deterministic tests). |
| `memory_space.list()` | Lists all spaces (summaries). |
| `remember_structured(space, payload_json, options?)` | Ingests a pre-built `MemoryObject`. Auto-fills missing summaries from `raw`. |
| `remember_raw(space, kind, raw_text, options?)` | Calls the semantic LLM (OpenAI/Anthropic/Google/Ollama/HF/custom) to infer path, summaries, tags, relevance, `auto_pin`. Options: `model`, `temperature`, `max_tokens`, `path_hint`, `project`, `tags`, `metadata`, `auto_pin`. Falls back to `options.model` → `LEXON_MEMORY_SEMANTIC_MODEL` → `config.llm_model`. |
| `pin_memory(space, id_or_path)` / `unpin_memory(...)` | Toggle `pinned`. |
| `set_memory_policy(space, policy_json)` | Persist project-specific auto-pin/retention/visibility rules. |
| `recall_context(space, topic, options_json?)` | Returns a bundle: `global_summary`, `sections[]`, optional `raw[]`, `generated_at`. Options: `limit`, `raw_limit`, `include_raw`, `include_metadata`, `prefer_kinds`, `prefer_tags`, `require_high_relevance`, `freeze_clock`. |
| `recall_kind(space, kind, options_json?)` | Returns ordered list of memories filtered by kind. |
| `before_action use_context project="..." topic="..."` | MCP hook to auto-inject recall bundles before agent tasks. |

Determinism hooks:
- `freeze_clock` overrides timestamps in bundles for goldens.
- `{"reset": true}` ensures predictable states in tests.

## 5. Backend designs

### 5.1 Basic backend
- Scoring: base relevance (high/medium/low), pinned bonus, topic path/summary/raw matches, `prefer_kinds`, `prefer_tags`.
- Implementation: `basic.rs` scoring function returning float; orders descending.

### 5.2 Patricia backend
- Compresses paths into trie segments for prefix lookups.
- Bonus for deeper prefix matches (`patricia_prefix_depth`).
- Falls back to `basic` heuristics plus prefix depth.

### 5.3 RAPTOR backend
- Lightweight clustering: overlaps tags and summary tokens with topic tokens.
- Recency score (`updated_at` timestamp) + pin bonus.
- `cluster_overlap` adds points for matching tags or summary tokens.

### 5.4 Hybrid (GraphRAG/MemTree) backend
- Extracts entity tokens from paths, tags, metadata.
- Builds frequency map to weight rare tokens.
- Scores overlap with topic tokens and pinned tokens, reuses `basic_score` for baseline.
- Simulates GraphRAG-style entity graphs without external service.

### 5.5 Backend selection
- `StructuredMemoryService::new` reads `LEXON_MEMORY_BACKEND` env var (default `basic`) and instantiates the backend trait via `build_backend(name)`.
- On unknown backend, logs warning and falls back to `basic`.
- Future roadmap: per-call overrides via options (`{"backend": "patricia"}`).

### 5.6 Pinning semantics
- `pin_memory` / `unpin_memory` delegate to `toggle_pin` which accepts either `id` or `path`.
- Updating the pin flag also refreshes `updated_at` so recalls reflect the latest ordering.
- Missing IDs/paths surface `ExecutorError::RuntimeError("Memory '...' not found in space '...'")`.

### 5.7 Backend guarantees at a glance

| Backend | Persistence | Consistency / guarantees | Recommended footprint |
|---------|-------------|--------------------------|-----------------------|
| `basic` | Single JSON file per space | Deterministic ordering, idempotent upserts by `id` + `path` | <5k objects per space |
| `patricia` | Same JSON file + in-memory trie | Prefix-aware ordering; serialization unaffected | Deep config/guide trees |
| `raptor` | JSON + cached tag clusters | At-least-once updates, recency decay (<30d) prevents stale bundles | Spaces dominated by tagged notes |
| `hybrid` | JSON + transient entity map | Entity-overlap scoring; requires `metadata.hash` for dedupe | Mixed specs + decision logs |

All backends share the same persistence primitive (`MemorySpaceFile`). Switching only changes ordering/scoring; the serialized data stays identical.


## 6. Bundled context format

`recall_context` returns:

```json
{
  "space": "lexon_demo",
  "topic": "runtime",
  "generated_at": "2025-01-01T00:00:00Z",
  "global_summary": "Context bundle for 'runtime' (N items): ...",
  "sections": [
    {
      "id": "...",
      "path": "...",
      "kind": "...",
      "summary_micro": "...",
      "summary_short": "...",
      "summary_long": "...",
      "relevance": "...",
      "pinned": true,
      "tags": [...],
      "metadata": {...},
      "updated_at": "..."
    }
  ],
  "raw": [
    {"path": "...", "raw": "...", "kind": "..."}
  ],
  "limit": 2
}
```

Ordering rules:
- Pinned + high relevance first.
- Respect `limit`, `raw_limit`.
- `require_high_relevance` filters out lower scores.
- `prefer_kinds` and `prefer_tags` boost relevant memories.

### 6.1 Ingest → index → query flow

```
watcher (git diff / filesystem) --> enqueue(raw_asset)
  -> normalizer (assign path/kind/tags)
  -> remember_raw / remember_structured
      -> dedupe by metadata.hash
      -> persist to MemorySpaceFile (JSON)
      -> backend caches ordering view
recall_context/topic --> backend.order_for_topic --> bundle --> before_action hook --> agents/prompts
```

Pseudo-API for deterministic ingest:

```lexon
let files = filesystem.watch("docs/runtime/*.md");
for file in files {
  let raw = read_file(file.path);
  remember_raw("runtime", "guide", raw, strings.json({
    path_hint: file.path,
    project: "lexon",
    tags: ["runtime", file.name]
  }));
}
```

Chunk metadata (`chunks[]`) is optional, but when present it lets downstream RAG pipelines correlate structured memory with vector search hits.

## 7. Governance & observability

- **Budgets & quotas**: `remember_*` / `recall_*` share the same budget manager as `ask_*`. Policies can cap USD/token spend per provider and per memory space.
- **Telemetry spans**: enabling `LEXON_OTEL=1` emits `lexon.memory.remember_raw`, `lexon.memory.recall_context`, `lexon.memory.gc`, and backend-specific spans with attributes (`space`, `topic`, `backend`, `items_returned`, `budget_usd`).
- **Counters**: Prometheus exporter surfaces `lexon_memory_objects_total`, `lexon_memory_gc_runs_total`, `lexon_memory_pin_toggle_total`, plus latency histograms.
- **Deterministic tests**: [`samples/memory/structured_semantic.lx`](../samples/memory/structured_semantic.lx) + [`golden/memory/structured_semantic.txt`](../golden/memory/structured_semantic.txt) rely on `{"reset": true}` + `freeze_clock`. They fail CI if bundles drift.
- **Storage layout**: `.lexon/structured_memory/<space>.json` and `.lexon/exports/*.json` live next to other runtime state so workspaces can be version-controlled or shipped as artifacts.

## 8. Error handling

- Missing spaces → file auto-created, but invalid JSON yields `ExecutorError::RuntimeError("Invalid space ...")`.
- Unknown backend name → warning + fallback to `basic`.
- Serialization issues when writing spaces/objects bubble up as runtime errors; callers see `ExecutorError::RuntimeError`.
- `toggle_pin`/`pin_memory` on unknown IDs/paths returns explicit `"Memory '...' not found"` errors.
- `remember_raw` surfaces adapter errors (LLM failures) just like `ask`.

## 9. End-to-end example (repo delta QA)

Goal: index today's repo changes and answer "What changed in telemetry?"

1. Reset + ingest recent diffs:
   ```bash
   memory_space.create("lexon_repo", """{"reset": true}""");
   git diff --name-only HEAD~1 | grep '.rs$' | while read file; do \
     lexc remember_raw --space lexon_repo --kind code --path "$file" "$file"; \
   done
   ```
2. Recall telemetry:
   ```lexon
   let bundle = recall_context("lexon_repo", "telemetry", """{"limit":3,"include_raw":true}""");
   print(bundle.global_summary);
   ```
3. Feed bundle into agents/RAG:
   ```lexon
   before_action use_context project="lexon_repo", topic="telemetry";
   let analyst = agent_create("lexon_repo_analyst", """{"model":"openai:gpt-4o-mini"}""");
   let report = agent_run(analyst, "Describe telemetry changes today", """{"deadline_ms":15000}""");
   print(report);
   ```
4. Export or prune:
   ```bash
   lexc memory export lexon_repo --out exports/lexon_repo.json
   lexc memory gc lexon_repo --ttl-days 30
   ```

The flow exercises ingestion, ordering, recall, MCP hooks, and governance knobs end-to-end.

## 10. Roadmap (structured memory track)

- Additional backends (TemporalTree decay, external GraphRAG adapters).
- Per-call backend overrides in `recall_context` / `recall_kind`.
- Cross-project memory links and role-based visibility.
- Memory inspector tooling (`lexc memory browse`), query/filter UI.
- Expanded MCP demos blending structured memory, agents, RAG, multioutput.

See [`ROADMAP.md`](../ROADMAP.md) for cross-cutting roadmap items (DX, providers, IR optimizations, networking/stdlib, sockets, CI hardening, etc.).

---

*Last updated:* 2025-12-03  
*Files referenced:* `lexc/src/executor/structured_memory.rs`, `lexc/src/executor/structured_memory/backends/*.rs`, `samples/memory/structured_semantic.lx`, `golden/memory/structured_semantic.txt`.

