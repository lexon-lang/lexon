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
2. **Storage Layer (PLuggable Backend)**
   - Persists `MemorySpaceFile` under `.lexon/structured_memory/<space>.json`.
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
- Policies can include auto-pin rules, retention hints, visibility flags (future).

## 4. API surface (Lexon primitives)

| Primitive | Summary |
|-----------|---------|
| `memory_space.create(name, metadata_json?)` | Initializes/resets a space. Pass `{"reset": true}` to wipe contents (deterministic tests). |
| `memory_space.list()` | Lists all spaces (summaries). |
| `remember_structured(space, payload_json, options?)` | Ingests a pre-built `MemoryObject`. Auto-fills missing summaries from `raw`. |
| `remember_raw(space, kind, raw_text, options?)` | Calls the semantic LLM (OpenAI/Anthropic/Google/Ollama/HF/custom) to infer path, summaries, tags, relevance, `auto_pin`. Options: `model`, `temperature`, `max_tokens`, `path_hint`, `project`, `tags`, `metadata`, `auto_pin`. |
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

## 7. Governance & observability

- Budgets: `remember_*` and `recall_*` respect provider budgets/quotas (shared with `ask_*`).
- Telemetry: each call emits spans (LLM call, file IO, backend scoring).
- Deterministic tests: `samples/memory/structured_semantic.lx` + `golden/memory/structured_semantic.txt` cover end-to-end behavior.
- Storage: `.lexon/structured_memory/*.json` kept alongside other runtime state; safe to vendor into reproducible bundles.

## 8. Roadmap (structured memory track)

- Additional backends (TemporalTree decay, external GraphRAG adapters).
- Per-call backend overrides in `recall_context` / `recall_kind`.
- Cross-project memory links and role-based visibility.
- Memory inspector tooling (`lexc memory browse`), query/filter UI.
- Expanded MCP demos blending structured memory, agents, RAG, multioutput.

See `ROADMAP.md` for cross-cutting roadmap items (DX, providers, IR optimizations, networking/stdlib, sockets, CI hardening, etc.).

---

*Last updated:* 2025-12-03  
*Files referenced:* `lexc/src/executor/structured_memory.rs`, `lexc/src/executor/structured_memory/backends/*.rs`, `samples/memory/structured_semantic.lx`, `golden/memory/structured_semantic.txt`.

