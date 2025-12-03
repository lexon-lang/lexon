use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fs;
use std::path::PathBuf;

use super::ExecutorError;
mod backends;
use backends::{build_backend, MemoryBackend};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryObject {
    pub id: String,
    pub path: String,
    pub kind: String,
    pub raw: String,
    pub summary_micro: String,
    pub summary_short: String,
    pub summary_long: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default = "default_relevance")]
    pub relevance: String,
    #[serde(default)]
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySpaceFile {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub metadata: Value,
    #[serde(default)]
    pub policies: Value,
    #[serde(default)]
    pub objects: Vec<MemoryObject>,
}

#[derive(Debug, Clone)]
pub struct RecallOptions {
    pub limit: usize,
    pub raw_limit: usize,
    pub include_raw: bool,
    pub include_metadata: bool,
    pub prefer_kinds: Vec<String>,
    pub prefer_tags: Vec<String>,
    pub require_high_relevance: bool,
    pub freeze_clock: Option<String>,
}

impl Default for RecallOptions {
    fn default() -> Self {
        Self {
            limit: 5,
            raw_limit: 2,
            include_raw: false,
            include_metadata: true,
            prefer_kinds: Vec::new(),
            prefer_tags: Vec::new(),
            require_high_relevance: false,
            freeze_clock: None,
        }
    }
}

impl RecallOptions {
    pub fn from_value(value: Option<&Value>) -> Self {
        let mut opts = RecallOptions::default();
        if let Some(Value::Object(map)) = value {
            if let Some(limit) = map.get("limit").and_then(|v| v.as_u64()) {
                if limit > 0 {
                    opts.limit = limit.min(20) as usize;
                }
            }
            if let Some(raw_limit) = map.get("raw_limit").and_then(|v| v.as_u64()) {
                if raw_limit > 0 {
                    opts.raw_limit = raw_limit.min(5) as usize;
                }
            }
            if let Some(include_raw) = map.get("include_raw").and_then(|v| v.as_bool()) {
                opts.include_raw = include_raw;
            }
            if let Some(include_metadata) = map.get("include_metadata").and_then(|v| v.as_bool()) {
                opts.include_metadata = include_metadata;
            }
            if let Some(require_high) = map
                .get("require_high_relevance")
                .and_then(|v| v.as_bool())
            {
                opts.require_high_relevance = require_high;
            }
            if let Some(kinds) = map.get("prefer_kinds").and_then(|v| v.as_array()) {
                opts.prefer_kinds = kinds
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_lowercase())
                    .collect();
            }
            if let Some(tags) = map.get("prefer_tags").and_then(|v| v.as_array()) {
                opts.prefer_tags = tags
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_lowercase())
                    .collect();
            }
            if let Some(clock) = map.get("freeze_clock").and_then(|v| v.as_str()) {
                opts.freeze_clock = Some(clock.to_string());
            }
        }
        opts
    }
}

pub struct StructuredMemoryService {
    base_dir: PathBuf,
    backend: Box<dyn MemoryBackend>,
}

impl StructuredMemoryService {
    pub fn new(memory_path: Option<String>) -> Self {
        let base = memory_path
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(".lexon"));
        let mut structured = base;
        structured.push("structured_memory");
        let backend_name =
            std::env::var("LEXON_MEMORY_BACKEND").unwrap_or_else(|_| "basic".to_string());
        let backend = build_backend(&backend_name).unwrap_or_else(|e| {
            println!(
                "[structured_memory] ⚠️ backend '{}' not available ({}), falling back to 'basic'",
                backend_name, e
            );
            build_backend("basic").expect("basic backend must exist")
        });
        println!(
            "[structured_memory] backend selected: {}",
            backend.id()
        );
        Self {
            base_dir: structured,
            backend,
        }
    }

    pub fn create_space(
        &self,
        raw_name: &str,
        metadata: Option<Value>,
    ) -> Result<Value, ExecutorError> {
        let name = Self::canonical_space_name(raw_name);
        self.ensure_parent_dir()?;
        let mut space = self.read_space(&name)?;
        if let Some(mut meta) = metadata {
            let reset = meta
                .get("reset")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if reset {
                space.objects.clear();
                if let Some(obj) = meta.as_object_mut() {
                    obj.remove("reset");
                }
            }
            space.metadata = merge_metadata(space.metadata, meta);
        }
        space.updated_at = Utc::now();
        self.write_space(&space)?;
        Ok(space_summary_value(&space))
    }

    pub fn list_spaces(&self) -> Result<Value, ExecutorError> {
        self.ensure_parent_dir()?;
        let mut summaries: Vec<Value> = Vec::new();
        if self.base_dir.exists() {
            for entry in fs::read_dir(&self.base_dir)
                .map_err(|e| ExecutorError::RuntimeError(format!("{}", e)))?
            {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() && path.extension().map(|e| e == "json").unwrap_or(false) {
                        if let Ok(contents) = fs::read_to_string(&path) {
                            if let Ok(space) = serde_json::from_str::<MemorySpaceFile>(&contents) {
                                summaries.push(space_summary_value(&space));
                            }
                        }
                    }
                }
            }
        }
        summaries.sort_by(|a, b| {
            let ad = a
                .get("updated_at")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            let bd = b
                .get("updated_at")
                .and_then(|v| v.as_str())
                .unwrap_or_default();
            bd.cmp(&ad)
        });
        Ok(Value::Array(summaries))
    }

    pub fn upsert_object(
        &self,
        raw_space: &str,
        mut object: MemoryObject,
        auto_pin: bool,
    ) -> Result<Value, ExecutorError> {
        let space_name = Self::canonical_space_name(raw_space);
        self.ensure_parent_dir()?;
        let mut space = self.read_space(&space_name)?;
        if auto_pin {
            object.pinned = true;
        }
        let mut updated = false;
        for existing in &mut space.objects {
            if existing.id == object.id {
                *existing = object.clone();
                updated = true;
                break;
            }
        }
        if !updated {
            space.objects.push(object.clone());
        }
        space.updated_at = Utc::now();
        self.write_space(&space)?;
        serde_json::to_value(object)
            .map_err(|e| ExecutorError::RuntimeError(format!("Failed to serialize object: {}", e)))
    }

    pub fn set_policy(
        &self,
        raw_space: &str,
        policy: Value,
    ) -> Result<Value, ExecutorError> {
        let space_name = Self::canonical_space_name(raw_space);
        self.ensure_parent_dir()?;
        let mut space = self.read_space(&space_name)?;
        space.policies = policy.clone();
        space.updated_at = Utc::now();
        self.write_space(&space)?;
        Ok(policy)
    }

    pub fn toggle_pin(
        &self,
        raw_space: &str,
        identifier: &str,
        pin: bool,
    ) -> Result<Value, ExecutorError> {
        let space_name = Self::canonical_space_name(raw_space);
        self.ensure_parent_dir()?;
        let mut space = self.read_space(&space_name)?;
        let mut target: Option<MemoryObject> = None;
        for obj in &mut space.objects {
            if obj.id == identifier || obj.path == identifier {
                obj.pinned = pin;
                obj.updated_at = Utc::now();
                target = Some(obj.clone());
                break;
            }
        }
        if target.is_none() {
            return Err(ExecutorError::RuntimeError(format!(
                "Memory '{}' not found in space '{}'",
                identifier, space_name
            )));
        }
        space.updated_at = Utc::now();
        self.write_space(&space)?;
        serde_json::to_value(target.unwrap())
            .map_err(|e| ExecutorError::RuntimeError(format!("Failed to serialize pin result: {}", e)))
    }

    pub fn recall_context(
        &self,
        raw_space: &str,
        topic: &str,
        options: &RecallOptions,
    ) -> Result<Value, ExecutorError> {
        let space_name = Self::canonical_space_name(raw_space);
        self.ensure_parent_dir()?;
        let space = self.read_space(&space_name)?;
        let ordered = self
            .backend
            .order_for_topic(&space, topic, options)
            .unwrap_or_else(|| space.objects.iter().collect());
        let mut bundle: Vec<Value> = Vec::new();
        let mut raw_snippets: Vec<Value> = Vec::new();
        for (idx, obj) in ordered.iter().take(options.limit).enumerate() {
            bundle.push(object_summary(obj, options.include_metadata));
            if options.include_raw && idx < options.raw_limit {
                raw_snippets.push(json!({
                    "path": obj.path,
                    "raw": clamp_text(&obj.raw, 1200),
                    "kind": obj.kind,
                }));
            }
        }
        let global_summary = render_global_summary(bundle.as_slice(), topic);
        let generated_at = options
            .freeze_clock
            .clone()
            .unwrap_or_else(|| Utc::now().to_rfc3339());
        Ok(json!({
            "space": space_name,
            "topic": topic,
            "generated_at": generated_at,
            "global_summary": global_summary,
            "sections": bundle,
            "raw": raw_snippets,
            "limit": options.limit,
        }))
    }

    pub fn recall_by_kind(
        &self,
        raw_space: &str,
        kind: &str,
        options: &RecallOptions,
    ) -> Result<Value, ExecutorError> {
        let space_name = Self::canonical_space_name(raw_space);
        self.ensure_parent_dir()?;
        let space = self.read_space(&space_name)?;
        let ordered = self
            .backend
            .order_for_kind(&space, kind, options)
            .unwrap_or_else(|| {
                space
                    .objects
                    .iter()
                    .filter(|obj| obj.kind.eq_ignore_ascii_case(kind))
                    .collect()
            });
        let mut matches: Vec<Value> = Vec::new();
        for obj in ordered.into_iter().take(options.limit) {
            matches.push(object_summary(obj, options.include_metadata));
        }
        Ok(Value::Array(matches))
    }

    fn ensure_parent_dir(&self) -> Result<(), ExecutorError> {
        if let Some(parent) = self.base_dir.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| ExecutorError::RuntimeError(format!("Failed to init memory dir: {}", e)))?;
        }
        fs::create_dir_all(&self.base_dir)
            .map_err(|e| ExecutorError::RuntimeError(format!("Failed to init memory dir: {}", e)))
    }

    fn read_space(&self, name: &str) -> Result<MemorySpaceFile, ExecutorError> {
        let path = self.space_path(name);
        if path.exists() {
            let text = fs::read_to_string(&path)
                .map_err(|e| ExecutorError::RuntimeError(format!("Failed to read space '{}': {}", name, e)))?;
            serde_json::from_str::<MemorySpaceFile>(&text)
                .map_err(|e| ExecutorError::RuntimeError(format!("Invalid space '{}': {}", name, e)))
        } else {
            Ok(MemorySpaceFile {
                name: name.to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                metadata: Value::Object(Map::new()),
                policies: Value::Object(Map::new()),
                objects: Vec::new(),
            })
        }
    }

    fn write_space(&self, space: &MemorySpaceFile) -> Result<(), ExecutorError> {
        let path = self.space_path(&space.name);
        let text = serde_json::to_string_pretty(space).map_err(|e| {
            ExecutorError::RuntimeError(format!("Failed to serialize space '{}': {}", space.name, e))
        })?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| ExecutorError::RuntimeError(format!("Failed to init memory dir: {}", e)))?;
        }
        fs::write(&path, text)
            .map_err(|e| ExecutorError::RuntimeError(format!("Failed to persist space '{}': {}", space.name, e)))
    }

    fn space_path(&self, name: &str) -> PathBuf {
        let mut path = self.base_dir.clone();
        path.push(format!("{}.json", name));
        path
    }

    fn canonical_space_name(raw: &str) -> String {
        let trimmed = raw.trim().to_lowercase();
        let mut clean = String::with_capacity(trimmed.len());
        for c in trimmed.chars() {
            if c.is_ascii_alphanumeric() || matches!(c, '/' | '_' | '-') {
                clean.push(c);
            } else if c.is_whitespace() {
                clean.push('_');
            }
        }
        if clean.is_empty() {
            "default".to_string()
        } else {
            clean
        }
    }
}

fn default_relevance() -> String {
    "medium".to_string()
}

fn space_summary_value(space: &MemorySpaceFile) -> Value {
    let pinned = space.objects.iter().filter(|o| o.pinned).count();
    json!({
        "name": space.name,
        "objects": space.objects.len(),
        "pinned": pinned,
        "created_at": space.created_at.to_rfc3339(),
        "updated_at": space.updated_at.to_rfc3339(),
        "metadata": space.metadata,
    })
}

fn merge_metadata(existing: Value, overlay: Value) -> Value {
    match (existing, overlay) {
        (Value::Object(mut base), Value::Object(new_vals)) => {
            for (k, v) in new_vals {
                base.insert(k, v);
            }
            Value::Object(base)
        }
        (_, other) => other,
    }
}

fn object_summary(obj: &MemoryObject, include_metadata: bool) -> Value {
    let mut value = json!({
        "id": obj.id,
        "path": obj.path,
        "kind": obj.kind,
        "summary_micro": obj.summary_micro,
        "summary_short": obj.summary_short,
        "summary_long": clamp_text(&obj.summary_long, 1000),
        "relevance": obj.relevance,
        "pinned": obj.pinned,
        "tags": obj.tags,
        "updated_at": obj.updated_at.to_rfc3339(),
    });
    if include_metadata {
        if let Value::Object(ref mut map) = value {
            map.insert("metadata".to_string(), obj.metadata.clone());
        }
    }
    value
}

fn clamp_text(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        text.to_string()
    } else {
        format!("{}...", &text[..limit])
    }
}

fn render_global_summary(sections: &[Value], topic: &str) -> String {
    if sections.is_empty() {
        return format!("No memories found for '{}'", topic);
    }
    let mut lines = Vec::new();
    lines.push(format!(
        "Context bundle for '{}' ({} items):",
        topic,
        sections.len()
    ));
    for value in sections.iter().take(6) {
        let path = value
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let summary = value
            .get("summary_micro")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        lines.push(format!("• {} — {}", path, summary));
    }
    lines.join("\n")
}

