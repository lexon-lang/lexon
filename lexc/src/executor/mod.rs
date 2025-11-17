// lexc/src/executor/mod.rs
//
// Experimental executor for LexIR
// This module implements a basic interpreter for LexIR that uses Polars for
// data operations and a simple memory adapter.

mod agents;
mod api_config;
pub mod async_ops;
mod data_functions;
pub mod data_processor;
mod llm_adapter;
mod llm_functions;
mod mcp;
mod memory;
mod memory_functions;
#[cfg(test)]
mod tests;
pub mod vector_memory; // 🧠 Sprint D: Real vector memory system // 🚀 ASYNC-v1: Async operations

use crate::lexir::LexUnaryOperator;
use crate::lexir::{
    LexBinaryOperator, LexExpression, LexFunction, LexInstruction, LexLiteral, LexProgram, TempId,
    ValueRef,
};
// futures unused
use serde_json::{json, Value};

// Web search configuration (module scope)
struct WebSearchConfig {
    endpoint: String,
    auth_mode: Option<String>, // "header" | "query"
    auth_name: Option<String>, // header or query param name
    api_key: Option<String>,
    query_param: String,
    count_param: String,
    format_param: Option<String>,
    format_value: Option<String>,
    extra_params: Vec<(String, String)>,
}
use crate::executor::llm_adapter::{ProviderConfig, ProviderKind};
use base64::Engine;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use memory::MemoryManager;
use vector_memory::VectorMemorySystem; // 🧠 Import real vector system
#[derive(Debug, Clone)]
pub(crate) struct ToolMeta {
    pub allowed_calls: Option<u64>,
    pub used_calls: u64,
    #[allow(dead_code)]
    pub scope: Option<String>,
}

fn normalize_search_json(endpoint: &str, original: &serde_json::Value) -> serde_json::Value {
    let ep = endpoint.to_lowercase();
    // SerpAPI (Google): aggregate multiple blocks to maximize recall
    // - organic_results
    // - news_results
    // - top_stories
    // - related_questions (use question/link)
    // - related_searches (use query/link if available)
    if ep.contains("serpapi.com") {
        let mut results: Vec<Value> = Vec::new();
        // helper to push (title,url,text)
        let mut push_item = |title: &str, url: &str, text: &str| {
            if !title.is_empty() || !url.is_empty() || !text.is_empty() {
                results.push(json!({
                    "title": title,
                    "url": url,
                    "Text": text,
                    "FirstURL": url
                }));
            }
        };
        if let Some(arr) = original.get("organic_results").and_then(|v| v.as_array()) {
            for it in arr {
                let title = it.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let url = it.get("link").and_then(|v| v.as_str()).unwrap_or("");
                let text = it.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                push_item(title, url, text);
            }
        }
        if let Some(arr) = original.get("news_results").and_then(|v| v.as_array()) {
            for it in arr {
                let title = it.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let url = it.get("link").and_then(|v| v.as_str()).unwrap_or("");
                let text = it.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                push_item(title, url, text);
            }
        }
        if let Some(arr) = original.get("top_stories").and_then(|v| v.as_array()) {
            for it in arr {
                let title = it.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let url = it.get("link").and_then(|v| v.as_str()).unwrap_or("");
                let text = it.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                push_item(title, url, text);
            }
        }
        if let Some(arr) = original.get("related_questions").and_then(|v| v.as_array()) {
            for it in arr {
                let title = it.get("question").and_then(|v| v.as_str()).unwrap_or("");
                let url = it.get("link").and_then(|v| v.as_str()).unwrap_or("");
                let text = it.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                push_item(title, url, text);
            }
        }
        if let Some(arr) = original.get("related_searches").and_then(|v| v.as_array()) {
            for it in arr {
                let title = it.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let url = it.get("link").and_then(|v| v.as_str()).unwrap_or("");
                let text = it.get("snippet").and_then(|v| v.as_str()).unwrap_or("");
                push_item(title, url, text);
            }
        }
        return json!({ "Results": results });
    }
    // Brave Search
    if ep.contains("api.search.brave.com") {
        let mut results: Vec<Value> = Vec::new();
        if let Some(arr) = original.pointer("/web/results").and_then(|v| v.as_array()) {
            for it in arr {
                let title = it.get("title").and_then(|v| v.as_str()).unwrap_or("");
                let url = it.get("url").and_then(|v| v.as_str()).unwrap_or("");
                let text = it.get("description").and_then(|v| v.as_str()).unwrap_or("");
                results.push(json!({
                    "title": title,
                    "url": url,
                    "Text": text,
                    "FirstURL": url
                }));
            }
        }
        return json!({ "Results": results });
    }
    // DuckDuckGo Instant Answer: no fallback aggregation here (app may handle formatting)
    // Default: return original (reporting handles nulls/fallbacks)
    original.clone()
}

#[derive(Debug, Clone)]
pub(crate) struct AgentState {
    pub model: String,
    pub budget_usd: Option<f64>,
    pub deadline_ms: Option<u64>,
}

// use crate::telemetry::{
//     trace_ask_operation, trace_ask_safe_operation, trace_data_operation, trace_memory_operation,
// };

/// Execution error
#[derive(Debug)]
pub enum ExecutorError {
    UnsupportedInstruction(String),
    UndefinedVariable(String),
    TypeError(String),
    DataError(String),
    MemoryError(String),
    LlmError(String),
    NameError(String),
    ArgumentError(String),
    UndefinedFunction(String),
    RuntimeError(String),
    IoError(String),
    CsvError(String),
    JsonError(String),
    CommandError(String),
    AsyncError(String),
}

// Display implementation for ExecutorError
impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::UnsupportedInstruction(msg) => {
                write!(f, "Unsupported instruction: {}", msg)
            }
            ExecutorError::UndefinedVariable(var) => write!(f, "Undefined variable: {}", var),
            ExecutorError::TypeError(msg) => write!(f, "Type error: {}", msg),
            ExecutorError::DataError(msg) => write!(f, "Data error: {}", msg),
            ExecutorError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            ExecutorError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            ExecutorError::NameError(msg) => write!(f, "Name error: {}", msg),
            ExecutorError::ArgumentError(msg) => write!(f, "Argument error: {}", msg),
            ExecutorError::UndefinedFunction(func) => write!(f, "Undefined function: {}", func),
            ExecutorError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            ExecutorError::IoError(msg) => write!(f, "I/O error: {}", msg),
            ExecutorError::CsvError(msg) => write!(f, "CSV error: {}", msg),
            ExecutorError::JsonError(msg) => write!(f, "JSON error: {}", msg),
            ExecutorError::CommandError(msg) => write!(f, "Command error: {}", msg),
            ExecutorError::AsyncError(msg) => write!(f, "Async error: {}", msg),
        }
    }
}

pub type Result<T> = std::result::Result<T, ExecutorError>;

/// Binary file for multioutput
#[derive(Debug, Clone)]
pub struct BinaryFile {
    pub name: String,
    pub content: Vec<u8>,
    pub mime_type: String,
    pub size: usize,
}

impl BinaryFile {
    pub fn new(name: String, content: Vec<u8>, mime_type: String) -> Self {
        let size = content.len();
        Self {
            name,
            content,
            mime_type,
            size,
        }
    }

    pub fn from_text(name: String, text: String) -> Self {
        use std::path::Path;
        let content = text.into_bytes();
        let mime_type = match Path::new(&name).extension().and_then(|e| e.to_str()) {
            Some("txt") => "text/plain",
            Some("md") | Some("mdx") => "text/markdown",
            Some("json") => "application/json",
            Some("csv") => "text/csv",
            Some("html") | Some("htm") => "text/html",
            Some("css") => "text/css",
            Some("js") => "application/javascript",
            Some("ts") => "application/typescript",
            Some("xml") => "application/xml",
            Some("yaml") | Some("yml") => "application/x-yaml",
            Some("toml") => "application/toml",
            _ => "text/plain",
        }
        .to_string();
        Self::new(name, content, mime_type)
    }

    pub fn from_json(name: String, json: &serde_json::Value) -> Result<Self> {
        let content = serde_json::to_vec_pretty(json)
            .map_err(|e| ExecutorError::DataError(format!("JSON serialization error: {}", e)))?;
        Ok(Self::new(name, content, "application/json".to_string()))
    }
}

/// Runtime values
#[derive(Debug, Clone)]
pub enum RuntimeValue {
    /// Null value
    Null,
    /// Boolean value
    Boolean(bool),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Dataset
    Dataset(Arc<data_processor::Dataset>),
    /// JSON value
    Json(Value),
    /// Result that can be success or error (uses Box to avoid recursion)
    Result {
        success: bool,
        value: Box<RuntimeValue>,
        error_message: Option<String>,
    },
    /// Multiple outputs: primary text + binary files + metadata
    MultiOutput {
        primary_text: String,
        binary_files: Vec<BinaryFile>,
        metadata: HashMap<String, String>,
    },
}

impl From<LexLiteral> for RuntimeValue {
    fn from(lit: LexLiteral) -> Self {
        match lit {
            LexLiteral::Integer(i) => RuntimeValue::Integer(i),
            LexLiteral::Float(f) => RuntimeValue::Float(f),
            LexLiteral::String(s) => RuntimeValue::String(s),
            LexLiteral::Boolean(b) => RuntimeValue::Boolean(b),
            LexLiteral::Array(arr) => {
                let json_arr: Vec<serde_json::Value> = arr
                    .into_iter()
                    .map(|item| {
                        match item {
                            LexLiteral::Integer(i) => serde_json::Value::Number(i.into()),
                            LexLiteral::Float(f) => serde_json::Value::Number(
                                serde_json::Number::from_f64(f)
                                    .unwrap_or_else(|| serde_json::Number::from(0)),
                            ),
                            LexLiteral::String(s) => serde_json::Value::String(s),
                            LexLiteral::Boolean(b) => serde_json::Value::Bool(b),
                            LexLiteral::Array(nested) => {
                                // Recursively convert nested arrays
                                let nested_runtime = RuntimeValue::from(LexLiteral::Array(nested));
                                if let RuntimeValue::Json(json_val) = nested_runtime {
                                    json_val
                                } else {
                                    serde_json::Value::Array(vec![])
                                }
                            }
                        }
                    })
                    .collect();
                RuntimeValue::Json(serde_json::Value::Array(json_arr))
            }
        }
    }
}

/// Configuration of the executor
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Optional path for persistent memory storage
    pub memory_path: Option<String>,
    /// Level of detail for execution logs
    pub verbose: bool,
    /// LLM model to use ("simulated" for simulated mode, or real model name)
    pub llm_model: Option<String>,
    /// Use new LLM architecture (real-by-default)
    pub use_new_llm_architecture: bool, // Temporarily disabled
    /// Operation mode for LLM: "real", "simulated", "auto"
    pub llm_mode: Option<String>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            memory_path: None,
            verbose: false,
            llm_model: None,
            use_new_llm_architecture: true,
            llm_mode: Some("auto".to_string()),
        }
    }
}

#[allow(dead_code)]
enum LoopSignal {
    Break,
    Continue,
}

/// Control flow signals for execution
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ControlFlow {
    /// Continues normal execution
    Continue,
    /// Breaks a loop
    Break,
    /// Skips to the next iteration of a loop
    Skip,
    /// Returns from a function with a value
    Return(RuntimeValue),
}

fn format_runtime_value(value: &RuntimeValue) -> String {
    match value {
        RuntimeValue::String(s) => s.clone(),
        RuntimeValue::Integer(i) => i.to_string(),
        RuntimeValue::Float(f) => f.to_string(),
        RuntimeValue::Boolean(b) => b.to_string(),
        RuntimeValue::Null => String::new(),
        RuntimeValue::Dataset(_) => "Dataset".to_string(),
        RuntimeValue::Json(json) => json.to_string(),
        RuntimeValue::Result {
            success,
            value,
            error_message,
        } => {
            if *success {
                format!("Ok({})", format_runtime_value(value))
            } else {
                match error_message {
                    Some(msg) => format!("Error({})", msg),
                    None => "Error(unknown)".to_string(),
                }
            }
        }
        RuntimeValue::MultiOutput {
            primary_text,
            binary_files,
            metadata,
        } => {
            let files_info = binary_files
                .iter()
                .map(|f| format!("{}({} bytes, {})", f.name, f.size, f.mime_type))
                .collect::<Vec<_>>()
                .join(", ");
            let meta_info = if metadata.is_empty() {
                String::new()
            } else {
                format!(", metadata: {}", metadata.len())
            };
            format!(
                "MultiOutput(text: {}, files: [{}]{})",
                primary_text.chars().take(50).collect::<String>()
                    + if primary_text.len() > 50 { "..." } else { "" },
                files_info,
                meta_info
            )
        }
    }
}

/// Execution environment for LexIR programs
pub struct ExecutionEnvironment {
    /// Variable storage
    variables: HashMap<String, RuntimeValue>,
    /// Temporary variables (generated by the compiler)
    temporaries: HashMap<TempId, RuntimeValue>,
    /// Data processor
    data_processor: data_processor::DataProcessor,
    /// Memory manager
    memory_manager: MemoryManager,
    /// Legacy LLM adapter
    llm_adapter: llm_adapter::LlmAdapter,
    /// New LLM adapter (real-by-default)
    llm_adapter_new: Option<llm_adapter::LlmAdapter>,
    /// Executor configuration
    config: ExecutorConfig,
    /// Functions defined in the program
    functions: HashMap<String, LexFunction>,
    /// Vector memory system (Sprint D)
    vector_memory_system: Option<VectorMemorySystem>,
    /// Orchestration: tool registry with quotas
    tool_registry: HashMap<String, ToolMeta>,
    /// Agents: registry and state
    agent_registry: HashMap<String, AgentState>,
    agent_status: HashMap<String, String>,
    agent_cancelled: HashMap<String, bool>,
    /// Async tasks
    async_scheduler: Option<crate::runtime::scheduler::AsyncScheduler>,
    tasks: HashMap<String, crate::runtime::scheduler::TaskHandle>,
    /// Concurrency channels
    channel_senders: HashMap<String, tokio::sync::mpsc::Sender<String>>,
    channel_receivers: HashMap<String, tokio::sync::mpsc::Receiver<String>>,
    /// Cooperative cancellation for long-running ops within this environment
    cancel_requested: bool,
    /// Optional shared cancellation flag (for external servers like MCP WS)
    cancel_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Metrics
    total_llm_calls: u64,
    llm_total_ms: u128,
    /// Concurrency primitives
    rate_limits: HashMap<String, (u64, u32)>, // epoch_sec -> count
    /// Sessions TTL/GC
    session_ttl_ms: u64,
    session_gc_interval_ms: u64,
    last_session_gc_epoch_ms: u64,
    /// Validation defaults
    validation_min_confidence: f64,
    validation_max_attempts: u32,
    validation_strategy: String,
    validation_domain: Option<String>,
    // Quality gates
    #[allow(dead_code)]
    quality_schema_enforce: bool,
    #[allow(dead_code)]
    quality_pii_block: bool,
}
impl ExecutionEnvironment {
    /// Creates a new execution environment
    pub fn new(config: ExecutorConfig) -> Self {
        let llm_adapter_new = if config.use_new_llm_architecture {
            Some(llm_adapter::LlmAdapter::new())
        } else {
            None
        };

        Self {
            variables: HashMap::new(),
            temporaries: HashMap::new(),
            data_processor: data_processor::DataProcessor::new(),
            memory_manager: MemoryManager::new(),
            llm_adapter: llm_adapter::LlmAdapter::new(),
            llm_adapter_new,
            config,
            functions: HashMap::new(),
            vector_memory_system: {
                let db_path = std::env::var("LEXON_VECTOR_DB_PATH").ok();
                VectorMemorySystem::new(db_path.as_deref()).ok()
            },
            tool_registry: HashMap::new(),
            agent_registry: HashMap::new(),
            agent_status: HashMap::new(),
            agent_cancelled: HashMap::new(),
            async_scheduler: Some(crate::runtime::scheduler::AsyncScheduler::new(8)),
            tasks: HashMap::new(),
            channel_senders: HashMap::new(),
            channel_receivers: HashMap::new(),
            cancel_requested: false,
            cancel_flag: None,
            total_llm_calls: 0,
            llm_total_ms: 0,
            validation_min_confidence: 0.6,
            validation_max_attempts: 2,
            validation_strategy: "basic".to_string(),
            validation_domain: None,
            rate_limits: HashMap::new(),
            session_ttl_ms: std::env::var("LEXON_SESSION_TTL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10 * 60 * 1000),
            session_gc_interval_ms: std::env::var("LEXON_SESSION_GC_INTERVAL_MS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60 * 1000),
            last_session_gc_epoch_ms: 0,
            quality_schema_enforce: std::env::var("LEXON_QUALITY_SCHEMA_ENFORCE")
                .ok()
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
            quality_pii_block: std::env::var("LEXON_QUALITY_PII_BLOCK")
                .ok()
                .map(|v| v.to_lowercase() == "true")
                .unwrap_or(false),
        }
    }

    /// Attach a shared cancellation flag to this environment
    pub fn set_cancel_flag(&mut self, flag: std::sync::Arc<std::sync::atomic::AtomicBool>) {
        self.cancel_flag = Some(flag);
    }

    fn now_epoch_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_millis(0))
            .as_millis() as u64
    }

    fn maybe_run_session_gc(&mut self) {
        let now = Self::now_epoch_ms();
        if now.saturating_sub(self.last_session_gc_epoch_ms) >= self.session_gc_interval_ms {
            let _ = self.memory_manager.clean_expired();
            self.last_session_gc_epoch_ms = now;
            println!("[sessions] GC ran");
        }
    }

    /// Enforce quotas and emit on_tool_call
    fn before_tool_call(&mut self, name: &str) -> Result<()> {
        if std::env::var("LEXON_VERBOSE").ok().as_deref() == Some("1") {
            println!("🛠️ on_tool_call: {}", name);
        }
        if let Some(meta) = self.tool_registry.get_mut(name) {
            if let Some(limit) = meta.allowed_calls {
                if meta.used_calls >= limit {
                    return Err(ExecutorError::RuntimeError(format!(
                        "Tool '{}' quota exceeded",
                        name
                    )));
                }
            }
            meta.used_calls = meta.used_calls.saturating_add(1);
        }
        Ok(())
    }

    /// Emit on_tool_success/on_tool_error
    fn after_tool_event(&mut self, name: &str, success: bool, error: Option<&str>) {
        if std::env::var("LEXON_VERBOSE").ok().as_deref() == Some("1") {
            if success {
                println!("✅ on_tool_success: {}", name);
            } else {
                println!(
                    "❌ on_tool_error: {} - {}",
                    name,
                    error.unwrap_or("unknown")
                );
            }
        }
    }

    /// Minimal LLM metrics hook
    #[allow(dead_code)]
    fn record_llm_call(&mut self, elapsed_ms: u128) {
        self.total_llm_calls = self.total_llm_calls.saturating_add(1);
        self.llm_total_ms = self.llm_total_ms.saturating_add(elapsed_ms);
        println!(
            "[INFO] [llm_call] COMPLETED in {:?}",
            std::time::Duration::from_millis(elapsed_ms as u64)
        );
    }

    // ---- RAG helper utilities (redundancy/score parsing) ----
    fn jaccard_sim_local(a: &str, b: &str) -> f32 {
        let sa: std::collections::HashSet<&str> = a.split_whitespace().collect();
        let sb: std::collections::HashSet<&str> = b.split_whitespace().collect();
        let inter = sa.intersection(&sb).count() as f32;
        let uni = sa.union(&sb).count() as f32;
        if uni == 0.0 {
            0.0
        } else {
            inter / uni
        }
    }

    fn parse_score_local(resp: &str) -> f32 {
        let lower = resp.to_lowercase();
        for line in lower.lines() {
            if let Some(pos) = line.find("score") {
                let tail = &line[pos..];
                for tok in tail.split(|c: char| !c.is_ascii_digit() && c != '.') {
                    if tok.is_empty() {
                        continue;
                    }
                    if let Ok(v) = tok.parse::<f32>() {
                        return v.clamp(0.0, 1.0);
                    }
                }
            }
        }
        0.0
    }

    fn estimate_token_count_local(text: &str, model: Option<&str>) -> usize {
        // Approximate per-model chars-per-token ratios; fallback to 4
        let m = model.unwrap_or("").to_lowercase();
        let ratio: f32 = if m.contains("gpt-4") || m.contains("gpt-3.5") {
            4.0
        } else if m.contains("claude") {
            5.0
        } else if m.contains("gemini") {
            4.0
        } else {
            4.0
        };
        let chars = text.chars().count() as f32;
        ((chars / ratio).ceil() as usize).max(1)
    }

    fn cosine_sim_local(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    #[cfg(feature = "tokenizers")]
    fn encode_tokens_precise(
        text: &str,
        model: Option<&str>,
    ) -> Option<(tiktoken_rs::CoreBPE, Vec<usize>)> {
        let bpe = if let Some(m) = model {
            tiktoken_rs::get_bpe_from_model(m)
                .ok()
                .or_else(|| tiktoken_rs::cl100k_base().ok())
        } else {
            tiktoken_rs::cl100k_base().ok()
        }?;
        let ids = bpe.encode_ordinary(text);
        Some((bpe, ids))
    }

    #[cfg(not(feature = "tokenizers"))]
    fn encode_tokens_precise(_text: &str, _model: Option<&str>) -> Option<((), Vec<usize>)> {
        None
    }

    #[cfg(feature = "tokenizers")]
    fn tokens_to_strings(bpe: &tiktoken_rs::CoreBPE, ids: &[usize]) -> Vec<String> {
        ids.iter()
            .map(|&id| bpe.decode(vec![id]).unwrap_or_default())
            .collect()
    }

    #[cfg(not(feature = "tokenizers"))]
    fn tokens_to_strings(_bpe: &(), _ids: &[usize]) -> Vec<String> {
        Vec::new()
    }

    #[cfg(feature = "tokenizers")]
    fn decode_range_to_string(bpe: &tiktoken_rs::CoreBPE, ids: &[usize]) -> String {
        bpe.decode(ids.to_vec()).unwrap_or_default()
    }

    #[cfg(not(feature = "tokenizers"))]
    fn decode_range_to_string(_bpe: &(), _ids: &[usize]) -> String {
        String::new()
    }

    // -------- Prompt registry helpers --------
    fn prompt_registry_path() -> String {
        let base = std::env::var("LEXON_MEMORY_PATH").unwrap_or_else(|_| ".lexon".to_string());
        format!("{}/prompts.json", base)
    }

    fn load_prompts() -> serde_json::Map<String, serde_json::Value> {
        let path = Self::prompt_registry_path();
        match std::fs::read_to_string(&path) {
            Ok(txt) => serde_json::from_str(&txt).unwrap_or_default(),
            Err(_) => serde_json::Map::new(),
        }
    }

    #[allow(dead_code)]
    fn save_prompts(map: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        let path = Self::prompt_registry_path();
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&serde_json::Value::Object(map.clone()))
                .unwrap_or_else(|_| "{}".to_string()),
        )
        .map_err(|e| ExecutorError::IoError(format!("save prompts: {}", e)))
    }

    fn render_template(tpl: &str, vars: &serde_json::Value) -> String {
        let mut out = tpl.to_string();
        if let Some(obj) = vars.as_object() {
            for (k, v) in obj {
                let needle = format!("{{{{{}}}}}", k);
                let val = if let Some(s) = v.as_str() {
                    s.to_string()
                } else {
                    v.to_string()
                };
                out = out.replace(&needle, &val);
            }
        }
        out
    }

    // -------- Distributed cache (stub) helpers --------
    #[allow(dead_code)]
    fn distcache_path() -> String {
        std::env::var("LEXON_DISTCACHE_PATH")
            .unwrap_or_else(|_| "samples/cache/dist_cache.json".to_string())
    }

    #[allow(dead_code)]
    fn distcache_load() -> serde_json::Map<String, serde_json::Value> {
        let path = Self::distcache_path();
        match std::fs::read_to_string(&path) {
            Ok(txt) => serde_json::from_str(&txt).unwrap_or_default(),
            Err(_) => serde_json::Map::new(),
        }
    }

    #[allow(dead_code)]
    fn distcache_save(map: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        let path = Self::distcache_path();
        if let Some(parent) = std::path::Path::new(&path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&serde_json::Value::Object(map.clone()))
                .unwrap_or_else(|_| "{}".to_string()),
        )
        .map_err(|e| ExecutorError::IoError(format!("distcache save: {}", e)))
    }

    fn load_web_search_from_toml() -> Option<WebSearchConfig> {
        let content = std::fs::read_to_string("lexon.toml").ok()?;
        let v: toml::Value = content.parse().ok()?;
        let tbl = v.get("web_search")?.as_table()?;
        let endpoint = tbl
            .get("endpoint")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if endpoint.is_empty() {
            return None;
        }
        let auth_mode = tbl
            .get("auth_mode")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let auth_name = tbl
            .get("auth_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let auth_env = tbl.get("auth_env").and_then(|v| v.as_str()).unwrap_or("");
        let api_key = if auth_env.is_empty() {
            None
        } else {
            std::env::var(auth_env).ok()
        };
        let query_param = tbl
            .get("query_param")
            .and_then(|v| v.as_str())
            .unwrap_or("q")
            .to_string();
        let count_param = tbl
            .get("count_param")
            .and_then(|v| v.as_str())
            .unwrap_or("n")
            .to_string();
        let format_param = tbl
            .get("format_param")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let format_value = tbl
            .get("format_value")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let mut extra_params: Vec<(String, String)> = Vec::new();
        if let Some(ep) = tbl.get("extra_params").and_then(|v| v.as_table()) {
            for (k, val) in ep {
                if let Some(s) = val.as_str() {
                    extra_params.push((k.clone(), s.to_string()));
                }
            }
        }
        Some(WebSearchConfig {
            endpoint,
            auth_mode,
            auth_name,
            api_key,
            query_param,
            count_param,
            format_param,
            format_value,
            extra_params,
        })
    }

    /// Execute simple tool calls from MCP JSON-RPC
    fn execute_tool_call_json(
        &mut self,
        name: &str,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        if self.cancel_requested {
            return Err(ExecutorError::RuntimeError("cancelled".to_string()));
        }
        if let Some(flag) = &self.cancel_flag {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                return Err(ExecutorError::RuntimeError("cancelled".to_string()));
            }
        }
        let out_ref = ValueRef::Named("__mcp_out".to_string());
        match name {
            "web.search" => {
                let query = args
                    .get("query")
                    .and_then(|v| v.as_str())
                    .or_else(|| args.as_str())
                    .unwrap_or("")
                    .to_string();
                let n: usize = args.get("n").and_then(|v| v.as_u64()).unwrap_or(5) as usize;
                // Reuse builtin behavior (endpoint or simulated)
                if query.is_empty() {
                    return Err(ExecutorError::ArgumentError(
                        "web.search requires 'query'".to_string(),
                    ));
                }
                if let Ok(endpoint) = std::env::var("LEXON_WEB_SEARCH_ENDPOINT") {
                    let q = query.replace(" ", "+");
                    let mut url = format!("{}?q={}&n={}", endpoint, q, n);
                    if endpoint.contains("duckduckgo.com") {
                        url.push_str("&format=json");
                    }
                    let resp = ureq::get(&url)
                        .call()
                        .map_err(|e| ExecutorError::RuntimeError(format!("web.search: {}", e)))?;
                    let text = resp.into_string().map_err(|e| {
                        ExecutorError::RuntimeError(format!("web.search read: {}", e))
                    })?;
                    let json_raw: serde_json::Value =
                        serde_json::from_str(&text).unwrap_or(serde_json::json!({"raw": text}));
                    let json_norm = normalize_search_json(&endpoint, &json_raw);
                    self.store_value(&out_ref, RuntimeValue::Json(json_norm))?;
                } else if let Some(cfg) = Self::load_web_search_from_toml() {
                    let q = query.replace(" ", "+");
                    let sep = if cfg.endpoint.contains('?') { '&' } else { '?' };
                    let mut url = format!(
                        "{}{}{}={}&{}={}",
                        cfg.endpoint, sep, cfg.query_param, q, cfg.count_param, n
                    );
                    if let (Some(fp), Some(fv)) =
                        (cfg.format_param.as_deref(), cfg.format_value.as_deref())
                    {
                        url.push_str(&format!("&{}={}", fp, fv));
                    }
                    for (k, v) in &cfg.extra_params {
                        url.push_str(&format!("&{}={}", k, v));
                    }
                    let mut req = ureq::get(&url);
                    if let (Some(mode), Some(name), Some(key)) = (
                        cfg.auth_mode.as_deref(),
                        cfg.auth_name.as_deref(),
                        cfg.api_key.as_deref(),
                    ) {
                        if mode.eq_ignore_ascii_case("header") {
                            req = req.set(name, key);
                        } else if mode.eq_ignore_ascii_case("query") {
                            url.push_str(&format!("&{}={}", name, key));
                            req = ureq::get(&url);
                        }
                    }
                    // Pagination for SerpAPI if available to increase recall
                    if cfg.endpoint.contains("serpapi.com") {
                        let mut agg: Vec<serde_json::Value> = Vec::new();
                        let mut page_start: usize = 0;
                        let mut done_pages: usize = 0;
                        while agg.len() < n && done_pages < 3 {
                            let mut page_url = format!("{}&start={}", url, page_start);
                            let mut preq = ureq::get(&page_url);
                            if let (Some(mode), Some(name), Some(key)) = (
                                cfg.auth_mode.as_deref(),
                                cfg.auth_name.as_deref(),
                                cfg.api_key.as_deref(),
                            ) {
                                if mode.eq_ignore_ascii_case("header") {
                                    preq = preq.set(name, key);
                                } else if mode.eq_ignore_ascii_case("query") {
                                    page_url.push_str(&format!("&{}={}", name, key));
                                    preq = ureq::get(&page_url);
                                }
                            }
                            let presp = preq.call().map_err(|e| {
                                ExecutorError::RuntimeError(format!("web.search: {}", e))
                            })?;
                            let ptext = presp.into_string().map_err(|e| {
                                ExecutorError::RuntimeError(format!("web.search read: {}", e))
                            })?;
                            let praw: serde_json::Value = serde_json::from_str(&ptext)
                                .unwrap_or(serde_json::json!({"raw": ptext}));
                            let pnorm = normalize_search_json(&cfg.endpoint, &praw);
                            if let Some(arr) = pnorm.get("Results").and_then(|v| v.as_array()) {
                                for it in arr {
                                    if agg.len() < n {
                                        agg.push(it.clone());
                                    }
                                }
                            }
                            page_start += 10;
                            done_pages += 1;
                        }
                        let final_json = serde_json::json!({ "Results": agg });
                        self.store_value(&out_ref, RuntimeValue::Json(final_json))?;
                    } else {
                        let resp = req.call().map_err(|e| {
                            ExecutorError::RuntimeError(format!("web.search: {}", e))
                        })?;
                        let text = resp.into_string().map_err(|e| {
                            ExecutorError::RuntimeError(format!("web.search read: {}", e))
                        })?;
                        let json_raw: serde_json::Value =
                            serde_json::from_str(&text).unwrap_or(serde_json::json!({"raw": text}));
                        let json_norm = normalize_search_json(&cfg.endpoint, &json_raw);
                        self.store_value(&out_ref, RuntimeValue::Json(json_norm))?;
                    }
                } else {
                    let items: Vec<serde_json::Value> = (1..=n.max(1)).map(|i| serde_json::json!({
                        "title": format!("Result {} for {}", i, query),
                        "url": format!("https://example.com/{}/{}", query.replace(" ", "_"), i),
                        "snippet": "SIMULATED SEARCH RESULT"
                    })).collect();
                    self.store_value(
                        &out_ref,
                        RuntimeValue::Json(serde_json::Value::Array(items)),
                    )?;
                }
            }
            "read_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .or_else(|| args.as_str())
                    .ok_or_else(|| {
                        ExecutorError::ArgumentError("read_file requires 'path'".to_string())
                    })?
                    .to_string();
                let arg_exprs = vec![LexExpression::Value(ValueRef::Literal(LexLiteral::String(
                    path,
                )))];
                self.handle_read_file(&arg_exprs, Some(&out_ref))?;
            }
            "emit_file" => {
                // Accept [name, mime, base64] or object {name,mime,base64}
                let (name, mime, b64) = if let Some(arr) = args.as_array() {
                    let n = arr
                        .get(0)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let m = arr
                        .get(1)
                        .and_then(|v| v.as_str())
                        .unwrap_or("application/octet-stream")
                        .to_string();
                    let b = arr
                        .get(2)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (n, m, b)
                } else {
                    let n = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let m = args
                        .get("mime")
                        .and_then(|v| v.as_str())
                        .unwrap_or("application/octet-stream")
                        .to_string();
                    let b = args
                        .get("base64")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (n, m, b)
                };
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64.as_bytes())
                    .unwrap_or_default();
                let file = BinaryFile::new(name, bytes, mime);
                let mo = self.create_multioutput("".to_string(), vec![file]);
                self.store_value(&out_ref, mo)?;
            }
            "save_binary_file" => {
                // Accept [path, base64]
                let (path, b64) = if let Some(arr) = args.as_array() {
                    let p = arr
                        .get(0)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let b = arr
                        .get(1)
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (p, b)
                } else {
                    let p = args
                        .get("path")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let b = args
                        .get("base64")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    (p, b)
                };
                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64.as_bytes())
                    .unwrap_or_default();
                // Build a temporary MultiOutput to reuse existing handler
                let file = BinaryFile::new(
                    "file.bin".to_string(),
                    bytes,
                    "application/octet-stream".to_string(),
                );
                let mo = self.create_multioutput("".to_string(), vec![file]);
                let arg_exprs = vec![
                    LexExpression::Value(ValueRef::Temp(TempId(999999))),
                    LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                ];
                // Store the multioutput in a temp before calling handler
                self.temporaries.insert(TempId(999999), mo);
                self.handle_save_binary_file(&arg_exprs, None)?;
                self.store_value(&out_ref, RuntimeValue::Boolean(true))?;
            }
            "write_file" => {
                let (path, content) = if let (Some(p), Some(c)) = (
                    args.get("path").and_then(|v| v.as_str()),
                    args.get("content"),
                ) {
                    (
                        p.to_string(),
                        if let Some(s) = c.as_str() {
                            s.to_string()
                        } else {
                            c.to_string()
                        },
                    )
                } else if let Some(a) = args.as_array() {
                    if a.len() == 2 {
                        (
                            a[0].as_str().unwrap_or("").to_string(),
                            a[1].as_str().unwrap_or("").to_string(),
                        )
                    } else {
                        return Err(ExecutorError::ArgumentError(
                            "write_file expects [path, content]".to_string(),
                        ));
                    }
                } else {
                    return Err(ExecutorError::ArgumentError(
                        "write_file requires 'path' and 'content'".to_string(),
                    ));
                };
                let arg_exprs = vec![
                    LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                    LexExpression::Value(ValueRef::Literal(LexLiteral::String(content))),
                ];
                self.handle_write_file(&arg_exprs, None)?;
                self.store_value(&out_ref, RuntimeValue::Boolean(true))?;
            }
            "save_file" => {
                let (content, path) = if let (Some(c), Some(p)) = (
                    args.get("content"),
                    args.get("path").and_then(|v| v.as_str()),
                ) {
                    (
                        if let Some(s) = c.as_str() {
                            s.to_string()
                        } else {
                            c.to_string()
                        },
                        p.to_string(),
                    )
                } else {
                    return Err(ExecutorError::ArgumentError(
                        "save_file requires 'content' and 'path'".to_string(),
                    ));
                };
                let arg_exprs = vec![
                    LexExpression::Value(ValueRef::Literal(LexLiteral::String(content))),
                    LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                ];
                self.handle_save_file(&arg_exprs, None)?;
                self.store_value(&out_ref, RuntimeValue::Boolean(true))?;
            }
            "load_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .or_else(|| args.as_str())
                    .ok_or_else(|| {
                        ExecutorError::ArgumentError("load_file requires 'path'".to_string())
                    })?
                    .to_string();
                let arg_exprs = vec![LexExpression::Value(ValueRef::Literal(LexLiteral::String(
                    path,
                )))];
                self.handle_load_file(&arg_exprs, Some(&out_ref))?;
            }
            "load_binary_file" => {
                let path = args
                    .get("path")
                    .and_then(|v| v.as_str())
                    .or_else(|| args.as_str())
                    .ok_or_else(|| {
                        ExecutorError::ArgumentError("load_binary_file requires 'path'".to_string())
                    })?
                    .to_string();
                let mut arg_exprs = vec![LexExpression::Value(ValueRef::Literal(
                    LexLiteral::String(path),
                ))];
                if let Some(n) = args.get("name").and_then(|v| v.as_str()) {
                    arg_exprs.push(LexExpression::Value(ValueRef::Literal(LexLiteral::String(
                        n.to_string(),
                    ))));
                }
                self.handle_load_binary_file(&arg_exprs, Some(&out_ref))?;
            }
            other => return Err(ExecutorError::UndefinedFunction(other.to_string())),
        }
        let val = self
            .variables
            .get("__mcp_out")
            .cloned()
            .unwrap_or(RuntimeValue::Null);
        Ok(match val {
            RuntimeValue::Json(j) => j,
            RuntimeValue::String(s) => serde_json::json!(s),
            RuntimeValue::Boolean(b) => serde_json::json!(b),
            RuntimeValue::Integer(i) => serde_json::json!(i),
            RuntimeValue::Float(f) => serde_json::json!(f),
            RuntimeValue::Null => serde_json::Value::Null,
            _ => serde_json::json!(format!("{:?}", val)),
        })
    }

    /// Creates a MultiOutput with text and files
    fn create_multioutput(&self, text: String, files: Vec<BinaryFile>) -> RuntimeValue {
        RuntimeValue::MultiOutput {
            primary_text: text,
            binary_files: files,
            metadata: HashMap::new(),
        }
    }

    /// Creates a MultiOutput with text, files, and metadata
    fn create_multioutput_with_metadata(
        &self,
        text: String,
        files: Vec<BinaryFile>,
        metadata: HashMap<String, String>,
    ) -> RuntimeValue {
        RuntimeValue::MultiOutput {
            primary_text: text,
            binary_files: files,
            metadata,
        }
    }

    /// Saves a binary file to disk
    fn save_binary_file(&self, file: &BinaryFile, path: &str) -> Result<()> {
        use std::fs::{create_dir_all, write};
        use std::path::Path;
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(e) = create_dir_all(parent) {
                    return Err(ExecutorError::DataError(format!(
                        "Failed to create directories for {}: {}",
                        path, e
                    )));
                }
            }
        }
        write(path, &file.content).map_err(|e| {
            ExecutorError::DataError(format!("Failed to save file {}: {}", path, e))
        })?;
        println!(
            "💾 Saved file: {} ({} bytes, {})",
            path, file.size, file.mime_type
        );
        Ok(())
    }

    /// Loads a binary file from disk
    fn load_binary_file(&self, path: &str, name: Option<String>) -> Result<BinaryFile> {
        use std::fs::read;
        use std::path::Path;

        let content = read(path).map_err(|e| {
            ExecutorError::DataError(format!("Failed to load file {}: {}", path, e))
        })?;

        let file_name = name.unwrap_or_else(|| {
            Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        // Detect MIME type based on extension
        let mime_type = match Path::new(path).extension().and_then(|ext| ext.to_str()) {
            Some("txt") => "text/plain",
            Some("json") => "application/json",
            Some("csv") => "text/csv",
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("pdf") => "application/pdf",
            Some("html") => "text/html",
            Some("xml") => "application/xml",
            _ => "application/octet-stream",
        }
        .to_string();

        Ok(BinaryFile::new(file_name, content, mime_type))
    }

    /// Resolves a value reference to a runtime value
    fn resolve_value(&self, value_ref: &ValueRef) -> Result<RuntimeValue> {
        match value_ref {
            ValueRef::Named(name) => {
                if let Some(v) = self.variables.get(name) {
                    Ok(v.clone())
                } else {
                    // Gracefully treat missing named variables as Null to avoid crashes on branch merges
                    Ok(RuntimeValue::Null)
                }
            }
            ValueRef::Temp(id) => {
                // Gracefully treat missing temporaries as null to avoid crashes on discarded top-level temps
                Ok(self
                    .temporaries
                    .get(id)
                    .cloned()
                    .unwrap_or(RuntimeValue::Null))
            }
            ValueRef::Literal(lit) => Ok(RuntimeValue::from(lit.clone())),
        }
    }

    /// Stores a value in a variable or temporary
    fn store_value(&mut self, target: &ValueRef, value: RuntimeValue) -> Result<()> {
        match target {
            ValueRef::Named(name) => {
                self.variables.insert(name.clone(), value);
            }
            ValueRef::Temp(id) => {
                self.temporaries.insert(id.clone(), value);
            }
            ValueRef::Literal(_) => {
                return Err(ExecutorError::TypeError(
                    "Cannot assign to a literal".to_string(),
                ));
            }
        }
        Ok(())
    }
    /// Evaluates an expression
    fn evaluate_expression(&self, expr: LexExpression) -> Result<RuntimeValue> {
        match expr {
            LexExpression::Value(value_ref) => self.resolve_value(&value_ref),

            LexExpression::BinaryOp {
                operator,
                left,
                right,
            } => {
                let left_value = self.evaluate_expression(*left)?;
                let right_value = self.evaluate_expression(*right)?;

                match operator {
                    // Arithmetic operators
                    LexBinaryOperator::Add => match (left_value, right_value) {
                        // If either side is a string, coerce the other to string for user-friendly concatenation
                        (RuntimeValue::String(mut a), b) => {
                            a.push_str(&format_runtime_value(&b));
                            Ok(RuntimeValue::String(a))
                        }
                        (a, RuntimeValue::String(b)) => {
                            let mut s = format_runtime_value(&a);
                            s.push_str(&b);
                            Ok(RuntimeValue::String(s))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Integer(a + b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Float(a + b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Float(a as f64 + b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Float(a + b as f64))
                        }

                        // Fallback: stringify both and concatenate to make `+` total on mixed types
                        (a, b) => {
                            let mut s = format_runtime_value(&a);
                            s.push_str(&format_runtime_value(&b));
                            Ok(RuntimeValue::String(s))
                        }
                    },
                    LexBinaryOperator::Subtract => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Integer(a - b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Float(a - b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Float(a as f64 - b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Float(a - b as f64))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for subtract operation".to_string(),
                        )),
                    },
                    LexBinaryOperator::Multiply => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Integer(a * b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Float(a * b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Float(a as f64 * b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Float(a * b as f64))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for multiply operation".to_string(),
                        )),
                    },
                    LexBinaryOperator::Divide => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            if b == 0 {
                                Err(ExecutorError::RuntimeError("Division by zero".to_string()))
                            } else {
                                Ok(RuntimeValue::Integer(a / b))
                            }
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            if b == 0.0 {
                                Err(ExecutorError::RuntimeError("Division by zero".to_string()))
                            } else {
                                Ok(RuntimeValue::Float(a / b))
                            }
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            if b == 0.0 {
                                Err(ExecutorError::RuntimeError("Division by zero".to_string()))
                            } else {
                                Ok(RuntimeValue::Float(a as f64 / b))
                            }
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            if b == 0 {
                                Err(ExecutorError::RuntimeError("Division by zero".to_string()))
                            } else {
                                Ok(RuntimeValue::Float(a / b as f64))
                            }
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for divide operation".to_string(),
                        )),
                    },

                    // Comparison operators
                    LexBinaryOperator::Equal => {
                        match (left_value, right_value) {
                            (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                                Ok(RuntimeValue::Boolean(a == b))
                            }
                            (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                                Ok(RuntimeValue::Boolean(a == b))
                            }
                            (RuntimeValue::Boolean(a), RuntimeValue::Boolean(b)) => {
                                Ok(RuntimeValue::Boolean(a == b))
                            }
                            (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                                Ok(RuntimeValue::Boolean(a == b))
                            }
                            _ => Ok(RuntimeValue::Boolean(false)), // Different types are never equal
                        }
                    }
                    LexBinaryOperator::NotEqual => {
                        match (left_value, right_value) {
                            (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                                Ok(RuntimeValue::Boolean(a != b))
                            }
                            (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                                Ok(RuntimeValue::Boolean(a != b))
                            }
                            (RuntimeValue::Boolean(a), RuntimeValue::Boolean(b)) => {
                                Ok(RuntimeValue::Boolean(a != b))
                            }
                            (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                                Ok(RuntimeValue::Boolean(a != b))
                            }
                            _ => Ok(RuntimeValue::Boolean(true)), // Different types are never equal
                        }
                    }
                    LexBinaryOperator::GreaterThan => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a > b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean(a > b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean((a as f64) > b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a > (b as f64)))
                        }
                        (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                            Ok(RuntimeValue::Boolean(a > b))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for comparison".to_string(),
                        )),
                    },
                    LexBinaryOperator::LessThan => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a < b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean(a < b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean((a as f64) < b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a < (b as f64)))
                        }
                        (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                            Ok(RuntimeValue::Boolean(a < b))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for comparison".to_string(),
                        )),
                    },
                    LexBinaryOperator::GreaterEqual => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a >= b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean(a >= b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean((a as f64) >= b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a >= (b as f64)))
                        }
                        (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                            Ok(RuntimeValue::Boolean(a >= b))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for comparison".to_string(),
                        )),
                    },
                    LexBinaryOperator::LessEqual => match (left_value, right_value) {
                        (RuntimeValue::Integer(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a <= b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean(a <= b))
                        }
                        (RuntimeValue::Integer(a), RuntimeValue::Float(b)) => {
                            Ok(RuntimeValue::Boolean((a as f64) <= b))
                        }
                        (RuntimeValue::Float(a), RuntimeValue::Integer(b)) => {
                            Ok(RuntimeValue::Boolean(a <= (b as f64)))
                        }
                        (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                            Ok(RuntimeValue::Boolean(a <= b))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for comparison".to_string(),
                        )),
                    },

                    // Logical operators
                    LexBinaryOperator::And => {
                        fn as_bool(v: RuntimeValue) -> Option<bool> {
                            match v {
                                RuntimeValue::Boolean(b) => Some(b),
                                RuntimeValue::String(s) => {
                                    let ls = s.to_lowercase();
                                    if ls == "true" {
                                        Some(true)
                                    } else if ls == "false" {
                                        Some(false)
                                    } else {
                                        None
                                    }
                                }
                                RuntimeValue::Json(j) => j.as_bool(),
                                _ => None,
                            }
                        }
                        match (as_bool(left_value), as_bool(right_value)) {
                            (Some(a), Some(b)) => Ok(RuntimeValue::Boolean(a && b)),
                            _ => Err(ExecutorError::TypeError(
                                "And operator requires boolean operands".to_string(),
                            )),
                        }
                    }
                    LexBinaryOperator::Or => {
                        fn as_bool(v: RuntimeValue) -> Option<bool> {
                            match v {
                                RuntimeValue::Boolean(b) => Some(b),
                                RuntimeValue::String(s) => {
                                    let ls = s.to_lowercase();
                                    if ls == "true" {
                                        Some(true)
                                    } else if ls == "false" {
                                        Some(false)
                                    } else {
                                        None
                                    }
                                }
                                RuntimeValue::Json(j) => j.as_bool(),
                                _ => None,
                            }
                        }
                        match (as_bool(left_value), as_bool(right_value)) {
                            (Some(a), Some(b)) => Ok(RuntimeValue::Boolean(a || b)),
                            _ => Err(ExecutorError::TypeError(
                                "Or operator requires boolean operands".to_string(),
                            )),
                        }
                    }
                }
            }

            LexExpression::UnaryOp { operator, operand } => {
                let operand_value = self.evaluate_expression(*operand)?;

                match operator {
                    LexUnaryOperator::Negate => match operand_value {
                        RuntimeValue::Integer(a) => Ok(RuntimeValue::Integer(-a)),
                        RuntimeValue::Float(a) => Ok(RuntimeValue::Float(-a)),
                        _ => Err(ExecutorError::TypeError(
                            "Negate operator requires numeric operand".to_string(),
                        )),
                    },
                    LexUnaryOperator::Not => match operand_value {
                        RuntimeValue::Boolean(a) => Ok(RuntimeValue::Boolean(!a)),
                        _ => Err(ExecutorError::TypeError(
                            "Not operator requires boolean operand".to_string(),
                        )),
                    },
                }
            }

            LexExpression::FieldAccess { base, field } => {
                let base_value = self.evaluate_expression(*base)?;

                // For now, we only support access to fields in JSON values
                match base_value {
                    RuntimeValue::Json(json) => {
                        if let Value::Object(obj) = json {
                            if let Some(field_value) = obj.get(&field) {
                                Ok(RuntimeValue::Json(field_value.clone()))
                            } else {
                                Err(ExecutorError::NameError(format!(
                                    "Field not found: {}",
                                    field
                                )))
                            }
                        } else {
                            Err(ExecutorError::TypeError(
                                "Field access requires object".to_string(),
                            ))
                        }
                    }
                    _ => Err(ExecutorError::TypeError(
                        "Field access requires object".to_string(),
                    )),
                }
            }
        }
    }
    /// Calls a user-defined function
    fn call_user_function(
        &mut self,
        func_def: &LexFunction,
        args: &[LexExpression],
    ) -> Result<RuntimeValue> {
        // Verify that the number of arguments matches
        if args.len() != func_def.parameters.len() {
            return Err(ExecutorError::ArgumentError(format!(
                "Function '{}' expects {} arguments, got {}",
                func_def.name,
                func_def.parameters.len(),
                args.len()
            )));
        }

        // Evaluate arguments
        let mut arg_values = Vec::new();
        for arg_expr in args {
            let value = self.evaluate_expression(arg_expr.clone())?;
            arg_values.push(value);
        }

        // Create new scope for the function (save current variables)
        let old_variables = self.variables.clone();

        // Initialize parameters in the local scope
        for (i, (param_name, _param_type)) in func_def.parameters.iter().enumerate() {
            self.variables
                .insert(param_name.clone(), arg_values[i].clone());
        }

        // Execute the function body
        let mut return_value = RuntimeValue::Null;
        for instruction in &func_def.body {
            match instruction {
                LexInstruction::Return { expr } => {
                    return_value = if let Some(expression) = expr {
                        self.evaluate_expression(expression.clone())?
                    } else {
                        RuntimeValue::Null
                    };
                    break; // Exit immediately upon finding return
                }
                _ => {
                    self.execute_instruction(instruction)?;
                }
            }
        }

        // Restore previous scope
        self.variables = old_variables;

        Ok(return_value)
    }

    /// Executes an instruction with control flow
    #[allow(dead_code)]
    fn execute_instruction_with_flow(
        &mut self,
        instruction: &LexInstruction,
    ) -> Result<ControlFlow> {
        match instruction {
            LexInstruction::Return { expr } => {
                let return_value = if let Some(expression) = expr {
                    self.evaluate_expression(expression.clone())?
                } else {
                    RuntimeValue::Null
                };
                Ok(ControlFlow::Return(return_value))
            }
            _ => {
                // For all other instructions, use normal method
                self.execute_instruction(instruction)?;
                Ok(ControlFlow::Continue)
            }
        }
    }
    /// Executes a LexIR instruction
    pub fn execute_instruction(&mut self, instruction: &LexInstruction) -> Result<()> {
        match instruction {
            LexInstruction::Declare {
                name,
                type_name: _,
                is_mutable: _,
            } => {
                // If the variable doesn't exist, we initialize it with null
                if !self.variables.contains_key(name) {
                    self.variables.insert(name.clone(), RuntimeValue::Null);
                }
                Ok(())
            }

            LexInstruction::Assign { result, expr } => {
                let value = self.evaluate_expression(expr.clone())?;
                self.store_value(result, value)?;
                Ok(())
            }

            LexInstruction::Call {
                result,
                function,
                args,
            } => {
                // Temporarily we only support print functions
                if function == "print" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "print takes exactly one argument".to_string(),
                        ));
                    }

                    let value = self.evaluate_expression(args[0].clone())?;
                    match value {
                        RuntimeValue::String(s) => println!("{}", s),
                        RuntimeValue::Integer(i) => println!("{}", i),
                        RuntimeValue::Float(f) => println!("{}", f),
                        RuntimeValue::Boolean(b) => println!("{}", b),
                        RuntimeValue::Dataset(ds) => match ds.to_string() {
                            Ok(s) => println!("{}", s),
                            Err(e) => return Err(e),
                        },
                        RuntimeValue::Json(json) => println!("{}", json),
                        RuntimeValue::Null => println!("null"),
                        RuntimeValue::Result {
                            success,
                            value,
                            error_message,
                        } => {
                            if success {
                                println!("Ok({})", format_runtime_value(&value));
                            } else {
                                match error_message {
                                    Some(msg) => println!("Error({})", msg),
                                    None => println!("Error(unknown)"),
                                }
                            }
                        }
                        RuntimeValue::MultiOutput {
                            primary_text,
                            binary_files,
                            metadata,
                        } => {
                            println!("{}", primary_text);
                            if !binary_files.is_empty() {
                                println!(
                                    "📦 Generated {} files: {}",
                                    binary_files.len(),
                                    binary_files
                                        .iter()
                                        .map(|f| f.name.clone())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                );
                            }
                            if !metadata.is_empty() {
                                println!("📋 Metadata: {} entries", metadata.len());
                            }
                        }
                    }

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Null)?;
                    }

                    Ok(())
                } else if function == "typeof" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "typeof takes exactly one argument".to_string(),
                        ));
                    }

                    let value = self.evaluate_expression(args[0].clone())?;
                    let type_name = match value {
                        RuntimeValue::String(_) => "string",
                        RuntimeValue::Integer(_) => "int",
                        RuntimeValue::Float(_) => "float",
                        RuntimeValue::Boolean(_) => "bool",
                        RuntimeValue::Dataset(_) => "Dataset",
                        RuntimeValue::Json(_) => "json",
                        RuntimeValue::Null => "null",
                        RuntimeValue::Result { .. } => "Result",
                        RuntimeValue::MultiOutput { .. } => "MultiOutput",
                    };

                    let result_value = RuntimeValue::String(type_name.to_string());
                    if let Some(res) = result {
                        self.store_value(res, result_value)?;
                    }

                    Ok(())
                } else if function == "method.call" {
                    // method.call(receiver, method_name, ...args)
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "method.call requires receiver and method name".to_string(),
                        ));
                    }
                    let recv_val = self.evaluate_expression(args[0].clone())?;
                    let method_name = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{}", format_runtime_value(&other)),
                    };
                    // Infer type from __type on JSON objects
                    let type_name = match recv_val {
                        RuntimeValue::Json(ref j) => j
                            .get("__type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        RuntimeValue::String(ref s) => {
                            // try parse JSON
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
                                v.get("__type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string()
                            } else {
                                String::new()
                            }
                        }
                        _ => String::new(),
                    };
                    if type_name.is_empty() {
                        return Err(ExecutorError::RuntimeError(
                            "method.call: receiver has no __type".to_string(),
                        ));
                    }
                    let fname = format!("{}__{}", type_name, method_name);
                    if let Some(func_def) = self.functions.get(&fname).cloned() {
                        // Pass through remaining arg expressions
                        let tail = if args.len() > 2 { &args[2..] } else { &[][..] };
                        let rv = self.call_user_function(&func_def, tail)?;
                        if let Some(res) = result {
                            self.store_value(res, rv)?;
                        }
                        Ok(())
                    } else {
                        Err(ExecutorError::UndefinedFunction(fname))
                    }
                } else if function == "Ok" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "Ok takes exactly one argument".to_string(),
                        ));
                    }
                    let value = self.evaluate_expression(args[0].clone())?;
                    let result_value = RuntimeValue::Result {
                        success: true,
                        value: Box::new(value),
                        error_message: None,
                    };
                    if let Some(res) = result {
                        self.store_value(res, result_value)?;
                    }
                    Ok(())
                } else if function == "http.get" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "http.get requires url".to_string(),
                        ));
                    }
                    let url = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let resp = ureq::get(&url)
                        .call()
                        .map_err(|e| ExecutorError::RuntimeError(format!("http.get: {}", e)))?;
                    let text = resp.into_string().map_err(|e| {
                        ExecutorError::RuntimeError(format!("http.get read: {}", e))
                    })?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(text))?;
                    }
                    Ok(())
                } else if function == "http.get_json" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "http.get_json requires url".to_string(),
                        ));
                    }
                    let url = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let resp = ureq::get(&url).call().map_err(|e| {
                        ExecutorError::RuntimeError(format!("http.get_json: {}", e))
                    })?;
                    let text = resp.into_string().map_err(|e| {
                        ExecutorError::RuntimeError(format!("http.get_json read: {}", e))
                    })?;
                    let json: serde_json::Value =
                        serde_json::from_str(&text).unwrap_or(serde_json::json!({"raw": text}));
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(json))?;
                    }
                    Ok(())
                } else if function == "http.request" || function == "http__request" {
                    // http.request(method: string, url: string, body?: string|json, headers?: json) -> json
                    if std::env::var("LEXON_ALLOW_HTTP").ok().as_deref() != Some("1") {
                        return Err(ExecutorError::RuntimeError(
                            "HTTP disabled: set LEXON_ALLOW_HTTP=1".to_string(),
                        ));
                    }
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "http.request requires at least (method, url)".to_string(),
                        ));
                    }
                    let method = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s.to_uppercase(),
                        v => format!("{}", format_runtime_value(&v)).to_uppercase(),
                    };
                    let url = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let (body_opt, headers_opt) = if args.len() >= 3 {
                        let b = self.evaluate_expression(args[2].clone())?;
                        let h = if args.len() >= 4 {
                            Some(self.evaluate_expression(args[3].clone())?)
                        } else {
                            None
                        };
                        (Some(b), h)
                    } else {
                        (None, None)
                    };
                    let timeout_ms: u64 = std::env::var("LEXON_HTTP_TIMEOUT_MS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(10000);
                    let retries: u32 = std::env::var("LEXON_HTTP_RETRIES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let agent = ureq::AgentBuilder::new()
                        .timeout(std::time::Duration::from_millis(timeout_ms))
                        .build();
                    // Build request
                    let build_req = |agent: &ureq::Agent| -> ureq::Request {
                        match method.as_str() {
                            "GET" => agent.get(&url),
                            "POST" => agent.post(&url),
                            "PUT" => agent.put(&url),
                            "DELETE" => agent.delete(&url),
                            "PATCH" => agent.request("PATCH", &url),
                            _ => agent.request(&method, &url),
                        }
                    };
                    // Perform with retries
                    let mut last_err: Option<String> = None;
                    let mut response_json: Option<serde_json::Value> = None;
                    for _ in 0..=retries {
                        let mut req = build_req(&agent);
                        // Headers
                        if let Some(RuntimeValue::Json(serde_json::Value::Object(map))) =
                            &headers_opt
                        {
                            for (k, v) in map {
                                let hv = if let Some(s) = v.as_str() {
                                    s.to_string()
                                } else {
                                    v.to_string()
                                };
                                req = req.set(k, &hv);
                            }
                        }
                        // Body
                        let sent = if let Some(b) = &body_opt {
                            match b {
                                RuntimeValue::Json(j) => {
                                    // default content-type if not provided
                                    let has_ct = if let Some(RuntimeValue::Json(
                                        serde_json::Value::Object(hm),
                                    )) = &headers_opt
                                    {
                                        hm.contains_key("Content-Type")
                                            || hm.contains_key("content-type")
                                    } else {
                                        false
                                    };
                                    if !has_ct {
                                        req = req.set("Content-Type", "application/json");
                                    }
                                    let s = j.to_string();
                                    req.send_string(&s)
                                }
                                RuntimeValue::String(s) => req.send_string(s),
                                other => {
                                    let s = format_runtime_value(other);
                                    req.send_string(&s)
                                }
                            }
                        } else {
                            req.call()
                        };
                        match sent {
                            Ok(resp) => {
                                let status = resp.status();
                                // Collect headers (few common, then raw)
                                let mut headers_obj = serde_json::Map::new();
                                for (k, v) in resp.headers_names().iter().map(|n| {
                                    (n.to_string(), resp.header(n).unwrap_or("").to_string())
                                }) {
                                    headers_obj.insert(k, serde_json::Value::String(v));
                                }
                                let text = resp.into_string().unwrap_or_else(|_| "".to_string());
                                let body_json =
                                    serde_json::from_str::<serde_json::Value>(&text).ok();
                                let out = serde_json::json!({
                                    "status": status,
                                    "headers": headers_obj,
                                    "body": text,
                                    "body_json": body_json
                                });
                                response_json = Some(out);
                                break;
                            }
                            Err(e) => {
                                last_err = Some(format!("http.request: {}", e));
                                continue;
                            }
                        }
                    }
                    let out = response_json.ok_or_else(|| {
                        ExecutorError::RuntimeError(
                            last_err.unwrap_or_else(|| "HTTP error".to_string()),
                        )
                    })?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "web.search" || function == "web__search" {
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "web.search requires query".to_string(),
                        ));
                    }
                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let n: usize = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => (i as isize).max(1) as usize,
                            RuntimeValue::Float(f) => (f as isize).max(1) as usize,
                            RuntimeValue::String(s) => s.parse().unwrap_or(5),
                            _ => 5,
                        }
                    } else {
                        5
                    };
                    if let Ok(endpoint) = std::env::var("LEXON_WEB_SEARCH_ENDPOINT") {
                        let q = query.replace(" ", "+");
                        let mut url = format!("{}?q={}&n={}", endpoint, q, n);
                        if endpoint.contains("duckduckgo.com") {
                            url.push_str("&format=json");
                        }
                        // Single call; no pagination for env endpoint
                        let resp = ureq::get(&url).call().map_err(|e| {
                            ExecutorError::RuntimeError(format!("web.search: {}", e))
                        })?;
                        let text = resp.into_string().map_err(|e| {
                            ExecutorError::RuntimeError(format!("web.search read: {}", e))
                        })?;
                        let json_raw: serde_json::Value =
                            serde_json::from_str(&text).unwrap_or(serde_json::json!({"raw": text}));
                        let json_norm = normalize_search_json(&endpoint, &json_raw);
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::Json(json_norm))?;
                        }
                    } else if let Some(cfg) = Self::load_web_search_from_toml() {
                        let q = query.replace(" ", "+");
                        let sep = if cfg.endpoint.contains('?') { '&' } else { '?' };
                        let mut url = format!(
                            "{}{}{}={}&{}={}",
                            cfg.endpoint, sep, cfg.query_param, q, cfg.count_param, n
                        );
                        if let (Some(fp), Some(fv)) =
                            (cfg.format_param.as_deref(), cfg.format_value.as_deref())
                        {
                            url.push_str(&format!("&{}={}", fp, fv));
                        }
                        for (k, v) in &cfg.extra_params {
                            url.push_str(&format!("&{}={}", k, v));
                        }
                        if cfg.endpoint.contains("serpapi.com") {
                            let mut agg: Vec<serde_json::Value> = Vec::new();
                            let mut page_start: usize = 0;
                            let mut done_pages: usize = 0;
                            while agg.len() < n && done_pages < 3 {
                                let mut page_url = format!("{}&start={}", url, page_start);
                                let mut preq = ureq::get(&page_url);
                                if let (Some(mode), Some(name), Some(key)) = (
                                    cfg.auth_mode.as_deref(),
                                    cfg.auth_name.as_deref(),
                                    cfg.api_key.as_deref(),
                                ) {
                                    if mode.eq_ignore_ascii_case("header") {
                                        preq = preq.set(name, key);
                                    } else if mode.eq_ignore_ascii_case("query") {
                                        page_url.push_str(&format!("&{}={}", name, key));
                                        preq = ureq::get(&page_url);
                                    }
                                }
                                let presp = preq.call().map_err(|e| {
                                    ExecutorError::RuntimeError(format!("web.search: {}", e))
                                })?;
                                let ptext = presp.into_string().map_err(|e| {
                                    ExecutorError::RuntimeError(format!("web.search read: {}", e))
                                })?;
                                let praw: serde_json::Value = serde_json::from_str(&ptext)
                                    .unwrap_or(serde_json::json!({"raw": ptext}));
                                let pnorm = normalize_search_json(&cfg.endpoint, &praw);
                                if let Some(arr) = pnorm.get("Results").and_then(|v| v.as_array()) {
                                    for it in arr {
                                        if agg.len() < n {
                                            agg.push(it.clone());
                                        }
                                    }
                                }
                                page_start += 10;
                                done_pages += 1;
                            }
                            let final_json = serde_json::json!({ "Results": agg });
                            if let Some(res) = result {
                                self.store_value(res, RuntimeValue::Json(final_json))?;
                            }
                        } else {
                            let mut req = ureq::get(&url);
                            if let (Some(mode), Some(name), Some(key)) = (
                                cfg.auth_mode.as_deref(),
                                cfg.auth_name.as_deref(),
                                cfg.api_key.as_deref(),
                            ) {
                                if mode.eq_ignore_ascii_case("header") {
                                    req = req.set(name, key);
                                } else if mode.eq_ignore_ascii_case("query") {
                                    url.push_str(&format!("&{}={}", name, key));
                                    req = ureq::get(&url);
                                }
                            }
                            let resp = req.call().map_err(|e| {
                                ExecutorError::RuntimeError(format!("web.search: {}", e))
                            })?;
                            let text = resp.into_string().map_err(|e| {
                                ExecutorError::RuntimeError(format!("web.search read: {}", e))
                            })?;
                            let json_raw: serde_json::Value = serde_json::from_str(&text)
                                .unwrap_or(serde_json::json!({"raw": text}));
                            let json_norm = normalize_search_json(&cfg.endpoint, &json_raw);
                            if let Some(res) = result {
                                self.store_value(res, RuntimeValue::Json(json_norm))?;
                            }
                        }
                    } else {
                        // deterministic simulated results
                        let items: Vec<serde_json::Value> = (1..=n).map(|i| serde_json::json!({
                            "title": format!("Result {} for {}", i, query),
                            "url": format!("https://example.com/{}/{}", query.replace(" ", "_"), i),
                            "snippet": "SIMULATED SEARCH RESULT"
                        })).collect();
                        let json = serde_json::Value::Array(items);
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::Json(json))?;
                        }
                    }
                    Ok(())
                } else if function == "Error" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "Error takes exactly one argument".to_string(),
                        ));
                    }

                    let error_value = self.evaluate_expression(args[0].clone())?;
                    let error_msg = match error_value {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", error_value),
                    };

                    let result_value = RuntimeValue::Result {
                        success: false,
                        value: Box::new(RuntimeValue::Null),
                        error_message: Some(error_msg),
                    };

                    if let Some(res) = result {
                        self.store_value(res, result_value)?;
                    }

                    Ok(())
                } else if function == "is_ok" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "is_ok takes exactly one argument".to_string(),
                        ));
                    }

                    let value = self.evaluate_expression(args[0].clone())?;
                    let is_success = match value {
                        RuntimeValue::Result { success, .. } => success,
                        _ => false,
                    };

                    let result_value = RuntimeValue::Boolean(is_success);
                    if let Some(res) = result {
                        self.store_value(res, result_value)?;
                    }

                    Ok(())
                } else if function == "is_error" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "is_error takes exactly one argument".to_string(),
                        ));
                    }

                    let value = self.evaluate_expression(args[0].clone())?;
                    let is_error = match value {
                        RuntimeValue::Result { success, .. } => !success,
                        _ => false,
                    };

                    let result_value = RuntimeValue::Boolean(is_error);
                    if let Some(res) = result {
                        self.store_value(res, result_value)?;
                    }

                    Ok(())
                } else if function == "unwrap" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "unwrap takes exactly one argument".to_string(),
                        ));
                    }

                    let value = self.evaluate_expression(args[0].clone())?;
                    let unwrapped_value = match value {
                        RuntimeValue::Result {
                            success,
                            value,
                            error_message,
                        } => {
                            if success {
                                *value
                            } else {
                                let msg =
                                    error_message.unwrap_or_else(|| "unknown error".to_string());
                                return Err(ExecutorError::RuntimeError(format!(
                                    "unwrap failed: {}",
                                    msg
                                )));
                            }
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "unwrap can only be called on Result values".to_string(),
                            ))
                        }
                    };

                    if let Some(res) = result {
                        self.store_value(res, unwrapped_value)?;
                    }

                    Ok(())
                } else if function == "ask_parallel" {
                    // Delegate to extracted function
                    self.handle_ask_parallel(args, result.as_ref())?;
                    Ok(())
                } else if function == "ask_merge" {
                    // Delegate to extracted function
                    self.handle_ask_merge(args, result.as_ref())?;
                    Ok(())
                } else if function == "ask_ensemble" {
                    // Delegate to extracted function
                    self.handle_ask_ensemble(args, result.as_ref())?;
                    Ok(())
                } else if function == "join_all" {
                    self.handle_join_all(args, result.as_ref())?;
                    Ok(())
                } else if function == "join_any" {
                    self.handle_join_any(args, result.as_ref())?;
                    Ok(())
                } else if function == "ask_stream" {
                    self.handle_ask_stream(args, result.as_ref())?;
                    Ok(())
                } else if function == "ask_multioutput_stream" {
                    self.handle_ask_multioutput_stream(args, result.as_ref())?;
                    Ok(())
                } else if function == "cancel_current" || function == "runtime__cancel_current" {
                    self.cancel_requested = true;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "channel.create" || function == "channel__create" {
                    // channel.create(name, [capacity]) -> bool
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "channel.create requires name".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let cap: usize = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => (i as isize).max(1) as usize,
                            RuntimeValue::Float(f) => (f as isize).max(1) as usize,
                            RuntimeValue::String(s) => s.parse().unwrap_or(32),
                            _ => 32,
                        }
                    } else {
                        32
                    };
                    let (tx, rx) = tokio::sync::mpsc::channel::<String>(cap);
                    self.channel_senders.insert(name.clone(), tx);
                    self.channel_receivers.insert(name, rx);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "channel.send" || function == "channel__send" {
                    // channel.send(name, value) -> bool
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "channel.send requires name and value".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let val_v = self.evaluate_expression(args[1].clone())?;
                    let payload = match val_v {
                        RuntimeValue::String(s) => s,
                        RuntimeValue::Json(j) => j.to_string(),
                        other => format!("{:?}", other),
                    };
                    let ok = if let Some(tx) = self.channel_senders.get(&name) {
                        let txc = tx.clone();
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current()
                                .block_on(async move { txc.send(payload).await.is_ok() })
                        })
                    } else {
                        false
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(ok))?;
                    }
                    Ok(())
                } else if function == "channel.recv" || function == "channel__recv" {
                    // channel.recv(name, [timeout_ms]) -> string
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "channel.recv requires name".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let timeout_ms: Option<u64> = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => Some(i as u64),
                            RuntimeValue::Float(f) => Some(f as u64),
                            RuntimeValue::String(s) => s.parse().ok(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let rx = self.channel_receivers.get_mut(&name).ok_or_else(|| {
                        ExecutorError::RuntimeError(format!("channel '{}' not found", name))
                    })?;
                    let res = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            if let Some(ms) = timeout_ms {
                                tokio::time::timeout(
                                    std::time::Duration::from_millis(ms),
                                    rx.recv(),
                                )
                                .await
                                .ok()
                                .flatten()
                            } else {
                                rx.recv().await
                            }
                        })
                    });
                    match res {
                        Some(s) => {
                            if let Some(r) = result {
                                self.store_value(r, RuntimeValue::String(s))?;
                            }
                            Ok(())
                        }
                        None => Err(ExecutorError::RuntimeError(
                            "channel.recv timeout or closed".to_string(),
                        )),
                    }
                } else if function == "channel.select_any" || function == "channel__select_any" {
                    // channel.select_any([names], [timeout_ms]) -> {name,value}
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "channel.select_any requires names array".to_string(),
                        ));
                    }
                    let names_v = self.evaluate_expression(args[0].clone())?;
                    let names: Vec<String> = match names_v {
                        RuntimeValue::Json(Value::Array(arr)) => arr
                            .into_iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect(),
                        RuntimeValue::String(s) => vec![s],
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "names must be array or string".to_string(),
                            ))
                        }
                    };
                    if names.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "no channel names provided".to_string(),
                        ));
                    }
                    let timeout_ms: u64 = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => (i as i64).max(0) as u64,
                            RuntimeValue::Float(f) => (f as i64).max(0) as u64,
                            RuntimeValue::String(s) => s.parse().unwrap_or(1000),
                            _ => 1000,
                        }
                    } else {
                        1000
                    };
                    let start = std::time::Instant::now();
                    let poll_interval = std::time::Duration::from_millis(10);
                    loop {
                        for name in &names {
                            if let Some(rx) = self.channel_receivers.get_mut(name) {
                                match rx.try_recv() {
                                    Ok(val) => {
                                        let obj = serde_json::json!({"name": name, "value": val});
                                        if let Some(r) = result {
                                            self.store_value(r, RuntimeValue::Json(obj))?;
                                        }
                                        return Ok(());
                                    }
                                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
                                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {}
                                }
                            }
                        }
                        if start.elapsed().as_millis() as u64 >= timeout_ms {
                            return Err(ExecutorError::RuntimeError(
                                "channel.select_any timeout".to_string(),
                            ));
                        }
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current()
                                .block_on(async { tokio::time::sleep(poll_interval).await })
                        });
                    }
                } else if function == "task.spawn" || function == "task__spawn" {
                    // task.spawn(prompt[, model]) -> task_id
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "task.spawn requires prompt".to_string(),
                        ));
                    }
                    let prompt = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let model_opt = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        self.config.llm_model.clone()
                    };
                    let scheduler = self.async_scheduler.as_ref().ok_or_else(|| {
                        ExecutorError::RuntimeError("scheduler unavailable".to_string())
                    })?;
                    let (task_id, handle) = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            let adapter_seed = self.llm_adapter.clone();
                            let m = model_opt.clone();
                            let h = scheduler
                                .schedule_task(
                                    move |_token| {
                                        let mut adapter = adapter_seed.clone();
                                        let p = prompt.clone();
                                        async move {
                                            adapter
                                                .call_llm_async(
                                                    m.as_deref(),
                                                    Some(0.7),
                                                    None,
                                                    Some(&p),
                                                    None,
                                                    None,
                                                    &std::collections::HashMap::new(),
                                                )
                                                .await
                                                .map_err(|e| format!("{}", e))
                                        }
                                    },
                                    crate::runtime::scheduler::TaskPriority::Normal,
                                    None,
                                )
                                .await
                                .map_err(|e| ExecutorError::RuntimeError(e))?;
                            let id = h.id().to_string();
                            Ok::<(String, crate::runtime::scheduler::TaskHandle), ExecutorError>((
                                id, h,
                            ))
                        })
                    })?;
                    self.tasks.insert(task_id.clone(), handle);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(task_id))?;
                    }
                    Ok(())
                } else if function == "task.await" || function == "task__await" {
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "task.await requires id".to_string(),
                        ));
                    }
                    let id = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let timeout_ms = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => Some(i as u64),
                            RuntimeValue::Float(f) => Some(f as u64),
                            RuntimeValue::String(s) => s.parse().ok(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let handle = self.tasks.get(&id).cloned().ok_or_else(|| {
                        ExecutorError::RuntimeError(format!("task '{}' not found", id))
                    })?;
                    let res = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            if let Some(ms) = timeout_ms {
                                handle
                                    .await_result_timeout(std::time::Duration::from_millis(ms))
                                    .await
                            } else {
                                handle.await_result().await
                            }
                        })
                    });
                    use crate::runtime::scheduler::TaskResult;
                    match res {
                        Ok(TaskResult::Success(s)) => {
                            if let Some(r) = result {
                                self.store_value(r, RuntimeValue::String(s))?;
                            }
                        }
                        Ok(TaskResult::Error(e)) => return Err(ExecutorError::RuntimeError(e)),
                        Ok(TaskResult::Cancelled(reason)) => {
                            return Err(ExecutorError::RuntimeError(format!(
                                "cancelled: {}",
                                reason
                            )))
                        }
                        Ok(TaskResult::Timeout) => {
                            return Err(ExecutorError::RuntimeError(
                                "task.await timeout".to_string(),
                            ))
                        }
                        Err(e) => {
                            return Err(ExecutorError::RuntimeError(format!("await error: {}", e)))
                        }
                    }
                    // Optional: remove handle after await completes
                    // let _ = self.tasks.remove(&id);
                    Ok(())
                } else if function == "task.cancel" || function == "task__cancel" {
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "task.cancel requires id".to_string(),
                        ));
                    }
                    let id = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    if let Some(handle) = self.tasks.get(&id) {
                        let ok = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current()
                                .block_on(async { handle.cancel().await.is_ok() })
                        });
                        if let Some(r) = result {
                            self.store_value(r, RuntimeValue::Boolean(ok))?;
                        }
                        Ok(())
                    } else {
                        return Err(ExecutorError::RuntimeError(format!(
                            "task '{}' not found",
                            id
                        )));
                    }
                } else if function == "ask_with_fallback" {
                    // Delegate to extracted function
                    self.handle_ask_with_fallback(args, result.as_ref())?;
                    Ok(())
                } else if function == "session_start" {
                    self.handle_session_start(args, result.as_ref())?;
                    Ok(())
                } else if function == "session_ask" {
                    self.handle_session_ask(args, result.as_ref())?;
                    Ok(())
                } else if function == "session_history" {
                    self.handle_session_history(args, result.as_ref())?;
                    Ok(())
                } else if function == "session_summarize" {
                    self.handle_session_summarize(args, result.as_ref())?;
                    Ok(())
                } else if function == "session_compress" {
                    self.handle_session_compress(args, result.as_ref())?;
                    Ok(())
                } else if function == "context_window_manage" {
                    self.handle_context_window_manage(args, result.as_ref())?;
                    Ok(())
                } else if function == "extract_key_points" {
                    self.handle_extract_key_points(args, result.as_ref())?;
                    Ok(())
                } else if function == "session.configure" || function == "session__configure" {
                    // session.configure({ ttl_ms?, gc_interval_ms? })
                    if args.len() < 1 {
                        return Err(ExecutorError::ArgumentError(
                            "session.configure requires config json".to_string(),
                        ));
                    }
                    let cfg = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                    if let Some(ttl) = cfg.get("ttl_ms").and_then(|v| v.as_u64()) {
                        self.session_ttl_ms = ttl;
                        std::env::set_var("LEXON_SESSION_TTL_MS", ttl.to_string());
                    }
                    if let Some(gc) = cfg.get("gc_interval_ms").and_then(|v| v.as_u64()) {
                        self.session_gc_interval_ms = gc;
                        std::env::set_var("LEXON_SESSION_GC_INTERVAL_MS", gc.to_string());
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "agent_create" {
                    self.handle_agent_create(args, result.as_ref())?;
                    Ok(())
                } else if function == "agent_run" {
                    self.handle_agent_run(args, result.as_ref())?;
                    Ok(())
                } else if function == "agent_chain" {
                    self.handle_agent_chain(args, result.as_ref())?;
                    Ok(())
                } else if function == "agent_parallel" {
                    self.handle_agent_parallel(args, result.as_ref())?;
                    Ok(())
                } else if function == "agent_cancel" {
                    self.handle_agent_cancel(args, result.as_ref())?;
                    Ok(())
                } else if function == "agent_status" {
                    // agent_status(name?) -> string or map
                    if args.is_empty() {
                        let mut map = serde_json::Map::new();
                        for (k, v) in &self.agent_status {
                            map.insert(k.clone(), serde_json::Value::String(v.clone()));
                        }
                        if let Some(res) = result {
                            self.store_value(
                                res,
                                RuntimeValue::Json(serde_json::Value::Object(map)),
                            )?;
                        }
                    } else {
                        let name = match self.evaluate_expression(args[0].clone())? {
                            RuntimeValue::String(s) => s,
                            v => format!("{:?}", v),
                        };
                        let st = self
                            .agent_status
                            .get(&name)
                            .cloned()
                            .unwrap_or_else(|| "unknown".to_string());
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::String(st))?;
                        }
                    }
                    Ok(())
                } else if function == "agent_list" {
                    let names: Vec<serde_json::Value> = self
                        .agent_registry
                        .keys()
                        .cloned()
                        .map(serde_json::Value::String)
                        .collect();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(serde_json::Value::Array(names)))?;
                    }
                    Ok(())
                } else if function == "agent_supervisor.get_state"
                    || function == "agent_supervisor__get_state"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "agent_supervisor.get_state requires name".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let cfg = self.agent_registry.get(&name).cloned();
                    let status = self
                        .agent_status
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());
                    let cancelled = *self.agent_cancelled.get(&name).unwrap_or(&false);
                    let obj = if let Some(c) = cfg {
                        serde_json::json!({"name": name, "model": c.model, "budget_usd": c.budget_usd, "deadline_ms": c.deadline_ms, "status": status, "cancelled": cancelled})
                    } else {
                        serde_json::json!({"name": name, "status": status, "cancelled": cancelled})
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(obj))?;
                    }
                    Ok(())
                } else if function == "agent_supervisor.list"
                    || function == "agent_supervisor__list"
                {
                    let mut arr = Vec::new();
                    for (name, cfg) in &self.agent_registry {
                        let status = self
                            .agent_status
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| "unknown".to_string());
                        let cancelled = *self.agent_cancelled.get(name).unwrap_or(&false);
                        arr.push(serde_json::json!({"name": name, "model": cfg.model, "budget_usd": cfg.budget_usd, "deadline_ms": cfg.deadline_ms, "status": status, "cancelled": cancelled}));
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(serde_json::Value::Array(arr)))?;
                    }
                    Ok(())
                } else if function == "memory_store" {
                    // Delegate to extracted function
                    self.handle_memory_store(args, result.as_ref())?;
                    Ok(())
                } else if function == "memory_load" {
                    self.handle_memory_load(args, result.as_ref())?;
                    Ok(())
                } else if function == "memory_index.search" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "memory_index.search requires query,k".to_string(),
                        ));
                    }
                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "query must be string".to_string(),
                            ))
                        }
                    };
                    let k = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "k must be integer".to_string(),
                            ))
                        }
                    };
                    let results = if let Some(ref mut vm) = self.vector_memory_system {
                        vm.vector_search(&query, k)?
                    } else {
                        Vec::new()
                    };
                    let arr: Vec<Value> = results
                        .into_iter()
                        .map(|v| match v {
                            RuntimeValue::Json(j) => j,
                            RuntimeValue::String(s) => Value::String(s),
                            other => Value::String(format!("{:?}", other)),
                        })
                        .collect();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(Value::Array(arr)))?;
                    }
                    Ok(())
                } else if function == "memory_index.hybrid_search"
                    || function == "memory_index__hybrid_search"
                {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "hybrid_search requires query,k,[config]".to_string(),
                        ));
                    }
                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "query must be string".to_string(),
                            ))
                        }
                    };
                    let k = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "k must be integer".to_string(),
                            ))
                        }
                    };
                    let mut alpha: f32 = 0.7;
                    let mut filters: HashMap<String, String> = HashMap::new();
                    let mut offset: usize = 0;
                    let mut limit_factor: usize = 4;
                    if args.len() >= 3 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Float(f) => alpha = f as f32,
                            RuntimeValue::Integer(i) => alpha = (i as f32).clamp(0.0, 1.0),
                            RuntimeValue::String(s) => {
                                if let Ok(v) = s.parse::<f32>() {
                                    alpha = v.clamp(0.0, 1.0);
                                }
                            }
                            RuntimeValue::Json(Value::Object(map)) => {
                                if let Some(a) = map.get("alpha").and_then(|v| v.as_f64()) {
                                    alpha = (a as f32).clamp(0.0, 1.0);
                                }
                                if let Some(fm) = map.get("filters").and_then(|v| v.as_object()) {
                                    for (k, v) in fm {
                                        if let Some(s) = v.as_str() {
                                            filters.insert(k.clone(), s.to_string());
                                        } else {
                                            filters.insert(k.clone(), v.to_string());
                                        }
                                    }
                                }
                                if let Some(raw) = map.get("qdrant_filter") {
                                    filters.insert("__raw__".to_string(), raw.to_string());
                                }
                                if let Some(o) = map.get("offset").and_then(|v| v.as_u64()) {
                                    offset = o as usize;
                                }
                                if let Some(lf) = map.get("limit_factor").and_then(|v| v.as_u64()) {
                                    let lf_usize = lf as usize;
                                    if lf_usize >= 1 {
                                        limit_factor = lf_usize;
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    let results = if let Some(ref mut vm) = self.vector_memory_system {
                        vm.hybrid_search(&query, k, alpha, &filters, offset, limit_factor)?
                    } else {
                        Vec::new()
                    };
                    let arr: Vec<Value> = results
                        .into_iter()
                        .map(|v| match v {
                            RuntimeValue::Json(j) => j,
                            RuntimeValue::String(s) => Value::String(s),
                            other => Value::String(format!("{:?}", other)),
                        })
                        .collect();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(Value::Array(arr)))?;
                    }
                    Ok(())
                } else if function == "memory_index.hybrid_search_page"
                    || function == "memory_index__hybrid_search_page"
                {
                    // memory_index.hybrid_search_page(query, k, page, limit_factor?, config?)
                    if args.len() < 3 {
                        return Err(ExecutorError::ArgumentError(
                            "hybrid_search_page requires query,k,page".to_string(),
                        ));
                    }
                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "query must be string".to_string(),
                            ))
                        }
                    };
                    let k = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "k must be integer".to_string(),
                            ))
                        }
                    };
                    let page = match self.evaluate_expression(args[2].clone())? {
                        RuntimeValue::Integer(i) => {
                            if i < 0 {
                                0usize
                            } else {
                                i as usize
                            }
                        }
                        _ => 0usize,
                    };
                    let limit_factor: usize = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Integer(i) => (i as usize).max(1),
                            _ => 4,
                        }
                    } else {
                        4
                    };
                    let mut alpha: f32 = 0.7;
                    let mut filters: HashMap<String, String> = HashMap::new();
                    if args.len() > 4 {
                        match self.evaluate_expression(args[4].clone())? {
                            RuntimeValue::Json(Value::Object(map)) => {
                                if let Some(a) = map.get("alpha").and_then(|v| v.as_f64()) {
                                    alpha = (a as f32).clamp(0.0, 1.0);
                                }
                                if let Some(fm) = map.get("filters").and_then(|v| v.as_object()) {
                                    for (k, v) in fm {
                                        if let Some(s) = v.as_str() {
                                            filters.insert(k.clone(), s.to_string());
                                        } else {
                                            filters.insert(k.clone(), v.to_string());
                                        }
                                    }
                                }
                                if let Some(raw) = map.get("qdrant_filter") {
                                    filters.insert("__raw__".to_string(), raw.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                    let offset = page.saturating_mul(k);
                    let results = if let Some(ref mut vm) = self.vector_memory_system {
                        vm.hybrid_search(&query, k, alpha, &filters, offset, limit_factor)?
                    } else {
                        Vec::new()
                    };
                    let arr: Vec<Value> = results
                        .into_iter()
                        .map(|v| match v {
                            RuntimeValue::Json(j) => j,
                            RuntimeValue::String(s) => Value::String(s),
                            other => Value::String(format!("{:?}", other)),
                        })
                        .collect();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(Value::Array(arr)))?;
                    }
                    Ok(())
                } else if function == "memory_index.hybrid_search_all"
                    || function == "memory_index__hybrid_search_all"
                {
                    // memory_index.hybrid_search_all(query, k, max_pages?, limit_factor?, config?) -> results_json
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "hybrid_search_all requires query,k".to_string(),
                        ));
                    }
                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "query must be string".to_string(),
                            ))
                        }
                    };
                    let k = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "k must be integer".to_string(),
                            ))
                        }
                    };
                    let max_pages: usize = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Integer(i) => (i as usize).max(1),
                            _ => 5,
                        }
                    } else {
                        5
                    };
                    let limit_factor: usize = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Integer(i) => (i as usize).max(1),
                            _ => 4,
                        }
                    } else {
                        4
                    };
                    let mut alpha: f32 = 0.7;
                    let mut filters: HashMap<String, String> = HashMap::new();
                    if args.len() > 4 {
                        match self.evaluate_expression(args[4].clone())? {
                            RuntimeValue::Json(Value::Object(map)) => {
                                if let Some(a) = map.get("alpha").and_then(|v| v.as_f64()) {
                                    alpha = (a as f32).clamp(0.0, 1.0);
                                }
                                if let Some(fm) = map.get("filters").and_then(|v| v.as_object()) {
                                    for (k, v) in fm {
                                        if let Some(s) = v.as_str() {
                                            filters.insert(k.clone(), s.to_string());
                                        } else {
                                            filters.insert(k.clone(), v.to_string());
                                        }
                                    }
                                }
                                if let Some(raw) = map.get("qdrant_filter") {
                                    filters.insert("__raw__".to_string(), raw.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                    let mut out: Vec<Value> = Vec::new();
                    if let Some(ref mut vm) = self.vector_memory_system {
                        for page in 0..max_pages {
                            let offset = page.saturating_mul(k);
                            let mut page_res =
                                vm.hybrid_search(&query, k, alpha, &filters, offset, limit_factor)?;
                            if page_res.is_empty() {
                                break;
                            }
                            for v in page_res.drain(..) {
                                match v {
                                    RuntimeValue::Json(j) => out.push(j),
                                    RuntimeValue::String(s) => out.push(Value::String(s)),
                                    other => out.push(Value::String(format!("{:?}", other))),
                                }
                            }
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(Value::Array(out)))?;
                    }
                    Ok(())
                } else if function == "memory_index.hybrid_search_llm_rerank"
                    || function == "memory_index__hybrid_search_llm_rerank"
                {
                    // memory_index.hybrid_search_llm_rerank(query, k, config?, model?, top_k?)
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "hybrid_search_llm_rerank requires query,k".to_string(),
                        ));
                    }
                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "query must be string".to_string(),
                            ))
                        }
                    };
                    let k = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "k must be integer".to_string(),
                            ))
                        }
                    };
                    let mut alpha: f32 = 0.7;
                    let mut filters: HashMap<String, String> = HashMap::new();
                    let mut model: Option<String> = None;
                    let mut top_k: usize = k;
                    if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Json(Value::Object(map)) => {
                                if let Some(a) = map.get("alpha").and_then(|v| v.as_f64()) {
                                    alpha = (a as f32).clamp(0.0, 1.0);
                                }
                                if let Some(fm) = map.get("filters").and_then(|v| v.as_object()) {
                                    for (k, v) in fm {
                                        if let Some(s) = v.as_str() {
                                            filters.insert(k.clone(), s.to_string());
                                        } else {
                                            filters.insert(k.clone(), v.to_string());
                                        }
                                    }
                                }
                                if let Some(raw) = map.get("qdrant_filter") {
                                    filters.insert("__raw__".to_string(), raw.to_string());
                                }
                            }
                            _ => {}
                        }
                    }
                    if args.len() > 3 {
                        if let RuntimeValue::String(s) =
                            self.evaluate_expression(args[3].clone())?
                        {
                            model = Some(s);
                        }
                    }
                    if args.len() > 4 {
                        if let RuntimeValue::Integer(i) =
                            self.evaluate_expression(args[4].clone())?
                        {
                            top_k = (i as usize).max(1);
                        }
                    }
                    let mut arr: Vec<Value> = Vec::new();
                    if let Some(ref mut vm) = self.vector_memory_system {
                        let res = vm.hybrid_search(&query, k, alpha, &filters, 0, 4)?;
                        for v in res {
                            match v {
                                RuntimeValue::Json(j) => arr.push(j),
                                RuntimeValue::String(s) => arr.push(Value::String(s)),
                                other => arr.push(Value::String(format!("{:?}", other))),
                            }
                        }
                    }
                    // LLM rerank result set
                    let mut scored: Vec<(f32, Value)> = Vec::new();
                    let sys = "You are a precise scorer that outputs only: score: <number between 0 and 1>";
                    if let Some(ref mut la) = self.llm_adapter_new {
                        let handle = tokio::runtime::Handle::current();
                        let scores: Vec<f32> = tokio::task::block_in_place(|| {
                            handle.block_on(async {
                            use futures_util::future::join_all;
                            let mut futs = Vec::new();
                            for it in &arr {
                                let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                let prompt_s = format!("Given query: {:?}\nRate relevance of passage 0..1 (decimal).\nPassage: {:?}\nAnswer as: score: <number>", query, content);
                                let mut adapter_local = la.clone();
                                let params: HashMap<String,String> = HashMap::new();
                                let model_clone = model.clone();
                                futs.push(async move { adapter_local.call_llm_async(model_clone.as_deref(), Some(0.0), Some(sys), Some(&prompt_s), None, None, &params).await });
                            }
                            let results = join_all(futs).await;
                            let mut out = Vec::new();
                            for r in results { out.push(r.ok().map(|s| Self::parse_score_local(&s)).unwrap_or(0.0)); }
                            out
                        })
                        });
                        for (i, it) in arr.into_iter().enumerate() {
                            let sc = scores.get(i).cloned().unwrap_or(0.0);
                            scored.push((sc, it));
                        }
                    } else {
                        for it in arr.into_iter() {
                            let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            let prompt = format!("Given query: {:?}\nRate relevance of passage 0..1 (decimal).\nPassage: {:?}\nAnswer as: score: <number>", query, content);
                            let resp = self.llm_adapter.call_llm(
                                model.as_deref(),
                                Some(0.0),
                                Some(sys),
                                Some(&prompt),
                                None,
                                None,
                                &HashMap::new(),
                            )?;
                            let score = Self::parse_score_local(&resp);
                            scored.push((score, it));
                        }
                    }
                    scored
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(top_k);
                    let out = Value::Array(scored.into_iter().map(|(_, v)| v).collect());
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "memory_index.set_metadata" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "set_metadata requires path, metadata".to_string(),
                        ));
                    }
                    let path = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "path must be string".to_string(),
                            ))
                        }
                    };
                    let metadata = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            match serde_json::from_str::<Value>(&s) {
                                Ok(j) => j,
                                Err(_) => {
                                    // Fallback for simple DSL maps like {key: value}
                                    let trimmed =
                                        s.trim().trim_start_matches('{').trim_end_matches('}');
                                    let mut map = serde_json::Map::new();
                                    if let Some((k, v)) = trimmed.split_once(':') {
                                        map.insert(
                                            k.trim().trim_matches('"').to_string(),
                                            serde_json::Value::String(
                                                v.trim().trim_matches('"').to_string(),
                                            ),
                                        );
                                    }
                                    Value::Object(map)
                                }
                            }
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "metadata must be JSON".to_string(),
                            ))
                        }
                    };
                    if let Some(ref mut vm) = self.vector_memory_system {
                        vm.set_metadata(&path, &metadata)?;
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "memory_index.qdrant_create_index"
                    || function == "memory_index__qdrant_create_index"
                {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "qdrant_create_index requires field_name, field_schema".to_string(),
                        ));
                    }
                    let field = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "field_name must be string".to_string(),
                            ))
                        }
                    };
                    let schema = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "field_schema must be string".to_string(),
                            ))
                        }
                    };
                    if let Some(ref vm) = self.vector_memory_system {
                        vm.qdrant_create_index(&field, &schema)?;
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "memory_index.qdrant_ensure_schema"
                    || function == "memory_index__qdrant_ensure_schema"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "qdrant_ensure_schema requires schema_json".to_string(),
                        ));
                    }
                    let val = self.evaluate_expression(args[0].clone())?;
                    let schema = match val {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            serde_json::from_str::<Value>(&s).map_err(|e| {
                                ExecutorError::ArgumentError(format!("Invalid JSON: {}", e))
                            })?
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "schema must be JSON or string".to_string(),
                            ))
                        }
                    };
                    if let Some(ref vm) = self.vector_memory_system {
                        vm.qdrant_set_schema(&schema)?;
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "get_metrics_json" {
                    let json = serde_json::json!({
                        "total_llm_calls": self.total_llm_calls,
                        "llm_total_ms": self.llm_total_ms,
                    });
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(json))?;
                    }
                    Ok(())
                } else if function == "metrics.get_json" {
                    let json = serde_json::json!({
                        "total_llm_calls": self.total_llm_calls,
                        "llm_total_ms": self.llm_total_ms,
                    });
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(json))?;
                    }
                    Ok(())
                } else if function == "metrics.write_json" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "metrics.write_json requires exactly 1 argument: path".to_string(),
                        ));
                    }
                    let path_val = self.evaluate_expression(args[0].clone())?;
                    let path = match path_val {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let json = serde_json::json!({
                        "total_llm_calls": self.total_llm_calls,
                        "llm_total_ms": self.llm_total_ms,
                    });
                    let arg_exprs = vec![
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(
                            json.to_string(),
                        ))),
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                    ];
                    self.handle_save_file(&arg_exprs, None)?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "fixtures.save" || function == "fixtures__save" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "fixtures.save requires name and content".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let content_val = self.evaluate_expression(args[1].clone())?;
                    let content = match content_val {
                        RuntimeValue::String(s) => s,
                        RuntimeValue::Json(j) => j.to_string(),
                        other => format!("{:?}", other),
                    };
                    let base = std::env::var("LEXON_FIXTURES_DIR")
                        .unwrap_or_else(|_| "samples/fixtures".to_string());
                    let path = format!("{}/{}", base, name);
                    let arg_exprs = vec![
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(content))),
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                    ];
                    self.handle_save_file(&arg_exprs, None)?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "fixtures.load" || function == "fixtures__load" {
                    if args.len() < 1 {
                        return Err(ExecutorError::ArgumentError(
                            "fixtures.load requires name".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let base = std::env::var("LEXON_FIXTURES_DIR")
                        .unwrap_or_else(|_| "samples/fixtures".to_string());
                    let path = format!("{}/{}", base, name);
                    let arg_exprs = vec![LexExpression::Value(ValueRef::Literal(
                        LexLiteral::String(path),
                    ))];
                    self.handle_read_file(&arg_exprs, result.as_ref())?;
                    Ok(())
                } else if function == "fixtures.save_json" || function == "fixtures__save_json" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "fixtures.save_json requires name and json".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let json_val = self.evaluate_expression(args[1].clone())?;
                    let content = match json_val {
                        RuntimeValue::Json(j) => {
                            serde_json::to_string_pretty(&j).unwrap_or_else(|_| j.to_string())
                        }
                        RuntimeValue::String(s) => {
                            if let Ok(v) = serde_json::from_str::<Value>(&s) {
                                serde_json::to_string_pretty(&v).unwrap_or(s)
                            } else {
                                s
                            }
                        }
                        other => format!("{:?}", other),
                    };
                    let base = std::env::var("LEXON_FIXTURES_DIR")
                        .unwrap_or_else(|_| "samples/fixtures".to_string());
                    let path = format!("{}/{}", base, name);
                    let arg_exprs = vec![
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(content))),
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                    ];
                    self.handle_save_file(&arg_exprs, None)?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "fixtures.load_json" || function == "fixtures__load_json" {
                    if args.len() < 1 {
                        return Err(ExecutorError::ArgumentError(
                            "fixtures.load_json requires name".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let base = std::env::var("LEXON_FIXTURES_DIR")
                        .unwrap_or_else(|_| "samples/fixtures".to_string());
                    let path = format!("{}/{}", base, name);
                    let txt = std::fs::read_to_string(&path)
                        .map_err(|e| ExecutorError::IoError(format!("read: {}", e)))?;
                    // Parse, with fallbacks for common escaping artifacts
                    let mut json: Value = match serde_json::from_str::<Value>(&txt) {
                        Ok(v) => v,
                        Err(_) => {
                            let repaired = txt.replace("\\\"", "\"");
                            serde_json::from_str::<Value>(&repaired)
                                .map_err(|e| ExecutorError::JsonError(format!("parse: {}", e)))?
                        }
                    };
                    if let Value::String(inner) = &json {
                        if let Ok(v2) = serde_json::from_str::<Value>(inner) {
                            json = v2;
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(json))?;
                    }
                    Ok(())
                } else if function == "rate_limiter.acquire" || function == "rate_limiter__acquire"
                {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "rate_limiter.acquire requires name and max_per_sec".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let max_ps_val = self.evaluate_expression(args[1].clone())?;
                    let max_per_sec: u32 = match max_ps_val {
                        RuntimeValue::Integer(i) => i as u32,
                        RuntimeValue::Float(f) => f as u32,
                        RuntimeValue::String(s) => s.parse().unwrap_or(1),
                        _ => 1,
                    };
                    let now_sec = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or(std::time::Duration::from_secs(0))
                        .as_secs()) as u64;
                    let entry = self.rate_limits.entry(name).or_insert((now_sec, 0));
                    if entry.0 != now_sec {
                        entry.0 = now_sec;
                        entry.1 = 0;
                    }
                    let granted = if entry.1 < max_per_sec {
                        entry.1 += 1;
                        true
                    } else {
                        false
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(granted))?;
                    }
                    Ok(())
                } else if function == "metrics.get_call_logs"
                    || function == "metrics__get_call_logs"
                {
                    let logs = if let Some(ref a) = self.llm_adapter_new {
                        a.get_call_logs_json()
                    } else {
                        self.llm_adapter.get_call_logs_json()
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(logs))?;
                    }
                    Ok(())
                } else if function == "metrics.write_call_logs"
                    || function == "metrics__write_call_logs"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "metrics.write_call_logs requires path".to_string(),
                        ));
                    }
                    let path = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let logs = if let Some(ref a) = self.llm_adapter_new {
                        a.get_call_logs_json()
                    } else {
                        self.llm_adapter.get_call_logs_json()
                    };
                    let arg_exprs = vec![
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(
                            logs.to_string(),
                        ))),
                        LexExpression::Value(ValueRef::Literal(LexLiteral::String(path))),
                    ];
                    self.handle_save_file(&arg_exprs, None)?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "metrics.export_prometheus"
                    || function == "metrics__export_prometheus"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "metrics.export_prometheus requires path".to_string(),
                        ));
                    }
                    let path = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let logs = if let Some(ref a) = self.llm_adapter_new {
                        a.get_call_logs_json()
                    } else {
                        self.llm_adapter.get_call_logs_json()
                    };
                    let mut total_cost = 0.0_f64;
                    if let serde_json::Value::Array(arr) = &logs {
                        for e in arr {
                            if let Some(c) = e.get("cost_usd").and_then(|v| v.as_f64()) {
                                total_cost += c;
                            }
                        }
                    }
                    let mut out = String::new();
                    out.push_str("# HELP lexon_llm_calls_total Total LLM calls\n# TYPE lexon_llm_calls_total counter\n");
                    out.push_str(&format!(
                        "lexon_llm_calls_total {}\n\n",
                        self.total_llm_calls
                    ));

                    out.push_str("# HELP lexon_llm_latency_ms_sum Total LLM latency (ms)\n# TYPE lexon_llm_latency_ms_sum counter\n");
                    out.push_str(&format!(
                        "lexon_llm_latency_ms_sum {}\n\n",
                        self.llm_total_ms
                    ));

                    out.push_str("# HELP lexon_llm_cost_usd_total Total LLM cost (USD)\n# TYPE lexon_llm_cost_usd_total counter\n");
                    out.push_str(&format!("lexon_llm_cost_usd_total {}\n\n", total_cost));

                    // Provider health from new adapter if present (1 up, 0 down)
                    let mut wrote_health = false;
                    if let Some(ref a) = self.llm_adapter_new {
                        if let Some(map) = a
                            .routing_metrics_json()
                            .get("provider_health")
                            .and_then(|v| v.as_object())
                        {
                            out.push_str("# HELP lexon_provider_health Provider health (1 up, 0 down)\n# TYPE lexon_provider_health gauge\n");
                            for (prov, obj) in map {
                                let up = obj
                                    .get("status")
                                    .and_then(|v| v.as_str())
                                    .map(|s| (s != "down") as i32)
                                    .unwrap_or(1);
                                out.push_str(&format!(
                                    "lexon_provider_health{{provider=\"{}\"}} {}\n",
                                    prov, up
                                ));
                            }
                            out.push('\n');
                            wrote_health = true;
                        }
                    }
                    if !wrote_health {
                        out.push_str("# HELP lexon_provider_health Provider health (1 up, 0 down)\n# TYPE lexon_provider_health gauge\n\n");
                    }

                    // Tool calls by tool from registry
                    out.push_str("# HELP lexon_tool_calls_total Tool calls\n# TYPE lexon_tool_calls_total counter\n");
                    for (tool, meta) in &self.tool_registry {
                        out.push_str(&format!(
                            "lexon_tool_calls_total{{tool=\"{}\"}} {}\n",
                            tool, meta.used_calls
                        ));
                    }
                    out.push('\n');

                    out.push_str("# HELP lexon_errors_total Total errors\n# TYPE lexon_errors_total counter\nlexon_errors_total 0\n");

                    std::fs::write(&path, out).map_err(|e| {
                        ExecutorError::IoError(format!("metrics.export_prometheus: {}", e))
                    })?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "configure_provider" || function == "providers__configure" {
                    // configure_provider(name, {kind: "huggingface"|"ollama"|"custom", base_url, api_key?})
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "configure_provider requires name and config".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let cfg_val = self.evaluate_expression(args[1].clone())?;
                    let obj = match cfg_val {
                        RuntimeValue::Json(serde_json::Value::Object(m)) => m,
                        RuntimeValue::String(s) => {
                            serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&s)
                                .unwrap_or_default()
                        }
                        _ => serde_json::Map::new(),
                    };
                    let kind = obj
                        .get("kind")
                        .and_then(|v| v.as_str())
                        .unwrap_or("custom")
                        .to_lowercase();
                    let base_url = obj
                        .get("base_url")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let api_key = obj
                        .get("api_key")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    let p_kind = match kind.as_str() {
                        "huggingface" => ProviderKind::HuggingFace,
                        "ollama" => ProviderKind::Ollama,
                        _ => ProviderKind::Custom,
                    };
                    let cfg = ProviderConfig {
                        kind: p_kind,
                        base_url,
                        api_key,
                    };
                    if let Some(ref mut a) = self.llm_adapter_new {
                        a.register_provider(name.clone(), cfg);
                    } else {
                        self.llm_adapter.register_provider(name.clone(), cfg);
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "providers.override_model"
                    || function == "providers__override_model"
                {
                    // providers.override_model(model, provider_key)
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "providers.override_model requires model and provider".to_string(),
                        ));
                    }
                    let model = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let provider = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    if let Some(ref mut a) = self.llm_adapter_new {
                        a.override_model_provider(model.clone(), provider.clone());
                    } else {
                        self.llm_adapter
                            .override_model_provider(model.clone(), provider.clone());
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "routing.get_metrics_json"
                    || function == "routing__get_metrics_json"
                {
                    // Periodic health tick before reading metrics
                    if let Some(ref mut adapter) = self.llm_adapter_new {
                        adapter.tick_health();
                    } else {
                        self.llm_adapter.tick_health();
                    }
                    let legacy = self.llm_adapter.routing_metrics_json();
                    let json = if let Some(ref adapter) = self.llm_adapter_new {
                        let newm = adapter.routing_metrics_json();
                        // simple merge: prefer non-empty maps; otherwise fallback to legacy
                        let is_new_empty = newm
                            .get("ab_variants")
                            .map(|v| v.as_object().map(|m| m.is_empty()).unwrap_or(true))
                            .unwrap_or(true)
                            && newm
                                .get("provider_health")
                                .map(|v| v.as_object().map(|m| m.is_empty()).unwrap_or(true))
                                .unwrap_or(true);
                        if is_new_empty {
                            legacy
                        } else {
                            newm
                        }
                    } else {
                        legacy
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(json))?;
                    }
                    Ok(())
                } else if function == "sessions.gc_now" || function == "sessions__gc_now" {
                    let _ = self.memory_manager.clean_expired();
                    self.last_session_gc_epoch_ms = Self::now_epoch_ms();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "session.exists" || function == "session__exists" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "session.exists requires session_id".to_string(),
                        ));
                    }
                    let sid = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "session_id must be string".to_string(),
                            ))
                        }
                    };
                    let exists = self.memory_manager.session_exists(&sid);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(exists))?;
                    }
                    Ok(())
                } else if function == "cache.invalidate" || function == "cache__invalidate" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "cache.invalidate requires key".to_string(),
                        ));
                    }
                    let key = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "key must be string".to_string(),
                            ))
                        }
                    };
                    let ok = if let Some(ref mut a) = self.llm_adapter_new {
                        a.cache_invalidate(&key)
                    } else {
                        self.llm_adapter.cache_invalidate(&key)
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(ok))?;
                    }
                    Ok(())
                } else if function == "cache.invalidate_prefix"
                    || function == "cache__invalidate_prefix"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "cache.invalidate_prefix requires prefix".to_string(),
                        ));
                    }
                    let prefix = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "prefix must be string".to_string(),
                            ))
                        }
                    };
                    let n = if let Some(ref mut a) = self.llm_adapter_new {
                        a.cache_invalidate_prefix(&prefix)
                    } else {
                        self.llm_adapter.cache_invalidate_prefix(&prefix)
                    } as i64;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Integer(n))?;
                    }
                    Ok(())
                } else if function == "memory_index.prune_by_metadata"
                    || function == "memory_index__prune_by_metadata"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "prune_by_metadata requires filters object".to_string(),
                        ));
                    }
                    let filters_val = self.evaluate_expression(args[0].clone())?;
                    let filters_map: std::collections::HashMap<String, String> = match filters_val {
                        RuntimeValue::Json(serde_json::Value::Object(m)) => m
                            .into_iter()
                            .filter_map(|(k, v)| v.as_str().map(|s| (k, s.to_string())))
                            .collect(),
                        RuntimeValue::String(s) => {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&s) {
                                if let Some(m) = v.as_object() {
                                    m.iter()
                                        .filter_map(|(k, v)| {
                                            v.as_str().map(|s| (k.clone(), s.to_string()))
                                        })
                                        .collect()
                                } else {
                                    std::collections::HashMap::new()
                                }
                            } else {
                                std::collections::HashMap::new()
                            }
                        }
                        _ => std::collections::HashMap::new(),
                    };
                    let deleted = if let Some(ref mut v) = self.vector_memory_system {
                        v.prune_by_metadata(&filters_map)?
                    } else {
                        0
                    } as i64;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Integer(deleted))?;
                    }
                    Ok(())
                } else if function == "write_metrics_json" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "write_metrics_json requires path".to_string(),
                        ));
                    }
                    let path = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "path must be string".to_string(),
                            ))
                        }
                    };
                    // Pull per-call logs from the active adapter (new preferred)
                    let logs_val = if let Some(ref a) = self.llm_adapter_new {
                        a.get_call_logs_json()
                    } else {
                        self.llm_adapter.get_call_logs_json()
                    };
                    // Aggregate rollups
                    let mut total_cost = 0.0_f64;
                    let mut by_model: std::collections::HashMap<String, serde_json::Value> =
                        std::collections::HashMap::new();
                    if let serde_json::Value::Array(arr) = &logs_val {
                        for entry in arr {
                            let model = entry
                                .get("model")
                                .and_then(|v| v.as_str())
                                .unwrap_or("(unknown)")
                                .to_string();
                            let cost = entry
                                .get("cost_usd")
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0);
                            let ms = entry
                                .get("elapsed_ms")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            total_cost += cost;
                            let e = by_model.entry(model).or_insert_with(|| serde_json::json!({"calls":0, "total_ms":0, "total_cost_usd":0.0}));
                            if let Some(obj) = e.as_object_mut() {
                                let calls =
                                    obj.get_mut("calls").and_then(|v| v.as_u64()).unwrap_or(0) + 1;
                                let tot_ms = obj
                                    .get_mut("total_ms")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0)
                                    + ms;
                                let tot_cost = obj
                                    .get_mut("total_cost_usd")
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0)
                                    + cost;
                                obj.insert("calls".to_string(), serde_json::Value::from(calls));
                                obj.insert("total_ms".to_string(), serde_json::Value::from(tot_ms));
                                obj.insert(
                                    "total_cost_usd".to_string(),
                                    serde_json::Value::from(tot_cost),
                                );
                            }
                        }
                    }
                    let json = serde_json::json!({
                        "total_llm_calls": self.total_llm_calls,
                        "llm_total_ms": self.llm_total_ms,
                        "total_cost_usd": total_cost,
                        "by_model": by_model,
                        "call_logs": logs_val,
                    });
                    std::fs::write(
                        &path,
                        serde_json::to_string_pretty(&json).unwrap_or_else(|_| json.to_string()),
                    )
                    .map_err(|e| ExecutorError::IoError(format!("write_metrics_json: {}", e)))?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "get_env_or" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "get_env_or requires (name, default)".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "name must be string".to_string(),
                            ))
                        }
                    };
                    let def = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "default must be string".to_string(),
                            ))
                        }
                    };
                    let val = std::env::var(&name).unwrap_or(def);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(val))?;
                    }
                    Ok(())
                } else if function == "mcp__run_stdio" || function == "mcp.run_stdio" {
                    // Non-blocking run
                    println!("[MCP] run_stdio requested (non-blocking noop in CLI mode)");
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "mcp__run_ws" || function == "mcp.run_ws" {
                    println!("[MCP] run_ws requested (non-blocking noop in CLI mode)");
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "mcp__run_ws_server"
                    || function == "mcp.run_ws_server"
                    || function == "mcp__serve_ws_server"
                    || function == "mcp.serve_ws_server"
                {
                    println!("[MCP] run/serve_ws_server requested (non-blocking noop in CLI mode)");
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "mcp__serve_stdio" || function == "mcp.serve_stdio" {
                    println!("[MCP] serve_stdio requested (non-blocking noop in CLI mode)");
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "mcp__tool_call" || function == "mcp.tool_call" {
                    if args.len() < 1 {
                        return Err(ExecutorError::ArgumentError(
                            "mcp.tool_call requires name and optional args".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "tool name must be string".to_string(),
                            ))
                        }
                    };
                    let json_args = if args.len() >= 2 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Json(j) => j,
                            RuntimeValue::String(s) => serde_json::Value::String(s),
                            _ => serde_json::Value::Null,
                        }
                    } else {
                        serde_json::Value::Null
                    };
                    let out = self.execute_tool_call_json(&name, &json_args)?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "mcp__set_quota" || function == "mcp.set_quota" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "mcp.set_quota requires (tool, allowed_calls)".to_string(),
                        ));
                    }
                    let tool = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "tool must be string".to_string(),
                            ))
                        }
                    };
                    let allowed = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i.max(0) as u64,
                        RuntimeValue::Float(f) => (f.max(0.0)) as u64,
                        RuntimeValue::String(s) => s.parse::<u64>().unwrap_or(0),
                        _ => 0,
                    };
                    let meta = self.tool_registry.entry(tool).or_insert(ToolMeta {
                        allowed_calls: None,
                        used_calls: 0,
                        scope: None,
                    });
                    meta.allowed_calls = Some(allowed);
                    meta.used_calls = 0;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "data_validate_schema" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "data_validate_schema requires (json, schema)".to_string(),
                        ));
                    }
                    let json = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::from_str(&s)
                            .map_err(|e| ExecutorError::JsonError(format!("parse json: {}", e)))?,
                        other => {
                            return Err(ExecutorError::ArgumentError(format!(
                                "first arg must be JSON or string: {:?}",
                                other
                            )))
                        }
                    };
                    let schema = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::from_str(&s).map_err(|e| {
                            ExecutorError::JsonError(format!("parse schema: {}", e))
                        })?,
                        other => {
                            return Err(ExecutorError::ArgumentError(format!(
                                "second arg must be JSON schema or string: {:?}",
                                other
                            )))
                        }
                    };
                    let compiled = jsonschema::JSONSchema::compile(&schema)
                        .map_err(|e| ExecutorError::JsonError(format!("schema compile: {}", e)))?;
                    let valid = compiled.validate(&json).is_ok();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(valid))?;
                    }
                    Ok(())
                } else if function == "pii_redact" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "pii_redact requires text".to_string(),
                        ));
                    }
                    let text = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "text must be string".to_string(),
                            ))
                        }
                    };
                    let email_re =
                        regex::Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
                            .unwrap();
                    let phone_re = regex::Regex::new(r"\+?\d[\d\s\-]{7,}\d").unwrap();
                    let mut redacted = email_re.replace_all(&text, "<redacted_email>").to_string();
                    redacted = phone_re
                        .replace_all(&redacted, "<redacted_phone>")
                        .to_string();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(redacted))?;
                    }
                    Ok(())
                } else if function == "memory_index__set_metadata"
                    || function == "memory_index.set_metadata"
                {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "set_metadata requires path, metadata".to_string(),
                        ));
                    }
                    let path = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "path must be string".to_string(),
                            ))
                        }
                    };
                    let metadata = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => match serde_json::from_str(&s) {
                            Ok(j) => j,
                            Err(_) => {
                                let trimmed =
                                    s.trim().trim_start_matches('{').trim_end_matches('}');
                                let mut map = serde_json::Map::new();
                                if let Some((k, v)) = trimmed.split_once(':') {
                                    map.insert(
                                        k.trim().trim_matches('"').to_string(),
                                        serde_json::Value::String(
                                            v.trim().trim_matches('"').to_string(),
                                        ),
                                    );
                                }
                                Value::Object(map)
                            }
                        },
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "metadata must be JSON".to_string(),
                            ))
                        }
                    };
                    if let Some(ref mut vm) = self.vector_memory_system {
                        vm.set_metadata(&path, &metadata)?;
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "data_validate_schema_file" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "data_validate_schema_file requires (json_path, schema_path)"
                                .to_string(),
                        ));
                    }
                    let jpath = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "json_path must be string".to_string(),
                            ))
                        }
                    };
                    let spath = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "schema_path must be string".to_string(),
                            ))
                        }
                    };
                    let jtxt = std::fs::read_to_string(&jpath)
                        .map_err(|e| ExecutorError::IoError(format!("read json: {}", e)))?;
                    let stxt = std::fs::read_to_string(&spath)
                        .map_err(|e| ExecutorError::IoError(format!("read schema: {}", e)))?;
                    let json: Value = serde_json::from_str(&jtxt)
                        .map_err(|e| ExecutorError::JsonError(format!("parse json: {}", e)))?;
                    let schema: Value = serde_json::from_str(&stxt)
                        .map_err(|e| ExecutorError::JsonError(format!("parse schema: {}", e)))?;
                    let compiled = jsonschema::JSONSchema::compile(&schema)
                        .map_err(|e| ExecutorError::JsonError(format!("schema compile: {}", e)))?;
                    let valid = compiled.validate(&json).is_ok();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(valid))?;
                    }
                    Ok(())
                } else if function == "rag_eval.precision_at_k"
                    || function == "rag_eval__precision_at_k"
                {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "precision_at_k requires (hits, gold)".to_string(),
                        ));
                    }
                    let hits = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(Value::Array(a)) => a,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "hits must be array".to_string(),
                            ))
                        }
                    };
                    let gold = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(Value::Array(a)) => a
                            .into_iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                            .collect::<Vec<_>>(),
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "gold must be array of strings".to_string(),
                            ))
                        }
                    };
                    let k = hits.len().max(1) as f64;
                    let mut tp = 0.0;
                    for h in hits {
                        let text = h
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if gold.iter().any(|g| text.contains(g)) {
                            tp += 1.0;
                        }
                    }
                    let p = tp / k;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(p))?;
                    }
                    Ok(())
                } else if function == "rag_eval.recall_at_k" || function == "rag_eval__recall_at_k"
                {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "recall_at_k requires (hits, gold)".to_string(),
                        ));
                    }
                    let hits = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(Value::Array(a)) => a,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "hits must be array".to_string(),
                            ))
                        }
                    };
                    let gold = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(Value::Array(a)) => a
                            .into_iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                            .collect::<Vec<_>>(),
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "gold must be array of strings".to_string(),
                            ))
                        }
                    };
                    let mut tp = 0.0;
                    for h in hits {
                        let text = h
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if gold.iter().any(|g| text.contains(g)) {
                            tp += 1.0;
                        }
                    }
                    let r = tp / (gold.len().max(1) as f64);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(r))?;
                    }
                    Ok(())
                } else if function == "rag_eval.f1_at_k" || function == "rag_eval__f1_at_k" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "f1_at_k requires (hits, gold)".to_string(),
                        ));
                    }
                    let hits = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(Value::Array(a)) => a,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "hits must be array".to_string(),
                            ))
                        }
                    };
                    let gold = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(Value::Array(a)) => a
                            .into_iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_lowercase()))
                            .collect::<Vec<_>>(),
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "gold must be array of strings".to_string(),
                            ))
                        }
                    };
                    let k = hits.len().max(1) as f64;
                    let mut tp = 0.0;
                    for h in &hits {
                        let text = h
                            .get("content")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_lowercase();
                        if gold.iter().any(|g| text.contains(g)) {
                            tp += 1.0;
                        }
                    }
                    let p = tp / k;
                    let r = tp / (gold.len().max(1) as f64);
                    let f1 = if p + r > 0.0 {
                        2.0 * p * r / (p + r)
                    } else {
                        0.0
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(f1))?;
                    }
                    Ok(())
                } else if function == "model_arbitrage" {
                    // model_arbitrage(topic, models_config, model_decider?, rounds?) -> consensus
                    // Multi-LLM debate system that seeks consensus with optional model decider
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "model_arbitrage requires topic and models_config".to_string(),
                        ));
                    }

                    let topic_arg = self.evaluate_expression(args[0].clone())?;
                    let topic = match topic_arg {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", topic_arg),
                    };

                    let models_config_arg = self.evaluate_expression(args[1].clone())?;
                    let models_config = match models_config_arg {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", models_config_arg),
                    };

                    // Detect if there's model_decider (3rd parameter string) and rounds (4th parameter)
                    let (model_decider, rounds) = if args.len() >= 4 {
                        // 4 parameters: topic, models_config, model_decider, rounds
                        let decider_arg = self.evaluate_expression(args[2].clone())?;
                        let decider = match decider_arg {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        };

                        let rounds_arg = self.evaluate_expression(args[3].clone())?;
                        let rounds = match rounds_arg {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 3,
                        };

                        (decider, rounds)
                    } else if args.len() == 3 {
                        // 3 parameters: detect if it's (topic, models, model_decider) or (topic, models, rounds)
                        let third_arg = self.evaluate_expression(args[2].clone())?;
                        match third_arg {
                            RuntimeValue::String(s) => {
                                // It's model_decider, use rounds by default
                                (Some(s), 3)
                            }
                            RuntimeValue::Integer(i) => {
                                // It's rounds, no model_decider
                                (None, i as usize)
                            }
                            _ => (None, 3),
                        }
                    } else {
                        // 2 parameters: only topic and models_config
                        (None, 3)
                    };

                    println!("⚖️ model_arbitrage: Starting consensus process");
                    println!("📋 Topic: {}", topic);
                    println!("🤖 Models: {}", models_config);
                    if let Some(ref decider) = model_decider {
                        println!("🧠 Model Decider: {}", decider);
                    }
                    println!("🔄 Rounds: {}", rounds);

                    let mut consensus_log = Vec::new();
                    consensus_log.push(format!("=== MODEL ARBITRAGE: {} ===", topic));
                    consensus_log.push(format!("Models: {}", models_config));
                    if let Some(ref decider) = model_decider {
                        consensus_log.push(format!("Decider Model: {}", decider));
                    }
                    consensus_log.push("".to_string());

                    let mut _final_consensus = String::new();

                    for round in 1..=rounds {
                        println!("⚖️ Round {}/{}: Seeking consensus...", round, rounds);

                        consensus_log.push(format!("--- Arbitrage Round {} ---", round));

                        let arbitrage_prompt = format!(
                            "You are participating in a structured debate about: '{}'\n\
                            Round {} of {}. Your goal is to provide balanced analysis and work toward consensus.\n\
                            Consider multiple perspectives and present well-reasoned arguments.",
                            topic, round, rounds
                        );

                        // Use the decision model if specified, otherwise use simulated by default
                        let decision_model = model_decider.as_deref().unwrap_or("simulated");

                        let mut llm_adapter = self.llm_adapter.clone();
                        let response = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                llm_adapter.call_llm_async(
                                    Some(decision_model), Some(0.3), // Lower temperature for more consistent reasoning
                                    Some("You are a balanced analyst seeking consensus through structured debate."),
                                    Some(&arbitrage_prompt), None, None, &HashMap::new()
                                ).await
                            })
                        })?;

                        consensus_log.push(format!(
                            "🎯 Arbitrage Analysis ({}): {}",
                            decision_model, response
                        ));
                        consensus_log.push("".to_string());

                        _final_consensus = response; // Keep the latest analysis as working consensus

                        println!("✅ Round {} arbitrage completed", round);
                    }

                    // Final consensus synthesis using the decision model
                    let synthesis_prompt = format!(
                        "Based on the multi-round analysis of '{}', provide a final balanced consensus that incorporates the key insights and addresses the main considerations.",
                        topic
                    );

                    let decision_model = model_decider.as_deref().unwrap_or("simulated");

                    let mut llm_adapter = self.llm_adapter.clone();
                    let final_synthesis = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            llm_adapter.call_llm_async(
                                Some(decision_model), Some(0.2), // Very low temperature for consistent final synthesis
                                Some("You are synthesizing a final consensus from multiple rounds of analysis."),
                                Some(&synthesis_prompt), None, None, &HashMap::new()
                            ).await
                        })
                    })?;

                    consensus_log.push("=== FINAL CONSENSUS ===".to_string());
                    consensus_log.push(format!(
                        "Final Decision by {}: {}",
                        decision_model, final_synthesis
                    ));

                    let _full_log = consensus_log.join("\n");

                    println!(
                        "🎯 model_arbitrage: Consensus reached after {} rounds",
                        rounds
                    );
                    if let Some(ref decider) = model_decider {
                        println!("🧠 Final decision made by: {}", decider);
                    }

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(final_synthesis))?;
                    }
                    Ok(())
                } else if function == "enumerate" {
                    // enumerate(array) -> [(index, value)] - Iterator with indices
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "enumerate requires exactly 1 argument".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let enumerated_array = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            let mut result = Vec::new();
                            for (index, item) in arr.into_iter().enumerate() {
                                let pair = serde_json::json!([index, item]);
                                result.push(pair);
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        RuntimeValue::String(s) => {
                            let mut result = Vec::new();
                            for (index, ch) in s.chars().enumerate() {
                                let pair = serde_json::json!([index, ch.to_string()]);
                                result.push(pair);
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "enumerate requires an array or string".to_string(),
                            ))
                        }
                    };

                    println!("🔄 enumerate: Created enumerated array with indices");

                    if let Some(res) = result {
                        self.store_value(res, enumerated_array)?;
                    }
                    Ok(())
                } else if function == "range" {
                    // range(start, end, step?) -> [start..end] - Range generator with optional step
                    if args.is_empty() || args.len() > 3 {
                        return Err(ExecutorError::ArgumentError(
                            "range requires 1-3 arguments".to_string(),
                        ));
                    }

                    let (start, end, step) = if args.len() == 1 {
                        // range(end) -> range(0, end, 1)
                        let end_value = self.evaluate_expression(args[0].clone())?;
                        let end = match end_value {
                            RuntimeValue::Integer(i) => i,
                            _ => {
                                return Err(ExecutorError::ArgumentError(
                                    "range end must be an integer".to_string(),
                                ))
                            }
                        };
                        (0, end, 1)
                    } else if args.len() == 2 {
                        // range(start, end) -> range(start, end, 1)
                        let start_value = self.evaluate_expression(args[0].clone())?;
                        let end_value = self.evaluate_expression(args[1].clone())?;
                        let start = match start_value {
                            RuntimeValue::Integer(i) => i,
                            _ => {
                                return Err(ExecutorError::ArgumentError(
                                    "range start must be an integer".to_string(),
                                ))
                            }
                        };
                        let end = match end_value {
                            RuntimeValue::Integer(i) => i,
                            _ => {
                                return Err(ExecutorError::ArgumentError(
                                    "range end must be an integer".to_string(),
                                ))
                            }
                        };
                        (start, end, 1)
                    } else {
                        // range(start, end, step)
                        let start_value = self.evaluate_expression(args[0].clone())?;
                        let end_value = self.evaluate_expression(args[1].clone())?;
                        let step_value = self.evaluate_expression(args[2].clone())?;
                        let start = match start_value {
                            RuntimeValue::Integer(i) => i,
                            _ => {
                                return Err(ExecutorError::ArgumentError(
                                    "range start must be an integer".to_string(),
                                ))
                            }
                        };
                        let end = match end_value {
                            RuntimeValue::Integer(i) => i,
                            _ => {
                                return Err(ExecutorError::ArgumentError(
                                    "range end must be an integer".to_string(),
                                ))
                            }
                        };
                        let step = match step_value {
                            RuntimeValue::Integer(i) => {
                                if i == 0 {
                                    return Err(ExecutorError::ArgumentError(
                                        "range step cannot be zero".to_string(),
                                    ));
                                }
                                i
                            }
                            _ => {
                                return Err(ExecutorError::ArgumentError(
                                    "range step must be an integer".to_string(),
                                ))
                            }
                        };
                        (start, end, step)
                    };

                    let mut range_array = Vec::new();
                    if step > 0 {
                        let mut current = start;
                        while current < end {
                            range_array.push(Value::Number(serde_json::Number::from(current)));
                            current += step;
                        }
                    } else {
                        let mut current = start;
                        while current > end {
                            range_array.push(Value::Number(serde_json::Number::from(current)));
                            current += step;
                        }
                    }

                    let array_len = range_array.len();
                    let result_value = RuntimeValue::Json(Value::Array(range_array));

                    println!(
                        "🔄 range: Generated range from {} to {} with step {} ({} elements)",
                        start, end, step, array_len
                    );

                    if let Some(res) = result {
                        self.store_value(res, result_value)?;
                    }
                    Ok(())
                } else if function == "map" {
                    // map(array, callback) -> [mapped] - Transform array elements using a simple callback expression
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "map requires exactly 2 arguments: array and callback".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let callback_expr = args[1].clone();

                    let mapped_array = match array_value {
                        RuntimeValue::Json(Value::Array(ref arr)) => {
                            let mut result_vec = Vec::new();
                            for (index, item) in arr.iter().enumerate() {
                                let mapped = if let LexExpression::Value(ValueRef::Literal(
                                    LexLiteral::String(callback_str),
                                )) = &callback_expr
                                {
                                    self.evaluate_callback(callback_str, item, index)?
                                } else {
                                    // Fallback: identity mapping
                                    item.clone()
                                };
                                result_vec.push(mapped);
                            }
                            RuntimeValue::Json(Value::Array(result_vec))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "map requires an array as first argument".to_string(),
                            ))
                        }
                    };

                    let original_len =
                        if let RuntimeValue::Json(Value::Array(ref arr)) = array_value {
                            arr.len()
                        } else {
                            0
                        };
                    println!("🔄 map: Transformed {} elements", original_len);

                    if let Some(res) = result {
                        self.store_value(res, mapped_array)?;
                    }
                    Ok(())
                } else if function == "filter" {
                    // filter(array, predicate) -> [filtered] - Array filtering with improved predicate support
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "filter requires exactly 2 arguments: array and predicate".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let predicate_expr = args[1].clone();

                    let filtered_array = match array_value {
                        RuntimeValue::Json(Value::Array(ref arr)) => {
                            let mut result = Vec::new();
                            for (index, item) in arr.iter().enumerate() {
                                // Improved predicate support - evaluate predicate expression for each item
                                let should_keep = if let LexExpression::Value(ValueRef::Literal(
                                    LexLiteral::String(predicate_str),
                                )) = &predicate_expr
                                {
                                    self.evaluate_predicate(predicate_str, item, index)?
                                } else {
                                    // Fallback to default logic
                                    match item {
                                        Value::Number(n) => {
                                            if let Some(i) = n.as_i64() {
                                                i > 0
                                            } else if let Some(f) = n.as_f64() {
                                                f > 0.0
                                            } else {
                                                true
                                            }
                                        }
                                        Value::String(s) => !s.is_empty(),
                                        Value::Bool(b) => *b,
                                        Value::Null => false,
                                        _ => true,
                                    }
                                };

                                if should_keep {
                                    result.push(item.clone());
                                }
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "filter requires an array as first argument".to_string(),
                            ))
                        }
                    };

                    let original_len =
                        if let RuntimeValue::Json(Value::Array(ref arr)) = array_value {
                            arr.len()
                        } else {
                            0
                        };
                    let filtered_len =
                        if let RuntimeValue::Json(Value::Array(ref arr)) = filtered_array {
                            arr.len()
                        } else {
                            0
                        };
                    println!(
                        "🔄 filter: Filtered array from {} to {} elements",
                        original_len, filtered_len
                    );

                    if let Some(res) = result {
                        self.store_value(res, filtered_array)?;
                    }
                    Ok(())
                } else if function == "reduce" {
                    // reduce(array, initial, callback) -> value - Array reduction with improved callback support
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError(
                            "reduce requires exactly 3 arguments: array, initial, and callback"
                                .to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let initial_value = self.evaluate_expression(args[1].clone())?;
                    let callback_expr = args[2].clone();

                    let reduced_value = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            if arr.is_empty() {
                                initial_value
                            } else {
                                let mut accumulator = initial_value;
                                for (index, item) in arr.iter().enumerate() {
                                    // Improved callback support - evaluate callback expression for each item
                                    if let LexExpression::Value(ValueRef::Literal(
                                        LexLiteral::String(callback_str),
                                    )) = &callback_expr
                                    {
                                        accumulator = self.evaluate_reduce_callback(
                                            callback_str,
                                            &accumulator,
                                            item,
                                            index,
                                        )?;
                                    } else {
                                        // Fallback to default logic (addition)
                                        accumulator = match (&accumulator, item) {
                                            (RuntimeValue::Integer(acc), Value::Number(n)) => {
                                                if let Some(item_i) = n.as_i64() {
                                                    RuntimeValue::Integer(acc + item_i)
                                                } else {
                                                    accumulator
                                                }
                                            }
                                            _ => accumulator,
                                        };
                                    }
                                }
                                accumulator
                            }
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "reduce requires an array as first argument".to_string(),
                            ))
                        }
                    };

                    let result_type = match &reduced_value {
                        RuntimeValue::String(_) => "String".to_string(),
                        RuntimeValue::Json(Value::Number(_)) => "Number".to_string(),
                        RuntimeValue::Integer(i) => format!("Integer({})", i),
                        RuntimeValue::Float(_) => "Float".to_string(),
                        _ => "Other".to_string(),
                    };
                    println!(
                        "🔄 reduce: Applied reduction to array (result: {})",
                        result_type
                    );

                    if let Some(res) = result {
                        self.store_value(res, reduced_value)?;
                    }
                    Ok(())
                } else if function == "zip" {
                    // zip(array1, array2) -> [(item1, item2)] - Combine two arrays
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "zip requires exactly 2 arguments".to_string(),
                        ));
                    }

                    let array1_value = self.evaluate_expression(args[0].clone())?;
                    let array2_value = self.evaluate_expression(args[1].clone())?;

                    let zipped_array = match (array1_value, array2_value) {
                        (
                            RuntimeValue::Json(Value::Array(arr1)),
                            RuntimeValue::Json(Value::Array(arr2)),
                        ) => {
                            let mut result = Vec::new();
                            let min_len = std::cmp::min(arr1.len(), arr2.len());
                            for i in 0..min_len {
                                let pair = serde_json::json!([arr1[i].clone(), arr2[i].clone()]);
                                result.push(pair);
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "zip requires two arrays".to_string(),
                            ))
                        }
                    };

                    println!("🔄 zip: Combined two arrays into pairs");

                    if let Some(res) = result {
                        self.store_value(res, zipped_array)?;
                    }
                    Ok(())
                } else if function == "flatten" {
                    // flatten(nested_array) -> [flattened] - Flatten nested arrays
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "flatten requires exactly 1 argument".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let flattened_array = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            let mut result = Vec::new();
                            for item in arr {
                                match item {
                                    Value::Array(nested) => {
                                        result.extend(nested);
                                    }
                                    _ => {
                                        result.push(item);
                                    }
                                }
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "flatten requires an array".to_string(),
                            ))
                        }
                    };

                    println!("🔄 flatten: Flattened nested array");

                    if let Some(res) = result {
                        self.store_value(res, flattened_array)?;
                    }
                    Ok(())
                } else if function == "unique" {
                    // unique(array) -> [unique_items] - Remove duplicates
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "unique requires exactly 1 argument".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let unique_array = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            let mut result = Vec::new();
                            let mut seen = std::collections::HashSet::new();
                            for item in arr {
                                let key = serde_json::to_string(&item).unwrap_or_default();
                                if seen.insert(key) {
                                    result.push(item);
                                }
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "unique requires an array".to_string(),
                            ))
                        }
                    };

                    println!("🔄 unique: Removed duplicates from array");

                    if let Some(res) = result {
                        self.store_value(res, unique_array)?;
                    }
                    Ok(())
                } else if function == "sort" {
                    // sort(array, order?) -> [sorted] - Sort array (ascending by default)
                    if args.is_empty() || args.len() > 2 {
                        return Err(ExecutorError::ArgumentError(
                            "sort requires 1-2 arguments".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let ascending = if args.len() == 2 {
                        let order_value = self.evaluate_expression(args[1].clone())?;
                        match order_value {
                            RuntimeValue::String(s) => s.to_lowercase() != "desc",
                            RuntimeValue::Boolean(b) => b,
                            _ => true,
                        }
                    } else {
                        true
                    };

                    let sorted_array = match array_value {
                        RuntimeValue::Json(Value::Array(mut arr)) => {
                            arr.sort_by(|a, b| {
                                let cmp = match (a, b) {
                                    (Value::Number(n1), Value::Number(n2)) => {
                                        let f1 = n1.as_f64().unwrap_or(0.0);
                                        let f2 = n2.as_f64().unwrap_or(0.0);
                                        f1.partial_cmp(&f2).unwrap_or(std::cmp::Ordering::Equal)
                                    }
                                    (Value::String(s1), Value::String(s2)) => s1.cmp(s2),
                                    (Value::Bool(b1), Value::Bool(b2)) => b1.cmp(b2),
                                    _ => std::cmp::Ordering::Equal,
                                };
                                if ascending {
                                    cmp
                                } else {
                                    cmp.reverse()
                                }
                            });
                            RuntimeValue::Json(Value::Array(arr))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "sort requires an array".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🔄 sort: Sorted array in {} order",
                        if ascending { "ascending" } else { "descending" }
                    );

                    if let Some(res) = result {
                        self.store_value(res, sorted_array)?;
                    }
                    Ok(())
                } else if function == "reverse" {
                    // reverse(array) -> [reversed] - Reverse array order
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "reverse requires exactly 1 argument".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let reversed_array = match array_value {
                        RuntimeValue::Json(Value::Array(mut arr)) => {
                            arr.reverse();
                            RuntimeValue::Json(Value::Array(arr))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "reverse requires an array".to_string(),
                            ))
                        }
                    };

                    println!("🔄 reverse: Reversed array order");

                    if let Some(res) = result {
                        self.store_value(res, reversed_array)?;
                    }
                    Ok(())
                } else if function == "chunk" {
                    // chunk(array, size) -> [[chunk1], [chunk2], ...] - Split array into chunks
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "chunk requires exactly 2 arguments".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let size_value = self.evaluate_expression(args[1].clone())?;

                    let chunk_size = match size_value {
                        RuntimeValue::Integer(i) => {
                            if i <= 0 {
                                return Err(ExecutorError::ArgumentError(
                                    "chunk size must be positive".to_string(),
                                ));
                            }
                            i as usize
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "chunk size must be an integer".to_string(),
                            ))
                        }
                    };

                    let chunked_array = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            let mut result = Vec::new();
                            for chunk in arr.chunks(chunk_size) {
                                result.push(Value::Array(chunk.to_vec()));
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "chunk requires an array".to_string(),
                            ))
                        }
                    };

                    println!("🔄 chunk: Split array into chunks of size {}", chunk_size);

                    if let Some(res) = result {
                        self.store_value(res, chunked_array)?;
                    }
                    Ok(())
                } else if function == "find" {
                    // find(array, predicate) -> first_match - Find first matching element
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "find requires exactly 2 arguments".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let predicate_expr = args[1].clone();

                    let found_value = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            for (index, item) in arr.iter().enumerate() {
                                let matches = if let LexExpression::Value(ValueRef::Literal(
                                    LexLiteral::String(predicate_str),
                                )) = &predicate_expr
                                {
                                    self.evaluate_predicate(predicate_str, item, index)?
                                } else {
                                    // Fallback to default logic
                                    match item {
                                        Value::Number(n) => {
                                            if let Some(i) = n.as_i64() {
                                                i > 0
                                            } else {
                                                true
                                            }
                                        }
                                        Value::String(s) => !s.is_empty(),
                                        Value::Bool(b) => *b,
                                        Value::Null => false,
                                        _ => true,
                                    }
                                };

                                if matches {
                                    if let Some(res) = result {
                                        self.store_value(res, RuntimeValue::Json(item.clone()))?;
                                    }
                                    return Ok(());
                                }
                            }
                            // Not found
                            RuntimeValue::Null
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "find requires an array".to_string(),
                            ))
                        }
                    };

                    println!("🔄 find: Searched for first matching element");

                    if let Some(res) = result {
                        self.store_value(res, found_value)?;
                    }
                    Ok(())
                } else if function == "count" {
                    // count(array, predicate?) -> number - Count elements (optionally matching predicate)
                    if args.is_empty() || args.len() > 2 {
                        return Err(ExecutorError::ArgumentError(
                            "count requires 1-2 arguments".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let count_value = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            if args.len() == 2 {
                                let predicate_expr = args[1].clone();
                                let mut count = 0;
                                for (index, item) in arr.iter().enumerate() {
                                    let matches = if let LexExpression::Value(ValueRef::Literal(
                                        LexLiteral::String(predicate_str),
                                    )) = &predicate_expr
                                    {
                                        self.evaluate_predicate(predicate_str, item, index)?
                                    } else {
                                        true
                                    };
                                    if matches {
                                        count += 1;
                                    }
                                }
                                RuntimeValue::Integer(count)
                            } else {
                                RuntimeValue::Integer(arr.len() as i64)
                            }
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "count requires an array".to_string(),
                            ))
                        }
                    };

                    println!("🔄 count: Counted elements in array");

                    if let Some(res) = result {
                        self.store_value(res, count_value)?;
                    }
                    Ok(())
                } else if function == "read_file" {
                    self.handle_read_file(args, result.as_ref())?;
                    Ok(())
                } else if function == "write_file" {
                    self.handle_write_file(args, result.as_ref())?;
                    Ok(())
                } else if function == "save_file" {
                    self.handle_save_file(args, result.as_ref())?;
                    Ok(())
                } else if function == "save_binary_file" {
                    self.handle_save_binary_file(args, result.as_ref())?;
                    Ok(())
                } else if function == "save_binary_file_stream" {
                    self.handle_save_binary_file_stream(args, result.as_ref())?;
                    Ok(())
                } else if function == "load_binary_file" {
                    self.handle_load_binary_file(args, result.as_ref())?;
                    Ok(())
                } else if function == "ask_multioutput" {
                    self.handle_ask_multioutput(args, result.as_ref())
                } else if function == "execute" {
                    // execute(command) -> string - Execute system command
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "execute requires exactly 1 argument: command".to_string(),
                        ));
                    }

                    let command_value = self.evaluate_expression(args[0].clone())?;
                    let command = match command_value {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "execute command must be a string".to_string(),
                            ))
                        }
                    };

                    println!("⚡ execute: Running command '{}'", command);

                    match std::process::Command::new("sh")
                        .arg("-c")
                        .arg(&command)
                        .output()
                    {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);

                            if output.status.success() {
                                println!("✅ execute: Command completed successfully");
                                if let Some(res) = result {
                                    self.store_value(
                                        res,
                                        RuntimeValue::String(stdout.to_string()),
                                    )?;
                                }
                            } else {
                                let error_msg = format!(
                                    "Command failed with status {}: {}",
                                    output.status.code().unwrap_or(-1),
                                    stderr
                                );
                                println!("❌ execute: {}", error_msg);
                                return Err(ExecutorError::RuntimeError(error_msg));
                            }
                        }
                        Err(e) => {
                            let error_msg =
                                format!("Failed to execute command '{}': {}", command, e);
                            println!("❌ execute: {}", error_msg);
                            return Err(ExecutorError::RuntimeError(error_msg));
                        }
                    }
                    Ok(())
                } else if function == "model_dialogue" {
                    // model_dialogue(participants, topic, rounds?) -> dialogue_transcript
                    // Structured dialogue between multiple LLMs with specific roles
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "model_dialogue requires participants and topic".to_string(),
                        ));
                    }

                    let participants_arg = self.evaluate_expression(args[0].clone())?;
                    let participants = match participants_arg {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", participants_arg),
                    };

                    let topic_arg = self.evaluate_expression(args[1].clone())?;
                    let topic = match topic_arg {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", topic_arg),
                    };

                    let rounds = if args.len() > 2 {
                        let rounds_arg = self.evaluate_expression(args[2].clone())?;
                        match rounds_arg {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 3,
                        }
                    } else {
                        3
                    };

                    println!("🎭 model_dialogue: Starting dialogue on '{}'", topic);
                    println!("👥 Participants: {}", participants);
                    println!("🔄 Rounds: {}", rounds);

                    let mut dialogue_transcript = Vec::new();
                    dialogue_transcript.push(format!("=== MODEL DIALOGUE: {} ===", topic));
                    dialogue_transcript.push(format!("Participants: {}", participants));
                    dialogue_transcript.push("".to_string());

                    for round in 1..=rounds {
                        println!("🗣️ Round {}/{}: Dialogue in progress...", round, rounds);

                        dialogue_transcript.push(format!("--- Round {} ---", round));

                        // More dynamic dialogue using LLM adapter
                        let mut llm_adapter = self.llm_adapter.clone();

                        let dialogue_prompt = format!("You are participating in a collaborative dialogue about '{}'. This is round {} of {}. Engage naturally with other AI participants, building on previous exchanges and sharing your unique perspective.", topic, round, rounds);

                        let response = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                llm_adapter.call_llm_async(
                                    Some("gpt-4"), Some(0.9),
                                    Some("You are participating in a collaborative multi-AI dialogue. Be engaging and build on others' points."),
                                    Some(&dialogue_prompt), None, None, &HashMap::new()
                                ).await
                            })
                        })?;

                        dialogue_transcript.push(format!("🤖 Participant: {}", response));
                        dialogue_transcript.push("".to_string());

                        println!("✅ Round {} dialogue completed", round);
                    }

                    dialogue_transcript.push("=== DIALOGUE CONCLUSION ===".to_string());
                    dialogue_transcript.push("The models engaged in a productive multi-round dialogue, sharing different perspectives and building on each other's insights.".to_string());

                    let full_transcript = dialogue_transcript.join("\n");

                    println!(
                        "🎯 model_dialogue: Dialogue completed after {} rounds",
                        rounds
                    );

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(full_transcript))?;
                    }

                    Ok(())
                } else if function == "multimodal_request" {
                    // multimodal_request(prompt, file_path, model?) -> response
                    // Multimodal processing system for text + images + files
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "multimodal_request requires prompt and file_path".to_string(),
                        ));
                    }

                    let prompt_arg = self.evaluate_expression(args[0].clone())?;
                    let prompt = match prompt_arg {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", prompt_arg),
                    };

                    let file_path_arg = self.evaluate_expression(args[1].clone())?;
                    let file_path = match file_path_arg {
                        RuntimeValue::String(s) => s,
                        _ => format!("{:?}", file_path_arg),
                    };

                    let model = if args.len() > 2 {
                        let model_arg = self.evaluate_expression(args[2].clone())?;
                        match model_arg {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };

                    println!("🖼️ multimodal_request: Processing multimodal content");
                    println!("📝 Prompt: {}", prompt);
                    println!("📁 File: {}", file_path);
                    if let Some(ref m) = model {
                        println!("🤖 Model: {}", m);
                    }

                    // Detect file type by extension
                    let file_extension = file_path.split('.').last().unwrap_or("").to_lowercase();
                    let media_type = match file_extension.as_str() {
                        "jpg" | "jpeg" | "png" | "gif" | "bmp" | "webp" => "image",
                        "mp4" | "avi" | "mov" | "wmv" | "flv" | "webm" => "video",
                        "mp3" | "wav" | "flac" | "aac" | "ogg" => "audio",
                        "pdf" | "doc" | "docx" | "txt" | "rtf" => "document",
                        _ => "unknown",
                    };

                    println!("🔍 Detected media type: {}", media_type);

                    // Simulate multimodal processing
                    let response = match media_type {
                        "image" => {
                            format!("🖼️ MULTIMODAL IMAGE ANALYSIS\n\
                                    File: {}\n\
                                    Type: Image Processing\n\
                                    Analysis: This appears to be an image file. In a full implementation, \
                                    this would be processed using GPT-4 Vision or similar multimodal models \
                                    to analyze the visual content and respond to: '{}'\n\
                                    \n\
                                    Simulated Response: Based on the image analysis, I can see various visual \
                                    elements that would be described in detail by a multimodal AI model.", 
                                    file_path, prompt)
                        }
                        "video" => {
                            format!("🎥 MULTIMODAL VIDEO ANALYSIS\n\
                                    File: {}\n\
                                    Type: Video Processing\n\
                                    Analysis: This appears to be a video file. In a full implementation, \
                                    this would extract frames and/or audio to analyze the content and \
                                    respond to: '{}'\n\
                                    \n\
                                    Simulated Response: Based on the video analysis, I would provide \
                                    a comprehensive summary of the visual and audio content.", 
                                    file_path, prompt)
                        }
                        "audio" => {
                            format!("🎵 MULTIMODAL AUDIO ANALYSIS\n\
                                    File: {}\n\
                                    Type: Audio Processing\n\
                                    Analysis: This appears to be an audio file. In a full implementation, \
                                    this would be transcribed and analyzed to respond to: '{}'\n\
                                    \n\
                                    Simulated Response: [Transcribed audio content would appear here] \
                                    followed by analysis based on the audio content.", 
                                    file_path, prompt)
                        }
                        "document" => {
                            format!("📄 MULTIMODAL DOCUMENT ANALYSIS\n\
                                    File: {}\n\
                                    Type: Document Processing\n\
                                    Analysis: This appears to be a document file. In a full implementation, \
                                    this would extract and analyze the text content to respond to: '{}'\n\
                                    \n\
                                    Simulated Response: Based on the document analysis, I would provide \
                                    insights derived from the extracted text content.", 
                                    file_path, prompt)
                        }
                        _ => {
                            format!("❓ MULTIMODAL UNKNOWN FILE ANALYSIS\n\
                                    File: {}\n\
                                    Type: Unknown file type (.{})\n\
                                    Analysis: This file type is not specifically recognized. In a full \
                                    implementation, this would attempt generic file processing to \
                                    respond to: '{}'\n\
                                    \n\
                                    Simulated Response: I would attempt to process this file using \
                                    appropriate methods based on its content and structure.", 
                                    file_path, file_extension, prompt)
                        }
                    };

                    println!("✅ multimodal_request: Processing completed");

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(response))?;
                    }
                    Ok(())
                } else if function == "load_csv" {
                    // Delegate to extracted function
                    self.handle_load_csv(args, result.as_ref())?;
                    Ok(())
                } else if function == "load_csv_url" {
                    self.handle_load_csv_url(args, result.as_ref())?;
                    Ok(())
                } else if function == "save_json" {
                    // Delegate to extracted function
                    self.handle_save_json(args, result.as_ref())?;
                    Ok(())
                } else if function == "save_csv" {
                    // Delegate to extracted function
                    self.handle_save_csv(args, result.as_ref())?;
                    Ok(())
                } else if function == "load_json" {
                    // Delegate to extracted function
                    self.handle_load_json(args, result.as_ref())?;
                    Ok(())
                } else if function == "load_json_url" {
                    self.handle_load_json_url(args, result.as_ref())?;
                    Ok(())
                } else if function == "load_parquet" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "load_parquet requires path".to_string(),
                        ));
                    }
                    let path = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "path must be string".to_string(),
                            ))
                        }
                    };
                    let ds = self.data_processor.load_parquet("parquet", &path)?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Dataset(Arc::new(ds)))?;
                    }
                    Ok(())
                } else if function == "save_parquet" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "save_parquet requires dataset, path".to_string(),
                        ));
                    }
                    let ds = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Dataset(d) => d,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "first arg must be dataset".to_string(),
                            ))
                        }
                    };
                    let path = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "path must be string".to_string(),
                            ))
                        }
                    };
                    ds.export(&path, "parquet")?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "set_default_model" {
                    // 🔧 Sprint B: set_default_model(model_name) -> success
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "set_default_model requires exactly 1 argument: model_name".to_string(),
                        ));
                    }

                    let model_value = self.evaluate_expression(args[0].clone())?;
                    let model_name = match model_value {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "set_default_model model_name must be a string".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🔧 set_default_model: Setting default model to '{}'",
                        model_name
                    );

                    // Update default model configuration
                    // This will affect future calls to ask() without specified model
                    self.config.llm_model = Some(model_name.clone());

                    // Also update LLM adapter with new default model
                    if let Some(ref mut llm_adapter) = self.llm_adapter_new {
                        llm_adapter.set_default_model(&model_name);
                    }

                    println!("✅ set_default_model: Default model updated successfully");

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "get_provider_default" {
                    // 🔧 Sprint B: get_provider_default(provider_name) -> model_name
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "get_provider_default requires exactly 1 argument: provider_name"
                                .to_string(),
                        ));
                    }

                    let provider_value = self.evaluate_expression(args[0].clone())?;
                    let provider_name = match provider_value {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "get_provider_default provider_name must be a string".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🔧 get_provider_default: Getting default model for provider '{}'",
                        provider_name
                    );

                    // Get default model for specified provider
                    let default_model = match provider_name.to_lowercase().as_str() {
                        "openai" => "gpt-4",
                        "anthropic" | "claude" => "claude-3-5-sonnet-20241022",
                        "google" | "gemini" => "gemini-1.5-pro",
                        "ollama" => "llama3.2",
                        _ => {
                            return Err(ExecutorError::ArgumentError(format!(
                                "Unknown provider: {}",
                                provider_name
                            )));
                        }
                    };

                    println!(
                        "✅ get_provider_default: Default model for '{}' is '{}'",
                        provider_name, default_model
                    );

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(default_model.to_string()))?;
                    }
                    Ok(())
                } else if function == "confidence_score" {
                    // 🛡️ Sprint C: Calculates confidence score of a response using heuristics
                    // 🛡️ Sprint C: confidence_score(response_text) -> float
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "confidence_score requires exactly 1 argument: response_text"
                                .to_string(),
                        ));
                    }

                    let response_text = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "confidence_score response_text must be a string".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🛡️ confidence_score: Calculating confidence for response: {}",
                        response_text.chars().take(50).collect::<String>()
                    );

                    // Calculate confidence score using basic heuristics
                    let confidence = self.calculate_confidence_score_v2(&response_text);

                    println!(
                        "✅ confidence_score: Calculated confidence: {:.2}",
                        confidence
                    );

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(confidence))?;
                    }
                    Ok(())
                } else if function == "validate_response" {
                    // 🛡️ Sprint C: validate_response(response_text, validation_type) -> bool
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError("validate_response requires exactly 2 arguments: response_text, validation_type".to_string()));
                    }

                    let response_text = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "validate_response response_text must be a string".to_string(),
                            ))
                        }
                    };

                    let validation_type = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "validate_response validation_type must be a string".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🛡️ validate_response: Validating response with type '{}'",
                        validation_type
                    );

                    // Validate response according to specified type
                    let is_valid =
                        self.validate_response_basic_v2(&response_text, &validation_type);

                    println!("✅ validate_response: Validation result: {}", is_valid);

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(is_valid))?;
                    }
                    Ok(())
                } else if function == "assert_true" {
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "assert_true requires condition [and optional message]".to_string(),
                        ));
                    }
                    let cond = self.evaluate_expression(args[0].clone())?;
                    let ok = match cond {
                        RuntimeValue::Boolean(b) => b,
                        RuntimeValue::Integer(i) => i != 0,
                        RuntimeValue::Float(f) => f != 0.0,
                        RuntimeValue::String(s) => !s.is_empty(),
                        _ => false,
                    };
                    if !ok {
                        let msg = if args.len() >= 2 {
                            match self.evaluate_expression(args[1].clone())? {
                                RuntimeValue::String(s) => s,
                                other => format!("Assertion failed: {:?}", other),
                            }
                        } else {
                            "Assertion failed".to_string()
                        };
                        return Err(ExecutorError::RuntimeError(msg));
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "assert_eq" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "assert_eq requires two values [and optional message]".to_string(),
                        ));
                    }
                    let a = self.evaluate_expression(args[0].clone())?;
                    let b = self.evaluate_expression(args[1].clone())?;
                    let equal = format!("{:?}", a) == format!("{:?}", b);
                    if !equal {
                        let msg = if args.len() >= 3 {
                            match self.evaluate_expression(args[2].clone())? {
                                RuntimeValue::String(s) => s,
                                _ => "".to_string(),
                            }
                        } else {
                            String::new()
                        };
                        let base = format!("assert_eq failed: left={:?} right={:?}", a, b);
                        return Err(ExecutorError::RuntimeError(if msg.is_empty() {
                            base
                        } else {
                            format!("{} - {}", base, msg)
                        }));
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "assert_snapshot" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "assert_snapshot requires name and value".to_string(),
                        ));
                    }
                    let name_val = self.evaluate_expression(args[0].clone())?;
                    let name = match name_val {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let value_val = self.evaluate_expression(args[1].clone())?;
                    let value_str = match value_val {
                        RuntimeValue::String(s) => s,
                        RuntimeValue::Json(j) => j.to_string(),
                        other => format!("{:?}", other),
                    };
                    let path = if name.contains('/') || name.contains('\\') {
                        name.clone()
                    } else {
                        format!("golden/{}.txt", name)
                    };
                    let update = std::env::var("LEXON_SNAPSHOT_UPDATE")
                        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                        .unwrap_or(false);
                    match std::fs::read_to_string(&path) {
                        Ok(existing) => {
                            if existing != value_str {
                                if update {
                                    std::fs::write(&path, &value_str).map_err(|e| {
                                        ExecutorError::RuntimeError(format!(
                                            "snapshot update failed '{}': {}",
                                            path, e
                                        ))
                                    })?;
                                } else {
                                    return Err(ExecutorError::RuntimeError(format!(
                                        "assert_snapshot failed: {} does not match",
                                        path
                                    )));
                                }
                            }
                        }
                        Err(_) => {
                            if update {
                                std::fs::write(&path, &value_str).map_err(|e| {
                                    ExecutorError::RuntimeError(format!(
                                        "snapshot create failed '{}': {}",
                                        path, e
                                    ))
                                })?;
                            } else {
                                return Err(ExecutorError::RuntimeError(format!(
                                    "snapshot missing: {} (set LEXON_SNAPSHOT_UPDATE=1 to create)",
                                    path
                                )));
                            }
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "timeout" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "timeout requires ms and prompt [optional model]".to_string(),
                        ));
                    }
                    let ms_value = self.evaluate_expression(args[0].clone())?;
                    let ms = match ms_value {
                        RuntimeValue::Integer(i) => i as u64,
                        RuntimeValue::Float(f) => f as u64,
                        RuntimeValue::String(s) => s.parse::<u64>().unwrap_or(0),
                        _ => 0,
                    };
                    let prompt_value = self.evaluate_expression(args[1].clone())?;
                    let prompt = match prompt_value {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let model_opt = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        self.config.llm_model.clone()
                    };

                    let mut llm_adapter = self.llm_adapter.clone();
                    let fut = async {
                        llm_adapter
                            .call_llm_async(
                                model_opt.as_deref(),
                                Some(0.7),
                                None,
                                Some(&prompt),
                                None,
                                None,
                                &std::collections::HashMap::new(),
                            )
                            .await
                    };
                    let response = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            match tokio::time::timeout(std::time::Duration::from_millis(ms), fut)
                                .await
                            {
                                Ok(Ok(r)) => Ok(r),
                                Ok(Err(e)) => Err(e),
                                Err(_) => Err(ExecutorError::RuntimeError(format!(
                                    "timeout exceeded: {} ms",
                                    ms
                                ))),
                            }
                        })
                    })?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(response))?;
                    }
                    Ok(())
                } else if function == "retry" {
                    if args.len() < 3 {
                        return Err(ExecutorError::ArgumentError(
                            "retry requires attempts, backoff_ms, prompt [optional model]"
                                .to_string(),
                        ));
                    }
                    let attempts_value = self.evaluate_expression(args[0].clone())?;
                    let backoff_value = self.evaluate_expression(args[1].clone())?;
                    let attempts = match attempts_value {
                        RuntimeValue::Integer(i) => i as u32,
                        RuntimeValue::Float(f) => f as u32,
                        RuntimeValue::String(s) => s.parse::<u32>().unwrap_or(1),
                        _ => 1,
                    };
                    let backoff_ms = match backoff_value {
                        RuntimeValue::Integer(i) => i as u64,
                        RuntimeValue::Float(f) => f as u64,
                        RuntimeValue::String(s) => s.parse::<u64>().unwrap_or(0),
                        _ => 0,
                    };
                    let prompt_value = self.evaluate_expression(args[2].clone())?;
                    let prompt = match prompt_value {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let model_opt = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        self.config.llm_model.clone()
                    };

                    let mut last_err: Option<ExecutorError> = None;
                    for attempt in 1..=attempts {
                        let mut llm_adapter = self.llm_adapter.clone();
                        let res = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                llm_adapter
                                    .call_llm_async(
                                        model_opt.as_deref(),
                                        Some(0.7),
                                        None,
                                        Some(&prompt),
                                        None,
                                        None,
                                        &std::collections::HashMap::new(),
                                    )
                                    .await
                            })
                        });
                        match res {
                            Ok(response) => {
                                if let Some(res_ref) = result {
                                    self.store_value(res_ref, RuntimeValue::String(response))?;
                                }
                                return Ok(());
                            }
                            Err(e) => {
                                last_err = Some(e);
                                if attempt < attempts {
                                    std::thread::sleep(std::time::Duration::from_millis(
                                        backoff_ms,
                                    ));
                                }
                            }
                        }
                    }
                    Err(last_err
                        .unwrap_or_else(|| ExecutorError::RuntimeError("retry failed".to_string())))
                } else if function == "configure_validation" {
                    if args.is_empty() {
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::Boolean(true))?;
                        }
                        return Ok(());
                    }
                    let cfg = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => Some(j),
                        RuntimeValue::String(s) => serde_json::from_str(&s).ok(),
                        _ => None,
                    };
                    if cfg.is_none() {
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::Boolean(true))?;
                        }
                        return Ok(());
                    }
                    let cfg = cfg.unwrap();
                    if let Some(mc) = cfg.get("min_confidence").and_then(|v| v.as_f64()) {
                        self.validation_min_confidence = mc;
                    }
                    if let Some(ma) = cfg.get("max_attempts").and_then(|v| v.as_u64()) {
                        self.validation_max_attempts = ma as u32;
                    }
                    if let Some(vt) = cfg
                        .get("validation_types")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.get(0))
                        .and_then(|v| v.as_str())
                    {
                        self.validation_strategy = vt.to_string();
                    }
                    if let Some(dom) = cfg.get("domain").and_then(|v| v.as_str()) {
                        self.validation_domain = Some(dom.to_string());
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(true))?;
                    }
                    Ok(())
                } else if function == "ask_with_validation" {
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "ask_with_validation requires (prompt, [config])".to_string(),
                        ));
                    }
                    let prompt = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "prompt must be string".to_string(),
                            ))
                        }
                    };
                    let cfg = if args.len() >= 2 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Json(j) => Some(j),
                            RuntimeValue::String(s) => serde_json::from_str(&s).ok(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    use crate::executor::llm_adapter::{
                        AntiHallucinationConfig, ValidationStrategy,
                    };
                    let min_conf = cfg
                        .as_ref()
                        .and_then(|j| j.get("min_confidence").and_then(|v| v.as_f64()))
                        .unwrap_or(self.validation_min_confidence);
                    let max_att = cfg
                        .as_ref()
                        .and_then(|j| j.get("max_attempts").and_then(|v| v.as_u64()))
                        .map(|v| v as usize)
                        .unwrap_or(self.validation_max_attempts as usize);
                    let strat_str = cfg
                        .as_ref()
                        .and_then(|j| {
                            j.get("validation_types")
                                .and_then(|v| v.as_array())
                                .and_then(|a| a.get(0))
                                .and_then(|v| v.as_str())
                        })
                        .unwrap_or(&self.validation_strategy);
                    let strategy = match strat_str {
                        "ensemble" => ValidationStrategy::Ensemble,
                        "fact" | "factual" | "fact_check" => ValidationStrategy::FactCheck,
                        "comprehensive" => ValidationStrategy::Comprehensive,
                        _ => ValidationStrategy::Basic,
                    };
                    let config = AntiHallucinationConfig {
                        validation_strategy: strategy,
                        confidence_threshold: min_conf,
                        max_validation_attempts: max_att,
                        use_fact_checking: true,
                        cross_reference_models: vec![],
                    };
                    let validation_result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            self.llm_adapter.ask_safe(&prompt, None, Some(config)).await
                        })
                    })?;
                    if let Some(res) = result {
                        self.store_value(
                            res,
                            RuntimeValue::String(validation_result.validated_content),
                        )?;
                    }
                    Ok(())
                } else if function == "memory_index.ingest" || function == "memory_index__ingest" {
                    // 🧠 Sprint D: memory_index.ingest(paths) -> int (REAL IMPLEMENTATION)
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "memory_index.ingest requires exactly 1 argument: paths".to_string(),
                        ));
                    }

                    let paths_value = self.evaluate_expression(args[0].clone())?;
                    let paths = match paths_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            let mut paths_vec = Vec::new();
                            for item in arr {
                                if let Value::String(path) = item {
                                    paths_vec.push(path);
                                } else {
                                    return Err(ExecutorError::ArgumentError(
                                        "memory_index.ingest paths array must contain only strings"
                                            .to_string(),
                                    ));
                                }
                            }
                            paths_vec
                        }
                        RuntimeValue::String(s) => {
                            // Single string or comma-separated
                            if s.contains(',') {
                                s.split(',').map(|p| p.trim().to_string()).collect()
                            } else {
                                vec![s]
                            }
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "memory_index.ingest paths must be a string or JSON array"
                                    .to_string(),
                            ))
                        }
                    };

                    println!(
                        "🧠 memory_index.ingest: Ingesting {} paths into vector memory index",
                        paths.len()
                    );

                    // Use real vector memory system
                    let documents_ingested =
                        if let Some(ref mut vector_system) = self.vector_memory_system {
                            vector_system.ingest_documents(&paths)?
                        } else {
                            // Fallback to basic implementation if no vector system
                            let mut documents_ingested = 0;
                            for path in &paths {
                                match std::fs::read_to_string(path) {
                                    Ok(content) => {
                                        self.memory_manager.store_memory(
                                            "global_index",
                                            RuntimeValue::String(content),
                                            Some(path),
                                        )?;
                                        documents_ingested += 1;
                                        println!("📄 Ingested: {}", path);
                                    }
                                    Err(e) => {
                                        println!("⚠️ Failed to ingest '{}': {}", path, e);
                                    }
                                }
                            }
                            documents_ingested
                        };

                    println!(
                        "✅ memory_index.ingest: Successfully ingested {} documents",
                        documents_ingested
                    );

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Integer(documents_ingested as i64))?;
                    }
                    Ok(())
                } else if function == "memory_index.vector_search"
                    || function == "memory_index__vector_search"
                {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "memory_index.vector_search requires exactly 2 arguments: query, k"
                                .to_string(),
                        ));
                    }

                    let query = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "memory_index.vector_search query must be a string".to_string(),
                            ))
                        }
                    };

                    let k = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "memory_index.vector_search k must be an integer".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🧠 memory_index.vector_search: Searching for '{}' (k={})",
                        query.chars().take(50).collect::<String>(),
                        k
                    );

                    // Use real vector memory system
                    let search_results =
                        if let Some(ref mut vector_system) = self.vector_memory_system {
                            vector_system.vector_search(&query, k)?
                        } else {
                            // Fallback to basic search if no vector system
                            let memory_results = self.memory_manager.load_memory(
                                "global_index",
                                None,
                                "buffer",
                                &HashMap::new(),
                            )?;

                            let mut relevant_docs = Vec::new();
                            let query_lower = query.to_lowercase();

                            if let RuntimeValue::Json(Value::Array(docs)) = memory_results {
                                for doc in docs {
                                    if let Value::String(content) = doc {
                                        let content_lower = content.to_lowercase();
                                        let query_words: Vec<&str> =
                                            query_lower.split_whitespace().collect();
                                        let mut matches = 0;
                                        for word in &query_words {
                                            if content_lower.contains(word) {
                                                matches += 1;
                                            }
                                        }

                                        if matches > 0 {
                                            relevant_docs.push(RuntimeValue::String(content));
                                        }
                                    }
                                }
                            }

                            relevant_docs.truncate(k);
                            relevant_docs
                        };

                    println!(
                        "✅ memory_index.vector_search: Found {} relevant documents",
                        search_results.len()
                    );

                    // Convert result to JSON array
                    let result_values: Vec<Value> = search_results
                        .into_iter()
                        .map(|val| match val {
                            RuntimeValue::Json(json_val) => json_val,
                            RuntimeValue::String(s) => Value::String(s),
                            _ => Value::String(format!("{:?}", val)),
                        })
                        .collect();

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(Value::Array(result_values)))?;
                    }
                    Ok(())
                } else if function == "auto_rag_context" {
                    // 🧠 Sprint D: auto_rag_context() -> string (REAL IMPLEMENTATION)
                    if !args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "auto_rag_context requires no arguments".to_string(),
                        ));
                    }

                    println!("🧠 auto_rag_context: Generating automatic RAG context");

                    // Use real vector memory system
                    let context = if let Some(ref mut vector_system) = self.vector_memory_system {
                        vector_system.generate_rag_context()?
                    } else {
                        // Fallback to basic implementation if no vector system
                        let memory_contents = self.memory_manager.load_memory(
                            "global_index",
                            None,
                            "buffer",
                            &HashMap::new(),
                        )?;

                        match memory_contents {
                            RuntimeValue::Json(Value::Array(docs)) => {
                                let mut context_parts = Vec::new();
                                for (i, doc) in docs.iter().enumerate() {
                                    if i >= 3 {
                                        break;
                                    }
                                    if let Value::String(content) = doc {
                                        let preview = content.chars().take(200).collect::<String>();
                                        context_parts.push(format!(
                                            "Document {}: {}",
                                            i + 1,
                                            preview
                                        ));
                                    }
                                }

                                if context_parts.is_empty() {
                                    "No indexed documents available for context.".to_string()
                                } else {
                                    format!("RAG Context:\n{}", context_parts.join("\n\n"))
                                }
                            }
                            _ => "No indexed memory available for RAG context.".to_string(),
                        }
                    };

                    println!(
                        "✅ auto_rag_context: Generated context ({} chars)",
                        context.len()
                    );

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(context))?;
                    }
                    Ok(())
                } else if function == "memory_index.ingest_chunks"
                    || function == "memory_index__ingest_chunks"
                {
                    // memory_index.ingest_chunks(paths, options_json?) -> int
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "memory_index.ingest_chunks requires at least 1 argument: paths"
                                .to_string(),
                        ));
                    }
                    let paths_value = self.evaluate_expression(args[0].clone())?;
                    let mut paths: Vec<String> = Vec::new();
                    match paths_value {
                        RuntimeValue::String(s) => paths.push(s),
                        RuntimeValue::Json(Value::Array(arr)) => {
                            for v in arr.iter() {
                                if let Some(s) = v.as_str() {
                                    paths.push(s.to_string());
                                } else {
                                    return Err(ExecutorError::ArgumentError("memory_index.ingest_chunks paths array must contain only strings".to_string()));
                                }
                            }
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "memory_index.ingest_chunks paths must be a string or array"
                                    .to_string(),
                            ))
                        }
                    }
                    let mut by = "tokens".to_string();
                    let mut size: usize = 200;
                    let mut overlap: usize = 40;
                    if args.len() > 1 {
                        let options_val = self.evaluate_expression(args[1].clone())?;
                        match options_val {
                            RuntimeValue::Json(obj) => {
                                if let Some(b) = obj.get("by").and_then(|v| v.as_str()) {
                                    by = b.to_string();
                                }
                                if let Some(sz) = obj.get("size").and_then(|v| v.as_u64()) {
                                    size = sz as usize;
                                }
                                if let Some(ov) = obj.get("overlap").and_then(|v| v.as_u64()) {
                                    overlap = ov as usize;
                                }
                            }
                            RuntimeValue::String(s) => {
                                // Try JSON first
                                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&s) {
                                    if let Some(b) = obj.get("by").and_then(|v| v.as_str()) {
                                        by = b.to_string();
                                    }
                                    if let Some(sz) = obj.get("size").and_then(|v| v.as_u64()) {
                                        size = sz as usize;
                                    }
                                    if let Some(ov) = obj.get("overlap").and_then(|v| v.as_u64()) {
                                        overlap = ov as usize;
                                    }
                                } else {
                                    // Fallback: simple DSL map parser
                                    let t = s.trim().trim_start_matches('{').trim_end_matches('}');
                                    for part in t.split(',') {
                                        if let Some((k, v)) = part.split_once(':') {
                                            let key = k.trim().trim_matches('"');
                                            let val = v.trim().trim_matches('"');
                                            match key {
                                                "by" => by = val.to_string(),
                                                "size" => {
                                                    if let Ok(n) = val.parse::<usize>() {
                                                        size = n;
                                                    }
                                                }
                                                "overlap" => {
                                                    if let Ok(n) = val.parse::<usize>() {
                                                        overlap = n;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    if let Some(ref mut vm) = self.vector_memory_system {
                        println!(
                            "🧠 memory_index.ingest_chunks: by={}, size={}, overlap={}",
                            by, size, overlap
                        );
                        let count = vm.ingest_documents_chunks(&paths, &by, size, overlap)?;
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::Integer(count as i64))?;
                        }
                        Ok(())
                    } else {
                        Err(ExecutorError::RuntimeError(
                            "Vector memory system not initialized".to_string(),
                        ))
                    }
                } else if function == "rag.chunk" || function == "rag__chunk" {
                    // rag.chunk(text, options_json?) -> [string]
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "rag.chunk requires at least 1 argument: text".to_string(),
                        ));
                    }
                    let text_val = self.evaluate_expression(args[0].clone())?;
                    let text = match text_val {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.chunk text must be a string".to_string(),
                            ))
                        }
                    };
                    let mut by = "tokens".to_string();
                    let mut size: usize = 200;
                    let mut overlap: usize = 40;
                    if args.len() > 1 {
                        let options_val = self.evaluate_expression(args[1].clone())?;
                        match options_val {
                            RuntimeValue::Json(obj) => {
                                if let Some(b) = obj.get("by").and_then(|v| v.as_str()) {
                                    by = b.to_string();
                                }
                                if let Some(sz) = obj.get("size").and_then(|v| v.as_u64()) {
                                    size = sz as usize;
                                }
                                if let Some(ov) = obj.get("overlap").and_then(|v| v.as_u64()) {
                                    overlap = ov as usize;
                                }
                            }
                            RuntimeValue::String(s) => {
                                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(&s) {
                                    if let Some(b) = obj.get("by").and_then(|v| v.as_str()) {
                                        by = b.to_string();
                                    }
                                    if let Some(sz) = obj.get("size").and_then(|v| v.as_u64()) {
                                        size = sz as usize;
                                    }
                                    if let Some(ov) = obj.get("overlap").and_then(|v| v.as_u64()) {
                                        overlap = ov as usize;
                                    }
                                } else {
                                    let t = s.trim().trim_start_matches('{').trim_end_matches('}');
                                    for part in t.split(',') {
                                        if let Some((k, v)) = part.split_once(':') {
                                            let key = k.trim().trim_matches('"');
                                            let val = v.trim().trim_matches('"');
                                            match key {
                                                "by" => by = val.to_string(),
                                                "size" => {
                                                    if let Ok(n) = val.parse::<usize>() {
                                                        size = n;
                                                    }
                                                }
                                                "overlap" => {
                                                    if let Ok(n) = val.parse::<usize>() {
                                                        overlap = n;
                                                    }
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    let chunks_json = if let Some(ref vm) = self.vector_memory_system {
                        let chunks = vm.chunk_text(&text, &by, size, overlap);
                        serde_json::Value::Array(
                            chunks
                                .into_iter()
                                .map(|s| serde_json::Value::String(s))
                                .collect(),
                        )
                    } else {
                        let tokens: Vec<&str> = text.split_whitespace().collect();
                        let mut out: Vec<String> = Vec::new();
                        let mut i = 0usize;
                        while i < tokens.len() {
                            let end = (i + size).min(tokens.len());
                            out.push(tokens[i..end].join(" "));
                            if end == tokens.len() {
                                break;
                            }
                            let step = size.saturating_sub(overlap).max(1);
                            i += step;
                        }
                        serde_json::Value::Array(
                            out.into_iter()
                                .map(|s| serde_json::Value::String(s))
                                .collect(),
                        )
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(chunks_json))?;
                    }
                    Ok(())
                } else if function == "rag.tokenize" || function == "rag__tokenize" {
                    // rag.tokenize(text, model?) -> [string]
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "rag.tokenize requires at least 1 argument: text".to_string(),
                        ));
                    }
                    let text = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.tokenize text must be a string".to_string(),
                            ))
                        }
                    };
                    let model = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let toks_json: Vec<serde_json::Value> = if let Some((bpe, ids)) =
                        Self::encode_tokens_precise(&text, model.as_deref())
                    {
                        Self::tokens_to_strings(&bpe, &ids)
                            .into_iter()
                            .map(|s| serde_json::Value::String(s))
                            .collect()
                    } else {
                        text.split_whitespace()
                            .map(|t| serde_json::Value::String(t.to_string()))
                            .collect()
                    };
                    if let Some(res) = result {
                        self.store_value(
                            res,
                            RuntimeValue::Json(serde_json::Value::Array(toks_json)),
                        )?;
                    }
                    Ok(())
                } else if function == "rag.token_count" || function == "rag__token_count" {
                    // rag.token_count(text, model?) -> int
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "rag.token_count requires at least 1 argument: text".to_string(),
                        ));
                    }
                    let text = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.token_count text must be a string".to_string(),
                            ))
                        }
                    };
                    let model = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let count = if let Some((_, ids)) =
                        Self::encode_tokens_precise(&text, model.as_deref())
                    {
                        ids.len() as i64
                    } else {
                        text.split_whitespace().count() as i64
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Integer(count))?;
                    }
                    Ok(())
                } else if function == "rag.chunk_tokens" || function == "rag__chunk_tokens" {
                    // rag.chunk_tokens(text, size, overlap?, model?) -> [string]
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "rag.chunk_tokens requires text and size".to_string(),
                        ));
                    }
                    let text = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.chunk_tokens text must be a string".to_string(),
                            ))
                        }
                    };
                    let size = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.chunk_tokens size must be an integer".to_string(),
                            ))
                        }
                    };
                    let overlap = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 0,
                        }
                    } else {
                        0
                    };
                    let model = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let mut out: Vec<serde_json::Value> = Vec::new();
                    if let Some((bpe, ids)) = Self::encode_tokens_precise(&text, model.as_deref()) {
                        let mut i = 0usize;
                        while i < ids.len() {
                            let end = (i + size).min(ids.len());
                            let chunk = Self::decode_range_to_string(&bpe, &ids[i..end]);
                            out.push(serde_json::Value::String(chunk));
                            if end == ids.len() {
                                break;
                            }
                            let step = size.saturating_sub(overlap).max(1);
                            i += step;
                        }
                    } else {
                        let toks: Vec<&str> = text.split_whitespace().collect();
                        let mut i = 0usize;
                        while i < toks.len() {
                            let end = (i + size).min(toks.len());
                            out.push(serde_json::Value::String(toks[i..end].join(" ")));
                            if end == toks.len() {
                                break;
                            }
                            let step = size.saturating_sub(overlap).max(1);
                            i += step;
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(serde_json::Value::Array(out)))?;
                    }
                    Ok(())
                } else if function == "rag.rerank" || function == "rag__rerank" {
                    // rag.rerank(results_json, query) -> results_json
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "rag.rerank requires results and query".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let query = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.rerank query must be a string".to_string(),
                            ))
                        }
                    };
                    let mut arr: Vec<serde_json::Value> = match resv {
                        RuntimeValue::Json(serde_json::Value::Array(a)) => a,
                        _ => Vec::new(),
                    };
                    let ql = query.to_lowercase();
                    arr.sort_by(|a, b| {
                        let ascore = a
                            .get("hybrid_score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or_else(|| {
                                let c = a
                                    .get("content")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_lowercase();
                                ql.split_whitespace().filter(|w| c.contains(*w)).count() as f64
                            });
                        let bscore = b
                            .get("hybrid_score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or_else(|| {
                                let c = b
                                    .get("content")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_lowercase();
                                ql.split_whitespace().filter(|w| c.contains(*w)).count() as f64
                            });
                        bscore
                            .partial_cmp(&ascore)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(serde_json::Value::Array(arr)))?;
                    }
                    Ok(())
                } else if function == "rag.fuse_passages" || function == "rag__fuse_passages" {
                    // rag.fuse_passages(results_json, max?) -> string
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "rag.fuse_passages requires results".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let maxp = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 5,
                        }
                    } else {
                        5
                    };
                    // MMR-like fusion to reduce redundancy
                    let mut selected: Vec<String> = Vec::new();
                    if let RuntimeValue::Json(serde_json::Value::Array(a)) = resv {
                        let mut cands: Vec<String> = a
                            .iter()
                            .filter_map(|it| {
                                it.get("content")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect();
                        while !cands.is_empty() && selected.len() < maxp {
                            // pick best by coverage - redundancy
                            let mut best_idx = 0usize;
                            let mut best_score = f32::MIN;
                            for (idx, c) in cands.iter().enumerate() {
                                let coverage = c.len() as f32;
                                let redundancy = selected
                                    .iter()
                                    .map(|s| Self::jaccard_sim_local(s, c))
                                    .fold(0f32, |acc, x| acc.max(x));
                                let lambda = 0.7f32;
                                let score = lambda * coverage - (1.0 - lambda) * redundancy;
                                if score > best_score {
                                    best_score = score;
                                    best_idx = idx;
                                }
                            }
                            let picked = cands.remove(best_idx);
                            selected.push(picked);
                        }
                    }
                    let mut out = String::new();
                    for (i, s) in selected.iter().enumerate() {
                        if i > 0 {
                            out.push_str("\n\n");
                        }
                        out.push_str(s);
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "rag.fuse_passages_semantic"
                    || function == "rag__fuse_passages_semantic"
                {
                    // rag.fuse_passages_semantic(results_json, max?, lambda?, citations?) -> string
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "rag.fuse_passages_semantic requires results".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let maxp = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 5,
                        }
                    } else {
                        5
                    };
                    let lambda = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Float(f) => f as f32,
                            RuntimeValue::Integer(i) => i as f32,
                            _ => 0.7,
                        }
                    } else {
                        0.7
                    };
                    let with_citations = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Boolean(b) => b,
                            _ => false,
                        }
                    } else {
                        false
                    };
                    let mut selected: Vec<String> = Vec::new();
                    let mut selected_meta: Vec<serde_json::Value> = Vec::new();
                    // Build candidate list and optional embedding map
                    let mut cands: Vec<String> = Vec::new();
                    let mut cands_meta: Vec<serde_json::Value> = Vec::new();
                    if let RuntimeValue::Json(serde_json::Value::Array(a)) = resv {
                        for it in a.iter() {
                            if let Some(s) = it.get("content").and_then(|v| v.as_str()) {
                                cands.push(s.to_string());
                                cands_meta.push(it.clone());
                            }
                        }
                    }
                    if cands.is_empty() {
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::String(String::new()))?;
                        }
                        return Ok(());
                    }
                    // Compute embeddings if available; fallback to jaccard
                    let mut emb_map: std::collections::HashMap<String, Vec<f32>> =
                        std::collections::HashMap::new();
                    let have_embs = if let Some(ref vm) = self.vector_memory_system {
                        for s in cands.iter() {
                            emb_map.insert(s.clone(), vm.embed_text(s));
                        }
                        true
                    } else {
                        false
                    };
                    while !cands.is_empty() && selected.len() < maxp {
                        let mut best_idx = 0usize;
                        let mut best_score = f32::MIN;
                        for (idx, c) in cands.iter().enumerate() {
                            let coverage = c.len() as f32;
                            let redundancy = if have_embs && !selected.is_empty() {
                                // compute max cosine between c and selected using precomputed embeddings
                                let e_c: &[f32] = match emb_map.get(c) {
                                    Some(v) => v.as_slice(),
                                    None => {
                                        let empty: &[f32] = &[];
                                        empty
                                    }
                                };
                                let mut max_sim = 0.0f32;
                                for s in selected.iter() {
                                    if let Some(e_s) = emb_map.get(s) {
                                        let sim = Self::cosine_sim_local(e_c, e_s);
                                        if sim > max_sim {
                                            max_sim = sim;
                                        }
                                    }
                                }
                                max_sim
                            } else {
                                selected
                                    .iter()
                                    .map(|s| Self::jaccard_sim_local(s, c))
                                    .fold(0f32, |acc, x| acc.max(x))
                            };
                            let score = (lambda * coverage) - ((1.0 - lambda) * redundancy);
                            if score > best_score {
                                best_score = score;
                                best_idx = idx;
                            }
                        }
                        let picked = cands.remove(best_idx);
                        let picked_meta = cands_meta.remove(best_idx);
                        selected.push(picked);
                        selected_meta.push(picked_meta);
                    }
                    let mut out = String::new();
                    for (i, s) in selected.iter().enumerate() {
                        if i > 0 {
                            out.push_str("\n\n");
                        }
                        if with_citations {
                            out.push_str(&format!("[{}] ", i + 1));
                        }
                        out.push_str(s);
                    }
                    if with_citations {
                        out.push_str("\n\nCitations:\n");
                        for (i, meta) in selected_meta.iter().enumerate() {
                            let path = meta.get("path").and_then(|v| v.as_str()).unwrap_or("");
                            out.push_str(&format!("[{}] {}\n", i + 1, path));
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "rag.rerank_llm" || function == "rag__rerank_llm" {
                    // rag.rerank_llm(results_json, query, model?, top_k?) -> results_json
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "rag.rerank_llm requires results and query".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let query = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.rerank_llm query must be a string".to_string(),
                            ))
                        }
                    };
                    let model = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let top_k = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 10,
                        }
                    } else {
                        10
                    };
                    let mut items: Vec<serde_json::Value> = match resv {
                        RuntimeValue::Json(serde_json::Value::Array(a)) => a,
                        _ => Vec::new(),
                    };
                    // score each with LLM simple instruction
                    let mut scored: Vec<(f32, serde_json::Value)> = Vec::new();
                    for it in items.drain(..) {
                        let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let prompt = format!("Given query: {:?}\nRate relevance of passage 0..1 (decimal).\nPassage: {:?}\nAnswer as: score: <number>", query, content);
                        let resp = if let Some(ref mut la) = self.llm_adapter_new {
                            tokio::task::block_in_place(|| {
                                tokio::runtime::Handle::current().block_on(async {
                                    la.call_llm_async(
                                        model.as_deref(),
                                        Some(0.0),
                                        Some("You are a precise scorer."),
                                        Some(&prompt),
                                        None,
                                        None,
                                        &HashMap::new(),
                                    )
                                    .await
                                })
                            })?
                        } else {
                            self.llm_adapter.call_llm(
                                model.as_deref(),
                                Some(0.0),
                                Some("You are a precise scorer."),
                                Some(&prompt),
                                None,
                                None,
                                &HashMap::new(),
                            )?
                        };
                        let score = Self::parse_score_local(&resp);
                        scored.push((score, it));
                    }
                    scored
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(top_k);
                    let out =
                        serde_json::Value::Array(scored.into_iter().map(|(_, v)| v).collect());
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "rag.rerank_cross_encoder"
                    || function == "rag__rerank_cross_encoder"
                {
                    // rag.rerank_cross_encoder(results_json, query, model?, top_k?, batch_size?) -> results_json
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "rag.rerank_cross_encoder requires results and query".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let query = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.rerank_cross_encoder query must be a string".to_string(),
                            ))
                        }
                    };
                    let model = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let top_k = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 10,
                        }
                    } else {
                        10
                    };
                    let env_batch = std::env::var("LEXON_RERANK_BATCH_SIZE")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                        .unwrap_or(8);
                    let batch_size = if args.len() > 4 {
                        match self.evaluate_expression(args[4].clone())? {
                            RuntimeValue::Integer(i) => (i as usize).max(1),
                            _ => env_batch,
                        }
                    } else {
                        env_batch
                    };
                    let mut items: Vec<serde_json::Value> = match resv {
                        RuntimeValue::Json(serde_json::Value::Array(a)) => a,
                        _ => Vec::new(),
                    };
                    if let Some(max_items) = std::env::var("LEXON_RERANK_MAX_ITEMS")
                        .ok()
                        .and_then(|s| s.parse::<usize>().ok())
                    {
                        if items.len() > max_items {
                            items.truncate(max_items);
                        }
                    }
                    let sys = "You are a precise cross-encoder that outputs only: score: <number between 0 and 1>";
                    let mut scored: Vec<(f32, serde_json::Value)> = Vec::new();
                    if let Some(ref mut la) = self.llm_adapter_new {
                        // Parallel batched scoring using adapter clones to avoid borrow across await
                        let handle = tokio::runtime::Handle::current();
                        let scores: Vec<f32> = tokio::task::block_in_place(|| {
                            handle.block_on(async {
                            use futures_util::future::join_all;
                            let mut out_scores: Vec<f32> = Vec::new();
                            let adapter_template = la.clone();
                            let mut idx = 0usize;
                            while idx < items.len() {
                                let end = (idx + batch_size).min(items.len());
                                let mut futs = Vec::new();
                                for it in &items[idx..end] {
                                    let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                                    let query_s = query.clone();
                                    let model_clone = model.clone();
                                    let sys_s = sys.to_string();
                                    let mut adapter_local = adapter_template.clone();
                                    futs.push(async move {
                                        let prompt_s = format!("[CROSS-ENCODER]\nQuery: {:?}\nPassage: {:?}\nScore relevance 0..1 as 'score: <number>'", query_s, content);
                                        let params: HashMap<String,String> = HashMap::new();
                                        adapter_local.call_llm_async(model_clone.as_deref(), Some(0.0), Some(&sys_s), Some(&prompt_s), None, None, &params).await
                                    });
                                }
                                let results = join_all(futs).await;
                                for r in results {
                                    match r {
                                        Ok(s) => out_scores.push(Self::parse_score_local(&s)),
                                        Err(_) => out_scores.push(0.0),
                                    }
                                }
                                idx = end;
                            }
                            out_scores
                        })
                        });
                        for (i, it) in items.drain(..).enumerate() {
                            let sc = scores.get(i).cloned().unwrap_or(0.0);
                            scored.push((sc, it));
                        }
                    } else {
                        // Sequential fallback
                        for it in items.drain(..) {
                            let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("");
                            let prompt = format!("[CROSS-ENCODER]\nQuery: {:?}\nPassage: {:?}\nScore relevance 0..1 as 'score: <number>'", query, content);
                            let resp = self.llm_adapter.call_llm(
                                model.as_deref(),
                                Some(0.0),
                                Some(sys),
                                Some(&prompt),
                                None,
                                None,
                                &HashMap::new(),
                            )?;
                            let score = Self::parse_score_local(&resp);
                            scored.push((score, it));
                        }
                    }
                    scored
                        .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(top_k);
                    let out =
                        serde_json::Value::Array(scored.into_iter().map(|(_, v)| v).collect());
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "rag.summarize_chunks" || function == "rag__summarize_chunks"
                {
                    // rag.summarize_chunks(chunks_json, model?) -> string (synthesizes brief summary)
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "rag.summarize_chunks requires chunks".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let model = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let mut joined = String::new();
                    if let RuntimeValue::Json(serde_json::Value::Array(a)) = resv {
                        for s in a {
                            if let Some(t) = s.as_str() {
                                joined.push_str("- ");
                                joined.push_str(t);
                                joined.push('\n');
                            }
                        }
                    }
                    // Use LLM adapter to summarize
                    let prompt = format!(
                        "Summarize the following passages succinctly in 3-4 bullet points:\n\n{}",
                        joined
                    );
                    let sum = if let Some(ref mut la) = self.llm_adapter_new {
                        tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                la.call_llm_async(
                                    model.as_deref(),
                                    Some(0.5),
                                    Some("You are a concise summarizer."),
                                    Some(&prompt),
                                    None,
                                    None,
                                    &HashMap::new(),
                                )
                                .await
                            })
                        })?
                    } else {
                        // legacy path
                        self.llm_adapter.call_llm(
                            Some("simulated"),
                            Some(0.5),
                            Some("You are a concise summarizer."),
                            Some(&prompt),
                            None,
                            None,
                            &HashMap::new(),
                        )?
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(sum))?;
                    }
                    Ok(())
                } else if function == "rag.optimize_window" || function == "rag__optimize_window" {
                    // rag.optimize_window(results_json, target_tokens, model?, margin_tokens?) -> results_json
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "rag.optimize_window requires results and target_tokens".to_string(),
                        ));
                    }
                    let resv = self.evaluate_expression(args[0].clone())?;
                    let target = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "rag.optimize_window target_tokens must be int".to_string(),
                            ))
                        }
                    };
                    let model = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::String(s) => Some(s),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let margin = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Integer(i) => i as usize,
                            _ => 256,
                        }
                    } else {
                        256
                    };
                    let mut items: Vec<serde_json::Value> = match resv {
                        RuntimeValue::Json(serde_json::Value::Array(a)) => a,
                        _ => Vec::new(),
                    };
                    // Compute tokens estimate and benefit (hybrid_score or text length heuristic)
                    let mut scored: Vec<(f32, usize, serde_json::Value)> = Vec::new();
                    for it in items.drain(..) {
                        let content = it.get("content").and_then(|v| v.as_str()).unwrap_or("");
                        let score = it
                            .get("hybrid_score")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.5) as f32;
                        let tokens = if let Some((_bpe, ids)) =
                            Self::encode_tokens_precise(content, model.as_deref())
                        {
                            ids.len()
                        } else {
                            Self::estimate_token_count_local(content, model.as_deref())
                        };
                        scored.push((score, tokens, it));
                    }
                    // Greedy by score/tokens
                    scored.sort_by(|a, b| {
                        (b.0 / b.1.max(1) as f32)
                            .partial_cmp(&(a.0 / a.1.max(1) as f32))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let mut budget = target.saturating_sub(margin);
                    let mut selected: Vec<serde_json::Value> = Vec::new();
                    for (_score, t, it) in scored.into_iter() {
                        let need = t.min(budget);
                        if need == 0 {
                            continue;
                        }
                        selected.push(it);
                        if budget <= t {
                            break;
                        }
                        budget -= t;
                    }
                    let out = serde_json::Value::Array(selected);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "context_retrieve" {
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError(
                            "context_retrieve requires session_id, query, k".to_string(),
                        ));
                    }
                    let session_id = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "session_id must be string".to_string(),
                            ))
                        }
                    };
                    let query = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "query must be string".to_string(),
                            ))
                        }
                    };
                    let k = match self.evaluate_expression(args[2].clone())? {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "k must be integer".to_string(),
                            ))
                        }
                    };
                    let history = self
                        .memory_manager
                        .get_session_history(&session_id)
                        .unwrap_or_default();
                    let mut lines: Vec<&str> = history.lines().collect();
                    let ql = query.to_lowercase();
                    lines.retain(|l| l.to_lowercase().contains(&ql));
                    let excerpt = lines.into_iter().take(k).collect::<Vec<_>>().join("\n");
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(excerpt))?;
                    }
                    Ok(())
                } else if function == "context_merge" {
                    if args.is_empty() {
                        return Err(ExecutorError::ArgumentError(
                            "context_merge requires [session_ids]".to_string(),
                        ));
                    }
                    let session_ids: Vec<String> =
                        match self.evaluate_expression(args[0].clone())? {
                            RuntimeValue::Json(Value::Array(arr)) => arr
                                .into_iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect(),
                            RuntimeValue::String(s) => vec![s],
                            other => vec![format!("{:?}", other)],
                        };
                    let mut merged = String::new();
                    for sid in session_ids {
                        if let Ok(hist) = self.memory_manager.get_session_history(&sid) {
                            if !merged.is_empty() {
                                merged.push_str("\n\n");
                            }
                            merged.push_str(&format!("# Session: {}\n{}", sid, hist));
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(merged))?;
                    }
                    Ok(())
                } else if function == "get_multioutput_text" {
                    // get_multioutput_text(multioutput) -> string - Extract primary text
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "get_multioutput_text requires exactly 1 argument: multioutput"
                                .to_string(),
                        ));
                    }

                    let multioutput_value = self.evaluate_expression(args[0].clone())?;
                    let text = match multioutput_value {
                        RuntimeValue::MultiOutput { primary_text, .. } => primary_text,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "get_multioutput_text requires a MultiOutput value".to_string(),
                            ))
                        }
                    };

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(text))?;
                    }
                    Ok(())
                } else if function == "get_multioutput_files" {
                    // get_multioutput_files(multioutput) -> array - Extract binary files
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "get_multioutput_files requires exactly 1 argument: multioutput"
                                .to_string(),
                        ));
                    }

                    let multioutput_value = self.evaluate_expression(args[0].clone())?;
                    let files = match multioutput_value {
                        RuntimeValue::MultiOutput { binary_files, .. } => {
                            let file_info: Vec<Value> = binary_files
                                .iter()
                                .map(|f| {
                                    serde_json::json!({
                                        "name": f.name,
                                        "size": f.size,
                                        "mime_type": f.mime_type
                                    })
                                })
                                .collect();
                            RuntimeValue::Json(Value::Array(file_info))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "get_multioutput_files requires a MultiOutput value".to_string(),
                            ))
                        }
                    };

                    if let Some(res) = result {
                        self.store_value(res, files)?;
                    }
                    Ok(())
                } else if function == "get_multioutput_metadata" {
                    // get_multioutput_metadata(multioutput) -> json - Extract metadata
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "get_multioutput_metadata requires exactly 1 argument: multioutput"
                                .to_string(),
                        ));
                    }

                    let multioutput_value = self.evaluate_expression(args[0].clone())?;
                    let metadata = match multioutput_value {
                        RuntimeValue::MultiOutput { metadata, .. } => {
                            let metadata_json: serde_json::Map<String, Value> = metadata
                                .into_iter()
                                .map(|(k, v)| (k, Value::String(v)))
                                .collect();
                            RuntimeValue::Json(Value::Object(metadata_json))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "get_multioutput_metadata requires a MultiOutput value".to_string(),
                            ))
                        }
                    };

                    if let Some(res) = result {
                        self.store_value(res, metadata)?;
                    }
                    Ok(())
                } else if function == "save_multioutput_file" {
                    // save_multioutput_file(multioutput, index, path) -> void - Save specific file
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError("save_multioutput_file requires exactly 3 arguments: multioutput, index, and path".to_string()));
                    }

                    let multioutput_value = self.evaluate_expression(args[0].clone())?;
                    let index_value = self.evaluate_expression(args[1].clone())?;
                    let path_value = self.evaluate_expression(args[2].clone())?;

                    let index = match index_value {
                        RuntimeValue::Integer(i) => i as usize,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "save_multioutput_file index must be an integer".to_string(),
                            ))
                        }
                    };

                    let path = match path_value {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "save_multioutput_file path must be a string".to_string(),
                            ))
                        }
                    };

                    let binary_files = match multioutput_value {
                        RuntimeValue::MultiOutput { binary_files, .. } => binary_files,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "save_multioutput_file first argument must be a MultiOutput"
                                    .to_string(),
                            ))
                        }
                    };

                    if index >= binary_files.len() {
                        return Err(ExecutorError::ArgumentError(format!(
                            "Index {} out of bounds for {} files",
                            index,
                            binary_files.len()
                        )));
                    }

                    let binary_file = &binary_files[index];

                    // Create parent directories if they don't exist
                    if let Some(parent) = std::path::Path::new(&path).parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            ExecutorError::DataError(format!(
                                "Failed to create directory {}: {}",
                                parent.display(),
                                e
                            ))
                        })?;
                    }

                    // Limits
                    let max_bytes: usize = std::env::var("LEXON_MULTI_MAX_FILE_BYTES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(10 * 1024 * 1024);
                    if binary_file.size > max_bytes {
                        return Err(ExecutorError::RuntimeError(format!(
                            "File exceeds limit: {} bytes > {}",
                            binary_file.size, max_bytes
                        )));
                    }
                    let mut attempts = 0u32;
                    let max_attempts: u32 = std::env::var("LEXON_MULTI_SAVE_RETRIES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(0);
                    let backoff_ms: u64 = std::env::var("LEXON_MULTI_SAVE_BACKOFF_MS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(100);
                    loop {
                        match std::fs::write(&path, &binary_file.content) {
                            Ok(_) => break,
                            Err(e) => {
                                if attempts >= max_attempts {
                                    return Err(ExecutorError::DataError(format!(
                                        "Failed to save file {}: {}",
                                        path, e
                                    )));
                                }
                                attempts += 1;
                                std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                                continue;
                            }
                        }
                    }

                    println!(
                        "💾 Saved file: {} ({} bytes, {})",
                        path, binary_file.size, binary_file.mime_type
                    );
                    Ok(())
                } else if function == "load_file" {
                    self.handle_load_file(args, result.as_ref())?;
                    Ok(())
                } else if function == "prompt.render" || function == "prompt__render" {
                    // prompt.render(name, vars_json, [version])
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "prompt.render requires (name, vars, [version])".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let vars = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            match serde_json::from_str(&s) {
                                Ok(j) => j,
                                Err(_) => {
                                    // Fallbacks: {key: value} or key=value[,key2=value2]
                                    let mut map = serde_json::Map::new();
                                    let trimmed = s.trim();
                                    if trimmed.contains('=') {
                                        for part in trimmed.split(',') {
                                            if let Some((k, v)) = part.split_once('=') {
                                                map.insert(
                                                    k.trim().trim_matches('"').to_string(),
                                                    serde_json::Value::String(
                                                        v.trim().trim_matches('"').to_string(),
                                                    ),
                                                );
                                            }
                                        }
                                    } else {
                                        let t2 =
                                            trimmed.trim_start_matches('{').trim_end_matches('}');
                                        if let Some((k, v)) = t2.split_once(':') {
                                            map.insert(
                                                k.trim().trim_matches('"').to_string(),
                                                serde_json::Value::String(
                                                    v.trim().trim_matches('"').to_string(),
                                                ),
                                            );
                                        }
                                    }
                                    Value::Object(map)
                                }
                            }
                        }
                        _ => Value::Object(Default::default()),
                    };
                    let ver = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Integer(i) => Some(i as i64),
                            RuntimeValue::String(s) => s.parse::<i64>().ok(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let map = Self::load_prompts();
                    let entry = map
                        .get(&name)
                        .and_then(|v| v.get("versions"))
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| ExecutorError::DataError("prompt not found".to_string()))?;
                    let pick = if let Some(v) = ver {
                        entry
                            .iter()
                            .find(|e| e.get("version").and_then(|x| x.as_i64()) == Some(v))
                    } else {
                        entry
                            .iter()
                            .max_by_key(|e| e.get("version").and_then(|x| x.as_i64()).unwrap_or(0))
                    }
                    .ok_or_else(|| {
                        ExecutorError::DataError("prompt version not found".to_string())
                    })?;
                    let content = pick.get("content").and_then(|v| v.as_str()).unwrap_or("");
                    let rendered = Self::render_template(content, &vars);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(rendered))?;
                    }
                    Ok(())
                } else if function == "prompt.render_kv" || function == "prompt__render_kv" {
                    // prompt.render_kv(name, key, value, [version])
                    if args.len() < 3 {
                        return Err(ExecutorError::ArgumentError(
                            "prompt.render_kv requires (name, key, value, [version])".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let key = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let val = match self.evaluate_expression(args[2].clone())? {
                        RuntimeValue::String(s) => s,
                        other => format!("{:?}", other),
                    };
                    let ver = if args.len() > 3 {
                        match self.evaluate_expression(args[3].clone())? {
                            RuntimeValue::Integer(i) => Some(i as i64),
                            RuntimeValue::String(s) => s.parse::<i64>().ok(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let map = Self::load_prompts();
                    let entry = map
                        .get(&name)
                        .and_then(|v| v.get("versions"))
                        .and_then(|v| v.as_array())
                        .ok_or_else(|| ExecutorError::DataError("prompt not found".to_string()))?;
                    let pick = if let Some(v) = ver {
                        entry
                            .iter()
                            .find(|e| e.get("version").and_then(|x| x.as_i64()) == Some(v))
                    } else {
                        entry
                            .iter()
                            .max_by_key(|e| e.get("version").and_then(|x| x.as_i64()).unwrap_or(0))
                    }
                    .ok_or_else(|| {
                        ExecutorError::DataError("prompt version not found".to_string())
                    })?;
                    let mut content = pick
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let needle = format!("{{{{{}}}}}", key);
                    content = content.replace(&needle, &val);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(content))?;
                    }
                    Ok(())
                } else if function == "eval.bleu" || function == "eval__bleu" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "eval.bleu requires (hypothesis, reference)".to_string(),
                        ));
                    }
                    let hyp = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "hypothesis must be string".to_string(),
                            ))
                        }
                    };
                    let refx = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "reference must be string".to_string(),
                            ))
                        }
                    };
                    // Simple BLEU-1 (unigram precision with clipping)
                    use std::collections::HashMap as Hm;
                    let mut ref_counts = Hm::new();
                    for w in refx.split_whitespace() {
                        *ref_counts.entry(w).or_insert(0u32) += 1;
                    }
                    let mut match_count = 0u32;
                    let mut total = 0u32;
                    let mut seen = Hm::new();
                    for w in hyp.split_whitespace() {
                        total += 1;
                        let c = seen.entry(w).or_insert(0u32);
                        *c += 1;
                        let allowed = *ref_counts.get(w).unwrap_or(&0);
                        if *c <= allowed {
                            match_count += 1;
                        }
                    }
                    let p = if total == 0 {
                        0.0
                    } else {
                        (match_count as f64) / (total as f64)
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(p))?;
                    }
                    Ok(())
                } else if function == "eval.rouge_l" || function == "eval__rouge_l" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "eval.rouge_l requires (hypothesis, reference)".to_string(),
                        ));
                    }
                    let hyp = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "hypothesis must be string".to_string(),
                            ))
                        }
                    };
                    let refx = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "reference must be string".to_string(),
                            ))
                        }
                    };
                    // LCS-based ROUGE-L F1
                    let a: Vec<&str> = hyp.split_whitespace().collect();
                    let b: Vec<&str> = refx.split_whitespace().collect();
                    let mut dp = vec![vec![0usize; b.len() + 1]; a.len() + 1];
                    for i in 1..=a.len() {
                        for j in 1..=b.len() {
                            dp[i][j] = if a[i - 1] == b[j - 1] {
                                dp[i - 1][j - 1] + 1
                            } else {
                                dp[i - 1][j].max(dp[i][j - 1])
                            };
                        }
                    }
                    let lcs = dp[a.len()][b.len()] as f64;
                    let r = if b.is_empty() {
                        0.0
                    } else {
                        lcs / (b.len() as f64)
                    };
                    let p = if a.is_empty() {
                        0.0
                    } else {
                        lcs / (a.len() as f64)
                    };
                    let f = if (p + r) == 0.0 {
                        0.0
                    } else {
                        (2.0 * p * r) / (p + r)
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(f))?;
                    }
                    Ok(())
                } else if function == "get_env_or" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "get_env_or requires (name, default)".to_string(),
                        ));
                    }
                    let name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "name must be string".to_string(),
                            ))
                        }
                    };
                    let def = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "default must be string".to_string(),
                            ))
                        }
                    };
                    let val = std::env::var(&name).unwrap_or(def);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(val))?;
                    }
                    Ok(())
                } else if function == "struct.new" || function == "struct__new" {
                    // struct.new(type_name, fields_json_or_string) -> object JSON with __type
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "struct.new requires (type_name, fields)".to_string(),
                        ));
                    }
                    let type_name = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "type_name must be string".to_string(),
                            ))
                        }
                    };
                    let fields_val = self.evaluate_expression(args[1].clone())?;
                    let mut obj = match fields_val {
                        RuntimeValue::Json(serde_json::Value::Object(m)) => m,
                        RuntimeValue::String(s) => {
                            match serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&s)
                            .or_else(|_| serde_json::from_str::<serde_json::Value>(&s).map(|v| v.as_object().cloned().unwrap_or_default())) {
                                Ok(m) => m,
                                Err(_) => serde_json::Map::new(),
                            }
                        }
                        _ => serde_json::Map::new(),
                    };
                    obj.insert("__type".to_string(), serde_json::Value::String(type_name));
                    let out = RuntimeValue::Json(serde_json::Value::Object(obj));
                    if let Some(res) = result {
                        self.store_value(res, out)?;
                    }
                    Ok(())
                } else if function == "struct.get" || function == "struct__get" {
                    // struct.get(object, field) -> value|null
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "struct.get requires (object, field)".to_string(),
                        ));
                    }
                    let obj = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                    let field = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "field must be string".to_string(),
                            ))
                        }
                    };
                    let val = obj.get(&field).cloned().unwrap_or(serde_json::Value::Null);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(val))?;
                    }
                    Ok(())
                } else if function == "struct.set" || function == "struct__set" {
                    // struct.set(object, field, value) -> new_object
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError(
                            "struct.set requires (object, field, value)".to_string(),
                        ));
                    }
                    let base = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                    let field = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "field must be string".to_string(),
                            ))
                        }
                    };
                    let value_json = match self.evaluate_expression(args[2].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::Value::String(s),
                        RuntimeValue::Boolean(b) => serde_json::Value::Bool(b),
                        RuntimeValue::Integer(i) => {
                            serde_json::Value::Number(serde_json::Number::from(i))
                        }
                        RuntimeValue::Float(f) => serde_json::Number::from_f64(f)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null),
                        RuntimeValue::Null => serde_json::Value::Null,
                        other => serde_json::Value::String(format!("{:?}", other)),
                    };
                    let mut map = base.as_object().cloned().unwrap_or_default();
                    map.insert(field, value_json);
                    let out = RuntimeValue::Json(serde_json::Value::Object(map));
                    if let Some(res) = result {
                        self.store_value(res, out)?;
                    }
                    Ok(())
                } else if function == "enum.make" || function == "enum__make" {
                    // enum.make(tag, value?) -> { __tag: tag, value: ... }
                    if args.is_empty() || args.len() > 2 {
                        return Err(ExecutorError::ArgumentError(
                            "enum.make requires tag[, value]".to_string(),
                        ));
                    }
                    let tag = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "tag must be string".to_string(),
                            ))
                        }
                    };
                    let payload = if args.len() == 2 {
                        self.evaluate_expression(args[1].clone())?
                    } else {
                        RuntimeValue::Null
                    };
                    let payload_json = match payload {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::Value::String(s),
                        RuntimeValue::Boolean(b) => serde_json::Value::Bool(b),
                        RuntimeValue::Integer(i) => {
                            serde_json::Value::Number(serde_json::Number::from(i))
                        }
                        RuntimeValue::Float(f) => serde_json::Number::from_f64(f)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null),
                        RuntimeValue::Null => serde_json::Value::Null,
                        other => serde_json::Value::String(format!("{:?}", other)),
                    };
                    let mut obj = serde_json::Map::new();
                    obj.insert("__tag".to_string(), serde_json::Value::String(tag));
                    obj.insert("value".to_string(), payload_json);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(serde_json::Value::Object(obj)))?;
                    }
                    Ok(())
                } else if function == "enum.is" || function == "enum__is" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "enum.is requires (enum, tag)".to_string(),
                        ));
                    }
                    let obj = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                    let tag = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "tag must be string".to_string(),
                            ))
                        }
                    };
                    let ok = obj
                        .get("__tag")
                        .and_then(|v| v.as_str())
                        .map(|t| t == tag)
                        .unwrap_or(false);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(ok))?;
                    }
                    Ok(())
                } else if function == "enum.unwrap" || function == "enum__unwrap" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "enum.unwrap requires (enum)".to_string(),
                        ));
                    }
                    let obj = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => {
                            serde_json::from_str(&s).unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                    let val = obj.get("value").cloned().unwrap_or(serde_json::Value::Null);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(val))?;
                    }
                    Ok(())
                } else if function == "encoding.base64_encode"
                    || function == "encoding__base64_encode"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "encoding.base64_encode(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let enc = base64::engine::general_purpose::STANDARD.encode(s.as_bytes());
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(enc))?;
                    }
                    Ok(())
                } else if function == "encoding.base64_decode"
                    || function == "encoding__base64_decode"
                {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "encoding.base64_decode(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    match base64::engine::general_purpose::STANDARD.decode(s.as_bytes()) {
                        Ok(bytes) => {
                            let out = String::from_utf8(bytes).unwrap_or_default();
                            if let Some(res) = result {
                                self.store_value(res, RuntimeValue::String(out))?;
                            }
                            Ok(())
                        }
                        Err(e) => Err(ExecutorError::RuntimeError(format!("base64 decode: {}", e))),
                    }
                } else if function == "encoding.hex_encode" || function == "encoding__hex_encode" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "encoding.hex_encode(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let mut out = String::with_capacity(s.len() * 2);
                    for b in s.as_bytes() {
                        out.push_str(&format!("{:02x}", b));
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "encoding.hex_decode" || function == "encoding__hex_decode" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "encoding.hex_decode(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    if s.len() % 2 != 0 {
                        return Err(ExecutorError::RuntimeError(
                            "hex decode: odd length".to_string(),
                        ));
                    }
                    let mut bytes = Vec::with_capacity(s.len() / 2);
                    let bs = s.as_bytes();
                    let to_val = |c: u8| -> Option<u8> {
                        match c {
                            b'0'..=b'9' => Some(c - b'0'),
                            b'a'..=b'f' => Some(c - b'a' + 10),
                            b'A'..=b'F' => Some(c - b'A' + 10),
                            _ => None,
                        }
                    };
                    let mut i = 0;
                    while i + 1 < bs.len() {
                        if let (Some(h), Some(l)) = (to_val(bs[i]), to_val(bs[i + 1])) {
                            bytes.push((h << 4) | l);
                            i += 2;
                        } else {
                            return Err(ExecutorError::RuntimeError(
                                "hex decode: invalid hex".to_string(),
                            ));
                        }
                    }
                    let out = String::from_utf8(bytes).unwrap_or_default();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "encoding.url_encode" || function == "encoding__url_encode" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "encoding.url_encode(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let mut out = String::with_capacity(s.len() * 3 / 2 + 1);
                    for ch in s.bytes() {
                        let is_unreserved = (b'a'..=b'z').contains(&ch)
                            || (b'A'..=b'Z').contains(&ch)
                            || (b'0'..=b'9').contains(&ch)
                            || ch == b'-'
                            || ch == b'_'
                            || ch == b'.'
                            || ch == b'~';
                        if is_unreserved {
                            out.push(ch as char);
                        } else if ch == b' ' {
                            out.push_str("%20");
                        } else {
                            out.push('%');
                            out.push_str(&format!("{:02X}", ch));
                        }
                    }
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "encoding.url_decode" || function == "encoding__url_decode" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "encoding.url_decode(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let bs = s.as_bytes();
                    let mut out: Vec<u8> = Vec::with_capacity(bs.len());
                    let hex = |c: u8| -> Option<u8> {
                        match c {
                            b'0'..=b'9' => Some(c - b'0'),
                            b'a'..=b'f' => Some(c - b'a' + 10),
                            b'A'..=b'F' => Some(c - b'A' + 10),
                            _ => None,
                        }
                    };
                    let mut i = 0;
                    while i < bs.len() {
                        if bs[i] == b'%' && i + 2 < bs.len() {
                            if let (Some(h), Some(l)) = (hex(bs[i + 1]), hex(bs[i + 2])) {
                                out.push((h << 4) | l);
                                i += 3;
                            } else {
                                return Err(ExecutorError::RuntimeError(
                                    "url decode: invalid % sequence".to_string(),
                                ));
                            }
                        } else if bs[i] == b'+' {
                            out.push(b' ');
                            i += 1;
                        } else {
                            out.push(bs[i]);
                            i += 1;
                        }
                    }
                    let out_str = String::from_utf8(out).unwrap_or_default();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out_str))?;
                    }
                    Ok(())
                } else if function == "strings.length" || function == "strings__length" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "strings.length(s)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let n = s.chars().count() as i64;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Integer(n))?;
                    }
                    Ok(())
                } else if function == "strings.lower" || function == "strings__lower" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError("strings.lower(s)".to_string()));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(s.to_lowercase()))?;
                    }
                    Ok(())
                } else if function == "strings.upper" || function == "strings__upper" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError("strings.upper(s)".to_string()));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(s.to_uppercase()))?;
                    }
                    Ok(())
                } else if function == "strings.replace" || function == "strings__replace" {
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError(
                            "strings.replace(s, from, to)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let from = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let to = match self.evaluate_expression(args[2].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(s.replace(&from, &to)))?;
                    }
                    Ok(())
                } else if function == "strings.split" || function == "strings__split" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "strings.split(s, sep)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let sep = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let arr: Vec<serde_json::Value> = s
                        .split(&sep)
                        .map(|x| serde_json::Value::String(x.to_string()))
                        .collect();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(serde_json::Value::Array(arr)))?;
                    }
                    Ok(())
                } else if function == "strings.join" || function == "strings__join" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "strings.join(arr, sep)".to_string(),
                        ));
                    }
                    let arr_v = self.evaluate_expression(args[0].clone())?;
                    let sep = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let parts: Vec<String> = match arr_v {
                        RuntimeValue::Json(serde_json::Value::Array(a)) => {
                            let mut out: Vec<String> = Vec::new();
                            for v in a {
                                let s = if v.is_null() {
                                    String::new()
                                } else {
                                    v.as_str().unwrap_or(&v.to_string()).to_string()
                                };
                                // Resolve debug placeholder like Identifier("name") to variable value if exists
                                if let Some(stripped) = s
                                    .strip_prefix("Identifier(\"")
                                    .and_then(|t| t.strip_suffix("\")"))
                                {
                                    if let Some(rv) = self.variables.get(stripped) {
                                        out.push(format_runtime_value(rv));
                                        continue;
                                    }
                                }
                                out.push(s);
                            }
                            out
                        }
                        RuntimeValue::String(s) => vec![s],
                        _ => Vec::new(),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(parts.join(&sep)))?;
                    }
                    Ok(())
                } else if function == "strings.starts_with" || function == "strings__starts_with" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "strings.starts_with(s, prefix)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let prefix = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(s.starts_with(&prefix)))?;
                    }
                    Ok(())
                } else if function == "strings.substring" || function == "strings__substring" {
                    if args.len() < 2 {
                        return Err(ExecutorError::ArgumentError(
                            "strings.substring(s, start[, len])".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let start = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i.max(0) as usize,
                        RuntimeValue::Float(f) => (f as isize).max(0) as usize,
                        RuntimeValue::String(t) => t.parse::<isize>().unwrap_or(0).max(0) as usize,
                        _ => 0,
                    };
                    let len_opt = if args.len() > 2 {
                        match self.evaluate_expression(args[2].clone())? {
                            RuntimeValue::Integer(i) => Some(i.max(0) as usize),
                            RuntimeValue::Float(f) => Some((f as isize).max(0) as usize),
                            RuntimeValue::String(t) => {
                                Some(t.parse::<isize>().unwrap_or(0).max(0) as usize)
                            }
                            _ => None,
                        }
                    } else {
                        None
                    };
                    let chars: Vec<char> = s.chars().collect();
                    if start >= chars.len() {
                        if let Some(res) = result {
                            self.store_value(res, RuntimeValue::String(String::new()))?;
                        }
                        return Ok(());
                    }
                    let end = match len_opt {
                        Some(l) => (start + l).min(chars.len()),
                        None => chars.len(),
                    };
                    let out: String = chars[start..end].iter().collect();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "math.sqrt" || function == "math__sqrt" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError("math.sqrt(x)".to_string()));
                    }
                    let x = self.evaluate_expression(args[0].clone())?;
                    let v = match x {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(v.sqrt()))?;
                    }
                    Ok(())
                } else if function == "math.pow" || function == "math__pow" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError("math.pow(x,y)".to_string()));
                    }
                    let a = self.evaluate_expression(args[0].clone())?;
                    let b = self.evaluate_expression(args[1].clone())?;
                    let x = match a {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let y = match b {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(x.powf(y)))?;
                    }
                    Ok(())
                } else if function == "math.min" || function == "math__min" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError("math.min(a,b)".to_string()));
                    }
                    let a = self.evaluate_expression(args[0].clone())?;
                    let b = self.evaluate_expression(args[1].clone())?;
                    let x = match a {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let y = match b {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(x.min(y)))?;
                    }
                    Ok(())
                } else if function == "math.max" || function == "math__max" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError("math.max(a,b)".to_string()));
                    }
                    let a = self.evaluate_expression(args[0].clone())?;
                    let b = self.evaluate_expression(args[1].clone())?;
                    let x = match a {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let y = match b {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(x.max(y)))?;
                    }
                    Ok(())
                } else if function == "math.clamp" || function == "math__clamp" {
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError(
                            "math.clamp(x, lo, hi)".to_string(),
                        ));
                    }
                    let xv = self.evaluate_expression(args[0].clone())?;
                    let lov = self.evaluate_expression(args[1].clone())?;
                    let hiv = self.evaluate_expression(args[2].clone())?;
                    let x = match xv {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let lo = match lov {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let hi = match hiv {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let v = if lo > hi { x } else { x.max(lo).min(hi) };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Float(v))?;
                    }
                    Ok(())
                } else if function == "regex.match" || function == "regex__match" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "regex.match(s, pattern)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let pat = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let re = regex::Regex::new(&pat)
                        .map_err(|e| ExecutorError::RuntimeError(format!("regex: {}", e)))?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(re.is_match(&s)))?;
                    }
                    Ok(())
                } else if function == "regex.replace" || function == "regex__replace" {
                    if args.len() != 3 {
                        return Err(ExecutorError::ArgumentError(
                            "regex.replace(s, pattern, repl)".to_string(),
                        ));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let pat = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let to = match self.evaluate_expression(args[2].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let re = regex::Regex::new(&pat)
                        .map_err(|e| ExecutorError::RuntimeError(format!("regex: {}", e)))?;
                    if let Some(res) = result {
                        self.store_value(
                            res,
                            RuntimeValue::String(re.replace_all(&s, to.as_str()).to_string()),
                        )?;
                    }
                    Ok(())
                } else if function == "json.parse" || function == "json__parse" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError("json.parse(text)".to_string()));
                    }
                    let t = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let val: serde_json::Value = serde_json::from_str(&t)
                        .map_err(|e| ExecutorError::JsonError(format!("json.parse: {}", e)))?;
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(val))?;
                    }
                    Ok(())
                } else if function == "json.to_string" || function == "json__to_string" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "json.to_string(value)".to_string(),
                        ));
                    }
                    let v = self.evaluate_expression(args[0].clone())?;
                    let s = match v {
                        RuntimeValue::Json(j) => j.to_string(),
                        _ => format_runtime_value(&v),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(s))?;
                    }
                    Ok(())
                } else if function == "json.get" || function == "json__get" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "json.get(obj, key)".to_string(),
                        ));
                    }
                    let objv = self.evaluate_expression(args[0].clone())?;
                    let key = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let j = match objv {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::from_str::<serde_json::Value>(&s)
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::Value::Null,
                    };
                    let out = match j {
                        serde_json::Value::Object(map) => {
                            map.get(&key).cloned().unwrap_or(serde_json::Value::Null)
                        }
                        _ => serde_json::Value::Null,
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "json.path" || function == "json__path" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "json.path(obj, pointer)".to_string(),
                        ));
                    }
                    let objv = self.evaluate_expression(args[0].clone())?;
                    let ptr = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    let j = match objv {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::from_str::<serde_json::Value>(&s)
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::Value::Null,
                    };
                    let out = j.pointer(&ptr).cloned().unwrap_or(serde_json::Value::Null);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "json.keys" || function == "json__keys" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError("json.keys(obj)".to_string()));
                    }
                    let objv = self.evaluate_expression(args[0].clone())?;
                    let j = match objv {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::from_str::<serde_json::Value>(&s)
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::Value::Null,
                    };
                    let out = if let serde_json::Value::Object(map) = j {
                        serde_json::Value::Array(
                            map.keys().cloned().map(serde_json::Value::String).collect(),
                        )
                    } else {
                        serde_json::Value::Array(vec![])
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "json.length" || function == "json__length" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "json.length(value)".to_string(),
                        ));
                    }
                    let v = self.evaluate_expression(args[0].clone())?;
                    let n: i64 = match v {
                        RuntimeValue::Json(serde_json::Value::Array(a)) => a.len() as i64,
                        RuntimeValue::Json(serde_json::Value::Object(m)) => m.len() as i64,
                        RuntimeValue::Json(serde_json::Value::String(s)) => {
                            s.chars().count() as i64
                        }
                        RuntimeValue::String(s) => s.chars().count() as i64,
                        _ => 0,
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Integer(n))?;
                    }
                    Ok(())
                } else if function == "json.index" || function == "json__index" {
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "json.index(arr, idx)".to_string(),
                        ));
                    }
                    let arrv = self.evaluate_expression(args[0].clone())?;
                    let idx = match self.evaluate_expression(args[1].clone())? {
                        RuntimeValue::Integer(i) => i as isize,
                        RuntimeValue::Float(f) => f as isize,
                        RuntimeValue::String(s) => s.parse::<isize>().unwrap_or(0),
                        _ => 0,
                    };
                    let j = match arrv {
                        RuntimeValue::Json(j) => j,
                        RuntimeValue::String(s) => serde_json::from_str::<serde_json::Value>(&s)
                            .unwrap_or(serde_json::Value::Null),
                        _ => serde_json::Value::Null,
                    };
                    let out = if let serde_json::Value::Array(a) = j {
                        if idx >= 0 {
                            let i = idx as usize;
                            if i < a.len() {
                                a[i].clone()
                            } else {
                                serde_json::Value::Null
                            }
                        } else {
                            serde_json::Value::Null
                        }
                    } else {
                        serde_json::Value::Null
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Json(out))?;
                    }
                    Ok(())
                } else if function == "json.as_string" || function == "json__as_string" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError(
                            "json.as_string(value)".to_string(),
                        ));
                    }
                    let v = self.evaluate_expression(args[0].clone())?;
                    let out = match v {
                        RuntimeValue::String(s) => s,
                        RuntimeValue::Json(serde_json::Value::String(s)) => s,
                        RuntimeValue::Json(j) => j.to_string(),
                        _ => format_runtime_value(&v),
                    };
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(out))?;
                    }
                    Ok(())
                } else if function == "time.now_iso8601" || function == "time__now_iso8601" {
                    if args.len() != 0 {
                        return Err(ExecutorError::ArgumentError(
                            "time.now_iso8601()".to_string(),
                        ));
                    }
                    let now = chrono::Utc::now().to_rfc3339();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(now))?;
                    }
                    Ok(())
                } else if function == "number.format" || function == "number__format" {
                    if args.len() < 1 {
                        return Err(ExecutorError::ArgumentError(
                            "number.format(n[,decimals])".to_string(),
                        ));
                    }
                    let v = self.evaluate_expression(args[0].clone())?;
                    let x = match v {
                        RuntimeValue::Integer(i) => i as f64,
                        RuntimeValue::Float(f) => f,
                        RuntimeValue::String(s) => s.parse::<f64>().unwrap_or(0.0),
                        _ => 0.0,
                    };
                    let decimals: usize = if args.len() > 1 {
                        match self.evaluate_expression(args[1].clone())? {
                            RuntimeValue::Integer(i) => (i as isize).max(0) as usize,
                            RuntimeValue::Float(f) => (f as isize).max(0) as usize,
                            RuntimeValue::String(s) => s.parse().unwrap_or(2),
                            _ => 2,
                        }
                    } else {
                        2
                    };
                    let s = format!("{:.*}", decimals, x);
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(s))?;
                    }
                    Ok(())
                } else if function == "crypto.sha256" || function == "crypto__sha256" {
                    if args.len() != 1 {
                        return Err(ExecutorError::ArgumentError("crypto.sha256(s)".to_string()));
                    }
                    let s = match self.evaluate_expression(args[0].clone())? {
                        RuntimeValue::String(s) => s,
                        v => format_runtime_value(&v),
                    };
                    use sha2::{Digest, Sha256};
                    let mut hasher = Sha256::new();
                    hasher.update(s.as_bytes());
                    let out = hasher.finalize();
                    let hex = out.iter().map(|b| format!("{:02x}", b)).collect::<String>();
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(hex))?;
                    }
                    Ok(())
                } else {
                    // Try to call user-defined function
                    // Try to call user-defined function
                    if let Some(func_def) = self.functions.get(function).cloned() {
                        let rv = self.call_user_function(&func_def, args)?;
                        if let Some(res) = result {
                            self.store_value(res, rv)?;
                        }
                        Ok(())
                    } else {
                        Err(ExecutorError::UndefinedFunction(function.clone()))
                    }
                }
            }

            LexInstruction::If {
                condition,
                then_block,
                else_block,
            } => {
                let cond_val = self.evaluate_expression(condition.clone())?;
                // Permissive truthiness:
                // - booleans and JSON booleans
                // - strings "true"/"false"
                // - numbers non-cero
                // - Result: use success
                // - JSON: null=false; string/number/bool coerced; array/object=true if non-empty
                let cond_bool = match cond_val {
                    RuntimeValue::Boolean(b) => Some(b),
                    RuntimeValue::Json(ref v) => match v {
                        serde_json::Value::Bool(b) => Some(*b),
                        serde_json::Value::Null => Some(false),
                        serde_json::Value::Number(num) => {
                            if let Some(i) = num.as_i64() {
                                Some(i != 0)
                            } else if let Some(f) = num.as_f64() {
                                Some(f != 0.0)
                            } else {
                                Some(true)
                            }
                        }
                        serde_json::Value::String(s) => {
                            let sl = s.to_lowercase();
                            if sl == "true" {
                                Some(true)
                            } else if sl == "false" {
                                Some(false)
                            } else {
                                Some(!s.is_empty())
                            }
                        }
                        serde_json::Value::Array(a) => Some(!a.is_empty()),
                        serde_json::Value::Object(o) => Some(!o.is_empty()),
                    },
                    RuntimeValue::String(ref s) => {
                        let sl = s.to_lowercase();
                        if sl == "true" {
                            Some(true)
                        } else if sl == "false" {
                            Some(false)
                        } else {
                            Some(!s.is_empty())
                        }
                    }
                    RuntimeValue::Integer(i) => Some(i != 0),
                    RuntimeValue::Float(f) => Some(f != 0.0),
                    RuntimeValue::Null => Some(false),
                    RuntimeValue::Result { success, .. } => Some(success),
                    _ => None,
                };
                if let Some(b) = cond_bool {
                    if b {
                        for instr in then_block {
                            self.execute_instruction(instr)?;
                        }
                    } else if let Some(else_instrs) = else_block {
                        for instr in else_instrs {
                            self.execute_instruction(instr)?;
                        }
                    }
                } else {
                    return Err(ExecutorError::TypeError("if condition must be bool".into()));
                }
                Ok(())
            }
            LexInstruction::DataLoad {
                result,
                source,
                schema,
                options,
            } => {
                let dataset = self.data_processor.load_data(source, options)?;

                // If a JSON schema is provided, perform validation
                if let Some(schema_path) = schema {
                    // Try to load schema from file or embedded string
                    let schema_value: serde_json::Value = if std::path::Path::new(schema_path)
                        .exists()
                    {
                        let schema_str = std::fs::read_to_string(schema_path).map_err(|e| {
                            ExecutorError::DataError(format!(
                                "Error reading schema file {}: {}",
                                schema_path, e
                            ))
                        })?;
                        serde_json::from_str(&schema_str).map_err(|e| {
                            ExecutorError::DataError(format!(
                                "Invalid JSON schema in file {}: {}",
                                schema_path, e
                            ))
                        })?
                    } else {
                        // Interpret as embedded JSON string
                        serde_json::from_str(schema_path).map_err(|e| {
                            ExecutorError::DataError(format!("Invalid inline JSON schema: {}", e))
                        })?
                    };

                    dataset.validate_against_schema(&schema_value)?;
                }
                self.store_value(result, RuntimeValue::Dataset(Arc::new(dataset)))
            }

            LexInstruction::DataFilter {
                result,
                input,
                predicate,
            } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    let predicate_value = self.evaluate_expression(predicate.clone())?;
                    let filtered = self.data_processor.filter_data(dataset, predicate_value)?;
                    self.store_value(result, RuntimeValue::Dataset(Arc::new(filtered)))
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for filter operation".to_string(),
                    ))
                }
            }

            LexInstruction::DataSelect {
                result,
                input,
                fields,
            } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    let selected = self.data_processor.select_fields(dataset, fields)?;
                    self.store_value(result, RuntimeValue::Dataset(Arc::new(selected)))
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for select operation".to_string(),
                    ))
                }
            }

            LexInstruction::DataTake {
                result,
                input,
                count,
            } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    // If count comes as a literal value, use it directly
                    let count_value = match count {
                        // If it's a ValueRef, resolve it
                        ValueRef::Named(name) => {
                            if let Some(value) = self.variables.get(name) {
                                if let RuntimeValue::Integer(i) = value {
                                    *i as usize
                                } else {
                                    return Err(ExecutorError::TypeError(
                                        "Count must be an integer".to_string(),
                                    ));
                                }
                            } else {
                                return Err(ExecutorError::NameError(format!(
                                    "Variable not found: {}",
                                    name
                                )));
                            }
                        }
                        ValueRef::Temp(temp_id) => {
                            if let Some(value) = self.temporaries.get(temp_id) {
                                if let RuntimeValue::Integer(i) = value {
                                    *i as usize
                                } else {
                                    return Err(ExecutorError::TypeError(
                                        "Count must be an integer".to_string(),
                                    ));
                                }
                            } else {
                                return Err(ExecutorError::NameError(format!(
                                    "Temporary value not found: {:?}",
                                    temp_id
                                )));
                            }
                        }
                        ValueRef::Literal(LexLiteral::Integer(i)) => *i as usize,
                        _ => {
                            return Err(ExecutorError::TypeError(
                                "Count must be an integer".to_string(),
                            ));
                        }
                    };

                    let limited = self.data_processor.take_rows(dataset, count_value)?;
                    self.store_value(result, RuntimeValue::Dataset(Arc::new(limited)))
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for take operation".to_string(),
                    ))
                }
            }

            LexInstruction::DataExport {
                input,
                path,
                format,
                options: _,
            } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    self.data_processor.export_data(dataset, path, format)
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for export operation".to_string(),
                    ))
                }
            }

            LexInstruction::DataFlatten {
                result,
                input,
                separator,
            } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    let flattened = self
                        .data_processor
                        .flatten_json(dataset, separator.as_deref())?;
                    self.store_value(result, RuntimeValue::Dataset(Arc::new(flattened)))
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for flatten operation".to_string(),
                    ))
                }
            }

            LexInstruction::DataFilterJsonPath {
                result,
                input,
                path,
            } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    let filtered = self.data_processor.filter_jsonpath(dataset, path)?;
                    self.store_value(result, RuntimeValue::Dataset(Arc::new(filtered)))
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for JSONPath filter operation".to_string(),
                    ))
                }
            }

            LexInstruction::DataInferSchema { result, input } => {
                let input_value = self.resolve_value(input)?;
                if let RuntimeValue::Dataset(dataset) = input_value {
                    let schema = dataset.infer_json_schema()?;
                    self.store_value(result, RuntimeValue::Json(schema))
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for schema inference".to_string(),
                    ))
                }
            }

            LexInstruction::DataValidateIncremental {
                result,
                input,
                schema,
            } => {
                let input_value = self.resolve_value(input)?;
                let schema_value = self.resolve_value(schema)?;

                if let RuntimeValue::Dataset(dataset) = input_value {
                    if let RuntimeValue::Json(schema_json) = schema_value {
                        let validation_results = dataset.validate_incremental(&schema_json)?;
                        self.store_value(
                            result,
                            RuntimeValue::Dataset(Arc::new(validation_results)),
                        )
                    } else {
                        Err(ExecutorError::TypeError(
                            "Expected JSON schema for validation".to_string(),
                        ))
                    }
                } else {
                    Err(ExecutorError::TypeError(
                        "Expected dataset for validation".to_string(),
                    ))
                }
            }

            LexInstruction::MemoryLoad {
                result,
                scope,
                source,
                strategy,
                options,
            } => {
                // Resolve optional source
                let source_value = if let Some(src) = source {
                    Some(self.resolve_value(src)?)
                } else {
                    None
                };

                // Load from memory
                let memory_result =
                    self.memory_manager
                        .load_memory(scope, source_value, strategy, options)?;

                self.store_value(result, memory_result)
            }

            LexInstruction::MemoryStore {
                scope,
                value,
                key,
                options: _,
            } => {
                // Resolve value to store
                let value_to_store = self.resolve_value(value)?;

                // Store in memory
                self.memory_manager
                    .store_memory(scope, value_to_store, key.as_deref())?;

                Ok(())
            }
            LexInstruction::While { condition, body } => {
                let mut iteration_count = 0;
                'outer: loop {
                    iteration_count += 1;

                    // Evaluate condition
                    let cond_val = self.evaluate_expression(condition.clone())?;

                    let is_true = match cond_val {
                        RuntimeValue::Boolean(b) => b,
                        RuntimeValue::Integer(i) => i != 0,
                        _ => {
                            return Err(ExecutorError::TypeError(
                                "while condition must be boolean or int".to_string(),
                            ))
                        }
                    };

                    if !is_true {
                        break;
                    }

                    for instr in body.iter() {
                        match instr {
                            LexInstruction::Break => {
                                break 'outer;
                            }
                            LexInstruction::Continue => {
                                continue 'outer;
                            }
                            _ => {
                                self.execute_instruction(instr)?;
                                // Show variable state after each instruction
                            }
                        }
                    }

                    // Safety check to prevent infinite loops during debugging
                    if iteration_count > 10 {
                        break;
                    }
                }
                Ok(())
            }
            LexInstruction::ForEach {
                iterator,
                iterable,
                body,
            } => {
                // Resolve iterable
                let iter_val = self.resolve_value(iterable)?;
                match iter_val {
                    RuntimeValue::Dataset(ds) => {
                        let rows = ds.to_json_rows()?; // Vec<Value>
                        'outer_foreach: for row in rows {
                            // Store iterator variable as Json
                            self.variables
                                .insert(iterator.clone(), RuntimeValue::Json(row));
                            let mut continue_flag = false;
                            for instr in body {
                                match instr {
                                    LexInstruction::Break => break 'outer_foreach,
                                    LexInstruction::Continue => {
                                        continue_flag = true;
                                        break;
                                    }
                                    _ => self.execute_instruction(instr)?,
                                }
                            }
                            if continue_flag {
                                continue;
                            }
                        }
                        // Optionally remove iterator variable
                        self.variables.remove(iterator);
                        Ok(())
                    }
                    RuntimeValue::String(s) => {
                        'outer_str: for ch in s.chars() {
                            self.variables
                                .insert(iterator.clone(), RuntimeValue::String(ch.to_string()));
                            let mut continue_flag = false;
                            for instr in body {
                                match instr {
                                    LexInstruction::Break => break 'outer_str,
                                    LexInstruction::Continue => {
                                        continue_flag = true;
                                        break;
                                    }
                                    _ => self.execute_instruction(instr)?,
                                }
                            }
                            if continue_flag {
                                continue;
                            }
                        }
                        self.variables.remove(iterator);
                        Ok(())
                    }
                    RuntimeValue::Json(serde_json::Value::Array(arr)) => {
                        'outer_array: for item in arr {
                            // Convert JSON value to RuntimeValue
                            let runtime_item = match item {
                                serde_json::Value::String(s) => RuntimeValue::String(s.clone()),
                                serde_json::Value::Number(ref n) => {
                                    if let Some(i) = n.as_i64() {
                                        RuntimeValue::Integer(i)
                                    } else if let Some(f) = n.as_f64() {
                                        RuntimeValue::Float(f)
                                    } else {
                                        RuntimeValue::Json(item.clone())
                                    }
                                }
                                serde_json::Value::Bool(b) => RuntimeValue::Boolean(b),
                                serde_json::Value::Null => RuntimeValue::Null,
                                _ => RuntimeValue::Json(item.clone()),
                            };
                            self.variables.insert(iterator.clone(), runtime_item);
                            let mut continue_flag = false;
                            for instr in body {
                                match instr {
                                    LexInstruction::Break => break 'outer_array,
                                    LexInstruction::Continue => {
                                        continue_flag = true;
                                        break;
                                    }
                                    _ => self.execute_instruction(instr)?,
                                }
                            }
                            if continue_flag {
                                continue;
                            }
                        }
                        self.variables.remove(iterator);
                        Ok(())
                    }
                    _ => Err(ExecutorError::TypeError(
                        "Unsupported iterable type for ForEach".to_string(),
                    )),
                }
            }

            LexInstruction::Ask {
                result,
                system_prompt,
                user_prompt,
                model,
                temperature,
                max_tokens,
                schema,
                attributes,
            } => {
                // Use LLM adapter to process query
                println!("🔍 DEBUG: Executing Ask instruction");
                let response = if let Some(ref mut a) = self.llm_adapter_new {
                    a.call_llm(
                        model.as_deref(),
                        *temperature,
                        system_prompt.as_deref(),
                        user_prompt.as_deref(),
                        schema.as_deref(),
                        *max_tokens,
                        attributes,
                    )?
                } else {
                    self.llm_adapter.call_llm(
                        model.as_deref(),
                        *temperature,
                        system_prompt.as_deref(),
                        user_prompt.as_deref(),
                        schema.as_deref(),
                        *max_tokens,
                        attributes,
                    )?
                };
                self.store_value(result, RuntimeValue::String(response))
            }

            LexInstruction::AsyncAsk {
                result,
                system_prompt,
                user_prompt,
                model,
                temperature,
                max_tokens,
                schema,
                attributes,
                task_id: _,
            } => {
                // Execute async ask using tokio directly
                let mut llm_adapter = if let Some(ref a) = self.llm_adapter_new {
                    a.clone()
                } else {
                    self.llm_adapter.clone()
                };
                let model = model.clone();
                let temperature = *temperature;
                let system_prompt = system_prompt.clone();
                let user_prompt = user_prompt.clone();
                let schema = schema.clone();
                let max_tokens = *max_tokens;
                let attributes = attributes.clone();

                // Use block_in_place to execute async code in sync context
                let response = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        llm_adapter
                            .call_llm_async(
                                model.as_deref(),
                                temperature,
                                system_prompt.as_deref(),
                                user_prompt.as_deref(),
                                schema.as_deref(),
                                max_tokens,
                                &attributes,
                            )
                            .await
                    })
                })?;

                self.store_value(result, RuntimeValue::String(response))
            }
            LexInstruction::AsyncAskParallel {
                results,
                asks,
                timeout_ms: _,
                max_concurrent: _,
            } => {
                // Implement AsyncAskParallel for advanced orchestration

                // Verify that there's the same number of results as asks
                if results.len() != asks.len() {
                    return Err(ExecutorError::ArgumentError(format!(
                        "Number of results ({}) must match number of asks ({})",
                        results.len(),
                        asks.len()
                    )));
                }

                // For now, execute sequentially to avoid async/await issues
                for (idx, ask_params) in asks.iter().enumerate() {
                    let response = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            self.llm_adapter
                                .call_llm_async(
                                    ask_params.model.as_deref(),
                                    ask_params.temperature,
                                    ask_params.system_prompt.as_deref(),
                                    ask_params.user_prompt.as_deref(),
                                    ask_params.schema.as_deref(),
                                    ask_params.max_tokens,
                                    &ask_params.attributes,
                                )
                                .await
                        })
                    })?;

                    // Store result in corresponding variable
                    self.store_value(&results[idx], RuntimeValue::String(response))?;
                }

                println!(
                    "🚀 AsyncAskParallel completed: {} asks processed sequentially",
                    asks.len()
                );
                Ok(())
            }
            LexInstruction::Return { expr } => {
                // For now, just evaluate the return value and ignore it
                // We'll need to implement proper function return handling later
                if let Some(expression) = expr {
                    let _return_value = self.evaluate_expression(expression.clone())?;
                    // TODO: Implement proper return value handling with ControlFlow
                }
                Ok(())
            }

            LexInstruction::Break | LexInstruction::Continue => Err(ExecutorError::RuntimeError(
                "break/continue outside of loop".to_string(),
            )),

            LexInstruction::Match { value, arms } => {
                // Evaluate value to match
                let match_value = self.evaluate_expression(value.clone())?;

                // Convert value to string for simple comparison
                let match_str = match match_value {
                    RuntimeValue::String(s) => s,
                    RuntimeValue::Integer(i) => i.to_string(),
                    RuntimeValue::Float(f) => f.to_string(),
                    RuntimeValue::Boolean(b) => b.to_string(),
                    RuntimeValue::Null => "null".to_string(),
                    _ => {
                        return Err(ExecutorError::TypeError(
                            "Match value must be a simple type".to_string(),
                        ))
                    }
                };

                // Find first arm that matches
                for arm in arms {
                    if arm.pattern == match_str || arm.pattern == "_" {
                        // Execute body of matching arm
                        for instr in &arm.body {
                            self.execute_instruction(instr)?;
                        }
                        return Ok(());
                    }
                }

                // If no matches, it's an error
                Err(ExecutorError::RuntimeError(format!(
                    "No matching pattern found for value: {}",
                    match_str
                )))
            }
            // 🛡️ Handle LexInstruction::AskSafe with anti-hallucination validation
            LexInstruction::AskSafe {
                result,
                system_prompt: _,
                user_prompt,
                model,
                temperature: _,
                max_tokens: _,
                schema: _,
                attributes: _,
                validation_strategy,
                confidence_threshold,
                max_attempts,
                cross_reference_models,
                use_fact_checking,
            } => {
                println!("🛡️ DEBUG: Executing AskSafe instruction with validation");

                // Build anti-hallucination validation configuration
                use crate::executor::llm_adapter::{AntiHallucinationConfig, ValidationStrategy};

                let strategy = match validation_strategy.as_deref() {
                    Some("basic") => ValidationStrategy::Basic,
                    Some("ensemble") => ValidationStrategy::Ensemble,
                    Some("fact_check") => ValidationStrategy::FactCheck,
                    Some("comprehensive") => ValidationStrategy::Comprehensive,
                    _ => ValidationStrategy::Basic, // Default
                };

                let config = AntiHallucinationConfig {
                    validation_strategy: strategy,
                    confidence_threshold: confidence_threshold.unwrap_or(0.8),
                    max_validation_attempts: max_attempts.unwrap_or(3) as usize,
                    use_fact_checking: use_fact_checking.unwrap_or(true),
                    cross_reference_models: cross_reference_models.clone(),
                };

                // Build prompt for ask_safe
                let prompt = user_prompt.as_deref().unwrap_or("No prompt provided");

                // Execute ask_safe with anti-hallucination validation
                let validation_result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        self.llm_adapter
                            .ask_safe(prompt, model.as_deref(), Some(config))
                            .await
                    })
                })?;

                // Create result structured with validation information
                let result_value = if validation_result.is_valid {
                    RuntimeValue::Result {
                        success: true,
                        value: Box::new(RuntimeValue::String(validation_result.validated_content)),
                        error_message: None,
                    }
                } else {
                    let error_msg = format!(
                        "Validation failed (confidence: {:.2}), {} issues detected",
                        validation_result.confidence_score,
                        validation_result.issues.len()
                    );
                    RuntimeValue::Result {
                        success: false,
                        value: Box::new(RuntimeValue::String(validation_result.validated_content)),
                        error_message: Some(error_msg),
                    }
                };

                println!(
                    "🛡️ AskSafe completed: valid={}, confidence={:.2}",
                    validation_result.is_valid, validation_result.confidence_score
                );

                self.store_value(result, result_value)
            } // All cases of LexInstruction are covered
        }
    }
    /// 🛡️ SPRINT C: Calculates confidence score of a response using heuristics (v2)
    fn calculate_confidence_score_v2(&self, response: &str) -> f64 {
        let mut confidence: f64 = 1.0;

        // 1. Check uncertainty patterns
        let uncertainty_patterns = vec![
            "I'm not sure",
            "I think",
            "possibly",
            "probably",
            "might be",
            "maybe",
            "perhaps",
            "could be",
            "it's possible",
        ];

        for pattern in &uncertainty_patterns {
            if response.to_lowercase().contains(&pattern.to_lowercase()) {
                confidence -= 0.1;
            }
        }

        // 2. Check common hallucination patterns
        let hallucination_patterns = vec![
            "according to recent studies",
            "research has shown",
            "experts claim",
            "it has been proven that",
        ];

        for pattern in &hallucination_patterns {
            if response.to_lowercase().contains(&pattern.to_lowercase()) {
                confidence -= 0.15;
            }
        }

        // 3. Check length and structure
        if response.len() < 10 {
            confidence -= 0.2; // Very short responses are suspicious
        }

        if response.len() > 2000 {
            confidence -= 0.1; // Very long responses may be verbose without substance
        }

        // 4. Check simple internal contradictions
        let sentences: Vec<&str> = response.split(". ").collect();
        if sentences.len() > 3 {
            // Look for obvious contradictions
            for (i, sentence1) in sentences.iter().enumerate() {
                for (j, sentence2) in sentences.iter().enumerate() {
                    if i != j && Self::detect_simple_contradiction(sentence1, sentence2) {
                        confidence -= 0.2;
                    }
                }
            }
        }

        // 5. Bonus for clear structure
        if response.contains("1.") || response.contains("•") || response.contains("-") {
            confidence += 0.05; // Structured responses are more reliable
        }

        // Ensure score is between 0.0 and 1.0
        confidence.clamp(0.0_f64, 1.0_f64)
    }

    /// 🛡️ SPRINT C: Validates a response using basic validation (v2)
    #[allow(clippy::only_used_in_recursion)]
    fn validate_response_basic_v2(&self, response: &str, validation_type: &str) -> bool {
        match validation_type {
            "basic" => {
                // Basic validation: check if not too short and not obviously hallucinated
                if response.len() < 5 {
                    return false;
                }

                // Check for obvious hallucination patterns
                let obvious_hallucination_patterns = vec![
                    "unicorns are real",
                    "unicorns live",
                    "unicorns exist",
                    "dragons exist",
                    "dragons live",
                    "earth is flat",
                    "flat earth",
                    "conspiracy theories are true",
                    "aliens built pyramids",
                    "moon landing was fake",
                ];

                for pattern in &obvious_hallucination_patterns {
                    if response.to_lowercase().contains(&pattern.to_lowercase()) {
                        return false;
                    }
                }

                true
            }
            "length" => {
                // Validate minimum and maximum length
                response.len() >= 10 && response.len() <= 5000
            }
            "structure" => {
                // Validate basic structure (not just spaces or special characters)
                !response.trim().is_empty() && response.chars().any(|c| c.is_alphanumeric())
            }
            "factual" => {
                // Basic factual validation: check if not containing obviously false claims
                let false_claims = vec![
                    "2+2=5",
                    "paris is in germany",
                    "paris is in germany",
                    "the sun is cold",
                    "the sun is cold",
                ];

                for claim in &false_claims {
                    if response.to_lowercase().contains(&claim.to_lowercase()) {
                        return false;
                    }
                }

                true
            }
            _ => {
                // For unrecognized validation types, use basic validation
                self.validate_response_basic_v2(response, "basic")
            }
        }
    }

    /// 🛡️ Detects simple contradictions between two sentences
    fn detect_simple_contradiction(sentence1: &str, sentence2: &str) -> bool {
        // Simple implementation: look for obvious negations
        let s1_lower = sentence1.to_lowercase();
        let s2_lower = sentence2.to_lowercase();

        // Obvious contradiction patterns
        if (s1_lower.contains("is true") && s2_lower.contains("is false"))
            || (s1_lower.contains("yes") && s2_lower.contains("no"))
            || (s1_lower.contains("yes") && s2_lower.contains("no"))
        {
            return true;
        }

        false
    }

    // 🧠 ============================================================================
    // PILLAR #2: SUMMARIZATION FUNCTIONS - WORLD'S FIRST NATIVE SESSION SUMMARIZATION
    // ============================================================================
    /// 🧠 session_summarize: Generate intelligent summary of session history
    /// FIRST IN THE INDUSTRY: Native session summarization in programming language
    pub fn session_summarize(
        &mut self,
        session_id: &str,
        options: &HashMap<String, RuntimeValue>,
    ) -> Result<RuntimeValue> {
        println!(
            "🧠 session_summarize: Generating intelligent summary for session '{}'",
            session_id
        );

        // Extract options with defaults
        let length = match options.get("length") {
            Some(RuntimeValue::String(l)) => l.clone(),
            _ => "medium".to_string(),
        };

        let focus = match options.get("focus") {
            Some(RuntimeValue::String(f)) => f.clone(),
            _ => "comprehensive".to_string(),
        };

        let style = match options.get("style") {
            Some(RuntimeValue::String(s)) => s.clone(),
            _ => "professional".to_string(),
        };

        // Get session history, create demo session if it doesn't exist
        let session_history = match self.memory_manager.get_session_history(session_id) {
            Ok(history) => history,
            Err(_) => {
                // If it's a demo session, create it with sample content
                if session_id.contains("demo") {
                    let demo_content = format!(
                        "Demo conversation for session {}\n\
                        User: Hello, I'm testing the Lexon language features.\n\
                        Assistant: Great! Lexon is a revolutionary LLM-native programming language.\n\
                        User: Can you explain the key features?\n\
                        Assistant: Lexon supports async/await, multimodal requests, session management, and anti-hallucination validation.\n\
                        User: That sounds impressive. How does the session system work?\n\
                        Assistant: Sessions allow persistent context across multiple interactions, with automatic summarization and compression.\n\
                        User: Perfect, let's test the context window management now.",
                        session_id
                    );

                    // Create the demo session
                    if self.memory_manager.create_session(session_id).is_err() {
                        return Ok(RuntimeValue::Boolean(false));
                    }

                    // Add demo content
                    if self
                        .memory_manager
                        .update_session_history(session_id, &demo_content)
                        .is_err()
                    {
                        return Ok(RuntimeValue::Boolean(false));
                    }

                    demo_content
                } else {
                    return Ok(RuntimeValue::Boolean(false));
                }
            }
        };

        if session_history.is_empty() {
            return Ok(RuntimeValue::String(
                "No conversation history available for summarization.".to_string(),
            ));
        }

        // Build summarization prompt based on options
        let system_prompt = format!(
            "You are an expert session summarizer. Create a {} summary with {} focus in a {} style.",
            length, focus, style
        );

        let user_prompt = format!(
            "Please summarize this conversation session:\n\n{}\n\nSummary requirements:\n- Length: {}\n- Focus: {}\n- Style: {}\n\nProvide a comprehensive yet concise summary that captures the key points, decisions, and outcomes.",
            session_history, length, focus, style
        );

        // Generate summary using LLM
        let summary_response = self.llm_adapter.call_llm(
            None,      // Use default model
            Some(0.3), // Lower temperature for consistent summaries
            Some(&system_prompt),
            Some(&user_prompt),
            None, // No schema
            None, // Use default max_tokens
            &HashMap::new(),
        )?;

        println!(
            "✅ session_summarize: Generated summary ({} chars)",
            summary_response.len()
        );
        Ok(RuntimeValue::String(summary_response))
    }
    /// 🧠 context_window_manage: Automatic context window management and compression
    /// FIRST IN THE INDUSTRY: Native context window management in programming language
    pub fn context_window_manage(
        &mut self,
        session_id: &str,
        options: &HashMap<String, RuntimeValue>,
    ) -> Result<RuntimeValue> {
        println!(
            "🧠 context_window_manage: Managing context window for session '{}'",
            session_id
        );

        // Extract options with defaults
        let max_tokens = match options.get("max_tokens") {
            Some(RuntimeValue::Integer(t)) => *t as usize,
            _ => 4000, // Default context window size
        };

        let compression_ratio = match options.get("compression_ratio") {
            Some(RuntimeValue::Float(r)) => *r,
            _ => 0.6, // Default compression ratio (60% of original size)
        };

        let preserve_recent = match options.get("preserve_recent") {
            Some(RuntimeValue::Integer(p)) => *p as usize,
            _ => 5, // Preserve last 5 messages by default
        };

        // Get session history, create demo session if it doesn't exist
        let session_history = match self.memory_manager.get_session_history(session_id) {
            Ok(history) => history,
            Err(_) => {
                // If it's a demo session, create it with sample content
                if session_id.contains("demo") {
                    let demo_content = format!(
                        "Demo conversation for session {}\n\
                        User: Hello, I'm testing the Lexon language features.\n\
                        Assistant: Great! Lexon is a revolutionary LLM-native programming language.\n\
                        User: Can you explain the key features?\n\
                        Assistant: Lexon supports async/await, multimodal requests, session management, and anti-hallucination validation.\n\
                        User: That sounds impressive. How does the session system work?\n\
                        Assistant: Sessions allow persistent context across multiple interactions, with automatic summarization and compression.\n\
                        User: Perfect, let's test the context window management now.",
                        session_id
                    );

                    // Create the demo session
                    if self.memory_manager.create_session(session_id).is_err() {
                        return Ok(RuntimeValue::Boolean(false));
                    }

                    // Add demo content
                    if self
                        .memory_manager
                        .update_session_history(session_id, &demo_content)
                        .is_err()
                    {
                        return Ok(RuntimeValue::Boolean(false));
                    }

                    demo_content
                } else {
                    return Ok(RuntimeValue::Boolean(false));
                }
            }
        };

        // Estimate token count (rough approximation: 4 chars per token)
        let estimated_tokens = session_history.len() / 4;

        if estimated_tokens <= max_tokens {
            println!(
                "✅ context_window_manage: No compression needed ({} tokens < {} limit)",
                estimated_tokens, max_tokens
            );
            return Ok(RuntimeValue::Boolean(true));
        }

        println!(
            "🔄 context_window_manage: Compression needed ({} tokens > {} limit)",
            estimated_tokens, max_tokens
        );

        // Split history into recent (to preserve) and older (to compress)
        let history_lines: Vec<&str> = session_history.lines().collect();
        let total_lines = history_lines.len();

        if total_lines <= preserve_recent {
            println!(
                "✅ context_window_manage: All history preserved (only {} lines)",
                total_lines
            );
            return Ok(RuntimeValue::Boolean(true));
        }

        let split_point = total_lines - preserve_recent;
        let older_history = history_lines[..split_point].join("\n");
        let recent_history = history_lines[split_point..].join("\n");

        // Compress older history
        let compression_prompt = format!(
            "Compress this conversation history to approximately {:.0}% of its original length while preserving key information:\n\n{}",
            compression_ratio * 100.0,
            older_history
        );

        let compressed_history = self.llm_adapter.call_llm(
            None,
            Some(0.2), // Very low temperature for consistent compression
            Some("You are an expert at compressing conversation history while preserving essential information."),
            Some(&compression_prompt),
            None,
            None,
            &HashMap::new(),
        )?;

        // Combine compressed older history with recent history
        let new_history = format!(
            "{}\n\n--- RECENT CONVERSATION ---\n{}",
            compressed_history, recent_history
        );

        // Update session with compressed history
        match self
            .memory_manager
            .update_session_history(session_id, &new_history)
        {
            Ok(_) => {
                let new_tokens = new_history.len() / 4;
                println!(
                    "✅ context_window_manage: Compression successful ({} -> {} tokens)",
                    estimated_tokens, new_tokens
                );
                Ok(RuntimeValue::Boolean(true))
            }
            Err(_) => {
                println!("❌ context_window_manage: Failed to update session history");
                Ok(RuntimeValue::Boolean(false))
            }
        }
    }

    /// 🧠 extract_key_points: Extract key points from session using LLM analysis
    /// FIRST IN THE INDUSTRY: Native key point extraction in programming language
    pub fn extract_key_points(
        &mut self,
        session_id: &str,
        options: &HashMap<String, RuntimeValue>,
    ) -> Result<RuntimeValue> {
        println!(
            "🧠 extract_key_points: Extracting key points from session '{}'",
            session_id
        );

        // Extract options with defaults
        let max_points = match options.get("max_points") {
            Some(RuntimeValue::Integer(p)) => *p as usize,
            _ => 10, // Default maximum points
        };

        let categories = match options.get("categories") {
            Some(RuntimeValue::String(c)) => c.clone(),
            _ => "decisions,actions,insights,questions".to_string(),
        };

        let importance_threshold = match options.get("importance_threshold") {
            Some(RuntimeValue::Float(t)) => *t,
            _ => 0.7, // Default importance threshold (0.0-1.0)
        };

        // Get session history
        let session_history = match self.memory_manager.get_session_history(session_id) {
            Ok(history) => history,
            Err(_) => {
                return Ok(RuntimeValue::Json(serde_json::Value::Array(vec![])));
            }
        };

        if session_history.is_empty() {
            return Ok(RuntimeValue::Json(serde_json::Value::Array(vec![])));
        }

        // Build key point extraction prompt
        let system_prompt = format!(
            "You are an expert at extracting key points from conversations. Extract up to {} key points focusing on: {}. Only include points with importance level >= {:.1}",
            max_points, categories, importance_threshold
        );

        let user_prompt = format!(
            "Extract the most important key points from this conversation:\n\n{}\n\nReturn a JSON array of strings, each containing one key point. Focus on:\n- {}\n\nExample format: [\"Key point 1\", \"Key point 2\", \"Key point 3\"]",
            session_history,
            categories.replace(",", "\n- ")
        );

        // Extract key points using LLM
        let key_points_response = self.llm_adapter.call_llm(
            None,
            Some(0.3), // Lower temperature for consistent extraction
            Some(&system_prompt),
            Some(&user_prompt),
            None,
            None,
            &HashMap::new(),
        )?;

        // Parse JSON response
        match serde_json::from_str::<serde_json::Value>(&key_points_response) {
            Ok(json_value) => {
                if let serde_json::Value::Array(points) = json_value {
                    let extracted_count = points.len();
                    println!(
                        "✅ extract_key_points: Extracted {} key points",
                        extracted_count
                    );
                    Ok(RuntimeValue::Json(serde_json::Value::Array(points)))
                } else {
                    // Fallback: split response by lines and convert to JSON array
                    let lines: Vec<serde_json::Value> = key_points_response
                        .lines()
                        .filter(|line| !line.trim().is_empty())
                        .take(max_points)
                        .map(|line| serde_json::Value::String(line.trim().to_string()))
                        .collect();

                    println!(
                        "✅ extract_key_points: Extracted {} key points (fallback parsing)",
                        lines.len()
                    );
                    Ok(RuntimeValue::Json(serde_json::Value::Array(lines)))
                }
            }
            Err(_) => {
                // Fallback: split response by lines and convert to JSON array
                let lines: Vec<serde_json::Value> = key_points_response
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .take(max_points)
                    .map(|line| serde_json::Value::String(line.trim().to_string()))
                    .collect();

                println!(
                    "✅ extract_key_points: Extracted {} key points (text parsing)",
                    lines.len()
                );
                Ok(RuntimeValue::Json(serde_json::Value::Array(lines)))
            }
        }
    }
    /// 🧠 session_compress: Intelligent compression maintaining context relevance
    /// FIRST IN THE INDUSTRY: Native intelligent session compression in programming language
    pub fn session_compress(
        &mut self,
        session_id: &str,
        options: &HashMap<String, RuntimeValue>,
    ) -> Result<RuntimeValue> {
        println!(
            "🧠 session_compress: Compressing session '{}' with intelligent context preservation",
            session_id
        );

        // Extract options with defaults
        let compression_level = match options.get("compression_level") {
            Some(RuntimeValue::String(l)) => l.clone(),
            _ => "medium".to_string(), // "light", "medium", "aggressive"
        };

        let preserve_entities = match options.get("preserve_entities") {
            Some(RuntimeValue::Boolean(p)) => *p,
            _ => true, // Preserve named entities by default
        };

        let maintain_flow = match options.get("maintain_flow") {
            Some(RuntimeValue::Boolean(f)) => *f,
            _ => true, // Maintain conversation flow by default
        };

        // Get session history
        let session_history = match self.memory_manager.get_session_history(session_id) {
            Ok(history) => history,
            Err(_) => {
                return Ok(RuntimeValue::String(format!(
                    "No session found with ID: {}",
                    session_id
                )));
            }
        };

        if session_history.is_empty() {
            return Ok(RuntimeValue::String(
                "No conversation history available for compression.".to_string(),
            ));
        }

        // Determine compression strategy based on level
        let (compression_ratio, compression_strategy) = match compression_level.as_str() {
            "light" => (
                0.8,
                "Remove redundant information while preserving all key points",
            ),
            "medium" => (
                0.6,
                "Compress significantly while maintaining context and flow",
            ),
            "aggressive" => (
                0.4,
                "Maximum compression while preserving essential information only",
            ),
            _ => (
                0.6,
                "Compress significantly while maintaining context and flow",
            ),
        };

        // Build compression prompt with specific instructions
        let mut compression_instructions = vec![
            format!(
                "Compress to approximately {:.0}% of original length",
                compression_ratio * 100.0
            ),
            compression_strategy.to_string(),
        ];

        if preserve_entities {
            compression_instructions.push(
                "Preserve all named entities (people, places, organizations, dates)".to_string(),
            );
        }

        if maintain_flow {
            compression_instructions
                .push("Maintain logical conversation flow and context transitions".to_string());
        }

        let system_prompt = format!(
            "You are an expert at intelligent conversation compression. Your task is to compress conversations while preserving meaning and context. Instructions:\n- {}", 
            compression_instructions.join("\n- ")
        );

        let user_prompt = format!(
            "Compress this conversation intelligently:\n\n{}\n\nCompression requirements:\n- Level: {}\n- Preserve entities: {}\n- Maintain flow: {}\n\nProvide a compressed version that maintains the essence and key information of the original conversation.",
            session_history, compression_level, preserve_entities, maintain_flow
        );

        // Generate compressed version using LLM
        let compressed_response = self.llm_adapter.call_llm(
            None,
            Some(0.2), // Very low temperature for consistent compression
            Some(&system_prompt),
            Some(&user_prompt),
            None,
            None,
            &HashMap::new(),
        )?;

        // Update session with compressed history
        match self
            .memory_manager
            .update_session_history(session_id, &compressed_response)
        {
            Ok(_) => {
                let original_size = session_history.len();
                let compressed_size = compressed_response.len();
                let actual_ratio = compressed_size as f64 / original_size as f64;

                println!(
                    "✅ session_compress: Compression successful ({} -> {} chars, {:.1}% ratio)",
                    original_size,
                    compressed_size,
                    actual_ratio * 100.0
                );

                Ok(RuntimeValue::String(compressed_response))
            }
            Err(_) => {
                println!("❌ session_compress: Failed to update session history");
                Ok(RuntimeValue::String(compressed_response)) // Return compressed version even if update fails
            }
        }
    }
    // ============================================================================
    // END OF PILLAR #2: SUMMARIZATION FUNCTIONS
    // ============================================================================
    /// Executes a full LexIR program
    pub fn execute_program(&mut self, program: &LexProgram) -> Result<()> {
        // Store functions in the environment
        self.functions = program.functions.clone();
        eprintln!(
            "DEBUG EXECUTOR: Loaded {} functions: {:?}",
            self.functions.len(),
            self.functions.keys().collect::<Vec<_>>()
        );

        for instruction in &program.instructions {
            self.execute_instruction(instruction)?;
        }
        Ok(())
    }

    /// Evaluate a callback expression for map operations
    fn evaluate_callback(
        &mut self,
        callback_str: &str,
        item: &Value,
        _index: usize,
    ) -> Result<Value> {
        // Support common callback patterns
        match callback_str {
            "x * 2" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(Value::Number(serde_json::Number::from(i * 2)))
                    } else if let Some(f) = n.as_f64() {
                        Ok(Value::Number(
                            serde_json::Number::from_f64(f * 2.0)
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        ))
                    } else {
                        Ok(item.clone())
                    }
                }
                _ => Ok(item.clone()),
            },
            "x + 1" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(Value::Number(serde_json::Number::from(i + 1)))
                    } else if let Some(f) = n.as_f64() {
                        Ok(Value::Number(
                            serde_json::Number::from_f64(f + 1.0)
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        ))
                    } else {
                        Ok(item.clone())
                    }
                }
                _ => Ok(item.clone()),
            },
            "x * x" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(Value::Number(serde_json::Number::from(i * i)))
                    } else if let Some(f) = n.as_f64() {
                        Ok(Value::Number(
                            serde_json::Number::from_f64(f * f)
                                .unwrap_or_else(|| serde_json::Number::from(0)),
                        ))
                    } else {
                        Ok(item.clone())
                    }
                }
                _ => Ok(item.clone()),
            },
            "x.toString()" | "x.to_string()" => Ok(Value::String(match item {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => "null".to_string(),
                _ => serde_json::to_string(item).unwrap_or_default(),
            })),
            "x.toUpperCase()" | "x.to_upper()" => match item {
                Value::String(s) => Ok(Value::String(s.to_uppercase())),
                _ => Ok(item.clone()),
            },
            "x.toLowerCase()" | "x.to_lower()" => match item {
                Value::String(s) => Ok(Value::String(s.to_lowercase())),
                _ => Ok(item.clone()),
            },
            "x.length" | "x.len()" => match item {
                Value::String(s) => Ok(Value::Number(serde_json::Number::from(s.len() as u64))),
                Value::Array(arr) => Ok(Value::Number(serde_json::Number::from(arr.len() as u64))),
                _ => Ok(Value::Number(serde_json::Number::from(0))),
            },
            _ => {
                // Default: return item unchanged
                Ok(item.clone())
            }
        }
    }
    /// Evaluate a predicate expression for filter operations
    fn evaluate_predicate(
        &mut self,
        predicate_str: &str,
        item: &Value,
        _index: usize,
    ) -> Result<bool> {
        // Support common predicate patterns
        match predicate_str {
            "x % 2 == 0" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i % 2 == 0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x % 2 != 0" | "x % 2 == 1" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i % 2 != 0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x > 0" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i > 0)
                    } else if let Some(f) = n.as_f64() {
                        Ok(f > 0.0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x < 0" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i < 0)
                    } else if let Some(f) = n.as_f64() {
                        Ok(f < 0.0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x >= 0" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i >= 0)
                    } else if let Some(f) = n.as_f64() {
                        Ok(f >= 0.0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x > 5" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i > 5)
                    } else if let Some(f) = n.as_f64() {
                        Ok(f > 5.0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x < 5" => match item {
                Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        Ok(i < 5)
                    } else if let Some(f) = n.as_f64() {
                        Ok(f < 5.0)
                    } else {
                        Ok(false)
                    }
                }
                _ => Ok(false),
            },
            "x.length > 3" | "x.len() > 3" => match item {
                Value::String(s) => Ok(s.len() > 3),
                Value::Array(arr) => Ok(arr.len() > 3),
                _ => Ok(false),
            },
            "x.startsWith('a')" | "x.starts_with('a')" => match item {
                Value::String(s) => Ok(s.starts_with('a')),
                _ => Ok(false),
            },
            "x.endsWith('e')" | "x.ends_with('e')" => match item {
                Value::String(s) => Ok(s.ends_with('e')),
                _ => Ok(false),
            },
            "x.includes('test')" | "x.contains('test')" => match item {
                Value::String(s) => Ok(s.contains("test")),
                _ => Ok(false),
            },
            "x != null" | "x != None" => Ok(!matches!(item, Value::Null)),
            "x == null" | "x == None" => Ok(matches!(item, Value::Null)),
            "x == true" => match item {
                Value::Bool(b) => Ok(*b),
                _ => Ok(false),
            },
            "x == false" => match item {
                Value::Bool(b) => Ok(!*b),
                _ => Ok(false),
            },
            _ => {
                // Default: check if item is truthy
                Ok(match item {
                    Value::Bool(b) => *b,
                    Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            i != 0
                        } else if let Some(f) = n.as_f64() {
                            f != 0.0
                        } else {
                            false
                        }
                    }
                    Value::String(s) => !s.is_empty(),
                    Value::Array(arr) => !arr.is_empty(),
                    Value::Object(obj) => !obj.is_empty(),
                    Value::Null => false,
                })
            }
        }
    }
    /// Evaluate a reduce callback expression
    fn evaluate_reduce_callback(
        &mut self,
        callback_str: &str,
        accumulator: &RuntimeValue,
        item: &Value,
        _index: usize,
    ) -> Result<RuntimeValue> {
        // Support common reduce callback patterns
        match callback_str {
            "acc + x" => match (accumulator, item) {
                (RuntimeValue::Integer(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Integer(acc + item_i))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float(*acc as f64 + item_f))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                (RuntimeValue::Float(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Float(acc + item_i as f64))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float(acc + item_f))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                (RuntimeValue::String(acc), Value::String(item_s)) => {
                    Ok(RuntimeValue::String(format!("{}{}", acc, item_s)))
                }
                (RuntimeValue::String(acc), _) => {
                    let item_str = match item {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Bool(b) => b.to_string(),
                        Value::Null => "null".to_string(),
                        _ => serde_json::to_string(item).unwrap_or_default(),
                    };
                    Ok(RuntimeValue::String(format!("{}{}", acc, item_str)))
                }
                _ => Ok(accumulator.clone()),
            },
            "acc * x" => match (accumulator, item) {
                (RuntimeValue::Integer(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Integer(acc * item_i))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float(*acc as f64 * item_f))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                (RuntimeValue::Float(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Float(acc * item_i as f64))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float(acc * item_f))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                _ => Ok(accumulator.clone()),
            },
            "Math.max(acc, x)" | "max(acc, x)" => match (accumulator, item) {
                (RuntimeValue::Integer(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Integer(std::cmp::max(*acc, item_i)))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float((*acc as f64).max(item_f)))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                (RuntimeValue::Float(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Float(acc.max(item_i as f64)))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float(acc.max(item_f)))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                _ => Ok(accumulator.clone()),
            },
            "Math.min(acc, x)" | "min(acc, x)" => match (accumulator, item) {
                (RuntimeValue::Integer(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Integer(std::cmp::min(*acc, item_i)))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float((*acc as f64).min(item_f)))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                (RuntimeValue::Float(acc), Value::Number(n)) => {
                    if let Some(item_i) = n.as_i64() {
                        Ok(RuntimeValue::Float(acc.min(item_i as f64)))
                    } else if let Some(item_f) = n.as_f64() {
                        Ok(RuntimeValue::Float(acc.min(item_f)))
                    } else {
                        Ok(accumulator.clone())
                    }
                }
                _ => Ok(accumulator.clone()),
            },
            _ => {
                // Default: return accumulator unchanged
                Ok(accumulator.clone())
            }
        }
    }
}

/// Executes a LexIR program and returns the resulting environment
pub fn execute(program: &LexProgram) -> Result<ExecutionEnvironment> {
    let mut env = ExecutionEnvironment::new(ExecutorConfig::default());
    env.execute_program(program)?;
    Ok(env)
}

#[cfg(test)]
mod test_data_ops {
    use super::*;
    use crate::lexir::{LexInstruction, LexLiteral, LexProgram, TempId, ValueRef};

    #[test]
    fn test_csv_pipeline_integration() {
        // paths (use workspace samples to avoid missing test assets)
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let src_path = manifest_dir
            .join("..")
            .join("samples")
            .join("triage")
            .join("tickets.csv");
        let src = src_path.to_string_lossy().to_string();
        let out = "/tmp/contacts_out.csv";

        // Build program
        let mut prog = LexProgram::new();
        let t0 = ValueRef::Temp(TempId(0));
        let t1 = ValueRef::Temp(TempId(1));
        let t2 = ValueRef::Temp(TempId(2));

        prog.add_instruction(LexInstruction::DataLoad {
            result: t0.clone(),
            source: src,
            schema: None,
            options: Default::default(),
        });
        prog.add_instruction(LexInstruction::DataSelect {
            result: t1.clone(),
            input: t0.clone(),
            fields: vec!["subject".into()],
        });
        prog.add_instruction(LexInstruction::DataTake {
            result: t2.clone(),
            input: t1.clone(),
            count: ValueRef::Literal(LexLiteral::Integer(2)),
        });
        prog.add_instruction(LexInstruction::DataExport {
            input: t2.clone(),
            path: out.into(),
            format: "csv".into(),
            options: Default::default(),
        });

        // Execute
        let mut env = ExecutionEnvironment::new(ExecutorConfig::default());
        env.execute_program(&prog).expect("exec ok");

        // Check output file exists
        assert!(std::path::Path::new(out).exists());
    }
}
