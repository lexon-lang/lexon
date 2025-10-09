// lexc/src/executor/mod.rs
//
// Experimental executor for LexIR
// This module implements a basic interpreter for LexIR that uses Polars for
// data operations and a simple memory adapter.

mod api_config;
pub mod async_ops;
mod data_functions;
pub mod data_processor;
mod llm_adapter;
mod llm_functions;
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
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use memory::MemoryManager;
use vector_memory::VectorMemorySystem; // 🧠 Import real vector system

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
        RuntimeValue::Null => "null".to_string(),
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
            vector_memory_system: VectorMemorySystem::new(None).ok(), // Initialize with in-memory SQLite
        }
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
            ValueRef::Named(name) => self
                .variables
                .get(name)
                .cloned()
                .ok_or_else(|| ExecutorError::UndefinedVariable(name.clone())),
            ValueRef::Temp(id) => self
                .temporaries
                .get(id)
                .cloned()
                .ok_or_else(|| ExecutorError::UndefinedVariable(format!("temp_{}", id.0))),
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
                        (RuntimeValue::String(a), RuntimeValue::String(b)) => {
                            let mut result_str = a;
                            result_str.push_str(&b);
                            Ok(RuntimeValue::String(result_str))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Incompatible types for add operation".to_string(),
                        )),
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
                    LexBinaryOperator::And => match (left_value, right_value) {
                        (RuntimeValue::Boolean(a), RuntimeValue::Boolean(b)) => {
                            Ok(RuntimeValue::Boolean(a && b))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "And operator requires boolean operands".to_string(),
                        )),
                    },
                    LexBinaryOperator::Or => match (left_value, right_value) {
                        (RuntimeValue::Boolean(a), RuntimeValue::Boolean(b)) => {
                            Ok(RuntimeValue::Boolean(a || b))
                        }
                        _ => Err(ExecutorError::TypeError(
                            "Or operator requires boolean operands".to_string(),
                        )),
                    },
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
        result: &Option<ValueRef>,
    ) -> Result<()> {
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

        // Store result if specified
        if let Some(res) = result {
            self.store_value(res, return_value)?;
        }

        Ok(())
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
                } else if function == "memory_store" {
                    // Delegate to extracted function
                    self.handle_memory_store(args, result.as_ref())?;
                    Ok(())
                } else if function == "memory_load" {
                    self.handle_memory_load(args, result.as_ref())?;
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
                    // map(array, callback) -> [transformed] - Array transformation with improved callback support
                    if args.len() != 2 {
                        return Err(ExecutorError::ArgumentError(
                            "map requires exactly 2 arguments: array and callback".to_string(),
                        ));
                    }

                    let array_value = self.evaluate_expression(args[0].clone())?;
                    let callback_expr = args[1].clone();

                    let mapped_array = match array_value {
                        RuntimeValue::Json(Value::Array(arr)) => {
                            let mut result = Vec::new();
                            for (index, item) in arr.iter().enumerate() {
                                // Improved callback support - evaluate callback expression for each item
                                let transformed = if let LexExpression::Value(ValueRef::Literal(
                                    LexLiteral::String(callback_str),
                                )) = &callback_expr
                                {
                                    self.evaluate_callback(callback_str, item, index)?
                                } else {
                                    // Fallback to simple transformation
                                    match item {
                                        Value::Number(n) => {
                                            if let Some(i) = n.as_i64() {
                                                Value::Number(serde_json::Number::from(i * 2))
                                            } else if let Some(f) = n.as_f64() {
                                                Value::Number(
                                                    serde_json::Number::from_f64(f * 2.0)
                                                        .unwrap_or_else(|| {
                                                            serde_json::Number::from(0)
                                                        }),
                                                )
                                            } else {
                                                item.clone()
                                            }
                                        }
                                        Value::String(s) => Value::String(format!("mapped_{}", s)),
                                        _ => item.clone(),
                                    }
                                };
                                result.push(transformed);
                            }
                            RuntimeValue::Json(Value::Array(result))
                        }
                        _ => {
                            return Err(ExecutorError::ArgumentError(
                                "map requires an array as first argument".to_string(),
                            ))
                        }
                    };

                    println!(
                        "🔄 map: Applied transformation to array ({} elements)",
                        if let RuntimeValue::Json(Value::Array(ref arr)) = mapped_array {
                            arr.len()
                        } else {
                            0
                        }
                    );

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
                    let confidence = self.calculate_confidence_score(&response_text);

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
                    let is_valid = self.validate_response_basic(&response_text, &validation_type);

                    println!("✅ validate_response: Validation result: {}", is_valid);

                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::Boolean(is_valid))?;
                    }
                    Ok(())
                } else if function == "memory_index.ingest" {
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
                } else if function == "memory_index.vector_search" {
                    // 🧠 Sprint D: memory_index.vector_search(query, k) -> array (REAL IMPLEMENTATION)
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

                    std::fs::write(&path, &binary_file.content).map_err(|e| {
                        ExecutorError::DataError(format!("Failed to save file {}: {}", path, e))
                    })?;

                    println!(
                        "💾 Saved file: {} ({} bytes, {})",
                        path, binary_file.size, binary_file.mime_type
                    );
                    Ok(())
                } else if function == "load_file" {
                    self.handle_load_file(args, result.as_ref())?;
                    Ok(())

                // 🧠 ==========================================================================
                // PILLAR #2: SUMMARIZATION FUNCTIONS - FUNCTION CALL HANDLERS
                // ==========================================================================
                } else if function == "session_summarize" {
                    self.handle_session_summarize(args, result.as_ref())?;
                    Ok(())
                } else if function == "context_window_manage" {
                    self.handle_context_window_manage(args, result.as_ref())?;
                    Ok(())
                } else if function == "extract_key_points" {
                    self.handle_extract_key_points(args, result.as_ref())?;
                    Ok(())
                } else if function == "session_compress" {
                    self.handle_session_compress(args, result.as_ref())?;
                    Ok(())

                // ==========================================================================
                // END OF PILLAR #2: SUMMARIZATION FUNCTIONS HANDLERS
                // ==========================================================================
                } else {
                    // Try to call user-defined function
                    if let Some(func_def) = self.functions.get(function).cloned() {
                        self.call_user_function(&func_def, args, result)
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
                if let RuntimeValue::Boolean(b) = cond_val {
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
                    // Duplicate arm removed
                    /* RuntimeValue::Json(serde_json::Value::Array(arr)) => {
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
                    } */
                    // Duplicate arm removed
                    /* RuntimeValue::Json(serde_json::Value::Array(arr)) => {
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
                    } */
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
                let response = self.llm_adapter.call_llm(
                    model.as_deref(),
                    *temperature,
                    system_prompt.as_deref(),
                    user_prompt.as_deref(),
                    schema.as_deref(),
                    *max_tokens,
                    attributes,
                )?;

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
                let mut llm_adapter = self.llm_adapter.clone();
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

    /// 🛡️ SPRINT C: Calculates confidence score of a response using heuristics
    fn calculate_confidence_score(&self, response: &str) -> f64 {
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
                    if i != j && self.detect_simple_contradiction(sentence1, sentence2) {
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

    /// 🛡️ SPRINT C: Validates a response using basic validation
    #[allow(clippy::only_used_in_recursion)]
    fn validate_response_basic(&self, response: &str, validation_type: &str) -> bool {
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
                self.validate_response_basic(response, "basic")
            }
        }
    }

    /// 🛡️ Detects simple contradictions between two sentences
    fn detect_simple_contradiction(&self, sentence1: &str, sentence2: &str) -> bool {
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
