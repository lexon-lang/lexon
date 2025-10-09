// src/runtime/mod.rs
//
// Runtime module to execute Lexon programs compiled to LexIR
// Supports asynchronous execution and resource management

pub mod data;
pub mod llm;
pub mod memory;
pub mod scheduler;

use crate::lexir::{LexExpression, LexInstruction, LexLiteral, LexProgram, ValueRef};
use crate::runtime::llm::{LlmMessage, LlmModel, LlmOptions};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
// use std::sync::{Arc, Mutex};

/// Error types that can occur during execution
#[derive(Debug, Clone)]
pub enum RuntimeError {
    /// Instruction error: the instruction cannot be executed
    InstructionError(String),
    /// Type error: the type of a value is not compatible with the operation
    TypeError(String),
    /// Value error: a value cannot be converted to the expected type
    ValueError(String),
    /// Reference error: a reference to a variable or value does not exist
    ReferenceError(String),
    /// Async error: an error during asynchronous execution
    AsyncError(String),
    /// LLM error: an error in communication with the language model
    LlmError(String),
    /// Memory error: an error in contextual memory management
    MemoryError(String),
    /// Data error: an error in data management
    DataError(String),
    /// Unexpected error
    Internal(String),
}

impl std::error::Error for RuntimeError {}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::InstructionError(msg) => write!(f, "Instruction error: {}", msg),
            RuntimeError::TypeError(msg) => write!(f, "Type error: {}", msg),
            RuntimeError::ValueError(msg) => write!(f, "Value error: {}", msg),
            RuntimeError::ReferenceError(msg) => write!(f, "Reference error: {}", msg),
            RuntimeError::AsyncError(msg) => write!(f, "Async error: {}", msg),
            RuntimeError::LlmError(msg) => write!(f, "LLM error: {}", msg),
            RuntimeError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
            RuntimeError::DataError(msg) => write!(f, "Data error: {}", msg),
            RuntimeError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

/// Result alias with RuntimeError
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Runtime values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RuntimeValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    List(Vec<RuntimeValue>),
    Map(HashMap<String, RuntimeValue>),
    Dataset(String), // Use String to enable serialization
    // Async execution specific values
    Future(String), // Use String as a placeholder for serialization
    Null,
}

impl RuntimeValue {
    // Helper methods if needed
}

// Keep a single definition; implement Default via a manual impl if needed later

/// Runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Path for contextual memory storage
    pub memory_path: Option<String>,
    /// Memory limit (in bytes)
    pub memory_limit: Option<usize>,
    /// LLM model name/version to use
    pub llm_model: Option<String>,
    /// API key for the LLM model
    pub llm_api_key: Option<String>,
    /// Enable verbose mode
    pub verbose: bool,
    /// Timeout for operations (milliseconds)
    pub timeout_ms: Option<u64>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        RuntimeConfig {
            memory_path: None,
            memory_limit: None,
            llm_model: None,
            llm_api_key: None,
            verbose: false,
            timeout_ms: Some(30000), // 30 seconds default
        }
    }
}

/// Runtime execution environment
pub struct Runtime {
    /// Runtime configuration
    config: RuntimeConfig,
    /// Contextual memory manager
    memory_manager: memory::MemoryManager,
    /// LLM adapter
    llm_adapter: llm::LlmAdapter,
    /// Data processor
    #[allow(dead_code)]
    data_processor: data::DataProcessor,
    /// Async scheduler
    #[allow(dead_code)]
    scheduler: scheduler::AsyncScheduler,
}

impl Runtime {
    /// Creates a new runtime with the specified configuration
    pub fn new(config: RuntimeConfig) -> Self {
        let memory_manager = memory::MemoryManager::new(&config);
        let llm_adapter = llm::LlmAdapter::new(&config);
        let data_processor = data::DataProcessor::new();
        let scheduler = scheduler::AsyncScheduler::new(8); // Default: 8 concurrent tasks

        Runtime {
            config,
            memory_manager,
            llm_adapter,
            data_processor,
            scheduler,
        }
    }

    /// Executes a LexIR program
    pub async fn execute_program(&mut self, program: &LexProgram) -> Result<()> {
        // For now, execute instructions sequentially
        // Later, we may implement parallel execution where possible
        for instruction in &program.instructions {
            self.execute_instruction(instruction).await?;
        }

        // If there is a main function, execute it
        if let Some(main_fn) = program.functions.get("main") {
            for instruction in &main_fn.body {
                self.execute_instruction(instruction).await?;
            }
        }

        Ok(())
    }

    /// Executes a LexIR instruction
    async fn execute_instruction(&mut self, instruction: &LexInstruction) -> Result<RuntimeValue> {
        match instruction {
            LexInstruction::Ask {
                result: _,
                system_prompt,
                user_prompt,
                model,
                temperature,
                max_tokens,
                schema: _,
                attributes: _,
            } => {
                // Build messages for the LLM in the classic order system→user
                let mut messages = Vec::new();
                if let Some(sys) = system_prompt {
                    messages.push(LlmMessage::system(sys));
                }
                if let Some(user) = user_prompt {
                    messages.push(LlmMessage::user(user));
                }

                // Select model: first the one specified in the instruction, then the runtime; default to Simulated
                let model_name = model.as_ref().or(self.config.llm_model.as_ref());
                let model_enum = match model_name {
                    Some(name) if name.contains("gpt") => LlmModel::OpenAI(name.clone()),
                    Some(name) => LlmModel::Custom(name.clone()),
                    None => LlmModel::Simulated,
                };

                // Options for the call
                let options = LlmOptions {
                    model: model_enum,
                    temperature: temperature.map(|t| t as f32),
                    max_tokens: *max_tokens,
                    ..Default::default()
                };

                // Call the LLM
                let response = self.llm_adapter.call(messages, Some(options)).await?;

                // Backward compatibility: print the generated response
                println!("{}", response.content);

                Ok(RuntimeValue::String(response.content))
            }
            LexInstruction::MemoryStore {
                scope,
                value,
                key,
                options: _,
            } => {
                // For now we only support simple literal values
                let val = match value {
                    ValueRef::Literal(lit) => self.literal_to_runtime(lit),
                    _ => RuntimeValue::Null,
                };

                // Ignore TTL and options in this basic phase
                self.memory_manager
                    .store(scope, val, key.as_deref(), None)?;

                Ok(RuntimeValue::Null)
            }
            LexInstruction::MemoryLoad {
                result: _,
                scope,
                source,
                strategy: _,
                options: _,
            } => {
                // If a key is passed as a literal string, use it; otherwise, retrieve all entries
                let key_opt = match source {
                    Some(ValueRef::Literal(LexLiteral::String(s))) => Some(s.as_str()),
                    _ => None,
                };

                let values = self.memory_manager.retrieve(scope, key_opt, Some(1))?;
                if let Some(v) = values.first() {
                    // Print for visibility
                    println!("[memory_load] {} => {:?}", scope, v);
                }
                Ok(RuntimeValue::Null)
            }
            _ => Err(RuntimeError::InstructionError(format!(
                "Instruction not implemented yet: {:?}",
                instruction
            ))),
        }
    }

    /// Evaluates a LexIR expression
    #[allow(dead_code)]
    async fn evaluate_expression(&self, expression: &LexExpression) -> Result<RuntimeValue> {
        Err(RuntimeError::InstructionError(format!(
            "Expression evaluation not implemented yet: {:?}",
            expression
        )))
    }

    /// Helper conversion from literal to RuntimeValue
    #[allow(clippy::only_used_in_recursion)]
    fn literal_to_runtime(&self, lit: &LexLiteral) -> RuntimeValue {
        match lit {
            LexLiteral::Integer(i) => RuntimeValue::Integer(*i),
            LexLiteral::Float(f) => RuntimeValue::Float(*f),
            LexLiteral::String(s) => RuntimeValue::String(s.clone()),
            LexLiteral::Boolean(b) => RuntimeValue::Boolean(*b),
            LexLiteral::Array(arr) => {
                let runtime_elements: Vec<RuntimeValue> =
                    arr.iter().map(|lit| self.literal_to_runtime(lit)).collect();
                RuntimeValue::List(runtime_elements)
            }
        }
    }
}

/// Helper function to execute a LexIR program
pub async fn execute(program: &LexProgram, config: Option<RuntimeConfig>) -> Result<()> {
    let config = config.unwrap_or_default();
    let mut runtime = Runtime::new(config);
    runtime.execute_program(program).await
}
