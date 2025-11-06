// lexc/src/executor/memory_functions.rs
//
// Memory and session functions extracted from executor/mod.rs
// This module handles all memory-related operations including:
// - memory_store, memory_load, memory_clear
// - session_start, session_ask, session_history
// - session_summarize, session_compress
// - context_window_manage, extract_key_points

use crate::lexir::{LexExpression, ValueRef};
// unused imports removed

use super::{ExecutionEnvironment, ExecutorError, RuntimeValue};
use std::collections::HashMap;

impl ExecutionEnvironment {
    /// Handle memory_store function - Store data in memory
    pub fn handle_memory_store(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // memory_store(key, value) -> void - Store a value in persistent memory
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "memory_store requires key and value".to_string(),
            ));
        }

        // Evaluate key
        let key_value = self.evaluate_expression(args[0].clone())?;
        let key = match key_value {
            RuntimeValue::String(s) => s,
            _ => format!("{:?}", key_value),
        };

        // Evaluate value to store
        let value_to_store = self.evaluate_expression(args[1].clone())?;

        println!(
            "💾 memory_store: Storing key '{}' with value type {:?}",
            key,
            match &value_to_store {
                RuntimeValue::String(_) => "String",
                RuntimeValue::Integer(_) => "Integer",
                RuntimeValue::Float(_) => "Float",
                RuntimeValue::Boolean(_) => "Boolean",
                RuntimeValue::Dataset(_) => "Dataset",
                RuntimeValue::Json(_) => "Json",
                RuntimeValue::Null => "Null",
                RuntimeValue::Result { .. } => "Result",
                RuntimeValue::MultiOutput { .. } => "MultiOutput",
            }
        );

        // Store in memory using default scope "global"
        self.memory_manager
            .store_memory("global", value_to_store, Some(&key))?;

        println!("✅ memory_store: Successfully stored key '{}'", key);

        // memory_store does not return a value, but if there's a result, we store true
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Boolean(true))?;
        }
        Ok(())
    }

    /// Handle memory_load function - Load data from memory
    pub fn handle_memory_load(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "memory_load requires key".to_string(),
            ));
        }

        let key_value = self.evaluate_expression(args[0].clone())?;
        let key = match key_value {
            RuntimeValue::String(s) => s,
            _ => format!("{:?}", key_value),
        };

        println!("🔍 memory_load: Loading key '{}'", key);
        match self.memory_manager.load_value_by_key("global", &key) {
            Ok(Some(loaded_value)) => {
                if let Some(res) = result {
                    self.store_value(res, loaded_value)?;
                }
            }
            Ok(None) => {
                if let Some(res) = result {
                    self.store_value(res, RuntimeValue::Null)?;
                }
            }
            Err(e) => {
                return Err(ExecutorError::DataError(format!(
                    "Failed to load memory key '{}': {}",
                    key, e
                )));
            }
        }
        Ok(())
    }

    /// Handle session_start function - Start a new session
    pub fn handle_session_start(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        self.maybe_run_session_gc();
        let model = if !args.is_empty() {
            let model_arg = self.evaluate_expression(args[0].clone())?;
            match model_arg {
                RuntimeValue::String(s) => s,
                _ => "gpt-3.5-turbo".to_string(),
            }
        } else {
            "gpt-3.5-turbo".to_string()
        };

        let session_name = if args.len() > 1 {
            let name_arg = self.evaluate_expression(args[1].clone())?;
            match name_arg {
                RuntimeValue::String(s) => s,
                _ => format!(
                    "session_{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                ),
            }
        } else {
            format!(
                "session_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            )
        };

        println!(
            "🔗 session_start: Creating session '{}' with model {}",
            session_name, model
        );
        let session_id = session_name.clone();
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(session_id))?;
        }
        Ok(())
    }

    /// Handle session_ask function - Ask within a session context
    pub fn handle_session_ask(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        self.maybe_run_session_gc();
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "session_ask requires session_id and user_prompt".to_string(),
            ));
        }

        let session_id_arg = self.evaluate_expression(args[0].clone())?;
        let session_id = match session_id_arg {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "session_id must be string".to_string(),
                ))
            }
        };
        let user_prompt_arg = self.evaluate_expression(args[1].clone())?;
        let user_prompt = match user_prompt_arg {
            RuntimeValue::String(s) => s,
            _ => format!("{:?}", user_prompt_arg),
        };

        println!(
            "💬 session_ask: Session '{}' - User: {}",
            session_id, user_prompt
        );
        let mut llm_adapter = self.llm_adapter.clone();
        let response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                llm_adapter
                    .call_llm_async(
                        Some("gpt-3.5-turbo"),
                        Some(0.7),
                        Some(&format!(
                            "You are in session '{}'. Maintain conversation context.",
                            session_id
                        )),
                        Some(&user_prompt),
                        None,
                        None,
                        &std::collections::HashMap::new(),
                    )
                    .await
            })
        })?;
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(response))?;
        }
        Ok(())
    }

    /// Handle session_history function - Get session history
    pub fn handle_session_history(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        self.maybe_run_session_gc();
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "session_history requires session_id".to_string(),
            ));
        }
        let session_id_arg = self.evaluate_expression(args[0].clone())?;
        let session_id = match session_id_arg {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "session_id must be string".to_string(),
                ))
            }
        };
        println!(
            "📚 session_history: Retrieving history for session '{}'",
            session_id
        );
        let history = format!(
            "Session History for '{}'\n============================\n\n\
                         This session contains the conversation history and context.\n\
                         In a full implementation, this would retrieve actual conversation logs.",
            session_id
        );
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(history))?;
        }
        Ok(())
    }

    /// Handle session_summarize function - Summarize session content
    pub fn handle_session_summarize(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        self.maybe_run_session_gc();
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "session_summarize requires session_id".to_string(),
            ));
        }

        let session_id_value = self.evaluate_expression(args[0].clone())?;
        let session_id = match session_id_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "session_id must be string".to_string(),
                ))
            }
        };

        let options = if args.len() > 1 {
            let options_arg = self.evaluate_expression(args[1].clone())?;
            match options_arg {
                RuntimeValue::Json(serde_json::Value::Object(map)) => {
                    let mut runtime_options = HashMap::new();
                    for (key, value) in map {
                        let runtime_value = match value {
                            serde_json::Value::String(s) => RuntimeValue::String(s),
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    RuntimeValue::Integer(i)
                                } else if let Some(f) = n.as_f64() {
                                    RuntimeValue::Float(f)
                                } else {
                                    RuntimeValue::String(n.to_string())
                                }
                            }
                            serde_json::Value::Bool(b) => RuntimeValue::Boolean(b),
                            _ => RuntimeValue::String(value.to_string()),
                        };
                        runtime_options.insert(key, runtime_value);
                    }
                    runtime_options
                }
                _ => HashMap::new(),
            }
        } else {
            HashMap::new()
        };

        let summary = self.session_summarize(&session_id, &options)?;
        if let Some(res) = result {
            self.store_value(res, summary)?;
        }
        Ok(())
    }

    /// Handle session_compress function - Compress session data
    pub fn handle_session_compress(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        self.maybe_run_session_gc();
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "session_compress requires session_id".to_string(),
            ));
        }

        let session_id_value = self.evaluate_expression(args[0].clone())?;
        let session_id = match session_id_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "session_id must be string".to_string(),
                ))
            }
        };

        let options = if args.len() > 1 {
            let options_arg = self.evaluate_expression(args[1].clone())?;
            match options_arg {
                RuntimeValue::Json(serde_json::Value::Object(map)) => {
                    let mut runtime_options = HashMap::new();
                    for (key, value) in map {
                        let runtime_value = match value {
                            serde_json::Value::String(s) => RuntimeValue::String(s),
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    RuntimeValue::Integer(i)
                                } else if let Some(f) = n.as_f64() {
                                    RuntimeValue::Float(f)
                                } else {
                                    RuntimeValue::String(n.to_string())
                                }
                            }
                            serde_json::Value::Bool(b) => RuntimeValue::Boolean(b),
                            _ => RuntimeValue::String(value.to_string()),
                        };
                        runtime_options.insert(key, runtime_value);
                    }
                    runtime_options
                }
                _ => HashMap::new(),
            }
        } else {
            HashMap::new()
        };

        let compressed = self.session_compress(&session_id, &options)?;
        if let Some(res) = result {
            self.store_value(res, compressed)?;
        }
        Ok(())
    }

    /// Handle context_window_manage function - Manage context window
    pub fn handle_context_window_manage(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "context_window_manage requires session_id".to_string(),
            ));
        }

        let session_id_value = self.evaluate_expression(args[0].clone())?;
        let session_id = match session_id_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "session_id must be string".to_string(),
                ))
            }
        };

        let options = if args.len() > 1 {
            let options_arg = self.evaluate_expression(args[1].clone())?;
            match options_arg {
                RuntimeValue::Json(serde_json::Value::Object(map)) => {
                    let mut runtime_options = HashMap::new();
                    for (key, value) in map {
                        let runtime_value = match value {
                            serde_json::Value::String(s) => RuntimeValue::String(s),
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    RuntimeValue::Integer(i)
                                } else if let Some(f) = n.as_f64() {
                                    RuntimeValue::Float(f)
                                } else {
                                    RuntimeValue::String(n.to_string())
                                }
                            }
                            serde_json::Value::Bool(b) => RuntimeValue::Boolean(b),
                            _ => RuntimeValue::String(value.to_string()),
                        };
                        runtime_options.insert(key, runtime_value);
                    }
                    runtime_options
                }
                _ => HashMap::new(),
            }
        } else {
            HashMap::new()
        };

        let managed = self.context_window_manage(&session_id, &options)?;
        if let Some(res) = result {
            self.store_value(res, managed)?;
        }
        Ok(())
    }

    /// Handle extract_key_points function - Extract key points from content
    pub fn handle_extract_key_points(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "extract_key_points requires session_id".to_string(),
            ));
        }

        let session_id_value = self.evaluate_expression(args[0].clone())?;
        let session_id = match session_id_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "session_id must be string".to_string(),
                ))
            }
        };

        let options = if args.len() > 1 {
            let options_arg = self.evaluate_expression(args[1].clone())?;
            match options_arg {
                RuntimeValue::Json(serde_json::Value::Object(map)) => {
                    let mut runtime_options = HashMap::new();
                    for (key, value) in map {
                        let runtime_value = match value {
                            serde_json::Value::String(s) => RuntimeValue::String(s),
                            serde_json::Value::Number(n) => {
                                if let Some(i) = n.as_i64() {
                                    RuntimeValue::Integer(i)
                                } else if let Some(f) = n.as_f64() {
                                    RuntimeValue::Float(f)
                                } else {
                                    RuntimeValue::String(n.to_string())
                                }
                            }
                            serde_json::Value::Bool(b) => RuntimeValue::Boolean(b),
                            _ => RuntimeValue::String(value.to_string()),
                        };
                        runtime_options.insert(key, runtime_value);
                    }
                    runtime_options
                }
                _ => HashMap::new(),
            }
        } else {
            HashMap::new()
        };

        let points = self.extract_key_points(&session_id, &options)?;
        if let Some(res) = result {
            self.store_value(res, points)?;
        }
        Ok(())
    }
}
