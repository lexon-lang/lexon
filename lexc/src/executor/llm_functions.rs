// lexc/src/executor/llm_functions.rs
//
// LLM-related functions extracted from executor/mod.rs
// This module handles all Large Language Model operations including:
// - ask_parallel, ask_merge, ask_ensemble
// - ask_with_fallback, ask_multioutput
// - confidence_score, validate_response
// - multimodal_request

use crate::lexir::{LexExpression, ValueRef};
use serde_json::Value;
use std::collections::HashMap;

use super::llm_adapter;
use super::BinaryFile;
use super::{ExecutionEnvironment, ExecutorError, RuntimeValue};
use crate::telemetry::trace_ask_operation;

impl ExecutionEnvironment {
    /// Handle ask_parallel function - Execute multiple ask operations in parallel
    pub fn handle_ask_parallel(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // ask_parallel(...args) -> JSON array - Advanced DSL function
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_parallel requires at least one argument".to_string(),
            ));
        }

        println!("🚀 ask_parallel: Processing {} asks", args.len());
        let mut responses = Vec::new();

        for (idx, arg) in args.iter().enumerate() {
            let arg_value = self.evaluate_expression(arg.clone())?;
            let user_prompt = match arg_value {
                RuntimeValue::String(s) => s,
                _ => format!("{:?}", arg_value),
            };

            let span = trace_ask_operation(&user_prompt, Some("gpt-4"));
            span.record_event("ask_parallel: calling LLM");

            let mut llm_adapter = self.llm_adapter.clone();
            let response = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    llm_adapter
                        .call_llm_async(
                            Some("gpt-4"),
                            Some(0.7),
                            None,
                            Some(&user_prompt),
                            None,
                            None,
                            &HashMap::new(),
                        )
                        .await
                })
            })?;

            responses.push(Value::String(response));
            println!("✅ ask_parallel: Completed {}/{}", idx + 1, args.len());
        }

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(Value::Array(responses)))?;
        }
        Ok(())
    }

    /// Handle ask_merge function - Merge results from parallel ask operations
    pub fn handle_ask_merge(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // ask_merge(responses, strategy?) -> string - Advanced DSL function
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_merge requires responses argument".to_string(),
            ));
        }

        let responses_value = self.evaluate_expression(args[0].clone())?;
        let strategy = if args.len() > 1 {
            match self.evaluate_expression(args[1].clone())? {
                RuntimeValue::String(s) => s,
                _ => "synthesize".to_string(),
            }
        } else {
            "synthesize".to_string()
        };

        let responses_text = match responses_value {
            RuntimeValue::Json(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>()
                .join("\n---\n"),
            RuntimeValue::String(s) => s,
            _ => "Invalid responses format".to_string(),
        };

        let merge_prompt = format!(
            "Consolidate these responses using {} strategy:\n\n{}\n\nConsolidated result:",
            strategy, responses_text
        );

        let mut llm_adapter = self.llm_adapter.clone();
        let span = trace_ask_operation(&merge_prompt, Some("gpt-4"));
        span.record_event("ask_merge: calling LLM");
        let merged_response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                llm_adapter
                    .call_llm_async(
                        Some("gpt-4"),
                        Some(0.3),
                        Some("You are an expert at consolidating information."),
                        Some(&merge_prompt),
                        None,
                        None,
                        &HashMap::new(),
                    )
                    .await
            })
        })?;

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(merged_response))?;
        }
        Ok(())
    }

    /// Handle ask_ensemble function - Ensemble consensus from multiple models
    pub fn handle_ask_ensemble(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // ask_ensemble(prompts, strategy?, model?) -> consensus_response - Advanced DSL function
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_ensemble requires prompts argument".to_string(),
            ));
        }

        // Evaluate prompts (can be JSON array or semicolon-separated string)
        let prompts_value = self.evaluate_expression(args[0].clone())?;
        let prompts = match prompts_value {
            RuntimeValue::Json(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>(),
            RuntimeValue::String(s) => {
                if s.contains(';') {
                    s.split(';').map(|p| p.trim().to_string()).collect()
                } else {
                    vec![s]
                }
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "prompts must be array or semicolon-separated string".to_string(),
                ))
            }
        };

        if prompts.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_ensemble requires at least one prompt".to_string(),
            ));
        }

        // Evaluate strategy (optional, default "synthesize")
        let strategy_str = if args.len() > 1 {
            match self.evaluate_expression(args[1].clone())? {
                RuntimeValue::String(s) => s,
                _ => "synthesize".to_string(),
            }
        } else {
            "synthesize".to_string()
        };

        // Convert string to enum
        let strategy = match strategy_str.to_lowercase().as_str() {
            "majority_vote" | "majority" => llm_adapter::EnsembleStrategy::MajorityVote,
            "weighted_average" | "weighted" => llm_adapter::EnsembleStrategy::WeightedAverage,
            "best_of_n" | "best" => llm_adapter::EnsembleStrategy::BestOfN,
            "synthesize" | "synthesis" => llm_adapter::EnsembleStrategy::Synthesize,
            _ => llm_adapter::EnsembleStrategy::Synthesize,
        };

        // Evaluate model (optional)
        let model = if args.len() > 2 {
            match self.evaluate_expression(args[2].clone())? {
                RuntimeValue::String(s) => Some(s),
                _ => None,
            }
        } else {
            None
        };

        println!(
            "🎯 ask_ensemble: Processing {} prompts with strategy {:?}",
            prompts.len(),
            strategy
        );

        // Use new LLM adapter if available
        if let Some(ref mut llm_adapter_new) = self.llm_adapter_new {
            let consensus_response = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    llm_adapter_new.ask_ensemble(prompts, strategy, model).await
                })
            })?;

            if let Some(res) = result {
                self.store_value(res, RuntimeValue::String(consensus_response))?;
            }
        } else {
            // Fallback to legacy adapter with basic functionality
            println!("⚠️ Using legacy LLM adapter for ask_ensemble (limited functionality)");

            let mut responses = Vec::new();
            for prompt in &prompts {
                let mut llm_adapter = self.llm_adapter.clone();
                let response = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async {
                        llm_adapter
                            .call_llm_async(
                                model.as_deref(),
                                Some(0.7),
                                None,
                                Some(prompt),
                                None,
                                None,
                                &HashMap::new(),
                            )
                            .await
                    })
                })?;
                responses.push(response);
            }

            // Basic synthesis for legacy adapter
            let synthesis_prompt = format!(
                "Synthesize these {} responses into a coherent consensus:\n\n{}",
                responses.len(),
                responses
                    .iter()
                    .enumerate()
                    .map(|(i, r)| format!("Response {}: {}", i + 1, r))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            );

            let mut llm_adapter = self.llm_adapter.clone();
            let span = trace_ask_operation(&synthesis_prompt, model.as_deref());
            span.record_event("ask_ensemble (legacy): calling LLM");
            let consensus_response = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    llm_adapter.call_llm_async(
                        model.as_deref(), Some(0.3),
                        Some("You are synthesizing a final consensus from multiple rounds of analysis."),
                        Some(&synthesis_prompt), None, None, &HashMap::new()
                    ).await
                })
            })?;

            if let Some(res) = result {
                self.store_value(res, RuntimeValue::String(consensus_response))?;
            }
        }

        Ok(())
    }

    /// Handle ask_with_fallback function - Ask with fallback model support
    pub fn handle_ask_with_fallback(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // ask_with_fallback([models], prompt) -> string - System for reliability fallback
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "ask_with_fallback requires models array and prompt".to_string(),
            ));
        }

        // Evaluate list of models (array or comma-separated string)
        let models_value = self.evaluate_expression(args[0].clone())?;
        let models = match models_value {
            RuntimeValue::Json(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect::<Vec<String>>(),
            RuntimeValue::String(s) => {
                if s.contains(',') {
                    s.split(',').map(|m| m.trim().to_string()).collect()
                } else {
                    vec![s]
                }
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "models must be array or comma-separated string".to_string(),
                ))
            }
        };

        if models.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_with_fallback requires at least one model".to_string(),
            ));
        }

        // Evaluate prompt
        let prompt_value = self.evaluate_expression(args[1].clone())?;
        let prompt = match prompt_value {
            RuntimeValue::String(s) => s,
            _ => format!("{:?}", prompt_value),
        };

        println!(
            "🔄 ask_with_fallback: Trying {} models for reliability",
            models.len()
        );

        let mut last_error = String::new();
        let mut attempt = 0;

        // Try each model in order until one works
        for model in &models {
            attempt += 1;
            println!(
                "🎯 ask_with_fallback: Attempt {}/{} with model '{}'",
                attempt,
                models.len(),
                model
            );

            let mut llm_adapter = self.llm_adapter.clone();
            let span = trace_ask_operation(&prompt, Some(model));
            span.record_event("ask_with_fallback: calling model");
            match tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    llm_adapter
                        .call_llm_async(
                            Some(model),
                            Some(0.7),
                            None,
                            Some(&prompt),
                            None,
                            None,
                            &HashMap::new(),
                        )
                        .await
                })
            }) {
                Ok(response) => {
                    println!(
                        "✅ ask_with_fallback: SUCCESS with model '{}' on attempt {}",
                        model, attempt
                    );
                    if let Some(res) = result {
                        self.store_value(res, RuntimeValue::String(response))?;
                    }
                    return Ok(());
                }
                Err(e) => {
                    last_error = format!("Model '{}': {}", model, e);
                    println!("❌ ask_with_fallback: Failed with model '{}': {}", model, e);
                    continue;
                }
            }
        }

        // If we reach here, all models failed
        let fallback_response = format!(
            "❌ FALLBACK ERROR: All {} models failed.\nLast error: {}\nModels tried: {}\nPrompt: {}",
            models.len(),
            last_error,
            models.join(", "),
            prompt
        );

        println!("💥 ask_with_fallback: All models failed, returning error response");

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(fallback_response))?;
        }
        Ok(())
    }

    /// Handle ask_multioutput function - Generate multiple files from single LLM call
    pub fn handle_ask_multioutput(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // ask_multioutput(prompt, output_files) -> MultiOutput
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "ask_multioutput requires exactly 2 arguments: prompt and output_files".to_string(),
            ));
        }

        let prompt_value = self.evaluate_expression(args[0].clone())?;
        let output_files_value = self.evaluate_expression(args[1].clone())?;

        let prompt = match prompt_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "ask_multioutput prompt must be a string".to_string(),
                ))
            }
        };

        let output_files = match output_files_value {
            RuntimeValue::Json(Value::Array(arr)) => {
                let mut files = Vec::new();
                for item in arr {
                    if let Value::String(filename) = item {
                        files.push(filename);
                    }
                }
                files
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "ask_multioutput output_files must be an array of strings".to_string(),
                ))
            }
        };

        println!(
            "🎯 ask_multioutput: Generating response with {} output files",
            output_files.len()
        );
        println!("📝 Prompt: {}", prompt);
        println!("📁 Output files: {:?}", output_files);

        // Generate LLM response
        let mut llm_adapter = self.llm_adapter.clone();
        let span = trace_ask_operation(&prompt, Some("auto"));
        span.record_event("ask_multioutput: calling LLM");
        // Honor default model set via set_default_model(...)
        let model_to_use: Option<String> = self.config.llm_model.clone();
        let llm_response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                llm_adapter
                    .call_llm_async(
                        model_to_use.as_deref(),
                        Some(0.7),
                        None,
                        Some(&prompt),
                        None,
                        None,
                        &HashMap::new(),
                    )
                    .await
            })
        })?;

        // Optional pseudo-streaming to stdout (opt-in)
        let stream_text = std::env::var("LEXON_STREAM_TEXT")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if stream_text {
            println!("📤 Streaming primary_text (pseudo-stream):");
            for line in llm_response.lines() {
                println!("{}", line);
            }
        }

        // Generate files based on requested names
        let mut binary_files = Vec::new();
        let real_binaries = std::env::var("LEXON_REAL_BINARIES")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        for filename in &output_files {
            let file = if real_binaries {
                // Real text payloads derived from LLM response with simple templating by extension
                match filename.split('.').last().unwrap_or("txt") {
                    "json" => {
                        let json_data = serde_json::json!({
                            "text": llm_response,
                            "filename": filename,
                            "generated_by": "ask_multioutput",
                        });
                        BinaryFile::from_json(filename.clone(), &json_data)?
                    }
                    "csv" => {
                        // Emit CSV with first line as header and a body synthesized from the response (truncated)
                        let snippet = llm_response.lines().next().unwrap_or("");
                        let csv_content =
                            format!("summary,snippet\nllm_output,{}", snippet.replace(',', ";"));
                        BinaryFile::from_text(filename.clone(), csv_content)
                    }
                    _ => {
                        // Default to text with the full LLM response
                        BinaryFile::from_text(filename.clone(), llm_response.clone())
                    }
                }
            } else {
                // Deterministic stubs (RC default)
                match filename.split('.').last().unwrap_or("txt") {
                    "json" => {
                        let json_data = serde_json::json!({
                            "generated_by": "ask_multioutput",
                            "prompt": prompt,
                            "filename": filename,
                            "data": "This is simulated JSON data generated by the multioutput system"
                        });
                        BinaryFile::from_json(filename.clone(), &json_data)?
                    }
                    "csv" => {
                        let csv_content =
                            "column1,column2,column3\nvalue1,value2,value3\ndata1,data2,data3";
                        BinaryFile::from_text(filename.clone(), csv_content.to_string())
                    }
                    "txt" => {
                        let txt_content = format!(
                            "Generated text file for prompt: {}\nFilename: {}\nGenerated by: ask_multioutput",
                            prompt, filename
                        );
                        BinaryFile::from_text(filename.clone(), txt_content)
                    }
                    _ => {
                        let default_content = format!("Binary file generated for: {}", filename);
                        BinaryFile::from_text(filename.clone(), default_content)
                    }
                }
            };
            binary_files.push(file);
        }

        let mut metadata = HashMap::new();
        metadata.insert("generated_by".to_string(), "ask_multioutput".to_string());
        metadata.insert("prompt_length".to_string(), prompt.len().to_string());
        metadata.insert("file_count".to_string(), binary_files.len().to_string());

        let multioutput =
            self.create_multioutput_with_metadata(llm_response, binary_files, metadata);

        println!(
            "✅ ask_multioutput: Generated response with {} files",
            output_files.len()
        );

        if let Some(res) = result {
            self.store_value(res, multioutput)?;
        }
        Ok(())
    }

    // confidence_score, validate_response, and multimodal_request are handled in executor/mod.rs.
    // Duplicate handlers were removed here to avoid confusion.

    /// Join all: run multiple prompts concurrently and return all responses
    pub fn handle_join_all(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "join_all requires prompts array [and optional model]".to_string(),
            ));
        }

        let prompts_val = self.evaluate_expression(args[0].clone())?;
        let prompts: Vec<String> = match prompts_val {
            RuntimeValue::Json(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            RuntimeValue::String(s) => {
                if s.contains(';') {
                    s.split(';').map(|p| p.trim().to_string()).collect()
                } else {
                    vec![s]
                }
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "prompts must be array or semicolon-separated string".to_string(),
                ))
            }
        };

        let model_opt = if args.len() > 1 {
            match self.evaluate_expression(args[1].clone())? {
                RuntimeValue::String(s) => Some(s),
                _ => None,
            }
        } else {
            self.config.llm_model.clone()
        };

        // Removed unused pre-allocation; results are collected below
        let vec_len = prompts.len();
        let join_res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut set = tokio::task::JoinSet::new();
                for p in prompts {
                    let mut adapter = self.llm_adapter.clone();
                    let m = model_opt.clone();
                    set.spawn(async move {
                        adapter
                            .call_llm_async(
                                m.as_deref(),
                                Some(0.7),
                                None,
                                Some(&p),
                                None,
                                None,
                                &HashMap::new(),
                            )
                            .await
                    });
                }
                let mut out = Vec::with_capacity(vec_len);
                while let Some(item) = set.join_next().await {
                    match item {
                        Ok(Ok(s)) => out.push(Value::String(s)),
                        Ok(Err(e)) => out.push(Value::String(format!("ERROR: {}", e))),
                        Err(e) => out.push(Value::String(format!("JOIN_ERROR: {}", e))),
                    }
                }
                out
            })
        });
        let responses = join_res;
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(Value::Array(responses)))?;
        }
        Ok(())
    }

    /// Join any: run multiple prompts concurrently and return the first successful response
    pub fn handle_join_any(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "join_any requires prompts array [and optional model]".to_string(),
            ));
        }

        let prompts_val = self.evaluate_expression(args[0].clone())?;
        let prompts: Vec<String> = match prompts_val {
            RuntimeValue::Json(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            RuntimeValue::String(s) => {
                if s.contains(';') {
                    s.split(';').map(|p| p.trim().to_string()).collect()
                } else {
                    vec![s]
                }
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "prompts must be array or semicolon-separated string".to_string(),
                ))
            }
        };

        let model_opt = if args.len() > 1 {
            match self.evaluate_expression(args[1].clone())? {
                RuntimeValue::String(s) => Some(s),
                _ => None,
            }
        } else {
            self.config.llm_model.clone()
        };

        let first_ok = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let mut set = tokio::task::JoinSet::new();
                for p in prompts {
                    let mut adapter = self.llm_adapter.clone();
                    let m = model_opt.clone();
                    set.spawn(async move {
                        adapter
                            .call_llm_async(
                                m.as_deref(),
                                Some(0.7),
                                None,
                                Some(&p),
                                None,
                                None,
                                &HashMap::new(),
                            )
                            .await
                    });
                }
                while let Some(item) = set.join_next().await {
                    if let Ok(Ok(s)) = item {
                        return Some(s);
                    }
                }
                None
            })
        });

        match first_ok {
            Some(s) => {
                if let Some(res) = result {
                    self.store_value(res, RuntimeValue::String(s))?;
                }
                Ok(())
            }
            None => Err(ExecutorError::RuntimeError(
                "join_any: all tasks failed".to_string(),
            )),
        }
    }

    /// ask_stream(prompt[, model]) -> string (streams chunks to stdout with structured lines)
    pub fn handle_ask_stream(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_stream requires prompt".to_string(),
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

        let mut llm_adapter = self.llm_adapter.clone();
        let response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                llm_adapter
                    .call_llm_async(
                        model_opt.as_deref(),
                        Some(0.7),
                        None,
                        Some(&prompt),
                        None,
                        None,
                        &HashMap::new(),
                    )
                    .await
            })
        })?;

        // Simulated chunked streaming: fixed-size chunks
        let chunk_size: usize = std::env::var("LEXON_STREAM_CHUNK_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(64);
        let total_len = response.len();
        let mut idx = 0usize;
        while idx < total_len {
            if self.cancel_requested {
                return Err(ExecutorError::RuntimeError("cancelled".to_string()));
            }
            let end = (idx + chunk_size).min(total_len);
            let chunk = &response[idx..end];
            println!("STREAM_CHUNK {{\"type\":\"llm\",\"offset\":{},\"len\":{},\"total\":{},\"data\":{:?}}}", idx, chunk.len(), total_len, chunk);
            idx = end;
        }
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(response))?;
        }
        Ok(())
    }

    /// ask_multioutput_stream(prompt, output_files[, model]) -> MultiOutput (streams chunk progress)
    pub fn handle_ask_multioutput_stream(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "ask_multioutput_stream requires prompt and output_files".to_string(),
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
        let files_val = self.evaluate_expression(args[1].clone())?;
        let output_files: Vec<String> = match files_val {
            RuntimeValue::Json(Value::Array(arr)) => arr
                .into_iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            RuntimeValue::String(s) => vec![s],
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "output_files must be array of strings".to_string(),
                ))
            }
        };
        let model_opt = if args.len() > 2 {
            match self.evaluate_expression(args[2].clone())? {
                RuntimeValue::String(s) => Some(s),
                _ => None,
            }
        } else {
            self.config.llm_model.clone()
        };

        // LLM call
        let mut llm_adapter = self.llm_adapter.clone();
        let llm_response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                llm_adapter
                    .call_llm_async(
                        model_opt.as_deref(),
                        Some(0.7),
                        None,
                        Some(&prompt),
                        None,
                        None,
                        &HashMap::new(),
                    )
                    .await
            })
        })?;

        // Stream pseudo-progress for file generation
        println!(
            "STREAM_EVENT {{\"type\":\"multioutput\",\"stage\":\"start\",\"files\":{}}}",
            output_files.len()
        );
        let mut binary_files = Vec::new();
        for (i, filename) in output_files.iter().enumerate() {
            println!("STREAM_EVENT {{\"type\":\"multioutput\",\"stage\":\"file_start\",\"index\":{},\"name\":{:?}}}", i, filename);
            let file = BinaryFile::from_text(filename.clone(), llm_response.clone());
            println!("STREAM_EVENT {{\"type\":\"multioutput\",\"stage\":\"file_done\",\"index\":{},\"name\":{:?},\"bytes\":{}}}", i, filename, file.size);
            binary_files.push(file);
        }
        println!("STREAM_EVENT {{\"type\":\"multioutput\",\"stage\":\"end\"}}");
        let mo = self.create_multioutput(llm_response, binary_files);
        if let Some(res) = result {
            self.store_value(res, mo)?;
        }
        Ok(())
    }
}
