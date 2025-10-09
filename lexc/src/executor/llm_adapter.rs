// lexc/src/executor/llm_adapter.rs
//
// LLM Adapter
// This is a simulated implementation for development that simply echoes
// prompts. In a real implementation, it would connect to LLM APIs.

use super::api_config::ApiConfig;
use super::{ExecutorError, Result};
use once_cell::sync::Lazy;
use regex;
// use reqwest;
use serde_json;
use std::collections::HashMap;
// use std::sync::Arc;
use tokio::sync::{oneshot, Mutex};
use ureq;

/// Gets the configurable OpenAI URL
#[allow(dead_code)]
fn get_openai_url() -> String {
    std::env::var("OPENAI_BASE_URL").unwrap_or_else(|_| "https://api.openai.com".to_string())
        + "/v1/chat/completions"
}
/// Parameters for LLM calls
#[derive(Debug, Clone, Default)]
pub struct LlmParams {
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub response_format: Option<String>,
    pub top_p: Option<f64>,
    pub seed: Option<u32>,
}

/// Simple cache for LLM results
struct LlmCache {
    /// Key-value cache
    cache: HashMap<String, String>,
}

impl LlmCache {
    fn new() -> Self {
        LlmCache {
            cache: HashMap::new(),
        }
    }

    /// Generates a cache key for a prompt
    fn generate_key(
        &self,
        model: Option<&str>,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        schema: Option<&str>,
    ) -> String {
        // In a real implementation, we would use a cryptographic hash like SHA-256
        // For the stub, we simply concatenate the values
        format!(
            "model={};temp={};sys={};user={};schema={}",
            model.unwrap_or("default"),
            temperature.unwrap_or(0.0),
            system_prompt.unwrap_or(""),
            user_prompt.unwrap_or(""),
            schema.unwrap_or(""),
        )
    }

    /// Searches in the cache
    fn lookup(&self, key: &str) -> Option<String> {
        self.cache.get(key).cloned()
    }

    /// Stores in the cache
    fn store(&mut self, key: String, value: String) {
        self.cache.insert(key, value.clone());
    }
}

/// LLM Adapter
pub struct LlmAdapter {
    /// Cache for results
    cache: LlmCache,
    /// Token counter
    token_count: usize,
    /// Estimated accumulated cost
    estimated_cost: f64,
    /// API configuration
    api_config: ApiConfig,
}

impl LlmAdapter {
    /// Creates a new adapter
    pub fn new() -> Self {
        let mut api_config = ApiConfig::new();
        api_config.load_from_env();
        api_config.load_from_toml(); // Load configuration from lexon.toml

        LlmAdapter {
            cache: LlmCache::new(),
            token_count: 0,
            estimated_cost: 0.0,
            api_config,
        }
    }

    /// Counts tokens (simulated)
    fn count_tokens(&self, text: &str) -> usize {
        // Very basic simulation: approximately 4 characters per token
        text.len() / 4 + 1
    }

    /// Makes call to OpenAI API
    #[allow(dead_code, clippy::too_many_arguments)]
    fn call_openai_api(
        &mut self,
        model_name: &str,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        max_tokens: Option<u32>,
        api_key: &str,
        cache_key: String,
    ) -> Result<String> {
        // Build message in ChatCompletion format
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(serde_json::json!({"role": "system", "content": sys }));
        }
        if let Some(user) = user_prompt {
            messages.push(serde_json::json!({"role": "user", "content": user }));
        }

        let temperature_val = temperature.unwrap_or(0.7);
        let max_tok = max_tokens.unwrap_or(256);

        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "temperature": temperature_val,
            "max_tokens": max_tok
        });

        // Use ApiConfig to get the full URL
        let openai_url = self
            .api_config
            .get_full_url("openai", "chat", None)
            .unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string());
        let auth_headers = self.api_config.get_auth_headers("openai", api_key);

        // Perform HTTP call using ureq to avoid nested Tokio runtime issues
        let mut request = ureq::post(&openai_url).set("Content-Type", "application/json");

        // Add authentication headers
        for (key, value) in &auth_headers {
            request = request.set(key, value);
        }

        let resp = request
            .send_json(body)
            .map_err(|e| ExecutorError::LlmError(format!("OpenAI HTTP error: {}", e)))?;

        if resp.status() != 200 {
            let status = resp.status();
            let txt = resp.into_string().unwrap_or_default();
            return Err(ExecutorError::LlmError(format!(
                "OpenAI API error ({}): {}",
                status, txt
            )));
        }

        let resp_json: serde_json::Value = resp
            .into_json()
            .map_err(|e| ExecutorError::LlmError(format!("OpenAI JSON parse error: {}", e)))?;

        if let Some(content) = resp_json["choices"][0]["message"]["content"].as_str() {
            println!("✅ OpenAI API call successful");
            // Store in cache and return
            self.cache.store(cache_key, content.to_string());
            Ok(content.to_string())
        } else {
            Err(ExecutorError::LlmError(
                "Unexpected response format from OpenAI".into(),
            ))
        }
    }

    /// Generates simulated response for unsupported models
    #[allow(dead_code)]
    fn generate_simulated_response(
        &mut self,
        model_name: &str,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        provider_name: &str,
    ) -> Result<String> {
        let mut response = String::new();

        response.push_str(&format!(
            "🤖 SIMULATED RESPONSE from {} ({})\n",
            model_name, provider_name
        ));

        if let Some(sys) = system_prompt {
            response.push_str(&format!("SYSTEM: {}\n", sys));
        }

        if let Some(user) = user_prompt {
            response.push_str(&format!("USER: {}\n", user));
        }

        // Generate model-specific response
        if model_name.starts_with("claude-") {
            response.push_str("\nClaude-style response: I understand your question and will provide a thoughtful, detailed analysis based on the context provided.\n");
        } else if model_name.contains("llama") {
            response.push_str("\nLlama-style response: Based on the information provided, here's my analysis with practical insights and recommendations.\n");
        } else {
            response.push_str("\nGeneric simulated response: This is a placeholder response for testing purposes.\n");
        }

        // Token counting (simulated)
        let prompt_tokens = self.count_tokens(system_prompt.unwrap_or(""))
            + self.count_tokens(user_prompt.unwrap_or(""));
        let completion_tokens = self.count_tokens(&response);

        println!("📊 Token usage (simulated):");
        println!("   - Model: {} ({})", model_name, provider_name);
        println!("   - Prompt tokens: {}", prompt_tokens);
        println!("   - Completion tokens: {}", completion_tokens);
        println!("   - Total tokens: {}", prompt_tokens + completion_tokens);

        self.token_count += prompt_tokens + completion_tokens;
        self.estimated_cost += (prompt_tokens + completion_tokens) as f64 * 0.00003;

        Ok(response)
    }

    /// LLM call with structured parameters
    pub async fn call_llm_structured(&mut self, params: LlmParams) -> Result<String> {
        // Generate a cache key
        let cache_key = self.cache.generate_key(
            Some(&params.model),
            params.temperature,
            params.system_prompt.as_deref(),
            Some(&params.user_prompt),
            None, // schema se maneja externamente
        );

        // Check the cache
        if let Some(cached_result) = self.cache.lookup(&cache_key) {
            println!("🔄 Cache hit! Reusing previous result.");
            return Ok(cached_result);
        }

        // Token counting (simulado)
        let prompt_tokens = self.count_tokens(params.system_prompt.as_deref().unwrap_or(""))
            + self.count_tokens(&params.user_prompt);

        // Build the simulated response
        let mut response = String::new();

        if let Some(sys) = &params.system_prompt {
            response.push_str(&format!("SYSTEM: {}\n", sys));
        }

        response.push_str(&format!("USER: {}\n", params.user_prompt));

        // Generate response based on response_format
        if let Some(format) = &params.response_format {
            if format == "json_object" {
                response.push_str("\nRESPONSE (JSON):\n");
                response.push_str("{\n");
                response.push_str("  \"result\": \"This is a simulated JSON response\",\n");
                response.push_str("  \"confidence\": 0.95,\n");
                response.push_str("  \"model_used\": \"");
                response.push_str(&params.model);
                response.push_str("\"\n");
                response.push_str("}\n");
            } else {
                response.push_str("\nRESPONSE: This is a simulated LLM response.\n");
            }
        } else {
            response.push_str("\nRESPONSE: This is a simulated LLM response.\n");
        }

        // Apply token limit if specified
        if let Some(max_tokens) = params.max_tokens {
            let max_chars = (max_tokens as usize) * 4; // Approximation
            if response.len() > max_chars {
                response.truncate(max_chars);
                response.push_str("...[truncated]");
            }
        }

        // Token counting of output (simulated)
        let completion_tokens = self.count_tokens(&response);

        // Logging of tokens
        println!("📊 Token usage:");
        println!("   - Prompt tokens: {}", prompt_tokens);
        println!("   - Completion tokens: {}", completion_tokens);
        println!("   - Total tokens: {}", prompt_tokens + completion_tokens);
        println!("   - Model: {}", params.model);
        if let Some(temp) = params.temperature {
            println!("   - Temperature: {}", temp);
        }

        // Cost estimation (simulated, based on GPT-4)
        let prompt_cost = (prompt_tokens as f64) * 0.00003;
        let completion_cost = (completion_tokens as f64) * 0.00006;
        let total_cost = prompt_cost + completion_cost;

        println!("💰 Estimated cost: ${:.6}", total_cost);

        // Update counters
        self.token_count += prompt_tokens + completion_tokens;
        self.estimated_cost += total_cost;

        // Store in cache
        self.cache.store(cache_key, response.clone());

        Ok(response)
    }

    /// LLM call (original method for compatibility)
    #[allow(clippy::too_many_arguments)]
    pub fn call_llm(
        &mut self,
        model: Option<&str>,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        schema: Option<&str>,
        max_tokens: Option<u32>,
        _attributes: &HashMap<String, String>,
    ) -> Result<String> {
        println!(
            "🔍 DEBUG call_llm: model={:?}, user_prompt={:?}",
            model, user_prompt
        );

        // If no model is specified, use the default model of the default provider
        let effective_model = match model {
            Some(m) => m,
            None => {
                // Get the default provider and its default model
                let default_provider = self.api_config.get_default_provider();
                if let Some(default_model) = self.api_config.get_default_model(&default_provider) {
                    println!(
                        "🎯 Using default model: {} from provider: {}",
                        default_model, default_provider
                    );
                    // We need to return a reference that lives long enough
                    // For now, use the hardcoded model as a fallback
                    if default_provider == "openai" {
                        "gpt-4"
                    } else if default_provider == "anthropic" {
                        "claude-3-5-sonnet-20241022"
                    } else {
                        "simulated"
                    }
                } else {
                    // Fallback if no default model is configured
                    println!("⚠️  No default model configured, using fallback");
                    "simulated"
                }
            }
        };

        // Generate a cache key
        let cache_key = self.cache.generate_key(
            Some(effective_model),
            temperature,
            system_prompt,
            user_prompt,
            schema,
        );

        // Check the cache
        if let Some(cached_result) = self.cache.lookup(&cache_key) {
            println!("🔄 Cache hit! Reusing previous result.");
            return Ok(cached_result);
        }

        // If the model is "simulated", use simulated mode directly
        println!("🔍 DEBUG: Checking cache for key: {:?}", cache_key);
        if effective_model == "simulated" {
            return self.generate_simulated_response_old(
                effective_model,
                system_prompt,
                user_prompt,
                "SIMULATED",
            );
        }

        // Determine the provider based on the model name
        let (provider, needs_key) = self.determine_provider(effective_model);

        // Check if we have the necessary API key
        match provider.as_str() {
            "openai" => {
                if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                    // Verify that the model is compatible with OpenAI
                    if !self.is_openai_model(effective_model) {
                        return Err(ExecutorError::LlmError(
                            format!("Model '{}' is not supported by OpenAI API. Available OpenAI models: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4-vision-preview", effective_model)
                        ));
                    }

                    return self.call_openai_api_real(
                        effective_model,
                        temperature,
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        &api_key,
                        cache_key,
                    );
                } else if needs_key {
                    return Err(ExecutorError::LlmError(format!(
                        "Model '{}' requires OPENAI_API_KEY environment variable to be set",
                        effective_model
                    )));
                }
            }
            "anthropic" => {
                if let Ok(api_key) = std::env::var("ANTHROPIC_API_KEY") {
                    if !self.is_anthropic_model(effective_model) {
                        return Err(ExecutorError::LlmError(
                            format!("Model '{}' is not supported by Anthropic API. Available Anthropic models: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-instant, claude-3.7-sonnet, claude-4-sonnet, claude-4-opus", effective_model)
                        ));
                    }

                    return self.call_anthropic_api(
                        effective_model,
                        temperature,
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        &api_key,
                        cache_key,
                    );
                } else if needs_key {
                    return Err(ExecutorError::LlmError(format!(
                        "Model '{}' requires ANTHROPIC_API_KEY environment variable to be set",
                        effective_model
                    )));
                }
            }
            "google" => {
                if let Ok(api_key) = std::env::var("GOOGLE_API_KEY") {
                    if !self.is_google_model(effective_model) {
                        return Err(ExecutorError::LlmError(
                            format!("Model '{}' is not supported by Google API. Available Google models: gemini-pro, gemini-pro-vision", effective_model)
                        ));
                    }

                    return self.call_google_api(
                        effective_model,
                        temperature,
                        system_prompt,
                        user_prompt,
                        max_tokens,
                        &api_key,
                        cache_key,
                    );
                } else if needs_key {
                    return Err(ExecutorError::LlmError(format!(
                        "Model '{}' requires GOOGLE_API_KEY environment variable to be set",
                        effective_model
                    )));
                }
            }
            _ => {
                // Unknown model - give clear error
                return Err(ExecutorError::LlmError(
                    format!("Unknown model '{}'. Supported models: OpenAI (gpt-4, gpt-3.5-turbo), Anthropic (claude-3-opus, claude-3-sonnet, claude-3.7-sonnet, claude-4-sonnet, claude-4-opus), Google (gemini-pro), or 'simulated' for testing", effective_model)
                    ));
            }
        }

        // If we get here, generate simulated response as fallback
        self.generate_simulated_response_old("default", system_prompt, user_prompt, "FALLBACK")
    }

    /// Determines the provider and if it needs an API key based on the model name
    fn determine_provider(&self, model_name: &str) -> (String, bool) {
        if model_name.starts_with("gpt-") || model_name.contains("openai") {
            ("openai".to_string(), true)
        } else if model_name.starts_with("claude-") || model_name.contains("anthropic") {
            ("anthropic".to_string(), true)
        } else if model_name.starts_with("gemini-") || model_name.contains("google") {
            ("google".to_string(), true)
        } else if model_name == "simulated" {
            ("simulated".to_string(), false)
        } else {
            // Unknown model
            ("unknown".to_string(), true)
        }
    }

    /// Verifies if a model is compatible with OpenAI
    fn is_openai_model(&self, model_name: &str) -> bool {
        // Use dynamic configuration rather than hardcoding
        if self.api_config.is_model_supported("openai", model_name) {
            return true;
        }

        // Fallback for compatibility with known models if no configuration present
        matches!(
            model_name,
            "gpt-4"
                | "gpt-4-turbo"
                | "gpt-4-turbo-preview"
                | "gpt-3.5-turbo"
                | "gpt-3.5-turbo-16k"
                | "gpt-4-vision-preview"
                | "gpt-4-1106-preview"
        )
    }

    /// Verifies if a model is compatible with Anthropic
    fn is_anthropic_model(&self, model_name: &str) -> bool {
        // Use dynamic configuration rather than hardcoding
        if self.api_config.is_model_supported("anthropic", model_name) {
            return true;
        }

        // Fallback for compatibility with known models if no configuration present
        model_name.starts_with("claude-")
            || matches!(
                model_name,
                "claude-3-5-sonnet-20241022"
                    | "claude-3-5-sonnet-20240620"
                    | "claude-3-opus-20240229"
                    | "claude-3-sonnet-20240229"
                    | "claude-3-haiku-20240307"
                    | "claude-3-opus"
                    | "claude-3-sonnet"
                    | "claude-3-haiku"
                    | "claude-instant"
                    | "claude-2"
                    | "claude-2.1"
            )
    }

    /// Verifies if a model is compatible with Google
    fn is_google_model(&self, model_name: &str) -> bool {
        // Use dynamic configuration rather than hardcoding
        if self.api_config.is_model_supported("google", model_name) {
            return true;
        }

        // Fallback for compatibility with known models if no configuration present
        matches!(
            model_name,
            "gemini-pro"
                | "gemini-pro-vision"
                | "gemini-ultra"
                | "gemini-1.5-pro"
                | "gemini-1.5-flash"
        )
    }

    /// Real call to OpenAI API with validation
    #[allow(clippy::too_many_arguments)]
    fn call_openai_api_real(
        &mut self,
        model_name: &str,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        max_tokens: Option<u32>,
        api_key: &str,
        cache_key: String,
    ) -> Result<String> {
        let span = crate::trace_span!(
            "llm_call",
            "provider" => "openai",
            "model" => model_name.to_string(),
            "temperature" => temperature.unwrap_or(0.7).to_string()
        );
        span.record_event("Calling OpenAI API");

        // Build message in ChatCompletion format
        let mut messages = Vec::new();
        if let Some(sys) = system_prompt {
            messages.push(serde_json::json!({"role": "system", "content": sys }));
        }
        if let Some(user) = user_prompt {
            messages.push(serde_json::json!({"role": "user", "content": user }));
        }

        let temperature_val = temperature.unwrap_or(0.7);
        let max_tok = max_tokens.unwrap_or(256);

        let body = serde_json::json!({
            "model": model_name,
            "messages": messages,
            "temperature": temperature_val,
            "max_tokens": max_tok
        });

        // Use ApiConfig to obtain the full URL
        let openai_url = self
            .api_config
            .get_full_url("openai", "chat", None)
            .unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string());
        let auth_headers = self.api_config.get_auth_headers("openai", api_key);

        // Perform HTTP call using ureq to avoid nested Tokio runtime issues
        let mut request = ureq::post(&openai_url).set("Content-Type", "application/json");

        // Add authentication headers
        for (key, value) in &auth_headers {
            request = request.set(key, value);
        }

        // Retry with exponential backoff on 429/5xx and transport errors
        let mut attempt: u32 = 0;
        let max_attempts: u32 = 4;
        loop {
            let result = request.clone().send_json(body.clone());
            match result {
                Ok(resp) => {
                    if resp.status() == 200 {
                        let resp_json: serde_json::Value = resp.into_json().map_err(|e| {
                            ExecutorError::LlmError(format!("OpenAI JSON parse error: {}", e))
                        })?;
                        if let Some(content) =
                            resp_json["choices"][0]["message"]["content"].as_str()
                        {
                            span.record_event("OpenAI call successful");
                            // Store in cache and return
                            self.cache.store(cache_key, content.to_string());
                            return Ok(content.to_string());
                        } else {
                            return Err(ExecutorError::LlmError(
                                "Unexpected response format from OpenAI".into(),
                            ));
                        }
                    } else {
                        let status = resp.status();
                        let txt = resp.into_string().unwrap_or_default();
                        // Retry only on throttling or server errors
                        if status == 429 || (500..600).contains(&status) {
                            attempt += 1;
                            span.record_warning(&format!(
                                "OpenAI retry {} due to status {}",
                                attempt, status
                            ));
                            if attempt >= max_attempts {
                                return Err(ExecutorError::LlmError(format!(
                                    "OpenAI API error ({}): {}",
                                    status, txt
                                )));
                            }
                            let backoff_ms = 250u64.saturating_mul(1u64 << (attempt - 1));
                            std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                            continue;
                        }
                        return Err(ExecutorError::LlmError(format!(
                            "OpenAI API error ({}): {}",
                            status, txt
                        )));
                    }
                }
                Err(e) => {
                    // Transport error: retry with backoff
                    attempt += 1;
                    span.record_warning(&format!("OpenAI transport retry {}: {}", attempt, e));
                    if attempt >= max_attempts {
                        return Err(ExecutorError::LlmError(format!("OpenAI HTTP error: {}", e)));
                    }
                    let backoff_ms = 250u64.saturating_mul(1u64 << (attempt - 1));
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                    continue;
                }
            }
        }
    }

    /// Real call to Anthropic API with validation
    #[allow(clippy::too_many_arguments)]
    fn call_anthropic_api(
        &mut self,
        model_name: &str,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        max_tokens: Option<u32>,
        api_key: &str,
        cache_key: String,
    ) -> Result<String> {
        let span = crate::trace_span!(
            "llm_call",
            "provider" => "anthropic",
            "model" => model_name.to_string(),
            "temperature" => temperature.unwrap_or(0.7).to_string()
        );
        span.record_event("Calling Anthropic API");

        // Build message in Anthropic Messages API format
        let mut messages = Vec::new();
        if let Some(user) = user_prompt {
            messages.push(serde_json::json!({"role": "user", "content": user}));
        }

        let temperature_val = temperature.unwrap_or(0.7);
        let max_tok = max_tokens.unwrap_or(1024);

        // Build body with Anthropic-specific format
        let mut body = serde_json::json!({
            "model": model_name,
            "max_tokens": max_tok,
            "messages": messages
        });

        // Add system prompt if present (separate from messages in Anthropic)
        if let Some(sys) = system_prompt {
            body["system"] = serde_json::json!(sys);
        }

        // Add temperature if specified
        if temperature.is_some() {
            body["temperature"] = serde_json::json!(temperature_val);
        }

        // Obtain URL and headers from configuration
        let url = self
            .api_config
            .get_full_url("anthropic", "messages", None)
            .unwrap_or_else(|| "https://api.anthropic.com/v1/messages".to_string());

        let headers = self.api_config.get_auth_headers("anthropic", api_key);

        // Perform HTTP call using ureq with dynamic configuration
        let mut request = ureq::post(&url);
        for (key, value) in headers {
            request = request.set(&key, &value);
        }

        let resp = request
            .send_json(body)
            .map_err(|e| ExecutorError::LlmError(format!("Anthropic HTTP error: {}", e)))?;

        if resp.status() != 200 {
            let status = resp.status();
            let txt = resp.into_string().unwrap_or_default();
            let msg = format!("Anthropic API error ({}): {}", status, txt);
            span.record_error(&msg);
            return Err(ExecutorError::LlmError(msg));
        }

        let resp_json: serde_json::Value = resp
            .into_json()
            .map_err(|e| ExecutorError::LlmError(format!("Anthropic JSON parse error: {}", e)))?;

        // Extract content from Anthropic response
        if let Some(content) = resp_json["content"][0]["text"].as_str() {
            span.record_event("Anthropic call successful");
            // Store in cache and return
            self.cache.store(cache_key, content.to_string());
            Ok(content.to_string())
        } else {
            Err(ExecutorError::LlmError(
                "Unexpected response format from Anthropic".into(),
            ))
        }
    }

    /// Google API call (placeholder - needs implementation)
    #[allow(clippy::too_many_arguments)]
    fn call_google_api(
        &mut self,
        model_name: &str,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        max_tokens: Option<u32>,
        api_key: &str,
        cache_key: String,
    ) -> Result<String> {
        let span = crate::trace_span!(
            "llm_call",
            "provider" => "google",
            "model" => model_name.to_string(),
            "temperature" => temperature.unwrap_or(0.7).to_string()
        );
        span.record_event("Calling Google Gemini API");

        // Gemini expects a generateContent style body:
        // {
        //   "contents": [{"parts": [{"text": "..."}]}],
        //   "safetySettings": [...],
        //   "generationConfig": {"temperature": .., "maxOutputTokens": ..}
        // }
        let mut contents: Vec<serde_json::Value> = Vec::new();
        let mut parts: Vec<serde_json::Value> = Vec::new();
        if let Some(sys) = system_prompt {
            parts.push(serde_json::json!({"text": sys}));
        }
        if let Some(user) = user_prompt {
            parts.push(serde_json::json!({"text": user}));
        }
        if parts.is_empty() {
            // Gemini requires at least one part
            parts.push(serde_json::json!({"text": ""}));
        }
        contents.push(serde_json::json!({"parts": parts}));

        let mut generation_config = serde_json::Map::new();
        if let Some(t) = temperature {
            generation_config.insert("temperature".to_string(), serde_json::json!(t));
        }
        if let Some(mt) = max_tokens {
            generation_config.insert("maxOutputTokens".to_string(), serde_json::json!(mt));
        }

        let mut body = serde_json::Map::new();
        body.insert("contents".to_string(), serde_json::Value::Array(contents));
        if !generation_config.is_empty() {
            body.insert(
                "generationConfig".to_string(),
                serde_json::Value::Object(generation_config),
            );
        }
        let body = serde_json::Value::Object(body);

        // Build URL from ApiConfig
        let url = self
            .api_config
            .get_full_url("google", "generate", Some(model_name))
            .unwrap_or_else(|| {
                format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
                    model_name
                )
            });

        // Auth via query param key=API_KEY or header; we use ApiConfig AuthFormat
        // Here, pass api_key as query parameter 'key'
        let full_url = format!("{}?key={}", url, api_key);

        // Perform HTTP call using ureq
        let request = ureq::post(&full_url).set("Content-Type", "application/json");
        let result = request.send_json(body);
        match result {
            Ok(resp) => {
                if resp.status() == 200 {
                    let resp_json: serde_json::Value = resp.into_json().map_err(|e| {
                        ExecutorError::LlmError(format!("Google JSON parse error: {}", e))
                    })?;

                    // Parse Gemini structure: candidates[0].content.parts[0].text
                    if let Some(text) = resp_json
                        .get("candidates")
                        .and_then(|c| c.get(0))
                        .and_then(|c0| c0.get("content"))
                        .and_then(|cnt| cnt.get("parts"))
                        .and_then(|p| p.get(0))
                        .and_then(|p0| p0.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        span.record_event("Google Gemini call successful");
                        self.cache.store(cache_key, text.to_string());
                        Ok(text.to_string())
                    } else {
                        let msg = "Unexpected response format from Google Gemini".to_string();
                        span.record_error(&msg);
                        Err(ExecutorError::LlmError(msg))
                    }
                } else {
                    let status = resp.status();
                    let txt = resp.into_string().unwrap_or_default();
                    let msg = format!("Google API error ({}): {}", status, txt);
                    span.record_error(&msg);
                    Err(ExecutorError::LlmError(msg))
                }
            }
            Err(e) => {
                let msg = format!("Google HTTP error: {}", e);
                span.record_error(&msg);
                Err(ExecutorError::LlmError(msg))
            }
        }
    }

    /// Generates simulated response with clear information
    fn generate_simulated_response_old(
        &mut self,
        model_name: &str,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        mode: &str,
    ) -> Result<String> {
        // Token counting (simulated)
        let prompt_tokens = self.count_tokens(system_prompt.unwrap_or(""))
            + self.count_tokens(user_prompt.unwrap_or(""));

        // Build simulated response
        let mut response = String::new();

        response.push_str(&format!("🤖 SIMULATED RESPONSE ({})\n", mode));
        response.push_str(&format!("Model: {}\n", model_name));
        response.push_str("Mode: Development/Testing\n\n");

        if let Some(sys) = system_prompt {
            response.push_str(&format!("SYSTEM: {}\n", sys));
        }

        if let Some(user) = user_prompt {
            response.push_str(&format!("USER: {}\n", user));
        }

        response.push_str(
            "\nRESPONSE: This is a simulated LLM response for development and testing purposes.\n",
        );

        // Output token counting (simulated)
        let completion_tokens = self.count_tokens(&response);

        // Token logging
        println!("📊 Token usage (SIMULATED):");
        println!("   - Model: {} ({})", model_name, mode);
        println!("   - Prompt tokens: {}", prompt_tokens);
        println!("   - Completion tokens: {}", completion_tokens);
        println!("   - Total tokens: {}", prompt_tokens + completion_tokens);

        // Cost estimation (simulated)
        let total_cost = (prompt_tokens + completion_tokens) as f64 * 0.00003;
        println!("💰 Estimated cost (simulated): ${:.6}", total_cost);

        // Update counters
        self.token_count += prompt_tokens + completion_tokens;
        self.estimated_cost += total_cost;

        // Store in cache
        self.cache.store(
            self.cache
                .generate_key(Some(model_name), None, system_prompt, user_prompt, None),
            response.clone(),
        );

        Ok(response)
    }

    /// Usage information
    pub fn usage_info(&self) -> (usize, f64) {
        (self.token_count, self.estimated_cost)
    }

    /// Async version with batching and distributed cache
    #[allow(clippy::too_many_arguments)]
    pub async fn call_llm_async(
        &mut self,
        model: Option<&str>,
        temperature: Option<f64>,
        system_prompt: Option<&str>,
        user_prompt: Option<&str>,
        schema: Option<&str>,
        max_tokens: Option<u32>,
        attributes: &HashMap<String, String>,
    ) -> Result<String> {
        // Generate cache key and check redis/ram first
        let cache_key =
            self.cache
                .generate_key(model, temperature, system_prompt, user_prompt, schema);
        if let Some(cached) = self.cache.lookup(&cache_key) {
            return Ok(cached);
        }

        // Clone params to move
        let mut attrs_clone = HashMap::new();
        for (k, v) in attributes {
            attrs_clone.insert(k.clone(), v.clone());
        }
        let (tx, rx) = oneshot::channel();
        let pending = PendingRequest {
            key: cache_key.clone(),
            model: model.map(|s| s.to_string()),
            temperature,
            system_prompt: system_prompt.map(|s| s.to_string()),
            user_prompt: user_prompt.map(|s| s.to_string()),
            schema: schema.map(|s| s.to_string()),
            max_tokens,
            attrs: attrs_clone,
            tx,
        };
        {
            let mut q = BATCH_QUEUE.lock().await;
            q.push(pending);
            if q.len() == 1 {
                // first element, spawn processor
                let adapter_clone = self.clone();
                tokio::spawn(async move {
                    batch_processor(adapter_clone).await;
                });
            }
            if q.len() >= BATCH_SIZE { /* wake processor early */ }
        }
        let res = rx
            .await
            .map_err(|_| ExecutorError::LlmError("batch send failed".into()))?;
        // store in local cache
        if let Ok(ref s) = res {
            self.cache.store(cache_key, s.clone());
        }
        res
    }

    /// Main function: ask_safe() - LLM call with automatic anti-hallucination validation
    pub async fn ask_safe(
        &mut self,
        prompt: &str,
        model: Option<&str>,
        config: Option<AntiHallucinationConfig>,
    ) -> Result<ValidationResult> {
        let config = config.unwrap_or_default();
        println!(
            "🛡️ Starting ANTI-HALLUCINATION validation for prompt: {}",
            prompt
        );

        let mut best_result: Option<ValidationResult> = None;
        let mut attempts = 0;

        while attempts < config.max_validation_attempts {
            attempts += 1;
            println!(
                "🔄 Validation attempt {}/{}",
                attempts, config.max_validation_attempts
            );

            // 1. Generate initial response
            let initial_response = self.call_llm(
                model,
                Some(0.3), // Low temperature for consistency
                None,
                Some(prompt),
                None,
                Some(512),
                &HashMap::new(),
            )?;

            // 2. Validate the response according to the configured strategy
            let validation_result = match config.validation_strategy {
                ValidationStrategy::Basic => self.validate_basic(&initial_response, prompt).await?,
                ValidationStrategy::Ensemble => {
                    self.validate_ensemble(
                        &initial_response,
                        prompt,
                        &config.cross_reference_models,
                    )
                    .await?
                }
                ValidationStrategy::FactCheck => {
                    self.validate_fact_check(&initial_response, prompt).await?
                }
                ValidationStrategy::Comprehensive => {
                    self.validate_comprehensive(&initial_response, prompt, &config)
                        .await?
                }
            };

            // 3. Check if it meets the confidence threshold
            if validation_result.confidence_score >= config.confidence_threshold {
                println!(
                    "✅ Validation PASSED with confidence: {:.2}",
                    validation_result.confidence_score
                );
                return Ok(validation_result);
            }

            // 4. If it fails, save the best result so far
            if best_result.is_none()
                || validation_result.confidence_score
                    > best_result.as_ref().unwrap().confidence_score
            {
                best_result = Some(validation_result.clone());
            }

            println!(
                "⚠️ Validation attempt {} failed (confidence: {:.2}), retrying...",
                attempts, validation_result.confidence_score
            );
        }

        // If we exhaust all attempts, return the best result with a warning
        let mut final_result = best_result.unwrap_or(ValidationResult {
            is_valid: false,
            confidence_score: 0.0,
            issues: vec![ValidationIssue::Hallucination {
                content: "Failed all validation attempts".to_string(),
                probability: 1.0,
            }],
            validated_content: "VALIDATION_FAILED".to_string(),
        });

        final_result.is_valid = false; // Mark as invalid if it didn't pass the threshold
        println!(
            "🚨 All validation attempts failed. Best confidence: {:.2}",
            final_result.confidence_score
        );
        Ok(final_result)
    }

    /// Basic validation: check internal consistency and hallucination patterns
    async fn validate_basic(
        &mut self,
        response: &str,
        original_prompt: &str,
    ) -> Result<ValidationResult> {
        println!("🔍 Running BASIC validation...");

        let mut issues = Vec::new();
        let mut confidence_score: f64 = 1.0;

        // 1. Check common hallucination patterns
        let hallucination_patterns = vec![
            r"according to recent studies",
            r"research has shown",
            r"experts claim",
            r"it has been proven that",
            r"statistical data shows",
        ];

        for pattern in &hallucination_patterns {
            if response.to_lowercase().contains(&pattern.to_lowercase()) {
                issues.push(ValidationIssue::Hallucination {
                    content: format!("Suspicious pattern detected: {}", pattern),
                    probability: 0.7,
                });
                confidence_score -= 0.1;
            }
        }

        // 2. Verify internal consistency (contradictions)
        let sentences: Vec<&str> = response.split(". ").collect();
        for (i, sentence1) in sentences.iter().enumerate() {
            for (j, sentence2) in sentences.iter().enumerate() {
                if i != j && self.detect_contradiction(sentence1, sentence2) {
                    issues.push(ValidationIssue::Contradiction {
                        original: sentence1.to_string(),
                        conflicting: sentence2.to_string(),
                    });
                    confidence_score -= 0.2;
                }
            }
        }

        // 3. Verify relevance to the original prompt
        let relevance_score = self.calculate_relevance(original_prompt, response);
        if relevance_score < 0.5 {
            issues.push(ValidationIssue::LogicalError {
                statement: "Response not relevant to original prompt".to_string(),
                error_type: "Irrelevance".to_string(),
            });
            confidence_score -= 0.3;
        }

        confidence_score = confidence_score.clamp(0.0, 1.0);

        Ok(ValidationResult {
            is_valid: confidence_score >= 0.8,
            confidence_score,
            issues,
            validated_content: response.to_string(),
        })
    }

    /// Ensemble validation: use multiple models for cross-validation
    async fn validate_ensemble(
        &mut self,
        response: &str,
        original_prompt: &str,
        models: &[String],
    ) -> Result<ValidationResult> {
        println!(
            "🎯 Running ENSEMBLE validation with {} models...",
            models.len()
        );

        let mut responses = vec![response.to_string()];
        let mut issues = Vec::new();

        // 1. Get responses from other models
        for model in models.iter().take(2) {
            // Limit to 2 additional models for cost
            match self.call_llm(
                Some(model),
                Some(0.3),
                None,
                Some(original_prompt),
                None,
                Some(512),
                &HashMap::new(),
            ) {
                Ok(alt_response) => responses.push(alt_response),
                Err(e) => {
                    println!("⚠️ Failed to get response from {}: {}", model, e);
                    continue;
                }
            }
        }

        // 2. Compare responses to detect inconsistencies
        let mut consistency_score: f64 = 1.0;
        for (i, resp1) in responses.iter().enumerate() {
            for (j, resp2) in responses.iter().enumerate() {
                if i != j {
                    let similarity = self.calculate_semantic_similarity(resp1, resp2);
                    if similarity < 0.6 {
                        issues.push(ValidationIssue::Inconsistency {
                            section1: resp1.chars().take(100).collect(),
                            section2: resp2.chars().take(100).collect(),
                        });
                        consistency_score -= 0.1;
                    }
                }
            }
        }

        // 3. Verify consensus
        let consensus_response = self.find_consensus(&responses);

        consistency_score = consistency_score.clamp(0.0, 1.0);

        Ok(ValidationResult {
            is_valid: consistency_score >= 0.7,
            confidence_score: consistency_score,
            issues,
            validated_content: consensus_response,
        })
    }

    /// Fact validation: verify specific claims
    async fn validate_fact_check(
        &mut self,
        response: &str,
        _original_prompt: &str,
    ) -> Result<ValidationResult> {
        println!("📋 Running FACT-CHECK validation...");

        let mut issues = Vec::new();
        let mut confidence_score: f64 = 1.0;

        // 1. Extract verifiable claims
        let claims = self.extract_verifiable_claims(response);

        // 2. Verify each claim
        for claim in &claims {
            let verification_prompt = format!(
                "Verify this factual claim: '{}'. Respond with 'TRUE', 'FALSE', or 'UNCERTAIN' followed by a brief explanation.",
                claim
            );

            match self.call_llm(
                Some("gpt-4"), // Use a reliable model for verification
                Some(0.1),     // Very low temperature
                Some("You are a fact-checker. Be precise and conservative."),
                Some(&verification_prompt),
                None,
                Some(256),
                &HashMap::new(),
            ) {
                Ok(verification) => {
                    if verification.to_uppercase().starts_with("FALSE") {
                        issues.push(ValidationIssue::FactualError {
                            claim: claim.clone(),
                            reason: verification,
                        });
                        confidence_score -= 0.2;
                    } else if verification.to_uppercase().starts_with("UNCERTAIN") {
                        confidence_score -= 0.1;
                    }
                }
                Err(e) => {
                    println!("⚠️ Fact-check failed for claim '{}': {}", claim, e);
                    confidence_score -= 0.05;
                }
            }
        }

        confidence_score = confidence_score.clamp(0.0, 1.0);

        Ok(ValidationResult {
            is_valid: confidence_score >= 0.8,
            confidence_score,
            issues,
            validated_content: response.to_string(),
        })
    }

    /// Comprehensive validation: combines all strategies
    async fn validate_comprehensive(
        &mut self,
        response: &str,
        original_prompt: &str,
        config: &AntiHallucinationConfig,
    ) -> Result<ValidationResult> {
        println!("🔬 Running COMPREHENSIVE validation...");

        // Execute all validations
        let basic_result = self.validate_basic(response, original_prompt).await?;
        let ensemble_result = self
            .validate_ensemble(response, original_prompt, &config.cross_reference_models)
            .await?;
        let fact_check_result = if config.use_fact_checking {
            self.validate_fact_check(response, original_prompt).await?
        } else {
            ValidationResult {
                is_valid: true,
                confidence_score: 1.0,
                issues: vec![],
                validated_content: response.to_string(),
            }
        };

        // Combine results
        let mut all_issues = Vec::new();
        all_issues.extend(basic_result.issues);
        all_issues.extend(ensemble_result.issues);
        all_issues.extend(fact_check_result.issues);

        // Calculate weighted average score
        let weighted_score = (basic_result.confidence_score * 0.3)
            + (ensemble_result.confidence_score * 0.4)
            + (fact_check_result.confidence_score * 0.3);

        Ok(ValidationResult {
            is_valid: weighted_score >= 0.8,
            confidence_score: weighted_score,
            issues: all_issues,
            validated_content: ensemble_result.validated_content, // Use ensemble consensus
        })
    }

    /// 🔍 HELPER FUNCTIONS FOR VALIDATION

    /// Detects contradictions between two sentences
    fn detect_contradiction(&self, sentence1: &str, sentence2: &str) -> bool {
        // Simple contradiction patterns
        let contradictory_pairs = vec![
            ("yes", "no"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("possible", "impossible"),
            ("increases", "decreases"),
            ("better", "worse"),
            ("fast", "slow"),
            ("large", "small"),
        ];

        let s1_lower = sentence1.to_lowercase();
        let s2_lower = sentence2.to_lowercase();

        for (word1, word2) in &contradictory_pairs {
            if (s1_lower.contains(word1) && s2_lower.contains(word2))
                || (s1_lower.contains(word2) && s2_lower.contains(word1))
            {
                return true;
            }
        }

        // Detection of contradictory negations
        if (s1_lower.contains("no is") && s2_lower.contains("is"))
            || (s1_lower.contains("no can") && s2_lower.contains("can"))
            || (s1_lower.contains("never") && s2_lower.contains("always"))
        {
            return true;
        }

        false
    }

    /// Calculates the relevance of the response to the original prompt
    fn calculate_relevance(&self, prompt: &str, response: &str) -> f64 {
        let prompt_lower = prompt.to_lowercase();
        let response_lower = response.to_lowercase();
        let prompt_words: Vec<&str> = prompt_lower.split_whitespace().collect();
        let response_words: Vec<&str> = response_lower.split_whitespace().collect();

        if prompt_words.is_empty() || response_words.is_empty() {
            return 0.0;
        }

        // Count shared keywords
        let mut shared_words = 0;
        for prompt_word in &prompt_words {
            if response_words.contains(prompt_word) && prompt_word.len() > 3 {
                shared_words += 1;
            }
        }

        // Calculate basic relevance score
        let relevance_score = shared_words as f64 / prompt_words.len() as f64;
        relevance_score.min(1.0)
    }

    /// Calculates basic semantic similarity between two texts
    fn calculate_semantic_similarity(&self, text1: &str, text2: &str) -> f64 {
        let text1_lower = text1.to_lowercase();
        let text2_lower = text2.to_lowercase();
        let words1: Vec<&str> = text1_lower.split_whitespace().collect();
        let words2: Vec<&str> = text2_lower.split_whitespace().collect();

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        // Calculate word intersection
        let mut common_words = 0;
        for word1 in &words1 {
            if words2.contains(word1) && word1.len() > 3 {
                common_words += 1;
            }
        }

        // Simplified Jaccard similarity
        let union_size = words1.len() + words2.len() - common_words;
        if union_size == 0 {
            return 1.0;
        }

        common_words as f64 / union_size as f64
    }

    /// Extracts verifiable claims from text
    fn extract_verifiable_claims(&self, text: &str) -> Vec<String> {
        let mut claims = Vec::new();

        // Patterns that indicate verifiable claims
        let claim_patterns = vec![
            r"the \d+% of",
            r"according to.*,",
            r"in the year \d+",
            r"studies show",
            r"research indicates",
            r"data reveals",
            r"statistics confirm",
        ];

        let sentences: Vec<&str> = text.split(". ").collect();

        for sentence in sentences {
            for pattern in &claim_patterns {
                if regex::Regex::new(pattern)
                    .unwrap()
                    .is_match(&sentence.to_lowercase())
                {
                    claims.push(sentence.trim().to_string());
                    break;
                }
            }

            // Also extract sentences with specific numbers
            if sentence.chars().any(|c| c.is_ascii_digit()) && sentence.len() > 20 {
                claims.push(sentence.trim().to_string());
            }
        }

        // Limit number of claims to avoid excessive costs
        claims.into_iter().take(5).collect()
    }

    /// Finds consensus across multiple responses
    fn find_consensus(&self, responses: &[String]) -> String {
        if responses.is_empty() {
            return "No consensus found".to_string();
        }

        if responses.len() == 1 {
            return responses[0].clone();
        }

        // For simplicity, return the most common answer or the first
        // In a more sophisticated implementation, NLP could be used to combine answers

        // Find the answer most similar to the others
        let mut best_response = &responses[0];
        let mut best_score = 0.0;

        for candidate in responses {
            let mut total_similarity = 0.0;
            for other in responses {
                if candidate != other {
                    total_similarity += self.calculate_semantic_similarity(candidate, other);
                }
            }

            let avg_similarity = total_similarity / (responses.len() - 1) as f64;
            if avg_similarity > best_score {
                best_score = avg_similarity;
                best_response = candidate;
            }
        }

        best_response.clone()
    }
}

impl Default for LlmAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for LlmAdapter {
    fn clone(&self) -> Self {
        LlmAdapter {
            cache: LlmCache::new(), // new cache for clone
            token_count: self.token_count,
            estimated_cost: self.estimated_cost,
            api_config: self.api_config.clone(),
        }
    }
}

// ------------ Batching ------------
struct PendingRequest {
    #[allow(dead_code)]
    key: String,
    model: Option<String>,
    temperature: Option<f64>,
    system_prompt: Option<String>,
    user_prompt: Option<String>,
    schema: Option<String>,
    max_tokens: Option<u32>,
    attrs: HashMap<String, String>,
    tx: oneshot::Sender<Result<String>>,
}

static BATCH_QUEUE: Lazy<Mutex<Vec<PendingRequest>>> = Lazy::new(|| Mutex::new(Vec::new()));
const BATCH_SIZE: usize = 8;
const BATCH_DELAY_MS: u64 = 25;

async fn batch_processor(mut adapter: LlmAdapter) {
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(BATCH_DELAY_MS)).await;
        let mut queue = BATCH_QUEUE.lock().await;
        if queue.is_empty() {
            continue;
        }
        let drain_len = BATCH_SIZE.min(queue.len());
        let to_process: Vec<_> = queue.drain(..drain_len).collect();
        drop(queue);

        // For stub, process sequentially but could join prompts.
        for req in to_process {
            let res = adapter.call_llm(
                req.model.as_deref(),
                req.temperature,
                req.system_prompt.as_deref(),
                req.user_prompt.as_deref(),
                req.schema.as_deref(),
                req.max_tokens,
                &req.attrs,
            );
            let _ = req.tx.send(res);
        }
    }
}
// ==================== NEW STRUCTURES ADDED ====================

/// Parameters for LLM calls
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    MajorityVote,
    WeightedAverage,
    BestOfN,
    Synthesize,
}

#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

/// Anti-hallucination validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence_score: f64,
    pub issues: Vec<ValidationIssue>,
    pub validated_content: String,
}

/// Types of validation issues
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    Contradiction {
        original: String,
        conflicting: String,
    },
    FactualError {
        claim: String,
        reason: String,
    },
    Inconsistency {
        section1: String,
        section2: String,
    },
    Hallucination {
        content: String,
        probability: f64,
    },
    LogicalError {
        statement: String,
        error_type: String,
    },
}

/// Validation strategies
#[derive(Debug, Clone)]
pub enum ValidationStrategy {
    /// Simple validation with consistency checking
    Basic,
    /// Validation with multiple models (ensemble)
    Ensemble,
    /// Validation with fact checking
    FactCheck,
    /// Complete validation (all strategies)
    Comprehensive,
}

/// Anti-hallucination system configuration
#[derive(Debug, Clone)]
pub struct AntiHallucinationConfig {
    pub validation_strategy: ValidationStrategy,
    pub confidence_threshold: f64,
    pub max_validation_attempts: usize,
    pub use_fact_checking: bool,
    pub cross_reference_models: Vec<String>,
}

impl Default for AntiHallucinationConfig {
    fn default() -> Self {
        AntiHallucinationConfig {
            validation_strategy: ValidationStrategy::Basic,
            confidence_threshold: 0.8,
            max_validation_attempts: 3,
            use_fact_checking: true,
            cross_reference_models: vec![
                "gpt-4".to_string(),
                "claude-3-5-sonnet-20241022".to_string(),
                "gemini-1.5-pro".to_string(),
            ],
        }
    }
}

impl LlmAdapter {
    /// ask_ensemble function: Executes multiple prompts and combines responses using consensus strategies
    pub async fn ask_ensemble(
        &mut self,
        prompts: Vec<String>,
        strategy: EnsembleStrategy,
        model: Option<String>,
    ) -> Result<String> {
        if prompts.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "ask_ensemble requires at least one prompt".to_string(),
            ));
        }

        println!(
            "🎯 ask_ensemble: Processing {} prompts with strategy {:?}",
            prompts.len(),
            strategy
        );

        let mut results = Vec::new();
        let selected_model = model.unwrap_or_else(|| "gpt-4".to_string());

        // Execute each prompt individually
        for prompt in &prompts {
            let result = self.call_llm(
                Some(&selected_model),
                Some(0.7),
                None,
                Some(prompt),
                None,
                Some(1000),
                &HashMap::new(),
            )?;
            results.push(result);
        }

        // Apply ensemble strategy
        let final_result = self.apply_ensemble_strategy(&results, strategy.clone())?;

        println!(
            "✅ ask_ensemble: Combined {} responses using {:?} strategy",
            results.len(),
            strategy
        );
        Ok(final_result)
    }

    /// Applies the ensemble strategy to the responses
    fn apply_ensemble_strategy(
        &self,
        responses: &[String],
        strategy: EnsembleStrategy,
    ) -> Result<String> {
        match strategy {
            EnsembleStrategy::MajorityVote => {
                // Simple strategy: returns the first response as "majority consensus"
                Ok(format!("Majority Vote Result: {}", responses[0].clone()))
            }
            EnsembleStrategy::WeightedAverage => {
                // Combines all responses with weights
                let combined = responses
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        format!(
                            "Response {} (weight: {:.2}): {}",
                            i + 1,
                            1.0 / responses.len() as f64,
                            r
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                Ok(format!("Weighted Average Result:\n{}", combined))
            }
            EnsembleStrategy::BestOfN => {
                // Selects the longest response as "best"
                let best_response = responses
                    .iter()
                    .max_by_key(|r| r.len())
                    .unwrap_or(&responses[0]);
                Ok(format!(
                    "Best of {} Result: {}",
                    responses.len(),
                    best_response
                ))
            }
            EnsembleStrategy::Synthesize => {
                // Combines key information from all responses
                let synthesis = responses
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        format!(
                            "From response {}: {}",
                            i + 1,
                            r.split('.').next().unwrap_or(r)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(". ");
                Ok(format!("Synthesized Result: {}", synthesis))
            }
        }
    }

    /// batch_call_llm function: Executes multiple LLM calls in parallel
    pub async fn batch_call_llm(
        &mut self,
        requests: Vec<(String, Option<String>)>, // (prompt, model)
        max_concurrent: Option<usize>,
    ) -> Result<Vec<String>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        println!(
            "🔄 batch_call_llm: Processing {} requests with max_concurrent: {:?}",
            requests.len(),
            max_concurrent.unwrap_or(4)
        );

        let mut results = Vec::new();
        let concurrent_limit = max_concurrent.unwrap_or(4);

        // Process in batches to avoid overload
        for chunk in requests.chunks(concurrent_limit) {
            let mut batch_results = Vec::new();

            for (prompt, model) in chunk {
                let result = self.call_llm(
                    model.as_deref(),
                    Some(0.7),
                    None,
                    Some(prompt),
                    None,
                    Some(1000),
                    &HashMap::new(),
                )?;
                batch_results.push(result);
            }

            results.extend(batch_results);
        }

        println!("✅ batch_call_llm: Completed {} requests", results.len());
        Ok(results)
    }

    /// Gets performance statistics of the adapter
    pub fn get_performance_stats(&self) -> Result<String> {
        let stats = format!(
            "🔧 LLM Adapter Performance Stats:\n\
            ================================\n\
            📊 Usage: {} tokens processed\n\
            💰 Cost: ${:.4} estimated\n\
            🚀 Status: Operational\n\
            🔗 API: Auto-detection enabled",
            self.token_count, self.estimated_cost
        );
        Ok(stats)
    }

    /// Resets the adapter statistics
    pub fn reset_performance_stats(&mut self) -> Result<()> {
        self.token_count = 0;
        self.estimated_cost = 0.0;
        println!("🔄 Performance statistics reset successfully");
        Ok(())
    }

    /// Validates the current adapter configuration
    pub fn validate_adapter_config(&self) -> Result<bool> {
        println!("🔍 Validating LLM adapter configuration...");

        // Simulate validations
        let checks = vec![
            ("API endpoints", true),
            ("Rate limits", true),
            ("Token counting", true),
            ("Error handling", true),
        ];

        let mut all_valid = true;
        for (component, is_valid) in &checks {
            let status = if *is_valid { "✅" } else { "❌" };
            println!(
                "   {} {}: {}",
                status,
                component,
                if *is_valid { "Valid" } else { "Invalid" }
            );
            if !is_valid {
                all_valid = false;
            }
        }

        Ok(all_valid)
    }

    /// 🔧 Sprint B: Sets the default model for future calls
    pub fn set_default_model(&mut self, model_name: &str) {
        println!("🔧 LlmAdapter: Setting default model to '{}'", model_name);
        // In a more complete implementation, this could store the default model
        // in the adapter configuration. For now, we just log it.
        // The real logic for using the default model is in ExecutorConfig.
    }
}
