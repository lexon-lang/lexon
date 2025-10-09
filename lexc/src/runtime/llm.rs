// src/runtime/llm.rs
//
// Adapter for Large Language Models (LLMs)
// Provides a unified interface across providers

use super::{Result, RuntimeConfig, RuntimeError};
use crate::telemetry::trace_ask_operation;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// LLM provider
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum LlmProvider {
    /// OpenAI (GPT-3.5, GPT-4, etc.)
    OpenAI,
    /// Anthropic (Claude, etc.)
    Anthropic,
    /// Simulated (for tests)
    Simulated,
    /// Custom (external models)
    Custom(u32),
}

/// LLM model type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmModel {
    /// OpenAI models
    OpenAI(String),
    /// Anthropic models
    Anthropic(String),
    /// Simulated model (for tests)
    Simulated,
    /// Custom model
    Custom(String),
}

impl LlmModel {
    /// Returns the model name
    pub fn name(&self) -> String {
        match self {
            LlmModel::OpenAI(name) => name.clone(),
            LlmModel::Anthropic(name) => name.clone(),
            LlmModel::Simulated => "simulated".to_string(),
            LlmModel::Custom(name) => name.clone(),
        }
    }

    /// Returns the model provider
    pub fn provider(&self) -> LlmProvider {
        match self {
            LlmModel::OpenAI(_) => LlmProvider::OpenAI,
            LlmModel::Anthropic(_) => LlmProvider::Anthropic,
            LlmModel::Simulated => LlmProvider::Simulated,
            LlmModel::Custom(_) => LlmProvider::Custom(0),
        }
    }
}

/// LLM message format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    /// Message role (system, user, assistant)
    pub role: String,
    /// Message content
    pub content: String,
    /// Extra metadata
    pub metadata: Option<HashMap<String, String>>,
}

impl LlmMessage {
    /// Create a system message
    pub fn system(content: &str) -> Self {
        LlmMessage {
            role: "system".to_string(),
            content: content.to_string(),
            metadata: None,
        }
    }

    /// Create a user message
    pub fn user(content: &str) -> Self {
        LlmMessage {
            role: "user".to_string(),
            content: content.to_string(),
            metadata: None,
        }
    }

    /// Create an assistant message
    pub fn assistant(content: &str) -> Self {
        LlmMessage {
            role: "assistant".to_string(),
            content: content.to_string(),
            metadata: None,
        }
    }

    /// Attach metadata to the message
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        let metadata = self.metadata.get_or_insert_with(HashMap::new);
        metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Options for LLM call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmOptions {
    /// Model to use
    pub model: LlmModel,
    /// Temperature (0.0 - 1.0)
    pub temperature: Option<f32>,
    /// Max tokens to generate
    pub max_tokens: Option<u32>,
    /// Stop tokens
    pub stop_tokens: Option<Vec<String>>,
    /// Top-p (nucleus sampling)
    pub top_p: Option<f32>,
    /// Frequency penalty
    pub frequency_penalty: Option<f32>,
    /// Presence penalty
    pub presence_penalty: Option<f32>,
    /// Token streaming
    pub stream: bool,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Session id for continuity
    pub session_id: Option<String>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for LlmOptions {
    fn default() -> Self {
        LlmOptions {
            model: LlmModel::OpenAI("gpt-3.5-turbo".to_string()),
            temperature: Some(0.7),
            max_tokens: None,
            stop_tokens: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stream: false,
            timeout_ms: Some(60000), // 60 seconds default
            session_id: None,
            seed: None,
        }
    }
}

/// Tokenization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizationMode {
    /// Approximate tokenization (faster)
    Approximate,
    /// Precise tokenization (model tokenizer)
    Precise,
}

/// LLM response
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Generated content
    pub content: String,
    /// Full message list (including history)
    pub messages: Vec<LlmMessage>,
    /// Token usage
    pub token_usage: TokenUsage,
    /// Model used
    pub model: LlmModel,
    /// Execution time
    pub execution_time: Duration,
    /// Extra metadata
    pub metadata: HashMap<String, String>,
}

/// Token usage info
#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    /// Prompt tokens
    pub prompt_tokens: usize,
    /// Completion tokens
    pub completion_tokens: usize,
    /// Total tokens
    pub total_tokens: usize,
    /// Estimated cost (USD)
    pub estimated_cost_usd: f64,
}

/// Cache for LLM responses
struct LlmCache {
    /// Internal cache (key: prompt hash, value: response)
    cache: HashMap<String, LlmResponse>,
    /// Max capacity
    capacity: usize,
    /// Keys in LRU order
    lru_keys: Vec<String>,
}

impl LlmCache {
    /// Create new cache
    fn new(capacity: usize) -> Self {
        LlmCache {
            cache: HashMap::with_capacity(capacity),
            capacity,
            lru_keys: Vec::with_capacity(capacity),
        }
    }

    /// Get entry from cache
    fn get(&mut self, key: &str) -> Option<LlmResponse> {
        if let Some(response) = self.cache.get(key) {
            // Update LRU
            if let Some(pos) = self.lru_keys.iter().position(|k| k == key) {
                self.lru_keys.remove(pos);
            }
            self.lru_keys.push(key.to_string());

            Some(response.clone())
        } else {
            None
        }
    }

    /// Insert entry into cache
    fn insert(&mut self, key: String, response: LlmResponse) {
        // If the cache is full, remove the least recently used entry
        if self.cache.len() >= self.capacity && !self.lru_keys.is_empty() {
            if let Some(lru_key) = self.lru_keys.first() {
                self.cache.remove(lru_key);
                self.lru_keys.remove(0);
            }
        }

        // Insert the new entry
        self.cache.insert(key.clone(), response);
        self.lru_keys.push(key);
    }

    /// Clear cache
    fn clear(&mut self) {
        self.cache.clear();
        self.lru_keys.clear();
    }
}

/// Token calculator
pub struct TokenCalculator {
    // Each model may have its own tokenization
    #[allow(clippy::type_complexity)]
    model_tokenizers: HashMap<String, Box<dyn Fn(&str) -> usize + Send + Sync>>,
}

impl TokenCalculator {
    /// Create a new token calculator
    pub fn new() -> Self {
        let mut calculator = TokenCalculator {
            model_tokenizers: HashMap::new(),
        };

        // Add default approximate tokenizer
        calculator.model_tokenizers.insert(
            "default".to_string(),
            Box::new(|text: &str| {
                // Simple approximation: ~4 characters per token for English
                (text.chars().count() as f32 / 4.0).ceil() as usize
            }),
        );

        calculator
    }

    /// Count tokens for text
    pub fn count_tokens(&self, text: &str, model: &LlmModel, mode: TokenizationMode) -> usize {
        match mode {
            TokenizationMode::Approximate => {
                // Use approximate tokenizer
                if let Some(tokenizer) = self.model_tokenizers.get("default") {
                    tokenizer(text)
                } else {
                    // Fallback: ~4 chars per token
                    (text.chars().count() as f32 / 4.0).ceil() as usize
                }
            }
            TokenizationMode::Precise => {
                // Try model-specific tokenizer
                let model_name = model.name();

                if let Some(tokenizer) = self.model_tokenizers.get(&model_name) {
                    tokenizer(text)
                } else {
                    // Fallback to default tokenizer
                    if let Some(tokenizer) = self.model_tokenizers.get("default") {
                        tokenizer(text)
                    } else {
                        // Last resort
                        (text.chars().count() as f32 / 4.0).ceil() as usize
                    }
                }
            }
        }
    }
}
impl Default for TokenCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter for LLM
pub struct LlmAdapter {
    /// Runtime configuration
    #[allow(dead_code)]
    config: RuntimeConfig,
    /// Response cache
    cache: Arc<RwLock<LlmCache>>,
    /// Token counter
    token_calculator: TokenCalculator,
    /// Clients per provider
    clients: HashMap<LlmProvider, Box<dyn LlmClient + Send + Sync>>,
    /// Usage statistics
    usage_stats: Arc<RwLock<HashMap<String, TokenUsage>>>,
}

impl LlmAdapter {
    /// Creates a new LLM adapter
    pub fn new(config: &RuntimeConfig) -> Self {
        let mut adapter = LlmAdapter {
            config: config.clone(),
            cache: Arc::new(RwLock::new(LlmCache::new(100))), // Cache with 100 entries
            token_calculator: TokenCalculator::new(),
            clients: HashMap::new(),
            usage_stats: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize default simulated client
        adapter.register_client(LlmProvider::Simulated, Box::new(SimulatedLlmClient::new()));

        // Also register simulated client for OpenAI by default
        adapter.register_client(LlmProvider::OpenAI, Box::new(SimulatedLlmClient::new()));

        // If there is an API key in config, replace simulated with real client
        adapter.try_register_openai(&config.llm_api_key);

        adapter
    }

    /// Registers a client for a provider
    pub fn register_client(
        &mut self,
        provider: LlmProvider,
        client: Box<dyn LlmClient + Send + Sync>,
    ) {
        self.clients.insert(provider, client);
    }

    /// Attempts to register a real OpenAI client if an API key is present
    pub fn try_register_openai(&mut self, api_key_opt: &Option<String>) {
        if let Some(key) = api_key_opt {
            // Override any previous (simulated) client with the real one
            self.register_client(
                LlmProvider::OpenAI,
                Box::new(OpenAIClient::new(key.clone())),
            );
        }
    }

    /// Creates a cache hash based on messages and options
    async fn create_cache_key(&self, messages: &[LlmMessage], options: &LlmOptions) -> String {
        // Serialize messages and relevant options
        let mut key = format!("model:{}|", options.model.name());

        if let Some(temp) = options.temperature {
            key.push_str(&format!("temp:{}|", temp));
        }

        // Add contents of the messages
        for msg in messages {
            key.push_str(&format!("{}:{}|", msg.role, msg.content));
        }

        // Generate hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Calls the LLM with messages
    pub async fn call(
        &self,
        messages: Vec<LlmMessage>,
        options: Option<LlmOptions>,
    ) -> Result<LlmResponse> {
        let options = options.unwrap_or_default();
        let start_time = Instant::now();

        // Telemetry span for ask operation
        let model_name = options.model.name();
        let first_user = messages
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");
        let span = trace_ask_operation(first_user, Some(&model_name));
        span.record_event("LLM call started");

        // Clone to avoid moving
        let messages_clone_for_key = messages.clone();

        // If cache is enabled and streaming is not used
        if !options.stream {
            let cache_key = self
                .create_cache_key(&messages_clone_for_key, &options)
                .await;

            // Attempt to get from the cache
            if let Some(cached_response) = {
                let mut cache = self.cache.write().await;
                cache.get(&cache_key)
            } {
                span.record_event("Cache hit");
                return Ok(cached_response);
            }
        }

        // Get the client for the provider
        let provider = options.model.provider();
        let client = self.clients.get(&provider).ok_or_else(|| {
            RuntimeError::LlmError(format!("No client registered for provider: {:?}", provider))
        })?;

        // Count prompt tokens
        let mut token_usage = TokenUsage::default();
        for message in &messages_clone_for_key {
            token_usage.prompt_tokens += self.token_calculator.count_tokens(
                &message.content,
                &options.model,
                TokenizationMode::Approximate,
            );
        }

        // Apply timeout if configured
        let timeout_duration = if let Some(timeout_ms) = options.timeout_ms {
            Duration::from_millis(timeout_ms)
        } else {
            Duration::from_secs(60) // 60 seconds default
        };

        // Call the LLM with a timeout
        let response = tokio::time::timeout(
            timeout_duration,
            client.generate(messages_clone_for_key.clone(), options.clone()),
        )
        .await
        .map_err(|_| {
            RuntimeError::LlmError(format!("LLM call timed out after {:?}", timeout_duration))
        })?
        .map_err(|e| RuntimeError::LlmError(format!("LLM client error: {}", e)))?;
        span.record_event("LLM call completed");

        // Count response tokens
        token_usage.completion_tokens = self.token_calculator.count_tokens(
            &response,
            &options.model,
            TokenizationMode::Approximate,
        );
        token_usage.total_tokens = token_usage.prompt_tokens + token_usage.completion_tokens;

        // Calculate estimated cost
        token_usage.estimated_cost_usd = self.estimate_cost(
            &options.model,
            token_usage.prompt_tokens,
            token_usage.completion_tokens,
        );

        // Build response
        let mut response_messages = messages_clone_for_key;
        response_messages.push(LlmMessage::assistant(&response));

        let llm_response = LlmResponse {
            content: response.clone(),
            messages: response_messages.clone(),
            token_usage,
            model: options.model.clone(),
            execution_time: start_time.elapsed(),
            metadata: HashMap::new(),
        };

        // Store in cache when not streaming
        if !options.stream {
            let cache_key = self
                .create_cache_key(&response_messages[..response_messages.len() - 1], &options)
                .await;
            let mut cache = self.cache.write().await;
            cache.insert(cache_key, llm_response.clone());
        }

        // Update usage statistics
        let model_name = options.model.name();
        let mut usage_stats = self.usage_stats.write().await;
        let model_usage = usage_stats.entry(model_name.clone()).or_default();
        model_usage.prompt_tokens += llm_response.token_usage.prompt_tokens;
        model_usage.completion_tokens += llm_response.token_usage.completion_tokens;
        model_usage.total_tokens += llm_response.token_usage.total_tokens;
        model_usage.estimated_cost_usd += llm_response.token_usage.estimated_cost_usd;

        span.set_attribute("model", &model_name);
        span.set_attribute(
            "tokens.total",
            &llm_response.token_usage.total_tokens.to_string(),
        );
        span.record_event("LLM response returned");
        Ok(llm_response)
    }

    /// Estimates the cost of an LLM call
    fn estimate_cost(
        &self,
        model: &LlmModel,
        prompt_tokens: usize,
        completion_tokens: usize,
    ) -> f64 {
        match model {
            LlmModel::OpenAI(name) => {
                match name.as_str() {
                    "gpt-4" => {
                        (prompt_tokens as f64 * 0.00003) + (completion_tokens as f64 * 0.00006)
                    }
                    "gpt-4-32k" => {
                        (prompt_tokens as f64 * 0.00006) + (completion_tokens as f64 * 0.00012)
                    }
                    "gpt-3.5-turbo" => {
                        (prompt_tokens as f64 * 0.0000015) + (completion_tokens as f64 * 0.000002)
                    }
                    _ => 0.0, // Unknown model
                }
            }
            LlmModel::Anthropic(name) => {
                match name.as_str() {
                    "claude-2" => {
                        (prompt_tokens as f64 * 0.00001102)
                            + (completion_tokens as f64 * 0.00003268)
                    }
                    _ => 0.0, // Unknown model
                }
            }
            LlmModel::Simulated => 0.0, // No cost
            LlmModel::Custom(_) => 0.0, // No default cost
        }
    }

    /// Gets usage statistics
    pub async fn get_usage_stats(&self) -> HashMap<String, TokenUsage> {
        let usage_stats = self.usage_stats.read().await;
        usage_stats.clone()
    }

    /// Clears the cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write().await;
        cache.clear();
        Ok(())
    }
}

/// Interface for LLM clients
#[async_trait]
pub trait LlmClient {
    /// Generates a response from messages
    async fn generate(&self, messages: Vec<LlmMessage>, options: LlmOptions) -> Result<String>;

    /// Generates a streaming response
    async fn generate_stream(
        &self,
        messages: Vec<LlmMessage>,
        options: LlmOptions,
    ) -> Result<tokio::sync::mpsc::Receiver<String>>;
}

/// Simulated LLM client (for tests)
pub struct SimulatedLlmClient {
    /// Artificial delay (ms)
    delay_ms: u64,
}

impl SimulatedLlmClient {
    /// Creates a new simulated client
    pub fn new() -> Self {
        SimulatedLlmClient {
            delay_ms: 500, // 500ms default
        }
    }

    /// Sets the artificial delay
    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }
}

impl Default for SimulatedLlmClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmClient for SimulatedLlmClient {
    async fn generate(&self, messages: Vec<LlmMessage>, options: LlmOptions) -> Result<String> {
        // Simulate delay
        tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;

        // Extract user message
        let user_message = messages
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "No user message".to_string());

        // Generate simulated response
        let response = format!(
            "This is a simulated response to: {}. [Model: {}]",
            user_message,
            options.model.name()
        );

        Ok(response)
    }

    async fn generate_stream(
        &self,
        messages: Vec<LlmMessage>,
        options: LlmOptions,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let (tx, rx) = tokio::sync::mpsc::channel(10);
        let delay_ms = self.delay_ms;

        // Extract user message
        let user_message = messages
            .iter()
            .find(|m| m.role == "user")
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "No user message".to_string());

        // Generate simulated response in chunks
        let response = format!(
            "This is a simulated streaming response to: {}. [Model: {}]",
            user_message,
            options.model.name()
        );

        let chunks: Vec<String> = response.split_whitespace().map(|s| s.to_string()).collect();
        let chunks_len = chunks.len() as u64;

        tokio::spawn(async move {
            for chunk in chunks {
                tokio::time::sleep(Duration::from_millis(delay_ms / chunks_len)).await;
                let _ = tx.send(chunk).await;
            }
        });

        Ok(rx)
    }
}

// -------- OpenAI real client ---------
pub struct OpenAIClient {
    api_key: String,
    http: reqwest::Client,
}

impl OpenAIClient {
    pub fn new(api_key: String) -> Self {
        OpenAIClient {
            api_key,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait::async_trait]
impl LlmClient for OpenAIClient {
    async fn generate(&self, messages: Vec<LlmMessage>, options: LlmOptions) -> Result<String> {
        use serde_json::json;
        // Transform messages
        let payload_msgs: Vec<_> = messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();

        let body = json!({
            "model": options.model.name(),
            "temperature": options.temperature.unwrap_or(0.8),
            "max_tokens": options.max_tokens.unwrap_or(1024),
            "messages": payload_msgs,
        });

        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| RuntimeError::LlmError(format!("OpenAI HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(RuntimeError::LlmError(format!(
                "OpenAI API error {}: {}",
                status, text
            )));
        }

        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| RuntimeError::LlmError(format!("OpenAI JSON error: {}", e)))?;
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        Ok(content)
    }

    async fn generate_stream(
        &self,
        _messages: Vec<LlmMessage>,
        _options: LlmOptions,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        Err(RuntimeError::LlmError(
            "OpenAI streaming not implemented".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simulated_client() {
        let client = SimulatedLlmClient::new().with_delay(100);
        let messages = vec![
            LlmMessage::system("You are a helpful assistant"),
            LlmMessage::user("Hello, world!"),
        ];

        let options = LlmOptions {
            model: LlmModel::Simulated,
            ..Default::default()
        };

        let response = client.generate(messages, options).await.unwrap();
        assert!(response.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_llm_adapter_basic() {
        let config = RuntimeConfig::default();
        let adapter = LlmAdapter::new(&config);

        let messages = vec![
            LlmMessage::system("You are a helpful assistant"),
            LlmMessage::user("Hello, world!"),
        ];

        let options = LlmOptions {
            model: LlmModel::Simulated,
            ..Default::default()
        };

        let response = adapter.call(messages, Some(options)).await.unwrap();
        assert!(response.content.contains("Hello, world!"));
    }

    #[tokio::test]
    async fn test_token_calculation() {
        let calculator = TokenCalculator::new();

        let text = "This is a simple test message with approximately 14 tokens.";
        let model = LlmModel::OpenAI("gpt-3.5-turbo".to_string());

        let tokens = calculator.count_tokens(text, &model, TokenizationMode::Approximate);
        assert!(tokens > 0);
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let config = RuntimeConfig::default();
        let adapter = LlmAdapter::new(&config);

        let messages = vec![
            LlmMessage::system("You are a helpful assistant"),
            LlmMessage::user("This should be cached"),
        ];

        let options = LlmOptions {
            model: LlmModel::Simulated,
            ..Default::default()
        };

        // First call (no cache)
        let start = Instant::now();
        let first_response = adapter
            .call(messages.clone(), Some(options.clone()))
            .await
            .unwrap();
        let first_duration = start.elapsed();

        // Second call (should hit cache)
        let start = Instant::now();
        let second_response = adapter
            .call(messages.clone(), Some(options.clone()))
            .await
            .unwrap();
        let second_duration = start.elapsed();

        // The second call should be faster
        assert!(second_duration < first_duration);

        // Responses should be identical
        assert_eq!(first_response.content, second_response.content);
    }
}
