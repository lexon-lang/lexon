// lexc/src/executor/api_config.rs
//
// Centralized configuration for external APIs
// Removes hardcoded URLs and enables flexible configuration

use std::collections::HashMap;
use toml::Value;

/// Centralized API endpoints configuration
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Base URLs per provider
    #[allow(dead_code)]
    pub endpoints: HashMap<String, String>,
    /// Provider-specific configurations
    pub provider_configs: HashMap<String, ProviderConfig>,
    /// Configuration loaded from TOML
    pub toml_config: Option<Value>,
    /// Default provider
    pub default_provider: String,
    /// Default timeout in seconds
    pub default_timeout: u64,
    /// Default retries
    pub default_retries: u32,
}

/// Provider-specific configuration
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// Provider base URL
    pub base_url: String,
    /// Additional required headers
    pub headers: HashMap<String, String>,
    /// Provider-specific timeout
    #[allow(dead_code)]
    pub timeout: Option<u64>,
    /// Authentication format
    pub auth_format: AuthFormat,
    /// Specific endpoints
    pub endpoints: HashMap<String, String>,
    /// Provider default model
    pub default_model: Option<String>,
    /// Supported models
    pub models: HashMap<String, ModelConfig>,
}

/// Model-specific configuration
#[derive(Debug, Clone)]
pub struct ModelConfig {
    #[allow(dead_code)]
    pub max_tokens: Option<u32>,
    #[allow(dead_code)]
    pub cost_per_1k: Option<f64>,
}

/// Supported authentication formats
#[derive(Debug, Clone)]
pub enum AuthFormat {
    Bearer,
    ApiKey(String),         // Header name
    Custom(String, String), // Header name, prefix
}

impl Default for ApiConfig {
    fn default() -> Self {
        let mut config = ApiConfig {
            endpoints: HashMap::new(),
            provider_configs: HashMap::new(),
            toml_config: None,
            default_provider: String::new(),
            default_timeout: 30,
            default_retries: 3,
        };

        // Configure default providers
        config.setup_default_providers();
        config
    }
}

impl ApiConfig {
    /// Creates a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure default providers
    fn setup_default_providers(&mut self) {
        // OpenAI
        let mut openai_headers = HashMap::new();
        openai_headers.insert("Content-Type".to_string(), "application/json".to_string());

        let mut openai_endpoints = HashMap::new();
        openai_endpoints.insert("chat".to_string(), "chat/completions".to_string());
        openai_endpoints.insert("completions".to_string(), "completions".to_string());

        self.provider_configs.insert(
            "openai".to_string(),
            ProviderConfig {
                base_url: std::env::var("OPENAI_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
                headers: openai_headers,
                timeout: Some(60),
                auth_format: AuthFormat::Bearer,
                endpoints: openai_endpoints,
                default_model: None,
                models: HashMap::new(),
            },
        );

        // Anthropic
        let mut anthropic_headers = HashMap::new();
        anthropic_headers.insert("Content-Type".to_string(), "application/json".to_string());
        anthropic_headers.insert("anthropic-version".to_string(), "2023-06-01".to_string());

        let mut anthropic_endpoints = HashMap::new();
        anthropic_endpoints.insert("messages".to_string(), "messages".to_string());

        self.provider_configs.insert(
            "anthropic".to_string(),
            ProviderConfig {
                base_url: std::env::var("ANTHROPIC_BASE_URL")
                    .unwrap_or_else(|_| "https://api.anthropic.com/v1".to_string()),
                headers: anthropic_headers,
                timeout: Some(60),
                auth_format: AuthFormat::ApiKey("x-api-key".to_string()),
                endpoints: anthropic_endpoints,
                default_model: None,
                models: HashMap::new(),
            },
        );

        // Google
        let mut google_headers = HashMap::new();
        google_headers.insert("Content-Type".to_string(), "application/json".to_string());

        let mut google_endpoints = HashMap::new();
        google_endpoints.insert(
            "generate".to_string(),
            "models/{model}:generateContent".to_string(),
        );

        self.provider_configs.insert(
            "google".to_string(),
            ProviderConfig {
                base_url: std::env::var("GOOGLE_BASE_URL").unwrap_or_else(|_| {
                    "https://generativelanguage.googleapis.com/v1beta".to_string()
                }),
                headers: google_headers,
                timeout: Some(45),
                auth_format: AuthFormat::Custom("key".to_string(), "".to_string()), // Query parameter
                endpoints: google_endpoints,
                default_model: None,
                models: HashMap::new(),
            },
        );
    }

    /// Gets the full URL for a specific endpoint
    pub fn get_full_url(
        &self,
        provider: &str,
        endpoint: &str,
        model: Option<&str>,
    ) -> Option<String> {
        if let Some(provider_config) = self.provider_configs.get(provider) {
            if let Some(endpoint_path) = provider_config.endpoints.get(endpoint) {
                let mut url = format!("{}/{}", provider_config.base_url, endpoint_path);

                // Replace placeholders like {model}
                if let Some(model_name) = model {
                    url = url.replace("{model}", model_name);
                }

                return Some(url);
            }
        }
        None
    }

    /// Gets configuration for a provider
    #[allow(dead_code)]
    pub fn get_provider_config(&self, provider: &str) -> Option<&ProviderConfig> {
        self.provider_configs.get(provider)
    }

    /// Updates configuration for a provider
    #[allow(dead_code)]
    pub fn update_provider(&mut self, provider: String, config: ProviderConfig) {
        self.provider_configs.insert(provider, config);
    }

    /// Validates that a provider is properly configured
    #[allow(dead_code)]
    pub fn validate_provider(&self, provider: &str) -> Result<(), String> {
        if let Some(config) = self.provider_configs.get(provider) {
            // Verify base URL is valid
            if config.base_url.is_empty() {
                return Err(format!("Base URL is empty for provider: {}", provider));
            }

            // Verify it has at least one endpoint
            if config.endpoints.is_empty() {
                return Err(format!(
                    "No endpoints configured for provider: {}",
                    provider
                ));
            }

            Ok(())
        } else {
            Err(format!("Provider not configured: {}", provider))
        }
    }

    /// Lists all configured providers
    #[allow(dead_code)]
    pub fn list_providers(&self) -> Vec<String> {
        self.provider_configs.keys().cloned().collect()
    }

    /// Loads configuration from environment variables
    pub fn load_from_env(&mut self) {
        // Load custom configurations from environment variables
        for (provider, config) in self.provider_configs.iter_mut() {
            let env_var = format!("{}_BASE_URL", provider.to_uppercase());
            if let Ok(custom_url) = std::env::var(&env_var) {
                config.base_url = custom_url;
                println!("🔧 Loaded custom URL for {}: {}", provider, config.base_url);
            }
        }

        // Try loading configuration from lexon.toml
        self.load_from_toml();
    }

    /// Loads configuration from TOML file
    pub fn load_from_toml(&mut self) {
        // Try loading lexon.toml from the current directory
        if let Ok(toml_content) = std::fs::read_to_string("lexon.toml") {
            if let Ok(parsed_toml) = toml_content.parse::<Value>() {
                self.toml_config = Some(parsed_toml.clone());
                self.apply_toml_config(&parsed_toml);
            }
        }
    }

    /// Applies configuration from TOML
    fn apply_toml_config(&mut self, toml_config: &Value) {
        // Get system configuration
        if let Some(system) = toml_config.get("system") {
            if let Some(default_provider) = system.get("default_provider").and_then(|v| v.as_str())
            {
                self.default_provider = default_provider.to_string();
            }
        }

        // Process providers
        if let Some(providers) = toml_config.get("providers").and_then(|v| v.as_table()) {
            for (provider_name, provider_config) in providers {
                if let Some(config_table) = provider_config.as_table() {
                    let mut provider_cfg = ProviderConfig {
                        base_url: config_table
                            .get("base_url")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string(),
                        headers: HashMap::new(),
                        timeout: config_table
                            .get("timeout")
                            .and_then(|v| v.as_integer())
                            .map(|i| i as u64),
                        auth_format: AuthFormat::Bearer, // Default
                        endpoints: HashMap::new(),
                        default_model: config_table
                            .get("default_model")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string()),
                        models: HashMap::new(),
                    };

                    // Process headers
                    if let Some(headers) = config_table.get("headers").and_then(|v| v.as_table()) {
                        for (key, value) in headers {
                            if let Some(val_str) = value.as_str() {
                                provider_cfg
                                    .headers
                                    .insert(key.clone(), val_str.to_string());
                            }
                        }
                    }

                    // Process endpoints
                    if let Some(endpoints) =
                        config_table.get("endpoints").and_then(|v| v.as_table())
                    {
                        for (key, value) in endpoints {
                            if let Some(val_str) = value.as_str() {
                                provider_cfg
                                    .endpoints
                                    .insert(key.clone(), val_str.to_string());
                            }
                        }
                    }

                    // Process models
                    if let Some(models) = config_table.get("models").and_then(|v| v.as_table()) {
                        for (model_name, model_config) in models {
                            if let Some(model_table) = model_config.as_table() {
                                let model_cfg = ModelConfig {
                                    max_tokens: model_table
                                        .get("max_tokens")
                                        .and_then(|v| v.as_integer())
                                        .map(|i| i as u32),
                                    cost_per_1k: model_table
                                        .get("cost_per_1k")
                                        .and_then(|v| v.as_float()),
                                };
                                provider_cfg.models.insert(model_name.clone(), model_cfg);
                            }
                        }
                    }

                    // Configure authentication format
                    if let Some(auth_header) =
                        config_table.get("auth_header").and_then(|v| v.as_str())
                    {
                        if auth_header == "Authorization" {
                            provider_cfg.auth_format = AuthFormat::Bearer;
                        } else {
                            provider_cfg.auth_format = AuthFormat::ApiKey(auth_header.to_string());
                        }
                    }

                    self.provider_configs
                        .insert(provider_name.clone(), provider_cfg);
                }
            }
        }
    }

    /// Gets the default provider
    pub fn get_default_provider(&self) -> String {
        if !self.default_provider.is_empty() {
            self.default_provider.clone()
        } else {
            "openai".to_string() // Fallback
        }
    }

    /// Gets the default model for a provider
    pub fn get_default_model(&self, provider: &str) -> Option<String> {
        self.provider_configs
            .get(provider)
            .and_then(|config| config.default_model.clone())
    }

    /// Verifies if a model is supported by a provider
    pub fn is_model_supported(&self, provider: &str, model: &str) -> bool {
        if let Some(provider_config) = self.provider_configs.get(provider) {
            provider_config.models.contains_key(model)
        } else {
            false
        }
    }

    /// Generates authentication headers for a provider
    pub fn get_auth_headers(&self, provider: &str, api_key: &str) -> HashMap<String, String> {
        let mut headers = HashMap::new();

        if let Some(config) = self.provider_configs.get(provider) {
            // Add base headers
            for (key, value) in &config.headers {
                headers.insert(key.clone(), value.clone());
            }

            // Add authentication
            match &config.auth_format {
                AuthFormat::Bearer => {
                    headers.insert("Authorization".to_string(), format!("Bearer {}", api_key));
                }
                AuthFormat::ApiKey(header_name) => {
                    headers.insert(header_name.clone(), api_key.to_string());
                }
                AuthFormat::Custom(header_name, prefix) => {
                    let auth_value = if prefix.is_empty() {
                        api_key.to_string()
                    } else {
                        format!("{} {}", prefix, api_key)
                    };
                    headers.insert(header_name.clone(), auth_value);
                }
            }
        }

        headers
    }
}

/// Builder to create custom API configurations
pub struct ApiConfigBuilder {
    config: ApiConfig,
}

#[allow(dead_code)]
impl ApiConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: ApiConfig::new(),
        }
    }

    pub fn add_provider(mut self, name: String, provider_config: ProviderConfig) -> Self {
        self.config.provider_configs.insert(name, provider_config);
        self
    }

    pub fn set_default_timeout(mut self, timeout: u64) -> Self {
        self.config.default_timeout = timeout;
        self
    }

    pub fn set_default_retries(mut self, retries: u32) -> Self {
        self.config.default_retries = retries;
        self
    }

    pub fn build(self) -> ApiConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ApiConfig::new();
        assert!(config.provider_configs.contains_key("openai"));
        assert!(config.provider_configs.contains_key("anthropic"));
        assert!(config.provider_configs.contains_key("google"));
    }

    #[test]
    fn test_get_full_url() {
        let config = ApiConfig::new();
        let url = config.get_full_url("openai", "chat", None);
        assert!(url.is_some());
        assert!(url.unwrap().contains("chat/completions"));
    }

    #[test]
    fn test_validate_provider() {
        let config = ApiConfig::new();
        assert!(config.validate_provider("openai").is_ok());
        assert!(config.validate_provider("nonexistent").is_err());
    }
}
