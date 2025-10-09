// src/telemetry.rs
//
// Simplified telemetry and tracing support for LEXON async operations
// This is a basic implementation without external dependencies for now

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for telemetry setup
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Service name for traces
    pub service_name: String,
    /// Enable console logging
    pub enable_console: bool,
    /// Log level filter
    pub log_level: String,
    /// Custom attributes to add to all spans
    pub global_attributes: HashMap<String, String>,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "lexon-runtime".to_string(),
            enable_console: true,
            log_level: "info".to_string(),
            global_attributes: HashMap::new(),
        }
    }
}

/// Telemetry manager for LEXON runtime
pub struct TelemetryManager {
    config: TelemetryConfig,
    initialized: bool,
}

impl TelemetryManager {
    /// Create a new telemetry manager
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            config,
            initialized: false,
        }
    }

    /// Initialize telemetry (simplified version)
    pub fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.initialized {
            return Ok(());
        }

        if self.config.enable_console {
            println!(
                "✅ Telemetry initialized for service: {}",
                self.config.service_name
            );
        }

        self.initialized = true;
        Ok(())
    }

    /// Shutdown telemetry
    pub fn shutdown(&self) {
        if self.config.enable_console {
            println!("🔄 Shutting down telemetry...");
        }
    }
}

/// Initialize OpenTelemetry tracing if feature `otel` is enabled and env `LEXON_OTEL=1`.
#[cfg(feature = "otel")]
pub fn init_tracing() -> anyhow::Result<()> {
    use opentelemetry_sdk::{trace as sdktrace, Resource};
    use tracing_subscriber::prelude::*;

    if std::env::var("LEXON_OTEL").ok().as_deref() == Some("1") {
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(opentelemetry_otlp::new_exporter().tonic())
            .with_trace_config(sdktrace::config().with_resource(Resource::default()))
            .install_batch(opentelemetry_sdk::runtime::Tokio)?;

        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(tracing_subscriber::fmt::layer())
            .with(otel_layer)
            .try_init()
            .ok();
    }

    Ok(())
}

/// No-op stub when `otel` feature is disabled.
#[cfg(not(feature = "otel"))]
pub fn init_tracing() -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

/// Span builder for creating traced operations
pub struct SpanBuilder {
    name: String,
    attributes: Vec<(&'static str, String)>,
    level: String,
}

impl SpanBuilder {
    /// Create a new span builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attributes: Vec::new(),
            level: "INFO".to_string(),
        }
    }

    /// Add an attribute to the span
    pub fn with_attribute(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.attributes.push((key, value.into()));
        self
    }

    /// Set the span level
    pub fn with_level(mut self, level: impl Into<String>) -> Self {
        self.level = level.into();
        self
    }

    /// Start the span and return a guard
    pub fn start(self) -> TracedSpan {
        TracedSpan {
            name: self.name,
            attributes: self.attributes,
            level: self.level,
            start_time: Instant::now(),
        }
    }
}

/// A traced span with automatic timing
pub struct TracedSpan {
    name: String,
    #[allow(dead_code)]
    attributes: Vec<(&'static str, String)>,
    level: String,
    start_time: Instant,
}

impl TracedSpan {
    /// Record an event in the span
    pub fn record_event(&self, message: &str) {
        println!("[{}] [{}] EVENT: {}", self.level, self.name, message);
    }

    /// Record an error in the span
    pub fn record_error(&self, error: &str) {
        println!("[ERROR] [{}] ERROR: {}", self.name, error);
    }

    /// Record a warning in the span
    pub fn record_warning(&self, warning: &str) {
        println!("[WARN] [{}] WARNING: {}", self.name, warning);
    }

    /// Record debug information
    pub fn record_debug(&self, debug_info: &str) {
        println!("[DEBUG] [{}] DEBUG: {}", self.name, debug_info);
    }

    /// Add an attribute to the span
    pub fn set_attribute(&self, key: &'static str, value: &str) {
        println!("[{}] [{}] ATTR: {}={}", self.level, self.name, key, value);
    }

    /// Get the elapsed time since span creation
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Drop for TracedSpan {
    fn drop(&mut self) {
        let elapsed = self.elapsed();
        println!(
            "[{}] [{}] COMPLETED in {:?}",
            self.level, self.name, elapsed
        );
    }
}

/// Convenience macros for creating traced spans
#[macro_export]
macro_rules! trace_span {
    ($name:expr) => {
        $crate::telemetry::SpanBuilder::new($name).start()
    };
    ($name:expr, $($key:expr => $value:expr),*) => {
        {
            let mut builder = $crate::telemetry::SpanBuilder::new($name);
            $(
                builder = builder.with_attribute($key, $value);
            )*
            builder.start()
        }
    };
}

/// Convenience macros for async function tracing
#[macro_export]
macro_rules! trace_async_fn {
    ($fn_name:expr, $future:expr) => {{
        let span = $crate::trace_span!($fn_name);
        span.record_event(&format!("Starting {}", $fn_name));

        let result = $future.await;

        match &result {
            Ok(_) => span.record_event(&format!("{} completed successfully", $fn_name)),
            Err(e) => span.record_error(&format!("{} failed: {:?}", $fn_name, e)),
        }

        result
    }};
}

/// Trace an ask operation
pub fn trace_ask_operation(prompt: &str, model: Option<&str>) -> TracedSpan {
    SpanBuilder::new("ask_operation")
        .with_attribute("operation.type", "ask")
        .with_attribute("prompt.length", prompt.len().to_string())
        .with_attribute("model", model.unwrap_or("default"))
        .with_level("INFO")
        .start()
}

/// Trace an ask_safe operation
pub fn trace_ask_safe_operation(prompt: &str, validation_strategy: Option<&str>) -> TracedSpan {
    SpanBuilder::new("ask_safe_operation")
        .with_attribute("operation.type", "ask_safe")
        .with_attribute("prompt.length", prompt.len().to_string())
        .with_attribute("validation.strategy", validation_strategy.unwrap_or("none"))
        .with_level("INFO")
        .start()
}

/// Trace a data operation
pub fn trace_data_operation(operation: &str, data_size: usize) -> TracedSpan {
    SpanBuilder::new("data_operation")
        .with_attribute("operation.type", operation)
        .with_attribute("data.size", data_size.to_string())
        .with_level("DEBUG")
        .start()
}

/// Trace a memory operation
pub fn trace_memory_operation(operation: &str, scope: &str) -> TracedSpan {
    SpanBuilder::new("memory_operation")
        .with_attribute("operation.type", operation)
        .with_attribute("memory.scope", scope)
        .with_level("DEBUG")
        .start()
}

/// Trace task scheduling
pub fn trace_task_scheduling(task_id: &str, priority: &str) -> TracedSpan {
    SpanBuilder::new("task_scheduling")
        .with_attribute("task.id", task_id)
        .with_attribute("task.priority", priority)
        .with_level("INFO")
        .start()
}

/// Trace task execution
pub fn trace_task_execution(task_id: &str) -> TracedSpan {
    SpanBuilder::new("task_execution")
        .with_attribute("task.id", task_id)
        .with_level("INFO")
        .start()
}

/// Trace task cancellation
pub fn trace_task_cancellation(task_id: &str, reason: &str) -> TracedSpan {
    SpanBuilder::new("task_cancellation")
        .with_attribute("task.id", task_id)
        .with_attribute("cancellation.reason", reason)
        .with_level("WARN")
        .start()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_telemetry_initialization() {
        let config = TelemetryConfig::default();
        let mut manager = TelemetryManager::new(config);

        assert!(manager.initialize().is_ok());

        manager.shutdown();
    }

    #[test]
    fn test_span_creation() {
        let config = TelemetryConfig::default();
        let mut manager = TelemetryManager::new(config);
        manager.initialize().unwrap();

        let span = SpanBuilder::new("test_operation")
            .with_attribute("test.key", "test_value")
            .with_level("INFO")
            .start();

        span.record_event("Test event");
        span.record_debug("Debug information");

        assert!(span.elapsed() < Duration::from_secs(1));

        manager.shutdown();
    }

    #[test]
    fn test_ask_operation_tracing() {
        let config = TelemetryConfig::default();
        let mut manager = TelemetryManager::new(config);
        manager.initialize().unwrap();

        let span = trace_ask_operation("What is 2+2?", Some("gpt-4"));
        span.record_event("Processing ask request");
        span.record_event("Ask request completed");

        manager.shutdown();
    }

    #[test]
    fn test_task_tracing() {
        let config = TelemetryConfig::default();
        let mut manager = TelemetryManager::new(config);
        manager.initialize().unwrap();

        let scheduling_span = trace_task_scheduling("task-123", "high");
        scheduling_span.record_event("Task scheduled");

        let execution_span = trace_task_execution("task-123");
        execution_span.record_event("Task started");
        execution_span.record_event("Task completed");

        manager.shutdown();
    }

    #[test]
    fn test_macro_tracing() {
        let config = TelemetryConfig::default();
        let mut manager = TelemetryManager::new(config);
        manager.initialize().unwrap();

        let span = trace_span!("test_macro", "key1" => "value1", "key2" => "value2");
        span.record_event("Macro test");

        manager.shutdown();
    }
}
