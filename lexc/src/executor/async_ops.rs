use super::llm_adapter::LlmAdapter;
use super::memory::MemoryManager;
use super::{ExecutorError, Result, RuntimeValue};
use polars::prelude::*;
use std::collections::HashMap;
use tokio::fs;
use tokio::process::Command;

/// Async I/O operations for LEXON
pub struct AsyncOps {
    pub llm_adapter: LlmAdapter,
    pub memory_manager: MemoryManager,
}

impl AsyncOps {
    pub fn new() -> Self {
        Self {
            llm_adapter: LlmAdapter::new(),
            memory_manager: MemoryManager::new(),
        }
    }

    /// Async version of read_file
    pub async fn read_file_async(&self, path: &str) -> Result<String> {
        let content = fs::read_to_string(path)
            .await
            .map_err(|e| ExecutorError::IoError(format!("Error reading file {}: {}", path, e)))?;
        Ok(content)
    }

    /// Async version of write_file
    pub async fn write_file_async(&self, path: &str, content: &str) -> Result<()> {
        fs::write(path, content)
            .await
            .map_err(|e| ExecutorError::IoError(format!("Error writing file {}: {}", path, e)))?;
        Ok(())
    }

    /// Async version of execute
    pub async fn execute_async(&self, command: &str) -> Result<String> {
        let output = if cfg!(target_os = "windows") {
            Command::new("cmd").args(["/C", command]).output().await
        } else {
            Command::new("sh").arg("-c").arg(command).output().await
        };

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    Ok(stdout.to_string())
                } else {
                    Err(ExecutorError::CommandError(format!(
                        "Command failed with exit code {:?}: {}",
                        output.status.code(),
                        stderr
                    )))
                }
            }
            Err(e) => Err(ExecutorError::CommandError(format!(
                "Failed to execute command: {}",
                e
            ))),
        }
    }

    /// Async version of load_csv
    pub async fn load_csv_async(&self, path: &str) -> Result<String> {
        // Read the CSV file using async fs first
        let _content = self.read_file_async(path).await?;

        // Use polars to process the CSV
        let df = polars::prelude::CsvReader::from_path(path)
            .map_err(|e| ExecutorError::CsvError(format!("Error opening CSV {}: {}", path, e)))?
            .finish()
            .map_err(|e| ExecutorError::CsvError(format!("Error reading CSV {}: {}", path, e)))?;

        // Convert to JSON for compatibility
        let mut buf = Vec::new();
        JsonWriter::new(&mut buf)
            .finish(&mut df.clone())
            .map_err(|e| ExecutorError::CsvError(format!("Error converting CSV to JSON: {}", e)))?;

        let json_string = String::from_utf8(buf).map_err(|e| {
            ExecutorError::CsvError(format!("Error converting bytes to string: {}", e))
        })?;

        Ok(json_string)
    }

    /// Async version of save_json
    pub async fn save_json_async(&self, path: &str, data: &str) -> Result<()> {
        // Verify content is valid JSON
        let _: serde_json::Value = serde_json::from_str(data)
            .map_err(|e| ExecutorError::JsonError(format!("Invalid JSON data: {}", e)))?;

        self.write_file_async(path, data).await
    }

    /// Async version of ask
    pub async fn ask_async(
        &mut self,
        prompt: &str,
        model: Option<&str>,
        attributes: &HashMap<String, String>,
    ) -> Result<String> {
        self.llm_adapter
            .call_llm_async(
                model,
                None, // temperature
                None, // system_prompt
                Some(prompt),
                None, // schema
                None, // max_tokens
                attributes,
            )
            .await
    }

    /// Async version of ask_safe
    pub async fn ask_safe_async(
        &mut self,
        prompt: &str,
        model: Option<&str>,
        config: Option<super::llm_adapter::AntiHallucinationConfig>,
    ) -> Result<super::llm_adapter::ValidationResult> {
        self.llm_adapter.ask_safe(prompt, model, config).await
    }

    /// Async version of memory_store
    pub async fn memory_store_async(&mut self, key: &str, value: &str) -> Result<()> {
        // Use sync method for now since MemoryManager has no async API
        self.memory_manager
            .store_memory(
                "default",
                RuntimeValue::String(value.to_string()),
                Some(key),
            )
            .map_err(|e| ExecutorError::MemoryError(format!("Memory store error: {}", e)))
    }

    /// Async version of memory_load
    pub async fn memory_load_async(&self, key: &str) -> Result<String> {
        // Use sync method for now since MemoryManager has no async API
        match self
            .memory_manager
            .load_value_by_key("default", key)
            .map_err(|e| ExecutorError::MemoryError(format!("Memory load error: {}", e)))?
        {
            Some(RuntimeValue::String(value)) => Ok(value),
            Some(_) => Err(ExecutorError::MemoryError(
                "Value is not a string".to_string(),
            )),
            None => Err(ExecutorError::MemoryError("Key not found".to_string())),
        }
    }

    /// Async version of ask_parallel
    pub async fn ask_parallel_async(
        &mut self,
        prompts: Vec<String>,
        model: Option<String>,
        max_concurrent: Option<usize>,
    ) -> Result<Vec<String>> {
        self.llm_adapter
            .batch_call_llm(
                prompts.into_iter().map(|p| (p, model.clone())).collect(),
                max_concurrent,
            )
            .await
    }

    /// Async version of ask_ensemble
    pub async fn ask_ensemble_async(
        &mut self,
        prompts: Vec<String>,
        strategy: super::llm_adapter::EnsembleStrategy,
        model: Option<String>,
    ) -> Result<String> {
        self.llm_adapter
            .ask_ensemble(prompts, strategy, model)
            .await
    }
}
impl Default for AsyncOps {
    fn default() -> Self {
        Self::new()
    }
}

/// Wrapper for compatibility with current executor
pub struct AsyncExecutor {
    pub ops: AsyncOps,
}

impl AsyncExecutor {
    pub fn new() -> Self {
        Self {
            ops: AsyncOps::new(),
        }
    }

    /// Executes an async operation from sync code
    pub fn block_on_async<F, T>(&mut self, future: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        // Use the current tokio runtime or create a new one
        let rt = tokio::runtime::Runtime::new().map_err(|e| {
            ExecutorError::AsyncError(format!("Failed to create async runtime: {}", e))
        })?;

        rt.block_on(future)
    }
}
impl Default for AsyncExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_read_write_file_async() {
        let mut temp_file = NamedTempFile::new().unwrap();
        let test_content = "Hello, async world!";

        // Write test content
        temp_file.write_all(test_content.as_bytes()).unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        let ops = AsyncOps::new();

        // Read file
        let content = ops.read_file_async(temp_path).await.unwrap();
        assert_eq!(content, test_content);

        // Write new content
        let new_content = "Hello, async write!";
        ops.write_file_async(temp_path, new_content).await.unwrap();

        // Verify it was written correctly
        let updated_content = ops.read_file_async(temp_path).await.unwrap();
        assert_eq!(updated_content, new_content);
    }

    #[tokio::test]
    async fn test_execute_async() {
        let ops = AsyncOps::new();

        // Execute simple command
        let result = ops
            .execute_async("echo 'Hello from async command'")
            .await
            .unwrap();
        assert!(result.contains("Hello from async command"));
    }

    #[tokio::test]
    async fn test_save_load_json_async() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        let ops = AsyncOps::new();
        let test_json = r#"{"message": "Hello, async JSON!"}"#;

        // Save JSON
        ops.save_json_async(temp_path, test_json).await.unwrap();

        // Load JSON
        let loaded_content = ops.read_file_async(temp_path).await.unwrap();
        assert_eq!(loaded_content, test_json);
    }
}
