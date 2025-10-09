// src/runtime/memory.rs
//
// Contextual memory manager for Lexon
// Allows storing and retrieving information across different scopes and strategies

use super::{Result, RuntimeConfig, RuntimeError, RuntimeValue};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
// use std::sync::Arc;
// use std::time::Duration;

// Prefer derive Default with explicit default variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MemoryStrategy {
    /// Simple FIFO buffer
    #[default]
    Buffer,
    /// Vector storage with semantic search
    Vector,
    /// Recursive summaries for large contexts
    Summary,
    /// Hybrid strategy
    Hybrid,
}

/// Memory entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Stored value
    pub value: RuntimeValue,
    /// Optional key (for direct access)
    pub key: Option<String>,
    /// Creation timestamp (milliseconds since epoch)
    pub created_at: u64, // timestamp in milliseconds
    /// Time to live (milliseconds from created_at)
    pub ttl: Option<u64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MemoryEntry {
    /// Creates a new memory entry
    pub fn new(value: RuntimeValue, key: Option<String>, ttl: Option<u64>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        MemoryEntry {
            value,
            key,
            created_at: now,
            ttl,
            metadata: HashMap::new(),
        }
    }

    /// Checks whether the entry has expired
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            (now - self.created_at) > ttl
        } else {
            false
        }
    }

    /// Adds a metadata pair to the entry
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// Memory scope
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryScope {
    /// Scope name
    pub name: String,
    /// Memory strategy
    pub strategy: MemoryStrategy,
    /// Maximum capacity (entries)
    pub capacity: usize,
    /// FIFO buffer for sequential memory
    pub buffer: VecDeque<MemoryEntry>,
    /// Key-value store for direct access
    pub key_value_store: HashMap<String, MemoryEntry>,
    /// Embeddings for vector search (serialized as HashMap<String, Vec<f32>>)
    #[serde(skip)]
    pub vector_embeddings: Option<HashMap<usize, Vec<f32>>>,
    /// Scope summary (for Summary strategy)
    pub summary: Option<String>,
    /// Last modification timestamp
    pub last_modified: u64,
}

impl MemoryScope {
    /// Creates a new memory scope
    pub fn new(name: &str, strategy: MemoryStrategy, capacity: Option<usize>) -> Self {
        let capacity = capacity.unwrap_or(100); // Default value
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        MemoryScope {
            name: name.to_string(),
            strategy,
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            key_value_store: HashMap::new(),
            vector_embeddings: None,
            summary: None,
            last_modified: now,
        }
    }

    /// Stores a value in the scope
    pub fn store(
        &mut self,
        value: RuntimeValue,
        key: Option<&str>,
        ttl: Option<u64>,
    ) -> Result<()> {
        let entry = MemoryEntry::new(value, key.map(|k| k.to_string()), ttl);

        // Update timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_modified = now;

        match self.strategy {
            MemoryStrategy::Buffer => {
                // In buffer mode, push to the end and drop from the front if exceeding capacity
                self.buffer.push_back(entry.clone());

                while self.buffer.len() > self.capacity {
                    self.buffer.pop_front();
                }

                // If it has a key, also store it in the map
                if let Some(key) = &entry.key {
                    self.key_value_store.insert(key.clone(), entry);
                }

                Ok(())
            }
            MemoryStrategy::Vector => {
                // TODO: Implement vector storage (embeddings)
                // For now, behave like Buffer
                self.buffer.push_back(entry.clone());

                while self.buffer.len() > self.capacity {
                    self.buffer.pop_front();
                }

                if let Some(key) = &entry.key {
                    self.key_value_store.insert(key.clone(), entry);
                }

                Ok(())
            }
            MemoryStrategy::Summary => {
                // TODO: Implement incremental summary
                // For now, behave like Buffer
                self.buffer.push_back(entry.clone());

                while self.buffer.len() > self.capacity {
                    self.buffer.pop_front();
                }

                if let Some(key) = &entry.key {
                    self.key_value_store.insert(key.clone(), entry);
                }

                Ok(())
            }
            MemoryStrategy::Hybrid => {
                // Combines Buffer + Vector
                self.buffer.push_back(entry.clone());

                while self.buffer.len() > self.capacity {
                    self.buffer.pop_front();
                }

                if let Some(key) = &entry.key {
                    self.key_value_store.insert(key.clone(), entry);
                }

                Ok(())
            }
        }
    }

    /// Retrieves values from the scope
    pub fn retrieve(&self, key: Option<&str>, limit: Option<usize>) -> Result<Vec<RuntimeValue>> {
        // Filter expired entries (conceptual, we don't mutate self here)
        let is_valid = |entry: &MemoryEntry| !entry.is_expired();

        // If a key is specified, look it up in the key-value store
        if let Some(key) = key {
            if let Some(entry) = self.key_value_store.get(key) {
                if is_valid(entry) {
                    return Ok(vec![entry.value.clone()]);
                } else {
                    return Ok(vec![]);
                }
            }
            return Err(RuntimeError::MemoryError(format!(
                "Key '{}' not found in scope '{}'",
                key, self.name
            )));
        }

        // Without a key, behavior depends on the strategy
        match self.strategy {
            MemoryStrategy::Buffer => {
                // Return the most recent entries (up to the limit)
                let limit = limit.unwrap_or(self.capacity);
                let valid_entries: Vec<_> = self
                    .buffer
                    .iter()
                    .filter(|e| is_valid(e))
                    .map(|e| e.value.clone())
                    .collect();

                // Take the last 'limit' entries
                let start = if valid_entries.len() > limit {
                    valid_entries.len() - limit
                } else {
                    0
                };

                Ok(valid_entries[start..].to_vec())
            }
            MemoryStrategy::Vector => {
                // TODO: Implement vector retrieval with embeddings
                // For now, same as Buffer
                let limit = limit.unwrap_or(self.capacity);
                let valid_entries: Vec<_> = self
                    .buffer
                    .iter()
                    .filter(|e| is_valid(e))
                    .map(|e| e.value.clone())
                    .collect();

                let start = if valid_entries.len() > limit {
                    valid_entries.len() - limit
                } else {
                    0
                };

                Ok(valid_entries[start..].to_vec())
            }
            MemoryStrategy::Summary => {
                // If there is a summary, return it as the first entry
                if let Some(summary) = &self.summary {
                    return Ok(vec![RuntimeValue::String(summary.clone())]);
                }

                // Without a summary, same behavior as Buffer
                let limit = limit.unwrap_or(self.capacity);
                let valid_entries: Vec<_> = self
                    .buffer
                    .iter()
                    .filter(|e| is_valid(e))
                    .map(|e| e.value.clone())
                    .collect();

                let start = if valid_entries.len() > limit {
                    valid_entries.len() - limit
                } else {
                    0
                };

                Ok(valid_entries[start..].to_vec())
            }
            MemoryStrategy::Hybrid => {
                // TODO: Implement hybrid strategy
                // For now, same as Buffer
                let limit = limit.unwrap_or(self.capacity);
                let valid_entries: Vec<_> = self
                    .buffer
                    .iter()
                    .filter(|e| is_valid(e))
                    .map(|e| e.value.clone())
                    .collect();

                let start = if valid_entries.len() > limit {
                    valid_entries.len() - limit
                } else {
                    0
                };

                Ok(valid_entries[start..].to_vec())
            }
        }
    }

    /// Cleans up expired entries
    pub fn clean_expired(&mut self) {
        // Update timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_modified = now;

        // Clean buffer
        self.buffer.retain(|e| !e.is_expired());

        // Clean key-value store
        let expired_keys: Vec<_> = self
            .key_value_store
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();

        for key in expired_keys {
            self.key_value_store.remove(&key);
        }
    }

    /// Removes all entries from the scope
    pub fn clear(&mut self) {
        // Update timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_modified = now;

        self.buffer.clear();
        self.key_value_store.clear();
        self.vector_embeddings = None;
        self.summary = None;
    }
}

/// Contextual memory manager
pub struct MemoryManager {
    /// Runtime configuration
    #[allow(dead_code)]
    config: RuntimeConfig,
    /// Available memory scopes
    scopes: HashMap<String, MemoryScope>,
    /// Persistence path
    persistence_path: Option<String>,
}

impl MemoryManager {
    /// Creates a new memory manager
    pub fn new(config: &RuntimeConfig) -> Self {
        let mut manager = MemoryManager {
            config: config.clone(),
            scopes: HashMap::new(),
            persistence_path: config.memory_path.clone(),
        };

        // Load from persistence if configured
        if let Some(path) = &manager.persistence_path {
            if Path::new(path).exists() {
                match fs::read_to_string(path) {
                    Ok(contents) => {
                        match serde_json::from_str::<HashMap<String, MemoryScope>>(&contents) {
                            Ok(loaded_scopes) => manager.scopes = loaded_scopes,
                            Err(e) => eprintln!("[memory] Failed to parse persistence file: {}", e),
                        }
                    }
                    Err(e) => eprintln!("[memory] Failed to read persistence file: {}", e),
                }
            }
        }

        manager
    }

    /// Creates a new memory scope
    pub fn create_scope(
        &mut self,
        name: &str,
        strategy: MemoryStrategy,
        capacity: Option<usize>,
    ) -> Result<()> {
        if self.scopes.contains_key(name) {
            return Err(RuntimeError::MemoryError(format!(
                "Memory scope '{}' already exists",
                name
            )));
        }

        let scope = MemoryScope::new(name, strategy, capacity);
        self.scopes.insert(name.to_string(), scope);

        // Persist changes if configured
        self.save_to_disk()?;

        Ok(())
    }

    /// Gets an existing scope or creates a new one
    pub fn get_or_create_scope(
        &mut self,
        name: &str,
        strategy: Option<MemoryStrategy>,
        capacity: Option<usize>,
    ) -> Result<&mut MemoryScope> {
        if !self.scopes.contains_key(name) {
            let strategy = strategy.unwrap_or_default();
            self.create_scope(name, strategy, capacity)?;
        }

        Ok(self.scopes.get_mut(name).unwrap())
    }

    /// Stores a value in a scope
    pub fn store(
        &mut self,
        scope: &str,
        value: RuntimeValue,
        key: Option<&str>,
        ttl: Option<u64>,
    ) -> Result<()> {
        let scope = self.get_or_create_scope(scope, None, None)?;
        let result = scope.store(value, key, ttl);

        // Persist changes if configured
        self.save_to_disk()?;

        result
    }

    /// Retrieves values from a scope
    pub fn retrieve(
        &self,
        scope: &str,
        key: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<RuntimeValue>> {
        let scope = self.scopes.get(scope).ok_or_else(|| {
            RuntimeError::MemoryError(format!("Memory scope '{}' not found", scope))
        })?;

        scope.retrieve(key, limit)
    }

    /// Removes a memory scope
    pub fn remove_scope(&mut self, name: &str) -> Result<()> {
        if !self.scopes.contains_key(name) {
            return Err(RuntimeError::MemoryError(format!(
                "Memory scope '{}' not found",
                name
            )));
        }

        self.scopes.remove(name);

        // Persist changes if configured
        self.save_to_disk()?;

        Ok(())
    }

    /// Cleans expired entries from all scopes
    pub fn clean_expired(&mut self) -> Result<()> {
        for scope in self.scopes.values_mut() {
            scope.clean_expired();
        }

        // Persist changes if configured
        self.save_to_disk()?;

        Ok(())
    }

    /// Saves state to disk (if configured)
    fn save_to_disk(&self) -> Result<()> {
        if let Some(path) = &self.persistence_path {
            // Create directory if it doesn't exist
            if let Some(parent) = Path::new(path).parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    RuntimeError::MemoryError(format!(
                        "Failed to create directory for memory persistence: {}",
                        e
                    ))
                })?;
            }

            // Serialize only the scopes (config does not need to be saved)
            let serialized = serde_json::to_string_pretty(&self.scopes).map_err(|e| {
                RuntimeError::MemoryError(format!("Failed to serialize memory: {}", e))
            })?;

            fs::write(path, serialized).map_err(|e| {
                RuntimeError::MemoryError(format!("Failed to write memory file: {}", e))
            })?;

            Ok(())
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_scope_basic() {
        let mut scope = MemoryScope::new("test", MemoryStrategy::Buffer, Some(5));

        // Store some values
        scope
            .store(RuntimeValue::String("value1".to_string()), None, None)
            .unwrap();
        scope
            .store(RuntimeValue::String("value2".to_string()), None, None)
            .unwrap();
        scope
            .store(
                RuntimeValue::String("value3".to_string()),
                Some("key3"),
                None,
            )
            .unwrap();

        // Retrieve without key (all values)
        let values = scope.retrieve(None, None).unwrap();
        assert_eq!(values.len(), 3);

        // Retrieve with key
        let values = scope.retrieve(Some("key3"), None).unwrap();
        assert_eq!(values.len(), 1);
        if let RuntimeValue::String(value) = &values[0] {
            assert_eq!(value, "value3");
        } else {
            panic!("Expected String");
        }

        // Verify capacity limit
        scope
            .store(RuntimeValue::String("value4".to_string()), None, None)
            .unwrap();
        scope
            .store(RuntimeValue::String("value5".to_string()), None, None)
            .unwrap();
        scope
            .store(RuntimeValue::String("value6".to_string()), None, None)
            .unwrap(); // Exceeds capacity

        // It should have removed the oldest value
        let values = scope.retrieve(None, None).unwrap();
        assert_eq!(values.len(), 5); // Maximum capacity
    }

    #[test]
    fn test_memory_entries_expiration() {
        let mut scope = MemoryScope::new("test", MemoryStrategy::Buffer, None);

        // Store a value with a short TTL
        scope
            .store(
                RuntimeValue::String("temp".to_string()),
                Some("temp"),
                Some(50), // 50ms
            )
            .unwrap();

        // It should be available immediately
        let values = scope.retrieve(Some("temp"), None).unwrap();
        assert_eq!(values.len(), 1);

        // Wait for it to expire
        std::thread::sleep(std::time::Duration::from_millis(100));

        // It should be expired now
        let values = scope.retrieve(Some("temp"), None).unwrap();
        assert_eq!(values.len(), 0);

        // Clean up explicitly
        scope.clean_expired();

        // Verify it was removed from the store
        assert!(!scope.key_value_store.contains_key("temp"));
    }

    #[test]
    fn test_memory_manager_basic() {
        let config = RuntimeConfig::default();
        let mut manager = MemoryManager::new(&config);

        // Create scopes
        manager
            .create_scope("scope1", MemoryStrategy::Buffer, None)
            .unwrap();
        manager
            .create_scope("scope2", MemoryStrategy::Vector, Some(10))
            .unwrap();

        // Store values
        manager
            .store(
                "scope1",
                RuntimeValue::String("value1".to_string()),
                None,
                None,
            )
            .unwrap();
        manager
            .store(
                "scope2",
                RuntimeValue::String("value2".to_string()),
                Some("key2"),
                None,
            )
            .unwrap();

        // Retrieve values
        let values = manager.retrieve("scope1", None, None).unwrap();
        assert_eq!(values.len(), 1);

        let values = manager.retrieve("scope2", Some("key2"), None).unwrap();
        assert_eq!(values.len(), 1);

        // Nonexistent scope
        let err = manager.retrieve("scope3", None, None);
        assert!(err.is_err());

        // Remove scope
        manager.remove_scope("scope1").unwrap();
        let err = manager.retrieve("scope1", None, None);
        assert!(err.is_err());
    }
}
