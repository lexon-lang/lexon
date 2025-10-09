// lexc/src/executor/vector_memory.rs
//
// Sprint D: Memory Index / RAG Lite - Real Vector Memory System
// Simplified implementation with SQLite + basic embeddings

use rusqlite::{params, Connection};
use serde_json::Value;
use std::collections::HashMap;

use crate::executor::{ExecutorError, RuntimeValue};

/// Real vector memory system for Sprint D
pub struct VectorMemorySystem {
    /// SQLite connection for metadata and texts
    db: Connection,
    /// In-memory cache of embeddings
    embedding_cache: HashMap<String, Vec<f32>>,
    /// Embedding dimension
    embedding_dim: usize,
}

impl VectorMemorySystem {
    /// Creates a new vector memory system
    pub fn new(db_path: Option<&str>) -> Result<Self, ExecutorError> {
        let db_path = db_path.unwrap_or(":memory:");
        let db = Connection::open(db_path).map_err(|e| {
            ExecutorError::MemoryError(format!("Failed to open SQLite database: {}", e))
        })?;

        // Initialize database schema
        Self::init_schema(&db)?;

        let system = VectorMemorySystem {
            db,
            embedding_cache: HashMap::new(),
            embedding_dim: 128, // Reduced dimension for simplicity
        };

        println!("📄 Vector Memory System initialized with SQLite backend");

        Ok(system)
    }

    /// Initializes the database schema
    fn init_schema(db: &Connection) -> Result<(), ExecutorError> {
        // Table for documents and metadata
        db.execute(
            "CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )",
            [],
        )
        .map_err(|e| {
            ExecutorError::MemoryError(format!("Failed to create documents table: {}", e))
        })?;

        // Table for embeddings (as BLOB)
        db.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                document_id INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )",
            [],
        )
        .map_err(|e| {
            ExecutorError::MemoryError(format!("Failed to create embeddings table: {}", e))
        })?;

        Ok(())
    }

    /// Generates a feature vector based on keywords
    fn generate_keyword_vector(&self, text: &str) -> Vec<f32> {
        let keywords = vec![
            "lexon",
            "llm",
            "ai",
            "model",
            "programming",
            "language",
            "rag",
            "search",
            "vector",
            "embedding",
            "memory",
            "index",
            "document",
            "query",
            "context",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "generation",
            "retrieval",
            "sprint",
            "function",
            "implementation",
            "system",
            "data",
            "algorithm",
            "database",
            "file",
            "content",
            "analysis",
            "processing",
            "framework",
        ];

        let text_lower = text.to_lowercase();
        let mut features = Vec::with_capacity(self.embedding_dim);

        // Keyword-based features (first 32 dimensions)
        for keyword in keywords.iter().take(32) {
            let count = text_lower.matches(keyword).count() as f32;
            let tf = count / text.split_whitespace().count().max(1) as f32;
            features.push(tf);
        }

        // Statistical text features (next dimensions)
        let text_len_normalized = (text.len() as f32).ln() / 10.0;
        features.push(text_len_normalized);

        let word_count_normalized = (text.split_whitespace().count() as f32).ln() / 5.0;
        features.push(word_count_normalized);

        // Punctuation features
        let punct_count = text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32;
        let punct_ratio = punct_count / text.len().max(1) as f32;
        features.push(punct_ratio);

        // Uppercase features
        let upper_count = text.chars().filter(|c| c.is_uppercase()).count() as f32;
        let upper_ratio = upper_count / text.len().max(1) as f32;
        features.push(upper_ratio);

        // Fill up to embedding_dim with derived values
        while features.len() < self.embedding_dim {
            let hash_val = self.simple_hash(text) as f32 / u64::MAX as f32;
            features.push(hash_val * 0.01); // Small values so they don't dominate
        }

        // Truncate if necessary
        features.truncate(self.embedding_dim);

        // Normalize vector (L2 norm)
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for feature in &mut features {
                *feature /= norm;
            }
        }

        features
    }

    /// Calculates cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    /// Ingests documents into the vector index
    pub fn ingest_documents(&mut self, paths: &[String]) -> Result<usize, ExecutorError> {
        let mut ingested_count = 0;

        for path in paths {
            // Read file content
            let content = std::fs::read_to_string(path).map_err(|e| {
                ExecutorError::MemoryError(format!("Failed to read file '{}': {}", path, e))
            })?;

            // Simple content hash
            let content_hash = format!("{:x}", self.simple_hash(&content));

            // Check if it already exists
            let exists: bool = self
                .db
                .query_row(
                    "SELECT 1 FROM documents WHERE content_hash = ?1",
                    params![content_hash],
                    |_| Ok(true),
                )
                .unwrap_or(false);

            if exists {
                println!(
                    "⚠️ Document '{}' already exists (same content), skipping",
                    path
                );
                continue;
            }

            // Generate embedding
            let embedding = self.generate_keyword_vector(&content);

            // Insert document
            self.db.execute(
                "INSERT INTO documents (path, content, content_hash, metadata) VALUES (?1, ?2, ?3, ?4)",
                params![path, content, content_hash, "{}"]
            ).map_err(|e| ExecutorError::MemoryError(format!("Failed to insert document: {}", e)))?;

            let doc_id = self.db.last_insert_rowid();

            // Serialize embedding as bytes
            let embedding_bytes: Vec<u8> = embedding
                .iter()
                .flat_map(|f| f.to_le_bytes().to_vec())
                .collect();

            // Insert embedding
            self.db
                .execute(
                    "INSERT INTO embeddings (document_id, embedding) VALUES (?1, ?2)",
                    params![doc_id, embedding_bytes],
                )
                .map_err(|e| {
                    ExecutorError::MemoryError(format!("Failed to insert embedding: {}", e))
                })?;

            println!(
                "📄 Ingested: {} (doc_id: {}, embedding_dim: {})",
                path,
                doc_id,
                embedding.len()
            );
            ingested_count += 1;
        }

        Ok(ingested_count)
    }

    /// Searches similar documents using vector embeddings
    pub fn vector_search(
        &mut self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RuntimeValue>, ExecutorError> {
        // Generate query embedding
        let query_embedding = self.generate_keyword_vector(query);

        // Get all documents and their embeddings
        let mut stmt = self
            .db
            .prepare(
                "SELECT d.id, d.path, d.content, e.embedding 
             FROM documents d 
             JOIN embeddings e ON d.id = e.document_id
             ORDER BY d.created_at DESC",
            )
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to prepare query: {}", e)))?;

        let doc_iter = stmt
            .query_map([], |row| {
                let doc_id: i64 = row.get(0)?;
                let path: String = row.get(1)?;
                let content: String = row.get(2)?;
                let embedding_bytes: Vec<u8> = row.get(3)?;

                Ok((doc_id, path, content, embedding_bytes))
            })
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to query documents: {}", e)))?;

        let mut scored_docs = Vec::new();

        for doc_result in doc_iter {
            let (doc_id, path, content, embedding_bytes) = doc_result.map_err(|e| {
                ExecutorError::MemoryError(format!("Failed to read document row: {}", e))
            })?;

            // Deserialize embedding
            if embedding_bytes.len() % 4 != 0 {
                println!("⚠️ Invalid embedding size for doc {}, skipping", doc_id);
                continue;
            }

            let embedding: Vec<f32> = embedding_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            // Compute similarity
            let similarity = self.cosine_similarity(&query_embedding, &embedding);

            // Only include documents with similarity > 0
            if similarity > 0.0 {
                scored_docs.push((similarity, content, path, doc_id));
            }
        }

        // Sort by similarity (descending)
        scored_docs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        scored_docs.truncate(k);

        // Convert to RuntimeValue
        let results: Vec<RuntimeValue> = scored_docs
            .into_iter()
            .map(|(similarity, content, path, doc_id)| {
                // Create JSON object with metadata
                let result_obj = serde_json::json!({
                    "content": content.chars().take(500).collect::<String>(), // Limit to 500 chars
                    "path": path,
                    "similarity": similarity,
                    "doc_id": doc_id
                });
                RuntimeValue::Json(result_obj)
            })
            .collect();

        Ok(results)
    }

    /// Generates automatic RAG context from indexed documents
    pub fn generate_rag_context(&mut self) -> Result<String, ExecutorError> {
        let mut stmt = self
            .db
            .prepare("SELECT content, path FROM documents ORDER BY created_at DESC LIMIT 3")
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to prepare query: {}", e)))?;

        let doc_iter = stmt
            .query_map([], |row| {
                let content: String = row.get(0)?;
                let path: String = row.get(1)?;
                Ok((content, path))
            })
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to query documents: {}", e)))?;

        let mut context_parts = Vec::new();

        for (i, doc_result) in doc_iter.enumerate() {
            let (content, path) = doc_result.map_err(|e| {
                ExecutorError::MemoryError(format!("Failed to read document: {}", e))
            })?;

            let preview = content.chars().take(200).collect::<String>();
            let preview = if content.len() > 200 {
                format!("{}...", preview)
            } else {
                preview
            };

            context_parts.push(format!("Document {}: {} (from: {})", i + 1, preview, path));
        }

        if context_parts.is_empty() {
            Ok("No indexed documents available for context.".to_string())
        } else {
            Ok(format!(
                "RAG Context from indexed documents:\n{}",
                context_parts.join("\n\n")
            ))
        }
    }

    /// Simple hash to generate deterministic features
    fn simple_hash(&self, data: &str) -> u64 {
        let mut hash = 0u64;
        for byte in data.bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
        hash
    }

    /// Gets statistics from the memory system
    pub fn get_stats(&self) -> Result<HashMap<String, Value>, ExecutorError> {
        let doc_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to count documents: {}", e)))?;

        let embedding_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .map_err(|e| {
                ExecutorError::MemoryError(format!("Failed to count embeddings: {}", e))
            })?;

        let mut stats = HashMap::new();
        stats.insert(
            "document_count".to_string(),
            Value::Number(doc_count.into()),
        );
        stats.insert(
            "embedding_count".to_string(),
            Value::Number(embedding_count.into()),
        );
        stats.insert(
            "embedding_dim".to_string(),
            Value::Number(self.embedding_dim.into()),
        );
        stats.insert(
            "cache_size".to_string(),
            Value::Number(self.embedding_cache.len().into()),
        );

        Ok(stats)
    }
}
