// lexc/src/executor/vector_memory.rs
//
// Sprint D: Memory Index / RAG Lite - Real Vector Memory System
// Simplified implementation with SQLite + basic embeddings

use rusqlite::{params, Connection};
use serde_json::Value;
use std::collections::HashMap;

use crate::executor::{ExecutorError, RuntimeValue};
use std::collections::HashMap as StdHashMap;

#[derive(Debug, Clone)]
enum VectorBackend {
    SqliteLocal,
    Qdrant { url: String, api_key: Option<String>, collection: String },
}

/// Real vector memory system for Sprint D
pub struct VectorMemorySystem {
    /// SQLite connection for metadata and texts
    db: Connection,
    /// In-memory cache of embeddings
    embedding_cache: HashMap<String, Vec<f32>>,
    /// Embedding dimension
    embedding_dim: usize,
    /// Backend selector
    backend: VectorBackend,
}

impl VectorMemorySystem {
    fn qdrant_post_json_retry(&self, url: &str, body: serde_json::Value) -> Result<ureq::Response, ExecutorError> {
        let retries: usize = std::env::var("LEXON_LLM_RETRIES").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(2);
        let backoff_ms: u64 = std::env::var("LEXON_LLM_BACKOFF_MS").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(100);
        let throttle_ms: u64 = std::env::var("LEXON_QDRANT_THROTTLE_MS").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
        let mut attempt = 0usize;
        loop {
            if throttle_ms>0 { std::thread::sleep(std::time::Duration::from_millis(throttle_ms)); }
            let req = self.qdrant_headers(ureq::post(url)).set("Content-Type","application/json");
            let resp = req.send_json(body.clone());
            match resp {
                Ok(r) => return Ok(r),
                Err(e) => {
                    if attempt >= retries { return Err(ExecutorError::RuntimeError(format!("qdrant req failed after {} retries: {}", attempt, e))); }
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms.saturating_mul((attempt as u64)+1)));
                    attempt += 1;
                }
            }
        }
    }
    // --- Qdrant adapter helpers (scaffold) ---
    fn qdrant_headers(&self, req: ureq::Request) -> ureq::Request {
        match &self.backend {
            VectorBackend::Qdrant { api_key: Some(key), .. } => req.set("api-key", key),
            _ => req,
        }
    }

    fn qdrant_upsert(&self, path: &str, content: &str, embedding: &[f32], metadata: Option<serde_json::Value>) -> Result<(), ExecutorError> {
        let (url, collection) = match &self.backend { VectorBackend::Qdrant { url, collection, .. } => (url, collection), _ => return Ok(()) };
        let upsert_url = format!("{}/collections/{}/points", url, collection);
        let payload = {
            let mut p = serde_json::json!({"path": path, "content": content});
            if let Some(m) = metadata { if let Some(obj)=p.as_object_mut(){ obj.insert("metadata".to_string(), m); } }
            p
        };
        let body = serde_json::json!({
            "points": [
                {"id": format!("{}", self.simple_hash(content)), "vector": embedding, "payload": payload}
            ]
        });
        let resp = self.qdrant_post_json_retry(&upsert_url, body)?;
        let status = resp.status();
        if (200..300).contains(&status) { Ok(()) } else { Err(ExecutorError::RuntimeError(format!("Qdrant upsert failed: status {}", status))) }
    }

    fn qdrant_search(&self, query: &str, k: usize) -> Result<Vec<RuntimeValue>, ExecutorError> {
        let (url, collection) = match &self.backend { VectorBackend::Qdrant { url, collection, .. } => (url, collection), _ => return Ok(vec![]) };
        let vec = self.embed_text(query);
        let search_url = format!("{}/collections/{}/points/search", url, collection);
        let body = serde_json::json!({"vector": vec, "limit": k});
        let resp = self.qdrant_post_json_retry(&search_url, body);
        let resp_ok = resp.map_err(|e| ExecutorError::RuntimeError(format!("qdrant req: {}", e)))?;
        let status = resp_ok.status();
        if !(200..300).contains(&status) { return Ok(vec![]); }
        let json: serde_json::Value = resp_ok.into_json().map_err(|e| ExecutorError::RuntimeError(format!("qdrant parse: {}", e)))?;
        let mut out = Vec::new();
        if let Some(arr) = json.get("result").and_then(|v| v.as_array()) {
            for it in arr {
                let payload = it.get("payload").cloned().unwrap_or(serde_json::json!({}));
                let path = payload.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let content = payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let score = it.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let obj = serde_json::json!({"content": content.chars().take(500).collect::<String>(), "path": path, "similarity": score});
                out.push(RuntimeValue::Json(obj));
            }
        }
        Ok(out)
    }
    /// Creates a new vector memory system
    pub fn new(db_path: Option<&str>) -> Result<Self, ExecutorError> {
        let backend = match std::env::var("LEXON_VECTOR_BACKEND").unwrap_or_else(|_| "sqlite_local".to_string()).to_lowercase().as_str() {
            "qdrant" => {
                let url = std::env::var("LEXON_QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
                let collection = std::env::var("LEXON_QDRANT_COLLECTION").unwrap_or_else(|_| "lexon_docs".to_string());
                let api_key = std::env::var("LEXON_QDRANT_API_KEY").ok();
                VectorBackend::Qdrant { url, api_key, collection }
            }
            _ => VectorBackend::SqliteLocal,
        };

        let db_path = db_path.unwrap_or(":memory:");
        let db = Connection::open(db_path).map_err(|e| {
            ExecutorError::MemoryError(format!("Failed to open SQLite database: {}", e))
        })?;

        // Initialize database schema
        Self::init_schema(&db)?;

        let mut system = VectorMemorySystem {
            db,
            embedding_cache: HashMap::new(),
            embedding_dim: 128, // Reduced dimension for simplicity
            backend,
        };

        match &system.backend {
            VectorBackend::SqliteLocal => {
        println!("📄 Vector Memory System initialized with SQLite backend");
            }
            VectorBackend::Qdrant { url, collection, .. } => {
                println!("🌐 Vector Memory System using Qdrant at {} (collection '{}')", url, collection);
                // Best-effort ensure collection exists
                let _ = system.qdrant_ensure_collection();
            }
        }

        Ok(system)
    }

    fn qdrant_ensure_collection(&mut self) -> Result<(), ExecutorError> {
        let (url, collection) = match &self.backend {
            VectorBackend::Qdrant { url, collection, .. } => (url.clone(), collection.clone()),
            _ => return Ok(())
        };
        let create_url = format!("{}/collections/{}", url, collection);
        let body = serde_json::json!({
            "vectors": {"size": self.embedding_dim, "distance": "Cosine"},
            "on_disk_payload": true
        });
        let resp = self.qdrant_post_json_retry(&create_url, body)?;
        let status = resp.status();
        if (200..300).contains(&status) || status == 409 { // 409 already exists
            Ok(())
        } else {
            Err(ExecutorError::RuntimeError(format!("Qdrant ensure collection failed: status {}", status)))
        }
    }

    pub fn qdrant_create_index(&self, field_name: &str, field_schema: &str) -> Result<(), ExecutorError> {
        let (url, collection) = match &self.backend {
            VectorBackend::Qdrant { url, collection, .. } => (url.clone(), collection.clone()),
            _ => return Ok(())
        };
        let idx_url = format!("{}/collections/{}/indexes", url, collection);
        let body = serde_json::json!({"field_name": field_name, "field_schema": field_schema});
        let resp = self.qdrant_post_json_retry(&idx_url, body)?;
        let status = resp.status();
        if (200..300).contains(&status) { Ok(()) } else { Err(ExecutorError::RuntimeError(format!("Qdrant create index failed: status {}", status))) }
    }

    pub fn qdrant_set_schema(&self, schema: &serde_json::Value) -> Result<(), ExecutorError> {
        let (url, collection) = match &self.backend {
            VectorBackend::Qdrant { url, collection, .. } => (url.clone(), collection.clone()),
            _ => return Ok(())
        };
        let schema_url = format!("{}/collections/{}/payload/index", url, collection);
        let resp = self.qdrant_post_json_retry(&schema_url, schema.clone())?;
        let status = resp.status();
        if (200..300).contains(&status) { Ok(()) } else { Err(ExecutorError::RuntimeError(format!("Qdrant set schema failed: status {}", status))) }
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

    pub fn embed_text(&self, text: &str) -> Vec<f32> {
        let provider = std::env::var("LEXON_EMBEDDINGS_PROVIDER").unwrap_or_default();
        if provider.eq_ignore_ascii_case("openai") {
            if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
                let model = std::env::var("LEXON_OPENAI_EMBEDDINGS_MODEL").unwrap_or_else(|_| "text-embedding-3-small".to_string());
                let url = "https://api.openai.com/v1/embeddings";
                let body = serde_json::json!({"model": model, "input": text});
                let resp = ureq::post(url)
                    .set("Authorization", &format!("Bearer {}", api_key))
                    .set("Content-Type", "application/json")
                    .send_json(body);
                if let Ok(r) = resp {
                    if let Ok(val) = r.into_json::<serde_json::Value>() {
                        if let Some(arr) = val.get("data").and_then(|v| v.as_array()).and_then(|a| a.get(0)).and_then(|o| o.get("embedding")).and_then(|e| e.as_array()) {
                            let mut out = Vec::new();
                            for v in arr { if let Some(f)=v.as_f64() { out.push(f as f32); } }
                            if !out.is_empty() { return out; }
                        }
                    }
                }
            }
        }
        // fallback
        self.generate_keyword_vector(text)
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
        // Backend dispatch: Qdrant or local SQLite
        if let VectorBackend::Qdrant { .. } = &self.backend {
            let mut ingested = 0usize;
            for path in paths {
                let content = std::fs::read_to_string(path).map_err(|e| {
                    ExecutorError::MemoryError(format!("Failed to read file '{}': {}", path, e))
                })?;
                let embedding = self.embed_text(&content);
                self.qdrant_upsert(path, &content, &embedding, None)?;
                println!("📄 Ingested to Qdrant: {} (dim: {})", path, embedding.len());
                ingested += 1;
            }
            return Ok(ingested);
        }

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

            // Generate embedding (provider or keyword)
            let embedding = self.embed_text(&content);
            let embedding = self.embed_text(&content);

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

    /// Ingests documents by chunking content before embedding/storing
    pub fn ingest_documents_chunks(
        &mut self,
        paths: &[String],
        by: &str,
        size: usize,
        overlap: usize,
    ) -> Result<usize, ExecutorError> {
        if let VectorBackend::Qdrant { .. } = &self.backend {
            let mut total = 0usize;
            for path in paths {
                let content = std::fs::read_to_string(path).map_err(|e| {
                    ExecutorError::MemoryError(format!("Failed to read file '{}': {}", path, e))
                })?;
                let chunks = self.chunk_text(&content, by, size, overlap);
                let chunk_count = chunks.len();
                for (idx, chunk) in chunks.into_iter().enumerate() {
                    let meta = serde_json::json!({
                        "source_path": path,
                        "chunk_index": idx,
                        "chunk_count": chunk_count,
                        "chunking": {"by": by, "size": size, "overlap": overlap}
                    });
                    let emb = self.embed_text(&chunk);
                    let emb = self.embed_text(&chunk);
                    self.qdrant_upsert(&format!("{}#chunk_{}", path, idx), &chunk, &emb, Some(meta))?;
                    total += 1;
                }
            }
            return Ok(total);
        }

        let mut total_chunks = 0usize;
        for path in paths {
            let content = std::fs::read_to_string(path).map_err(|e| {
                ExecutorError::MemoryError(format!("Failed to read file '{}': {}", path, e))
            })?;
            let chunks = self.chunk_text(&content, by, size, overlap);
            let chunk_count = chunks.len();
            for (idx, chunk) in chunks.into_iter().enumerate() {
                let content_hash = format!("{:x}", self.simple_hash(&chunk));
                let exists: bool = self
                    .db
                    .query_row(
                        "SELECT 1 FROM documents WHERE content_hash = ?1",
                        params![content_hash],
                        |_| Ok(true),
                    )
                    .unwrap_or(false);
                if exists { continue; }
                let meta_obj = serde_json::json!({
                    "source_path": path,
                    "chunk_index": idx,
                    "chunk_count": chunk_count,
                    "chunking": {"by": by, "size": size, "overlap": overlap}
                });
                let embedding = self.embed_text(&chunk);
                self.db.execute(
                    "INSERT INTO documents (path, content, content_hash, metadata) VALUES (?1, ?2, ?3, ?4)",
                    params![format!("{}#chunk_{}", path, idx), chunk, content_hash, meta_obj.to_string()]
                ).map_err(|e| ExecutorError::MemoryError(format!("Failed to insert document: {}", e)))?;
                let doc_id = self.db.last_insert_rowid();
                let embedding_bytes: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes().to_vec()).collect();
                self.db.execute(
                    "INSERT INTO embeddings (document_id, embedding) VALUES (?1, ?2)",
                    params![doc_id, embedding_bytes]
                ).map_err(|e| ExecutorError::MemoryError(format!("Failed to insert embedding: {}", e)))?;
                println!("📄 Ingested chunk {} of {} from {} (doc_id: {}, dim: {})", idx+1, chunk_count, path, doc_id, embedding.len());
                total_chunks += 1;
            }
        }
        Ok(total_chunks)
    }

    /// Simple chunker (tokens|chars|paragraphs) with overlap
    pub fn chunk_text(&self, text: &str, by: &str, size: usize, overlap: usize) -> Vec<String> {
        if size == 0 { return vec![]; }
        match by.to_ascii_lowercase().as_str() {
            "chars" => {
                let mut out = Vec::new();
                let mut start = 0usize;
                let len = text.len();
                while start < len {
                    let end = (start + size).min(len);
                    out.push(text[start..end].to_string());
                    if end == len { break; }
                    let step = size.saturating_sub(overlap).max(1);
                    start += step;
                }
                out
            }
            "paragraphs" => {
                let paras: Vec<&str> = text.split("\n\n").collect();
                // Greedy pack paragraphs to reach approx size (by chars length)
                let mut out = Vec::new();
                let mut buf = String::new();
                for p in paras {
                    if buf.len() + p.len() + 2 > size && !buf.is_empty() {
                        out.push(buf.clone());
                        if overlap > 0 { /* naive overlap: append last para again */ out.push(p.to_string()); }
                        buf.clear();
                    }
                    if !buf.is_empty() { buf.push_str("\n\n"); }
                    buf.push_str(p);
                }
                if !buf.is_empty() { out.push(buf); }
                out
            }
            _ => {
                // tokens (default): whitespace tokens
                let tokens: Vec<&str> = text.split_whitespace().collect();
                let mut out = Vec::new();
                let mut i = 0usize;
                while i < tokens.len() {
                    let end = (i + size).min(tokens.len());
                    out.push(tokens[i..end].join(" "));
                    if end == tokens.len() { break; }
                    let step = size.saturating_sub(overlap).max(1);
                    i += step;
                }
                out
            }
        }
    }

    /// Searches similar documents using vector embeddings
    pub fn vector_search(
        &mut self,
        query: &str,
        k: usize,
    ) -> Result<Vec<RuntimeValue>, ExecutorError> {
        if let VectorBackend::Qdrant { .. } = &self.backend {
            return self.qdrant_search(query, k);
        }
        // Generate query embedding
        let query_embedding = self.embed_text(query);

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

    pub fn set_metadata(&self, path: &str, metadata: &serde_json::Value) -> Result<(), ExecutorError> {
        let meta_str = metadata.to_string();
        self.db
            .execute(
                "UPDATE documents SET metadata = ?1 WHERE path = ?2",
                params![meta_str, path],
            )
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to update metadata: {}", e)))?;
        Ok(())
    }

    /// Prunes documents whose metadata matches all provided filters
    pub fn prune_by_metadata(
        &mut self,
        filters: &StdHashMap<String, String>,
    ) -> Result<usize, ExecutorError> {
        if filters.is_empty() {
            return Ok(0);
        }
        let mut stmt = self
            .db
            .prepare("SELECT id, metadata FROM documents")
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to prepare prune query: {}", e)))?;
        let rows = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let metadata_text: String = row.get(1)?;
                Ok((id, metadata_text))
            })
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to query documents for prune: {}", e)))?;

        let mut to_delete: Vec<i64> = Vec::new();
        for r in rows {
            let (id, meta_text) = r
                .map_err(|e| ExecutorError::MemoryError(format!("Failed to read row: {}", e)))?;
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta_text) {
                let mut pass = true;
                for (k, v) in filters.iter() {
                    let mv = meta.get(k).and_then(|x| x.as_str()).unwrap_or("");
                    if mv != v {
                        pass = false;
                        break;
                    }
                }
                if pass {
                    to_delete.push(id);
                }
            }
        }

        let mut deleted = 0usize;
        for id in to_delete {
            self.db
                .execute("DELETE FROM embeddings WHERE document_id = ?1", params![id])
                .map_err(|e| ExecutorError::MemoryError(format!("Failed to delete embeddings: {}", e)))?;
            self.db
                .execute("DELETE FROM documents WHERE id = ?1", params![id])
                .map_err(|e| ExecutorError::MemoryError(format!("Failed to delete document: {}", e)))?;
            deleted += 1;
        }
        Ok(deleted)
    }

    pub fn hybrid_search(
        &mut self,
        query: &str,
        k: usize,
        alpha: f32,
        filters: &StdHashMap<String, String>,
        offset: usize,
        limit_factor: usize,
    ) -> Result<Vec<RuntimeValue>, ExecutorError> {
        if let VectorBackend::Qdrant { .. } = &self.backend {
            return self.qdrant_hybrid_search(query, k, alpha, filters, offset, limit_factor);
        }
        let qvec = self.generate_keyword_vector(query);
        let qlower = query.to_lowercase();
        let words: Vec<&str> = qlower.split_whitespace().filter(|w| !w.is_empty()).collect();

        let mut stmt = self.db.prepare("SELECT d.id, d.path, d.content, d.metadata, e.embedding FROM documents d JOIN embeddings e ON d.id = e.document_id ORDER BY d.created_at DESC")
            .map_err(|e| ExecutorError::MemoryError(format!("Failed to prepare query: {}", e)))?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            let content: String = row.get(2)?;
            let metadata_text: String = row.get(3)?;
            let embedding_bytes: Vec<u8> = row.get(4)?;
            Ok((id, path, content, metadata_text, embedding_bytes))
        }).map_err(|e| ExecutorError::MemoryError(format!("Failed to query documents: {}", e)))?;

        let mut scored = Vec::new();
        for row in rows {
            let (id, path, content, metadata_text, emb_bytes) = row.map_err(|e| ExecutorError::MemoryError(format!("Failed to read row: {}", e)))?;
            if !filters.is_empty() {
                if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&metadata_text) {
                    let mut pass = true;
                    for (k, v) in filters.iter() {
                        let mv = meta.get(k).and_then(|x| x.as_str()).unwrap_or("");
                        if mv != v { pass = false; break; }
                    }
                    if !pass { continue; }
                }
            }
            if emb_bytes.len()%4!=0 { continue; }
            let emb: Vec<f32> = emb_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            let vec_sim = self.cosine_similarity(&qvec, &emb);
            let clower = content.to_lowercase();
            let matches = words.iter().filter(|w| clower.contains(**w)).count() as f32;
            let text_score = if words.is_empty() { 0.0 } else { matches / (words.len() as f32) };
            let a = alpha.clamp(0.0, 1.0);
            let hybrid = a*vec_sim + (1.0-a)*text_score;
            if hybrid>0.0 {
                let obj = serde_json::json!({
                    "content": content.chars().take(500).collect::<String>(),
                    "path": path,
                    "hybrid_score": hybrid,
                    "vector_score": vec_sim,
                    "text_score": text_score,
                    "doc_id": id
                });
                scored.push((hybrid, RuntimeValue::Json(obj)));
            }
        }
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // Apply offset then take k
        if offset >= scored.len() { return Ok(Vec::new()); }
        let start = offset;
        let end = (start + k).min(scored.len());
        let slice = &scored[start..end];
        Ok(slice.iter().map(|(_,v)| v.clone()).collect())
    }

    fn qdrant_hybrid_search(
        &self,
        query: &str,
        k: usize,
        alpha: f32,
        filters: &StdHashMap<String, String>,
        offset: usize,
        limit_factor: usize,
    ) -> Result<Vec<RuntimeValue>, ExecutorError> {
        let (url, collection) = match &self.backend { VectorBackend::Qdrant { url, collection, .. } => (url, collection), _ => return Ok(vec![]) };
        // Build server-side filter from metadata map
        let mut must: Vec<serde_json::Value> = Vec::new();
        if let Some(raw) = filters.get("__raw__") {
            if let Ok(filter_json) = serde_json::from_str::<serde_json::Value>(raw) {
                let vec = self.embed_text(query);
                let search_url = format!("{}/collections/{}/points/search", url, collection);
                let limit = k.saturating_mul(limit_factor.max(1));
                let mut body = serde_json::json!({"vector": vec, "limit": limit, "offset": offset, "with_payload": true});
                body["filter"] = filter_json;
                let resp = self.qdrant_post_json_retry(&search_url, body);
                let resp_ok = resp.map_err(|e| ExecutorError::RuntimeError(format!("qdrant req: {}", e)))?;
                let status = resp_ok.status();
                if !(200..300).contains(&status) { return Ok(vec![]); }
                let json: serde_json::Value = resp_ok.into_json().map_err(|e| ExecutorError::RuntimeError(format!("qdrant parse: {}", e)))?;
                let qlower = query.to_lowercase();
                let words: Vec<&str> = qlower.split_whitespace().filter(|w| !w.is_empty()).collect();
                let mut scored: Vec<(f32, RuntimeValue)> = Vec::new();
                if let Some(arr) = json.get("result").and_then(|v| v.as_array()) {
                    for it in arr {
                        let payload = it.get("payload").cloned().unwrap_or(serde_json::json!({}));
                        let path = payload.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let content = payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                        let vector_sim = it.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        let clower = content.to_lowercase();
                        let text_score = if words.is_empty() { 0.0 } else { words.iter().filter(|w| clower.contains(**w)).count() as f32 / (words.len() as f32) };
                        let a = alpha.clamp(0.0, 1.0);
                        let hybrid = a*vector_sim + (1.0-a)*text_score;
                        let obj = serde_json::json!({
                            "content": content.chars().take(500).collect::<String>(),
                            "path": path,
                            "hybrid_score": hybrid,
                            "vector_score": vector_sim,
                            "text_score": text_score
                        });
                        scored.push((hybrid, RuntimeValue::Json(obj)));
                    }
                }
                scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(k);
                return Ok(scored.into_iter().map(|(_,v)| v).collect());
            }
        }
        for (k,v) in filters.iter() {
            // Try to parse value as JSON for rich conditions
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(v) {
                match val {
                    serde_json::Value::Array(arr) => {
                        // any-of values
                        let mut should: Vec<serde_json::Value> = Vec::new();
                        for item in arr {
                            should.push(serde_json::json!({"key": format!("metadata.{}", k), "match": {"value": item}}));
                        }
                        if !should.is_empty() { must.push(serde_json::json!({"should": should})); }
                    }
                    serde_json::Value::Object(obj) => {
                        // range e.g. {"gte": 10, "lte": 20}
                        let mut range = serde_json::Map::new();
                        for (rk, rv) in &obj {
                            if rk == "gte" || rk == "lte" || rk == "gt" || rk == "lt" { range.insert(rk.to_string(), rv.clone()); }
                        }
                        if !range.is_empty() {
                            must.push(serde_json::json!({"key": format!("metadata.{}", k), "range": serde_json::Value::Object(range)}));
                        } else {
                            must.push(serde_json::json!({"key": format!("metadata.{}", k), "match": {"value": serde_json::Value::Object(obj)}}));
                        }
                    }
                    other => {
                        must.push(serde_json::json!({"key": format!("metadata.{}", k), "match": {"value": other}}));
                    }
                }
            } else {
                must.push(serde_json::json!({
                    "key": format!("metadata.{}", k),
                    "match": { "value": v }
                }));
            }
        }
        let vec = self.embed_text(query);
        let search_url = format!("{}/collections/{}/points/search", url, collection);
        let limit = k.saturating_mul(limit_factor.max(1));
        let mut body = serde_json::json!({"vector": vec, "limit": limit, "offset": offset, "with_payload": true});
        if !must.is_empty() {
            body["filter"] = serde_json::json!({"must": must});
        }
        let resp = self.qdrant_post_json_retry(&search_url, body);
        let resp_ok = resp.map_err(|e| ExecutorError::RuntimeError(format!("qdrant req: {}", e)))?;
        let status = resp_ok.status();
        if !(200..300).contains(&status) { return Ok(vec![]); }
        let json: serde_json::Value = resp_ok.into_json().map_err(|e| ExecutorError::RuntimeError(format!("qdrant parse: {}", e)))?;
        let qlower = query.to_lowercase();
        let words: Vec<&str> = qlower.split_whitespace().filter(|w| !w.is_empty()).collect();
        let mut scored: Vec<(f32, RuntimeValue)> = Vec::new();
        if let Some(arr) = json.get("result").and_then(|v| v.as_array()) {
            for it in arr {
                let payload = it.get("payload").cloned().unwrap_or(serde_json::json!({}));
                let path = payload.get("path").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let content = payload.get("content").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let vector_sim = it.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let clower = content.to_lowercase();
                let text_score = if words.is_empty() { 0.0 } else { words.iter().filter(|w| clower.contains(**w)).count() as f32 / (words.len() as f32) };
                let a = alpha.clamp(0.0, 1.0);
                let hybrid = a*vector_sim + (1.0-a)*text_score;
                let obj = serde_json::json!({
                    "content": content.chars().take(500).collect::<String>(),
                    "path": path,
                    "hybrid_score": hybrid,
                    "vector_score": vector_sim,
                    "text_score": text_score
                });
                scored.push((hybrid, RuntimeValue::Json(obj)));
            }
        }
        scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored.into_iter().map(|(_,v)| v).collect())
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
