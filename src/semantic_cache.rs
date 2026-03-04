// src/semantic_cache.rs
// High-performance semantic cache with HNSW vector index and Moka cache

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use moka::future::Cache;
use hnsw_rs::prelude::*;
use serde::{Serialize, Deserialize};
use tracing::{info, debug, warn};
use anyhow::Result;

use crate::openai_proxy::{OpenAiRequest, OpenAiResponse};

/// Semantic cache entry with vector embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheEntry {
    pub id: u32,
    pub request: OpenAiRequest,
    pub response: OpenAiResponse,
    pub embedding: Vec<f32>,
    pub access_count: u64,
    pub last_accessed: u64,
}

/// High-performance semantic cache with HNSW vector index
pub struct SemanticCache {
    // In-memory cache for fast response retrieval
    cache: Cache<u32, SemanticCacheEntry>,
    // HNSW vector index for approximate nearest neighbor search
    vector_index: Hnsw<'static, f32, DistCosine>,
    // Mapping from prompt hash to cache ID
    hash_to_id: HashMap<String, u32>,
    // Current ID counter
    next_id: u32,
    // Maximum cache size
    max_cache_size: u64,
    // Embedding dimension
    embedding_dim: usize,
    // Similarity threshold
    similarity_threshold: f32,
}

impl SemanticCache {
    /// Create a new semantic cache with specified capacity
    pub fn new(
        max_cache_size: u64,
        embedding_dim: usize,
        similarity_threshold: f32,
    ) -> Self {
        // Configure HNSW parameters
        let max_nb_connection = 16;
        let nb_layer = 16.min((max_cache_size as f32).log2() as usize);
        let ef_c = 200;
        
        let vector_index = Hnsw::<'static, f32, DistCosine>::new(
            max_nb_connection,
            max_cache_size as usize,
            nb_layer,
            ef_c,
            DistCosine {},
        );
        
        // Configure Moka cache with time-based eviction
        let cache = Cache::builder()
            .max_capacity(max_cache_size)
            .time_to_idle(std::time::Duration::from_secs(3600)) // 1 hour idle
            .time_to_live(std::time::Duration::from_secs(86400)) // 24 hours max
            .build();
        
        Self {
            cache,
            vector_index,
            hash_to_id: HashMap::new(),
            next_id: 1,
            max_cache_size,
            embedding_dim,
            similarity_threshold,
        }
    }
    
    /// Generate hash for a request (for exact matching)
    fn generate_request_hash(&self, request: &OpenAiRequest) -> String {
        use sha2::{Sha256, Digest};
        
        let request_json = serde_json::to_string(request).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(request_json);
        let result = hasher.finalize();
        format!("{:x}", result)
    }
    
    /// Check cache for similar requests using vector similarity
    pub async fn check_similarity(
        &mut self,
        embedding: &[f32],
        request: &OpenAiRequest,
    ) -> Option<OpenAiResponse> {
        // First check for exact match
        let request_hash = self.generate_request_hash(request);
        if let Some(&id) = self.hash_to_id.get(&request_hash) {
            if let Some(entry) = self.cache.get(&id).await {
                // Update access stats
                let mut updated_entry = entry.clone();
                updated_entry.access_count += 1;
                updated_entry.last_accessed = current_timestamp();
                self.cache.insert(id, updated_entry.clone()).await;
                debug!("Exact cache hit for request hash: {}", request_hash);
                return Some(entry.response.clone());
            }
        }
        
        // If no exact match, search for similar embeddings
        if self.vector_index.get_nb_point() == 0 {
            return None;
        }
        
        // Search for nearest neighbors
        let search_result = self.vector_index.search(embedding, 1, 24);
        
        if let Some(nearest) = search_result.first() {
            // Distance is 1.0 - CosineSimilarity, so we invert it
            let similarity = 1.0 - nearest.distance;
            
            if similarity >= self.similarity_threshold {
                let id = nearest.d_id as u32;
                if let Some(entry) = self.cache.get(&id).await {
                    // Update access stats
                    let mut updated_entry = entry.clone();
                    updated_entry.access_count += 1;
                    updated_entry.last_accessed = current_timestamp();
                    self.cache.insert(id, updated_entry.clone()).await;
                    debug!("Semantic cache hit with similarity: {:.4}", similarity);
                    return Some(entry.response.clone());
                }
            }
        }
        
        None
    }
    
    /// Add new request-response pair to cache
    pub async fn add_to_cache(
        &mut self,
        request: OpenAiRequest,
        response: OpenAiResponse,
        embedding: Vec<f32>,
    ) -> Result<()> {
        // Ensure embedding has correct dimension
        if embedding.len() != self.embedding_dim {
            return Err(anyhow::anyhow!(
                "Embedding dimension mismatch: expected {}, got {}",
                self.embedding_dim,
                embedding.len()
            ));
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        let request_hash = self.generate_request_hash(&request);
        
        let entry = SemanticCacheEntry {
            id,
            request: request.clone(),
            response: response.clone(),
            embedding: embedding.clone(),
            access_count: 1,
            last_accessed: current_timestamp(),
        };
        
        // Add to Moka cache
        self.cache.insert(id, entry.clone()).await;
        
        // Add to hash mapping
        self.hash_to_id.insert(request_hash, id);
        
        // Add to HNSW vector index
        self.vector_index.insert((&embedding, id as usize));
        
        debug!("Added new entry to semantic cache with ID: {}", id);
        Ok(())
    }
    
    /// Get cache statistics
    pub async fn get_stats(&self) -> CacheStats {
        let cache_size = self.cache.entry_count();
        let vector_index_size = self.vector_index.get_nb_point();
        let hash_map_size = self.hash_to_id.len();
        
        CacheStats {
            cache_size,
            vector_index_size,
            hash_map_size,
            next_id: self.next_id,
            max_cache_size: self.max_cache_size,
            embedding_dim: self.embedding_dim,
        }
    }
    
    /// Clear the entire cache
    pub async fn clear(&mut self) {
        self.cache.invalidate_all();
        self.hash_to_id.clear();
        // Note: HNSW doesn't have a clear method, we need to create a new one
        let max_nb_connection = 16;
        let nb_layer = 16.min((self.max_cache_size as f32).log2() as usize);
        let ef_c = 200;
        
        self.vector_index = Hnsw::<'static, f32, DistCosine>::new(
            max_nb_connection,
            self.max_cache_size as usize,
            nb_layer,
            ef_c,
            DistCosine {},
        );
        self.next_id = 1;
        
        info!("Semantic cache cleared");
    }
    
    /// Load cache from database (to be implemented with persistence)
    pub async fn load_from_db(&mut self, _db_path: &str) -> Result<()> {
        // TODO: Implement loading from SQLite
        warn!("Loading from database not yet implemented");
        Ok(())
    }
    
    /// Save cache to database (to be implemented with persistence)
    pub async fn save_to_db(&self, _db_path: &str) -> Result<()> {
        // TODO: Implement saving to SQLite
        warn!("Saving to database not yet implemented");
        Ok(())
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub cache_size: u64,
    pub vector_index_size: usize,
    pub hash_map_size: usize,
    pub next_id: u32,
    pub max_cache_size: u64,
    pub embedding_dim: usize,
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}