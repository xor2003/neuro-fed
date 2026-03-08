// src/openai_proxy/components.rs
use serde_json::Value;

/// Semantic cache for storing and retrieving embeddings
pub struct SemanticCache {
    // Placeholder
}

impl SemanticCache {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn get(&self, _key: &str) -> Option<Value> {
        None
    }
    
    pub fn put(&mut self, _key: String, _value: Value) {
        // Placeholder
    }
}

/// Configuration for the OpenAI proxy
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProxyConfig {
    pub ollama_url: String,
    pub fallback_url: String,
    pub enable_cache: bool,
    pub cache_size: usize,
    pub timeout_seconds: u64,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            ollama_url: "http://localhost:11434".to_string(),
            fallback_url: "https://api.openai.com".to_string(),
            enable_cache: true,
            cache_size: 1000,
            timeout_seconds: 30,
        }
    }
}