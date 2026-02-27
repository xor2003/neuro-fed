// src/openai_proxy.rs
// OpenAI API transparent proxy with local fallback and learning

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::path::Path;

use axum::{routing, Router, Json, http::StatusCode};
use axum::extract::{Path, Query, State};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};
use thiserror::Error;
use tokio::sync::mpsc;
use tokio::time::sleep;
use crate::ml_engine::MLEngine;
use crate::config::NodeConfig;

/// Error types for OpenAI proxy operations
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("Request parsing failed: {0}")]
    RequestParseError(String),
    #[error("Response parsing failed: {0}")]
    ResponseParseError(String),
    #[error("OpenAI API error: {0}")]
    OpenAiError(String),
    #[error("Local model error: {0}")]
    LocalModelError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
}

/// OpenAI API request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
}

/// OpenAI API response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: i32,
    pub message: Message,
    pub finish_reason: Option<String>,
    pub logprobs: Option<LogProbs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<HashMap<String, f32>>>,
    pub text_offset: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
}

/// Cache entry for responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub request_hash: String,
    pub response: OpenAiResponse,
    pub created_at: i64,
    pub expires_at: i64,
    pub cost: f64,
}

/// Proxy statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyStats {
    pub total_requests: u64,
    pub local_responses: u64,
    pub openai_responses: u64,
    pub cache_hits: u64,
    pub total_cost: f64,
    pub last_reset: i64,
}

/// OpenAI proxy server
#[derive(Debug)]
pub struct OpenAiProxy {
    client: Client,
    local_engine: Arc<Mutex<MLEngine>>,
    config: NodeConfig,
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    stats: Arc<Mutex<ProxyStats>>,
    api_key: String,
    cache_ttl: Duration,
}

impl OpenAiProxy {
    /// Create a new OpenAI proxy
    pub fn new(config: NodeConfig, api_key: String, local_engine: Arc<Mutex<MLEngine>>) -> Self {
        Self {
            client: Client::new(),
            local_engine,
            config,
            cache: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(ProxyStats {
                total_requests: 0,
                local_responses: 0,
                openai_responses: 0,
                cache_hits: 0,
                total_cost: 0.0,
                last_reset: chrono::Utc::now().timestamp(),
            })),
            api_key,
            cache_ttl: Duration::from_secs(24 * 60 * 60), // 24 hours
        }
    }
    
    /// Start the proxy server
    pub async fn start(&self, port: u16) -> Result<(), ProxyError> {
        info!("Starting OpenAI proxy server on port {}", port);
        
        let app = Router::new()
            .route("/v1/models", routing::get(self.list_models))
            .route("/v1/completions", routing::post(self.handle_completion))
            .route("/v1/chat/completions", routing::post(self.handle_chat_completion))
            .route("/v1/embeddings", routing::post(self.handle_embeddings))
            .route("/v1/chat/completions/:id", routing::get(self.get_chat_completion))
            .with_state(self.clone());
        
        axum::Server::bind(&format!("0.0.0.0:{}", port).parse().unwrap())
            .serve(app.into_make_service())
            .await
            .map_err(|e| ProxyError::RequestParseError(e.to_string()))?;
            
        Ok(())
    }
    
    /// List available models
    async fn list_models(&self) -> Result<Json<serde_json::Value>, StatusCode> {
        // For now, return a static list of models
        let models = serde_json::json!([
            "gpt-3.5-turbo",
            "gpt-4",
            "local-model",
        ]);
        
        Ok(Json(models))
    }
    
    /// Handle chat completion requests
    async fn handle_chat_completion(
        &self,
        Json(req): Json<OpenAiRequest>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        self.stats.lock().unwrap().total_requests += 1;
        
        // Check cache first
        if let Some(cached) = self.check_cache(&req) {
            self.stats.lock().unwrap().cache_hits += 1;
            return Ok(Json(cached.response));
        }
        
        // Try local model first
        if let Ok(local_response) = self.handle_local_request(&req).await {
            self.stats.lock().unwrap().local_responses += 1;
            return Ok(Json(local_response));
        }
        
        // Fall back to OpenAI API
        match self.forward_to_openai(&req, "chat/completions").await {
            Ok(response) => {
                self.stats.lock().unwrap().openai_responses += 1;
                self.update_cache(&req, &response).await;
                Ok(Json(response))
            }
            Err(e) => {
                error!("Failed to get response from OpenAI: {}", e);
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
    
    /// Handle completion requests
    async fn handle_completion(
        &self,
        Json(req): Json<OpenAiRequest>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        self.handle_chat_completion(Json(req)).await
    }
    
    /// Handle embeddings requests
    async fn handle_embeddings(
        &self,
        Json(req): Json<OpenAiRequest>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        self.handle_chat_completion(Json(req)).await
    }
    
    /// Get chat completion by ID
    async fn get_chat_completion(
        &self,
        Path(id): Path<String>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        // For now, return a 404
        Err(StatusCode::NOT_FOUND)
    }
    
    /// Check if request is in cache
    fn check_cache(&self, req: &OpenAiRequest) -> Option<CacheEntry> {
        let cache = self.cache.lock().unwrap();
        let hash = self.hash_request(req);
        
        if let Some(entry) = cache.get(&hash) {
            let now = chrono::Utc::now().timestamp();
            if entry.expires_at > now {
                return Some(entry.clone());
            }
        }
        
        None
    }
    
    /// Update cache with new response
    async fn update_cache(&self, req: &OpenAiRequest, response: &OpenAiResponse) {
        let mut cache = self.cache.lock().unwrap();
        let hash = self.hash_request(req);
        
        let entry = CacheEntry {
            request_hash: hash.clone(),
            response: response.clone(),
            created_at: chrono::Utc::now().timestamp(),
            expires_at: chrono::Utc::now().timestamp() + self.cache_ttl.as_secs() as i64,
            cost: self.estimate_cost(response),
        };
        
        cache.insert(hash, entry);
        
        // Update stats
        let mut stats = self.stats.lock().unwrap();
        stats.total_cost += entry.cost;
    }
    
    /// Handle request with local model
    async fn handle_local_request(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let engine = self.local_engine.lock().unwrap();
        
        // Generate embedding for the prompt
        let embedding = engine.generate_embedding(&req.prompt).await.map_err(|e| {
            ProxyError::LocalModelError(format!("Failed to generate embedding: {}", e))
        })?;
        
        // For now, return a simple response based on the embedding
        let response = OpenAiResponse {
            id: uuid::Uuid::new_v4().to_string(),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: "local-model".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: format!("Local response based on embedding: {:?}", embedding),
                },
                finish_reason: Some("length".to_string()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: req.prompt.split_whitespace().count() as i32,
                completion_tokens: 10,
                total_tokens: req.prompt.split_whitespace().count() as i32 + 10,
            },
        };
        
        Ok(response)
    }
    
    /// Forward request to OpenAI API
    async fn forward_to_openai(
        &self,
        req: &OpenAiRequest,
        endpoint: &str,
    ) -> Result<OpenAiResponse, ProxyError> {
        let url = format!("https://api.openai.com/v1/{}", endpoint);
        
        let client = &self.client;
        let api_key = &self.api_key;
        
        let request_body = match endpoint {
            "chat/completions" => {
                serde_json::json!({
                    "model": req.model,
                    "messages": [{"role": "user", "content": req.prompt}],
                    "max_tokens": req.max_tokens.unwrap_or(100),
                    "temperature": req.temperature.unwrap_or(0.7),
                })
            }
            _ => {
                serde_json::json!({
                    "model": req.model,
                    "prompt": req.prompt,
                    "max_tokens": req.max_tokens.unwrap_or(100),
                    "temperature": req.temperature.unwrap_or(0.7),
                })
            }
        };
        
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ProxyError::OpenAiError(e.to_string()))?;
        
        if response.status().is_success() {
            let text = response.text().await.map_err(|e| {
                ProxyError::ResponseParseError(format!("Failed to read response: {}", e))
            })?;
            
            let openai_response: OpenAiResponse = serde_json::from_str(&text)
                .map_err(|e| ProxyError::ResponseParseError(e.to_string()))?;
            
            Ok(openai_response)
        } else {
            let status = response.status();
            let text = response.text().await.map_err(|e| {
                ProxyError::ResponseParseError(format!("Failed to read error response: {}", e))
            })?;
            
            Err(ProxyError::OpenAiError(format!("HTTP {}: {}", status, text)))
        }
    }
    
    /// Hash a request for caching
    fn hash_request(&self, req: &OpenAiRequest) -> String {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        req.model.hash(&mut hasher);
        req.prompt.hash(&mut hasher);
        req.max_tokens.hash(&mut hasher);
        req.temperature.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }
    
    /// Estimate cost of a response
    fn estimate_cost(&self, response: &OpenAiResponse) -> f64 {
        // Simple cost estimation based on token count
        let tokens = response.usage.total_tokens as f64;
        let cost_per_token = 0.0000015; // Approximate cost per token
        
        tokens * cost_per_token
    }
    
    /// Get proxy statistics
    pub fn get_stats(&self) -> ProxyStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_requests = 0;
        stats.local_responses = 0;
        stats.openai_responses = 0;
        stats.cache_hits = 0;
        stats.total_cost = 0.0;
        stats.last_reset = chrono::Utc::now().timestamp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_proxy_creation() {
        let config = NodeConfig::default();
        let api_key = "test-key".to_string();
        let engine = Arc::new(Mutex::new(MLEngine::new(MLConfig::default()).await.unwrap()));
        
        let proxy = OpenAiProxy::new(config, api_key, engine);
        assert!(proxy.client.post("https://example.com").send().await.is_err());
    }
    
    #[tokio::test]
    async fn test_request_hashing() {
        let config = NodeConfig::default();
        let api_key = "test-key".to_string();
        let engine = Arc::new(Mutex::new(MLEngine::new(MLConfig::default()).await.unwrap()));
        
        let proxy = OpenAiProxy::new(config, api_key, engine);
        
        let req1 = OpenAiRequest {
            model: "gpt-3.5-turbo".to_string(),
            prompt: "test prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
        };
        
        let req2 = OpenAiRequest {
            model: "gpt-3.5-turbo".to_string(),
            prompt: "test prompt".to_string(),
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
        };
        
        let hash1 = proxy.hash_request(&req1);
        let hash2 = proxy.hash_request(&req2);
        
        assert_eq!(hash1, hash2);
    }
}