// src/openai_proxy.rs
// OpenAI API transparent proxy with local fallback and learning
// Enhanced with tool calling, semantic caching, and predictive coding integration

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use std::sync::Arc;
use chrono::Utc;

use axum::{routing, Router, Json, http::StatusCode};
use axum::extract::State;
use tokio::sync::Mutex;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json;
use tracing::{info, error, debug, warn};
use ndarray::Array2;

use crate::ml_engine::MLEngine;
use crate::config::{NodeConfig, BackendConfig};
use crate::pc_hierarchy::PredictiveCoding;
use crate::types::{FunctionCall, Tool, ToolCall, ProxyStats};
use thiserror::Error;

/// Error types for OpenAI proxy operations
#[derive(Debug, Error)]
pub enum ProxyError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    #[error("Request failed: {0}")]
    RequestError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Invalid response: {0}")]
    InvalidResponse(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("PC hierarchy error: {0}")]
    PCError(String),
    #[error("Embedding generation error: {0}")]
    EmbeddingError(String),
    #[error("Backend communication error: {0}")]
    BackendError(String),
}

/// Request structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub n: Option<usize>,
    pub echo: Option<bool>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub function_call: Option<FunctionCall>,
    pub tools: Option<Vec<Tool>>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Option<Usage>,
}

/// Response structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

/// Choice structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: Option<String>,
    pub logprobs: Option<LogProbs>,
}

/// Message structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
    pub name: Option<String>,
}

/// Log probabilities structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f64>,
    pub top_logprobs: Option<HashMap<String, f64>>,
    pub text_offset: usize,
}

/// Usage structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Cache entry with semantic embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheEntry {
    pub request: OpenAiRequest,
    pub response: OpenAiResponse,
    pub timestamp: i64,
    pub embedding: Vec<f32>, // Embedding vector for semantic similarity
    pub embedding_dim: usize,
}

/// Metrics collected by the smart proxy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tool_bypass_requests: u64,
    pub semantic_similarity_hits: u64,
    pub pc_inference_calls: u64,
    pub pc_learning_calls: u64,
    pub openai_backend_calls: u64,
    pub ollama_backend_calls: u64,
    pub local_fallback_calls: u64,
    pub average_response_time_ms: f64,
    pub total_tokens_saved: u64,
    pub start_time: SystemTime,
    pub last_updated: SystemTime,
}

impl Default for ProxyMetrics {
    fn default() -> Self {
        Self {
            total_requests: 0,
            cache_hits: 0,
            cache_misses: 0,
            tool_bypass_requests: 0,
            semantic_similarity_hits: 0,
            pc_inference_calls: 0,
            pc_learning_calls: 0,
            openai_backend_calls: 0,
            ollama_backend_calls: 0,
            local_fallback_calls: 0,
            average_response_time_ms: 0.0,
            total_tokens_saved: 0,
            start_time: SystemTime::now(),
            last_updated: SystemTime::now(),
        }
    }
}

/// OpenAI proxy implementation with enhanced features
#[derive(Clone)]
pub struct OpenAiProxy {
    config: NodeConfig,
    backend_config: BackendConfig,
    local_engine: Arc<tokio::sync::Mutex<MLEngine>>,
    pc_hierarchy: Arc<tokio::sync::Mutex<PredictiveCoding>>,
    client: Client,
    semantic_cache: Arc<tokio::sync::Mutex<HashMap<String, SemanticCacheEntry>>>,
    metrics: Arc<tokio::sync::Mutex<ProxyMetrics>>,
    stats: Arc<tokio::sync::Mutex<ProxyStats>>,
}

impl OpenAiProxy {
    /// Create a new OpenAI proxy with enhanced features
    pub fn new(
        config: NodeConfig,
        backend_config: BackendConfig,
        local_engine: Arc<Mutex<MLEngine>>,
        pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
    ) -> Self {
        Self {
            config,
            backend_config,
            local_engine,
            pc_hierarchy,
            client: Client::new(),
            semantic_cache: Arc::new(Mutex::new(HashMap::new())),
            metrics: Arc::new(Mutex::new(ProxyMetrics::default())),
            stats: Arc::new(Mutex::new(ProxyStats {
                requests_total: 0,
                requests_successful: 0,
                requests_failed: 0,
                average_response_time: 0.0,
                last_reset: SystemTime::now(),
            })),
        }
    }

    /// Start the OpenAI proxy server
    pub async fn start(self, port: u16) -> Result<(), ProxyError> {
        // Create a shared state - NO OUTER MUTEX NEEDED
        let state = Arc::new(self);
        
        let app = Router::new()
            .route("/v1/models", routing::get(|State(_state): State<Arc<OpenAiProxy>>| async move {
                Ok::<Json<serde_json::Value>, StatusCode>(Json(serde_json::json!({ "data": [] })))
            }))
            .route("/v1/chat/completions", routing::post(|State(state): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>| async move {
                OpenAiProxy::handle_chat_completion(State(state), Json(req)).await
            }))
            .route("/v1/completions", routing::post(|State(state): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>| async move {
                OpenAiProxy::handle_completion(State(state), Json(req)).await
            }))
            .route("/v1/embeddings", routing::post(|State(state): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>| async move {
                OpenAiProxy::handle_embeddings(State(state), Json(req)).await
            }))
            .route("/v1/metrics", routing::get(|State(state): State<Arc<OpenAiProxy>>| async move {
                OpenAiProxy::handle_metrics(State(state)).await
            }))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await
            .map_err(|e| ProxyError::ConfigError(format!("Failed to bind to port {}: {}", port, e)))?;
        axum::serve(listener, app).await
            .map_err(|e| ProxyError::ConfigError(format!("Failed to start server: {}", e)))?;
        Ok(())
    }

    /// Handle chat completion requests with enhanced routing logic
    pub async fn handle_chat_completion(
        State(proxy): State<Arc<OpenAiProxy>>,
        Json(req): Json<OpenAiRequest>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        let start_time = Instant::now();
        
        // Update metrics
        proxy.metrics.lock().await.total_requests += 1;
        proxy.metrics.lock().await.last_updated = SystemTime::now();
        
        // 1. Check for tool calls - Tool Bypass Mode
        let has_tools = proxy.has_tool_calls(&req);
        if has_tools && proxy.backend_config.tool_bypass_enabled {
            proxy.metrics.lock().await.tool_bypass_requests += 1;
            debug!("Tool bypass mode activated, forwarding directly to backend");
            match proxy.forward_to_backend(&req).await {
                Ok(response) => {
                    let elapsed = start_time.elapsed();
                    proxy.update_metrics_success(elapsed, &response).await;
                    return Ok(Json(response));
                }
                Err(e) => {
                    error!("Tool bypass forwarding failed: {}", e);
                    return Err(StatusCode::INTERNAL_SERVER_ERROR);
                }
            }
        }
        
        // 2. Semantic caching check
        if proxy.backend_config.semantic_cache_enabled {
            if let Some(cached_response) = proxy.check_semantic_cache(&req).await {
                proxy.metrics.lock().await.cache_hits += 1;
                proxy.metrics.lock().await.semantic_similarity_hits += 1;
                debug!("Semantic cache hit, returning cached response");
                let elapsed = start_time.elapsed();
                proxy.update_metrics_success(elapsed, &cached_response).await;
                return Ok(Json(cached_response));
            }
        }
        
        // 3. PC Inference if enabled
        if proxy.backend_config.pc_inference_enabled {
            proxy.metrics.lock().await.pc_inference_calls += 1;
            if let Some(inferred_response) = proxy.pc_inference(&req).await {
                debug!("PC inference generated response");
                let elapsed = start_time.elapsed();
                proxy.update_metrics_success(elapsed, &inferred_response).await;
                return Ok(Json(inferred_response));
            }
        }
        
        // 4. Forward to appropriate backend
        proxy.metrics.lock().await.cache_misses += 1;
        let response = match proxy.forward_to_backend(&req).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Backend forwarding failed: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        };
        
        // 5. Learn from response if PC learning enabled
        if proxy.backend_config.pc_learning_enabled {
            proxy.metrics.lock().await.pc_learning_calls += 1;
            if let Err(e) = proxy.learn_from_response(&req, &response).await {
                warn!("PC learning failed: {}", e);
            }
        }
        
        // 6. Update semantic cache
        if proxy.backend_config.semantic_cache_enabled {
            if let Err(e) = proxy.update_semantic_cache(&req, &response).await {
                warn!("Failed to update semantic cache: {}", e);
            }
        }
        
        let elapsed = start_time.elapsed();
        proxy.update_metrics_success(elapsed, &response).await;
        Ok(Json(response))
    }

    /// Check if request contains tool calls
    fn has_tool_calls(&self, req: &OpenAiRequest) -> bool {
        req.tools.is_some() || req.tool_calls.is_some() || req.function_call.is_some()
    }

    /// Generate embedding for a request using ML Engine
    async fn generate_embedding(&self, req: &OpenAiRequest) -> Result<Vec<f32>, ProxyError> {
        // Convert request to text representation for embedding
        let text = self.request_to_text(req);
        
        let engine = self.local_engine.lock().await;
        let tensor = engine.process_text(&text).await
            .map_err(|e| ProxyError::EmbeddingError(format!("Failed to process text: {}", e)))?;
        
        // Convert candle tensor to Vec<f32>
        let shape = tensor.shape();
        let total_elements: usize = shape.dims().iter().product();
        let data = tensor.to_vec1::<f32>()
            .map_err(|e| ProxyError::EmbeddingError(format!("Failed to convert tensor: {}", e)))?;
        
        // Ensure we have enough elements
        if data.len() < total_elements {
            return Err(ProxyError::EmbeddingError("Tensor data mismatch".to_string()));
        }
        
        Ok(data[..total_elements].to_vec())
    }

    /// Convert request to text for embedding generation
    fn request_to_text(&self, req: &OpenAiRequest) -> String {
        let mut text = String::new();
        text.push_str(&format!("Model: {}\n", req.model));
        for message in &req.messages {
            text.push_str(&format!("{}: {}\n", message.role, message.content));
        }
        if let Some(tools) = &req.tools {
            text.push_str(&format!("Tools: {:?}\n", tools));
        }
        text
    }

    /// Check semantic cache for similar requests
    async fn check_semantic_cache(&self, req: &OpenAiRequest) -> Option<OpenAiResponse> {
        let embedding = match self.generate_embedding(req).await {
            Ok(emb) => emb,
            Err(e) => {
                warn!("Failed to generate embedding for cache check: {}", e);
                return None;
            }
        };
        
        let cache = self.semantic_cache.lock().await;
        let threshold = self.backend_config.semantic_similarity_threshold;
        
        for (_, entry) in cache.iter() {
            if self.cosine_similarity(&embedding, &entry.embedding) >= threshold {
                debug!("Semantic cache hit with similarity above threshold");
                return Some(entry.response.clone());
            }
        }
        
        None
    }

    /// Update semantic cache with new request-response pair
    async fn update_semantic_cache(&self, req: &OpenAiRequest, response: &OpenAiResponse) -> Result<(), ProxyError> {
        let embedding = self.generate_embedding(req).await?;
        
        let mut cache = self.semantic_cache.lock().await;
        
        // Implement LRU eviction if cache exceeds max size
        if cache.len() >= self.backend_config.max_cache_size {
            // Find the entry with the smallest timestamp (oldest)
            let mut oldest_key = None;
            let mut oldest_timestamp = i64::MAX;
            for (key, entry) in cache.iter() {
                if entry.timestamp < oldest_timestamp {
                    oldest_timestamp = entry.timestamp;
                    oldest_key = Some(key.clone());
                }
            }
            if let Some(key) = oldest_key {
                cache.remove(&key);
                debug!("Evicted oldest cache entry (timestamp: {}) due to size limit", oldest_timestamp);
            }
        }
        
        let embedding_dim = embedding.len();
        let entry = SemanticCacheEntry {
            request: req.clone(),
            response: response.clone(),
            timestamp: Utc::now().timestamp(),
            embedding,
            embedding_dim,
        };
        
        let key = self.hash_request(req);
        cache.insert(key, entry);
        debug!("Updated semantic cache with new entry");
        
        Ok(())
    }

    /// Perform PC inference on the request
    async fn pc_inference(&self, req: &OpenAiRequest) -> Option<OpenAiResponse> {
        let embedding = match self.generate_embedding(req).await {
            Ok(emb) => emb,
            Err(e) => {
                warn!("Failed to generate embedding for PC inference: {}", e);
                return None;
            }
        };
        
        // Convert embedding to ndarray for PC hierarchy
        let embedding_clone = embedding.clone();
        let embedding_array = Array2::from_shape_vec((1, embedding_clone.len()), embedding_clone)
            .expect("Failed to create embedding array");
        
        // Clone the PC hierarchy for spawn_blocking
        let pc_hierarchy = self.pc_hierarchy.clone();
        
        // Perform heavy CPU inference in spawn_blocking
        match tokio::task::spawn_blocking(move || {
            let mut pc = pc_hierarchy.blocking_lock();
            pc.infer(&embedding_array, 10)
        }).await {
            Ok(Ok(stats)) => {
                debug!("PC inference completed with surprise: {}", stats.total_surprise);
                // For now, return None to let backend handle it
                // In a more advanced implementation, we could generate response from PC beliefs
                None
            }
            Ok(Err(e)) => {
                warn!("PC inference failed: {}", e);
                None
            }
            Err(join_err) => {
                error!("PC inference task panicked: {}", join_err);
                None
            }
        }
    }

    /// Learn from API response using PC hierarchy
    async fn learn_from_response(&self, req: &OpenAiRequest, response: &OpenAiResponse) -> Result<(), ProxyError> {
        // Generate embedding for request
        let req_embedding = self.generate_embedding(req).await?;
        
        // Generate embedding for response (simplified - use response text)
        let response_text = self.response_to_text(response);
        let engine = self.local_engine.lock().await;
        let resp_tensor = engine.process_text(&response_text).await
            .map_err(|e| ProxyError::EmbeddingError(format!("Failed to process response text: {}", e)))?;
        let resp_data = resp_tensor.to_vec1::<f32>()
            .map_err(|e| ProxyError::EmbeddingError(format!("Failed to convert response tensor: {}", e)))?;
        
        // Create combined embedding for learning
        let mut combined = req_embedding;
        combined.extend(resp_data);
        
        // Ensure dimensions match PC hierarchy input
        let combined_array = Array2::from_shape_vec((1, combined.len()), combined)
            .map_err(|e| ProxyError::PCError(format!("Failed to create learning array: {}", e)))?;
        
        // Clone the PC hierarchy for spawn_blocking
        let pc_hierarchy = self.pc_hierarchy.clone();
        
        // Perform heavy CPU learning in spawn_blocking
        tokio::task::spawn_blocking(move || {
            let mut pc = pc_hierarchy.blocking_lock();
            pc.learn_legacy(&combined_array)
        }).await
        .map_err(|join_err| ProxyError::PCError(format!("PC learning task panicked: {}", join_err)))?
        .map_err(|e| ProxyError::PCError(format!("PC learning failed: {}", e)))?;
        
        debug!("PC learning completed successfully");
        Ok(())
    }

    /// Convert response to text for embedding
    fn response_to_text(&self, response: &OpenAiResponse) -> String {
        let mut text = String::new();
        text.push_str(&format!("Model: {}\n", response.model));
        for choice in &response.choices {
            text.push_str(&format!("Choice {}: {}\n", choice.index, choice.message.content));
        }
        text
    }

    /// Forward request to appropriate backend (OpenAI, Ollama, or local fallback)
    async fn forward_to_backend(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        // Determine which backend to use based on configuration and request
        let use_openai = self.backend_config.openai_api_key.is_some();
        let use_ollama = !use_openai || req.model.contains("ollama") || req.model.contains("local");
        
        if use_openai && !use_ollama {
            self.metrics.lock().await.openai_backend_calls += 1;
            self.forward_to_openai(req).await
        } else if self.backend_config.local_fallback_enabled {
            self.metrics.lock().await.local_fallback_calls += 1;
            self.forward_to_ollama(req).await
        } else {
            self.metrics.lock().await.ollama_backend_calls += 1;
            self.forward_to_ollama(req).await
        }
    }

    /// Forward request to OpenAI API
    async fn forward_to_openai(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let api_key = self.backend_config.openai_api_key.as_ref()
            .ok_or_else(|| ProxyError::ConfigError("OpenAI API key not configured".to_string()))?;
        
        let client = &self.client;
        let url = format!("{}/v1/chat/completions", self.backend_config.openai_base_url);
        
        let response = client.post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(req)
            .send()
            .await
            .map_err(|e| ProxyError::BackendError(format!("OpenAI request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::BackendError(format!("OpenAI API error {}: {}", status, body)));
        }
        
        let response_json: OpenAiResponse = response.json()
            .await
            .map_err(|e| ProxyError::SerializationError(format!("Failed to parse OpenAI response: {}", e)))?;
        
        Ok(response_json)
    }

    /// Forward request to Ollama API
    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let client = &self.client;
        let url = format!("{}/api/chat", self.backend_config.ollama_base_url);
        
        // Convert OpenAI request format to Ollama format
        let ollama_req = self.convert_to_ollama_format(req);
        
        let response = client.post(&url)
            .header("Content-Type", "application/json")
            .json(&ollama_req)
            .send()
            .await
            .map_err(|e| ProxyError::BackendError(format!("Ollama request failed: {}", e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(ProxyError::BackendError(format!("Ollama API error {}: {}", status, body)));
        }
        
        // Convert Ollama response to OpenAI format
        let ollama_resp: serde_json::Value = response.json()
            .await
            .map_err(|e| ProxyError::SerializationError(format!("Failed to parse Ollama response: {}", e)))?;
        
        self.convert_from_ollama_format(&ollama_resp, req)
    }

    /// Convert OpenAI request to Ollama format
    fn convert_to_ollama_format(&self, req: &OpenAiRequest) -> serde_json::Value {
        let mut messages = Vec::new();
        for msg in &req.messages {
            messages.push(serde_json::json!({
                "role": msg.role,
                "content": msg.content.to_string(),
            }));
        }
        
        serde_json::json!({
            "model": req.model,
            "messages": messages,
            "options": {
                "temperature": req.temperature.unwrap_or(0.7),
                "top_p": req.top_p.unwrap_or(1.0),
                "max_tokens": req.max_tokens.unwrap_or(2048),
            }
        })
    }

    /// Convert Ollama response to OpenAI format
    fn convert_from_ollama_format(&self, ollama_resp: &serde_json::Value, original_req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let message = ollama_resp.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        
        let response = OpenAiResponse {
            id: ollama_resp.get("created_at")
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string(),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: original_req.model.clone(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String(message.to_string()),
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 0, // Ollama doesn't provide token counts
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        
        Ok(response)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }

    /// Hash request for cache key
    fn hash_request(&self, req: &OpenAiRequest) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        let json = serde_json::to_string(req).unwrap_or_default();
        hasher.update(json);
        format!("{:x}", hasher.finalize())
    }

    /// Update metrics after successful request
    async fn update_metrics_success(&self, elapsed: Duration, response: &OpenAiResponse) {
        let mut metrics = self.metrics.lock().await;
        let mut stats = self.stats.lock().await;
        
        stats.requests_total += 1;
        stats.requests_successful += 1;
        
        // Update average response time (exponential moving average)
        let elapsed_ms = elapsed.as_millis() as f64;
        if stats.average_response_time == 0.0 {
            stats.average_response_time = elapsed_ms as f32;
        } else {
            stats.average_response_time = 0.9 * stats.average_response_time + 0.1 * elapsed_ms as f32;
        }
        
        metrics.average_response_time_ms = (metrics.average_response_time_ms * (metrics.total_requests as f64 - 1.0) + elapsed_ms) / metrics.total_requests as f64;
        metrics.total_tokens_saved += response.usage.total_tokens as u64;
        metrics.last_updated = SystemTime::now();
    }

    /// Handle metrics endpoint
    pub async fn handle_metrics(
        State(proxy): State<Arc<OpenAiProxy>>,
    ) -> Result<Json<ProxyMetrics>, StatusCode> {
        let metrics = proxy.metrics.lock().await.clone();
        Ok(Json(metrics))
    }

    /// List models endpoint
    pub async fn list_models(
        State(_proxy): State<Arc<OpenAiProxy>>,
    ) -> Result<Json<serde_json::Value>, StatusCode> {
        Ok(Json(serde_json::json!({ "data": [] })))
    }

    /// Handle completion requests (legacy)
    pub async fn handle_completion(
        State(proxy): State<Arc<OpenAiProxy>>,
        Json(req): Json<OpenAiRequest>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        // Convert completion request to chat completion format
        Self::handle_chat_completion(State(proxy), Json(req)).await
    }

    /// Handle embeddings requests
    pub async fn handle_embeddings(
        State(proxy): State<Arc<OpenAiProxy>>,
        Json(req): Json<OpenAiRequest>,
    ) -> Result<Json<OpenAiResponse>, StatusCode> {
        // Generate embedding using ML engine
        let text = proxy.request_to_text(&req);
        let engine = proxy.local_engine.lock().await;
        let tensor = match engine.process_text(&text).await {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to generate embedding: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        };
        
        // Convert to response format
        let _data = tensor.to_vec1::<f32>().unwrap_or_default();
        let response = OpenAiResponse {
            id: "embed-".to_string() + &Utc::now().timestamp().to_string(),
            object: "list".to_string(),
            created: Utc::now().timestamp(),
            model: req.model,
            choices: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
        };
        
        Ok(Json(response))
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> ProxyMetrics {
        self.metrics.lock().await.clone()
    }

    /// Get cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.semantic_cache.lock().await;
        (cache.len(), self.backend_config.max_cache_size)
    }

    /// Clear semantic cache
    pub async fn clear_cache(&self) {
        let mut cache = self.semantic_cache.lock().await;
        cache.clear();
        info!("Semantic cache cleared");
    }

    /// Reset metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().await;
        *metrics = ProxyMetrics::default();
        info!("Proxy metrics reset");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use crate::pc_hierarchy::PCConfig;
    use crate::types::DeviceType;

    fn create_test_proxy() -> OpenAiProxy {
        let config = NodeConfig::default();
        let backend_config = BackendConfig {
            openai_api_key: Some("test-key".to_string()),
            openai_base_url: "https://api.openai.com".to_string(),
            ollama_base_url: "http://localhost:11434".to_string(),
            local_fallback_enabled: true,
            tool_bypass_enabled: true,
            semantic_cache_enabled: true,
            semantic_similarity_threshold: 0.8,
            pc_inference_enabled: true,
            pc_learning_enabled: true,
            max_cache_size: 100,
        };
        
        let device_type = DeviceType {
            name: "CPU".to_string(),
            description: "CPU device".to_string(),
            supported: true,
        };
        let local_engine = Arc::new(Mutex::new(MLEngine::new("test-model", device_type).unwrap()));
        
        let pc_config = PCConfig::new(3, vec![512, 256, 128]);
        let pc_hierarchy = Arc::new(Mutex::new(PredictiveCoding::new(pc_config).unwrap()));
        
        OpenAiProxy::new(config, backend_config, local_engine, pc_hierarchy)
    }

    #[test]
    fn test_has_tool_calls() {
        let proxy = create_test_proxy();
        
        let req_with_tools = OpenAiRequest {
            model: "test".to_string(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            n: None,
            echo: None,
            logit_bias: None,
            function_call: Some(FunctionCall {
                name: "test".to_string(),
                arguments: HashMap::new(),
            }),
            tools: None,
            tool_calls: None,
            usage: None,
        };
        
        assert!(proxy.has_tool_calls(&req_with_tools));
        
        let req_without_tools = OpenAiRequest {
            model: "test".to_string(),
            messages: vec![],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            n: None,
            echo: None,
            logit_bias: None,
            function_call: None,
            tools: None,
            tool_calls: None,
            usage: None,
        };
        
        assert!(!proxy.has_tool_calls(&req_without_tools));
    }

    #[test]
    fn test_cosine_similarity() {
        let proxy = create_test_proxy();
        
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((proxy.cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        
        let c = vec![0.0, 1.0, 0.0];
        assert!((proxy.cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
        
        let d = vec![1.0, 1.0, 0.0];
        let similarity = proxy.cosine_similarity(&a, &d);
        assert!(similarity > 0.7 && similarity < 0.8);
    }

    #[test]
    fn test_hash_request() {
        let proxy = create_test_proxy();
        
        let req = OpenAiRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("Hello".to_string()),
                name: None,
            }],
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            n: None,
            echo: None,
            logit_bias: None,
            function_call: None,
            tools: None,
            tool_calls: None,
            usage: None,
        };
        
        let hash = proxy.hash_request(&req);
        assert_eq!(hash.len(), 64); // SHA256 hex string length
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let proxy = create_test_proxy();
        
        let initial_metrics = proxy.get_metrics().await;
        assert_eq!(initial_metrics.total_requests, 0);
        
        // Simulate a request (we can't actually call handle_chat_completion without mocking)
        // But we can test that metrics struct is properly initialized
        assert_eq!(initial_metrics.cache_hits, 0);
        assert_eq!(initial_metrics.cache_misses, 0);
    }

    #[test]
    fn test_request_to_text() {
        let proxy = create_test_proxy();
        
        let req = OpenAiRequest {
            model: "gpt-4".to_string(),
            messages: vec![
                Message {
                    role: "user".to_string(),
                    content: serde_json::Value::String("Hello, world!".to_string()),
                    name: None,
                },
                Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String("Hi there!".to_string()),
                    name: None,
                },
            ],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: None,
            n: None,
            echo: None,
            logit_bias: None,
            function_call: None,
            tools: None,
            tool_calls: None,
            usage: None,
        };
        
        let text = proxy.request_to_text(&req);
        assert!(text.contains("gpt-4"));
        assert!(text.contains("Hello, world!"));
        assert!(text.contains("Hi there!"));
    }
}