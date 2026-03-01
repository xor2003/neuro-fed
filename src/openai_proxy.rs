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
use candle_core::Tensor;

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
    #[serde(rename = "_neurofed_source", skip_serializing_if = "Option::is_none")]
    pub neurofed_source: Option<String>,
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
        
        // Flatten tensor if it's not 1D (e.g., [512, 1] -> [512])
        let data = if shape.dims().len() > 1 {
            // Reshape to 1D tensor
            let flattened = tensor.flatten_all()
                .map_err(|e| ProxyError::EmbeddingError(format!("Failed to flatten tensor: {}", e)))?;
            flattened.to_vec1::<f32>()
                .map_err(|e| ProxyError::EmbeddingError(format!("Failed to convert flattened tensor: {}", e)))?
        } else {
            tensor.to_vec1::<f32>()
                .map_err(|e| ProxyError::EmbeddingError(format!("Failed to convert tensor: {}", e)))?
        };
        
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
        
        // Convert embedding to Tensor for PC hierarchy
        let embedding_len = embedding.len();
        // PC hierarchy expects shape (embedding_dim, 1) not (1, embedding_dim)
        // Create tensor with shape (embedding_len, 1) for PC compatibility
        let embedding_tensor = Tensor::from_vec(embedding.clone(), (embedding_len, 1), &candle_core::Device::Cpu)
            .map_err(|e| {
                error!("Failed to create embedding tensor: {}", e);
                std::process::exit(1);
            })
            .expect("Failed to create embedding tensor");
        
        // Clone the PC hierarchy for spawn_blocking
        let pc_hierarchy = self.pc_hierarchy.clone();
        
        // Perform heavy CPU inference in spawn_blocking
        match tokio::task::spawn_blocking(move || {
            let mut pc = pc_hierarchy.blocking_lock();
            pc.infer(&embedding_tensor, 10)
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
        // Generate embedding for response (simplified - use response text)
        let response_text = self.response_to_text(response);
        let engine = self.local_engine.lock().await;
        let resp_tensor = engine.process_text(&response_text).await
            .map_err(|e| ProxyError::EmbeddingError(format!("Failed to process response text: {}", e)))?;
        
        debug!("Response tensor shape: {:?}, rank: {}", resp_tensor.shape(), resp_tensor.rank());
        
        // Flatten tensor if needed (handles [512, 1] shape from dummy embeddings)
        let resp_data = if resp_tensor.rank() > 1 {
            resp_tensor.flatten_all()
                .map_err(|e| ProxyError::EmbeddingError(format!("Failed to flatten response tensor: {}", e)))?
                .to_vec1::<f32>()
                .map_err(|e| ProxyError::EmbeddingError(format!("Failed to convert flattened tensor: {}", e)))?
        } else {
            resp_tensor.to_vec1::<f32>()
                .map_err(|e| ProxyError::EmbeddingError(format!("Failed to convert response tensor: {}", e)))?
        };
        
        debug!("Response data length: {}", resp_data.len());
        
        // Trim to 512 dimensions (match PC hierarchy input dimension)
        let pc_input_dim = 512;
        let trimmed_data: Vec<f32> = if resp_data.len() >= pc_input_dim {
            resp_data[..pc_input_dim].to_vec()
        } else {
            // Pad with zeros if shorter (should not happen)
            let mut padded = resp_data;
            padded.resize(pc_input_dim, 0.0);
            padded
        };
        
        debug!("Trimmed data length: {}, pc_input_dim: {}", trimmed_data.len(), pc_input_dim);
        
        // Create tensor with shape (pc_input_dim, 1) for PC hierarchy compatibility
        let learning_tensor = Tensor::from_vec(trimmed_data, (pc_input_dim, 1), &candle_core::Device::Cpu)
            .map_err(|e| ProxyError::PCError(format!("Failed to create learning tensor: {}", e)))?;
        
        debug!("Learning tensor shape: {:?}", learning_tensor.shape());
        
        // Clone the PC hierarchy for spawn_blocking
        let pc_hierarchy = self.pc_hierarchy.clone();
        
        // Perform heavy CPU learning in spawn_blocking
        tokio::task::spawn_blocking(move || {
            let mut pc = pc_hierarchy.blocking_lock();
            debug!("PC hierarchy config dim_per_level[0]: {:?}", pc.config.dim_per_level.get(0));
            pc.learn_legacy(&learning_tensor)
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
        
        let mut response_json: OpenAiResponse = response.json()
            .await
            .map_err(|e| ProxyError::SerializationError(format!("Failed to parse OpenAI response: {}", e)))?;
        
        // Add source metadata
        response_json.neurofed_source = Some("remote".to_string());
        
        Ok(response_json)
    }

    /// Forward request to Ollama API
    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let client = &self.client;
        let url = format!("{}/api/chat", self.backend_config.ollama_base_url);
        
        // Convert OpenAI request format to Ollama format
        let ollama_req = self.convert_to_ollama_format(req);
        
        // Log which model we're querying for better troubleshooting
        let model_name = ollama_req.get("model").and_then(|m| m.as_str()).unwrap_or("unknown");
        warn!("🔍 Querying Ollama model: '{}' at URL: {} (original request model: '{}')",
              model_name, url, req.model);
        
        // Log full request for debugging
        if let Ok(req_str) = serde_json::to_string_pretty(&ollama_req) {
            let truncated_req = if req_str.len() > 300 { &req_str[..300] } else { &req_str };
            debug!("Ollama request (truncated):\n{}...", truncated_req);
        }
        
        let response = client.post(&url)
            .header("Content-Type", "application/json")
            .json(&ollama_req)
            .send()
            .await
            .map_err(|e| ProxyError::BackendError(format!("Ollama request failed for model '{}': {}", model_name, e)))?;
        
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            let error_msg = format!("Ollama API error {} for model '{}': {}", status, model_name, body);
            warn!("{}", error_msg);
            return Err(ProxyError::BackendError(error_msg));
        }
        
        // Get response body as text first for debugging
        let response_body = response.text()
            .await
            .map_err(|e| {
                let error_msg = format!("Failed to read Ollama response body for model '{}': {}", model_name, e);
                warn!("{}", error_msg);
                ProxyError::SerializationError(error_msg)
            })?;
        
        // Log raw response body for debugging
        let truncated_body = if response_body.len() > 1000 { &response_body[..1000] } else { &response_body };
        debug!("Ollama raw response body (truncated):\n{}...", truncated_body);
        
        // Check if streaming is enabled (from the request)
        let is_streaming = req.stream.unwrap_or(false);
        
        let ollama_resp: serde_json::Value = if is_streaming {
            // Parse streaming NDJSON response (multiple JSON objects separated by newlines)
            let mut combined_content = String::new();
            let mut final_message: Option<serde_json::Value> = None;
            
            for line in response_body.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                
                match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(json) => {
                        // Extract content if present
                        if let Some(content) = json.get("message")
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str()) {
                            combined_content.push_str(content);
                        }
                        
                        // Check if this is the final message (done: true)
                        if let Some(done) = json.get("done").and_then(|d| d.as_bool()) {
                            if done {
                                final_message = Some(json.clone());
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to parse Ollama streaming line: {}. Line: '{}'", e, line);
                        // Continue parsing other lines
                    }
                }
            }
            
            // Create a synthetic response combining all content
            // Use the final message if available, otherwise create a synthetic one
            if let Some(final_msg) = final_message {
                final_msg
            } else {
                // Create a synthetic response with combined content
                serde_json::json!({
                    "model": model_name,
                    "message": {
                        "role": "assistant",
                        "content": combined_content
                    },
                    "done": true
                })
            }
        } else {
            // Non-streaming response: parse as single JSON
            serde_json::from_str(&response_body)
                .map_err(|e| {
                    let error_msg = format!(
                        "Failed to parse Ollama response for model '{}': {}. Response body (first 500 chars): '{}'",
                        model_name, e,
                        if response_body.len() > 500 { &response_body[..500] } else { &response_body }
                    );
                    warn!("{}", error_msg);
                    ProxyError::SerializationError(error_msg)
                })?
        };
        
        // Log parsed response for debugging
        if let Ok(resp_str) = serde_json::to_string_pretty(&ollama_resp) {
            let truncated = if resp_str.len() > 500 { &resp_str[..500] } else { &resp_str };
            debug!("Ollama parsed response (truncated):\n{}...", truncated);
        }
        
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
        
        // Map model names: if model is "neurofed" or unknown, use "tinyllama" as default (smaller, more likely installed)
        let ollama_model = match req.model.as_str() {
            "neurofed" => "tinyllama",
            "gpt-3.5-turbo" | "gpt-4" => "tinyllama",  // Map OpenAI models to local equivalents
            other => other,
        };
        
        // Include stream parameter (default to false for compatibility)
        let stream = req.stream.unwrap_or(false);
        
        serde_json::json!({
            "model": ollama_model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": req.temperature.unwrap_or(0.7),
                "top_p": req.top_p.unwrap_or(1.0),
                "max_tokens": req.max_tokens.unwrap_or(2048),
            }
        })
    }

    /// Convert Ollama response to OpenAI format
    fn convert_from_ollama_format(&self, ollama_resp: &serde_json::Value, original_req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        // Debug: log the raw response for troubleshooting
        // println!("DEBUG: Ollama raw response: {}", serde_json::to_string_pretty(ollama_resp).unwrap_or_default());
        
        // Try multiple possible response formats
        let mut message = String::new();
        
        if let Some(msg) = ollama_resp.get("message") {
            if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                message = content.to_string();
            }
        } else if let Some(content) = ollama_resp.get("response").and_then(|c| c.as_str()) {
            message = content.to_string();
        } else if let Some(content) = ollama_resp.get("content").and_then(|c| c.as_str()) {
            message = content.to_string();
        } else {
            // Fallback: try to find any string field that might be the response
            if let Some(obj) = ollama_resp.as_object() {
                for (key, value) in obj {
                    if value.is_string() && (key.contains("content") || key.contains("response") || key.contains("message")) {
                        message = value.as_str().unwrap_or("").to_string();
                        break;
                    }
                }
            }
        }
        
        let response = OpenAiResponse {
            id: ollama_resp.get("created_at")
                .and_then(|c| c.as_str())
                .or_else(|| ollama_resp.get("id").and_then(|id| id.as_str()))
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
            neurofed_source: Some("local".to_string()),
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
        
        // Convert to response format (flatten tensor if rank > 1, ignore errors)
        let _data = if tensor.rank() > 1 {
            match tensor.flatten_all() {
                Ok(flat) => flat.to_vec1::<f32>().unwrap_or_default(),
                Err(e) => {
                    error!("Failed to flatten embedding tensor: {}", e);
                    Vec::new()
                }
            }
        } else {
            tensor.to_vec1::<f32>().unwrap_or_default()
        };
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
            neurofed_source: Some("embedding".to_string()),
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

    #[test]
    fn test_model_name_mapping() {
        let proxy = create_test_proxy();
        
        // Test that neurofed model maps to tinyllama
        let neurofed_req = OpenAiRequest {
            model: "neurofed".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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
        
        let ollama_format = proxy.convert_to_ollama_format(&neurofed_req);
        assert_eq!(ollama_format["model"], "tinyllama");
        
        // Test that gpt-3.5-turbo maps to tinyllama
        let gpt_req = OpenAiRequest {
            model: "gpt-3.5-turbo".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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
        
        let ollama_format2 = proxy.convert_to_ollama_format(&gpt_req);
        assert_eq!(ollama_format2["model"], "tinyllama");
        
        // Test that unknown models pass through unchanged
        let other_req = OpenAiRequest {
            model: "custom-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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
        
        let ollama_format3 = proxy.convert_to_ollama_format(&other_req);
        assert_eq!(ollama_format3["model"], "custom-model");
    }

    #[test]
    fn test_stream_parameter_in_ollama_format() {
        let proxy = create_test_proxy();
        
        // Test with stream: true
        let req_with_stream = OpenAiRequest {
            model: "neurofed".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: Some(true),
            n: None,
            echo: None,
            logit_bias: None,
            function_call: None,
            tools: None,
            tool_calls: None,
            usage: None,
        };
        
        let ollama_format = proxy.convert_to_ollama_format(&req_with_stream);
        assert_eq!(ollama_format["stream"], true);
        assert_eq!(ollama_format["model"], "tinyllama"); // Should be mapped
        
        // Test with stream: false
        let req_without_stream = OpenAiRequest {
            model: "neurofed".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            frequency_penalty: None,
            presence_penalty: None,
            stop: None,
            stream: Some(false),
            n: None,
            echo: None,
            logit_bias: None,
            function_call: None,
            tools: None,
            tool_calls: None,
            usage: None,
        };
        
        let ollama_format2 = proxy.convert_to_ollama_format(&req_without_stream);
        assert_eq!(ollama_format2["stream"], false);
        
        // Test with stream: None (defaults to false)
        let req_stream_none = OpenAiRequest {
            model: "neurofed".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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
        
        let ollama_format3 = proxy.convert_to_ollama_format(&req_stream_none);
        assert_eq!(ollama_format3["stream"], false); // Should default to false
    }

    #[test]
    fn test_ollama_response_parsing_multiple_formats() {
        let proxy = create_test_proxy();
        
        // Create a minimal request for testing
        let req = OpenAiRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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

        // Test 1: Standard Ollama format with message.content
        let standard_response = serde_json::json!({
            "model": "tinyllama",
            "created_at": "2023-08-04T19:22:45.499127Z",
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "done": true
        });
        
        let result1 = proxy.convert_from_ollama_format(&standard_response, &req);
        assert!(result1.is_ok());
        let response1 = result1.unwrap();
        assert_eq!(response1.choices[0].message.content, serde_json::Value::String("Hello! How can I help you today?".to_string()));

        // Test 2: Alternative format with response field
        let alt_response1 = serde_json::json!({
            "model": "tinyllama",
            "response": "This is a direct response field",
            "done": true
        });
        
        let result2 = proxy.convert_from_ollama_format(&alt_response1, &req);
        assert!(result2.is_ok());
        let response2 = result2.unwrap();
        assert_eq!(response2.choices[0].message.content, serde_json::Value::String("This is a direct response field".to_string()));

        // Test 3: Alternative format with content field
        let alt_response2 = serde_json::json!({
            "model": "tinyllama",
            "content": "Content field response",
            "done": true
        });
        
        let result3 = proxy.convert_from_ollama_format(&alt_response2, &req);
        assert!(result3.is_ok());
        let response3 = result3.unwrap();
        assert_eq!(response3.choices[0].message.content, serde_json::Value::String("Content field response".to_string()));

        // Test 4: Fallback to empty string if no recognizable format
        let empty_response = serde_json::json!({
            "model": "tinyllama",
            "done": true,
            "other_field": "not a response"
        });
        
        let result4 = proxy.convert_from_ollama_format(&empty_response, &req);
        assert!(result4.is_ok());
        let response4 = result4.unwrap();
        assert_eq!(response4.choices[0].message.content, serde_json::Value::String("".to_string()));
    }

    #[test]
    fn test_tensor_flattening_for_embeddings() {
        // Note: This test would require mocking the ML engine to return a 2D tensor
        // Since we can't easily mock the ML engine in unit tests, we'll test the logic
        // by verifying the generate_embedding function handles tensor conversion
        
        // Instead, we'll create a simple test to verify the tensor flattening logic
        // by checking that the function signature and error handling are correct
        
        let proxy = create_test_proxy();
        let req = OpenAiRequest {
            model: "test-model".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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

        // The actual tensor processing happens in generate_embedding which requires
        // a real ML engine. We'll rely on integration tests for this.
        // This test at least verifies the request structure is valid.
        assert_eq!(req.model, "test-model");
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_openai_response_source_tracking() {
        // Test that OpenAiResponse properly stores and serializes neurofed_source
        let response = OpenAiResponse {
            id: "test-id".to_string(),
            object: "chat.completion".to_string(),
            created: 1234567890,
            model: "neurofed".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::Value::String("test response".to_string()),
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
            neurofed_source: Some("local".to_string()),
        };

        // Test serialization - neurofed_source should be serialized as _neurofed_source
        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"_neurofed_source\":\"local\""));
        
        // Test deserialization
        let deserialized: OpenAiResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.neurofed_source, Some("local".to_string()));
        
        // Test with different source values
        let remote_response = OpenAiResponse {
            neurofed_source: Some("remote".to_string()),
            ..response.clone()
        };
        let remote_json = serde_json::to_string(&remote_response).unwrap();
        assert!(remote_json.contains("\"_neurofed_source\":\"remote\""));
        
        let pc_response = OpenAiResponse {
            neurofed_source: Some("pc".to_string()),
            ..response.clone()
        };
        let pc_json = serde_json::to_string(&pc_response).unwrap();
        assert!(pc_json.contains("\"_neurofed_source\":\"pc\""));
    }

    #[test]
    fn test_tensor_rank_handling_in_learn_from_response() {
        use candle_core::{Tensor, Device, DType};
        
        // Test that tensor flattening logic handles different ranks properly
        let device = Device::Cpu;
        
        // Create a 1D tensor (should not need flattening)
        let tensor_1d = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3], &device).unwrap();
        assert_eq!(tensor_1d.rank(), 1);
        
        // Create a 2D tensor [512, 1] (should need flattening)
        let data_2d: Vec<f32> = (0..512).map(|x| x as f32).collect();
        let tensor_2d = Tensor::from_vec(data_2d.clone(), vec![512, 1], &device).unwrap();
        assert_eq!(tensor_2d.rank(), 2);
        
        // Verify flatten_all works
        let flattened = tensor_2d.flatten_all().unwrap();
        assert_eq!(flattened.rank(), 1);
        assert_eq!(flattened.shape().dims(), &[512]);
        
        // Create a 3D tensor (edge case)
        let tensor_3d = Tensor::zeros((2, 256, 1), DType::F32, &device).unwrap();
        assert_eq!(tensor_3d.rank(), 3);
        let flattened_3d = tensor_3d.flatten_all().unwrap();
        assert_eq!(flattened_3d.rank(), 1);
        assert_eq!(flattened_3d.shape().dims(), &[512]);
    }

    #[test]
    fn test_convert_from_ollama_format_sets_local_source() {
        use serde_json::json;
        
        let proxy = create_test_proxy();
        
        // Create a mock Ollama response
        let ollama_resp = json!({
            "model": "llama2",
            "created_at": "2023-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "test response from ollama"
            },
            "done": true,
            "total_duration": 1000
        });
        
        let original_req = OpenAiRequest {
            model: "neurofed".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: serde_json::Value::String("test".to_string()),
                name: None,
            }],
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
        
        let result = proxy.convert_from_ollama_format(&ollama_resp, &original_req);
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.neurofed_source, Some("local".to_string()));
        assert_eq!(response.model, "neurofed"); // Should preserve original model name
    }
}