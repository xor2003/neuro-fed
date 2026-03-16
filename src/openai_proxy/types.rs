// src/openai_proxy/types.rs
use crate::types::{FunctionCall, Tool, ToolCall};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAiRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing, skip_deserializing)]
    pub api_key: Option<String>,
}

/// Response structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: Option<String>,
    pub logprobs: Option<LogProbs>,
}

/// Message structure for OpenAI API
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
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

impl OpenAiResponse {
    /// Create an error response with a given error message
    pub fn error(_message: &str) -> Self {
        Self {
            id: "error".to_string(),
            object: "error".to_string(),
            created: 0,
            model: "neurofed-error".to_string(),
            choices: vec![],
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            neurofed_source: Some("error".to_string()),
        }
    }
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
