// src/types.rs
// Common types used across the NeuroFed Node

use std::path::PathBuf;
use std::time::SystemTime;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInput {
    pub id: String,
    pub content: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCommand {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEvent {
    pub path: PathBuf,
    pub event_type: FileEventType,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileEventType {
    Created,
    Modified,
    Deleted,
    Renamed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResponse {
    pub id: String,
    pub content: String,
    pub confidence: f32,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeError {
    pub id: String,
    pub message: String,
    pub error_type: ErrorType,
    pub timestamp: SystemTime,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    ValidationError,
    ProcessingError,
    NetworkError,
    StorageError,
    ConfigurationError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NostrEvent {
    pub id: String,
    pub content: String,
    pub author: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub device_type: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    pub port: u16,
    pub host: String,
    pub log_level: String,
    pub enable_nostr: bool,
    pub enable_openai_proxy: bool,
    pub enable_web_ui: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    pub success: bool,
    pub message: String,
    pub model_loaded: bool,
    pub config_loaded: bool,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub model_path: String,
    pub device_type: String,
    pub config_path: String,
    pub enable_gpu: bool,
    pub enable_nostr: bool,
    pub enable_openai_proxy: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCConfig {
    pub learning_rate: f32,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub dropout_rate: f32,
    pub max_sequence_length: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NostrConfig {
    pub pubkey: String,
    pub privkey: String,
    pub relay_urls: Vec<String>,
    pub enable_relay_discovery: bool,
    pub max_message_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIProxyConfig {
    pub port: u16,
    pub host: String,
    pub enable_local_fallback: bool,
    pub local_model_path: String,
    pub rate_limit: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub function: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: String,
    pub arguments: HashMap<String, String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxumPath {
    pub segments: Vec<String>,
    pub query: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyStats {
    pub requests_total: u64,
    pub requests_successful: u64,
    pub requests_failed: u64,
    pub average_response_time: f32,
    pub last_reset: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoModel {
    pub name: String,
    pub version: String,
    pub parameters: u64,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTokenizer {
    pub vocab_size: usize,
    pub max_length: usize,
    pub special_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub device_type: String,
    pub memory_total: u64,
    pub memory_available: u64,
    pub cores: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceType {
    pub name: String,
    pub description: String,
    pub supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub name: String,
    pub layer_type: String,
    pub parameters: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapError {
    pub message: String,
    pub error_type: String,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCError {
    pub message: String,
    pub error_type: String,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveCoding {
    pub layers: Vec<Layer>,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub learning_rate: f32,
    pub dropout_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NostrFederation {
    pub config: NostrConfig,
    pub connected: bool,
    pub relay_count: usize,
    pub message_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIProxy {
    pub config: OpenAIProxyConfig,
    pub running: bool,
    pub request_count: u64,
    pub error_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub node: NodeConfig,
    pub model: ModelConfig,
    pub nostr: NostrConfig,
    pub openai: OpenAIProxyConfig,
    pub pc: PCConfig,
}

// Added MLError type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLError {
    ModelLoadError(String),
    RequestError(String),
    SerializationError(String),
    InvalidResponse(String),
    ConfigurationError(String),
}