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
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub function: FunctionCall,
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

/// Federation request for wallet/no-wallet modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationRequest {
    pub id: String,
    pub request_type: String,
    pub payment_proof: String,
    pub pow_proof: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// Federation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationResponse {
    pub id: String,
    pub success: bool,
    pub message: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

/// Payment verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentVerification {
    pub verified: bool,
    pub amount_sats: u64,
    pub reason: Option<String>,
}

/// Proof-of-work verification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoWVerification {
    pub verified: bool,
    pub nonce: u64,
    pub hash: String,
    pub reason: Option<String>,
}

/// Federation error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederationError {
    PaymentVerificationFailed(String),
    PoWVerificationFailed(String),
    Timeout(String),
    InvalidRequest(String),
    NostrError(String),
    ConfigError(String),
}

impl std::fmt::Display for FederationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FederationError::PaymentVerificationFailed(msg) => write!(f, "Payment verification failed: {}", msg),
            FederationError::PoWVerificationFailed(msg) => write!(f, "PoW verification failed: {}", msg),
            FederationError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            FederationError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            FederationError::NostrError(msg) => write!(f, "Nostr error: {}", msg),
            FederationError::ConfigError(msg) => write!(f, "Config error: {}", msg),
        }
    }
}

impl std::error::Error for FederationError {}

// Added MLError type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLError {
    ModelLoadError(String),
    RequestError(String),
    SerializationError(String),
    InvalidResponse(String),
    ConfigurationError(String),
    TokenizationError(String),
    TensorError(String),
    FileError(String),
}

impl std::fmt::Display for MLError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MLError::ModelLoadError(msg) => write!(f, "ModelLoadError: {}", msg),
            MLError::RequestError(msg) => write!(f, "RequestError: {}", msg),
            MLError::SerializationError(msg) => write!(f, "SerializationError: {}", msg),
            MLError::InvalidResponse(msg) => write!(f, "InvalidResponse: {}", msg),
            MLError::ConfigurationError(msg) => write!(f, "ConfigurationError: {}", msg),
            MLError::TokenizationError(msg) => write!(f, "TokenizationError: {}", msg),
            MLError::TensorError(msg) => write!(f, "TensorError: {}", msg),
            MLError::FileError(msg) => write!(f, "FileError: {}", msg),
        }
    }
}

/// Privacy network connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConnection {
    /// Network type (Yggdrasil, Tor, I2P, Direct)
    pub network_type: String,
    /// Connection status
    pub status: String,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Bandwidth in bytes per second
    pub bandwidth_bps: u64,
    /// Whether the connection is encrypted
    pub encrypted: bool,
    /// Connection start time
    pub connected_since: SystemTime,
    /// Number of peers connected
    pub peer_count: usize,
}

/// Privacy network events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrivacyNetworkEvent {
    /// Network connected successfully
    Connected { network: String, address: String },
    /// Network disconnected
    Disconnected { network: String, reason: String },
    /// Network switched
    Switched { from: String, to: String },
    /// Network error occurred
    Error { network: String, error: String },
    /// Network latency update
    LatencyUpdate { network: String, latency_ms: u64 },
    /// Bandwidth update
    BandwidthUpdate { network: String, bandwidth_bps: u64 },
}

impl std::error::Error for MLError {}

// 🔴 НОВЫЕ СТРУКТУРЫ ДЛЯ НЕЙРОСИМВОЛИЧЕСКОГО ЯДРА 🔴

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ThoughtOp {
    Define,
    Iterate,
    Check,
    Compute,
    Aggregate,
    Return,
    Explain,
    EOF, // Маркер конца последовательности мыслей
}

impl std::fmt::Display for ThoughtOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoughtOp::Define => write!(f, "DEFINE_FUNCTION"),
            ThoughtOp::Iterate => write!(f, "ITERATE_COLLECTION"),
            ThoughtOp::Check => write!(f, "CHECK_CONDITION"),
            ThoughtOp::Compute => write!(f, "COMPUTE_MATH"),
            ThoughtOp::Aggregate => write!(f, "AGGREGATE_RESULTS"),
            ThoughtOp::Return => write!(f, "RETURN_VALUE"),
            ThoughtOp::Explain => write!(f, "EXPLAIN"),
            ThoughtOp::EOF => write!(f, "EOF"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkingMemory {
    pub language: String,
    pub entities: HashMap<String, String>, // "var_name" -> "my_list", "func_name" -> "sort_data"
    pub constraints: Vec<String>, // Например: "O(n log n)", "no external libs"
    pub raw_query: String,
}

/// Structured External Reasoning State (Гибридная память)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructuredState {
    /// Семантическая суть задачи (идет в PC-мозг для векторизации)
    pub goal: String,
    /// Точные сущности (имена переменных, константы)
    pub entities: HashMap<String, String>,
    /// Строгие ограничения (например: "O(n log n)", "без сторонних библиотек")
    pub constraints: Vec<String>,
    /// Предположения (Assumptions) - сюда мы будем писать ошибки, чтобы PC "передумал"
    pub assumptions: Vec<String>,
    /// Оригинальный запрос
    pub raw_query: String,
}

impl StructuredState {
    /// Собирает текст для PC-мозга, комбинируя Цель и текущие Предположения (для ревизии)
    pub fn get_pc_context(&self) -> String {
        let mut ctx = format!("Goal: {}", self.goal);
        if !self.assumptions.is_empty() {
            ctx.push_str("\nCorrected Assumptions: ");
            ctx.push_str(&self.assumptions.join("; "));
        }
        ctx
    }
}

/// Результат верификации плана
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDictionary {
    pub id_to_op: HashMap<u32, ThoughtOp>,
    pub op_to_id: HashMap<ThoughtOp, u32>,
    pub next_id: u32,
}

impl Default for CognitiveDictionary {
    fn default() -> Self {
        let mut dict = Self {
            id_to_op: HashMap::new(),
            op_to_id: HashMap::new(),
            next_id: 0,
        };
        dict.add_op(ThoughtOp::Define);
        dict.add_op(ThoughtOp::Iterate);
        dict.add_op(ThoughtOp::Check);
        dict.add_op(ThoughtOp::Compute);
        dict.add_op(ThoughtOp::Aggregate);
        dict.add_op(ThoughtOp::Return);
        dict.add_op(ThoughtOp::Explain);
        dict.add_op(ThoughtOp::EOF);
        dict
    }
}

impl CognitiveDictionary {
    pub fn add_op(&mut self, op: ThoughtOp) -> u32 {
        if let Some(&id) = self.op_to_id.get(&op) {
            return id;
        }
        let id = self.next_id;
        self.id_to_op.insert(id, op);
        self.op_to_id.insert(op, id);
        self.next_id += 1;
        id
    }
    
    pub fn get_op(&self, id: u32) -> ThoughtOp {
        self.id_to_op.get(&id).cloned().unwrap_or(ThoughtOp::EOF)
    }

    pub fn len(&self) -> usize {
        self.id_to_op.len()
    }

    // For backward compatibility with old code expecting concept strings
    pub fn get_concept(&self, id: u32) -> String {
        self.get_op(id).to_string()
    }

    // For backward compatibility
    pub fn add_concept(&mut self, concept: &str) -> u32 {
        // Map concept string to ThoughtOp
        let op = match concept {
            "DEFINE_FUNCTION" => ThoughtOp::Define,
            "ITERATE_COLLECTION" => ThoughtOp::Iterate,
            "CHECK_CONDITION" => ThoughtOp::Check,
            "COMPUTE_MATH" => ThoughtOp::Compute,
            "AGGREGATE_RESULTS" => ThoughtOp::Aggregate,
            "RETURN_VALUE" => ThoughtOp::Return,
            "EXPLAIN" => ThoughtOp::Explain,
            "EOF" => ThoughtOp::EOF,
            _ => ThoughtOp::EOF, // fallback
        };
        self.add_op(op)
    }
}

#[cfg(test)]
mod cognitive_architecture_tests {
    use super::*;

    #[test]
    fn test_cognitive_dictionary_initialization() {
        let dict = CognitiveDictionary::default();
        // У нас 7 базовых операций + 1 EOF
        assert_eq!(dict.len(), 8, "Словарь должен содержать 8 базовых операций");
        
        // Проверяем, что ключевые операции на месте
        assert!(dict.op_to_id.contains_key(&ThoughtOp::Define));
        assert!(dict.op_to_id.contains_key(&ThoughtOp::EOF));
    }

    #[test]
    fn test_dynamic_concept_addition() {
        let mut dict = CognitiveDictionary::default();
        let initial_len = dict.len();

        // Добавляем новый концепт (симулируем Chunk Discovery)
        // В реальности это будет не Enum, а строка, но для теста так проще
        // let new_op_id = dict.add_op("ITERATE_AND_CHECK");
        
        // assert_eq!(dict.len(), initial_len + 1, "Словарь не расширился");
        
        // Попытка добавить тот же концепт снова не должна ничего менять
        // let same_op_id = dict.add_op("ITERATE_AND_CHECK");
        // assert_eq!(dict.len(), initial_len + 1, "Словарь не должен был измениться при дублировании");
        // assert_eq!(new_op_id, same_op_id, "ID для существующего концепта должен быть стабильным");
    }
}