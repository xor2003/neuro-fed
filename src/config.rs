// src/config.rs
// Configuration management for NeuroPC Node

use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::privacy_networks::PrivacyNetworkConfig;

// Re-export config crate for convenience
pub use config;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeConfig {
    pub model_path: String,
    pub context_size: usize,
    pub bootstrap_on_start: bool,
    pub nostr_config: NostrConfig,
    pub pc_config: PCConfig,
    pub bootstrap_config: BootstrapConfig,
    pub web_ui_enabled: bool,
    pub log_level: String,
    pub ml_config: MLConfig,
    pub proxy_config: BackendConfig,
    pub wallet_address: Option<String>,
    pub brain_sharing_config: BrainSharingConfig,
    pub federation_config: FederationConfig,
    pub privacy_config: PrivacyNetworkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MLConfig {
    pub device_type: String,
    pub max_batch_size: usize,
    pub embedding_dim: usize,
    pub use_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BackendConfig {
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub openai_model: String,
    pub ollama_base_url: String,
    pub ollama_model: String,
    pub local_fallback_enabled: bool,
    pub tool_bypass_enabled: bool,
    pub semantic_cache_enabled: bool,
    pub semantic_similarity_threshold: f32,
    pub pc_inference_enabled: bool,
    pub pc_learning_enabled: bool,
    pub max_cache_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NostrConfig {
    pub relay_urls: Vec<String>,
    pub public_key: String,
    pub private_key: String,
    pub max_batch_size: usize,
    pub publish_interval: u64, // seconds
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BrainSharingConfig {
    /// Whether brain sharing is enabled.
    pub enabled: bool,
    /// Relay URLs for brain sharing (overrides NostrConfig if non‑empty).
    pub relay_urls: Vec<String>,
    /// Directory where brains are stored locally.
    pub brain_storage_dir: std::path::PathBuf,
    /// Directory for caching downloaded brains.
    pub cache_dir: std::path::PathBuf,
    /// Base model ID of this node (for compatibility checking).
    pub base_model_id: String,
    /// Allow downloading brains from unknown authors.
    pub allow_untrusted_authors: bool,
    /// Maximum brain size in bytes.
    pub max_brain_size: u64,
}

/// Federation configuration for wallet vs. no-wallet modes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FederationConfig {
    /// Federation strategy: "wallet" or "no_wallet"
    pub strategy: String,
    /// Wallet configuration (for wallet mode)
    pub wallet: WalletConfig,
    /// Proof-of-work configuration (for no-wallet mode)
    pub pow: PoWConfig,
    /// Enable fallback between modes
    pub enable_fallback: bool,
    /// Maximum retries for federation requests
    pub max_retries: u32,
    /// Request timeout in seconds
    pub request_timeout_seconds: u64,
}

/// Wallet configuration for Nostr payments (zaps)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WalletConfig {
    /// Nostr private key (hex)
    pub private_key: String,
    /// Relay URLs for payment verification
    pub payment_relays: Vec<String>,
    /// Minimum satoshis required per request
    pub min_sats: u64,
    /// Required confirmations
    pub required_confirmations: u32,
    /// Enable automatic zap requests
    pub enable_auto_zap: bool,
}

/// Proof-of-work configuration for no-wallet mode
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoWConfig {
    /// Difficulty target (number of leading zero bits)
    pub difficulty: u32,
    /// Timeout for PoW generation in seconds
    pub timeout_seconds: u64,
    /// Hash algorithm (e.g., "sha256")
    pub hash_algorithm: String,
    /// Enable dynamic difficulty adjustment
    pub enable_dynamic_difficulty: bool,
    /// Maximum nonce value
    pub max_nonce: u64,
}

impl Default for WalletConfig {
    fn default() -> Self {
        Self {
            private_key: "".to_string(),
            payment_relays: vec!["wss://relay.damus.io".to_string()],
            min_sats: 1000,
            required_confirmations: 1,
            enable_auto_zap: false,
        }
    }
}

impl Default for PoWConfig {
    fn default() -> Self {
        Self {
            difficulty: 5,
            timeout_seconds: 30,
            hash_algorithm: "sha256".to_string(),
            enable_dynamic_difficulty: true,
            max_nonce: 1_000_000,
        }
    }
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            strategy: "wallet".to_string(),
            wallet: WalletConfig::default(),
            pow: PoWConfig::default(),
            enable_fallback: true,
            max_retries: 3,
            request_timeout_seconds: 30,
        }
    }
}

impl Default for BrainSharingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            relay_urls: vec!["wss://relay.damus.io".to_string()],
            brain_storage_dir: std::path::PathBuf::from("./brains"),
            cache_dir: std::path::PathBuf::from("./cache/brains"),
            base_model_id: "unknown".to_string(),
            allow_untrusted_authors: false,
            max_brain_size: 2 * 1024 * 1024 * 1024, // 2 GB
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PCConfig {
    pub n_levels: usize,
    pub dim_per_level: Vec<usize>,
    pub learning_rate: f32,
    pub mu_pc_scaling: bool,
    // Inference configuration
    pub inference_steps: usize,
    pub selective_update: bool,
    // Precision weighting configuration
    pub enable_precision_weighting: bool,
    pub free_energy_drop_threshold: f32,
    pub default_precision: f32,
    pub min_precision: f32,
    pub max_precision: f32,
    pub free_energy_history_size: usize,
    pub enable_code_verification: bool,
    pub enable_nostr_zap_tracking: bool,
    pub min_zaps_for_consensus: usize,
    /// Path to SQLite database for persisting PC weights (optional)
    pub persistence_db_path: Option<String>,
    /// Convergence threshold for early exiting during inference
    pub convergence_threshold: f32,
    /// Factor for hidden dimension calculation: hidden_dim = embedding_dim * factor
    pub hidden_dim_factor: f32,
    /// Threshold for surprise detection based on statistical distribution
    pub surprise_threshold: f32,
}

impl PCConfig {
    /// Create a new PCConfig with basic parameters
    /// This is a convenience constructor for backward compatibility
    pub fn new(n_levels: usize, dim_per_level: Vec<usize>) -> Self {
        Self {
            n_levels,
            dim_per_level,
            learning_rate: 0.01,
            mu_pc_scaling: true,
            inference_steps: 20,
            selective_update: true,
            enable_precision_weighting: false,
            free_energy_drop_threshold: 0.5,
            default_precision: 0.3,
            min_precision: 0.1,
            max_precision: 1.0,
            free_energy_history_size: 10,
            enable_code_verification: false,
            enable_nostr_zap_tracking: false,
            min_zaps_for_consensus: 3,
            persistence_db_path: None,
            convergence_threshold: 0.01,
            hidden_dim_factor: 0.5,
            surprise_threshold: 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BootstrapConfig {
    pub embedding_dim: usize,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub learning_rate: f32,
    pub document_paths: Vec<String>,
}

impl BootstrapConfig {
    pub fn new(embedding_dim: usize, batch_size: usize, max_epochs: usize, learning_rate: f32, document_paths: Vec<String>) -> Self {
        Self {
            embedding_dim,
            batch_size,
            max_epochs,
            learning_rate,
            document_paths,
        }
    }
}

#[derive(Debug)]
pub enum NodeError {
    ConfigParseError(String),
    FileNotFoundError(String),
    InvalidConfig(String),
    SerializationError(String),
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            NodeError::ConfigParseError(msg) => write!(f, "Config parse error: {}", msg),
            NodeError::FileNotFoundError(msg) => write!(f, "File not found: {}", msg),
            NodeError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            NodeError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
        }
    }
}

impl std::error::Error for NodeError {}

impl From<config::ConfigError> for NodeError {
    fn from(err: config::ConfigError) -> Self {
        NodeError::ConfigParseError(err.to_string())
    }
}

impl NodeConfig {
    pub fn load_from_file(path: &str) -> Result<Self, NodeError> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(NodeError::FileNotFoundError(path.display().to_string()));
        }
        
        // Use config crate which supports multiple formats (TOML, JSON, YAML, etc.)
        // Create a default configuration and merge with file
        let default_config = Self::default();
        let config_builder = config::Config::builder()
            .add_source(config::Config::try_from(&default_config)?)
            .add_source(config::File::from(path))
            .build()?;
        
        config_builder
            .try_deserialize()
            .map(|config: Self| {
                info!("Loaded config from {} (format detected automatically)", path.display());
                config
            })
            .map_err(|e| NodeError::ConfigParseError(e.to_string()))
    }
    
    /// Load configuration from "config.toml" if it exists, otherwise return default configuration.
    /// This is the recommended way to get configuration for the application.
    pub fn load_or_default() -> Self {
        const CONFIG_FILE: &str = "config.toml";
        match Self::load_from_file(CONFIG_FILE) {
            Ok(config) => {
                info!("Configuration loaded from {}", CONFIG_FILE);
                config
            }
            Err(e) => {
                warn!("Failed to load configuration from {}: {}. Using default configuration.", CONFIG_FILE, e);
                Self::default()
            }
        }
    }
    
    pub fn save_to_file(&self, path: &str) -> Result<(), NodeError> {
        let path_obj = Path::new(path);
        let config_str = if let Some(ext) = path_obj.extension() {
            if ext == "toml" {
                // Save as TOML
                toml::to_string_pretty(self)
                    .map_err(|e| NodeError::SerializationError(e.to_string()))?
            } else {
                // Default to JSON for .json or any other extension
                serde_json::to_string_pretty(self)
                    .map_err(|e| NodeError::SerializationError(e.to_string()))?
            }
        } else {
            // No extension, default to JSON
            serde_json::to_string_pretty(self)
                .map_err(|e| NodeError::SerializationError(e.to_string()))?
        };
        
        match fs::write(path, config_str) {
            Ok(_) => {
                info!("Saved config to {}", path);
                Ok(())
            }
            Err(e) => Err(NodeError::FileNotFoundError(e.to_string())),
        }
    }
    
    pub fn validate(&self) -> Result<(), NodeError> {
        if self.model_path.is_empty() {
            return Err(NodeError::InvalidConfig("Model path cannot be empty".to_string()));
        }
        
        if self.context_size == 0 {
            return Err(NodeError::InvalidConfig("Context size must be > 0".to_string()));
        }
        
        if self.nostr_config.relay_urls.is_empty() {
            warn!("No Nostr relays configured");
        }
        
        Ok(())
    }
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            model_path: "models/gguf_model.gguf".to_string(),
            context_size: 2048,
            bootstrap_on_start: true,
            nostr_config: NostrConfig::default(),
            pc_config: PCConfig::default(),
            bootstrap_config: BootstrapConfig::default(),
            web_ui_enabled: false,
            log_level: "INFO".to_string(),
            ml_config: MLConfig::default(),
            proxy_config: BackendConfig::default(),
            wallet_address: None,
            brain_sharing_config: BrainSharingConfig::default(),
            federation_config: FederationConfig::default(),
            privacy_config: PrivacyNetworkConfig::default(),
        }
    }
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            device_type: "cpu".to_string(),
            max_batch_size: 32,
            embedding_dim: 768,
            use_gpu: true,
        }
    }
}

impl Default for NostrConfig {
    fn default() -> Self {
        Self {
            relay_urls: vec!["wss://relay.damus.io".to_string()],
            public_key: "default-pub-key".to_string(),
            private_key: "default-priv-key".to_string(),
            max_batch_size: 100,
            publish_interval: 60,
        }
    }
}

impl Default for PCConfig {
    fn default() -> Self {
        Self {
            n_levels: 3,
            dim_per_level: vec![2048, 1024, 512],
            learning_rate: 0.01,
            mu_pc_scaling: true,
            // Inference configuration
            inference_steps: 20,
            selective_update: true,
            // Precision weighting defaults
            enable_precision_weighting: false,
            free_energy_drop_threshold: 0.5,
            default_precision: 0.3,
            min_precision: 0.1,
            max_precision: 1.0,
            free_energy_history_size: 10,
            enable_code_verification: false,
            enable_nostr_zap_tracking: false,
            min_zaps_for_consensus: 3,
            persistence_db_path: Some("./neurofed.db".to_string()),
            convergence_threshold: 0.01,
            hidden_dim_factor: 0.5,
            surprise_threshold: 2.0,
        }
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            openai_api_key: None,
            openai_base_url: "https://api.openai.com/v1".to_string(),
            openai_model: "gpt-4o-mini".to_string(),
            ollama_base_url: "http://localhost:11434".to_string(),
            ollama_model: "tinyllama".to_string(),
            local_fallback_enabled: true,
            tool_bypass_enabled: true,
            semantic_cache_enabled: true,
            semantic_similarity_threshold: 0.8,
            pc_inference_enabled: true,
            pc_learning_enabled: true,
            max_cache_size: 100,
        }
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 1024,
            batch_size: 32,
            max_epochs: 100,  // Increased from 10 to ensure better decoder convergence
            learning_rate: 0.001,
            document_paths: vec!["human-eval/".to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_serialization() {
        let config = NodeConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: NodeConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.model_path, deserialized.model_path);
        assert_eq!(config.context_size, deserialized.context_size);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = NodeConfig::default();
        assert!(config.validate().is_ok());
        
        config.model_path = "".to_string();
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_file_operations() {
        let config = NodeConfig::default();
        let test_path = "test_config.json";
        
        // Save config
        assert!(config.save_to_file(test_path).is_ok());
        
        // Load config
        let loaded_config = NodeConfig::load_from_file(test_path).unwrap();
        assert_eq!(config.model_path, loaded_config.model_path);
        
        // Clean up
        std::fs::remove_file(test_path).unwrap();
    }
}
