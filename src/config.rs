// src/config.rs
// Configuration management for NeuroPC Node

use std::fs;
use std::path::Path;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub wallet_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub model_path: String,
    pub device_type: String,
    pub max_batch_size: usize,
    pub embedding_dim: usize,
    pub use_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NostrConfig {
    pub relay_urls: Vec<String>,
    pub public_key: String,
    pub private_key: String,
    pub max_batch_size: usize,
    pub publish_interval: u64, // seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCConfig {
    pub n_levels: usize,
    pub dim_per_level: Vec<usize>,
    pub learning_rate: f32,
    pub muPC_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapConfig {
    pub model_path: String,
    pub embedding_dim: usize,
    pub batch_size: usize,
    pub max_epochs: usize,
    pub learning_rate: f32,
    pub document_paths: Vec<String>,
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

impl NodeConfig {
    pub fn load_from_file(path: &str) -> Result<Self, NodeError> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(NodeError::FileNotFoundError(path.display().to_string()));
        }
        
        let config_str = match fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => return Err(NodeError::ConfigParseError(e.to_string())),
        };
        
        match serde_json::from_str(&config_str) {
            Ok(config) => {
                info!("Loaded config from {}", path.display());
                Ok(config)
            }
            Err(e) => Err(NodeError::ConfigParseError(e.to_string())),
        }
    }
    
    pub fn save_to_file(&self, path: &str) -> Result<(), NodeError> {
        let config_str = match serde_json::to_string_pretty(self) {
            Ok(json) => json,
            Err(e) => return Err(NodeError::SerializationError(e.to_string())),
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
            wallet_address: None,
        }
    }
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model_path: "models/bert-base-uncased".to_string(),
            device_type: "cpu".to_string(),
            max_batch_size: 32,
            embedding_dim: 768,
            use_gpu: false,
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
            muPC_scaling: false,
        }
    }
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            model_path: "models/gguf_model.gguf".to_string(),
            embedding_dim: 1024,
            batch_size: 32,
            max_epochs: 10,
            learning_rate: 0.001,
            document_paths: vec!["docs/".to_string()],
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