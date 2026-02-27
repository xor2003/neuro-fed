// src/ml_engine.rs
// ML Engine using candle framework for pure Rust CPU/GPU operations

use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;

use candle_core::{Device, Tensor, DeviceType};
use candle_transformers::{AutoModel, AutoTokenizer, AutoConfig};
use candle_nn::layers::{Linear, Layer};
use tracing::{info, error, debug, warn};
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// Error types for ML engine operations
#[derive(Debug, Error)]
pub enum MLError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    #[error("Tensor operation failed: {0}")]
    TensorError(String),
    #[error("Device not available: {0}")]
    DeviceError(String),
    #[error("Invalid model path: {0}")]
    InvalidPath(String),
    #[error("Embedding generation failed: {0}")]
    EmbeddingError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Configuration for ML engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    pub model_path: String,
    pub device_type: String,
    pub max_batch_size: usize,
    pub embedding_dim: usize,
    pub use_gpu: bool,
}

/// ML Engine using candle framework
#[derive(Debug)]
pub struct MLEngine {
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: Device,
    embedding_dim: usize,
    config: MLConfig,
}

impl MLEngine {
    /// Create a new ML engine instance
    pub async fn new(config: MLConfig) -> Result<Self, MLError> {
        info!("Creating ML engine with config: {:?}", config);
        
        // Detect available devices
        let device = if config.use_gpu {
            match Device::cuda_if_available() {
                Ok(device) => {
                    info!("Using CUDA device: {:?}", device);
                    device
                }
                Err(_) => {
                    warn!("CUDA not available, falling back to CPU");
                    Device::cpu()
                }
            }
        } else {
            Device::cpu()
        };
        
        // Load model and tokenizer
        let model = match AutoModel::from_pretrained(&config.model_path, &device).await {
            Ok(model) => model,
            Err(e) => return Err(MLError::ModelLoadError(e.to_string())),
        };
        
        let tokenizer = match AutoTokenizer::from_pretrained(&config.model_path).await {
            Ok(tokenizer) => tokenizer,
            Err(e) => return Err(MLError::ModelLoadError(e.to_string())),
        };
        
        // Get embedding dimension from model
        let embedding_dim = model.config().get("dim").and_then(|v| v.as_i64())
            .unwrap_or(config.embedding_dim as i64) as usize;
        
        Ok(Self {
            model,
            tokenizer,
            device,
            embedding_dim,
            config,
        })
    }
    
    /// Generate embeddings for the given text
    pub async fn generate_embedding(&self, text: &str) -> Result<Tensor, MLError> {
        debug!("Generating embedding for text: {:?}", text);
        
        // Tokenize input
        let inputs = self.tokenizer.encode(text, true).await
            .map_err(|e| MLError::EmbeddingError(e.to_string()))?;
        
        // Get model output
        let outputs = self.model.forward_t(Some(inputs), None, false).await
            .map_err(|e| MLError::EmbeddingError(e.to_string()))?;
        
        // Extract embeddings (typically from last hidden state)
        let embeddings = outputs.get("last_hidden_state")
            .ok_or_else(|| MLError::EmbeddingError("No last_hidden_state found".to_string()))?
            .clone();
        
        // Pool embeddings if needed (e.g., mean pooling)
        let pooled_embedding = embeddings.mean_dim(Some(&[1]), false, self.device.clone())
            .map_err(|e| MLError::TensorError(e.to_string()))?;
        
        Ok(pooled_embedding)
    }
    
    /// Generate embeddings for batch of texts
    pub async fn generate_batch_embeddings(&self, texts: &[&str]) -> Result<Vec<Tensor>, MLError> {
        let mut embeddings = Vec::with_capacity(texts.len());
        
        for text in texts {
            let embedding = self.generate_embedding(text).await?;
            embeddings.push(embedding);
        }
        
        Ok(embeddings)
    }
    
    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
    
    /// Get device information
    pub fn device_info(&self) -> String {
        format!("Device: {:?}, Type: {:?}, Memory: {:?}", 
                self.device.index(), 
                self.device.device_type(),
                self.device.total_memory())
    }
    
    /// Cleanup resources
    pub fn cleanup(&mut self) {
        info!("Cleaning up ML engine");
        // In candle, cleanup is handled by dropping the model
    }
    
    /// Save model state
    pub async fn save_state(&self, path: &str) -> Result<(), MLError> {
        // Save model weights if needed
        self.model.save_pretrained(path).await
            .map_err(|e| MLError::SerializationError(e.to_string()))?;
        
        Ok(())
    }
    
    /// Load model state
    pub async fn load_state(&mut self, path: &str) -> Result<(), MLError> {
        // Load model weights if needed
        self.model = match AutoModel::from_pretrained(path, &self.device).await {
            Ok(model) => model,
            Err(e) => return Err(MLError::ModelLoadError(e.to_string())),
        };
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    
    #[tokio::test]
    async fn test_engine_creation() {
        let config = MLConfig {
            model_path: "models/bert-base-uncased".to_string(),
            device_type: "cpu".to_string(),
            max_batch_size: 32,
            embedding_dim: 768,
            use_gpu: false,
        };
        
        let engine = MLEngine::new(config).await;
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_embedding_generation() {
        let config = MLConfig {
            model_path: "models/bert-base-uncased".to_string(),
            device_type: "cpu".to_string(),
            max_batch_size: 32,
            embedding_dim: 768,
            use_gpu: false,
        };
        
        let engine = MLEngine::new(config).await.unwrap();
        let embedding = engine.generate_embedding("test text").await;
        assert!(embedding.is_ok());
        
        let embedding = embedding.unwrap();
        assert_eq!(embedding.size()[0], 1);
        assert_eq!(embedding.size()[1], 768);
    }
    
    #[tokio::test]
    async fn test_batch_embeddings() {
        let config = MLConfig {
            model_path: "models/bert-base-uncased".to_string(),
            device_type: "cpu".to_string(),
            max_batch_size: 32,
            embedding_dim: 768,
            use_gpu: false,
        };
        
        let engine = MLEngine::new(config).await.unwrap();
        let texts = ["text1", "text2", "text3"];
        let embeddings = engine.generate_batch_embeddings(&texts).await;
        assert!(embeddings.is_ok());
        
        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.len(), 3);
    }
    
    #[tokio::test]
    async fn test_device_info() {
        let config = MLConfig {
            model_path: "models/bert-base-uncased".to_string(),
            device_type: "cpu".to_string(),
            max_batch_size: 32,
            embedding_dim: 768,
            use_gpu: false,
        };
        
        let engine = MLEngine::new(config).await.unwrap();
        let info = engine.device_info();
        assert!(!info.is_empty());
    }
}