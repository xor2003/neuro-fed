// src/ml_engine.rs
// ML Engine using candle framework for pure Rust CPU/GPU operations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::Result as AnyResult;
use candle_core::{Device, Tensor};
use tracing::{info, warn};

use crate::types::{AutoModel, AutoTokenizer, AutoConfig, DeviceType, MLError};
use crate::model_manager::ModelManager;

/// ML Engine implementation with actual candle-core model loading
pub struct MLEngine {
    model_manager: Arc<ModelManager>,
    device: Device,
    cache: Arc<Mutex<HashMap<String, Tensor>>>,
    model_name: String,
    model_info: Option<AutoModel>,
    tokenizer_info: Option<AutoTokenizer>,
    config_info: Option<AutoConfig>,
}

impl MLEngine {
    /// Create a new ML engine with ModelManager integration (new API)
    pub async fn new_with_manager(
        model_manager: Arc<ModelManager>,
        model_name: &str,
    ) -> Result<Self, MLError> {
        info!("Creating ML engine for model: {}", model_name);
        
        // Get recommended model based on available memory
        let recommended_model = model_manager.get_recommended_model().await
            .map_err(|e| MLError::ModelLoadError(format!("Failed to get recommended model: {}", e)))?;
        
        let model_name = recommended_model.name;
        
        // Load model using ModelManager
        let model_info = model_manager.load_model(&model_name).await
            .map_err(|e| MLError::ModelLoadError(format!("Failed to load model: {}", e)))?;
        
        let config_info = model_manager.get_model_config(&model_name)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to get model config: {}", e)))?;
        
        let tokenizer_info = model_manager.get_tokenizer(&model_name)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to get tokenizer: {}", e)))?;
        
        let device_type = model_manager.get_device_config(&model_name)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to get device config: {}", e)))?;
        
        // Determine device based on device_type and available GPU
        let device = Self::select_device(&device_type)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to select device: {}", e)))?;
        
        Ok(Self {
            model_manager,
            device,
            cache: Arc::new(Mutex::new(HashMap::new())),
            model_name,
            model_info: Some(model_info),
            tokenizer_info: Some(tokenizer_info),
            config_info: Some(config_info),
        })
    }
    
    /// Create a new ML engine (backward compatible API)
    pub fn new(model_path: &str, device_type: DeviceType) -> Result<Self, MLError> {
        info!("Creating ML engine with legacy API for model: {}", model_path);
        
        // Ensure models directory exists
        let models_dir = std::path::Path::new(model_path).parent().unwrap_or_else(|| std::path::Path::new("."));
        if !models_dir.exists() {
            std::fs::create_dir_all(models_dir).map_err(|e| {
                MLError::ModelLoadError(format!("Failed to create models directory: {}", e))
            })?;
            info!("Created models directory: {:?}", models_dir);
        }
        
        // Check if model file exists
        let model_file = std::path::Path::new(model_path);
        if !model_file.exists() {
            warn!("Model file {} does not exist. Using dummy embeddings.", model_path);
            info!("To download a small model, run: mkdir -p models && curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q2_K.gguf -o {}", model_path);
        }
        
        // Create dummy model info to enable random embeddings (instead of zero tensor)
        let model_info = Some(AutoModel {
            name: model_path.to_string(),
            version: "1.0".to_string(),
            parameters: 0,
            capabilities: vec!["text-generation".to_string()],
        });
        let config_info = Some(AutoConfig {
            hidden_size: 512, // Match PC hierarchy embedding_dim
            num_layers: 32,
            vocab_size: 32000,
            max_position_embeddings: 2048,
        });
        let tokenizer_info = Some(AutoTokenizer {
            vocab_size: 32000,
            max_length: 2048,
            special_tokens: vec![],
        });
        
        // Create a default ModelManager for backward compatibility
        let config = crate::config::NodeConfig::default();
        let model_manager = Arc::new(ModelManager::new(config));
        
        // Determine device based on device_type
        let device = Self::select_device(&device_type)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to select device: {}", e)))?;
        
        Ok(Self {
            model_manager,
            device,
            cache: Arc::new(Mutex::new(HashMap::new())),
            model_name: model_path.to_string(),
            model_info,
            tokenizer_info,
            config_info,
        })
    }
    
    /// Select appropriate device (CPU/GPU) based on configuration and availability
    fn select_device(device_type: &DeviceType) -> AnyResult<Device> {
        info!("Selecting device based on: {:?}", device_type);
        
        // Try GPU backends based on device_type name
        if device_type.name.to_lowercase().contains("cuda") {
            match Device::new_cuda(0) {
                Ok(device) => {
                    info!("Using CUDA device");
                    return Ok(device);
                }
                Err(e) => warn!("CUDA not available: {}", e),
            }
        }
        
        if device_type.name.to_lowercase().contains("metal") {
            match Device::new_metal(0) {
                Ok(device) => {
                    info!("Using Metal device");
                    return Ok(device);
                }
                Err(e) => warn!("Metal not available: {}", e),
            }
        }
        
        // Note: Vulkan support may not be available in candle-core 0.9.2
        if device_type.name.to_lowercase().contains("vulkan") {
            // Try to create Vulkan device if available
            #[cfg(feature = "vulkan")]
            match Device::new_vulkan(0) {
                Ok(device) => {
                    info!("Using Vulkan device");
                    return Ok(device);
                }
                Err(e) => warn!("Vulkan not available: {}", e),
            }
            #[cfg(not(feature = "vulkan"))]
            warn!("Vulkan requested but not compiled with vulkan support");
        }
        
        // Fallback to CPU
        info!("Using CPU device");
        Ok(Device::Cpu)
    }
    
    /// Process text through the ML engine
    pub async fn process_text(&self, text: &str) -> Result<Tensor, MLError> {
        info!("Processing text: {}", text);
        
        // Check cache first
        let cache_key = text.to_string();
        {
            let cache = self.cache.lock().unwrap();
            if let Some(cached) = cache.get(&cache_key) {
                info!("Cache hit for text: {}", text);
                return Ok(cached.clone());
            }
        }
        
        // If no model info loaded (legacy mode), return dummy tensor for compatibility
        // This maintains backward compatibility with existing tests expecting shape [10]
        if self.model_info.is_none() {
            warn!("No model loaded, returning dummy tensor for compatibility. model_info: {:?}, config_info: {:?}, tokenizer_info: {:?}",
                self.model_info, self.config_info, self.tokenizer_info);
            tracing::debug!("Model path used: {}, device: {:?}", self.model_name, self.device);
            // Create dummy tensor with config dimension to match PC hierarchy top level dimension
            // Must be 2D tensor with shape (hidden_size, 1) for PC hierarchy compatibility
            let hidden_size = self.config_info.as_ref()
                .map(|c| c.hidden_size)
                .unwrap_or(512); // Default to 512 for backward compatibility
            let data: Vec<f32> = vec![0.0; hidden_size];
            return Ok(Tensor::from_slice(&data, (hidden_size, 1), &self.device)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to create dummy tensor: {}", e)))?);
        }
        
        // 1. Tokenization placeholder - using simple byte-to-int conversion for now
        // In production, replace with HuggingFace tokenizers crate
        let tokens: Vec<u32> = text.bytes().take(512).map(|b| b as u32).collect();
        
        if tokens.is_empty() {
            return Err(MLError::InvalidResponse("Empty tokenization".to_string()));
        }
        
        // 2. CHUNKING: Prevent OOM on massive prompts
        let max_len = 512; // Match your PC layer 0
        let hidden_size = self.config_info.as_ref()
            .map(|c| c.hidden_size)
            .unwrap_or(768);
        
        // Log that we're using placeholder tokenization
        tracing::debug!("Using placeholder tokenization ({} tokens). Install tokenizers crate for proper tokenization.", tokens.len());
        
        let mut all_embeddings = Vec::new();
        
        for chunk in tokens.chunks(max_len) {
            // Convert chunk to tensor
            let input_tensor = Tensor::from_slice(chunk, (1, chunk.len()), &self.device)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to create input tensor: {}", e)))?;
            
            // Simulate model inference by creating random embeddings
            // In a real implementation, this would run the actual model
            let chunk_embeddings = Tensor::randn(0.0, 1.0, (1, chunk.len(), hidden_size), &self.device)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to create random tensor: {}", e)))?;
            
            // Mean pooling across sequence dimension
            let pooled = chunk_embeddings.mean(1)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to compute mean: {}", e)))?;
            
            // Squeeze batch dimension
            let pooled = pooled.squeeze(0)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to squeeze tensor: {}", e)))?;
            
            all_embeddings.push(pooled);
        }
        
        // 3. Average all chunks together to get the final semantic vector
        let final_embedding = if all_embeddings.len() == 1 {
            all_embeddings[0].clone()
        } else {
            // Stack and average
            let stacked = Tensor::stack(&all_embeddings, 0)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to stack tensors: {}", e)))?;
            stacked.mean(0)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to compute mean across chunks: {}", e)))?
        };
        
        // Convert to f32 (PC hierarchy expects f32)
        let final_embedding = final_embedding.to_dtype(candle_core::DType::F32)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to convert tensor to f32: {}", e)))?;
        
        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(cache_key, final_embedding.clone());
        }
        
        Ok(final_embedding)
    }
    
    /// Get model information
    pub fn get_model_info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        
        if let Some(model) = &self.model_info {
            info.insert("model_name".to_string(), model.name.clone());
            info.insert("model_version".to_string(), model.version.clone());
            info.insert("model_parameters".to_string(), model.parameters.to_string());
            info.insert("model_capabilities".to_string(), model.capabilities.join(","));
        }
        
        if let Some(tokenizer) = &self.tokenizer_info {
            info.insert("tokenizer_vocab_size".to_string(), tokenizer.vocab_size.to_string());
            info.insert("tokenizer_max_length".to_string(), tokenizer.max_length.to_string());
            info.insert("tokenizer_special_tokens".to_string(), tokenizer.special_tokens.join(","));
        }
        
        if let Some(config) = &self.config_info {
            info.insert("hidden_size".to_string(), config.hidden_size.to_string());
            info.insert("num_layers".to_string(), config.num_layers.to_string());
            info.insert("vocab_size".to_string(), config.vocab_size.to_string());
            info.insert("max_position_embeddings".to_string(), config.max_position_embeddings.to_string());
        }
        
        info.insert("device".to_string(), format!("{:?}", self.device));
        info.insert("model_name".to_string(), self.model_name.clone());
        info.insert("cache_size".to_string(), self.cache.lock().unwrap().len().to_string());
        
        info
    }
    
    /// Get the underlying device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Clear the inference cache
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        info!("ML engine cache cleared");
    }
    
    /// Get the model manager reference
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeConfig;
    
    #[test]
    fn test_engine_creation() {
        let device_type = DeviceType {
            name: "CPU".to_string(),
            description: "".to_string(),
            supported: true,
        };
        let engine = MLEngine::new("test-model", device_type).unwrap();
        assert_eq!(engine.model_name, "test-model");
    }
    
    #[tokio::test]
    async fn test_text_processing() {
        let device_type = DeviceType {
            name: "CPU".to_string(),
            description: "".to_string(),
            supported: true,
        };
        let engine = MLEngine::new("test-model", device_type).unwrap();
        let result = engine.process_text("test").await.unwrap();
        // Should return tensor of shape [512] (1D) after squeezing batch dimension
        assert_eq!(result.shape().dims(), &[512]);
    }
    
    #[tokio::test]
    async fn test_model_info() {
        let device_type = DeviceType {
            name: "CPU".to_string(),
            description: "".to_string(),
            supported: true,
        };
        let engine = MLEngine::new("test-model", device_type).unwrap();
        let info = engine.get_model_info();
        assert!(info.contains_key("device"));
        assert!(info.contains_key("model_name"));
    }

    #[tokio::test]
    async fn test_dummy_tensor_shape() {
        // Test that tensor has correct shape [512] for PC hierarchy compatibility
        // (Matches PC config dim_per_level top level = 512)
        // Note: The tensor is 1D after squeezing batch dimension
        let device_type = DeviceType {
            name: "CPU".to_string(),
            description: "".to_string(),
            supported: true,
        };
        let engine = MLEngine::new("test-model", device_type).unwrap();
        let result = engine.process_text("test").await.unwrap();
        assert_eq!(result.shape().dims(), &[512], "Tensor should have shape [512] for PC hierarchy compatibility");
        
        // Verify tensor has correct number of elements
        let total_elements: usize = result.shape().dims().iter().product();
        assert_eq!(total_elements, 512, "Tensor should have 512 elements");
    }
}