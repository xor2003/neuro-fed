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
            model_info: None,
            tokenizer_info: None,
            config_info: None,
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
            warn!("No model loaded, returning dummy tensor for compatibility");
            // Create dummy tensor with size 512 to match PC hierarchy top level dimension
            // Must be 2D tensor with shape (512, 1) for PC hierarchy compatibility
            let data: Vec<f32> = vec![0.0; 512];
            return Ok(Tensor::from_slice(&data, (512, 1), &self.device)
                .map_err(|e| MLError::ModelLoadError(format!("Failed to create dummy tensor: {}", e)))?);
        }
        
        // Simple tokenization (placeholder)
        let tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        if tokens.is_empty() {
            return Err(MLError::InvalidResponse("Empty tokenization".to_string()));
        }
        
        // Convert to tensor
        let _input_tensor = Tensor::from_slice(&tokens, (1, tokens.len()), &self.device)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to create input tensor: {}", e)))?;
        
        // Simulate model inference by creating random embeddings
        // In a real implementation, this would run the actual model
        let hidden_size = self.config_info.as_ref()
            .map(|c| c.hidden_size)
            .unwrap_or(768);
        
        let embeddings = Tensor::randn(0.0, 1.0, (1, tokens.len(), hidden_size), &self.device)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to create random tensor: {}", e)))?;
        
        // Mean pooling across sequence dimension
        let embeddings = embeddings.mean(1)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to compute mean: {}", e)))?;
        
        // Squeeze batch dimension
        let embeddings = embeddings.squeeze(0)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to squeeze tensor: {}", e)))?;
        
        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.insert(cache_key, embeddings.clone());
        }
        
        Ok(embeddings)
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
        // Should return tensor of shape [512, 1] for PC hierarchy compatibility
        assert_eq!(result.shape().dims(), &[512, 1]);
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
        // Test that dummy tensor has correct shape (512, 1) for PC hierarchy compatibility
        // (Matches PC config dim_per_level top level = 512)
        let device_type = DeviceType {
            name: "CPU".to_string(),
            description: "".to_string(),
            supported: true,
        };
        let engine = MLEngine::new("test-model", device_type).unwrap();
        let result = engine.process_text("test").await.unwrap();
        assert_eq!(result.shape().dims(), &[512, 1], "Dummy tensor should have shape (512, 1) for PC hierarchy compatibility");
        
        // Verify tensor has correct number of elements
        let total_elements: usize = result.shape().dims().iter().product();
        assert_eq!(total_elements, 512, "Dummy tensor should have 512 elements");
    }
}