// src/ml_engine.rs
// Motor Cortex Implementation: Extracting "Eyes and Mouth" from GGUF
// This transforms the PC Brain from random noise to real language processing.

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use tokenizers::Tokenizer;
use std::path::Path;
use std::sync::Arc;
use crate::types::{DeviceType, MLError};
use crate::model_manager::ModelManager;

pub struct MLEngine {
    token_embeddings: Tensor,
    lm_head: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    embedding_dim: usize, // Store the actual embedding dimension from the model
}

impl MLEngine {
    /// Load the Motor Cortex (Eyes and Mouth) from a GGUF file
    pub fn new(model_path: &str, _device_type: DeviceType) -> Result<Self, MLError> {
        let device = Device::Cpu; // Keep it under 100MB RAM by staying on CPU/Mmap
        
        // 1. Load Tokenizer (Essential for "Eyes")
        // We expect tokenizer.json to be in the same directory or we use a default
        let tokenizer_path = Path::new(model_path).parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json");
        
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MLError::ModelLoadError(format!("Missing tokenizer.json: {}", e)))?;

        // 2. Open GGUF File
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to open GGUF: {}", e)))?;
        
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| MLError::ModelLoadError(format!("GGUF Read Error: {}", e)))?;

        // 3. Extract "The Eyes" (Token Embeddings)
        // In TinyLlama GGUF, this is usually "token_embd.weight"
        let token_embeddings = content.tensor(&mut file, "token_embd.weight", &device)
            .map_err(|e| MLError::ModelLoadError(format!("Missing token_embd.weight: {}", e)))?
            .dequantize(&device)
            .map_err(|e| MLError::ModelLoadError(format!("Dequantize Error: {}", e)))?;

        // 4. Extract "The Mouth" (LM Head)
        // In TinyLlama GGUF, this is usually "output.weight"
        let lm_head = content.tensor(&mut file, "output.weight", &device)
            .map_err(|e| MLError::ModelLoadError(format!("Missing output.weight: {}", e)))?
            .dequantize(&device)
            .map_err(|e| MLError::ModelLoadError(format!("Dequantize Error: {}", e)))?;

        // 5. Extract embedding dimension from GGUF metadata or tensor shape
        let embedding_dim = Self::extract_embedding_dim(&content, &token_embeddings)
            .unwrap_or_else(|| {
                tracing::warn!("Could not extract embedding dimension from GGUF metadata, using tensor shape inference");
                // Fallback: use token_embeddings shape [vocab_size, embedding_dim]
                let shape = token_embeddings.shape();
                let dims = shape.dims();
                if dims.len() >= 2 {
                    dims[1]
                } else {
                    2048 // Ultimate fallback
                }
            });

        tracing::info!("Motor Cortex Loaded: Embeddings {:?}, LM Head {:?}, Embedding Dim: {}",
            token_embeddings.shape(), lm_head.shape(), embedding_dim);

        Ok(Self {
            token_embeddings,
            lm_head,
            tokenizer,
            device,
            embedding_dim,
        })
    }

    /// Create an ML engine using a ModelManager to handle model selection and downloading
    pub async fn new_with_manager(
        model_manager: Arc<ModelManager>,
        model_name: &str,
    ) -> Result<Self, MLError> {
        // Get recommended model from manager
        let recommended_model = model_manager
            .get_recommended_model()
            .await
            .map_err(|e| MLError::ModelLoadError(e.to_string()))?;

        // Ensure model is downloaded
        if !model_manager.is_model_downloaded(&recommended_model.name).await {
            model_manager
                .download_model(&recommended_model.name)
                .await
                .map_err(|e| MLError::ModelLoadError(e.to_string()))?;
        }

        // Use the existing constructor with the local path
        // Device type is not used in current implementation; we pass a dummy.
        let device_type = DeviceType {
            name: "cpu".to_string(),
            description: "CPU".to_string(),
            supported: true,
        };
        Self::new(&recommended_model.local_path, device_type)
    }

    /// Extract embedding dimension from GGUF metadata or tensor shape
    fn extract_embedding_dim(content: &candle_core::quantized::gguf_file::Content, token_embeddings: &candle_core::Tensor) -> Option<usize> {
        // Try to get embedding dimension from metadata
        // Common GGUF metadata keys for embedding dimension
        let metadata = &content.metadata;
        
        // Try llama.embedding_length (common in Llama-based models)
        if let Some(value) = metadata.get("llama.embedding_length") {
            // Try to extract as u64
            match value {
                candle_core::quantized::gguf_file::Value::U64(dim) => return Some(*dim as usize),
                candle_core::quantized::gguf_file::Value::I64(dim) => return Some(*dim as usize),
                candle_core::quantized::gguf_file::Value::U32(dim) => return Some(*dim as usize),
                candle_core::quantized::gguf_file::Value::I32(dim) => return Some(*dim as usize),
                _ => {}
            }
        }
        
        // Try general.architecture and related fields
        if let Some(value) = metadata.get("general.architecture") {
            // Check if it's a string value
            if let candle_core::quantized::gguf_file::Value::String(arch_str) = value {
                // For llama-based models, try hidden_size
                if arch_str.contains("llama") {
                    if let Some(hidden_size) = metadata.get("llama.hidden_size") {
                        match hidden_size {
                            candle_core::quantized::gguf_file::Value::U64(dim) => return Some(*dim as usize),
                            candle_core::quantized::gguf_file::Value::I64(dim) => return Some(*dim as usize),
                            candle_core::quantized::gguf_file::Value::U32(dim) => return Some(*dim as usize),
                            candle_core::quantized::gguf_file::Value::I32(dim) => return Some(*dim as usize),
                            _ => {}
                        }
                    }
                }
            }
        }
        
        // Try hidden_size directly
        if let Some(value) = metadata.get("hidden_size") {
            match value {
                candle_core::quantized::gguf_file::Value::U64(dim) => return Some(*dim as usize),
                candle_core::quantized::gguf_file::Value::I64(dim) => return Some(*dim as usize),
                candle_core::quantized::gguf_file::Value::U32(dim) => return Some(*dim as usize),
                candle_core::quantized::gguf_file::Value::I32(dim) => return Some(*dim as usize),
                _ => {}
            }
        }
        
        // Fallback: use token_embeddings shape [vocab_size, embedding_dim]
        let shape = token_embeddings.shape();
        let dims = shape.dims();
        if dims.len() >= 2 {
            Some(dims[1])
        } else {
            None
        }
    }

    /// "The Eyes": Convert text into a semantic vector for the PC Brain
    pub async fn process_text(&self, text: &str) -> Result<Tensor, MLError> {
        let tokens = self.tokenizer.encode(text, true)
            .map_err(|e| MLError::InvalidResponse(format!("Tokenization error: {}", e)))?;
        
        let token_ids = tokens.get_ids();
        if token_ids.is_empty() {
            return Ok(Tensor::zeros((1, self.embedding_dim), DType::F32, &self.device).unwrap());
        }

        // Get embeddings for all tokens
        let mut embeddings = Vec::new();
        for &id in token_ids {
            let emb = self.token_embeddings.get(id as usize)
                .map_err(|e| MLError::InvalidResponse(format!("Embedding lookup error: {}", e)))?;
            embeddings.push(emb);
        }

        // Mean pool to get a single "thought vector"
        let stacked = Tensor::stack(&embeddings, 0)
            .map_err(|e| MLError::InvalidResponse(format!("Stack error: {}", e)))?;
        
        let mean_emb = stacked.mean(0)
            .map_err(|e| MLError::InvalidResponse(format!("Mean pool error: {}", e)))?;

        Ok(mean_emb.reshape((1, self.embedding_dim)).unwrap())
    }

    /// "The Mouth": Convert a PC Brain "belief" vector back into English tokens
    pub fn decode_belief(&self, belief: &Tensor) -> Result<String, MLError> {
        // 1. Project belief (2048) through LM Head to get Vocab Logits
        // lm_head is (vocab_size, 2048)
        let logits = belief.matmul(&self.lm_head.t().map_err(|e| MLError::InvalidResponse(e.to_string()))?)
            .map_err(|e| MLError::InvalidResponse(format!("LM Head projection failed: {}", e)))?;

        // 2. Sample the most likely token (Greedy for now)
        let logits_vec = logits.squeeze(0)
            .map_err(|e| MLError::InvalidResponse(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| MLError::InvalidResponse(e.to_string()))?;

        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // 3. Convert ID back to string
        let word = self.tokenizer.decode(&[max_idx as u32], true)
            .map_err(|e| MLError::InvalidResponse(format!("Decode error: {}", e)))?;

        Ok(word)
    }

    /// Get the embedding dimension detected from the model
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn get_model_info(&self) -> std::collections::HashMap<String, String> {
        let mut info = std::collections::HashMap::new();
        info.insert("motor_cortex".to_string(), "active".to_string());
        info.insert("embedding_dim".to_string(), self.embedding_dim.to_string());
        info.insert("vocab_size".to_string(), self.tokenizer.get_vocab_size(true).to_string());
        info
    }

    pub fn clear_cache(&self) {}
}
