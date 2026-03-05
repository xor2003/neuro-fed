// src/ml_engine.rs
// Motor Cortex Implementation: Extracting "Eyes and Mouth" from GGUF
// This transforms the PC Brain from random noise to real language processing.

use anyhow::Result;
use candle_core::{Device, Tensor, DType};
use tokenizers::Tokenizer;
use std::path::Path;
use std::sync::Arc;
use tracing::warn;
use crate::types::{DeviceType, MLError};
use crate::model_manager::ModelManager;

pub struct MLEngine {
    token_embeddings: Tensor,
    lm_head: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    embedding_dim: usize, // Store the actual embedding dimension from the model
    model_path: String,   // Store the path to the GGUF file for later weight extraction
}

impl MLEngine {
    /// Load the Motor Cortex (Eyes and Mouth) from a GGUF file
    pub fn new(model_path: &str, _device_type: DeviceType) -> Result<Self, MLError> {
        // Default tokenizer path: tokenizer.json in same directory as model
        let tokenizer_path = Path::new(model_path).parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json")
            .to_string_lossy()
            .to_string();
        
        Self::new_with_tokenizer(model_path, &tokenizer_path, _device_type)
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

        // Ensure model is downloaded (this will also download tokenizer if needed)
        if !model_manager.is_model_downloaded(&recommended_model.name).await {
            model_manager
                .download_model(&recommended_model.name)
                .await
                .map_err(|e| MLError::ModelLoadError(e.to_string()))?;
        }

        // Get tokenizer path from model info
        let tokenizer_path = recommended_model.tokenizer_local_path
            .unwrap_or_else(|| {
                // Fallback: use tokenizer.json in same directory as model
                let model_path = Path::new(&recommended_model.local_path);
                model_path.parent()
                    .unwrap_or(Path::new("."))
                    .join("tokenizer.json")
                    .to_string_lossy()
                    .to_string()
            });

        // Use the existing constructor with the local path and tokenizer path
        // Device type is not used in current implementation; we pass a dummy.
        let device_type = DeviceType {
            name: "cpu".to_string(),
            description: "CPU".to_string(),
            supported: true,
        };
        Self::new_with_tokenizer(&recommended_model.local_path, &tokenizer_path, device_type)
    }

    /// Create an ML engine with explicit tokenizer path
    pub fn new_with_tokenizer(
        model_path: &str,
        tokenizer_path: &str,
        _device_type: DeviceType,
    ) -> Result<Self, MLError> {
        let device = Device::Cpu; // Keep it under 100MB RAM by staying on CPU/Mmap
        
        // 1. Load Tokenizer from specified path
        let tokenizer = if Path::new(tokenizer_path).exists() {
            Tokenizer::from_file(tokenizer_path)
                .map_err(|e| MLError::ModelLoadError(format!("Invalid tokenizer.json at {}: {}", tokenizer_path, e)))?
        } else {
            // Fallback: Attempt to download tokenizer if missing
            warn!("tokenizer.json not found at {}. Trying to download from HuggingFace Hub...", tokenizer_path);
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| MLError::ModelLoadError(e.to_string()))?;
            // Use TinyLlama as a generic fallback - this should work for most Llama-based models
            let repo = api.model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string());
            let tok_path = repo.get("tokenizer.json")
                .map_err(|e| MLError::ModelLoadError(format!("Failed to download tokenizer: {}", e)))?;
            Tokenizer::from_file(tok_path)
                .map_err(|e| MLError::ModelLoadError(e.to_string()))?
        };

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
            model_path: model_path.to_string(),
        })
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
        // Log tensor shapes for debugging
        let belief_shape = belief.shape();
        let lm_head_shape = self.lm_head.shape();
        tracing::debug!("decode_belief: belief shape {:?}, lm_head shape {:?}", belief_shape, lm_head_shape);

        // Check vocabulary size compatibility
        let tokenizer_vocab_size = self.tokenizer.get_vocab_size(true);
        let lm_head_vocab_size = lm_head_shape.dims()[0];
        if tokenizer_vocab_size as usize != lm_head_vocab_size {
            tracing::warn!("Vocabulary size mismatch: tokenizer={}, lm_head={}. This may cause decoding issues.",
                tokenizer_vocab_size, lm_head_vocab_size);
        }

        // 1. Project belief through LM Head to get Vocab Logits
        // belief shape should be (1, embedding_dim) or (embedding_dim, 1)
        // lm_head shape is (vocab_size, embedding_dim)
        // We need belief * lm_head^T to get (1, vocab_size)
        
        // First, ensure belief is (1, embedding_dim)
        let belief_2d = if belief_shape.rank() == 1 {
            // Reshape from (embedding_dim,) to (1, embedding_dim)
            belief.reshape((1, belief_shape.dims()[0]))
                .map_err(|e| MLError::InvalidResponse(format!("Failed to reshape belief tensor: {}", e)))?
        } else if belief_shape.rank() == 2 && belief_shape.dims()[0] == 1 {
            // Already (1, embedding_dim)
            belief.clone()
        } else if belief_shape.rank() == 2 && belief_shape.dims()[1] == 1 {
            // (embedding_dim, 1) -> transpose to (1, embedding_dim)
            belief.t()
                .map_err(|e| MLError::InvalidResponse(format!("Failed to transpose belief tensor: {}", e)))?
        } else {
            return Err(MLError::InvalidResponse(format!("Unexpected belief tensor shape: {:?}", belief_shape)));
        };

        // Transpose LM Head: (vocab_size, embedding_dim) -> (embedding_dim, vocab_size)
        let lm_head_t = self.lm_head.t()
            .map_err(|e| MLError::InvalidResponse(format!("Failed to transpose LM Head: {}", e)))?;

        // Matrix multiply: (1, embedding_dim) * (embedding_dim, vocab_size) = (1, vocab_size)
        let logits = belief_2d.matmul(&lm_head_t)
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

        // Ensure token ID is within vocabulary bounds
        if max_idx >= tokenizer_vocab_size as usize {
            tracing::warn!("Token ID {} exceeds tokenizer vocabulary size {}, clamping to {}",
                max_idx, tokenizer_vocab_size, tokenizer_vocab_size - 1);
            max_idx = tokenizer_vocab_size as usize - 1;
        }

        // 3. Convert ID back to string
        let word = self.tokenizer.decode(&[max_idx as u32], true)
            .map_err(|e| MLError::InvalidResponse(format!("Decode error: {}", e)))?;

        tracing::debug!("decode_belief: decoded token ID {} -> '{}'", max_idx, word);
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

    /// Extract a specific layer weight tensor from the GGUF file
    /// This allows loading pre-trained weights for PC hierarchy initialization
    pub fn extract_layer_weight(&self, tensor_name: &str) -> Result<Tensor, MLError> {
        // Reopen the GGUF file to extract additional tensors
        let mut file = std::fs::File::open(&self.model_path)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to open GGUF for weight extraction: {}", e)))?;
        
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| MLError::ModelLoadError(format!("GGUF Read Error for weight extraction: {}", e)))?;
        
        // Extract the requested tensor
        let tensor = content.tensor(&mut file, tensor_name, &self.device)
            .map_err(|e| MLError::ModelLoadError(format!("Missing tensor {}: {}", tensor_name, e)))?
            .dequantize(&self.device)
            .map_err(|e| MLError::ModelLoadError(format!("Dequantize Error for {}: {}", tensor_name, e)))?;
        
        tracing::info!("Extracted layer weight {} with shape {:?}", tensor_name, tensor.shape());
        Ok(tensor)
    }

    /// Get a list of available layer weight tensor names from the GGUF file
    pub fn list_layer_weights(&self) -> Result<Vec<String>, MLError> {
        let mut file = std::fs::File::open(&self.model_path)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to open GGUF for listing: {}", e)))?;
        
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| MLError::ModelLoadError(format!("GGUF Read Error for listing: {}", e)))?;
        
        // Filter for layer weights (e.g., blk.*.ffn_down.weight, blk.*.attn_q.weight, etc.)
        let layer_weights: Vec<String> = content.tensor_infos.iter()
            .filter_map(|(name, _)| {
                if name.contains("blk.") && (name.contains(".weight") || name.contains(".bias")) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        
        tracing::info!("Found {} layer weights in GGUF file", layer_weights.len());
        Ok(layer_weights)
    }

    pub fn clear_cache(&self) {}
}
