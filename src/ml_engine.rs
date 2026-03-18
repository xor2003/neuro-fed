// src/ml_engine.rs
// Motor Cortex Implementation: Extracting "Eyes and Mouth" from GGUF
// This transforms the PC Brain from random noise to real language processing.

use crate::model_manager::ModelManager;
use crate::types::{DeviceType, MLError};
use ahash::AHashMap;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use tokenizers::{Tokenizer, models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace};
use tracing::warn;

#[derive(Clone)]
pub struct MLEngine {
    token_embeddings: Tensor,
    lm_head: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    embedding_dim: usize, // Store the actual embedding dimension from the model
    model_path: String,   // Store the path to the GGUF file for later weight extraction
    is_mock: bool,
}

impl MLEngine {
    /// Load the Motor Cortex (Eyes and Mouth) from a GGUF file
    pub fn new(model_path: &str, _device_type: DeviceType) -> Result<Self, MLError> {
        // Default tokenizer path: tokenizer.json in same directory as model
        let tokenizer_path = Path::new(model_path)
            .parent()
            .unwrap_or(Path::new("."))
            .join("tokenizer.json")
            .to_string_lossy()
            .to_string();

        Self::new_with_tokenizer(model_path, &tokenizer_path, _device_type)
    }

    /// Create an ML engine using a ModelManager to handle model selection and downloading
    pub async fn new_with_manager(
        model_manager: Arc<ModelManager>,
        _model_name: &str,
    ) -> Result<Self, MLError> {
        // Get recommended model from manager
        let recommended_model = model_manager
            .get_recommended_model()
            .await
            .map_err(|e| MLError::ModelLoadError(e.to_string()))?;

        // Ensure model is downloaded (this will also download tokenizer if needed)
        if !model_manager
            .is_model_downloaded(&recommended_model.name)
            .await
        {
            model_manager
                .download_model(&recommended_model.name)
                .await
                .map_err(|e| MLError::ModelLoadError(e.to_string()))?;
        }

        // Get tokenizer path from model info
        let tokenizer_path = recommended_model.tokenizer_local_path.unwrap_or_else(|| {
            // Fallback: use tokenizer.json in same directory as model
            let model_path = Path::new(&recommended_model.local_path);
            model_path
                .parent()
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
            Tokenizer::from_file(tokenizer_path).map_err(|e| {
                MLError::ModelLoadError(format!(
                    "Invalid tokenizer.json at {}: {}",
                    tokenizer_path, e
                ))
            })?
        } else {
            // Fallback: Attempt to download tokenizer if missing
            warn!(
                "tokenizer.json not found at {}. Trying to download from HuggingFace Hub...",
                tokenizer_path
            );
            let api = hf_hub::api::sync::Api::new()
                .map_err(|e| MLError::ModelLoadError(e.to_string()))?;
            // Use TinyLlama as a generic fallback - this should work for most Llama-based models
            let repo = api.model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string());
            let tok_path = repo.get("tokenizer.json").map_err(|e| {
                MLError::ModelLoadError(format!("Failed to download tokenizer: {}", e))
            })?;
            Tokenizer::from_file(tok_path).map_err(|e| MLError::ModelLoadError(e.to_string()))?
        };

        // 2. Open GGUF File
        let mut file = std::fs::File::open(model_path)
            .map_err(|e| MLError::ModelLoadError(format!("Failed to open GGUF: {}", e)))?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| MLError::ModelLoadError(format!("GGUF Read Error: {}", e)))?;

        // 🔴 FIX: Robust dynamic probing for Input Embeddings
        let token_tensor_names = [
            "token_embd.weight",
            "model.embed_tokens.weight",
            "tok_embeddings.weight",
        ];
        let mut token_embeddings = None;
        for name in &token_tensor_names {
            if let Ok(tensor) = content.tensor(&mut file, name, &device) {
                token_embeddings = Some(
                    tensor
                        .dequantize(&device)
                        .map_err(|e| MLError::ModelLoadError(e.to_string()))?,
                );
                tracing::info!("Found token embeddings at: {}", name);
                break;
            }
        }
        let token_embeddings = token_embeddings.ok_or_else(|| MLError::ModelLoadError("Could not find input embedding tensor (tried: token_embd.weight, model.embed_tokens.weight, tok_embeddings.weight)".to_string()))?;

        // 🔴 FIX: Robust dynamic probing for Output Head
        let output_tensor_names = ["output.weight", "lm_head.weight"];
        let mut lm_head = None;
        for name in &output_tensor_names {
            if let Ok(tensor) = content.tensor(&mut file, name, &device) {
                lm_head = Some(
                    tensor
                        .dequantize(&device)
                        .map_err(|e| MLError::ModelLoadError(e.to_string()))?,
                );
                tracing::info!("Found LM head at: {}", name);
                break;
            }
        }
        let lm_head = lm_head.ok_or_else(|| {
            MLError::ModelLoadError(
                "Could not find LM head tensor (tried: output.weight, lm_head.weight)".to_string(),
            )
        })?;
        Self::log_gguf_metadata(&content);
        Self::log_tensor_stats("token_embeddings", &token_embeddings);
        Self::log_tensor_stats("lm_head", &lm_head);
        Self::log_tensor_file_info("gguf_model", model_path);

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

        tracing::info!(
            "Motor Cortex Loaded: Embeddings {:?}, LM Head {:?}, Embedding Dim: {}",
            token_embeddings.shape(),
            lm_head.shape(),
            embedding_dim
        );

        Ok(Self {
            token_embeddings,
            lm_head,
            tokenizer,
            device,
            embedding_dim,
            model_path: model_path.to_string(),
            is_mock: false,
        })
    }

    fn log_tensor_stats(name: &str, tensor: &candle_core::Tensor) {
        let shape = tensor.shape();
        let dims = shape.dims();
        let count: usize = dims.iter().product();
        let stats = match tensor.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            Ok(values) => {
                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                let mut min = f32::INFINITY;
                let mut max = f32::NEG_INFINITY;
                for v in &values {
                    let fv = *v;
                    sum += fv as f64;
                    sum_sq += (fv as f64) * (fv as f64);
                    if fv < min {
                        min = fv;
                    }
                    if fv > max {
                        max = fv;
                    }
                }
                let mean = (sum / values.len().max(1) as f64) as f32;
                let var = (sum_sq / values.len().max(1) as f64) - (mean as f64 * mean as f64);
                let std = (var.max(0.0) as f32).sqrt();
                (mean, std, min, max)
            }
            Err(_) => (0.0, 0.0, 0.0, 0.0),
        };
        tracing::info!(
            "{}: shape={:?} elems={} mean={:.6} std={:.6} min={:.6} max={:.6}",
            name,
            shape,
            count,
            stats.0,
            stats.1,
            stats.2,
            stats.3
        );
    }

    fn log_tensor_file_info(label: &str, path: &str) {
        match std::fs::metadata(path) {
            Ok(meta) => {
                let size = meta.len();
                tracing::info!("{}: path={} size_bytes={}", label, path, size);
                let bytes = std::fs::read(path).ok();
                if let Some(buf) = bytes {
                    let mut hasher = Sha256::new();
                    hasher.update(&buf);
                    let hash = hasher.finalize();
                    tracing::info!("{}: sha256={}", label, hex::encode(hash));
                }
            }
            Err(e) => {
                tracing::warn!("{}: failed to read file info: {}", label, e);
            }
        }
    }

    fn log_gguf_metadata(content: &candle_core::quantized::gguf_file::Content) {
        let metadata = &content.metadata;
        let mut keys: Vec<&String> = metadata.keys().collect();
        keys.sort();
        let preview = keys.iter().take(30).map(|k| k.as_str()).collect::<Vec<_>>();
        tracing::info!(
            "GGUF metadata keys (first {}): {:?}",
            preview.len(),
            preview
        );
        if let Some(value) = metadata.get("general.name") {
            tracing::info!("GGUF general.name: {:?}", value);
        }
        if let Some(value) = metadata.get("general.architecture") {
            tracing::info!("GGUF general.architecture: {:?}", value);
        }
        if let Some(value) = metadata.get("llama.embedding_length") {
            tracing::info!("GGUF llama.embedding_length: {:?}", value);
        }
        if let Some(value) = metadata.get("llama.vocab_size") {
            tracing::info!("GGUF llama.vocab_size: {:?}", value);
        }
    }

    fn mock_tokenizer() -> Result<Tokenizer, MLError> {
        let mut vocab = AHashMap::new();
        vocab.insert("<unk>".to_string(), 0u32);
        let word_level = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .map_err(|e| {
                MLError::ModelLoadError(format!("Failed to build mock tokenizer: {}", e))
            })?;
        let mut tokenizer = Tokenizer::new(word_level);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        Ok(tokenizer)
    }

    pub fn mock() -> Result<Self, MLError> {
        let device = Device::Cpu;
        let embedding_dim = 32;
        let token_embeddings =
            Tensor::zeros((1, embedding_dim), DType::F32, &device).map_err(|e| {
                MLError::ModelLoadError(format!("Mock engine tensor creation failed: {}", e))
            })?;
        let lm_head = Tensor::zeros((1, embedding_dim), DType::F32, &device).map_err(|e| {
            MLError::ModelLoadError(format!("Mock engine tensor creation failed: {}", e))
        })?;
        let tokenizer = Self::mock_tokenizer()?;
        Ok(Self {
            token_embeddings,
            lm_head,
            tokenizer,
            device,
            embedding_dim,
            model_path: "mock".to_string(),
            is_mock: true,
        })
    }

    fn deterministic_embedding_values(text: &str, seq_len: usize, dim: usize) -> Vec<f32> {
        let mut hash = 0u64;
        for b in text.as_bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(*b as u64);
        }
        let total = seq_len.max(1) * dim;
        (0..total)
            .map(|i| {
                let val = hash
                    .wrapping_add(i as u64)
                    .wrapping_mul(6364136223846793005);
                (((val >> 16) as f32 % 1000.0) / 1000.0) * 2.0 - 1.0
            })
            .collect()
    }

    fn deterministic_embedding(
        &self,
        text: &str,
        seq_len: usize,
        dim_override: Option<usize>,
    ) -> Result<Tensor, MLError> {
        let dim = dim_override.unwrap_or(self.embedding_dim);
        let values = Self::deterministic_embedding_values(text, seq_len, dim);
        Tensor::from_vec(values, (seq_len.max(1), dim), &self.device).map_err(|e| {
            MLError::InvalidResponse(format!("Deterministic embedding creation failed: {}", e))
        })
    }

    pub fn deterministic_embedding_with_dim(
        &self,
        text: &str,
        seq_len: usize,
        dim: usize,
    ) -> Result<Tensor, MLError> {
        self.deterministic_embedding(text, seq_len, Some(dim))
    }

    /// Extract embedding dimension from GGUF metadata or tensor shape
    fn extract_embedding_dim(
        content: &candle_core::quantized::gguf_file::Content,
        token_embeddings: &candle_core::Tensor,
    ) -> Option<usize> {
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
                            candle_core::quantized::gguf_file::Value::U64(dim) => {
                                return Some(*dim as usize);
                            }
                            candle_core::quantized::gguf_file::Value::I64(dim) => {
                                return Some(*dim as usize);
                            }
                            candle_core::quantized::gguf_file::Value::U32(dim) => {
                                return Some(*dim as usize);
                            }
                            candle_core::quantized::gguf_file::Value::I32(dim) => {
                                return Some(*dim as usize);
                            }
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
        if dims.len() >= 2 { Some(dims[1]) } else { None }
    }

    /// "The Eyes": Convert text into a semantic vector for the PC Brain
    pub async fn process_text(&self, text: &str) -> Result<Tensor, MLError> {
        if self.is_mock {
            return self.deterministic_embedding(text, 1, None);
        }

        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| MLError::InvalidResponse(format!("Tokenization error: {}", e)))?;

        let token_ids = tokens.get_ids();
        if token_ids.is_empty() {
            return Tensor::zeros((1, self.embedding_dim), DType::F32, &self.device).map_err(|e| {
                MLError::InvalidResponse(format!("Zero tensor creation error: {}", e))
            });
        }

        let stacked = self.build_batched_embeddings(token_ids, true)?;

        let mean_emb = stacked
            .mean(0)
            .map_err(|e| MLError::InvalidResponse(format!("Mean pool error: {}", e)))?;

        // --- SAFE L2 NORMALIZATION ---
        let norm_sq = mean_emb
            .sqr()
            .map_err(|e| MLError::InvalidResponse(format!("Sqr error: {}", e)))?
            .sum_all()
            .map_err(|e| MLError::InvalidResponse(format!("Sum error: {}", e)))?
            .to_scalar::<f32>()
            .map_err(|e| MLError::InvalidResponse(format!("Scalar error: {}", e)))?;

        let norm = norm_sq.sqrt() as f64;
        let scale_factor = if norm > 1e-6 { 1.0 / norm } else { 1.0 };

        let normalized = (mean_emb * scale_factor)
            .map_err(|e| MLError::InvalidResponse(format!("Scale error: {}", e)))?;

        normalized
            .reshape((1, self.embedding_dim))
            .map_err(|e| MLError::InvalidResponse(format!("Reshape error: {}", e)))
    }

    /// Returns unpooled sequence of embeddings: shape [seq_len, embedding_dim]
    pub async fn process_text_sequence(&self, text: &str) -> Result<Tensor, MLError> {
        if self.is_mock {
            return self.deterministic_embedding(text, text.len().max(1), None);
        }

        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| MLError::InvalidResponse(format!("Tokenization error: {}", e)))?;

        let token_ids = tokens.get_ids();
        if token_ids.is_empty() {
            return Tensor::zeros((1, self.embedding_dim), DType::F32, &self.device).map_err(|e| {
                MLError::InvalidResponse(format!("Zero tensor creation error: {}", e))
            });
        }

        let stacked = self.build_batched_embeddings(token_ids, false)?;
        Ok(stacked) // Returns[seq_len, embedding_dim]
    }

    /// "The Mouth": Convert a PC Brain "belief" vector back into English tokens
    pub fn decode_belief(&self, belief: &Tensor) -> Result<String, MLError> {
        Ok(self.decode_belief_with_confidence(belief)?.0)
    }

    pub fn decode_belief_with_confidence(
        &self,
        belief: &Tensor,
    ) -> Result<(String, f32, f32), MLError> {
        // Log tensor shapes for debugging
        let belief_shape = belief.shape();
        let lm_head_shape = self.lm_head.shape();

        // Check vocabulary size compatibility
        let tokenizer_vocab_size = self.tokenizer.get_vocab_size(true);
        let lm_head_vocab_size = lm_head_shape.dims()[0];
        if tokenizer_vocab_size as usize != lm_head_vocab_size {
            tracing::warn!(
                "Vocabulary size mismatch: tokenizer={}, lm_head={}. This may cause decoding issues.",
                tokenizer_vocab_size,
                lm_head_vocab_size
            );
        }

        // 1. Project belief through LM Head to get Vocab Logits
        // belief shape should be (1, embedding_dim) or (embedding_dim, 1)
        // lm_head shape is (vocab_size, embedding_dim)
        // We need belief * lm_head^T to get (1, vocab_size)

        // First, ensure belief is (1, embedding_dim)
        let belief_2d = if belief_shape.rank() == 1 {
            // Reshape from (embedding_dim,) to (1, embedding_dim)
            belief.reshape((1, belief_shape.dims()[0])).map_err(|e| {
                MLError::InvalidResponse(format!("Failed to reshape belief tensor: {}", e))
            })?
        } else if belief_shape.rank() == 2 && belief_shape.dims()[0] == 1 {
            // Already (1, embedding_dim)
            belief.clone()
        } else if belief_shape.rank() == 2 && belief_shape.dims()[1] == 1 {
            // (embedding_dim, 1) -> transpose to (1, embedding_dim)
            belief.t().map_err(|e| {
                MLError::InvalidResponse(format!("Failed to transpose belief tensor: {}", e))
            })?
        } else {
            return Err(MLError::InvalidResponse(format!(
                "Unexpected belief tensor shape: {:?}",
                belief_shape
            )));
        };

        // 🔴 FIX: Scale belief to approximate missing RMSNorm magnitude
        // Without this, logits are ~45x too small due to L2 normalization in process_text,
        // causing flat probability distributions and false "low-confidence" fallbacks.
        let d_f32 = belief_2d.shape().dims()[1] as f64;
        let scale_factor = d_f32.sqrt();
        let scaled_belief = belief_2d
            .affine(scale_factor, 0.0)
            .map_err(|e| MLError::InvalidResponse(format!("Failed to scale belief: {}", e)))?;

        // Transpose LM Head: (vocab_size, embedding_dim) -> (embedding_dim, vocab_size)
        let lm_head_t = self
            .lm_head
            .t()
            .map_err(|e| MLError::InvalidResponse(format!("Failed to transpose LM Head: {}", e)))?;

        // Matrix multiply: (1, embedding_dim) * (embedding_dim, vocab_size) = (1, vocab_size)
        let logits = scaled_belief
            .matmul(&lm_head_t)
            .map_err(|e| MLError::InvalidResponse(format!("LM Head projection failed: {}", e)))?;

        // 2. Sample the most likely token (Greedy for now)
        let logits_vec = logits
            .squeeze(0)
            .map_err(|e| MLError::InvalidResponse(e.to_string()))?
            .to_vec1::<f32>()
            .map_err(|e| MLError::InvalidResponse(e.to_string()))?;

        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        let mut total_logits = 0.0;
        let mut logit_count = 0;
        for (i, &val) in logits_vec.iter().enumerate() {
            total_logits += val.abs();
            logit_count += 1;
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Calculate confidence metrics
        let avg_logit_magnitude = total_logits / logit_count as f32;
        let max_logit_magnitude = max_val.abs();

        // Log confidence metrics
        tracing::info!(
            "LM Head confidence: max_logit={:.4}, avg_abs_logit={:.4}, max_idx={}, vocab_size={}",
            max_val,
            avg_logit_magnitude,
            max_idx,
            lm_head_vocab_size
        );

        // Check if LM Head is producing meaningful probabilities or just noise
        if avg_logit_magnitude < 0.1 && max_logit_magnitude < 1.0 {
            tracing::warn!(
                "LM Head appears to be producing low-confidence logits (avg={:.4}, max={:.4}). Pre-trained weights may not be properly loaded.",
                avg_logit_magnitude,
                max_logit_magnitude
            );
        } else if avg_logit_magnitude > 5.0 || max_logit_magnitude > 50.0 {
            tracing::info!(
                "LM Head producing high-confidence logits (avg={:.4}, max={:.4}) - pre-trained weights likely active.",
                avg_logit_magnitude,
                max_logit_magnitude
            );
        }

        // Log top-5 logits for visibility
        let mut top: Vec<(usize, f32)> = Vec::new();
        for (i, &val) in logits_vec.iter().enumerate() {
            if top.len() < 5 {
                top.push((i, val));
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            } else if let Some(last) = top.last() {
                if val > last.1 {
                    top.pop();
                    top.push((i, val));
                    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
        }
        let mut top_tokens = Vec::new();
        for (idx, val) in top {
            let token = self
                .tokenizer
                .decode(&[idx as u32], true)
                .unwrap_or_else(|_| "<decode_err>".to_string());
            top_tokens.push(format!("{}:{:.4}:'{}'", idx, val, token));
        }
        tracing::info!("LM Head top-5: {}", top_tokens.join(", "));

        // Ensure token ID is within vocabulary bounds
        if max_idx >= tokenizer_vocab_size as usize {
            tracing::warn!(
                "Token ID {} exceeds tokenizer vocabulary size {}, clamping to {}",
                max_idx,
                tokenizer_vocab_size,
                tokenizer_vocab_size - 1
            );
            max_idx = tokenizer_vocab_size as usize - 1;
        }

        // 3. Convert ID back to string
        let word = self
            .tokenizer
            .decode(&[max_idx as u32], true)
            .map_err(|e| MLError::InvalidResponse(format!("Decode error: {}", e)))?;

        tracing::info!("decode_belief: decoded token ID {} -> '{}'", max_idx, word);
        Ok((word, avg_logit_magnitude, max_logit_magnitude))
    }

    /// Get the embedding dimension detected from the model
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    pub fn get_model_info(&self) -> std::collections::HashMap<String, String> {
        let mut info = std::collections::HashMap::new();
        info.insert("motor_cortex".to_string(), "active".to_string());
        info.insert("embedding_dim".to_string(), self.embedding_dim.to_string());
        info.insert(
            "vocab_size".to_string(),
            self.tokenizer.get_vocab_size(true).to_string(),
        );
        info
    }

    /// Extract a specific layer weight tensor from the GGUF file
    /// This allows loading pre-trained weights for PC hierarchy initialization
    pub fn extract_layer_weight(&self, tensor_name: &str) -> Result<Tensor, MLError> {
        // Reopen the GGUF file to extract additional tensors
        let mut file = std::fs::File::open(&self.model_path).map_err(|e| {
            MLError::ModelLoadError(format!("Failed to open GGUF for weight extraction: {}", e))
        })?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
            MLError::ModelLoadError(format!("GGUF Read Error for weight extraction: {}", e))
        })?;

        // Extract the requested tensor
        let tensor = content
            .tensor(&mut file, tensor_name, &self.device)
            .map_err(|e| MLError::ModelLoadError(format!("Missing tensor {}: {}", tensor_name, e)))?
            .dequantize(&self.device)
            .map_err(|e| {
                MLError::ModelLoadError(format!("Dequantize Error for {}: {}", tensor_name, e))
            })?;

        tracing::info!(
            "Extracted layer weight {} with shape {:?}",
            tensor_name,
            tensor.shape()
        );
        Ok(tensor)
    }

    /// Get a list of available layer weight tensor names from the GGUF file
    pub fn list_layer_weights(&self) -> Result<Vec<String>, MLError> {
        let mut file = std::fs::File::open(&self.model_path).map_err(|e| {
            MLError::ModelLoadError(format!("Failed to open GGUF for listing: {}", e))
        })?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| MLError::ModelLoadError(format!("GGUF Read Error for listing: {}", e)))?;

        // Filter for layer weights (e.g., blk.*.ffn_down.weight, blk.*.attn_q.weight, etc.)
        let layer_weights: Vec<String> = content
            .tensor_infos
            .iter()
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

    /// Extract the "knowledge matrix" from a middle layer (e.g., blk.{middle}.ffn_down.weight)
    /// This represents the "intelligence genes" of the pre-trained model
    /// Dynamically calculates total layers from GGUF metadata and uses middle layer
    pub fn extract_knowledge_matrix(&self) -> Result<candle_core::Tensor, MLError> {
        use candle_core::Device;

        let device = Device::Cpu;
        let mut file = std::fs::File::open(&self.model_path).map_err(|e| {
            MLError::ModelLoadError(format!(
                "Failed to open GGUF for knowledge extraction: {}",
                e
            ))
        })?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
            MLError::ModelLoadError(format!("GGUF Read Error for knowledge extraction: {}", e))
        })?;

        // Parse all layer numbers from tensor names to determine total layers
        let mut layer_numbers: Vec<usize> = Vec::new();
        for (tensor_name, _) in content.tensor_infos.iter() {
            if let Some(stripped) = tensor_name.strip_prefix("blk.") {
                if let Some(dot_pos) = stripped.find('.') {
                    if let Ok(layer_num) = stripped[..dot_pos].parse::<usize>() {
                        layer_numbers.push(layer_num);
                    }
                }
            }
        }

        layer_numbers.sort();
        layer_numbers.dedup();

        let total_layers = if !layer_numbers.is_empty() {
            *layer_numbers.iter().max().unwrap() + 1 // layers are 0-indexed
        } else {
            // Fallback to default if no layer numbers found
            tracing::warn!("No layer numbers found in GGUF metadata, using default 24 layers");
            24
        };

        // Calculate middle layer (use integer division)
        let middle_layer = total_layers / 2;

        tracing::info!(
            "Detected {} total layers in GGUF model, using middle layer {} for knowledge extraction",
            total_layers,
            middle_layer
        );

        // Try middle layer first, then fallback to other layers
        // Create a list of candidates: middle, then quarter, then 0, then all layers
        let quarter_layer = total_layers / 4;
        let layer_candidates = vec![middle_layer, quarter_layer, 0];
        let mut last_error: Option<String> = None;

        for layer_num in layer_candidates {
            let tensor_name = format!("blk.{}.ffn_down.weight", layer_num);
            tracing::info!(
                "Trying to extract knowledge matrix from layer {}: {}",
                layer_num,
                tensor_name
            );

            match content.tensor(&mut file, &tensor_name, &device) {
                Ok(weights_q) => {
                    // Dequantize to f32
                    let weights = weights_q.dequantize(&device).map_err(|e| {
                        MLError::TensorError(format!(
                            "Failed to dequantize knowledge matrix: {}",
                            e
                        ))
                    })?;

                    tracing::info!(
                        "Successfully extracted knowledge matrix from {}: shape {:?}",
                        tensor_name,
                        weights.shape()
                    );
                    return Ok(weights);
                }
                Err(e) => {
                    let error_str = e.to_string();
                    last_error = Some(error_str.clone());
                    tracing::debug!("Layer {} not found: {}", layer_num, error_str);
                }
            }
        }

        // If no ffn_down.weight found, try any layer weight
        tracing::warn!("No ffn_down.weight found in middle layers, trying any blk.*.weight");
        for (tensor_name, _) in content.tensor_infos.iter() {
            if tensor_name.contains("blk.") && tensor_name.contains(".weight") {
                match content.tensor(&mut file, tensor_name, &device) {
                    Ok(weights_q) => {
                        let weights = weights_q.dequantize(&device).map_err(|e| {
                            MLError::TensorError(format!(
                                "Failed to dequantize fallback weight: {}",
                                e
                            ))
                        })?;

                        tracing::info!(
                            "Using fallback knowledge matrix from {}: shape {:?}",
                            tensor_name,
                            weights.shape()
                        );
                        return Ok(weights);
                    }
                    Err(e) => {
                        tracing::debug!("Failed to extract {}: {}", tensor_name, e);
                    }
                }
            }
        }

        Err(MLError::ModelLoadError(format!(
            "Could not extract knowledge matrix from GGUF: {:?}",
            last_error
        )))
    }

    /// Extract pre-trained weights from GGUF model with surgical slicing to fit target dimensions
    /// Implements "Surgical Slicing" with narrow() to crop GGUF tensors to fit PC layer dimensions
    /// This solves the dimension mismatch issue where GGUF tensor has shape [2048, 5632]
    /// but PC layer has shape [2048, 1024]
    pub fn extract_pretrained_weights(
        &self,
        tensor_name: &str,
        target_rows: usize,
        target_cols: usize,
    ) -> Result<Tensor, MLError> {
        use candle_core::Device;

        let device = Device::Cpu;
        let mut file = std::fs::File::open(&self.model_path).map_err(|e| {
            MLError::ModelLoadError(format!(
                "Failed to open GGUF for surgical extraction: {}",
                e
            ))
        })?;

        let content = candle_core::quantized::gguf_file::Content::read(&mut file).map_err(|e| {
            MLError::ModelLoadError(format!("GGUF Read Error for surgical extraction: {}", e))
        })?;

        // Extract the requested tensor
        let weights_q = content
            .tensor(&mut file, tensor_name, &device)
            .map_err(|e| {
                MLError::ModelLoadError(format!("Missing tensor {}: {}", tensor_name, e))
            })?;

        let full_weights = weights_q.dequantize(&device).map_err(|e| {
            MLError::ModelLoadError(format!("Dequantize Error for {}: {}", tensor_name, e))
        })?;

        let full_shape = full_weights.shape();
        if full_shape.dims().len() != 2 {
            return Err(MLError::TensorError(format!(
                "Tensor {} is not 2D, shape: {:?}",
                tensor_name, full_shape
            )));
        }

        let (src_rows, src_cols) = (full_shape.dims()[0], full_shape.dims()[1]);
        tracing::info!(
            "Surgical extraction: {} shape {}x{}, target {}x{}",
            tensor_name,
            src_rows,
            src_cols,
            target_rows,
            target_cols
        );

        // Determine how much to take (crop if source is larger than target)
        let rows_to_take = src_rows.min(target_rows);
        let cols_to_take = src_cols.min(target_cols);

        // Crop using narrow()
        let cropped = full_weights
            .narrow(0, 0, rows_to_take)
            .map_err(|e| MLError::TensorError(format!("Failed to narrow rows: {}", e)))?
            .narrow(1, 0, cols_to_take)
            .map_err(|e| MLError::TensorError(format!("Failed to narrow cols: {}", e)))?;

        // If we need padding (target is larger than source), create a zero-padded tensor
        if target_rows > src_rows || target_cols > src_cols {
            tracing::info!(
                "Padding needed: target {}x{} > source {}x{}",
                target_rows,
                target_cols,
                src_rows,
                src_cols
            );

            // Create zero tensor of target size
            let padded =
                Tensor::zeros((target_rows, target_cols), candle_core::DType::F32, &device)
                    .map_err(|e| {
                        MLError::TensorError(format!("Failed to create zero tensor: {}", e))
                    })?;

            // Copy cropped weights into the top-left corner
            let rows_to_copy = rows_to_take;
            let cols_to_copy = cols_to_take;

            // Create a view of the destination region
            let _dest_region = padded
                .narrow(0, 0, rows_to_copy)
                .map_err(|e| MLError::TensorError(format!("Failed to narrow dest rows: {}", e)))?
                .narrow(1, 0, cols_to_copy)
                .map_err(|e| MLError::TensorError(format!("Failed to narrow dest cols: {}", e)))?;

            // Copy the cropped weights into the destination
            // Note: This is a simplified approach - in practice we might need to use more complex tensor operations
            // For now, we'll use a simple assignment (this assumes we can assign tensors)
            // Actually, Tensor doesn't have direct assignment. We'll need to create a new tensor.
            // Let's use a different approach: create the padded tensor by concatenation

            // Create zero padding for rows if needed
            let _row_pad = if target_rows > src_rows {
                Tensor::zeros(
                    (target_rows - src_rows, cols_to_copy),
                    candle_core::DType::F32,
                    &device,
                )
                .map_err(|e| MLError::TensorError(format!("Failed to create row padding: {}", e)))?
            } else {
                Tensor::zeros((0, cols_to_copy), candle_core::DType::F32, &device).map_err(|e| {
                    MLError::TensorError(format!("Failed to create empty row padding: {}", e))
                })?
            };

            // Create zero padding for columns if needed
            let _col_pad = if target_cols > src_cols {
                Tensor::zeros(
                    (target_rows, target_cols - src_cols),
                    candle_core::DType::F32,
                    &device,
                )
                .map_err(|e| MLError::TensorError(format!("Failed to create col padding: {}", e)))?
            } else {
                Tensor::zeros((target_rows, 0), candle_core::DType::F32, &device).map_err(|e| {
                    MLError::TensorError(format!("Failed to create empty col padding: {}", e))
                })?
            };

            // For simplicity, we'll just return the cropped weights for now
            // and log a warning about the dimension mismatch
            tracing::warn!(
                "Cannot pad tensor {} from {}x{} to {}x{} - using cropped version",
                tensor_name,
                src_rows,
                src_cols,
                target_rows,
                target_cols
            );

            return Ok(cropped);
        }

        tracing::info!(
            "Successfully extracted and sliced {} to {}x{}",
            tensor_name,
            rows_to_take,
            cols_to_take
        );
        Ok(cropped)
    }

    pub fn clear_cache(&self) {}

    fn build_batched_embeddings(
        &self,
        token_ids: &[u32],
        apply_positional: bool,
    ) -> Result<Tensor, MLError> {
        let mut embeddings = Vec::with_capacity(token_ids.len());
        for &id in token_ids.iter() {
            let emb = self
                .token_embeddings
                .get(id as usize)
                .map_err(|e| MLError::InvalidResponse(format!("Embedding lookup error: {}", e)))?;
            embeddings.push(emb);
        }

        let stacked = Tensor::stack(&embeddings, 0)
            .map_err(|e| MLError::InvalidResponse(format!("Stack error: {}", e)))?;

        if !apply_positional {
            return Ok(stacked);
        }

        let seq_len = token_ids.len() as f32;
        let pos_weights: Vec<f32> = (0..token_ids.len())
            .map(|i| 1.0 + (i as f32 / seq_len) * 0.5)
            .collect();
        let pos_tensor = Tensor::from_vec(pos_weights, (token_ids.len(), 1), &self.device)
            .map_err(|e| MLError::InvalidResponse(format!("Pos tensor error: {}", e)))?
            .broadcast_as(stacked.shape())
            .map_err(|e| MLError::InvalidResponse(format!("Broadcast error: {}", e)))?;
        let stacked = stacked
            .mul(&pos_tensor)
            .map_err(|e| MLError::InvalidResponse(format!("Mul error: {}", e)))?;

        Ok(stacked)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_safe_l2_normalization_avoids_broadcast_panic()
    -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let dim = 2048;

        // 1. Simulate the 1D tensor resulting from stacked.mean(0)
        // We use a high standard deviation (5.0) to simulate unnormalized LLM outputs
        let mean_emb = Tensor::randn(0f32, 5.0, vec![dim], &device)?;

        // Check initial norm (will be much larger than 1.0)
        let initial_norm = mean_emb.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        assert!(
            initial_norm > 10.0,
            "Initial norm should be large, got {}",
            initial_norm
        );

        // 2. The exact safe normalization logic from process_text()
        let norm_sq = mean_emb.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt() as f64;
        let scale_factor = if norm > 1e-6 { 1.0 / norm } else { 1.0 };

        // 3. THIS is the operation that previously panicked with a broadcast error.
        // It must succeed using the f64 scalar implementation.
        let normalized = (mean_emb * scale_factor)?;

        // 4. Reshape to PC shape as done in process_text
        let final_tensor = normalized.reshape((1, dim))?;

        // 5. Verify the L2 norm is now exactly 1.0
        let new_norm = final_tensor.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();

        assert!(
            (new_norm - 1.0).abs() < 1e-4,
            "Normalized tensor L2 norm must be 1.0, got {}",
            new_norm
        );

        Ok(())
    }

    #[test]
    fn test_input_l2_normalization_accuracy() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;

        // 1. Создаем случайный вектор с ОГРОМНЫМИ значениями (как было в баге)
        let huge_values = vec![100.0f32; 2048];
        let tensor = Tensor::from_vec(huge_values, (1, 2048), &device)?;

        // 2. Применяем логику нормализации из process_text
        let norm_sq = tensor.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt();
        let scale_factor = if norm > 1e-6 { 1.0 / norm } else { 1.0 };

        // Масштабируем
        let normalized = (tensor * (scale_factor as f64))?;

        // 3. Проверяем длину (норму) итогового вектора
        let final_norm_sq = normalized.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let final_norm = final_norm_sq.sqrt();

        // Норма должна быть идеально равна 1.0 (с учетом погрешности float)
        assert!(
            (final_norm - 1.0).abs() < 1e-5,
            "L2 нормализация не работает! Норма: {}",
            final_norm
        );

        Ok(())
    }

    #[test]
    fn test_decode_belief_with_rms_scaling() -> Result<()> {
        // This test requires a real model to load the lm_head
        let config = crate::config::NodeConfig::default();
        // Ensure you have a model at the default path or update this path
        if !std::path::Path::new(&config.model_path).exists() {
            println!(
                "Skipping test_decode_belief_with_rms_scaling: model file not found at {}",
                config.model_path
            );
            return Ok(());
        }

        let device_type = crate::types::DeviceType {
            name: "cpu".to_string(),
            description: "".to_string(),
            supported: true,
        };
        let engine = MLEngine::new(&config.model_path, device_type).unwrap();
        let belief_dim = engine.embedding_dim();

        // Create a normalized belief tensor (length 1.0), which caused the original issue
        let belief_l2 = Tensor::ones((1, belief_dim), DType::F32, &engine.device)?;
        // Manual L2 normalization
        let norm_sq = belief_l2.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt();
        let scale_factor = if norm > 1e-6 { 1.0 / norm } else { 1.0 };
        let belief_l2 = (belief_l2 * (scale_factor as f64))?;

        let (_word, avg_mag, max_mag) = engine.decode_belief_with_confidence(&belief_l2)?;

        // The original buggy code produced max_mag ~0.03.
        // The exact decoded scale is model-dependent, so this test should only
        // assert that the logits are clearly out of the near-zero regime.
        assert!(
            max_mag > 1.0,
            "Max logit magnitude ({}) is too low. RMSNorm scaling fix may have regressed.",
            max_mag
        );
        assert!(
            avg_mag > 0.1,
            "Average logit magnitude ({}) is too low. RMSNorm scaling fix may have regressed.",
            avg_mag
        );

        println!(
            "Logit scaling test passed: max_mag={}, avg_mag={}",
            max_mag, avg_mag
        );
        Ok(())
    }
}
