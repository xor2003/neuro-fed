# Llama FFI Component Technical Specifications

## Overview
`llama_ffi.rs` provides Rust FFI bindings to llama.cpp for:
- Text embeddings via `llama_embed()`
- Optional text decoding via `llama_decode()`
- GGML matrix operations for GPU/CPU acceleration
- GGUF model management and context handling

## Architecture

### Core Components
```rust
// Public API
pub struct LlamaContext {
    model: *mut LlamaModel,
    ctx: *mut LlamaContextHandle,
    gpu_config: GpuConfig,
}

pub struct GpuConfig {
    backend: GpuBackend,
    device_id: usize,
    memory_fraction: f32,
}

pub struct Embedding {
    data: Vec<f32>,
    dim: usize,
}

pub enum GpuBackend {
    CUDA,
    Metal,
    Vulkan,
    CPU,
}
```

### FFI Bindings
```rust
// C function signatures (llama.cpp)
extern "C" {
    fn llama_init_from_file(model_path: *const c_char) -> *mut LlamaModel;
    fn llama_context_create(model: *mut LlamaModel, n_ctx: c_int) -> *mut LlamaContextHandle;
    fn llama_embed(ctx: *mut LlamaContextHandle, text: *const c_char) -> *mut f32;
    fn llama_get_embedding_dim(model: *mut LlamaModel) -> c_int;
    fn llama_get_embedding(embedding: *mut f32, dim: c_int) -> *mut c_float;
    fn llama_context_free(ctx: *mut LlamaContextHandle);
    fn llama_model_free(model: *mut LlamaModel);
}
```

## Implementation Details

### Context Management
```rust
impl LlamaContext {
    pub fn new(model_path: &str, context_size: usize) -> Result<Self, LlamaError> {
        // Initialize model from GGUF file
        let model_ptr = unsafe { llama_init_from_file(model_path.as_ptr() as *const c_char) };
        
        // Detect GPU capabilities and configure
        let gpu_config = Self::detect_gpu()?
        
        // Create context with specified size
        let ctx_ptr = unsafe { llama_context_create(model_ptr, context_size as c_int) };
        
        Ok(LlamaContext {
            model: model_ptr,
            ctx: ctx_ptr,
            gpu_config,
        })
    }
    
    pub fn embed(&self, text: &str) -> Result<Embedding, LlamaError> {
        // Convert text to C string
        let c_text = CString::new(text)?;
        
        // Get embedding dimension
        let dim = unsafe { llama_get_embedding_dim(self.model) } as usize;
        
        // Perform embedding
        let embedding_ptr = unsafe { llama_embed(self.ctx, c_text.as_ptr()) };
        
        // Copy embedding data to Rust Vec
        let data = unsafe { 
            std::slice::from_raw_parts(embedding_ptr, dim)
                .to_vec()
        };
        
        Ok(Embedding { data, dim })
    }
    
    pub fn decode(&self, embedding: &Embedding) -> Result<String, LlamaError> {
        // Optional: decode embedding back to text
        // Implementation depends on llama.cpp capabilities
        unimplemented!()
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            llama_context_free(self.ctx);
            llama_model_free(self.model);
        }
    }
}
```

### GPU Detection and Configuration
```rust
impl LlamaContext {
    fn detect_gpu() -> Result<GpuConfig, LlamaError> {
        // Check for CUDA first
        if cuda_available() {
            return Ok(GpuConfig {
                backend: GpuBackend::CUDA,
                device_id: 0,
                memory_fraction: 0.8,
            });
        }
        
        // Fallback to Metal (macOS)
        if metal_available() {
            return Ok(GpuConfig {
                backend: GpuBackend::Metal,
                device_id: 0,
                memory_fraction: 0.8,
            });
        }
        
        // Fallback to Vulkan
        if vulkan_available() {
            return Ok(GpuConfig {
                backend: GpuBackend::Vulkan,
                device_id: 0,
                memory_fraction: 0.8,
            });
        }
        
        // CPU fallback
        Ok(GpuConfig {
            backend: GpuBackend::CPU,
            device_id: 0,
            memory_fraction: 1.0,
        })
    }
}
```

## Error Handling

### Custom Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum LlamaError {
    #[error("FFI call failed: {0}")]
    FfiCallFailed(String),
    
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    
    #[error("GPU not available: {0}")]
    GpuNotAvailable(String),
    
    #[error("Invalid embedding dimension: {0}")]
    InvalidEmbeddingDim(i32),
    
    #[error("Memory allocation failed")]
    MemoryAllocationFailed,
    
    #[error("Text encoding failed: {0}")]
    TextEncodingFailed(String),
}
```

## Performance Considerations

### Memory Management
- Use `Vec<f32>` for embeddings to leverage Rust's memory safety
- Implement proper cleanup in `Drop` trait
- Consider memory pooling for high-throughput scenarios

### GPU Optimization
- Configure memory fraction based on available VRAM
- Use appropriate backend for target platform
- Implement async embedding calls for non-blocking operation

### Thread Safety
- Ensure FFI calls are thread-safe using `Send` and `Sync` traits
- Consider using `Arc<Mutex<>>` for shared context access

## Integration Points

### With PC Hierarchy
```rust
// In pc_hierarchy.rs
impl PredictiveCoding {
    pub fn process_input(&mut self, text: &str, llama_ctx: &LlamaContext) -> Result<(), LlamaError> {
        // Embed text using llama FFI
        let embedding = llama_ctx.embed(text)?;
        
        // Process embedding through PC hierarchy
        self.infer(&embedding.data)?;
        
        Ok(())
    }
}
```

### Configuration
```toml
# config.toml
[llama]
model_path = "models/llama-3.2-3B.Q4_K_M.gguf"
context_size = 2048
gpu_backend = "cuda"  # or "metal", "vulkan", "cpu"
gpu_device_id = 0
gpu_memory_fraction = 0.8
```

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_embedding_creation() {
        let ctx = LlamaContext::new("test-model.gguf", 512).unwrap();
        let embedding = ctx.embed("test text").unwrap();
        assert_eq!(embedding.dim, 512);
        assert!(!embedding.data.is_empty());
    }
    
    #[test]
    fn test_gpu_detection() {
        let config = LlamaContext::detect_gpu().unwrap();
        assert!(matches!(config.backend, GpuBackend::CUDA | GpuBackend::Metal | GpuBackend::Vulkan | GpuBackend::CPU));
    }
}
```

### Integration Tests
- Test with actual GGUF models
- Verify GPU acceleration works correctly
- Test error handling for missing models, invalid text, etc.

## Dependencies

### Required
- `libc = "0.2"` - For FFI bindings
- `thiserror = "1.0"` - For error handling

### Optional
- `serde = { version = "1.0", features = ["derive"] }` - For configuration serialization
- `tracing = "0.1"` - For structured logging

## Security Considerations

- Validate all input text before embedding
- Handle FFI errors gracefully to prevent crashes
- Ensure proper memory cleanup to prevent leaks
- Consider sandboxing for untrusted model files