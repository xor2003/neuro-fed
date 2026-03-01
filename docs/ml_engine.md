# ML Engine Component Documentation

## Overview

The `MLEngine` component is the core machine learning inference engine for NeuroFed Node. It integrates with the ModelManager to automatically select and load appropriate models based on available system memory, and uses candle-core for pure Rust CPU/GPU operations. The engine provides text processing, embedding generation, and model information retrieval with comprehensive error handling and GPU support.

## Key Features

- **ModelManager Integration**: Automatically selects and loads models based on available system memory
- **Candle-Core Backend**: Pure Rust tensor operations with CPU/GPU acceleration
- **Multi-Backend Support**: Supports Llama 3 8B and Qwen2.5-1.5B GGUF models
- **GPU Acceleration**: Automatic detection and utilization of CUDA, Metal, and Vulkan backends
- **Embedding Generation**: High-quality vector embeddings for text processing
- **Caching**: Intelligent caching of processed text embeddings
- **Async Operations**: Fully asynchronous model loading and processing
- **Backward Compatibility**: Maintains existing API for seamless integration

## Architecture

### Core Components

1. **MLEngine**: Main struct that orchestrates ML operations
2. **ModelWrapper**: Trait abstraction for different model types (Llama, Qwen)
3. **Device Selection**: Automatic GPU detection with CPU fallback
4. **Embedding Cache**: LRU cache for processed embeddings

### Model Integration

The MLEngine works seamlessly with the ModelManager:

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   MLEngine  │────│ ModelManager │────│ Model Files │
└─────────────┘    └──────────────┘    └─────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Inference  │    │ Auto-Select  │    │ GGUF Models │
└─────────────┘    └──────────────┘    └─────────────┘
```

## API Reference

### MLEngine

#### Creation

**With ModelManager (Recommended):**
```rust
use neuro_fed_node::{MLEngine, ModelManager, NodeConfig};
use std::sync::Arc;

let config = NodeConfig::default();
let model_manager = ModelManager::new(config);
let engine = MLEngine::new_with_manager(Arc::new(model_manager)).await?;
```

**Direct Initialization (Legacy):**
```rust
use neuro_fed_node::{MLEngine, DeviceType};

let engine = MLEngine::new("path/to/model", DeviceType::Cpu)?;
```

#### Key Methods

##### `new_with_manager(model_manager: Arc<ModelManager>) -> Result<Self, MLError>`
Creates a new MLEngine using ModelManager for automatic model selection.

**Example:**
```rust
let config = NodeConfig::default();
let manager = ModelManager::new(config);
let engine = MLEngine::new_with_manager(Arc::new(manager)).await?;
```

##### `new(model_path: &str, device_type: DeviceType) -> Result<Self, MLError>`
Creates a new MLEngine with explicit model path and device type (backward compatibility).

**Example:**
```rust
let engine = MLEngine::new("models/llama-3-8b-instruct", DeviceType::Cuda)?;
```

##### `process_text(text: &str) -> Result<Tensor, MLError>`
Processes text and returns embedding tensor.

**Example:**
```rust
let text = "Hello, world! This is a test.";
let embedding = engine.process_text(text).await?;
println!("Embedding shape: {:?}", embedding.shape());
```

##### `get_model_info() -> HashMap<String, String>`
Returns metadata about the currently loaded model.

**Example:**
```rust
let info = engine.get_model_info();
for (key, value) in info {
    println!("{}: {}", key, value);
}
```

##### `clear_cache()`
Clears the embedding cache.

**Example:**
```rust
engine.clear_cache();
```

### Device Types

The MLEngine supports multiple device types:

```rust
pub enum DeviceType {
    Cpu,           // CPU only
    Cuda,          // NVIDIA CUDA GPU
    Metal,         // Apple Metal GPU
    Vulkan,        // Vulkan GPU
    BestAvailable, // Automatically selects the best available device
}
```

**Automatic Device Selection:**
```rust
let engine = MLEngine::new("models/llama-3-8b-instruct", DeviceType::BestAvailable)?;
```

## Usage Examples

### Basic Text Processing

```rust
use neuro_fed_node::{MLEngine, ModelManager, NodeConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create model manager and engine
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    let engine = MLEngine::new_with_manager(Arc::new(manager)).await?;
    
    // Process text
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Rust provides memory safety without garbage collection.",
    ];
    
    for text in texts {
        let embedding = engine.process_text(text).await?;
        println!("Processed '{}': {:?}", &text[..20], embedding.shape());
    }
    
    Ok(())
}
```

### GPU Acceleration

```rust
use neuro_fed_node::{MLEngine, DeviceType};

// Explicit GPU selection
let engine = MLEngine::new("models/qwen2.5-1.5b-instruct", DeviceType::Cuda)?;

// Or let the engine decide
let engine = MLEngine::new("models/llama-3-8b-instruct", DeviceType::BestAvailable)?;

// Check which device is being used
let info = engine.get_model_info();
if let Some(device) = info.get("device") {
    println!("Running on: {}", device);
}
```

### Integration with OpenAI Proxy

The MLEngine integrates seamlessly with the OpenAI proxy for local inference:

```rust
use neuro_fed_node::{MLEngine, ModelManager, OpenAiProxy, NodeConfig};
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    let engine = MLEngine::new_with_manager(Arc::new(manager)).await?;
    
    // Wrap in Arc<Mutex> for shared access
    let local_engine = Arc::new(Mutex::new(engine));
    
    // Create OpenAI proxy with local fallback
    let proxy = OpenAiProxy::new(
        NodeConfig::default(),
        "dummy-api-key".to_string(),
        local_engine,
    );
    
    // Start proxy server
    proxy.start(8080).await?;
    
    Ok(())
}
```

### Model Information and Monitoring

```rust
use neuro_fed_node::{MLEngine, ModelManager, NodeConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    let engine = MLEngine::new_with_manager(Arc::new(manager)).await?;
    
    // Get comprehensive model information
    let info = engine.get_model_info();
    
    println!("=== Model Information ===");
    println!("Model: {}", info.get("model_name").unwrap_or(&"Unknown".to_string()));
    println!("Device: {}", info.get("device").unwrap_or(&"Unknown".to_string()));
    println!("Embedding Dim: {}", info.get("embedding_dim").unwrap_or(&"Unknown".to_string()));
    println!("Cache Size: {}", info.get("cache_size").unwrap_or(&"Unknown".to_string()));
    
    // Monitor performance
    let start = std::time::Instant::now();
    let embedding = engine.process_text("Benchmark test").await?;
    let duration = start.elapsed();
    
    println!("Processing time: {:?}", duration);
    println!("Embedding shape: {:?}", embedding.shape());
    
    Ok(())
}
```

## Error Handling

The MLEngine uses comprehensive error handling with the `MLError` type:

```rust
pub enum MLError {
    ModelLoadError(String),
    InferenceError(String),
    DeviceError(String),
    TokenizationError(String),
    CacheError(String),
    UnknownError(String),
}
```

**Error Handling Example:**
```rust
match engine.process_text(text).await {
    Ok(embedding) => {
        // Process embedding
    }
    Err(MLError::ModelLoadError(msg)) => {
        eprintln!("Failed to load model: {}", msg);
        // Try fallback model
    }
    Err(MLError::DeviceError(msg)) => {
        eprintln!("GPU error: {}", msg);
        // Fall back to CPU
    }
    Err(e) => {
        eprintln!("Unexpected error: {:?}", e);
    }
}
```

## Performance Optimization

### Caching Strategy

The MLEngine implements an intelligent LRU cache for embeddings:

```rust
// Cache configuration (from ml_engine.rs)
const DEFAULT_CACHE_CAPACITY: usize = 1000;

// Manual cache management
engine.clear_cache(); // Clear all cached embeddings

// Automatic cache eviction occurs when capacity is reached
```

### GPU Memory Management

For GPU operations, memory is managed automatically:

```rust
// GPU memory fraction can be configured via environment variable
export GPU_MEMORY_FRACTION=0.8  // Use 80% of available GPU memory

// Or programmatically
std::env::set_var("GPU_MEMORY_FRACTION", "0.8");
```

### Batch Processing

For optimal performance, process multiple texts in sequence:

```rust
let texts = vec![/* large batch of texts */];
let mut embeddings = Vec::new();

for text in texts {
    let embedding = engine.process_text(text).await?;
    embeddings.push(embedding);
}

// Or use parallel processing with tokio
use futures::future::join_all;

let futures = texts.iter().map(|text| engine.process_text(text));
let embeddings = join_all(futures).await;
```

## Integration Guide

### With Existing Components

**OpenAI Proxy Integration:**
The MLEngine provides local inference capabilities for the OpenAI proxy, reducing API costs and improving latency.

**Bootstrap Integration:**
The bootstrap process can use the MLEngine for initial model distillation and embedding generation.

**Predictive Coding Hierarchy:**
Generated embeddings feed into the PC hierarchy for hierarchical predictive coding.

### Migration from Legacy Implementation

If you're upgrading from the previous placeholder implementation:

1. **API Changes:**
   - `LlamaContext` is replaced by `MLEngine`
   - Model loading is now managed by `ModelManager`
   - Device selection is more flexible

2. **Code Migration Example:**

**Before:**
```rust
use neuro_fed_node::LlamaContext;
let context = LlamaContext::new("path/to/model", 512);
let embedding = context.get_embedding("text")?;
```

**After:**
```rust
use neuro_fed_node::{MLEngine, DeviceType};
let engine = MLEngine::new("path/to/model", DeviceType::BestAvailable)?;
let embedding = engine.process_text("text").await?;
```

## Configuration

### Environment Variables

```bash
# Device selection
export DEVICE_TYPE=cuda          # cuda, metal, vulkan, cpu
export GPU_DEVICE_ID=0           # GPU device index
export GPU_MEMORY_FRACTION=0.8   # GPU memory usage limit

# Performance tuning
export EMBEDDING_CACHE_SIZE=1000 # LRU cache capacity
export MAX_SEQUENCE_LENGTH=512   # Maximum token length

# Debugging
export RUST_LOG=ml_engine=debug
export RUST_BACKTRACE=1
```

### Configuration File

The MLEngine respects the global `NodeConfig`:

```toml
[ml_engine]
device_type = "best_available"
cache_capacity = 1000
max_sequence_length = 512
enable_gpu = true

[models]
default = "llama-3-8b-instruct"
fallback = "qwen2.5-1.5b-instruct"
```

## Testing

### Unit Tests

Run ML Engine tests:
```bash
cargo test ml_engine --lib
```

### Integration Tests

Test ML Engine with ModelManager:
```bash
cargo test ml_engine_integration --test integration
```

### Performance Benchmarks

```bash
cargo bench ml_engine
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected:**
   - Check driver installation
   - Verify CUDA/Metal/Vulkan support
   - Set `DEVICE_TYPE=cpu` as fallback

2. **Model Loading Failed:**
   - Verify model file exists
   - Check file permissions
   - Ensure sufficient disk space

3. **Out of Memory:**
   - Reduce batch size
   - Use smaller model (Qwen2.5-1.5B)
   - Enable GPU memory fraction limit

4. **Slow Performance:**
   - Enable GPU acceleration
   - Increase cache capacity
   - Use batch processing

### Debug Logging

Enable detailed logging:
```rust
std::env::set_var("RUST_LOG", "ml_engine=debug,candle_core=info");
```

## Future Enhancements

Planned improvements for the ML Engine:

1. **Multi-Model Support**: Simultaneous loading of multiple models
2. **Quantization**: Support for different quantization levels (Q2_K, Q3_K, Q4_K, etc.)
3. **Distributed Inference**: Load balancing across multiple GPUs
4. **Custom Model Integration**: Support for user-provided GGUF models
5. **Real-time Metrics**: Prometheus integration for performance monitoring
6. **Plugin Architecture**: Extensible backend support for other ML frameworks

## See Also

- [Model Manager Documentation](./model_manager.md)
- [OpenAI Proxy Documentation](../src/openai_proxy.rs)
- [Candle-Core Documentation](https://github.com/huggingface/candle)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)