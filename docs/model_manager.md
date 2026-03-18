# Model Manager Component Documentation

## Overview

The `ModelManager` component is responsible for automatic model download and selection based on available system memory. It supports Llama 3 8B and Qwen2.5-1.5B models as default options when no specific model is configured. The manager handles the complete model lifecycle including detection, download, loading, and cleanup.

## Key Features

- **Automatic Model Selection**: Dynamically selects the most appropriate model based on available system memory
- **Cross-Platform Memory Detection**: Works on Linux, macOS, and Windows systems
- **Windows-Friendly Detection**: Uses PowerShell/CIM-based memory detection instead of relying only on deprecated `wmic`
- **Progress Tracking**: Real-time download progress with callbacks
- **Error Handling**: Comprehensive error types and recovery mechanisms
- **Candle-Core Integration**: Seamless integration with the candle-core framework for model operations
- **Async Operations**: Fully asynchronous download and loading operations
- **Configurable**: Integrates with the existing `NodeConfig` system

## Architecture

### Core Components

1. **ModelManager**: Main struct that orchestrates model management
2. **ModelInfo**: Metadata structure for model specifications
3. **DownloadProgress**: Real-time progress tracking structure
4. **ModelManagerError**: Comprehensive error enumeration

### Default Models

The ModelManager comes pre-configured with two default models:

1. **Llama 3 8B** (`llama-3-8b`)
   - Size: ~4.5GB
   - Minimum Memory: 8GB
   - Maximum Memory: 16GB
   - Quantization: Q4_K_M

2. **Qwen2.5 1.5B** (`qwen2.5-1.5b`)
   - Size: ~1GB
   - Minimum Memory: 2GB
   - Maximum Memory: 4GB
   - Quantization: Q4_K_M

## API Reference

### ModelManager

#### Creation
```rust
let config = NodeConfig::default();
let manager = ModelManager::new(config);
```

#### Key Methods

##### `get_recommended_model() -> Result<ModelInfo, ModelManagerError>`
Automatically selects the best model based on available system memory.

**Example:**
```rust
let recommended = manager.get_recommended_model().await?;
println!("Recommended model: {}", recommended.name);
```

##### `download_model(model_name: &str) -> Result<(), ModelManagerError>`
Downloads a model with progress tracking.

**Example:**
```rust
manager.set_progress_callback(|progress| {
    println!("Download progress: {:.2}%", progress.percentage);
});

manager.download_model("llama-3-8b").await?;
```

##### `load_model(model_name: &str) -> Result<AutoModel, ModelManagerError>`
Loads a model using candle-core. Automatically downloads if not present.

**Example:**
```rust
let model = manager.load_model("qwen2.5-1.5b").await?;
println!("Loaded model: {}", model.name);
```

##### `detect_available_memory() -> Result<u64, String>`
Detects available system memory in MB.

**Example:**
```rust
let memory_mb = manager.detect_available_memory().await?;
println!("Available memory: {} MB", memory_mb);
```

### Detection Strategy Notes

The manager now tries multiple platform-specific probes in order and falls back conservatively when they fail:
- Linux: `free -m`
- macOS: `vm_stat`
- Windows: PowerShell `Get-CimInstance Win32_OperatingSystem`
- Windows fallback: `systeminfo`
- Final conservative fallback: `4096` MB

The fallback is deliberate. Model recommendation should degrade conservatively rather than fail outright when host memory introspection is blocked or unavailable.

##### `get_available_models() -> Vec<ModelInfo>`
Returns a list of all available models.

**Example:**
```rust
let models = manager.get_available_models();
for model in models {
    println!("Model: {} ({} MB)", model.name, model.size_mb);
}
```

##### `is_model_downloaded(model_name: &str) -> bool`
Checks if a model is already downloaded locally.

**Example:**
```rust
if manager.is_model_downloaded("llama-3-8b").await {
    println!("Model already downloaded");
}
```

##### `get_model_config(model_name: &str) -> Result<AutoConfig, ModelManagerError>`
Gets model configuration for ML engine integration.

**Example:**
```rust
let config = manager.get_model_config("llama-3-8b")?;
println!("Hidden size: {}", config.hidden_size);
```

##### `get_tokenizer(model_name: &str) -> Result<AutoTokenizer, ModelManagerError>`
Gets tokenizer configuration for the model.

**Example:**
```rust
let tokenizer = manager.get_tokenizer("llama-3-8b")?;
println!("Vocabulary size: {}", tokenizer.vocab_size);
```

##### `get_device_config(model_name: &str) -> Result<DeviceType, ModelManagerError>`
Determines the appropriate device configuration based on model size.

**Example:**
```rust
let device = manager.get_device_config("llama-3-8b")?;
println!("Device type: {}", device.name);
```

##### `cleanup() -> Result<(), ModelManagerError>`
Cleans up downloaded model files.

**Example:**
```rust
manager.cleanup().await?;
println!("All model files cleaned up");
```

### ModelInfo Structure

```rust
pub struct ModelInfo {
    pub name: String,           // Model identifier (e.g., "llama-3-8b")
    pub version: String,        // Model version
    pub size_mb: u64,           // Model size in megabytes
    pub min_memory_mb: u64,     // Minimum required memory in MB
    pub max_memory_mb: u64,     // Maximum recommended memory in MB
    pub quantization: String,   // Quantization type (e.g., "Q4_K_M")
    pub download_url: String,   // URL to download the model
    pub local_path: String,     // Local file path
}
```

### DownloadProgress Structure

```rust
pub struct DownloadProgress {
    pub bytes_downloaded: u64,  // Bytes downloaded so far
    pub total_bytes: u64,       // Total bytes to download
    pub percentage: f64,        // Download percentage (0-100)
    pub speed_kbps: f64,        // Download speed in KB/s
    pub eta_seconds: u64,       // Estimated time remaining in seconds
}
```

### Error Handling

The `ModelManagerError` enum provides comprehensive error types:

```rust
pub enum ModelManagerError {
    DownloadError(String),          // Download-related errors
    FileError(String),              // File system errors
    MemoryDetectionError(String),   // Memory detection failures
    ModelLoadError(String),         // Model loading errors
    InvalidModelError(String),      // Invalid model specification
    NetworkError(String),           // Network-related errors
    ConfigurationError(String),     // Configuration errors
}
```

## Integration Points

### With Configuration System
The ModelManager integrates with the existing `NodeConfig` system:

```rust
use crate::config::NodeConfig;
use crate::model_manager::ModelManager;

let config = NodeConfig::default();
let manager = ModelManager::new(config);
```

### With ML Engine
The ModelManager provides model configurations for the ML Engine:

```rust
use crate::ml_engine::MLEngine;
use crate::model_manager::ModelManager;

let manager = ModelManager::new(config);
let model_config = manager.get_model_config("llama-3-8b")?;
let device_config = manager.get_device_config("llama-3-8b")?;

let engine = MLEngine::new(&model_config, device_config)?;
```

### With Bootstrap System
The bootstrap system can use ModelManager for model initialization:

```rust
use crate::bootstrap::Bootstrap;
use crate::model_manager::ModelManager;

let manager = ModelManager::new(config);
let recommended_model = manager.get_recommended_model().await?;

let bootstrap = Bootstrap::new(&recommended_model.local_path);
```

## Usage Examples

### Basic Usage
```rust
use crate::model_manager::ModelManager;
use crate::config::NodeConfig;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NodeConfig::default();
    let mut manager = ModelManager::new(config);
    
    // Set up progress callback
    manager.set_progress_callback(|progress| {
        println!("Progress: {:.1}% ({:.1} KB/s)", 
            progress.percentage, progress.speed_kbps);
    });
    
    // Get recommended model based on system memory
    let model = manager.get_recommended_model().await?;
    println!("Selected model: {}", model.name);
    
    // Download if not present
    if !manager.is_model_downloaded(&model.name).await {
        manager.download_model(&model.name).await?;
    }
    
    // Load the model
    let loaded_model = manager.load_model(&model.name).await?;
    println!("Model loaded successfully: {}", loaded_model.name);
    
    Ok(())
}
```

### Advanced Usage with Custom Models
```rust
use crate::model_manager::{ModelManager, ModelInfo};
use std::collections::HashMap;

let mut custom_models = HashMap::new();
custom_models.insert("custom-model".to_string(), ModelInfo {
    name: "custom-model".to_string(),
    version: "1.0".to_string(),
    size_mb: 2000,
    min_memory_mb: 4096,
    max_memory_mb: 8192,
    quantization: "Q4_K_M".to_string(),
    download_url: "https://example.com/custom-model.gguf".to_string(),
    local_path: "models/custom-model.gguf".to_string(),
});

let config = NodeConfig::default();
let mut manager = ModelManager::new(config);
manager.models = custom_models;
```

## Testing

The ModelManager includes comprehensive unit tests:

```rust
#[tokio::test]
async fn test_model_manager_creation() {
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    
    assert_eq!(manager.models.len(), 2);
    assert!(manager.models.contains_key("llama-3-8b"));
    assert!(manager.models.contains_key("qwen2.5-1.5b"));
}

#[tokio::test]
async fn test_memory_detection() {
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    
    let memory = manager.detect_available_memory().await;
    assert!(memory.is_ok());
    assert!(memory.unwrap() > 0);
}

#[tokio::test]
async fn test_model_recommendation() {
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    
    let recommended = manager.get_recommended_model().await;
    assert!(recommended.is_ok());
    let model = recommended.unwrap();
    
    assert!(model.name == "qwen2.5-1.5b" || model.name == "llama-3-8b");
}
```

## Configuration

### Default Configuration
The ModelManager uses sensible defaults:
- Download directory: `models/`
- Timeout: 300 seconds for downloads
- Default models: Llama 3 8B and Qwen2.5 1.5B

### Custom Configuration
You can extend the ModelManager with custom models:

```rust
let mut manager = ModelManager::new(config);

// Add custom model
manager.models.insert("my-model".to_string(), ModelInfo {
    name: "my-model".to_string(),
    version: "1.0".to_string(),
    size_mb: 1500,
    min_memory_mb: 3072,
    max_memory_mb: 6144,
    quantization: "Q4_K_M".to_string(),
    download_url: "https://my-server.com/model.gguf".to_string(),
    local_path: "models/my-model.gguf".to_string(),
});
```

## Performance Considerations

1. **Memory Detection**: The memory detection uses platform-specific commands (`free`, `vm_stat`, `wmic`)
2. **Download Resumption**: Currently downloads from scratch; consider adding resumable downloads
3. **Caching**: Downloaded models are cached locally for future use
4. **Concurrency**: Multiple downloads can be handled concurrently with proper resource management

## Platform Support

- **Linux**: Uses `free -m` command
- **macOS**: Uses `vm_stat` command
- **Windows**: Uses `wmic OS get FreePhysicalMemory` command
- **Fallback**: Returns error if platform detection fails

## Error Recovery

The ModelManager implements graceful error recovery:

1. **Network Errors**: Retry logic can be added
2. **Disk Space**: Checks available space before downloading
3. **Corrupted Downloads**: Hash verification can be added
4. **Memory Constraints**: Falls back to smaller models

## Future Enhancements

1. **Resumable Downloads**: Add support for partial downloads
2. **Model Validation**: Add hash verification for downloaded files
3. **Multiple Sources**: Support for multiple download mirrors
4. **GPU Detection**: Automatic GPU availability detection
5. **Model Quantization**: On-the-fly quantization support
6. **Version Management**: Multiple model version support
7. **Cache Management**: Automatic cache cleanup based on LRU

## Dependencies

- `reqwest`: HTTP client for downloads
- `tokio`: Async runtime
- `futures`: Stream utilities
- `candle-core`: Model loading and inference
- `tracing`: Logging and instrumentation

## See Also

- [ML Engine Documentation](ml_engine.md)
- [Configuration System](config.md)
- [Bootstrap System](bootstrap.md)
