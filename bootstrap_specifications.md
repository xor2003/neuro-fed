# Bootstrap Component Technical Specifications

## Overview
`bootstrap.rs` provides one-time distillation from frozen LLM to seed the PC hierarchy with meaningful initial beliefs and weights. It runs on first launch and can be re-run to update the model with new data.

## Architecture

### Core Data Structures
```rust
// Public API
pub struct Bootstrap {
    config: BootstrapConfig,
    llama_ctx: LlamaContext,
    pc_hierarchy: PredictiveCoding,
    progress_callback: Option<Box<dyn Fn(BootstrapProgress) -> bool>>,
}

#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    model_path: String,
    data_paths: Vec<String>,
    public_corpus_path: Option<String>,
    n_layers: usize,
    n_samples: usize,
    max_tokens: usize,
    distillation_method: DistillationMethod,
    save_path: String,
}

#[derive(Debug, Clone)]
pub enum DistillationMethod {
    LinearProjection,
    SimpleFFInit,
    LayerMatching,
}

#[derive(Debug, Clone)]
pub struct BootstrapProgress {
    current_step: usize,
    total_steps: usize,
    current_file: String,
    progress_percent: f32,
    message: String,
}

#[derive(Debug, Clone)]
pub struct BootstrapResult {
    beliefs: Vec<Array2<f32>>,
    weights: Vec<Array3<f32>>,
    metadata: BootstrapMetadata,
}

#[derive(Debug, Clone)]
pub struct BootstrapMetadata {
    timestamp: DateTime<Utc>,
    model_info: ModelInfo,
    data_sources: Vec<String>,
    distillation_method: DistillationMethod,
    duration_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    model_name: String,
    model_size: String,
    n_parameters: usize,
    embedding_dim: usize,
}
```

### Distillation Process Flow
```rust
impl Bootstrap {
    pub fn new(config: BootstrapConfig) -> Result<Self, BootstrapError> {
        // Initialize llama context
        let llama_ctx = LlamaContext::new(&config.model_path, config.max_tokens)?;
        
        // Initialize empty PC hierarchy (will be configured during bootstrap)
        let pc_hierarchy = PredictiveCoding::new(PCConfig::default());
        
        Ok(Bootstrap {
            config,
            llama_ctx,
            pc_hierarchy,
            progress_callback: None,
        })
    }
    
    pub fn set_progress_callback(&mut self, callback: impl Fn(BootstrapProgress) -> bool + 'static) {
        self.progress_callback = Some(Box::new(callback));
    }
    
    pub fn run(&mut self) -> Result<BootstrapResult, BootstrapError> {
        self.report_progress(0, "Starting bootstrap process");
        
        // Step 1: Load and preprocess data
        let documents = self.load_documents()?;
        self.report_progress(10, "Documents loaded");
        
        // Step 2: Extract embeddings from LLM
        let embeddings = self.extract_embeddings(&documents)?;
        self.report_progress(40, "Embeddings extracted");
        
        // Step 3: Initialize PC hierarchy
        let result = self.initialize_hierarchy(&embeddings)?;
        self.report_progress(80, "Hierarchy initialized");
        
        // Step 4: Save results
        self.save_results(&result)?;
        self.report_progress(100, "Bootstrap completed");
        
        Ok(result)
    }
}
```

## Data Processing Pipeline

### Document Loading
```rust
impl Bootstrap {
    fn load_documents(&self) -> Result<Vec<Document>, BootstrapError> {
        let mut documents = Vec::new();
        
        for path in &self.config.data_paths {
            let file_documents = self.load_document_file(path)?;
            documents.extend(file_documents);
        }
        
        // Add public corpus if specified
        if let Some(corpus_path) = &self.config.public_corpus_path {
            let corpus_docs = self.load_document_file(corpus_path)?;
            documents.extend(corpus_docs);
        }
        
        Ok(documents)
    }
    
    fn load_document_file(&self, path: &str) -> Result<Vec<Document>, BootstrapError> {
        let ext = Path::new(path).extension().unwrap_or_default();
        
        match ext.to_str() {
            Some("txt") => self.load_text_file(path),
            Some("pdf") => self.load_pdf_file(path),
            Some("md") => self.load_markdown_file(path),
            Some("docx") => self.load_docx_file(path),
            _ => Err(BootstrapError::UnsupportedFormat(path.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Document {
    text: String,
    source: String,
    metadata: DocumentMetadata,
}

#[derive(Debug, Clone)]
pub struct DocumentMetadata {
    file_path: String,
    created_at: Option<DateTime<Utc>>,
    modified_at: Option<DateTime<Utc>>,
    author: Option<String>,
}
```

### Embedding Extraction
```rust
impl Bootstrap {
    fn extract_embeddings(&self, documents: &[Document]) -> Result<Vec<EmbeddingBatch>, BootstrapError> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        
        for (i, doc) in documents.iter().enumerate() {
            // Tokenize and embed document
            let tokens = self.llama_ctx.tokenize(&doc.text)?;
            let embedding = self.llama_ctx.embed_from_tokens(&tokens)?;
            
            current_batch.push(embedding);
            
            // Create batch every N samples or at end
            if current_batch.len() >= self.config.n_samples || i == documents.len() - 1 {
                let batch = EmbeddingBatch::new(current_batch.clone());
                batches.push(batch);
                current_batch.clear();
                
                self.report_progress(
                    (i as f32 / documents.len() as f32 * 30.0 + 40.0) as usize,
                    &format!("Extracted embeddings for {} documents", i + 1)
                );
            }
        }
        
        Ok(batches)
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
    embeddings: Vec<Array2<f32>>,
    source_indices: Vec<usize>,
}

impl EmbeddingBatch {
    pub fn new(embeddings: Vec<Array2<f32>>) -> Self {
        let source_indices = (0..embeddings.len()).collect();
        EmbeddingBatch { embeddings, source_indices }
    }
}
```

## Hierarchy Initialization

### Linear Projection Method
```rust
impl Bootstrap {
    fn initialize_hierarchy(&mut self, batches: &[EmbeddingBatch]) -> Result<BootstrapResult, BootstrapError> {
        match self.config.distillation_method {
            DistillationMethod::LinearProjection => {
                self.linear_projection_init(batches)
            }
            DistillationMethod::SimpleFFInit => {
                self.simple_ff_init(batches)
            }
            DistillationMethod::LayerMatching => {
                self.layer_matching_init(batches)
            }
        }
    }
    
    fn linear_projection_init(&mut self, batches: &[EmbeddingBatch]) -> Result<BootstrapResult, BootstrapError> {
        // Collect all embeddings
        let all_embeddings: Vec<_> = batches.iter()
            .flat_map(|batch| batch.embeddings.iter().cloned())
            .collect();
        
        // Compute mean and covariance
        let mean_embedding = Self::compute_mean(&all_embeddings);
        let covariance = Self::compute_covariance(&all_embeddings, &mean_embedding);
        
        // Perform PCA or similar dimensionality reduction
        let (principal_components, explained_variance) = Self::pca(&covariance);
        
        // Initialize beliefs with principal components
        let mut beliefs = Vec::new();
        let mut weights = Vec::new();
        
        let mut current_dim = self.llama_ctx.embedding_dim();
        for target_dim in &self.config.dim_per_level {
            // Project to target dimension
            let projection_matrix = &principal_components.slice(s![.., 0..*target_dim]);
            let projected = projection_matrix.dot(&mean_embedding);
            
            beliefs.push(projected.clone());
            
            // Initialize weights (random for now)
            let weight_shape = (*target_dim, current_dim);
            let weight_matrix = Array2::random(weight_shape, RandomDistribution::Uniform);
            weights.push(weight_matrix);
            
            current_dim = *target_dim;
        }
        
        Ok(BootstrapResult {
            beliefs,
            weights,
            metadata: BootstrapMetadata::new(
                &self.config,
                &self.llama_ctx.model_info(),
                &batches
            ),
        })
    }
}
```

### Simple FF Initialization
```rust
impl Bootstrap {
    fn simple_ff_init(&mut self, batches: &[EmbeddingBatch]) -> Result<BootstrapResult, BootstrapError> {
        // Use a simple feedforward initialization
        let mut beliefs = Vec::new();
        let mut weights = Vec::new();
        
        let mut current_input = Array2::zeros((self.llama_ctx.embedding_dim(), 1));
        
        for (i, target_dim) in self.config.dim_per_level.iter().enumerate() {
            // Initialize belief as average of embeddings
            let avg_embedding = Self::compute_mean_from_batches(batches);
            beliefs.push(avg_embedding.clone());
            
            // Initialize weights with small random values
            let weight_shape = (*target_dim, current_input.shape()[0]);
            let weight_matrix = Array2::random(weight_shape, RandomDistribution::Uniform) * 0.01;
            weights.push(weight_matrix);
            
            // Update current input for next layer
            current_input = avg_embedding.clone();
        }
        
        Ok(BootstrapResult {
            beliefs,
            weights,
            metadata: BootstrapMetadata::new(
                &self.config,
                &self.llama_ctx.model_info(),
                &batches
            ),
        })
    }
}
```

## Integration with PC Hierarchy

### Applying Bootstrap Results
```rust
impl PredictiveCoding {
    pub fn apply_bootstrap(&mut self, result: &BootstrapResult) -> Result<(), PCError> {
        // Configure hierarchy based on bootstrap result
        if self.levels.len() != result.beliefs.len() {
            return Err(PCError::DimensionMismatch {
                expected: self.levels.len(),
                got: result.beliefs.len()
            });
        }
        
        // Apply beliefs and weights
        for (i, level) in self.levels.iter_mut().enumerate() {
            // Set beliefs
            if let Some(belief) = result.beliefs.get(i) {
                level.beliefs = belief.clone();
            }
            
            // Set weights
            if let Some(weight) = result.weights.get(i) {
                // Convert Array2 to Array3 (add batch dimension)
                let weight_3d = weight.into_shape((weight.shape()[0], weight.shape()[1], 1))
                    .map_err(|_| PCError::MatrixOperationFailed("Reshape failed".to_string()))?;
                level.weights = weight_3d;
            }
        }
        
        Ok(())
    }
}
```

## Configuration Examples

### Basic Bootstrap Configuration
```rust
let basic_config = BootstrapConfig {
    model_path: "models/llama-3.2-3B.Q4_K_M.gguf".to_string(),
    data_paths: vec![
        "/home/user/documents".to_string(),
        "/home/user/chat_history".to_string(),
    ],
    public_corpus_path: Some("/usr/share/public_corpus".to_string()),
    n_layers: 3,
    n_samples: 100,
    max_tokens: 2048,
    distillation_method: DistillationMethod::LinearProjection,
    save_path: "bootstrap_results.json".to_string(),
    dim_per_level: vec![512, 256, 128],
};
```

### Advanced Configuration with Progress Callback
```rust
let mut bootstrap = Bootstrap::new(advanced_config);

bootstrap.set_progress_callback(|progress| {
    println!("Bootstrap progress: {:.1}% - {}", 
             progress.progress_percent, progress.message);
    true // Continue
});
```

## Error Handling

### Custom Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("File I/O error: {0}")]
    FileIOError(String),
    
    #[error("Unsupported document format: {0}")]
    UnsupportedFormat(String),
    
    #[error("LLM embedding failed: {0}")]
    EmbeddingFailed(String),
    
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Progress callback returned false, aborting")]
    CallbackAborted,
    
    #[error("Serialization failed: {0}")]
    SerializationFailed(String),
}
```

## Performance Considerations

### Memory Management
```rust
impl Bootstrap {
    fn optimize_memory(&mut self) {
        // Pre-allocate all arrays
        self.pc_hierarchy.optimize_memory();
        
        // Use memory-efficient data structures
        self.config.data_paths.shrink_to_fit();
    }
    
    fn report_progress(&self, percent: usize, message: &str) -> Result<(), BootstrapError> {
        if let Some(callback) = &self.progress_callback {
            let progress = BootstrapProgress {
                current_step: percent,
                total_steps: 100,
                current_file: "".to_string(),
                progress_percent: percent as f32,
                message: message.to_string(),
            };
            
            if !callback(progress) {
                return Err(BootstrapError::CallbackAborted);
            }
        }
        Ok(())
    }
}
```

### Parallel Processing
```rust
impl Bootstrap {
    fn parallel_embedding(&self, documents: &[Document]) -> Result<Vec<Embedding>, BootstrapError> {
        // Process documents in parallel
        documents.par_iter()
            .map(|doc| {
                let tokens = self.llama_ctx.tokenize(&doc.text)?;
                let embedding = self.llama_ctx.embed_from_tokens(&tokens)?;
                Ok(embedding)
            })
            .collect()
    }
}
```

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_bootstrap() {
        let config = BootstrapConfig::default();
        let mut bootstrap = Bootstrap::new(config).unwrap();
        
        let result = bootstrap.run().unwrap();
        assert!(!result.beliefs.is_empty());
        assert!(!result.weights.is_empty());
    }
    
    #[test]
    fn test_progress_callback() {
        let config = BootstrapConfig::default();
        let mut bootstrap = Bootstrap::new(config).unwrap();
        
        let mut progress_called = false;
        bootstrap.set_progress_callback(|_| {
            progress_called = true;
            true
        });
        
        bootstrap.run().unwrap();
        assert!(progress_called);
    }
}
```

### Integration Tests
- Test with actual GGUF models and document files
- Verify bootstrap results can be applied to PC hierarchy
- Test different distillation methods
- Benchmark performance with various data sizes

## Dependencies

### Required
- `ndarray = "0.15"` - Core array operations
- `thiserror = "1.0"` - Error handling
- `chrono = "0.4"` - Timestamp handling
- `serde = { version = "1.0", features = ["derive"] }` - Serialization
- `serde_json = "1.0"` - JSON serialization for results

### Optional
- `rayon = "1.0"` - Parallel processing
- `pdf_extract = "0.1"` - PDF document processing
- `docx = "0.1"` - DOCX document processing
- `tracing = "0.1"` - Structured logging

## Security Considerations

- Validate all document paths before loading
- Sanitize document content before processing
- Handle large files gracefully to prevent memory exhaustion
- Use secure random number generation for weight initialization

This specification provides a complete blueprint for implementing the bootstrap component with all necessary data processing pipelines, initialization methods, and integration points needed for the Phase 0 development.