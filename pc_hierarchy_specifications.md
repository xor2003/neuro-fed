# PC Hierarchy Component Technical Specifications

## Overview
`pc_hierarchy.rs` implements the core Pure Predictive Coding (PC) algorithm based on Rao-Ballard/Friston free-energy minimization. It provides a configurable 3-6 level hierarchy for continuous learning from prediction errors.

## Architecture

### Core Data Structures
```rust
// Public API
pub struct PredictiveCoding {
    levels: Vec<PCLevel>,
    config: PCConfig,
    surprise_threshold: f32,
    free_energy: f32,
}

pub struct PCLevel {
    beliefs: Array2<f32>,        // r_l: current beliefs at level l
    predictions: Array2<f32>,    // r_hat_l: top-down predictions
    errors: Array2<f32>,         // epsilon_l: precision-weighted prediction errors
    weights: Array3<f32>,        // U_l: connection weights between levels
    precision: Array2<f32>,      // precision matrix for error weighting
}

#[derive(Debug, Clone)]
pub struct PCConfig {
    n_levels: usize,
    dim_per_level: Vec<usize>,
    learning_rate: f32,
    inference_steps: usize,
    surprise_threshold: f32,
    selective_update: bool,
    muPC_scaling: bool,
}

pub struct SurpriseStats {
    total_surprise: f32,
    high_surprise_indices: Vec<usize>,
    free_energy_history: Vec<f32>,
}
```

### Mathematical Implementation

#### Prediction Step
```rust
// r_hat_l = f(U_l * r_{l+1})
// Using simple linear transformation for f()
impl PCLevel {
    pub fn predict(&mut self) {
        // Matrix multiplication: U_l (dim_l x dim_{l+1}) * r_{l+1} (dim_{l+1} x batch_size)
        let pre_activation = &self.weights.dot(&self.beliefs_next_level());
        
        // Apply activation function (identity for simplicity, can be extended)
        self.predictions = pre_activation.clone();
    }
}
```

#### Error Computation
```rust
// epsilon_l = (r_l - r_hat_l) .* precision
impl PCLevel {
    pub fn compute_errors(&mut self) {
        // Compute raw prediction error
        let raw_error = &self.beliefs - &self.predictions;
        
        // Apply precision weighting
        self.errors = &raw_error * &self.precision;
    }
}
```

#### Inference (Free-Energy Minimization)
```rust
impl PredictiveCoding {
    pub fn infer(&mut self, input: &Array2<f32>, steps: usize) -> Result<SurpriseStats, PCError> {
        // Initialize bottom level with input
        self.levels[0].beliefs = input.clone();
        
        let mut stats = SurpriseStats::default();
        
        for step in 0..steps {
            // Upward pass: compute predictions and errors
            for l in 0..self.levels.len() - 1 {
                self.levels[l].predict();
                self.levels[l].compute_errors();
            }
            
            // Downward pass: update beliefs based on errors
            for l in (0..self.levels.len() - 1).rev() {
                // Belief update: r_l = r_l + eta * epsilon_l
                self.levels[l].beliefs = &self.levels[l].beliefs + 
                    self.config.learning_rate * &self.levels[l].errors;
                
                // Propagate error upward
                if l < self.levels.len() - 2 {
                    self.levels[l+1].beliefs = &self.levels[l+1].beliefs + 
                        &self.levels[l].errors;
                }
            }
            
            // Track free energy and surprise
            let fe = self.compute_free_energy();
            stats.free_energy_history.push(fe);
            
            if fe > self.config.surprise_threshold {
                stats.high_surprise_indices.push(step);
            }
        }
        
        stats.total_surprise = stats.free_energy_history.iter().sum::<f32>();
        Ok(stats)
    }
}
```

#### Learning (Hebbian Updates)
```rust
impl PCLevel {
    pub fn update_weights(&mut self, eta: f32) {
        // Delta U_l = eta * epsilon_l * r_{l+1}^T
        let delta_weights = eta * &self.errors.dot(&self.beliefs_next_level().t());
        self.weights = &self.weights + &delta_weights;
    }
}

impl PredictiveCoding {
    pub fn learn(&mut self, input: &Array2<f32>) -> Result<SurpriseStats, PCError> {
        // Perform inference to compute errors
        let stats = self.infer(input, self.config.inference_steps)?;
        
        // Update weights only for high-surprise components
        if self.config.selective_update {
            for (l, level) in self.levels.iter_mut().enumerate() {
                if stats.high_surprise_indices.is_empty() {
                    level.update_weights(self.config.learning_rate);
                }
            }
        } else {
            // Update all weights
            for level in self.levels.iter_mut() {
                level.update_weights(self.config.learning_rate);
            }
        }
        
        self.free_energy = stats.free_energy_history.last().unwrap_or(&0.0).clone();
        Ok(stats)
    }
}
```

## Configuration Options

### Hierarchy Configuration
```rust
impl PCConfig {
    pub fn new(n_levels: usize, dim_per_level: Vec<usize>) -> Self {
        PCConfig {
            n_levels,
            dim_per_level,
            learning_rate: 0.01,
            inference_steps: 20,
            surprise_threshold: 1.0,
            selective_update: true,
            muPC_scaling: false,
        }
    }
    
    // μPC-style scaling for deeper hierarchies
    pub fn with_muPC_scaling(mut self, enabled: bool) -> Self {
        self.muPC_scaling = enabled;
        self
    }
    
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
}
```

### Example Configurations
```rust
// Basic 3-level hierarchy (512-256-128 dimensions)
let basic_config = PCConfig::new(3, vec![512, 256, 128]);

// Deeper 6-level hierarchy with μPC scaling
let deep_config = PCConfig::new(6, vec![1024, 512, 256, 128, 64, 32])
    .with_muPC_scaling(true)
    .with_learning_rate(0.005);

// Small hierarchy for resource-constrained devices
let small_config = PCConfig::new(3, vec![128, 64, 32])
    .with_learning_rate(0.02);
```

## Integration with Other Components

### With Llama FFI
```rust
impl PredictiveCoding {
    pub fn process_text(&mut self, text: &str, llama_ctx: &LlamaContext) -> Result<SurpriseStats, PCError> {
        // Get embedding from llama FFI
        let embedding = llama_ctx.embed(text)?;
        
        // Convert to Array2<f32>
        let input = Array2::from_shape_vec(
            (embedding.dim, 1),
            embedding.data.clone()
        ).unwrap();
        
        // Perform learning
        self.learn(&input)
    }
}
```

### With Bootstrap
```rust
impl PredictiveCoding {
    pub fn bootstrap_from_llm(&mut self, layer_activations: &[Array2<f32>]) -> Result<(), PCError> {
        // Initialize beliefs and weights from LLM layer activations
        for (l, activation) in layer_activations.iter().enumerate() {
            if l < self.levels.len() {
                // Initialize beliefs with activation
                self.levels[l].beliefs = activation.clone();
                
                // Initialize weights (random initialization for now)
                let weight_shape = (
                    self.config.dim_per_level[l],
                    self.config.dim_per_level.get(l+1).unwrap_or(&0)
                );
                self.levels[l].weights = Array3::random(weight_shape, RandomDistribution::Uniform);
            }
        }
        Ok(())
    }
}
```

## Performance Optimizations

### Matrix Operations
- Use `ndarray` with BLAS backend for optimized matrix operations
- Consider `ndarray-linalg` for additional linear algebra operations
- Implement batch processing for multiple inputs

### Memory Management
```rust
impl PredictiveCoding {
    pub fn optimize_memory(&mut self) {
        // Pre-allocate all arrays to avoid reallocations
        for level in self.levels.iter_mut() {
            level.beliefs = Array2::zeros(level.beliefs.raw_dim());
            level.predictions = Array2::zeros(level.predictions.raw_dim());
            level.errors = Array2::zeros(level.errors.raw_dim());
            level.weights = Array3::zeros(level.weights.raw_dim());
            level.precision = Array2::ones(level.precision.raw_dim());
        }
    }
}
```

### Parallel Processing
```rust
impl PredictiveCoding {
    pub fn parallel_inference(&mut self, inputs: &[Array2<f32>]) -> Vec<SurpriseStats> {
        // Process multiple inputs in parallel
        inputs.par_iter()
            .map(|input| self.infer(input, self.config.inference_steps).unwrap())
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
    fn test_basic_inference() {
        let mut pc = PredictiveCoding::new(PCConfig::new(3, vec![10, 5, 3]));
        let input = Array2::ones((10, 1));
        
        let stats = pc.infer(&input, 10).unwrap();
        assert!(!stats.free_energy_history.is_empty());
        assert!(stats.total_surprise >= 0.0);
    }
    
    #[test]
    fn test_weight_updates() {
        let mut pc = PredictiveCoding::new(PCConfig::new(3, vec![10, 5, 3]));
        let input = Array2::ones((10, 1));
        
        let initial_weights = pc.levels[0].weights.clone();
        pc.learn(&input).unwrap();
        let updated_weights = pc.levels[0].weights.clone();
        
        assert_ne!(initial_weights, updated_weights);
    }
}
```

### Integration Tests
- Test with actual embeddings from llama FFI
- Verify learning converges on simple patterns
- Test with different hierarchy configurations
- Benchmark performance with various input sizes

## Error Handling

### Custom Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum PCError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    
    #[error("Invalid hierarchy configuration")]
    InvalidHierarchy,
    
    #[error("Matrix operation failed: {0}")]
    MatrixOperationFailed(String),
    
    #[error("Inference did not converge after {0} steps")]
    NonConvergence(usize),
    
    #[error("NaN or Inf detected in computation")]
    NumericalError,
}
```

## Dependencies

### Required
- `ndarray = "0.15"` - Core array operations
- `ndarray-linalg = "0.16"` - Linear algebra operations (optional, for advanced features)
- `thiserror = "1.0"` - Error handling
- `rand = "0.8"` - Random weight initialization

### Optional
- `rayon = "1.0"` - Parallel processing
- `serde = { version = "1.0", features = ["derive"] }` - Serialization for configuration

## Mathematical Foundations

### Free Energy Calculation
```rust
impl PredictiveCoding {
    fn compute_free_energy(&self) -> f32 {
        // F = sum(epsilon_l^2) + complexity_penalty
        let mut free_energy = 0.0;
        
        for level in &self.levels {
            let error_norm = level.errors.mapv(|x| x.powi(2)).sum();
            free_energy += error_norm;
        }
        
        // Add complexity penalty (weight norms)
        for level in &self.levels {
            let weight_norm = level.weights.mapv(|x| x.powi(2)).sum();
            free_energy += 0.01 * weight_norm; // regularization parameter
        }
        
        free_energy
    }
}
```

### μPC Scaling (for deeper hierarchies)
```rust
impl PCLevel {
    pub fn apply_muPC_scaling(&mut self, depth: usize) {
        if depth > 3 {
            // Scale learning rate based on depth
            let depth_factor = 1.0 / (depth as f32);
            self.weights = &self.weights * depth_factor;
        }
    }
}
```

## Configuration Examples

### Production Configuration
```rust
let production_config = PCConfig::new(4, vec![768, 384, 192, 96])
    .with_learning_rate(0.005)
    .with_inference_steps(30)
    .with_selective_update(true);
```

### Development Configuration
```rust
let dev_config = PCConfig::new(3, vec![128, 64, 32])
    .with_learning_rate(0.02)
    .with_inference_steps(10)
    .with_selective_update(false);
```

This specification provides a complete blueprint for implementing the PC hierarchy component with all necessary mathematical foundations, performance optimizations, and integration points.