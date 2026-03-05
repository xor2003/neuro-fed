// src/pc_hierarchy.rs
// Pure Predictive Coding (PC) implementation based on Rao-Ballard/Friston free-energy minimization
// Migrated from ndarray to candle-core for GPU acceleration

use candle_core::{Device, Tensor, DType, Result as CandleResult, IndexOp};
use std::error::Error;
use std::fmt;

use crate::knowledge_filter::{PrecisionCalculator, PrecisionConfig, PrecisionContext};
use crate::persistence::{PCPersistence, PersistenceError};

#[derive(Debug)]
pub struct PCError(String);

impl fmt::Display for PCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PCError: {}", self.0)
    }
}

impl Error for PCError {}

impl From<candle_core::Error> for PCError {
    fn from(err: candle_core::Error) -> Self {
        PCError(format!("Candle error: {}", err))
    }
}

#[derive(Debug, Clone)]
pub struct PCConfig {
    pub n_levels: usize,
    pub dim_per_level: Vec<usize>,
    pub learning_rate: f32,
    pub inference_steps: usize,
    pub surprise_threshold: f32,
    pub convergence_threshold: f32,
    pub selective_update: bool,
    pub mu_pc_scaling: bool,
    // Precision weighting configuration
    pub enable_precision_weighting: bool,
    pub free_energy_drop_threshold: f32,
    pub default_precision: f32,
    pub min_precision: f32,
    pub max_precision: f32,
    pub free_energy_history_size: usize,
    pub enable_code_verification: bool,
    pub enable_nostr_zap_tracking: bool,
    pub min_zaps_for_consensus: usize,
    /// Path to SQLite database for persisting PC weights (optional)
    pub persistence_db_path: Option<String>,
}

impl PCConfig {
    pub fn new(n_levels: usize, dim_per_level: Vec<usize>) -> Self {
        PCConfig {
            n_levels,
            dim_per_level,
            learning_rate: 0.01,
            inference_steps: 20,
            surprise_threshold: 1.0,
            convergence_threshold: 0.01,
            selective_update: true,
            mu_pc_scaling: true,
            // Precision weighting defaults
            enable_precision_weighting: false,
            free_energy_drop_threshold: 0.5,
            default_precision: 0.3,
            min_precision: 0.1,
            max_precision: 1.0,
            free_energy_history_size: 10,
            enable_code_verification: false,
            enable_nostr_zap_tracking: false,
            min_zaps_for_consensus: 3,
            persistence_db_path: None,
        }
    }

    pub fn with_mu_pc_scaling(mut self, enabled: bool) -> Self {
        self.mu_pc_scaling = enabled;
        self
    }

    pub fn with_convergence_threshold(mut self, threshold: f32) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
}

#[derive(Debug, Clone)]
pub struct PCLevel {
    pub beliefs: Tensor,
    pub predictions: Tensor,
    pub errors: Tensor,
    pub weights: Tensor,
    pub precision: Tensor,
    device: Device,
}

impl PCLevel {
    pub fn new(input_dim: usize, output_dim: usize, device: &Device) -> CandleResult<Self> {
        let beliefs = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let predictions = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let errors = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let weights = Tensor::randn(0f32, 0.01f32, (input_dim, output_dim), device)?;
        let precision = Tensor::ones((input_dim, 1), DType::F32, device)?;

        Ok(PCLevel {
            beliefs,
            predictions,
            errors,
            weights,
            precision,
            device: device.clone(),
        })
    }

    pub fn predict(&mut self) -> CandleResult<()> {
        // r_hat_l = U_l * r_{l+1}
        // Matrix multiplication: weights (input_dim x output_dim) * beliefs_next_level (output_dim x batch)
        let beliefs_next = self.beliefs_next_level()?;
        self.predictions = self.weights.matmul(&beliefs_next)?;
        Ok(())
    }

    pub fn compute_errors(&mut self) -> CandleResult<()> {
        // epsilon_l = (r_l - r_hat_l) .* precision
        let raw_error = (&self.beliefs - &self.predictions)?;
        self.errors = (&raw_error * &self.precision)?;
        Ok(())
    }

    fn beliefs_next_level(&self) -> CandleResult<Tensor> {
        // This would normally come from the level above
        // For now, return a zero tensor of appropriate shape
        let (_, output_dim) = self.weights.shape().dims2()?;
        let (_, batch) = self.beliefs.shape().dims2()?;
        Tensor::zeros((output_dim, batch), DType::F32, &self.device)
    }

    pub fn update_weights(&mut self, eta: f32, next_level_beliefs: &Tensor, precision: Option<&Tensor>, mu_pc_scaling: bool) -> CandleResult<()> {
        // Delta U_l = eta * epsilon_l * r_{l+1}^T * π
        // If precision is provided, apply element-wise multiplication
        let next_t = next_level_beliefs.t()?;
        let matmul_result = self.errors.matmul(&next_t)?;
        
        // --- NEW: MU-PC SCALING MATH ---
        let (input_dim, _) = self.weights.shape().dims2()?;
        let mut effective_lr = eta;
        if mu_pc_scaling {
            // Scale learning rate by 1 / sqrt(dimension) to prevent exploding gradients
            effective_lr = eta / (input_dim as f32).sqrt();
        }
        // --------------------------------
        
        // Create scalar eta tensor and broadcast to match matmul_result shape
        let eta_tensor = Tensor::from_slice(&[effective_lr], (1, 1), &matmul_result.device())?
            .broadcast_as(matmul_result.shape())?;
        let mut delta_weights = matmul_result.mul(&eta_tensor)?;
        
        if let Some(precision_matrix) = precision {
            // Apply precision weighting element-wise
            // Note: precision_matrix should have shape (input_dim, 1) for broadcasting
            // Broadcast precision_matrix to match delta_weights shape
            let broadcasted_precision = precision_matrix.broadcast_as(delta_weights.shape())?;
            delta_weights = (&delta_weights * &broadcasted_precision)?;
        }
        
        // --- NEW: GRADIENT CLIPPING TO PREVENT NUMERICAL EXPLOSION ---
        let clip_threshold = 1.0; // Maximum L2 norm for weight updates
        let norm_sq = delta_weights.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = norm_sq.sqrt();
        
        if norm > clip_threshold {
            // Scale delta_weights to have norm = clip_threshold
            let scale = clip_threshold / norm;
            let scale_tensor = Tensor::from_slice(&[scale], (1, 1), &delta_weights.device())?
                .broadcast_as(delta_weights.shape())?;
            delta_weights = (&delta_weights * &scale_tensor)?;
            tracing::debug!("Gradient clipped: norm {} > threshold {}, scaled by {}", norm, clip_threshold, scale);
        }
        // -------------------------------------------------------------
        
        // Use broadcast_add for in-place addition without reallocating
        self.weights = self.weights.broadcast_add(&delta_weights)?;
        Ok(())
    }
    
    /// Legacy method for backward compatibility
    pub fn update_weights_legacy(&mut self, eta: f32, next_level_beliefs: &Tensor) -> CandleResult<()> {
        self.update_weights(eta, next_level_beliefs, None, false)
    }
}

#[derive(Debug, Clone)]
pub struct SurpriseStats {
    pub total_surprise: f32,
    pub high_surprise_indices: Vec<usize>,
    pub free_energy_history: Vec<f32>,
}

impl Default for SurpriseStats {
    fn default() -> Self {
        SurpriseStats {
            total_surprise: 0.0,
            high_surprise_indices: Vec::new(),
            free_energy_history: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictiveCoding {
    pub levels: Vec<PCLevel>,
    pub config: PCConfig,
    pub surprise_threshold: f32,
    pub free_energy: f32,
    pub precision_calculator: Option<PrecisionCalculator>,
    device: Device,
}

impl PredictiveCoding {
    pub fn new(config: PCConfig) -> Result<Self, PCError> {
        Self::new_with_device(config, &Device::Cpu)
    }

    pub fn new_with_device(config: PCConfig, device: &Device) -> Result<Self, PCError> {
        if config.n_levels < 2 {
            return Err(PCError("Hierarchy must have at least 2 levels".to_string()));
        }
        
        if config.dim_per_level.len() != config.n_levels {
            return Err(PCError(format!(
                "Dimension vector length {} does not match n_levels {}",
                config.dim_per_level.len(),
                config.n_levels
            )));
        }

        let mut levels = Vec::new();
        
        // Create levels with decreasing dimensions
        for i in 0..config.n_levels {
            let input_dim = config.dim_per_level[i];
            let output_dim = if i < config.n_levels - 1 {
                config.dim_per_level[i + 1]
            } else {
                // Last level has no output
                1
            };
            
            let mut level = PCLevel::new(input_dim, output_dim, device)?;
            
            // Initialize weights with small random values (already done in PCLevel::new)
            // Scale down weights
            if i < config.n_levels - 1 {
                level.weights = (&level.weights * 0.01)?;
            }
            
            levels.push(level);
        }

        // Create precision calculator if precision weighting is enabled
        let precision_calculator = if config.enable_precision_weighting {
            let precision_config = PrecisionConfig {
                free_energy_drop_threshold: config.free_energy_drop_threshold,
                default_precision: config.default_precision,
                min_precision: config.min_precision,
                max_precision: config.max_precision,
                free_energy_history_size: config.free_energy_history_size,
                enable_code_verification: config.enable_code_verification,
                enable_nostr_zap_tracking: config.enable_nostr_zap_tracking,
                min_zaps_for_consensus: config.min_zaps_for_consensus,
                trusted_node_keys: Vec::new(), // Will be populated from config if needed
            };
            Some(PrecisionCalculator::new(precision_config))
        } else {
            None
        };

        Ok(PredictiveCoding {
            levels,
            config: config.clone(),
            surprise_threshold: config.surprise_threshold,
            free_energy: 0.0,
            precision_calculator,
            device: device.clone(),
        })
    }

    pub fn infer(&mut self, input: &Tensor, steps: usize) -> Result<SurpriseStats, PCError> {
        // Auto-align input tensor shape: PC expects (embedding_dim, 1)
        // If input is (1, embedding_dim), transpose it
        let input = if input.shape().dims()[0] == 1 && input.shape().dims()[1] == self.config.dim_per_level[0] {
            match input.t() {
                Ok(transposed) => {
                    tracing::debug!("PC infer: transposed input from {:?} to {:?}", input.shape(), transposed.shape());
                    transposed
                }
                Err(e) => return Err(PCError(format!("Failed to transpose input tensor: {}", e))),
            }
        } else {
            input.clone()
        };
        
        let (input_dim, _) = input.shape().dims2()?;
        if input_dim != self.config.dim_per_level[0] {
            tracing::debug!("PC infer: input_dim={}, config.dim_per_level[0]={}, config.dim_per_level={:?}",
                input_dim, self.config.dim_per_level[0], self.config.dim_per_level);
            return Err(PCError(format!("Input dimension {} does not match level 0 dimension {}",
                input_dim, self.config.dim_per_level[0])));
        }

        tracing::debug!("PC infer: input shape {:?}, steps {}", input.shape(), steps);
        
        // Initialize bottom level with input
        self.levels[0].beliefs = input;
        
        let mut stats = SurpriseStats::default();
        
        for step in 0..steps {
            // Upward pass: compute predictions and errors
            for l in 0..self.levels.len() - 1 {
                self.levels[l].predict()?;
                self.levels[l].compute_errors()?;
            }
            
            // Downward pass: update beliefs based on errors
            for l in (0..self.levels.len() - 1).rev() {
                // Belief update: r_l = r_l - eta * epsilon_l (MUST BE MINUS to minimize free energy)
                let lr_tensor = Tensor::from_slice(&[self.config.learning_rate], (1, 1), &self.levels[l].errors.device())?
                    .broadcast_as(self.levels[l].errors.shape())?;
                let update = self.levels[l].errors.mul(&lr_tensor)?;
                
                // 🔴 BUG FIX: Changed + to -
                self.levels[l].beliefs = (&self.levels[l].beliefs - &update)?;
                
                // Propagate error upward to influence beliefs at next level: r_{l+1} += eta * U_l^T · epsilon_l
                // Only propagate if there is a next level (l+1 exists)
                if l + 1 < self.levels.len() {
                    let weight_transpose = self.levels[l].weights.t()?;
                    let matmul_result = weight_transpose.matmul(&self.levels[l].errors)?;
                    let lr_tensor2 = Tensor::from_slice(&[self.config.learning_rate], (1, 1), &matmul_result.device())?
                        .broadcast_as(matmul_result.shape())?;
                    let belief_update = matmul_result.mul(&lr_tensor2)?;
                    // NOTE: Upward propagation stays PLUS (Mathematically correct: r_{l+1} = r_{l+1} - eta * dF/dr_{l+1})
                    self.levels[l+1].beliefs = (&self.levels[l+1].beliefs + &belief_update)?;
                }
            }
            
            // Track free energy and surprise
            let fe = self.compute_free_energy()?;
            stats.free_energy_history.push(fe);
            
            if fe > self.config.surprise_threshold {
                stats.high_surprise_indices.push(step);
            }

            // ИЗМЕНИ ЭТОТ БЛОК:
            // Early exiting: stop ONLY if FE is extremely low AND we've thought for at least 3 steps
            if fe < 0.0001 && step > 3 {
                tracing::debug!("PC inference converged early at step {} (FE: {:.4})", step, fe);
                break;
            }
        }
        
        stats.total_surprise = stats.free_energy_history.iter().sum::<f32>();
        Ok(stats)
    }

    pub fn learn(&mut self, input: &Tensor, context: Option<PrecisionContext>) -> Result<SurpriseStats, PCError> {
        // Perform inference to compute errors
        let stats = self.infer(input, self.config.inference_steps)?;
        
        // Calculate free energy drop for gossip trigger
        let free_energy_drop = if stats.free_energy_history.len() >= 2 {
            let initial_fe = stats.free_energy_history.first().unwrap_or(&0.0);
            let final_fe = stats.free_energy_history.last().unwrap_or(&0.0);
            initial_fe - final_fe  // Positive drop means free energy decreased (good)
        } else {
            0.0
        };
        
        // Check if free energy drop exceeds threshold for gossip trigger
        if free_energy_drop > self.config.free_energy_drop_threshold {
            tracing::info!(
                "GOSSIP TRIGGER: Free energy drop {:.4} exceeds threshold {:.4}",
                free_energy_drop,
                self.config.free_energy_drop_threshold
            );
            
            // TODO: Package delta weights into NIP-8700 event and send to mpsc::channel
            // The delta weights (ΔU) would need to be captured during weight updates
            // For now, log that gossip would be triggered
            tracing::debug!("Would package delta weights for Nostr gossip (NIP-8700)");
        }
        
        // Record free energy for tracking
        if let Some(ref mut calculator) = self.precision_calculator {
            let current_free_energy = stats.free_energy_history.last().unwrap_or(&0.0);
            calculator.record_free_energy(*current_free_energy);
        }
        
        // Clone beliefs for all levels to avoid borrow issues
        let next_level_beliefs: Vec<Tensor> = self.levels.iter().map(|level| level.beliefs.clone()).collect();
        
        // Update weights only for high-surprise components
        if self.config.selective_update {
            for l in 0..self.levels.len() - 1 {
                if stats.high_surprise_indices.is_empty() {
                    // Create precision matrix for this level if enabled
                    let level_precision_matrix = if let Some(ref calculator) = self.precision_calculator {
                        if let Some(ref context) = context {
                            let precision_result = calculator.calculate_precision(context);
                            // Create a precision matrix with the input dimension of this level
                            let (input_dim, _) = self.levels[l].beliefs.shape().dims2()?;
                            let ones = Tensor::ones((input_dim, 1), DType::F32, &self.device)?;
                            let precision_tensor = Tensor::from_slice(&[precision_result.precision], (1, 1), &self.device)?
                                .broadcast_as(ones.shape())?;
                            Some(ones.mul(&precision_tensor)?)
                        } else {
                            // Default precision matrix (all ones) if no context provided
                            let (input_dim, _) = self.levels[l].beliefs.shape().dims2()?;
                            let ones = Tensor::ones((input_dim, 1), DType::F32, &self.device)?;
                            let default_precision_tensor = Tensor::from_slice(&[self.config.default_precision], (1, 1), &self.device)?
                                .broadcast_as(ones.shape())?;
                            Some(ones.mul(&default_precision_tensor)?)
                        }
                    } else {
                        None
                    };
                    
                    self.levels[l].update_weights(
                        self.config.learning_rate,
                        &next_level_beliefs[l + 1],
                        level_precision_matrix.as_ref(),
                        self.config.mu_pc_scaling
                    )?;
                }
            }
        } else {
            // Update all weights
            for l in 0..self.levels.len() - 1 {
                // Create precision matrix for this level if enabled
                let level_precision_matrix = if let Some(ref calculator) = self.precision_calculator {
                    if let Some(ref context) = context {
                        let precision_result = calculator.calculate_precision(context);
                        // Create a precision matrix with the input dimension of this level
                        let (input_dim, _) = self.levels[l].beliefs.shape().dims2()?;
                        let ones = Tensor::ones((input_dim, 1), DType::F32, &self.device)?;
                        let precision_tensor = Tensor::from_slice(&[precision_result.precision], (1, 1), &self.device)?
                            .broadcast_as(ones.shape())?;
                        Some(ones.mul(&precision_tensor)?)
                    } else {
                        // Default precision matrix (all ones) if no context provided
                        let (input_dim, _) = self.levels[l].beliefs.shape().dims2()?;
                        let ones = Tensor::ones((input_dim, 1), DType::F32, &self.device)?;
                        let default_precision_tensor = Tensor::from_slice(&[self.config.default_precision], (1, 1), &self.device)?
                            .broadcast_as(ones.shape())?;
                        Some(ones.mul(&default_precision_tensor)?)
                    }
                } else {
                    None
                };
                
                self.levels[l].update_weights(
                    self.config.learning_rate,
                    &next_level_beliefs[l + 1],
                    level_precision_matrix.as_ref(),
                    self.config.mu_pc_scaling
                )?;
            }
        }
        
        self.free_energy = stats.free_energy_history.last().unwrap_or(&0.0).clone();
        Ok(stats)
    }
    
    /// Legacy method for backward compatibility
    pub fn learn_legacy(&mut self, input: &Tensor) -> Result<SurpriseStats, PCError> {
        self.learn(input, None)
    }

    fn compute_free_energy(&self) -> Result<f32, PCError> {
        // Simplified free energy computation
        // In a full implementation, this would involve KL divergence and other terms
        let mut fe = 0.0f32;
        
        for l in 0..self.levels.len() - 1 {
            let prediction_error = (&self.levels[l].beliefs - &self.levels[l].predictions)?;
            let squared_error = prediction_error.sqr()?.sum_all()?.to_scalar::<f32>()?;
            
            // --- FIX: Convert SSE to MSE ---
            // Divide the total error by the number of dimensions
            let (dim, _) = self.levels[l].beliefs.shape().dims2()?;
            fe += squared_error / (dim as f32);
        }
        
        Ok(fe / self.levels.len() as f32)
    }

    pub fn surprise(&mut self, input: &Tensor) -> Result<f32, PCError> {
        let stats = self.infer(input, 1)?;
        Ok(stats.free_energy_history.last().unwrap_or(&0.0).clone())
    }

    pub fn get_beliefs(&self, level: usize) -> Result<&Tensor, PCError> {
        if level >= self.levels.len() {
            return Err(PCError("Level index out of bounds".to_string()));
        }
        Ok(&self.levels[level].beliefs)
    }

    pub fn get_predictions(&self, level: usize) -> Result<&Tensor, PCError> {
        if level >= self.levels.len() {
            return Err(PCError("Level index out of bounds".to_string()));
        }
        Ok(&self.levels[level].predictions)
    }

    pub fn get_errors(&self, level: usize) -> Result<&Tensor, PCError> {
        if level >= self.levels.len() {
            return Err(PCError("Level index out of bounds".to_string()));
        }
        Ok(&self.levels[level].errors)
    }

    /// Export memory as a JSON-serializable structure for human-readable inspection.
    /// Returns a serde_json::Value containing hierarchy configuration, weights, beliefs, and errors.
    pub fn export_memory(&self) -> Result<serde_json::Value, PCError> {
        use serde_json::json;

        let mut levels_json = Vec::new();
        for (i, level) in self.levels.iter().enumerate() {
            // Extract tensor data as flat vectors (could be large)
            let weights_shape = level.weights.shape().dims2()?;
            let beliefs_shape = level.beliefs.shape().dims2()?;
            let errors_shape = level.errors.shape().dims2()?;

            // Convert tensors to Vec<f32> (flatten)
            let weights_vec = level.weights.flatten_all()?.to_vec1::<f32>()?;
            let beliefs_vec = level.beliefs.flatten_all()?.to_vec1::<f32>()?;
            let errors_vec = level.errors.flatten_all()?.to_vec1::<f32>()?;

            // Limit output size: take first 10 elements for preview
            let weights_preview: Vec<f32> = weights_vec.iter().take(10).cloned().collect();
            let beliefs_preview: Vec<f32> = beliefs_vec.iter().take(10).cloned().collect();
            let errors_preview: Vec<f32> = errors_vec.iter().take(10).cloned().collect();

            levels_json.push(json!({
                "level": i,
                "input_dim": weights_shape.0,
                "output_dim": weights_shape.1,
                "beliefs_shape": [beliefs_shape.0, beliefs_shape.1],
                "errors_shape": [errors_shape.0, errors_shape.1],
                "weights_preview": weights_preview,
                "beliefs_preview": beliefs_preview,
                "errors_preview": errors_preview,
                "weights_total_elements": weights_vec.len(),
                "beliefs_total_elements": beliefs_vec.len(),
                "errors_total_elements": errors_vec.len(),
            }));
        }

        let config_json = json!({
            "n_levels": self.config.n_levels,
            "dim_per_level": self.config.dim_per_level,
            "learning_rate": self.config.learning_rate,
            "inference_steps": self.config.inference_steps,
            "surprise_threshold": self.config.surprise_threshold,
            "convergence_threshold": self.config.convergence_threshold,
            "selective_update": self.config.selective_update,
            "mu_pc_scaling": self.config.mu_pc_scaling,
            "enable_precision_weighting": self.config.enable_precision_weighting,
            "free_energy_drop_threshold": self.config.free_energy_drop_threshold,
            "default_precision": self.config.default_precision,
            "min_precision": self.config.min_precision,
            "max_precision": self.config.max_precision,
            "free_energy_history_size": self.config.free_energy_history_size,
            "enable_code_verification": self.config.enable_code_verification,
            "enable_nostr_zap_tracking": self.config.enable_nostr_zap_tracking,
            "min_zaps_for_consensus": self.config.min_zaps_for_consensus,
            "persistence_db_path": self.config.persistence_db_path,
        });

        let result = json!({
            "config": config_json,
            "levels": levels_json,
            "free_energy": self.free_energy,
            "surprise_threshold": self.surprise_threshold,
        });

        Ok(result)
    }

    /// Dream (top‑down generative translation): produce a bottom‑level representation from a top‑level seed.
    /// The seed should have shape `(batch, dim_per_level.last())`. The returned tensor has shape
    /// `(batch, dim_per_level[0])`.
    pub fn dream(&self, top_seed: &Tensor) -> Result<Tensor, PCError> {
        let (batch, top_dim) = top_seed.shape().dims2()?;
        let expected_top_dim = self.config.dim_per_level.last().ok_or_else(||
            PCError("Hierarchy has no levels".to_string())
        )?;
        if top_dim != *expected_top_dim {
            return Err(PCError(format!(
                "Top seed dimension {} does not match top level dimension {}",
                top_dim, expected_top_dim
            )));
        }

        // Start with the seed as the current representation at the top level
        let mut current = top_seed.clone();

        // Traverse levels from top to bottom (excluding the bottom level, which is the target)
        for l in (0..self.levels.len() - 1).rev() {
            // Use the weight matrix U_l^T to project from level l+1 to level l
            let weight_t = self.levels[l].weights.t()?;
            // current shape: (batch, dim_{l+1})
            // weight_t shape: (dim_{l+1}, dim_l)
            // result shape: (batch, dim_l)
            current = current.matmul(&weight_t)?;
            // Optionally apply a non‑linearity? For pure linear generative mapping we keep as is.
            // Could add ReLU or sigmoid, but we keep it linear for now.
        }

        tracing::info!(
            "PC dreaming completed: seed shape ({}, {}), generated shape {:?}",
            batch,
            top_dim,
            current.shape()
        );

        Ok(current)
    }

    pub fn inject_pretrained_weights(&mut self, ml_engine: &crate::ml_engine::MLEngine) -> Result<(), PCError> {
        tracing::info!("Injecting pre-trained weights from GGUF into PC hierarchy");
        
        let knowledge_matrix = ml_engine.extract_knowledge_matrix()
            .map_err(|e| PCError(format!("Failed to extract knowledge matrix: {}", e)))?;
        
        let knowledge_shape = knowledge_matrix.shape();
        
        let levels_len = self.levels.len();
        for (i, level) in self.levels.iter_mut().enumerate() {
            // Для последнего слоя веса не обновляем (у него выход 1)
            if i == levels_len - 1 { continue; }

            let (target_rows, target_cols) = level.weights.shape().dims2()?;
            let (src_rows, src_cols) = knowledge_shape.dims2()?;
            
            // Вычисляем, сколько мы можем отрезать (Surgical Slicing)
            let rows_to_take = src_rows.min(target_rows);
            let cols_to_take = src_cols.min(target_cols);
            
            if rows_to_take > 0 && cols_to_take > 0 {
                // Отрезаем нужный кусок от большой матрицы GGUF
                let sliced = knowledge_matrix
                    .narrow(0, 0, rows_to_take).unwrap()
                    .narrow(1, 0, cols_to_take).unwrap();
                
                // Если отрезанный кусок идеально совпадает с нашим слоем — вставляем!
                if rows_to_take == target_rows && cols_to_take == target_cols {
                    // Assign the sliced tensor directly (Note: user suggested .set() but we use Tensor)
                    level.weights = sliced;
                    tracing::info!("✅ Level {}: Successfully injected {}x{} weights from GGUF", i, target_rows, target_cols);
                } else {
                    tracing::warn!("Level {}: Sliced matrix {}x{} doesn't fill target {}x{}. Keeping random.", 
                        i, rows_to_take, cols_to_take, target_rows, target_cols);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_basic_hierarchy_creation() -> Result<(), PCError> {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let pc = PredictiveCoding::new(config)?;
        assert_eq!(pc.levels.len(), 3);
        assert_eq!(pc.levels[0].beliefs.shape().dims2()?.0, 512);
        assert_eq!(pc.levels[1].beliefs.shape().dims2()?.0, 256);
        assert_eq!(pc.levels[2].beliefs.shape().dims2()?.0, 128);
        Ok(())
    }

    #[test]
    fn test_inference() -> Result<(), PCError> {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config)?;
        
        // Create random input
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (512, 1), &device)?;
        
        let stats = pc.infer(&input, 10)?;
        assert_eq!(stats.free_energy_history.len(), 10);
        assert!(stats.total_surprise >= 0.0);
        Ok(())
    }

    #[test]
    fn test_learning() -> Result<(), PCError> {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (512, 1), &device)?;
        
        let stats = pc.learn_legacy(&input)?;
        assert!(stats.free_energy_history.len() > 0);
        Ok(())
    }

    #[test]
    fn test_dimension_mismatch_error() -> Result<(), PCError> {
        // Test that incorrect input dimension triggers proper error
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        // Create input with wrong dimension (1 instead of 512) - must be 2D tensor
        let wrong_input = Tensor::ones((1, 1), candle_core::DType::F32, &device)?;
        
        // This should fail with dimension mismatch error
        let result = pc.infer(&wrong_input, 10);
        assert!(result.is_err());
        
        // Verify error message contains dimension information
        let err = result.unwrap_err();
        let err_msg = err.to_string();
        assert!(err_msg.contains("dimension"));
        assert!(err_msg.contains("512"));
        
        Ok(())
    }
    
    #[test]
    fn test_correct_dimension_works() -> Result<(), PCError> {
        // Test that correct input dimension works
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        let correct_input = Tensor::ones((512, 1), candle_core::DType::F32, &device)?;
        
        // This should succeed
        let result = pc.infer(&correct_input, 10);
        assert!(result.is_ok());
        
        Ok(())
    }

    #[test]
    fn test_surprise() -> Result<(), PCError> {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (512, 1), &device)?;
        let surprise = pc.surprise(&input)?;
        assert!(surprise >= 0.0);
        Ok(())
    }

    #[test]
    fn test_accessors() -> Result<(), PCError> {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let pc = PredictiveCoding::new(config)?;
        
        assert!(pc.get_beliefs(0).is_ok());
        assert!(pc.get_predictions(0).is_ok());
        assert!(pc.get_errors(0).is_ok());
        
        // Test out of bounds
        assert!(pc.get_beliefs(10).is_err());
        Ok(())
    }

    #[test]
    fn test_edge_case_zero_input() -> Result<(), PCError> {
        let config = PCConfig::new(3, vec![8, 4, 2]).with_convergence_threshold(-1.0);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        let zero_input = Tensor::zeros((8, 1), DType::F32, &device)?;
        
        // Inference should work
        let stats = pc.infer(&zero_input, 5)?;
        assert_eq!(stats.free_energy_history.len(), 5);
        
        // Surprise should be zero or very small
        let surprise = pc.surprise(&zero_input)?;
        assert!(surprise >= 0.0);
        assert!(surprise < 0.001, "Zero input should produce near-zero surprise");
        Ok(())
    }

    #[test]
    fn test_edge_case_single_level_hierarchy() -> Result<(), PCError> {
        // Test minimum hierarchy size (2 levels)
        let config = PCConfig::new(2, vec![4, 2]);
        let pc = PredictiveCoding::new(config.clone())?;
        assert_eq!(pc.levels.len(), 2);
        
        // Should work with inference
        let mut pc = PredictiveCoding::new(config)?;
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (4, 1), &device)?;
        let stats = pc.infer(&input, 3)?;
        assert_eq!(stats.free_energy_history.len(), 3);
        Ok(())
    }

    #[test]
    fn test_edge_case_large_learning_rate() -> Result<(), PCError> {
        // Test with very large learning rate (should still work, though may be unstable)
        let config = PCConfig::new(3, vec![8, 4, 2]).with_learning_rate(1.0);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (8, 1), &device)?;
        
        // Should not panic
        let result = pc.infer(&input, 2);
        assert!(result.is_ok());
        Ok(())
    }

    #[test]
    fn test_edge_case_small_learning_rate() -> Result<(), PCError> {
        // Test with very small learning rate
        let config = PCConfig::new(3, vec![8, 4, 2]).with_learning_rate(0.0001);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (8, 1), &device)?;
        
        let stats1 = pc.infer(&input, 5)?;
        let _ = pc.learn_legacy(&input)?;
        let stats2 = pc.infer(&input, 5)?;
        
        // With tiny learning rate, free energy should change very little
        let fe1 = stats1.free_energy_history.last().unwrap();
        let fe2 = stats2.free_energy_history.last().unwrap();
        let change = (fe2 - fe1).abs();
        assert!(change < 0.01, "Free energy change too large for small LR: {}", change);
        Ok(())
    }

    #[test]
    fn test_property_beliefs_non_nan() -> Result<(), PCError> {
        // Property: beliefs should never contain NaN values
        let config = PCConfig::new(3, vec![12, 6, 3]);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        for _ in 0..10 {
            let input = Tensor::randn(0f32, 1.0, (12, 1), &device)?;
            let _ = pc.infer(&input, 3)?;
            
            // Check all beliefs
            for level in &pc.levels {
                let beliefs_vec = level.beliefs.flatten_all()?.to_vec1::<f32>()?;
                assert!(!beliefs_vec.iter().any(|&x| x.is_nan()), "Beliefs contain NaN");
                
                let predictions_vec = level.predictions.flatten_all()?.to_vec1::<f32>()?;
                assert!(!predictions_vec.iter().any(|&x| x.is_nan()), "Predictions contain NaN");
                
                let errors_vec = level.errors.flatten_all()?.to_vec1::<f32>()?;
                assert!(!errors_vec.iter().any(|&x| x.is_nan()), "Errors contain NaN");
            }
        }
        Ok(())
    }

    #[test]
    fn test_property_surprise_non_negative() -> Result<(), PCError> {
        // Property: surprise should always be non-negative
        let config = PCConfig::new(3, vec![10, 5, 2]);
        let mut pc = PredictiveCoding::new(config)?;
        
        let device = Device::Cpu;
        for _ in 0..5 {
            let input = Tensor::randn(0f32, 1.0, (10, 1), &device)?;
            let surprise = pc.surprise(&input)?;
            assert!(surprise >= 0.0, "Surprise should be non-negative, got {}", surprise);
        }
        Ok(())
    }

    #[test]
    fn test_config_validation() -> Result<(), PCError> {
        // Test invalid configurations
        let config = PCConfig::new(1, vec![512]); // Only 1 level
        let result = PredictiveCoding::new(config);
        assert!(result.is_err(), "Should reject hierarchy with less than 2 levels");
        
        let config = PCConfig::new(3, vec![512, 256]); // Mismatched dimensions
        let result = PredictiveCoding::new(config);
        assert!(result.is_err(), "Should reject mismatched dimension vector");
        
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let pc = PredictiveCoding::new(config)?;
        assert_eq!(pc.levels.len(), 3);
        Ok(())
    }

    #[test]
    fn test_early_exiting_prevention() -> Result<(), PCError> {
        let device = Device::Cpu;
        let config = PCConfig::new(3, vec![2048, 1024, 512]);
        let mut pc = PredictiveCoding::new_with_device(config, &device)?;
        
        let zero_input = Tensor::zeros((2048, 1), DType::F32, &device)
            .map_err(|e| PCError(e.to_string()))?;
        
        // Инференс нулей мгновенно дает нулевую ошибку.
        // Мы ожидаем, что он выполнит МИНИМУМ 4 шага (step 0, 1, 2, 3), прежде чем прервется.
        let stats = pc.infer(&zero_input, 15)?;
        
        assert!(stats.free_energy_history.len() > 3, "Inference must not exit before step 3");
        Ok(())
    }

    #[test]
    fn test_free_energy_is_mse_not_sse() -> Result<(), PCError> {
        let device = Device::Cpu;
        
        // Create a TINY hierarchy
        let config_small = PCConfig::new(2, vec![10, 5]);
        let mut pc_small = PredictiveCoding::new_with_device(config_small, &device)?;
        let input_small = Tensor::randn(0f32, 1.0, (10, 1), &device)?;
        
        // Create a MASSIVE hierarchy
        let config_large = PCConfig::new(2, vec![2000, 1000]);
        let mut pc_large = PredictiveCoding::new_with_device(config_large, &device)?;
        let input_large = Tensor::randn(0f32, 1.0, (2000, 1), &device)?;

        // Run 1 step of inference
        let stats_small = pc_small.infer(&input_small, 1)?;
        let stats_large = pc_large.infer(&input_large, 1)?;

        let fe_small = stats_small.free_energy_history.last().unwrap();
        let fe_large = stats_large.free_energy_history.last().unwrap();

        // If the codebase regresses to Sum of Squared Errors (SSE),
        // the large hierarchy will have ~200x more Free Energy than the small one.
        // If the code is correctly using Mean Squared Error (MSE),
        // the ratio between the two will be close to 1.0.
        let ratio = fe_large / fe_small;
        
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "REGRESSION DETECTED: Free Energy is scaling with dimensionality! \
            This means you are using SSE instead of MSE in `compute_free_energy`. \
            fe_small: {:.4}, fe_large: {:.4}, Ratio: {:.2}",
            fe_small, fe_large, ratio
        );

        Ok(())
    }

// Example usage
#[cfg(test)]
pub fn example_usage() -> Result<(), PCError> {
    // Create a basic 3-level hierarchy
    let config = PCConfig::new(3, vec![512, 256, 128]);
    let mut pc = PredictiveCoding::new(config)?;
    
    // Create random input
    let device = Device::Cpu;
    let input = Tensor::randn(0f32, 1.0, (512, 1), &device)?;
    
    // Perform inference
    let stats = pc.infer(&input, 20)?;
    println!("Free energy history: {:?}", stats.free_energy_history);
    println!("Total surprise: {}", stats.total_surprise);
    
    // Learn from input
    let learning_stats = pc.learn_legacy(&input)?;
    println!("Learning surprise: {}", learning_stats.total_surprise);
    
    // Get beliefs from level 1
    let beliefs = pc.get_beliefs(1)?;
    println!("Level 1 beliefs shape: {:?}", beliefs.shape());
    Ok(())
}
}