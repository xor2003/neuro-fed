// src/pc_hierarchy.rs
// Pure Predictive Coding (PC) implementation based on Rao-Ballard/Friston free-energy minimization
// Migrated from ndarray to candle-core for GPU acceleration

use candle_core::{Device, Tensor, DType};

use crate::knowledge_filter::{PrecisionCalculator, PrecisionConfig, PrecisionContext};
pub use crate::pc_types::{PCError, PCConfig, SurpriseStats};
use crate::pc_level::PCLevel;

/// Main Predictive Coding hierarchy
pub struct PredictiveCoding {
    pub levels: Vec<PCLevel>,
    pub config: PCConfig,
    pub device: Device,
    pub precision_calculator: Option<PrecisionCalculator>,
    pub free_energy: f32,
    pub belief_history: Vec<Vec<Tensor>>,
}

impl PredictiveCoding {
    pub fn new(config: PCConfig) -> Result<Self, PCError> {
        let device = Device::Cpu;
        Self::new_with_device(config, device)
    }
    
    pub fn new_with_device(config: PCConfig, device: Device) -> Result<Self, PCError> {
        let mut levels = Vec::new();
        
        for i in 0..config.n_levels {
            let input_dim = config.dim_per_level[i];
            let output_dim = if i + 1 < config.n_levels {
                config.dim_per_level[i + 1]
            } else {
                // Top level has no upward projection
                config.dim_per_level[i]
            };
            
            let level = PCLevel::new(input_dim, output_dim, &device)?;
            levels.push(level);
        }
        
        let precision_calculator = if config.enable_precision_weighting {
            Some(PrecisionCalculator::new(PrecisionConfig {
                free_energy_drop_threshold: config.free_energy_drop_threshold,
                default_precision: config.default_precision,
                min_precision: config.min_precision,
                max_precision: config.max_precision,
                free_energy_history_size: config.free_energy_history_size,
                enable_code_verification: config.enable_code_verification,
                enable_nostr_zap_tracking: config.enable_nostr_zap_tracking,
                min_zaps_for_consensus: config.min_zaps_for_consensus,
                trusted_node_keys: Vec::new(),
            }))
        } else {
            None
        };
        
        Ok(PredictiveCoding {
            levels,
            config,
            device,
            precision_calculator,
            free_energy: 0.0,
            belief_history: Vec::new(),
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
                // Use split_at_mut to avoid simultaneous mutable and immutable borrows
                let (left, right) = self.levels.split_at_mut(l + 1);
                let current = &mut left[l];
                let next_beliefs = &right[0].beliefs;
                current.predict(next_beliefs)?;
                current.compute_errors()?;
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

            // Early exiting: stop ONLY if FE is extremely low AND we've thought for at least 3 steps
            if fe < 0.0001 && step > 3 {
                tracing::debug!("PC inference converged early at step {} (FE: {:.4})", step, fe);
                break;
            }
        }
        
        stats.total_surprise = stats.free_energy_history.iter().sum::<f32>();
        
        // Compute explicit uncertainty metrics
        if !stats.free_energy_history.is_empty() {
            // Novelty: Initial free energy (how unexpected was the input)
            stats.novelty_score = *stats.free_energy_history.first().unwrap_or(&0.0);
            
            // Confidence: Inverse of final free energy (how stable the belief became)
            let final_fe = *stats.free_energy_history.last().unwrap_or(&0.0);
            stats.confidence_score = 1.0 / (1.0 + final_fe); // Maps [0, ∞) → (0, 1]
        }
        
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
            let error_sq = prediction_error.sqr()?.sum_all()?.to_scalar::<f32>()?;
            fe += error_sq;
        }
        
        Ok(fe)
    }
    
    
    pub fn get_top_belief(&self) -> Result<Tensor, PCError> {
        if let Some(top_level) = self.levels.last() {
            Ok(top_level.beliefs.clone())
        } else {
            Err(PCError("No levels in PC hierarchy".to_string()))
        }
    }
    
    pub fn add_to_belief_history(&mut self) {
        let beliefs: Vec<Tensor> = self.levels.iter().map(|level| level.beliefs.clone()).collect();
        self.belief_history.push(beliefs);
        if self.belief_history.len() > 10 {
            self.belief_history.remove(0);
        }
    }

    /// Advances the temporal state of all levels
    pub fn step_time(&mut self) -> Result<(), PCError> {
        for level in &mut self.levels {
            level.step_time()?;
        }
        Ok(())
    }

    /// NEW: Structurally grounds confidence into the PC precision matrices
    pub fn modulate_precision(&mut self, calibrated_confidence: f32) -> Result<(), PCError> {
        for level in &mut self.levels {
            let ones = Tensor::ones_like(&level.precision)?;
            // Prevent absolute zero precision to avoid gradient vanishing
            let safe_confidence = calibrated_confidence.max(0.01);
            let scale = Tensor::from_slice(&[safe_confidence], (1, 1), &self.device)?
                .broadcast_as(ones.shape())?;
            level.precision = ones.mul(&scale)?;
        }
        Ok(())
    }

    pub fn checkpoint_weights(&mut self) -> Result<(), PCError> {
        for level in &mut self.levels {
            level.checkpoint()?;
        }
        Ok(())
    }

    pub fn rollback_weights(&mut self) -> Result<(), PCError> {
        for level in &mut self.levels {
            level.rollback()?;
        }
        tracing::warn!("PC weights rolled back due to unstable Free Energy rise.");
        Ok(())
    }

    /// Processes a sequence of latents, updating the causal temporal state at each step.
    pub fn infer_sequence(&mut self, sequence_tensor: &Tensor, steps_per_token: usize) -> Result<SurpriseStats, PCError> {
        let seq_len = sequence_tensor.shape().dims()[0];
        let mut overall_stats = SurpriseStats::default();
        
        for t in 0..seq_len {
            // Extract [1, embedding_dim] and reshape to[embedding_dim, 1]
            let token_emb = sequence_tensor.narrow(0, t, 1)?
                .reshape((self.config.dim_per_level[0], 1))?;
            
            let stats = self.infer(&token_emb, steps_per_token)?;
            overall_stats.free_energy_history.extend(stats.free_energy_history);
            overall_stats.total_surprise += stats.total_surprise;
            
            // Critical: Advance temporal state for the next token!
            self.step_time()?;
        }
        
        overall_stats.novelty_score = *overall_stats.free_energy_history.first().unwrap_or(&0.0);
        let final_fe = *overall_stats.free_energy_history.last().unwrap_or(&0.0);
        overall_stats.confidence_score = 1.0 / (1.0 + final_fe);
        
        Ok(overall_stats)
    }

    /// NEW: Learn over entire sequences (safe sleep phase processing)
    pub fn learn_sequence(&mut self, sequence_tensor: &Tensor, context: Option<PrecisionContext>) -> Result<SurpriseStats, PCError> {
        let seq_len = sequence_tensor.shape().dims()[0];
        let mut overall_stats = SurpriseStats::default();

        for t in 0..seq_len {
            let token_emb = sequence_tensor.narrow(0, t, 1)?
                .reshape((self.config.dim_per_level[0], 1))?;
            
            let stats = self.learn(&token_emb, context.clone())?;
            overall_stats.free_energy_history.extend(stats.free_energy_history);
            overall_stats.total_surprise += stats.total_surprise;
            
            self.step_time()?;
        }
        Ok(overall_stats)
    }
}

#[cfg(test)]
mod sequence_and_calibration_tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_modulate_precision_scales_correctly() -> Result<(), PCError> {
        let device = Device::Cpu;
        let config = PCConfig::new(2, vec![8, 4]);
        let mut pc = PredictiveCoding::new(config)?;

        // Base precision is initialized to 1.0
        let initial_precision = pc.levels[0].precision.to_vec2::<f32>()?;
        assert_eq!(initial_precision[0][0], 1.0);

        // Apply a calibrated confidence of 20% (0.2)
        pc.modulate_precision(0.2)?;

        let modulated_precision = pc.levels[0].precision.to_vec2::<f32>()?;
        
        // The precision matrix should now be scaled down to 0.2
        assert!((modulated_precision[0][0] - 0.2).abs() < 0.001,
            "Precision matrix did not scale correctly. Got {}", modulated_precision[0][0]);
            
        Ok(())
    }

    #[test]
    fn test_modulate_precision_prevents_zero_vanishing_gradient() -> Result<(), PCError> {
        let device = Device::Cpu;
        let config = PCConfig::new(2, vec![8, 4]);
        let mut pc = PredictiveCoding::new(config)?;

        // If confidence is absolutely 0.0 (total failure historical rate)
        pc.modulate_precision(0.0)?;

        let modulated_precision = pc.levels[0].precision.to_vec2::<f32>()?;
        
        // It must clamp to 0.01 to prevent the network from entirely dying (vanishing gradients)
        assert!((modulated_precision[0][0] - 0.01).abs() < 0.0001,
            "Precision failed to clamp minimum bound. Got {}", modulated_precision[0][0]);
            
        Ok(())
    }

    #[test]
    fn test_infer_sequence_accumulates_stats_and_progresses_time() -> Result<(), PCError> {
        let device = Device::Cpu;
        let config = PCConfig::new(2, vec![4, 2]);
        let mut pc = PredictiveCoding::new(config)?;

        // Create a sequence of 3 tokens, each with dimension 4. Shape: [3, 4]
        let seq_data = vec![
            0.1f32, 0.2, 0.3, 0.4, // Token 1
            0.5, 0.6, 0.7, 0.8, // Token 2
            0.9, 1.0, 1.1, 1.2, // Token 3
        ];
        let seq_tensor = Tensor::from_vec(seq_data, (3, 4), &device)?;

        let stats = pc.infer_sequence(&seq_tensor, 5)?;

        // Ensure stats accumulated over the 3 steps
        assert!(stats.free_energy_history.len() > 5, "Should have multiple steps of history");
        assert!(stats.total_surprise > 0.0, "Total surprise should be non-zero");

        // Verify temporal progression actually happened (prev_beliefs != 0)
        let prev_beliefs = pc.levels[0].prev_beliefs.to_vec2::<f32>()?;
        let sum_prev: f32 = prev_beliefs.iter().map(|row| row[0]).sum();
        
        assert!(sum_prev > 0.0, "Temporal state (prev_beliefs) was not updated during sequence inference");

        Ok(())
    }
}

