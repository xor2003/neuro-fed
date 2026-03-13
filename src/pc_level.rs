// src/pc_level.rs
use candle_core::{Device, Tensor, DType, Result as CandleResult};

#[derive(Debug, Clone)]
pub struct PCLevel {
    pub beliefs: Tensor,
    pub predictions: Tensor,
    pub errors: Tensor,
    pub weights: Tensor,
    pub precision: Tensor,
    pub prev_beliefs: Tensor,
    pub temporal_weights: Tensor,
    pub adapter_weights: Tensor,
    pub device: Device,
    pub is_dirty: bool, // For error-driven recomputation
    
    // NEW: Rollback buffers
    pub backup_weights: Option<Tensor>,
    pub backup_temporal: Option<Tensor>,
    // Cached learning-rate tensors to reduce per-step allocations
    pub cached_spatial_eta: Option<(f32, Tensor)>,
    pub cached_temporal_eta: Option<(f32, Tensor)>,
    // Delta propagation cache
    pub last_prediction_input: Option<Tensor>,
    pub last_spatial_prediction: Option<Tensor>,
}

impl PCLevel {
    pub fn new(input_dim: usize, output_dim: usize, device: &Device) -> CandleResult<Self> {
        Self::new_with_weights(input_dim, output_dim, None, device)
    }
    
    /// NEW: Create PCLevel with optional pre-trained weights for knowledge-guided initialization
    pub fn new_with_weights(
        input_dim: usize,
        output_dim: usize,
        initial_weights: Option<Tensor>,
        device: &Device
    ) -> CandleResult<Self> {
        let weights = match initial_weights {
            Some(w) => w,
            None => Tensor::randn(0f32, (1.0 / input_dim as f32).sqrt(), (input_dim, output_dim), device)?,
        };
        
        Ok(PCLevel {
            beliefs: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            predictions: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            errors: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            weights,
            precision: Tensor::ones((input_dim, 1), DType::F32, device)?,
            prev_beliefs: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            // Initialize temporal weights as identity + noise to encourage stability
            temporal_weights: Tensor::eye(input_dim, DType::F32, device)?
                .broadcast_add(&Tensor::randn(0f32, 0.001f32, (input_dim, input_dim), device)?)?,
            adapter_weights: Tensor::randn(0f32, 0.01f32, (input_dim, output_dim), device)?,
            device: device.clone(),
            is_dirty: true,
            backup_weights: None,
            backup_temporal: None,
            cached_spatial_eta: None,
            cached_temporal_eta: None,
            last_prediction_input: None,
            last_spatial_prediction: None,
        })
    }

    /// NEW: Causal Transition Model. Predictions come from BOTH top-down spatial weights AND lateral temporal weights.
    /// ИСПОЛЬЗУЕМ ТЕМПОРАЛЬНЫЕ ВЕСА ТОЛЬКО ЕСЛИ ЕСТЬ ИСТОРИЯ
    pub fn predict(&mut self, beliefs_next: &Tensor) -> CandleResult<()> {
        // 🚀 DELTA PROPAGATION: Only multiply the difference if we have cached previous input
        let spatial_pred = if let (Some(old_input), Some(old_spatial)) = (&self.last_prediction_input, &self.last_spatial_prediction) {
            let delta_r = (beliefs_next - old_input)?;
            let delta_pred = self.weights.matmul(&delta_r.contiguous()?)?;
            (old_spatial + &delta_pred)?
        } else {
            // Initial step: full matmul
            self.weights.matmul(&beliefs_next.contiguous()?)?
        };
        
        // Cache for next delta propagation
        self.last_spatial_prediction = Some(spatial_pred.clone());
        self.last_prediction_input = Some(beliefs_next.clone());

        // ИСПОЛЬЗУЕМ ТЕМПОРАЛЬНЫЕ ВЕСА ТОЛЬКО ЕСЛИ ЕСТЬ ИСТОРИЯ
        let prev_beliefs_sum = self.prev_beliefs.sum_all()?.to_scalar::<f32>()?;
        if prev_beliefs_sum.abs() > 1e-6 {
            let prev_beliefs_cont = self.prev_beliefs.contiguous()?;
            let temporal_pred = self.temporal_weights.matmul(&prev_beliefs_cont)?;
            self.predictions = spatial_pred.broadcast_add(&temporal_pred)?;
        } else {
            self.predictions = spatial_pred;
        }
        
        Ok(())
    }

    /// Advance time step
    pub fn step_time(&mut self) -> CandleResult<()> {
        self.prev_beliefs = self.beliefs.clone();
        Ok(())
    }

    // NEW: Safe Consolidation Mechanisms
    pub fn checkpoint(&mut self) -> CandleResult<()> {
        self.backup_weights = Some(self.weights.clone());
        self.backup_temporal = Some(self.temporal_weights.clone());
        Ok(())
    }

    pub fn rollback(&mut self) -> CandleResult<()> {
        if let Some(bw) = &self.backup_weights {
            self.weights = bw.clone();
        }
        if let Some(bt) = &self.backup_temporal {
            self.temporal_weights = bt.clone();
        }
        Ok(())
    }

    pub fn compute_errors(&mut self) -> CandleResult<()> {
        let raw_error = (&self.beliefs - &self.predictions)?;
        self.errors = (&raw_error * &self.precision)?;
        Ok(())
    }
    
    pub fn update_weights(&mut self, eta: f32, next_level_beliefs: &Tensor, precision: Option<&Tensor>, mu_pc_scaling: bool) -> CandleResult<()> {
        // 🚀 OPTIMIZATION: Force contiguous memory layout after transpose for fast BLAS/Native matmul
        let next_t = next_level_beliefs.t()?.contiguous()?.to_device(&self.device)?;
        let matmul_result = self.errors.matmul(&next_t)?;
        
        // --- NEW: NUMERICAL GUARDRAIL ---
        // Check if the update is finite. If it's NaN or Inf, skip this update.
        let is_finite = matmul_result.abs()?.max_all()?.to_scalar::<f32>()?.is_finite();
        if !is_finite {
            tracing::error!("⚠️ Skipping weight update: Matmul resulted in non-finite values (Explosion prevented)");
            return Ok(());
        }
        // --------------------------------
        
        let (input_dim, _) = self.weights.shape().dims2()?;
        let effective_lr = if mu_pc_scaling { eta / (input_dim as f32).sqrt() } else { eta };
        
        // Create eta tensor for spatial weights update (cached)
        let eta_tensor_spatial = match &self.cached_spatial_eta {
            Some((cached_lr, cached_tensor)) if (cached_lr - effective_lr).abs() < 1e-9 => cached_tensor.clone(),
            _ => {
                let t = Tensor::from_slice(&[effective_lr], (1, 1), &matmul_result.device())?
                    .broadcast_as(matmul_result.shape())?;
                self.cached_spatial_eta = Some((effective_lr, t.clone()));
                t
            }
        };
        let mut delta_weights = matmul_result.mul(&eta_tensor_spatial)?;
        
        if let Some(precision_matrix) = precision {
            let broadcasted_precision = precision_matrix.broadcast_as(delta_weights.shape())?;
            delta_weights = (&delta_weights * &broadcasted_precision)?;
        }
        
        // --- GRADIENT CLIPPING ---
        let clip_val = 1.0f64;
        let total_norm = delta_weights.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        
        if total_norm > clip_val {
            let scale = clip_val / total_norm;
            delta_weights = delta_weights.affine(scale, 0.0)?;
        }
        
        // L2 Regularization / Weight Decay to prevent infinite weight explosion
        let weight_decay = 1e-4;
        let decayed_weights = (&self.weights * (1.0 - weight_decay))?;
        
        // 🚀 CACHE LOCALITY: After updating weights, we MUST ensure the new tensor
        // is contiguous. This allows the NEXT token's matmul to use BLAS/SIMD optimally.
        let new_weights = decayed_weights.broadcast_add(&delta_weights)?;
        self.weights = new_weights.contiguous()?;
        
        // NEW: Also update temporal causal weights using Hebbian learning on prev_beliefs
        // 🚀 OPTIMIZATION: Force contiguous memory layout
        let prev_t = self.prev_beliefs.t()?.contiguous()?;
        let temporal_matmul = self.errors.matmul(&prev_t)?;
        
        // --- NEW: NUMERICAL GUARDRAIL for temporal update ---
        let is_temporal_finite = temporal_matmul.abs()?.max_all()?.to_scalar::<f32>()?.is_finite();
        if !is_temporal_finite {
            tracing::error!("⚠️ Skipping temporal weight update: Matmul resulted in non-finite values");
            // Continue without temporal update, but spatial update already passed guardrail
        } else {
            // Create separate eta tensor for temporal update with correct shape (cached)
            let eta_tensor_temporal = match &self.cached_temporal_eta {
                Some((cached_lr, cached_tensor)) if (cached_lr - effective_lr).abs() < 1e-9 => cached_tensor.clone(),
                _ => {
                    let t = Tensor::from_slice(&[effective_lr], (1, 1), &temporal_matmul.device())?
                        .broadcast_as(temporal_matmul.shape())?;
                    self.cached_temporal_eta = Some((effective_lr, t.clone()));
                    t
                }
            };
            let mut temporal_update = temporal_matmul.mul(&eta_tensor_temporal)?;
            
            // --- GRADIENT CLIPPING for temporal update ---
            let temporal_norm = temporal_update.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            if temporal_norm > clip_val {
                let scale = clip_val / temporal_norm;
                temporal_update = temporal_update.affine(scale, 0.0)?;
            }
            
            let decayed_temporal = (&self.temporal_weights * (1.0 - weight_decay))?;
            
            // 🚀 CACHE LOCALITY: Ensure temporal weights are also contiguous
            let new_temporal_weights = decayed_temporal.broadcast_add(&temporal_update)?;
            self.temporal_weights = new_temporal_weights.contiguous()?;
        }
        
        Ok(())
    }

    /// 🚀 NEW: Exact Learning Rule (Prospective Configuration)
    /// This implementation uses normalized beliefs to prevent weight explosion
    /// and adaptive learning rates based on error magnitude.
    /// precision_scale: Optional scaling factor from hyper-network (0.1 to 2.0)
    pub fn update_weights_exact(&mut self, error: &Tensor, next_beliefs: &Tensor, precision_scale: Option<f32>) -> CandleResult<()> {
        // Compute surprise magnitude
        let surprise = error.sqr()?.sum_all()?.to_scalar::<f32>()?;
        
        // ─────────────────────────────────────────────────────
        // Aha! Optimizer: adaptive LR + normalized Hebbian
        // ─────────────────────────────────────────────────────
        let surprise_threshold = 2.0;
        let base_eta = 0.01;
        let aha_boost = 5.0; // во сколько раз увеличиваем шаг при "Ага!"
        
        let mut adaptive_eta = if surprise > surprise_threshold {
            base_eta * aha_boost
        } else {
            base_eta
        };
        
        // Apply precision scaling from hyper-network if provided
        if let Some(scale) = precision_scale {
            adaptive_eta *= scale;
        }
        
        // Нормализация входа (предотвращает взрывы весов)
        let input_norm = (next_beliefs.sqr()?.sum_all()?.to_scalar::<f32>()? + 1e-6).sqrt();
        let normalized_input = next_beliefs.affine((1.0 / input_norm) as f64, 0.0)?;
        
        // Delta = error * normalized_input^T
        let delta = error.matmul(&normalized_input.t()?)?;
        
        // L2 decay + большой шаг при Aha!
        let l2_decay = 1e-4;
        let new_weights = self.weights
            .affine(1.0 - l2_decay, 0.0)?                     // L2 decay
            .broadcast_add(&delta.affine(adaptive_eta as f64, 0.0)?)?;
        
        self.weights = new_weights.contiguous()?;
        Ok(())
    }

    pub fn add_to_memory(&mut self, beliefs: Tensor) {
        // Keep memory_buffer for backward compatibility
        // In a full implementation, this would be used for temporal context
        let _ = beliefs;
    }

    /// Set weights from a flattened vector (row-major)
    pub fn set_weights_from_vec(&mut self, weights_vec: Vec<f32>) -> CandleResult<()> {
        let rows = self.weights.dim(0)?;
        let cols = self.weights.dim(1)?;
        
        if weights_vec.len() != rows * cols {
            return Err(candle_core::Error::Msg(format!(
                "Weight vector size mismatch: expected {} ({}x{}), got {}",
                rows * cols, rows, cols, weights_vec.len()
            )));
        }
        
        // 🔴 FIX: Bake contiguity into the weights immediately on load
        self.weights = Tensor::from_vec(weights_vec, (rows, cols), &self.device)?.contiguous()?;
        Ok(())
    }

    /// Set temporal weights from a flattened vector (row-major)
    pub fn set_temporal_weights_from_vec(&mut self, weights_vec: Vec<f32>) -> CandleResult<()> {
        let rows = self.temporal_weights.dim(0)?;
        let cols = self.temporal_weights.dim(1)?;
        
        if weights_vec.len() != rows * cols {
            return Err(candle_core::Error::Msg(format!(
                "Temporal weight vector size mismatch: expected {} ({}x{}), got {}",
                rows * cols, rows, cols, weights_vec.len()
            )));
        }
        
        self.temporal_weights = Tensor::from_vec(weights_vec, (rows, cols), &self.device)?;
        Ok(())
    }
}

#[cfg(test)]
mod stability_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_gradient_clipping_enforces_max_norm() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(10, 5, &device)?;
        
        // 1. Create a "Nuclear" error (1 million)
        level.errors = Tensor::full(1_000_000.0f32, (10, 1), &device)?;
        let next_beliefs = Tensor::full(1.0f32, (5, 1), &device)?;
        
        let weights_before = level.weights.to_vec2::<f32>()?;
        
        // 2. Perform update.
        // Without clipping, weights would move by ~1000.0.
        // With clipping (max_norm = 1.0), they should move very little.
        level.update_weights(0.1, &next_beliefs, None, false)?;
        
        let weights_after = level.weights.to_vec2::<f32>()?;
        
        // 3. Measure how far the weights moved
        let mut max_move = 0.0f32;
        for i in 0..10 {
            for j in 0..5 {
                let diff = (weights_after[i][j] - weights_before[i][j]).abs();
                if diff > max_move { max_move = diff; }
            }
        }

        // It must be small because of the clip
        assert!(max_move < 1.1, "Weights moved too far ({}), clipping failed!", max_move);
        Ok(())
    }

    #[test]
    fn test_nan_protection_fuse() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(4, 2, &device)?;
        let original_weights = level.weights.to_vec2::<f32>()?;

        // 1. Inject a NaN error
        level.errors = Tensor::full(f32::NAN, (4, 1), &device)?;
        let next_beliefs = Tensor::zeros((2, 1), candle_core::DType::F32, &device)?;

        // 2. Update. This should trigger the ERROR log but NOT change the weights.
        level.update_weights(0.1, &next_beliefs, None, false)?;

        let final_weights = level.weights.to_vec2::<f32>()?;
        assert_eq!(original_weights, final_weights, "NaN error corrupted the weights!");
        Ok(())
    }
}

#[cfg(test)]
mod temporal_logic_tests {
    use super::*;

    #[test]
    fn test_temporal_prediction_skipped_if_prev_state_is_zero() -> CandleResult<()> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(4, 2, &device)?;
        let next_beliefs = Tensor::randn(0f32, 1.0, (2, 1), &device)?;
        
        // Изначально prev_beliefs = 0. Предсказание должно быть чисто пространственным.
        let spatial_prediction = level.weights.matmul(&next_beliefs)?;
        level.predict(&next_beliefs)?;
        
        // Compute difference manually
        let predictions_vec = level.predictions.to_vec2::<f32>()?;
        let spatial_vec = spatial_prediction.to_vec2::<f32>()?;
        let mut diff = 0.0;
        for (pred_row, spatial_row) in predictions_vec.iter().zip(spatial_vec.iter()) {
            for (&p, &s) in pred_row.iter().zip(spatial_row.iter()) {
                diff += (p - s).abs();
            }
        }
        assert!(diff < 1e-6, "Temporal prediction was not skipped for zero prev_beliefs");
        
        Ok(())
    }
}

#[cfg(test)]
mod numerical_guardrail_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_nan_protection_fuse() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(10, 5, &device)?;
        
        let original_weights = level.weights.to_vec2::<f32>()?;
        
        // 1. Срочно создаем ошибку типа NaN (Not a Number)
        // Simulate a mathematical "explosion" in the error tensor
        level.errors = Tensor::full(f32::NAN, (10, 1), &device)?;
        let next_beliefs = Tensor::zeros((5, 1), candle_core::DType::F32, &device)?;
        
        // 2. Пытаемся обновить веса.
        // Logic fix: update_weights should see the NaN and skip the update.
        level.update_weights(0.1, &next_beliefs, None, false)?;
        
        // 3. Проверяем, что веса НЕ изменились и НЕ стали NaN
        let updated_weights = level.weights.to_vec2::<f32>()?;
        assert_eq!(original_weights, updated_weights, "Weights were corrupted by NaN! Guardrail failed.");
        Ok(())
    }

    #[test]
    fn test_gradient_clipping_bounds_updates() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(2, 2, &device)?;
        
        // 1. Создаем гигантскую ошибку в 1 миллион
        // Without clipping, this would teleport weights to infinity
        level.errors = Tensor::full(1_000_000.0f32, (2, 1), &device)?;
        let next_beliefs = Tensor::full(1.0f32, (2, 1), &device)?;
        
        let weights_before = level.weights.to_vec2::<f32>()?;
        
        // 2. Обновляем веса
        level.update_weights(1.0, &next_beliefs, None, false)?;
        
        let weights_after = level.weights.to_vec2::<f32>()?;
        
        // 3. Считаем, как далеко ушли веса.
        // Максимальный сдвиг не должен превышать max_norm (1.0) + погрешность
        let mut max_diff = 0.0f32;
        for i in 0..2 {
            for j in 0..2 {
                let diff = (weights_after[i][j] - weights_before[i][j]).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }
        
        assert!(max_diff < 1.5, "Gradient clipping failed! Weights moved too far: {}", max_diff);
        Ok(())
    }
}

#[cfg(test)]
mod delta_propagation_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_spatial_delta_propagation_vs_fresh_matmul() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(4, 4, &device)?;
        
        // Initialize temporal state to guarantee temporal_pred is active
        level.prev_beliefs = Tensor::ones((4, 1), candle_core::DType::F32, &device)?;
        level.temporal_weights = Tensor::eye(4, candle_core::DType::F32, &device)?;
        
        let input1 = Tensor::full(1.0f32, (4, 1), &device)?;
        let input2 = Tensor::full(2.0f32, (4, 1), &device)?;

        // 1. Run with delta propagation
        level.predict(&input1)?; // Populates caches
        level.predict(&input2)?; // Uses delta propagation
        let pred_with_delta = level.predictions.to_vec2::<f32>()?;

        // 2. Run fresh (simulate what it SHOULD be without the delta cache)
        let mut level_fresh = PCLevel::new(4, 4, &device)?;
        // Copy weights so they are identical
        level_fresh.weights = level.weights.clone();
        level_fresh.temporal_weights = level.temporal_weights.clone();
        level_fresh.prev_beliefs = level.prev_beliefs.clone();
        
        // Only predict input2 directly (no cache)
        level_fresh.predict(&input2)?;
        let pred_fresh = level_fresh.predictions.to_vec2::<f32>()?;

        // Verify they are practically identical
        let mut max_diff = 0.0f32;
        for i in 0..4 {
            for j in 0..1 {
                let diff = (pred_with_delta[i][j] - pred_fresh[i][j]).abs();
                if diff > max_diff { max_diff = diff; }
            }
        }
        
        // If the temporal bug was still there, the difference would be massive
        assert!(max_diff < 1e-4, "Delta propagation diverged from fresh matmul! Max diff: {}", max_diff);
        Ok(())
    }
}

#[cfg(test)]
mod aha_optimizer_tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_aha_optimizer_boosts_on_high_surprise() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(8, 4, &device)?;
        
        // Маленькая surprise → маленький шаг
        let small_error = Tensor::full(0.1f32, (8, 1), &device)?;
        let next = Tensor::ones((4, 1), candle_core::DType::F32, &device)?;
        
        let initial_weights = level.weights.clone();
        level.update_weights_exact(&small_error, &next, None)?;
        let small_change = (level.weights.sub(&initial_weights)?.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
        
        // Большая surprise → большой шаг
        let huge_error = Tensor::full(10.0f32, (8, 1), &device)?;
        level.weights = initial_weights.clone(); // сброс
        level.update_weights_exact(&huge_error, &next, None)?;
        let huge_change = (level.weights.sub(&initial_weights)?.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt();
        
        assert!(huge_change > small_change * 3.0, "Aha! should boost update significantly. small_change={:.4}, huge_change={:.4}", small_change, huge_change);
        println!("✅ Aha! Optimizer test passed: small_change={:.4}, huge_change={:.4}", small_change, huge_change);
        
        Ok(())
    }

    #[test]
    fn test_normalized_hebbian_prevents_explosion() -> Result<(), Box<dyn std::error::Error>> {
        let device = Device::Cpu;
        let mut level = PCLevel::new(8, 4, &device)?;
        
        // Огромный вход → без нормализации веса взорвались бы
        let huge_next = Tensor::full(1000.0f32, (4, 1), &device)?;
        let error = Tensor::ones((8, 1), candle_core::DType::F32, &device)?;
        
        level.update_weights_exact(&error, &huge_next, None)?;
        
        let weights_norm = level.weights.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        assert!(weights_norm < 50.0, "Weights exploded! Norm: {}", weights_norm);
        
        println!("✅ Normalized Hebbian test passed: final norm={:.2}", weights_norm);
        Ok(())
    }
}
