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
    
    // NEW: Rollback buffers
    pub backup_weights: Option<Tensor>,
    pub backup_temporal: Option<Tensor>,
}

impl PCLevel {
    pub fn new(input_dim: usize, output_dim: usize, device: &Device) -> CandleResult<Self> {
        Ok(PCLevel {
            beliefs: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            predictions: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            errors: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            weights: Tensor::randn(0f32, 0.01f32, (input_dim, output_dim), device)?,
            precision: Tensor::ones((input_dim, 1), DType::F32, device)?,
            prev_beliefs: Tensor::zeros((input_dim, 1), DType::F32, device)?,
            // Initialize temporal weights as identity + noise to encourage stability
            temporal_weights: Tensor::eye(input_dim, DType::F32, device)?
                .broadcast_add(&Tensor::randn(0f32, 0.001f32, (input_dim, input_dim), device)?)?,
            adapter_weights: Tensor::randn(0f32, 0.01f32, (input_dim, output_dim), device)?,
            device: device.clone(),
            backup_weights: None,
            backup_temporal: None,
        })
    }

    /// NEW: Causal Transition Model. Predictions come from BOTH top-down spatial weights AND lateral temporal weights.
    pub fn predict(&mut self, beliefs_next: &Tensor) -> CandleResult<()> {
        let spatial_pred = self.weights.matmul(beliefs_next)?;
        let temporal_pred = self.temporal_weights.matmul(&self.prev_beliefs)?;
        self.predictions = spatial_pred.broadcast_add(&temporal_pred)?;
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
        let next_t = next_level_beliefs.t()?;
        let matmul_result = self.errors.matmul(&next_t)?;
        
        let (input_dim, _) = self.weights.shape().dims2()?;
        let effective_lr = if mu_pc_scaling { eta / (input_dim as f32).sqrt() } else { eta };
        
        let eta_tensor = Tensor::from_slice(&[effective_lr], (1, 1), &matmul_result.device())?
            .broadcast_as(matmul_result.shape())?;
        let mut delta_weights = matmul_result.mul(&eta_tensor)?;
        
        if let Some(precision_matrix) = precision {
            let broadcasted_precision = precision_matrix.broadcast_as(delta_weights.shape())?;
            delta_weights = (&delta_weights * &broadcasted_precision)?;
        }
        
        // L2 Regularization / Weight Decay to prevent infinite weight explosion
        let weight_decay = 1e-4;
        let decayed_weights = (&self.weights * (1.0 - weight_decay))?;
        self.weights = decayed_weights.broadcast_add(&delta_weights)?;
        
        // NEW: Also update temporal causal weights using Hebbian learning on prev_beliefs
        let prev_t = self.prev_beliefs.t()?;
        let temporal_update = self.errors.matmul(&prev_t)?.mul(&eta_tensor)?; // Use same learning rate
        let decayed_temporal = (&self.temporal_weights * (1.0 - weight_decay))?;
        self.temporal_weights = decayed_temporal.broadcast_add(&temporal_update)?;
        
        Ok(())
    }

    pub fn add_to_memory(&mut self, beliefs: Tensor) {
        // Keep memory_buffer for backward compatibility
        // In a full implementation, this would be used for temporal context
        let _ = beliefs;
    }
}
