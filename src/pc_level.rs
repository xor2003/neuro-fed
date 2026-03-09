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
    pub memory_buffer: Vec<Tensor>,
    pub adapter_weights: Tensor,
    pub device: Device,
}

impl PCLevel {
    pub fn new(input_dim: usize, output_dim: usize, device: &Device) -> CandleResult<Self> {
        let beliefs = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let predictions = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let errors = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let weights = Tensor::randn(0f32, 0.01f32, (input_dim, output_dim), device)?;
        let precision = Tensor::ones((input_dim, 1), DType::F32, device)?;
        let prev_beliefs = Tensor::zeros((input_dim, 1), DType::F32, device)?;
        let temporal_weights = Tensor::randn(0f32, 0.01f32, (input_dim, input_dim), device)?;
        let adapter_weights = Tensor::randn(0f32, 0.01f32, (input_dim, output_dim), device)?;

        Ok(PCLevel {
            beliefs,
            predictions,
            errors,
            weights,
            precision,
            prev_beliefs,
            temporal_weights,
            memory_buffer: Vec::new(),
            adapter_weights,
            device: device.clone(),
        })
    }

    pub fn predict(&mut self, beliefs_next: &Tensor) -> CandleResult<()> {
        self.predictions = self.weights.matmul(beliefs_next)?;
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
        
        let clip_threshold = 1.0;
        let norm = delta_weights.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
        if norm > clip_threshold {
            let scale = clip_threshold / norm;
            delta_weights = (delta_weights * (scale as f64))?;
        }
        
        self.weights = self.weights.broadcast_add(&delta_weights)?;
        Ok(())
    }

    pub fn add_to_memory(&mut self, beliefs: Tensor) {
        self.memory_buffer.push(beliefs);
        if self.memory_buffer.len() > 10 { self.memory_buffer.remove(0); }
    }
}
