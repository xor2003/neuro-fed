// src/pc_decoder.rs
use candle_core::{Tensor, Device};
use candle_nn::{ops, loss};
use crate::pc_types::PCError;
use crate::types::ThoughtOp;

pub struct ThoughtDecoder {
    pub w_update: Tensor,
    pub w_hidden: Tensor,
    pub w_vocab: Tensor,
    pub device: Device,
}

impl ThoughtDecoder {
    pub fn new(belief_dim: usize, vocab_size: usize, device: &Device) -> Result<Self, PCError> {
        let combined_dim = belief_dim * 2;
        Ok(Self {
            w_update: Tensor::randn(0f32, 0.02f32, (belief_dim, combined_dim), device)?,
            w_hidden: Tensor::randn(0f32, 0.02f32, (belief_dim, combined_dim), device)?,
            w_vocab: Tensor::randn(0f32, 0.02f32, (vocab_size, belief_dim), device)?,
            device: device.clone(),
        })
    }

    pub fn decode_sequence(&self, anchor_belief: &Tensor, max_steps: usize) -> Result<Vec<u32>, PCError> {
        let mut sequence = Vec::new();
        let mut h_t = anchor_belief.clone();

        for _ in 0..max_steps {
            let combined_input = Tensor::cat(&[&h_t, anchor_belief], 0)?;
            let update_gate = combined_input.matmul(&self.w_update.t()?)?;
            let h_hat = combined_input.matmul(&self.w_hidden.t()?)?;
            h_t = (h_t.mul(&update_gate)? + h_hat.mul(&(update_gate.ones_like()? - &update_gate)?)?)?;
            
            let logits = h_t.matmul(&self.w_vocab.t()?)?;
            let probs = ops::softmax(&logits, 0)?;

            if self.calculate_entropy(&probs)? > 2.5 {
                break;
            }
            
            let token_id = probs.argmax(0)?.to_scalar::<u32>()?;
            sequence.push(token_id);
            if token_id == ThoughtOp::EOF as u32 { break; }
        }
        Ok(sequence)
    }

    pub fn train_step(&mut self, belief: &Tensor, target_seq: &[u32], lr: f32) -> Result<f32, PCError> {
        let mut h_t = belief.clone();
        let mut total_loss = 0.0;
        for &target_id in target_seq {
            let combined = Tensor::cat(&[&h_t, belief], 0)?;
            let h_next = combined.matmul(&self.w_hidden.t()?)?.tanh()?;
            let logits = h_next.matmul(&self.w_vocab.t()?)?;
            let loss = loss::cross_entropy(&logits.t()?, &Tensor::new(&[target_id], &self.device)?)?;
            total_loss += loss.to_scalar::<f32>()?;
            
            let probs = ops::softmax(&logits, 0)?;
            let mut grad_vec = probs.to_vec1::<f32>()?;
            if (target_id as usize) < grad_vec.len() { grad_vec[target_id as usize] -= 1.0; }
            let grad_tensor = Tensor::from_vec(grad_vec, probs.shape(), &self.device)?;
            let dw_vocab = grad_tensor.matmul(&h_next.t()?)?;
            self.w_vocab = (&self.w_vocab - dw_vocab.mul(&Tensor::from_slice(&[lr], (1, 1), &self.device)?)?)?;
            h_t = h_next;
        }
        Ok(total_loss / target_seq.len() as f32)
    }

    fn calculate_entropy(&self, probs: &Tensor) -> Result<f32, candle_core::Error> {
        let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
        let entropy: f32 = probs_vec.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
        Ok(entropy)
    }
}
