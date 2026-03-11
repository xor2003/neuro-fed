// src/pc_decoder.rs
use candle_core::{Device, Tensor, Var};
use candle_nn::ops;
use crate::pc_types::PCError;

pub struct ThoughtDecoder {
    pub w_update: Var,
    pub w_hidden: Var,
    pub w_vocab: Var,
    pub device: Device,
}

impl ThoughtDecoder {
    pub fn new(belief_dim: usize, vocab_size: usize, device: &Device) -> Result<Self, PCError> {
        let combined_dim = belief_dim * 2;
        Ok(Self {
            w_update: Var::randn(0f32, 0.01f32, (belief_dim, combined_dim), device)?,
            w_hidden: Var::randn(0f32, 0.01f32, (belief_dim, combined_dim), device)?,
            w_vocab: Var::randn(0f32, 0.01f32, (vocab_size, belief_dim), device)?,
            device: device.clone(),
        })
    }

    /// NEW: Dynamically expands the vocabulary matrix when new chunks are discovered
    pub fn resize_vocab(&mut self, new_vocab_size: usize) -> Result<(), PCError> {
        let old_vocab_size = self.w_vocab.shape().dims()[0];
        if new_vocab_size <= old_vocab_size {
            return Ok(());
        }
        
        let old_tensor = self.w_vocab.as_tensor();
        let belief_dim = old_tensor.dims()[1];
        let added_size = new_vocab_size - old_vocab_size;
        
        // Initialize new vocabulary rows
        let new_weights = Tensor::randn(0f32, 0.01f32, (added_size, belief_dim), &self.device)?;
        
        // Concatenate old weights with new weights to preserve existing knowledge
        let combined = Tensor::cat(&[old_tensor, &new_weights], 0)?;
        self.w_vocab = Var::from_tensor(&combined)?;
        
        tracing::info!("Expanded ThoughtDecoder vocabulary from {} to {}", old_vocab_size, new_vocab_size);
        Ok(())
    }

    pub fn decode_sequence(&self, anchor_belief: &Tensor, max_steps: usize, beam_width: usize) -> Result<Vec<u32>, PCError> {
        self.decode_sequence_with_costs(anchor_belief, max_steps, beam_width, None)
    }

    pub fn decode_sequence_with_costs(
        &self,
        anchor_belief: &Tensor,
        max_steps: usize,
        beam_width: usize,
        action_costs: Option<&std::collections::HashMap<u32, f32>>
    ) -> Result<Vec<u32>, PCError> {
        let anchor_flat = anchor_belief.flatten_all()?;
        let belief_dim = anchor_flat.dims()[0];
        let anchor_2d = anchor_flat.reshape((1, belief_dim))?;

        let mut beams = vec![(0.0f32, 0.0f32, Vec::<u32>::new(), anchor_2d.clone(), false)];

        for _step in 1..=max_steps {
            let mut new_beams = Vec::new();
            
            for (_, raw_score, seq, h_t, is_done) in &beams {
                if *is_done {
                    let norm_score = *raw_score / (seq.len() as f32).powf(0.7); 
                    new_beams.push((norm_score, *raw_score, seq.clone(), h_t.clone(), true));
                    continue;
                }

                let combined_input = Tensor::cat(&[h_t, &anchor_2d], 1)?;
                
                let update_gate = ops::sigmoid(&combined_input.matmul(&self.w_update.as_tensor().t()?)?)?;
                let h_hat = combined_input.matmul(&self.w_hidden.as_tensor().t()?)?.tanh()?;
                let ones = update_gate.ones_like()?;
                let diff = (ones - &update_gate)?;
                let h_next = (h_t.mul(&update_gate)? + h_hat.mul(&diff)?)?;
                
                let logits = h_next.matmul(&self.w_vocab.as_tensor().t()?)?;
                let log_probs = ops::log_softmax(&logits, 1)?;
                let log_probs_vec = log_probs.flatten_all()?.to_vec1::<f32>()?;

                for (token_id, &lp) in log_probs_vec.iter().enumerate() {
                    let mut new_seq = seq.clone();
                    new_seq.push(token_id as u32);
                    
                    let cost = action_costs
                        .and_then(|costs| costs.get(&(token_id as u32)))
                        .copied()
                        .unwrap_or(0.0);
                        
                    let new_raw_score = raw_score + lp - cost;
                    let norm_score = new_raw_score / (new_seq.len() as f32).powf(0.7);
                    
                    let done = token_id as u32 == 7; // EOF
                    new_beams.push((norm_score, new_raw_score, new_seq, h_next.clone(), done));
                }
            }

            new_beams.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            new_beams.truncate(beam_width);
            beams = new_beams;

            if beams.iter().all(|b| b.4) {
                break;
            }
        }
        Ok(beams[0].2.clone())
    }

    /// Robust Pseudo-BPTT Training Step with Eligibility Trace (Backpropagation through time)
    pub fn train_step(&mut self, belief: &Tensor, target_seq: &[u32], lr: f64) -> Result<f32, PCError> {
        let belief_flat = belief.flatten_all()?;
        let belief_dim = belief_flat.dims()[0];
        let belief_2d = belief_flat.reshape((1, belief_dim))?;
        
        let mut h_t = belief_2d.clone();
        let mut total_loss = 0.0f32;
        let lr_f32 = lr as f32;
        
        let mut dh_accumulated = Tensor::zeros_like(&h_t)?; // Eligibility trace
        
        for &target_id in target_seq.iter().rev() { // BACKWARD pass approximation
            let combined = Tensor::cat(&[&h_t, &belief_2d], 1)?;
            
            // Forward (recomputed for the trace)
            let update_gate = ops::sigmoid(&combined.matmul(&self.w_update.as_tensor().t()?)?)?;
            let h_hat = combined.matmul(&self.w_hidden.as_tensor().t()?)?.tanh()?;
            let diff = (update_gate.ones_like()? - &update_gate)?;
            let h_next = (h_t.mul(&update_gate)? + h_hat.mul(&diff)?)?;
            
            let logits = h_next.matmul(&self.w_vocab.as_tensor().t()?)?;
            let probs = ops::softmax(&logits, 1)?;
            let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
            
            let p_target = probs_vec[target_id as usize].max(1e-7);
            total_loss += -p_target.ln();

            // 1. Update w_vocab
            let mut grad_vocab = probs_vec.clone();
            grad_vocab[target_id as usize] -= 1.0;
            let grad_vocab_tensor = Tensor::from_vec(grad_vocab, probs.shape(), &self.device)?;
            
            let dw_vocab = grad_vocab_tensor.t()?.matmul(&h_next)?;
            let lr_tensor_vocab = Tensor::new(&[lr_f32], &self.device)?.broadcast_as(dw_vocab.shape())?;
            let new_w_vocab = self.w_vocab.as_tensor().sub(&dw_vocab.mul(&lr_tensor_vocab)?)?;
            self.w_vocab.set(&new_w_vocab)?;
            
            // 2. Accumulate gradient through time (BPTT Trace)
            let dh_current = grad_vocab_tensor.matmul(&self.w_vocab.as_tensor())?;
            dh_accumulated = (dh_accumulated + dh_current)?; 
            
            let dw_hidden = dh_accumulated.t()?.matmul(&combined)?;
            let lr_tensor_hidden = Tensor::new(&[lr_f32], &self.device)?.broadcast_as(dw_hidden.shape())?;
            let new_w_hidden = self.w_hidden.as_tensor().sub(&dw_hidden.mul(&lr_tensor_hidden)?)?;
            self.w_hidden.set(&new_w_hidden)?;
            
            h_t = h_next;
        }

        Ok(total_loss / target_seq.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_decoder_dynamic_resizing() -> Result<(), PCError> {
        let device = Device::Cpu;
        let belief_dim = 16;
        let mut decoder = ThoughtDecoder::new(belief_dim, 8, &device)?;

        let original_weights = decoder.w_vocab.as_tensor().to_vec2::<f32>()?;
        decoder.resize_vocab(10)?;
        let new_weights = decoder.w_vocab.as_tensor().to_vec2::<f32>()?;
        
        assert_eq!(new_weights.len(), 10);
        for i in 0..8 {
            assert_eq!(new_weights[i], original_weights[i]);
        }
        Ok(())
    }
}
