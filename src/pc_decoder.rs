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
        // Xavier/Kaiming-style initialization
        let std_hidden = (1.0 / combined_dim as f32).sqrt();
        let std_vocab = (1.0 / belief_dim as f32).sqrt();

        Ok(Self {
            w_update: Var::randn(0f32, std_hidden, (belief_dim, combined_dim), device)?,
            w_hidden: Var::randn(0f32, std_hidden, (belief_dim, combined_dim), device)?,
            w_vocab: Var::randn(0f32, std_vocab, (vocab_size, belief_dim), device)?,
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

        // 🔴 FIX: Normalize and scale the anchor to match what the network was trained on
        let anchor_norm = anchor_2d.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let target_norm = (belief_dim as f64).sqrt();
        let scale_factor = if anchor_norm > 1e-6 { target_norm / anchor_norm } else { 1.0 };
        let anchor_2d = anchor_2d.affine(scale_factor, 0.0)?;

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
        
        // 🔴 FIX: Normalize and scale the belief to match Xavier expected variance
        let belief_norm = belief_2d.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let target_norm = (belief_dim as f64).sqrt();
        let scale_factor = if belief_norm > 1e-6 { target_norm / belief_norm } else { 1.0 };
        let belief_2d = belief_2d.affine(scale_factor, 0.0)?;
        
        let mut h_t = belief_2d.clone();
        let mut total_loss = 0.0f32;
        
        // 🔴 BUG FIX: Exact BPTT requires storing forward states, not iterating backwards!
        struct StepState {
            combined: Tensor,
            update_gate: Tensor,
            h_hat: Tensor,
            h_prev: Tensor,
            h_next: Tensor,
            probs_vec: Vec<f32>,
            target_id: u32,
        }
        
        let mut states = Vec::with_capacity(target_seq.len());
        
        // --- FORWARD PASS ---
        for &target_id in target_seq.iter() {
            let combined = Tensor::cat(&[&h_t, &belief_2d], 1)?;
            
            let update_gate = ops::sigmoid(&combined.matmul(&self.w_update.as_tensor().t()?)?)?;
            let h_hat = combined.matmul(&self.w_hidden.as_tensor().t()?)?.tanh()?;
            let diff = (update_gate.ones_like()? - &update_gate)?;
            let h_next = (h_t.mul(&update_gate)? + h_hat.mul(&diff)?)?;
            
            let logits = h_next.matmul(&self.w_vocab.as_tensor().t()?)?;
            let probs = ops::softmax(&logits, 1)?;
            let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
            
            let p_target = probs_vec[target_id as usize].max(1e-7);
            total_loss += -p_target.ln();
            
            states.push(StepState {
                combined,
                update_gate,
                h_hat,
                h_prev: h_t.clone(),
                h_next: h_next.clone(),
                probs_vec,
                target_id,
            });
            
            h_t = h_next;
        }
        
        // --- BACKWARD PASS ---
        let mut dh_next_step = Tensor::zeros_like(&h_t)?;
        let mut dw_vocab_acc = Tensor::zeros_like(self.w_vocab.as_tensor())?;
        let mut dw_hidden_acc = Tensor::zeros_like(self.w_hidden.as_tensor())?;
        let mut dw_update_acc = Tensor::zeros_like(self.w_update.as_tensor())?;
        
        for state in states.into_iter().rev() {
            // 1. Loss gradient w.r.t logits
            let mut grad_vocab = state.probs_vec;
            grad_vocab[state.target_id as usize] -= 1.0;
            let grad_vocab_tensor = Tensor::from_vec(grad_vocab, (1, self.w_vocab.shape().dims()[0]), &self.device)?;
            
            // 2. Gradients for w_vocab
            let dw_vocab = grad_vocab_tensor.t()?.matmul(&state.h_next)?;
            dw_vocab_acc = (dw_vocab_acc + dw_vocab)?;
            
            // 3. Gradient flowing into h_next
            let dh_from_loss = grad_vocab_tensor.matmul(&self.w_vocab.as_tensor())?;
            let dh_total = (dh_from_loss + dh_next_step)?;
            
            // 4. Backprop through the GRU-like step
            let diff = state.update_gate.ones_like()?.sub(&state.update_gate)?;
            let dh_hat = (&dh_total * &diff)?;
            let dh_update = (&dh_total * &(state.h_prev.sub(&state.h_hat)?))?;
            
            // 5. Gradients for w_hidden (through tanh)
            let tanh_deriv = state.h_hat.ones_like()?.sub(&state.h_hat.sqr()?)?;
            let d_pre_tanh = (dh_hat * tanh_deriv)?;
            let dw_hidden = d_pre_tanh.t()?.matmul(&state.combined)?;
            dw_hidden_acc = (dw_hidden_acc + dw_hidden)?;
            
            // 6. Gradients for w_update (through sigmoid)
            let sig_deriv = (&state.update_gate * &(state.update_gate.ones_like()?.sub(&state.update_gate)?))?;
            let d_pre_sig = (dh_update * sig_deriv)?;
            let dw_update = d_pre_sig.t()?.matmul(&state.combined)?;
            dw_update_acc = (dw_update_acc + dw_update)?;
            
            // 7. Compute dh_next_step for the previous iteration
            let d_combined_from_hidden = d_pre_tanh.matmul(&self.w_hidden.as_tensor())?;
            let d_combined_from_update = d_pre_sig.matmul(&self.w_update.as_tensor())?;
            let d_combined = (d_combined_from_hidden + d_combined_from_update)?;
            
            // Extract the h_prev part (first belief_dim columns)
            let d_h_prev = d_combined.narrow(1, 0, belief_dim)?;
            dh_next_step = ((dh_total * state.update_gate)? + d_h_prev)?;
        }
        
        // --- GRADIENT CLIPPING ---
        // 🔴 FIX: Use safe .affine() math to scale gradients, preventing tensor shape bugs
        let clip_val = 5.0f64;
        let dw_vocab_norm = dw_vocab_acc.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let dw_vocab_clipped = if dw_vocab_norm > clip_val { dw_vocab_acc.affine(clip_val / dw_vocab_norm, 0.0)? } else { dw_vocab_acc };
        
        let dw_hidden_norm = dw_hidden_acc.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let dw_hidden_clipped = if dw_hidden_norm > clip_val { dw_hidden_acc.affine(clip_val / dw_hidden_norm, 0.0)? } else { dw_hidden_acc };
        
        let dw_update_norm = dw_update_acc.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let dw_update_clipped = if dw_update_norm > clip_val { dw_update_acc.affine(clip_val / dw_update_norm, 0.0)? } else { dw_update_acc };
        
        // --- APPLY GRADIENTS ---
        self.w_vocab.set(&self.w_vocab.as_tensor().broadcast_add(&dw_vocab_clipped.affine(-lr, 0.0)?)?)?;
        self.w_hidden.set(&self.w_hidden.as_tensor().broadcast_add(&dw_hidden_clipped.affine(-lr, 0.0)?)?)?;
        self.w_update.set(&self.w_update.as_tensor().broadcast_add(&dw_update_clipped.affine(-lr, 0.0)?)?)?;

        Ok(total_loss / target_seq.len() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use tracing::info;

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

    #[test]
    fn test_bptt_actually_learns_a_sequence() -> Result<(), PCError> {
        let device = Device::Cpu;
        let belief_dim = 16;
        let vocab_size = 8;
        let mut decoder = ThoughtDecoder::new(belief_dim, vocab_size, &device)?;

        let belief = Tensor::randn(0f32, 1.0, (belief_dim, 1), &device)?;
        let target_seq = vec![1, 2, 3, 4, 7]; // A simple sequence ending in EOF

        // Clone initial weights to verify they change
        let w_vocab_before = decoder.w_vocab.as_tensor().to_vec2::<f32>()?;
        let w_hidden_before = decoder.w_hidden.as_tensor().to_vec2::<f32>()?;
        let w_update_before = decoder.w_update.as_tensor().to_vec2::<f32>()?;

        // Train for a few epochs
        let mut last_loss = f32::MAX;
        let mut initial_loss = 0.0;

        for epoch in 0..20 {
            let loss = decoder.train_step(&belief, &target_seq, 0.01)?;
            if epoch == 0 {
                initial_loss = loss;
            }
            last_loss = loss;
        }

        // Assert that the loss has decreased significantly
        assert!(last_loss < initial_loss, "Decoder loss did not decrease after training. Initial: {}, Final: {}", initial_loss, last_loss);
        // The loss should have decreased by at least 5% (reasonable for 20 epochs)
        let loss_decrease = (initial_loss - last_loss) / initial_loss;
        assert!(loss_decrease > 0.05, "Loss decrease too small: {:.2}% (initial: {}, final: {})", loss_decrease * 100.0, initial_loss, last_loss);

        // Assert that all weight matrices were updated
        let w_vocab_after = decoder.w_vocab.as_tensor().to_vec2::<f32>()?;
        let w_hidden_after = decoder.w_hidden.as_tensor().to_vec2::<f32>()?;
        let w_update_after = decoder.w_update.as_tensor().to_vec2::<f32>()?;

        assert_ne!(w_vocab_before, w_vocab_after, "w_vocab was not updated during training.");
        assert_ne!(w_hidden_before, w_hidden_after, "w_hidden was not updated during training.");
        assert_ne!(w_update_before, w_update_after, "w_update was not updated during training.");

        Ok(())
    }

    #[test]
    fn test_decoder_gradient_flow() -> Result<(), PCError> {
        let device = Device::Cpu;
        let mut decoder = ThoughtDecoder::new(32, 8, &device)?;
        let belief = Tensor::ones((32, 1), candle_core::DType::F32, &device)?;
        let target = vec![1, 7]; // Define -> EOF

        let loss_1 = decoder.train_step(&belief, &target, 0.5)?;
        let loss_2 = decoder.train_step(&belief, &target, 0.5)?;

        assert!(loss_2 < loss_1, "Loss must decrease. Gradients are likely zero. Check initialization.");
        Ok(())
    }

    #[test]
    fn test_escape_from_dead_loss() -> Result<(), PCError> {
        // 🔴 CRITICAL TEST: Ensure decoder can escape from the "dead network" state
        // where loss is stuck at exactly 2.0795 (uniform probability distribution)
        let device = Device::Cpu;
        let belief_dim = 64;
        let vocab_size = 8;
        let mut decoder = ThoughtDecoder::new(belief_dim, vocab_size, &device)?;

        // Create a random belief tensor
        let belief = Tensor::randn(0f32, 1.0, (belief_dim, 1), &device)?;
        let target_seq = vec![0, 1, 2, 3, 4, 5, 6, 7]; // All tokens to avoid bias

        // Train for several steps with a healthy learning rate
        let mut losses = Vec::new();
        for step in 0..30 {
            let loss = decoder.train_step(&belief, &target_seq, 0.05)?;
            losses.push(loss);
            
            // If we ever hit exactly 2.0795 (within floating epsilon), that's the dead network bug
            if (loss - 2.0795).abs() < 1e-4 {
                panic!("Decoder stuck in dead network state (loss = {:.4}) at step {}. This indicates uniform probability distribution bug.", loss, step);
            }
        }

        // Verify that loss is NOT stuck at the uniform probability value
        let final_loss = losses.last().unwrap();
        assert!(
            (*final_loss - 2.0795).abs() > 0.1,
            "Decoder failed to escape dead network state. Final loss {:.4} is too close to uniform probability 2.0795.",
            final_loss
        );

        // Verify that loss decreased overall (not required to be monotonic, but should trend down)
        let avg_first_5 = losses.iter().take(5).sum::<f32>() / 5.0;
        let avg_last_5 = losses.iter().rev().take(5).sum::<f32>() / 5.0;
        assert!(
            avg_last_5 < avg_first_5,
            "Decoder loss did not improve: first 5 avg = {:.4}, last 5 avg = {:.4}",
            avg_first_5, avg_last_5
        );

        info!("✅ Escape-from-dead-loss test passed. Loss trajectory: {:.4} -> {:.4}", losses[0], losses.last().unwrap());
        Ok(())
    }
}
