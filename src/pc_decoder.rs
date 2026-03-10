// src/pc_decoder.rs
use candle_core::{Tensor, Device, Var};
use candle_nn::ops;
use crate::pc_types::PCError;
use crate::types::ThoughtOp;

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
            // Initialize with smaller variance to prevent early saturation
            w_update: Var::randn(0f32, 0.01f32, (belief_dim, combined_dim), device)?,
            w_hidden: Var::randn(0f32, 0.01f32, (belief_dim, combined_dim), device)?,
            w_vocab: Var::randn(0f32, 0.01f32, (vocab_size, belief_dim), device)?,
            device: device.clone(),
        })
    }

    /// Graph of Thoughts (Beam Search) with Length Normalization
    pub fn decode_sequence(&self, anchor_belief: &Tensor, max_steps: usize, beam_width: usize) -> Result<Vec<u32>, PCError> {
        self.decode_sequence_with_costs(anchor_belief, max_steps, beam_width, None)
    }

    /// Graph of Thoughts with Length Normalization AND Action-Cost routing
    pub fn decode_sequence_with_costs(
        &self,
        anchor_belief: &Tensor,
        max_steps: usize,
        beam_width: usize,
        action_costs: Option<&std::collections::HashMap<u32, f32>> // NEW
    ) -> Result<Vec<u32>, PCError> {
        use std::collections::HashMap;
        
        let anchor_flat = anchor_belief.flatten_all()?;
        let belief_dim = anchor_flat.dims()[0];
        let anchor_2d = anchor_flat.reshape((1, belief_dim))?;

        // Beam structure: (Normalized Score, Raw Score, Sequence, Hidden State, Is Done)
        let mut beams = vec![(0.0f32, 0.0f32, Vec::<u32>::new(), anchor_2d.clone(), false)];

        for _step in 1..=max_steps {
            let mut new_beams = Vec::new();
            
            for (_, raw_score, seq, h_t, is_done) in &beams {
                if *is_done {
                    // Re-calculate normalized score for finished beams to compare fairly
                    let norm_score = *raw_score / (seq.len() as f32).powf(0.7); // 0.7 is a standard length penalty
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
                    
                    // Apply explicit action costs if provided
                    let cost = action_costs
                        .and_then(|costs| costs.get(&(token_id as u32)))
                        .copied()
                        .unwrap_or(0.0); // Default zero extra cost
                        
                    // Penalize score by cost
                    let new_raw_score = raw_score + lp - cost;
                    
                    let norm_score = new_raw_score / (new_seq.len() as f32).powf(0.7);
                    let done = token_id as u32 == 7; // EOF
                    new_beams.push((norm_score, new_raw_score, new_seq, h_next.clone(), done));
                }
            }

            // Sort by normalized score (descending)
            new_beams.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            new_beams.truncate(beam_width);
            beams = new_beams;

            if beams.iter().all(|b| b.4) {
                break;
            }
        }

        // Return the sequence of the best beam
        Ok(beams[0].2.clone())
    }

    /// Robust Pseudo-BPTT Training Step
    pub fn train_step(&mut self, belief: &Tensor, target_seq: &[u32], lr: f64) -> Result<f32, PCError> {
        let belief_flat = belief.flatten_all()?;
        let belief_dim = belief_flat.dims()[0];
        let belief_2d = belief_flat.reshape((1, belief_dim))?;
        
        let mut h_t = belief_2d.clone();
        let mut total_loss = 0.0f32;
        let lr_f32 = lr as f32;
        
        for &target_id in target_seq {
            let combined = Tensor::cat(&[&h_t, &belief_2d], 1)?;
            
            // Forward pass for the step
            let matmul_update = combined.matmul(&self.w_update.as_tensor().t()?)?;
            let update_gate = ops::sigmoid(&matmul_update)?;
            let h_hat = combined.matmul(&self.w_hidden.as_tensor().t()?)?.tanh()?;
            
            let ones = update_gate.ones_like()?;
            let diff = (ones - &update_gate)?;
            let h_next = (h_t.mul(&update_gate)? + h_hat.mul(&diff)?)?;
            
            let logits = h_next.matmul(&self.w_vocab.as_tensor().t()?)?;
            let probs = ops::softmax(&logits, 1)?;
            
            // Calculate scalar loss (Cross Entropy: -log(p_target))
            let probs_vec = probs.flatten_all()?.to_vec1::<f32>()?;
            let p_target = probs_vec[target_id as usize].max(1e-7); // Avoid log(0)
            let step_loss = -p_target.ln();
            total_loss += step_loss;

            // --- Pseudo-BPTT Updates ---
            // 1. Update w_vocab
            let mut grad_vocab = probs_vec.clone();
            grad_vocab[target_id as usize] -= 1.0;
            let grad_vocab_tensor = Tensor::from_vec(grad_vocab, probs.shape(), &self.device)?;
            
            let dw_vocab = grad_vocab_tensor.t()?.matmul(&h_next)?;
            let lr_tensor_vocab = Tensor::new(&[lr_f32], &self.device)?.broadcast_as(dw_vocab.shape())?;
            let dw_scaled = dw_vocab.mul(&lr_tensor_vocab)?;
            let new_w_vocab = self.w_vocab.as_tensor().sub(&dw_scaled)?;
            
            // 2. Propagate gradient back to h_next to update internal GRU weights
            // dh = W_vocab^T * grad_vocab
            let dh = grad_vocab_tensor.matmul(&self.w_vocab.as_tensor())?;
            
            // Simplified update for w_hidden based on dh
            // This ensures the internal state transitions are actually learned
            let dw_hidden = dh.t()?.matmul(&combined)?;
            let lr_tensor_hidden = Tensor::new(&[lr_f32], &self.device)?.broadcast_as(dw_hidden.shape())?;
            let dw_hidden_scaled = dw_hidden.mul(&lr_tensor_hidden)?;
            let new_w_hidden = self.w_hidden.as_tensor().sub(&dw_hidden_scaled)?;
            
            // Apply updates
            self.w_vocab.set(&new_w_vocab)?;
            self.w_hidden.set(&new_w_hidden)?;
            // We omit w_update here for stability, allowing the hidden projection to do the heavy lifting
            
            h_t = h_next;
        }

        let scalar_loss = total_loss / target_seq.len() as f32;
        Ok(scalar_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// 1. Тест на защиту от паники при разных формах входного тензора Belief
    #[test]
    fn test_decoder_accepts_various_input_shapes() -> Result<(), PCError> {
        let device = Device::Cpu;
        let dim = 32;
        let mut decoder = ThoughtDecoder::new(dim, 10, &device)?;
        let target_seq = vec![1, 2, 3];

        // Симуляция 1: Одномерный тензор [32]
        let belief_1d = Tensor::randn(0f32, 1.0, (dim,), &device)?;
        decoder.train_step(&belief_1d, &target_seq, 0.01)?;
        let _ = decoder.decode_sequence(&belief_1d, 5, 3)?;

        // Симуляция 2: Двумерный тензор-строка [1, 32]
        let belief_row = Tensor::randn(0f32, 1.0, (1, dim), &device)?;
        decoder.train_step(&belief_row, &target_seq, 0.01)?;
        let _ = decoder.decode_sequence(&belief_row, 5, 3)?;

        // Симуляция 3: Двумерный тензор-столбец [32, 1] (Как выдает PC Hierarchy)
        // ИМЕННО ОН ВЫЗЫВАЛ ОШИБКУ lhs: [4096] ДО ФИКСА
        let belief_col = Tensor::randn(0f32, 1.0, (dim, 1), &device)?;
        decoder.train_step(&belief_col, &target_seq, 0.01)?;
        let _ = decoder.decode_sequence(&belief_col, 5, 3)?;

        Ok(())
    }

    /// 2. Тест на правильное протекание градиентов (BPTT)
    #[test]
    fn test_train_step_updates_multiple_matrices() -> Result<(), PCError> {
        let device = Device::Cpu;
        let mut decoder = ThoughtDecoder::new(16, 8, &device)?;
        let belief = Tensor::randn(0f32, 1.0, (16,), &device)?;
        let target_seq = vec![0, 1, 4];

        // Запоминаем состояние весов ДО обучения
        let old_w_vocab = decoder.w_vocab.as_tensor().to_vec2::<f32>()?;
        let old_w_hidden = decoder.w_hidden.as_tensor().to_vec2::<f32>()?;

        // Делаем шаг обучения
        let loss = decoder.train_step(&belief, &target_seq, 0.1)?;
        assert!(loss > 0.0, "Loss не должна быть нулевой");

        // Запоминаем состояние весов ПОСЛЕ обучения
        let new_w_vocab = decoder.w_vocab.as_tensor().to_vec2::<f32>()?;
        let new_w_hidden = decoder.w_hidden.as_tensor().to_vec2::<f32>()?;

        // Убеждаемся, что веса изменились. 
        // Если w_hidden не меняется — сеть страдает от Mode Collapse.
        assert_ne!(old_w_vocab, new_w_vocab, "CRITICAL: Матрица классификатора (w_vocab) не обновилась!");
        assert_ne!(old_w_hidden, new_w_hidden, "CRITICAL: Скрытая матрица переходов (w_hidden) не обновилась! Риск коллапса.");

        Ok(())
    }

    /// 3. Интеграционный тест: Обучение + Beam Search Генерация
    #[test]
    fn test_overfitting_and_beam_search_retrieval() -> Result<(), PCError> {
        let device = Device::Cpu;
        let mut decoder = ThoughtDecoder::new(32, 8, &device)?;
        let belief = Tensor::randn(0f32, 1.0, (32,), &device)?;
        
        // Целевая последовательность мыслей: Define (0) -> Iterate (1) -> Compute (3) -> EOF (7)
        let target_seq = vec![0, 1, 3, 7];
        
        // Жестко переобучаем декодер на этот один пример (Overfitting)
        let mut loss = f32::MAX;
        for _ in 0..500 {
            loss = decoder.train_step(&belief, &target_seq, 0.1)?;
        }
        
        // Loss должна стремиться к нулю (допускаем значение < 1.0 для стабильности)
        assert!(loss < 1.0, "Декодер не смог сойтись (упасть в локальный минимум). Финальная loss: {}", loss);

        // Теперь просим Beam Search сгенерировать последовательность из того же Belief
        let generated_seq = decoder.decode_sequence(&belief, 5, 3)?;
        
        // Beam Search должен идеально выдать то, чему мы его научили
        assert_eq!(
            generated_seq, 
            target_seq, 
            "Beam search сгенерировал неправильный план: {:?} вместо {:?}", 
            generated_seq, target_seq
        );

        Ok(())
    }
}
