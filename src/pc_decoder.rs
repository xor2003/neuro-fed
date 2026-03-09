// src/pc_decoder.rs
use candle_core::{Tensor, Device, Var, DType};
use candle_nn::{ops, loss, Optimizer};
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
            // Используем Var для того, чтобы градиенты накапливались
            w_update: Var::randn(0f32, 0.02f32, (belief_dim, combined_dim), device)?,
            w_hidden: Var::randn(0f32, 0.02f32, (belief_dim, combined_dim), device)?,
            w_vocab: Var::randn(0f32, 0.02f32, (vocab_size, belief_dim), device)?,
            device: device.clone(),
        })
    }

    /// Graph of Thoughts (Beam Search). Ищет лучший путь вместо жадного выбора первого попавшегося.
    pub fn decode_sequence(&self, anchor_belief: &Tensor, max_steps: usize, beam_width: usize) -> Result<Vec<u32>, PCError> {
        // ENFORCE 2D SHAPE: Ensure anchor is [1, belief_dim]
        let anchor_flat = anchor_belief.flatten_all()?;
        let belief_dim = anchor_flat.dims()[0];
        let anchor_2d = anchor_flat.reshape((1, belief_dim))?;

        let mut beams = vec![(0.0f32, Vec::<u32>::new(), anchor_2d.clone(), false)];

        for _ in 0..max_steps {
            let mut new_beams = Vec::new();
            
            for (score, seq, h_t, is_done) in &beams {
                if *is_done {
                    new_beams.push((*score, seq.clone(), h_t.clone(), true));
                    continue;
                }

                // 1. Cat along dimension 1 (columns) -> [1, belief_dim * 2]
                let combined_input = Tensor::cat(&[h_t, &anchor_2d], 1)?;
                
                // 2. Pseudo-GRU step
                let matmul_update = combined_input.matmul(&self.w_update.as_tensor().t()?)?;
                let update_gate = ops::sigmoid(&matmul_update)?;
                let h_hat = combined_input.matmul(&self.w_hidden.as_tensor().t()?)?.tanh()?;
                let ones = update_gate.ones_like()?;
                let diff = (ones - &update_gate)?;
                let h_next = (h_t.mul(&update_gate)? + h_hat.mul(&diff)?)?;
                
                // 3. Logits
                let logits = h_next.matmul(&self.w_vocab.as_tensor().t()?)?;
                // Calculate log softmax along the vocabulary dimension (dim 1)
                let log_probs = ops::log_softmax(&logits, 1)?;
                let log_probs_vec = log_probs.flatten_all()?.to_vec1::<f32>()?;

                for (token_id, &lp) in log_probs_vec.iter().enumerate() {
                    let mut new_seq = seq.clone();
                    new_seq.push(token_id as u32);
                    let new_score = score + lp;
                    let done = token_id as u32 == ThoughtOp::EOF as u32;
                    new_beams.push((new_score, new_seq, h_next.clone(), done));
                }
            }

            new_beams.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            new_beams.truncate(beam_width);
            beams = new_beams;

            if beams.iter().all(|b| b.3) {
                break;
            }
        }

        Ok(beams[0].1.clone())
    }

    pub fn train_step(&mut self, belief: &Tensor, target_seq: &[u32], lr: f64) -> Result<f32, PCError> {
        // ENFORCE 2D SHAPE: Ensure belief is [1, belief_dim]
        let belief_flat = belief.flatten_all()?;
        let belief_dim = belief_flat.dims()[0];
        let belief_2d = belief_flat.reshape((1, belief_dim))?;
        
        let mut h_t = belief_2d.clone();
        let mut total_loss = 0.0f32;
        
        for &target_id in target_seq {
            // Cat along dimension 1 (columns) ->[1, belief_dim * 2]
            let combined = Tensor::cat(&[&h_t, &belief_2d], 1)?;
            
            let matmul_update = combined.matmul(&self.w_update.as_tensor().t()?)?;
            let update_gate = ops::sigmoid(&matmul_update)?;
            let h_hat = combined.matmul(&self.w_hidden.as_tensor().t()?)?.tanh()?;
            let ones = update_gate.ones_like()?;
            let diff = (ones - &update_gate)?;
            h_t = (h_t.mul(&update_gate)? + h_hat.mul(&diff)?)?;
            
            // Logits shape: [1, vocab_size]
            let logits = h_t.matmul(&self.w_vocab.as_tensor().t()?)?;
            
            // Cross entropy expects [batch, classes] and [batch] for targets
            let target_tensor = Tensor::new(&[target_id], &self.device)?;
            let step_loss = loss::cross_entropy(&logits, &target_tensor)?;
            
            // Convert scalar tensor to f32
            let step_loss_scalar = step_loss.to_scalar::<f32>()?;
            total_loss += step_loss_scalar;
        }

        let scalar_loss = total_loss / target_seq.len() as f32;

        // 🔴 FIX: Temporary fallback to pseudo-SGD.
        // Real BPTT with `candle_nn::SGD` requires the loss to be a `Var` graph,
        // but `cross_entropy` currently breaks the gradient trace in this setup.
        // We will manually update weights to ensure stability without crashing.
        let lr_f32 = lr as f32;
        
        // Manual update for w_vocab (simplest pseudo-gradient)
        // This prevents the node from crashing while still allowing basic learning
        let final_logits = h_t.matmul(&self.w_vocab.as_tensor().t()?)?;
        let probs = ops::softmax(&final_logits, 1)?;
        let mut grad_vec = probs.flatten_all()?.to_vec1::<f32>()?;
        if let Some(last_target) = target_seq.last() {
             grad_vec[*last_target as usize] -= 1.0;
        }
        let grad_tensor = Tensor::from_vec(grad_vec, probs.shape(), &self.device)?;
        let dw_vocab = grad_tensor.t()?.matmul(&h_t)?;
        // Scale by learning rate - create scalar tensor and broadcast
        let lr_tensor = Tensor::new(&[lr_f32], &self.device)?;
        let lr_broadcasted = lr_tensor.broadcast_as(dw_vocab.shape())?;
        let dw_scaled = dw_vocab.mul(&lr_broadcasted)?;
        let new_w_vocab = self.w_vocab.as_tensor().sub(&dw_scaled)?;
        self.w_vocab.set(&new_w_vocab)?;

        Ok(scalar_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_search_and_backprop() -> Result<(), PCError> {
        let device = Device::Cpu;
        let mut decoder = ThoughtDecoder::new(16, 8, &device)?;
        let belief = Tensor::randn(0f32, 1.0, (16, 1), &device)?;
        
        // 1. Проверяем, что Beam Search работает и не падает
        let seq = decoder.decode_sequence(&belief, 5, 3)?;
        assert!(!seq.is_empty(), "Beam search должен выдать результат");

        // 2. Проверяем, что Backprop обновляет скрытые веса, а не только словарь
        let old_hidden = decoder.w_hidden.as_tensor().to_vec2::<f32>()?;
        let target = vec![0, 1, 2, 7];
        decoder.train_step(&belief, &target, 0.1)?;
        let new_hidden = decoder.w_hidden.as_tensor().to_vec2::<f32>()?;

        assert_ne!(old_hidden, new_hidden, "BPTT не сработал: скрытые веса не обновились!");
        Ok(())
    }

    #[tokio::test]
    async fn test_end_to_end_bootstrap_training_pipeline() -> Result<(), Box<dyn std::error::Error>> {
        // This integration test prevents the dreaded "shape mismatch in matmul" error
        // by ensuring the entire data pipeline from text to decoder training is correct.

        use std::sync::Arc;
        use tokio::sync::Mutex;
        use candle_core::Device;

        // 1. Full system setup (as in main.rs)
        let device = Device::Cpu;
        let config = crate::config::NodeConfig::default();
        
        // We can't use the real ModelManager here, so we'll mock the MLEngine init
        // For this test, we assume a model is downloaded.
        let model_path = "models/TinyLlama-1.1B-Chat-v1.0.Q2_K.gguf"; // Make sure this file exists
        if !std::path::Path::new(model_path).exists() {
            // Skip the test if the model isn't available to avoid failing CI/local tests
            // In a real project, you might auto-download this or use a smaller test-specific model
            println!("Skipping bootstrap pipeline test: GGUF model not found at {}", model_path);
            return Ok(());
        }
        
        let engine = Arc::new(Mutex::new(crate::ml_engine::MLEngine::new(
            model_path,
            crate::types::DeviceType { name: "cpu".into(), description: "".into(), supported: true }
        )?));

        let pc_config: crate::pc_hierarchy::PCConfig = config.pc_config.into();
        let pc_hierarchy = Arc::new(Mutex::new(crate::pc_hierarchy::PredictiveCoding::new(pc_config)?));

        let dict = Arc::new(Mutex::new(crate::types::CognitiveDictionary::default()));
        
        let top_belief_dim = *pc_hierarchy.lock().await.config.dim_per_level.last().unwrap();
        let vocab_size = dict.lock().await.len();
        
        let mut decoder = ThoughtDecoder::new(top_belief_dim, vocab_size, &device)?;

        // 2. Simulate data generation from bootstrap.rs
        let text = "Write a quicksort function in Python".to_string();
        let target_seq = vec![0, 1, 2, 3, 4, 7]; // Example sequence
        
        // A) Get 2048-dim embedding from raw text
        let embedding_2048d = engine.lock().await.process_text(&text).await?;
        assert_eq!(embedding_2048d.dims(), &[1, 2048], "MLEngine should produce a [1, 2048] embedding");

        // B) Run PC inference to compress it to a 512-dim belief
        let mut pc = pc_hierarchy.lock().await;
        pc.infer(&embedding_2048d, 15)?;
        let belief_512d = pc.levels.last().unwrap().beliefs.clone(); // Shape [512, 1]
        
        // 3. THE CRITICAL ASSERTION
        // The decoder's weights expect an input of `belief_dim * 2` (1024)
        // after concatenating h_t (512) and anchor_belief (512).
        // Our input `belief` must match the `h_t` dimension.
        let expected_decoder_input_dim = decoder.w_hidden.dims()[1] / 2;
        assert_eq!(
            belief_512d.dims()[0],
            expected_decoder_input_dim,
            "Dimension mismatch! PC top layer belief ({}) does not match ThoughtDecoder input dimension ({}).",
            belief_512d.dims()[0],
            expected_decoder_input_dim
        );

        // 4. Run one train step to confirm no `matmul` error occurs
        let loss_result = decoder.train_step(&belief_512d, &target_seq, 0.01);
        
        assert!(loss_result.is_ok(), "train_step failed with an error: {:?}", loss_result.err());
        let loss = loss_result?;
        assert!(loss.is_finite() && loss > 0.0, "Loss calculation is incorrect");
        
        println!("Bootstrap pipeline test passed with loss: {}", loss);

        Ok(())
    }

    #[test]
    fn test_decoder_shape_safety_prevents_4096_crash() -> Result<(), PCError> {
        let device = Device::Cpu;
        let belief_dim = 512;
        let vocab_size = 8;
        
        let mut decoder = ThoughtDecoder::new(belief_dim, vocab_size, &device)?;
        
        // Simulate a raw belief tensor from PC (which might accidentally be 2D[512, 1] or [1, 512])
        // We test the worst case:[512, 1] (column vector)
        let raw_pc_belief = Tensor::randn(0f32, 1.0, (512, 1), &device)?;
        
        let target_seq = vec![0, 1, 2, 7];
        
        // This call MUST NOT PANIC with "shape mismatch in matmul, lhs: [4096]"
        let loss = decoder.train_step(&raw_pc_belief, &target_seq, 0.01)?;
        
        assert!(loss > 0.0, "Loss should be calculated successfully");
        
        // Ensure decode sequence also works with raw vectors
        let sequence = decoder.decode_sequence(&raw_pc_belief, 5, 3)?;
        assert!(!sequence.is_empty(), "Decoder should generate a sequence");
        
        Ok(())
    }
    
    #[cfg(test)]
    mod pipeline_tests {
        use super::*;
        use candle_core::{Device, Tensor, DType};
        use crate::pc_hierarchy::{PredictiveCoding, PCConfig};
        use crate::types::ThoughtOp;

        #[test]
        fn test_thought_decoder_training_pipeline_dimensions() -> Result<(), Box<dyn std::error::Error>> {
            let device = Device::Cpu;

            // 1. Настройка иерархии (как в реальности: 2048 -> 1024 -> 512)
            let embedding_dim = 2048;
            let pc_dims = vec![2048, 1024, 512];
            let pc_config = PCConfig::new(3, pc_dims.clone());
            let mut pc = PredictiveCoding::new_with_device(pc_config, device.clone())?;

            // 2. Настройка декодера (должен принимать размерность верхнего слоя PC - 512)
            let top_belief_dim = 512;
            let vocab_size = 8;
            let mut decoder = ThoughtDecoder::new(top_belief_dim, vocab_size, &device)?;

            // 3. Имитируем выход MLEngine (вектор 2048)
            let raw_embedding = Tensor::randn(0f32, 1.0, (1, embedding_dim), &device)?;

            // 4. Прогоняем через PC (Инференс), чтобы сжать 2048 -> 512
            pc.infer(&raw_embedding, 5)?;
            let compressed_belief = pc.levels.last().unwrap().beliefs.clone();
            
            // Проверяем, что PC реально выдал 512 (в форме [512, 1])
            assert_eq!(compressed_belief.dims(), &[512, 1], "PC должен сжать вектор до 512");

            // 5. Подготавливаем данные для декодера (цепочка мыслей)
            let target_sequence = vec![
                ThoughtOp::Define as u32,
                ThoughtOp::Iterate as u32,
                ThoughtOp::Return as u32,
                ThoughtOp::EOF as u32,
            ];

            // 6. КРИТИЧЕСКАЯ ПРОВЕРКА: Попытка обучения
            // Если здесь будет ошибка макс. размера [4096], тест упадет.
            // Мы используем flatten_all(), чтобы передать чистый вектор [512]
            let belief_for_train = compressed_belief.flatten_all()?;
            let loss = decoder.train_step(&belief_for_train, &target_sequence, 0.01);

            assert!(loss.is_ok(), "Ошибка размерности при обучении: {:?}", loss.err());
            println!("Loss calculated successfully: {}", loss.unwrap());

            // 7. Проверка генерации (Beam Search) с тем же вектором
            let gen_seq = decoder.decode_sequence(&belief_for_train, 5, 3);
            assert!(gen_seq.is_ok(), "Ошибка размерности при генерации");
            assert!(!gen_seq.unwrap().is_empty());

            Ok(())
        }
    }
}
