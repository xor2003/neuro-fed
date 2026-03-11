use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;
use candle_core::Tensor;
use crate::ml_engine::MLEngine;
use crate::pc_decoder::ThoughtDecoder;
use crate::pc_hierarchy::PredictiveCoding;
use crate::types::{CognitiveDictionary, ThoughtOp};

pub struct BootstrapManager {
    ml_engine: Arc<RwLock<MLEngine>>,
    thought_decoder: Arc<RwLock<ThoughtDecoder>>,
    dict: Arc<RwLock<CognitiveDictionary>>,
    pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
    config: crate::config::BootstrapConfig,
}

impl BootstrapManager {
    pub fn new(
        ml_engine: Arc<RwLock<MLEngine>>,
        thought_decoder: Arc<RwLock<ThoughtDecoder>>,
        dict: Arc<RwLock<CognitiveDictionary>>,
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        config: crate::config::BootstrapConfig,
    ) -> Self {
        Self { ml_engine, thought_decoder, dict, pc_hierarchy, config }
    }

    pub async fn run_synthetic_training(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Запускаем Curriculum Learning для Декодера Мыслей...");
        let mut synthetic_data = self.generate_synthetic_dataset().await;
        
        synthetic_data.sort_by_key(|(_, seq)| seq.len());
        
        let mut decoder = self.thought_decoder.write().await;

        let batch_size = self.config.batch_size.max(1);
        let max_epochs = self.config.max_epochs.max(100); // Ensure enough epochs for actual learning
        
        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;
            let mut steps = 0;
            for batch in synthetic_data.chunks(batch_size) {
                for (belief, seq) in batch {
                    // 🔴 FIX: Ensure learning rate doesn't drop below 0.01 during bootstrap
                    let lr = self.config.learning_rate.max(0.01) as f64;
                    let loss = decoder.train_step(belief, seq, lr)?;
                    total_loss += loss;
                    steps += 1;
                }
            }
            if epoch % 20 == 0 || epoch == max_epochs - 1 {
                 info!("Эпоха {}: Loss = {:.4}", epoch, total_loss / steps as f32);
            }
        }
        
        info!("✅ Синтетическое обучение декодера завершено.");
        Ok(())
    }

    async fn generate_synthetic_dataset(&self) -> Vec<(Tensor, Vec<u32>)> {
        let engine = self.ml_engine.read().await;
        let dict = self.dict.read().await;

        let q1 = "Напиши функцию quicksort на Python".to_string();
        let seq1 = vec![
            dict.op_to_id[&ThoughtOp::Define],
            dict.op_to_id[&ThoughtOp::Check],
            dict.op_to_id[&ThoughtOp::Iterate],
            dict.op_to_id[&ThoughtOp::Compute],
            dict.op_to_id[&ThoughtOp::Return],
            dict.op_to_id[&ThoughtOp::EOF],
        ];

        let q2 = "Что такое фотосинтез?".to_string();
        let seq2 = vec![
            dict.op_to_id[&ThoughtOp::Explain],
            dict.op_to_id[&ThoughtOp::Define],
            dict.op_to_id[&ThoughtOp::EOF],
        ];
        
        let q1_emb = engine.process_text(&q1).await.unwrap();
        let q2_emb = engine.process_text(&q2).await.unwrap();

        // 🔴 THE FIX: Run inference to get the compressed 512-dim belief
        let mut pc = self.pc_hierarchy.write().await;
        
        // 🔴 CRITICAL FIX: Reset PC state before processing a new unrelated prompt
        for level in pc.levels.iter_mut() {
            level.beliefs = level.beliefs.zeros_like().unwrap();
            level.prev_beliefs = level.prev_beliefs.zeros_like().unwrap();
        }
        
        // 🔴 BUG FIX: We must use pc.learn() during bootstrap so PC weights actually adapt to the embeddings
        pc.learn(&q1_emb, None).unwrap();
        // Flatten the[512, 1] tensor to a 1D tensor [512] for the decoder
        let q1_belief = pc.levels.last().unwrap().beliefs.flatten_all().unwrap();
        
        // 🔴 CRITICAL FIX: Reset PC state before processing a new unrelated prompt
        for level in pc.levels.iter_mut() {
            level.beliefs = level.beliefs.zeros_like().unwrap();
            level.prev_beliefs = level.prev_beliefs.zeros_like().unwrap();
        }
        
        // 🔴 BUG FIX: We must use pc.learn() during bootstrap so PC weights actually adapt to the embeddings
        pc.learn(&q2_emb, None).unwrap();
        let q2_belief = pc.levels.last().unwrap().beliefs.flatten_all().unwrap();

        let mut dataset = vec![(q1_belief, seq1), (q2_belief, seq2)];
        if !self.config.document_paths.is_empty() {
            for path in &self.config.document_paths {
                let query = format!("Summarize: {}", path);
                if let Ok(emb) = engine.process_text(&query).await {
                    // 🔴 CRITICAL FIX: Reset PC state before processing a new unrelated prompt
                    for level in pc.levels.iter_mut() {
                        level.beliefs = level.beliefs.zeros_like().unwrap();
                        level.prev_beliefs = level.prev_beliefs.zeros_like().unwrap();
                    }
                    
                    // 🔴 BUG FIX: We must use pc.learn() during bootstrap so PC weights actually adapt to the embeddings
                    pc.learn(&emb, None).ok();
                    if let Some(top) = pc.levels.last() {
                        let belief = top.beliefs.flatten_all().unwrap();
                        dataset.push((belief, vec![dict.op_to_id[&ThoughtOp::Explain], dict.op_to_id[&ThoughtOp::EOF]]));
                    }
                }
            }
        }
        dataset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pc_hierarchy::PCConfig;
    use crate::types::{CognitiveDictionary, DeviceType};
    use crate::pc_decoder::ThoughtDecoder;
    use candle_core::Device;

    #[tokio::test]
    async fn test_bootstrap_run_completes_and_trains_decoder() -> Result<(), Box<dyn std::error::Error>> {
        // This is a full integration test and may be slow.
        
        // 1. Setup all required components
        let ml_config = crate::config::MLConfig { embedding_dim: 32, ..Default::default() };
        let node_config = crate::config::NodeConfig { ml_config, ..Default::default() };

        // Ensure model file exists before running
        if !std::path::Path::new(&node_config.model_path).exists() {
            println!("Skipping bootstrap integration test: model file not found at {}", node_config.model_path);
            return Ok(());
        }

        let device = Device::Cpu;
        let ml_engine = Arc::new(RwLock::new(MLEngine::new(&node_config.model_path, DeviceType { name: "cpu".into(), ..Default::default() })?));
        let embedding_dim = ml_engine.read().await.embedding_dim();

        let pc_config = PCConfig::new(3, vec![embedding_dim, 128, 64]);
        let pc_hierarchy = Arc::new(RwLock::new(PredictiveCoding::new(pc_config)?));
        
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let top_belief_dim = *pc_hierarchy.read().await.config.dim_per_level.last().unwrap();
        let thought_decoder = Arc::new(RwLock::new(ThoughtDecoder::new(top_belief_dim, dict.read().await.len(), &device)?));

        // Clone initial decoder weights for comparison
        let decoder_weights_before = thought_decoder.read().await.w_vocab.as_tensor().to_vec2::<f32>()?;

        // 2. Setup BootstrapManager
        let bootstrap_config = crate::config::BootstrapConfig {
            max_epochs: 2, // Run only for a few epochs for a quick test
            ..Default::default()
        };
        let manager = BootstrapManager::new(
            ml_engine,
            thought_decoder.clone(),
            dict,
            pc_hierarchy,
            bootstrap_config,
        );

        // 3. Run the training
        let result = manager.run_synthetic_training().await;
        assert!(result.is_ok(), "Bootstrap training failed: {:?}", result.err());

        // 4. Verify that the decoder was actually trained
        let decoder_weights_after = thought_decoder.read().await.w_vocab.as_tensor().to_vec2::<f32>()?;
        assert_ne!(decoder_weights_before, decoder_weights_after, "ThoughtDecoder weights did not change after bootstrap training.");

        println!("Bootstrap integration test passed.");
        Ok(())
    }
}
