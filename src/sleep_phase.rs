// src/sleep_phase.rs
// Offline Consolidation: Fast-to-Slow memory transfer and Chunk Discovery

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::pc_hierarchy::PredictiveCoding;
use crate::pc_decoder::ThoughtDecoder;
use crate::types::{CognitiveDictionary, Episode};
use candle_core::{Tensor, Device};

pub struct SleepManager {
    pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
    decoder: Arc<RwLock<ThoughtDecoder>>,
    dict: Arc<RwLock<CognitiveDictionary>>,
    episodic_memory: Arc<RwLock<std::collections::VecDeque<Episode>>>,
}

impl SleepManager {
    pub fn new(
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        decoder: Arc<RwLock<ThoughtDecoder>>,
        dict: Arc<RwLock<CognitiveDictionary>>,
        episodic_memory: Arc<RwLock<std::collections::VecDeque<Episode>>>,
    ) -> Self {
        Self { pc_hierarchy, decoder, dict, episodic_memory }
    }

    /// Triggers offline consolidation
    pub async fn process_sleep_cycle(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("🌙 Entering SLEEP PHASE: Consolidating memory...");
        
        let mut memory = self.episodic_memory.write().await;
        if memory.is_empty() {
            tracing::info!("No new episodes to consolidate.");
            return Ok(());
        }

        let episodes: Vec<Episode> = memory.iter().cloned().collect();
        
        // 1. Chunk Discovery on successful episodes
        let mut dict = self.dict.write().await;
        let new_chunks = dict.discover_chunks(&episodes);
        let new_dict_len = dict.len();
        drop(dict);

        // 1.5. Expand Decoder Matrix if new thoughts were chunked
        if new_chunks > 0 {
            tracing::info!("🧠 Discovered {} new thought chunks! Expanding Decoder Matrix...", new_chunks);
            let mut decoder = self.decoder.write().await;
            decoder.resize_vocab(new_dict_len).map_err(|e| e.to_string())?;
        }

        // 2. Safe Slow Memory Update over Sequences
        let mut decoder = self.decoder.write().await;
        let mut pc = self.pc_hierarchy.write().await;
        let mut total_loss = 0.0;
        let mut learned_count = 0;

        for ep in episodes.iter().filter(|e| e.success) {
            if ep.query_sequence.is_empty() { continue; }
            let seq_len = ep.query_sequence.len();
            let dim = ep.query_sequence[0].len();
            if dim == 0 { continue; }
            
            let flat_data: Vec<f32> = ep.query_sequence.iter().flatten().copied().collect();
            let embed_tensor = Tensor::from_vec(flat_data, (seq_len, dim), &Device::Cpu)?;
            
            pc.reset_state().map_err(|e| e.to_string())?;

            // Use learn_sequence for proper temporal processing
            if ep.novelty > 2.0 {
                pc.checkpoint_weights().map_err(|e| e.to_string())?;
                
                let stats = pc.learn_sequence(&embed_tensor, None).map_err(|e| e.to_string())?;
                
                // Rollback if learning caused divergence
                if stats.total_surprise.is_nan() || stats.total_surprise > 1000.0 {
                    pc.rollback_weights().map_err(|e| e.to_string())?;
                }
            } else {
                pc.infer_sequence(&embed_tensor, 5).map_err(|e| e.to_string())?;
            }

            let belief = pc.levels.last().unwrap().beliefs.flatten_all()?;
            
            if let Ok(loss) = decoder.train_step(&belief, &ep.thought_sequence, 0.01) {
                total_loss += loss;
                learned_count += 1;
            }
        }

        if learned_count > 0 {
            tracing::info!("📈 Sleep training complete. Avg Loss: {:.4}", total_loss / learned_count as f32);
        }

        memory.clear();
        tracing::info!("☀️ Waking up. Episodic memory cleared.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pc_hierarchy::PCConfig;
    use crate::pc_decoder::ThoughtDecoder;
    use crate::types::{CognitiveDictionary, Episode, ThoughtOp};
    use std::collections::VecDeque;

    #[tokio::test]
    async fn test_sleep_manager_clears_memory() {
        let pc_config = PCConfig::new(2, vec![3, 2]); // Match embedding dimension
        let pc = PredictiveCoding::new(pc_config).unwrap();
        let pc_hierarchy = Arc::new(RwLock::new(pc));
        
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(2, 8, &Device::Cpu).unwrap() // belief_dim = top level dim (2)
        ));
        
        let mut episodes = VecDeque::new();
        episodes.push_back(Episode {
            raw_query: "test".into(),
            query_sequence: vec![vec![0.1, 0.2, 0.3]], // dim=3 matches PC input dim
            novelty: 3.0,
            confidence: 0.9,
            generated_code: "".into(),
            thought_sequence: vec![0, 1, 2],
            success: true,
        });
        
        let episodic_memory = Arc::new(RwLock::new(episodes));
        
        let sleep_mgr = SleepManager::new(
            pc_hierarchy,
            decoder,
            dict,
            episodic_memory.clone(),
        );
        
        // Process sleep cycle
        let result = sleep_mgr.process_sleep_cycle().await;
        assert!(result.is_ok(), "Sleep phase failed: {:?}", result.err());
        
        // Check that memory is cleared
        let memory = episodic_memory.read().await;
        assert!(memory.is_empty(), "Episodic memory should be cleared after sleep");
    }
}

#[cfg(test)]
mod sleep_phase_integration_tests {
    use super::*;
    use crate::pc_hierarchy::PCConfig;
    use crate::pc_decoder::ThoughtDecoder;
    use crate::types::{CognitiveDictionary, Episode};
    use std::collections::VecDeque;
    use candle_core::Device;

    #[tokio::test]
    async fn test_sleep_phase_processes_sequences_and_resizes_vocab() {
        let pc_config = PCConfig::new(2, vec![4, 2]);
        let pc = PredictiveCoding::new(pc_config).unwrap();
        let pc_hierarchy = Arc::new(RwLock::new(pc));
        
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        // Original dict has 8 elements
        let decoder = Arc::new(RwLock::new(ThoughtDecoder::new(2, 8, &Device::Cpu).unwrap()));
        
        let mut episodes = VecDeque::new();
        
        // Insert 4 identical successful episodes.
        // Because there are >3 identical bigrams (0->1), the chunk discoverer will create a new token.
        for _ in 0..4 {
            episodes.push_back(Episode {
                raw_query: "Solve X".into(),
                // Simulate a sequence of 3 tokens, each embedding size 4
                query_sequence: vec![
                    vec![0.1, 0.2, 0.3, 0.4],
                    vec![0.5, 0.6, 0.7, 0.8],
                    vec![0.9, 1.0, 1.1, 1.2]
                ],
                novelty: 3.0,
                confidence: 0.9,
                generated_code: "print('X')".into(),
                thought_sequence: vec![0, 1, 7], // Define -> Iterate -> EOF
                success: true,
            });
        }
        
        let episodic_memory = Arc::new(RwLock::new(episodes));
        
        let manager = SleepManager::new(
            pc_hierarchy,
            decoder.clone(),
            dict.clone(),
            episodic_memory.clone()
        );
        
        // Trigger consolidation
        let result = manager.process_sleep_cycle().await;
        assert!(result.is_ok(), "Sleep phase crashed: {:?}", result.err());
        
        // 1. Ensure memory was wiped clean
        assert!(episodic_memory.read().await.is_empty(), "Memory should be cleared after sleep phase");
        
        // 2. Ensure Dictionary Grew
        let new_dict_len = dict.read().await.len();
        assert!(new_dict_len > 8, "Dictionary should have discovered the 0->1 bigram and grown");
        
        // 3. Ensure Decoder w_vocab Resized to match Dictionary
        let current_vocab_size = decoder.read().await.w_vocab.shape().dims()[0];
        assert_eq!(current_vocab_size, new_dict_len,
            "Decoder w_vocab rows ({}) must match Dictionary length ({})!", current_vocab_size, new_dict_len);
    }
}
