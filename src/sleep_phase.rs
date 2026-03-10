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

        // 1. Chunk Discovery on successful episodes
        let episodes: Vec<Episode> = memory.iter().cloned().collect();
        let new_chunks = self.dict.write().await.discover_chunks(&episodes);
        if new_chunks > 0 {
            tracing::info!("🧠 Discovered {} new thought chunks!", new_chunks);
            // Note: In production, expanding dict requires expanding the decoder w_vocab matrix.
        }

        // 2. Train Decoder on successful paths (Distillation)
        let mut decoder = self.decoder.write().await;
        let mut pc = self.pc_hierarchy.write().await;
        
        let mut total_loss = 0.0;
        let mut learned_count = 0;

        for ep in episodes.iter().filter(|e| e.success) {
            // Convert stored embedding vec back to Tensor
            let embed_tensor = Tensor::from_vec(
                ep.query_embedding.clone(), 
                (1, ep.query_embedding.len()), 
                &Device::Cpu
            )?;
            
            // Slow Memory Update: Update deep PC weights for highly novel successful queries
            if ep.novelty > 2.0 {
                let _ = pc.learn_legacy(&embed_tensor);
            }

            // Get stable belief for decoder training
            pc.infer(&embed_tensor, 5)?;
            let belief = pc.levels.last().unwrap().beliefs.flatten_all()?;
            
            if let Ok(loss) = decoder.train_step(&belief, &ep.thought_sequence, 0.01) {
                total_loss += loss;
                learned_count += 1;
            }
        }

        if learned_count > 0 {
            tracing::info!("📈 Sleep training complete. Avg Loss: {:.4}", total_loss / learned_count as f32);
        }

        // Clear short-term memory after consolidation
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
        let pc_config = PCConfig::new(2, vec![10, 5]);
        let pc = PredictiveCoding::new(pc_config).unwrap();
        let pc_hierarchy = Arc::new(RwLock::new(pc));
        
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(5, 8, &Device::Cpu).unwrap()
        ));
        
        let mut episodes = VecDeque::new();
        episodes.push_back(Episode {
            raw_query: "test".into(),
            query_embedding: vec![0.1, 0.2, 0.3],
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
        assert!(result.is_ok());
        
        // Check that memory is cleared
        let memory = episodic_memory.read().await;
        assert!(memory.is_empty(), "Episodic memory should be cleared after sleep");
    }
}