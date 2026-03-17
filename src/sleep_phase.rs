// src/sleep_phase.rs
// Offline Consolidation: Fast-to-Slow memory transfer and Chunk Discovery

use crate::pc_hierarchy::PredictiveCoding;
use crate::types::{CognitiveDictionary, Episode};
use crate::reasoning_state::{state_error, text_error};
use crate::{learning_log::append_learning_detail, pc_decoder::ThoughtDecoder};
use candle_core::{Device, Tensor};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task;

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
        Self {
            pc_hierarchy,
            decoder,
            dict,
            episodic_memory,
        }
    }

    /// Triggers offline consolidation
    pub async fn process_sleep_cycle(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("🌙 Entering SLEEP PHASE: Consolidating memory...");

        let episodes = {
            let mut memory = self.episodic_memory.write().await;
            if memory.is_empty() {
                tracing::info!("No new episodes to consolidate.");
                return Ok(());
            }
            memory.drain(..).collect::<Vec<Episode>>()
        };

        // 1. Chunk Discovery on successful episodes
        let (new_chunks, new_dict_len) = {
            let mut dict = self.dict.write().await;
            let chunks = dict.discover_chunks(&episodes);
            let len = dict.len();
            (chunks, len)
        };

        if new_chunks > 0 {
            tracing::info!(
                "🧠 Discovered {} new thought chunks! Expanding Decoder Matrix...",
                new_chunks
            );
            let mut decoder = self.decoder.write().await;
            decoder
                .resize_vocab(new_dict_len)
                .map_err(|e| e.to_string())?;
            drop(decoder);
        }

        let mut total_loss = 0.0;
        let mut learned_count = 0;
        let mut detail_logs = Vec::new();

        for ep in episodes.into_iter().filter(|e| e.success) {
            if ep.query_sequence.is_empty() {
                continue;
            }
            let seq_len = ep.query_sequence.len();
            let dim = ep.query_sequence[0].len();
            if dim == 0 {
                continue;
            }

            let flat_data: Vec<f32> = ep.query_sequence.iter().flatten().copied().collect();
            let embed_tensor = Tensor::from_vec(flat_data, (seq_len, dim), &Device::Cpu)?;

            let mut pc = self.pc_hierarchy.write().await;
            let mut decoder = self.decoder.write().await;

            pc.reset_state().map_err(|e| e.to_string())?;

            if ep.novelty > 2.0 {
                pc.checkpoint_weights().map_err(|e| e.to_string())?;
                let stats = pc
                    .learn_sequence(&embed_tensor, None)
                    .map_err(|e| e.to_string())?;
                if stats.total_surprise.is_nan() || stats.total_surprise > 1000.0 {
                    pc.rollback_weights().map_err(|e| e.to_string())?;
                }
            } else {
                pc.infer_sequence(&embed_tensor, 5)
                    .map_err(|e| e.to_string())?;
            }

            let belief = pc.levels.last().unwrap().beliefs.flatten_all()?;

            let mut state_loss = 0.0f32;
            let mut text_loss = 0.0f32;
            let mut state_debug: Option<String> = None;

            if let Some(task) = &ep.reasoning_task {
                let dict_guard = self.dict.read().await;
                let ops: Vec<_> = ep
                    .thought_sequence
                    .iter()
                    .map(|id| dict_guard.get_op(*id))
                    .collect();
                let (err, outcome) = state_error(task, &ops);
                state_loss = err;
                if !outcome.errors.is_empty() {
                    state_debug = Some(outcome.errors.join(" | "));
                }
            }

            if let Some(expected) = &ep.expected_output {
                text_loss = text_error(expected, &ep.generated_code);
            }

            let loss_scale = (1.0 + state_loss + text_loss).clamp(0.5, 3.0);
            let adjusted_lr = 0.05 * loss_scale as f64;

            if let Ok(reasoning_loss) =
                decoder.train_step(&belief, &ep.thought_sequence, adjusted_lr)
            {
                let combined_loss = reasoning_loss + state_loss + text_loss;
                total_loss += combined_loss;
                learned_count += 1;
                let trajectory = {
                    let dict = self.dict.read().await;
                    ep.thought_sequence
                        .iter()
                        .map(|&id| dict.get_op(id).to_string())
                        .collect::<Vec<_>>()
                        .join(" -> ")
                };
                detail_logs.push(format!(
                    "Input Question: {}\nAnswer: {}\nTrajectory: {}\nThought sequence: {:?}\nConfidence: {}\nNovelty: {}\nReasoning loss: {:.4}\nState loss: {:.2}\nText loss: {:.2}\nCombined loss: {:.4}{}\nLearning rate: {:.4}",
                    ep.raw_query,
                    ep.generated_code,
                    trajectory,
                    ep.thought_sequence,
                    ep.confidence,
                    ep.novelty,
                    reasoning_loss,
                    state_loss,
                    text_loss,
                    combined_loss,
                    state_debug
                        .as_ref()
                        .map(|s| format!("\nState errors: {}", s))
                        .unwrap_or_default(),
                    adjusted_lr
                ));
            }

            drop(decoder);
            drop(pc);
            task::yield_now().await;
        }

        if !detail_logs.is_empty() {
            detail_logs.push(format!(
                "Sleep Summary: episodes={}, learned_chunks={}, total_loss={:.4}",
                learned_count, learned_count, total_loss
            ));
            let log_body = detail_logs.join("\n\n---\n\n");
            append_learning_detail(&log_body);
        }

        if learned_count > 0 {
            tracing::info!(
                "📈 Sleep training complete. Avg Loss: {:.4}",
                total_loss / learned_count as f32
            );
        }

        self.episodic_memory.write().await.clear();
        tracing::info!("☀️ Waking up. Episodic memory cleared.");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pc_decoder::ThoughtDecoder;
    use crate::pc_hierarchy::PCConfig;
    use crate::types::{CognitiveDictionary, Episode};
    use std::collections::VecDeque;

    #[tokio::test]
    async fn test_sleep_manager_clears_memory() {
        let pc_config = PCConfig::new(2, vec![3, 2]); // Match embedding dimension
        let pc = PredictiveCoding::new(pc_config).unwrap();
        let pc_hierarchy = Arc::new(RwLock::new(pc));

        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(2, 8, &Device::Cpu).unwrap(), // belief_dim = top level dim (2)
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
            reasoning_task: None,
            expected_output: None,
        });

        let episodic_memory = Arc::new(RwLock::new(episodes));

        let sleep_mgr = SleepManager::new(pc_hierarchy, decoder, dict, episodic_memory.clone());

        // Process sleep cycle
        let result = sleep_mgr.process_sleep_cycle().await;
        assert!(result.is_ok(), "Sleep phase failed: {:?}", result.err());

        // Check that memory is cleared
        let memory = episodic_memory.read().await;
        assert!(
            memory.is_empty(),
            "Episodic memory should be cleared after sleep"
        );
    }
}

#[cfg(test)]
mod sleep_phase_integration_tests {
    use super::*;
    use crate::pc_decoder::ThoughtDecoder;
    use crate::pc_hierarchy::PCConfig;
    use crate::types::{CognitiveDictionary, Episode, ThoughtOp};
    use candle_core::Device;
    use std::collections::VecDeque;

    #[tokio::test]
    async fn test_sleep_phase_processes_sequences_and_resizes_vocab() {
        let pc_config = PCConfig::new(2, vec![4, 2]);
        let pc = PredictiveCoding::new(pc_config).unwrap();
        let pc_hierarchy = Arc::new(RwLock::new(pc));

        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        // Decoder vocab can be smaller for tests; it will resize on new chunks.
        let decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(2, 8, &Device::Cpu).unwrap(),
        ));

        let mut episodes = VecDeque::new();
        let eof_id = { dict.read().await.op_to_id[&ThoughtOp::EOF] };

        // Insert 4 identical successful episodes.
        // Because there are >3 identical bigrams (0->1), the chunk discoverer will create a new token.
        for _ in 0..4 {
            episodes.push_back(Episode {
                raw_query: "Solve X".into(),
                // Simulate a sequence of 3 tokens, each embedding size 4
                query_sequence: vec![
                    vec![0.1, 0.2, 0.3, 0.4],
                    vec![0.5, 0.6, 0.7, 0.8],
                    vec![0.9, 1.0, 1.1, 1.2],
                ],
                novelty: 3.0,
                confidence: 0.9,
                generated_code: "print('X')".into(),
                thought_sequence: vec![0, 1, eof_id], // Define -> Iterate -> EOF
                success: true,
                reasoning_task: None,
                expected_output: None,
            });
        }

        let episodic_memory = Arc::new(RwLock::new(episodes));

        let manager = SleepManager::new(
            pc_hierarchy,
            decoder.clone(),
            dict.clone(),
            episodic_memory.clone(),
        );

        // Trigger consolidation
        let result = manager.process_sleep_cycle().await;
        assert!(result.is_ok(), "Sleep phase crashed: {:?}", result.err());

        // 1. Ensure memory was wiped clean
        assert!(
            episodic_memory.read().await.is_empty(),
            "Memory should be cleared after sleep phase"
        );

        // 2. Ensure Dictionary Grew
        let new_dict_len = dict.read().await.len();
        assert!(
            new_dict_len > 8,
            "Dictionary should have discovered the 0->1 bigram and grown"
        );

        // 3. Ensure Decoder w_vocab Resized to match Dictionary
        let current_vocab_size = decoder.read().await.w_vocab.shape().dims()[0];
        assert_eq!(
            current_vocab_size, new_dict_len,
            "Decoder w_vocab rows ({}) must match Dictionary length ({})!",
            current_vocab_size, new_dict_len
        );
    }
}

#[cfg(test)]
mod deep_consolidation_tests {
    use super::*;
    use crate::pc_hierarchy::PCConfig;
    use crate::types::ThoughtOp;
    use candle_core::Device;

    #[tokio::test]
    async fn test_full_sleep_cycle_dimension_stability() {
        // PROVES: When chunk discovery creates a new token, the Decoder matrix resizes
        // safely and backpropagation doesn't crash due to dimension mismatches.
        let pc_config = PCConfig::new(2, vec![4, 2]);
        let pc_hierarchy = Arc::new(RwLock::new(PredictiveCoding::new(pc_config).unwrap()));
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        // Base dictionary has 12 items (new commands included).
        let decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(2, 8, &Device::Cpu).unwrap(),
        ));
        let episodic_memory = Arc::new(RwLock::new(std::collections::VecDeque::new()));

        let sleep_mgr = SleepManager::new(
            pc_hierarchy,
            decoder.clone(),
            dict.clone(),
            episodic_memory.clone(),
        );

        let base_dict_len = { dict.read().await.len() };
        let define_id = { dict.read().await.op_to_id[&ThoughtOp::Define] };
        let iterate_id = { dict.read().await.op_to_id[&ThoughtOp::Iterate] };
        let eof_id = { dict.read().await.op_to_id[&ThoughtOp::EOF] };

        // Push 5 successful identical episodes to force Chunk Discovery of [0, 1]
        for _ in 0..5 {
            episodic_memory.write().await.push_back(Episode {
                raw_query: "Test".into(),
                query_sequence: vec![vec![0.1, 0.2, 0.3, 0.4], vec![0.5, 0.6, 0.7, 0.8]],
                novelty: 5.0,
                confidence: 0.9,
                generated_code: "pass".into(),
                thought_sequence: vec![define_id, iterate_id, eof_id],
                // Will chunk Define -> Iterate (EOF ensures termination)
                success: true,
                reasoning_task: None,
                expected_output: None,
            });
        }

        // Run sleep cycle
        let result = sleep_mgr.process_sleep_cycle().await;
        assert!(result.is_ok(), "Sleep cycle crashed: {:?}", result.err());

        // 1. Verify Dictionary expanded
        let new_dict_size = dict.read().await.len();
        assert!(
            new_dict_size > base_dict_len,
            "Dictionary failed to discover new chunk!"
        );

        // 2. Verify Neural Network resized WITHOUT losing old data
        let w_vocab_shape = decoder.read().await.w_vocab.shape().clone();
        assert_eq!(
            w_vocab_shape.dims()[0],
            new_dict_size,
            "Decoder matrix did not resize to fit new vocabulary!"
        );

        // 3. Verify it can STILL run a forward/backward pass with the new dimensions
        let test_belief = Tensor::randn(0f32, 1.0, (2, 1), &Device::Cpu).unwrap();
        // Train it on the newly discovered token ID (8)
        let new_token_id = (new_dict_size - 1) as u32;
        let train_res = decoder
            .write()
            .await
            .train_step(&test_belief, &[new_token_id], 0.01);
        assert!(
            train_res.is_ok(),
            "BPTT Panicked after matrix resize! Dimension mismatch likely."
        );
    }
}
