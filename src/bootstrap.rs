use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;
use candle_core::Tensor;
use crate::ml_engine::MLEngine;
use crate::pc_decoder::ThoughtDecoder;
use crate::pc_hierarchy::PredictiveCoding;
use crate::types::{CognitiveDictionary, ThoughtOp};

pub struct BootstrapManager {
    ml_engine: Arc<Mutex<MLEngine>>,
    thought_decoder: Arc<Mutex<ThoughtDecoder>>,
    dict: Arc<Mutex<CognitiveDictionary>>,
    pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
}

impl BootstrapManager {
    pub fn new(
        ml_engine: Arc<Mutex<MLEngine>>,
        thought_decoder: Arc<Mutex<ThoughtDecoder>>,
        dict: Arc<Mutex<CognitiveDictionary>>,
        pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
    ) -> Self {
        Self { ml_engine, thought_decoder, dict, pc_hierarchy }
    }

    pub async fn run_synthetic_training(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Запускаем Curriculum Learning для Декодера Мыслей...");
        let mut synthetic_data = self.generate_synthetic_dataset().await;
        
        synthetic_data.sort_by_key(|(_, seq)| seq.len());
        
        let mut decoder = self.thought_decoder.lock().await;

        for epoch in 0..100 { 
            let mut total_loss = 0.0;
            for (belief, seq) in &synthetic_data {
                let loss = decoder.train_step(belief, seq, 0.01)?;
                total_loss += loss;
            }
            if epoch % 10 == 0 {
                 info!("Эпоха {}: Loss = {:.4}", epoch, total_loss / synthetic_data.len() as f32);
            }
        }
        
        info!("✅ Синтетическое обучение декодера завершено.");
        Ok(())
    }

    async fn generate_synthetic_dataset(&self) -> Vec<(Tensor, Vec<u32>)> {
        let engine = self.ml_engine.lock().await;
        let dict = self.dict.lock().await;

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
        let mut pc = self.pc_hierarchy.lock().await;
        
        pc.infer(&q1_emb, 15).unwrap();
        // Flatten the[512, 1] tensor to a 1D tensor [512] for the decoder
        let q1_belief = pc.levels.last().unwrap().beliefs.flatten_all().unwrap();
        
        pc.infer(&q2_emb, 15).unwrap();
        let q2_belief = pc.levels.last().unwrap().beliefs.flatten_all().unwrap();

        vec![(q1_belief, seq1), (q2_belief, seq2)]
    }
}
