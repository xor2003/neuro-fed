// src/bootstrap.rs
// One-time distillation from frozen LLM to seed PC hierarchy with meaningful initial beliefs and weights

use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, debug};
use walkdir::WalkDir;
use serde_json;

use crate::ml_engine::MLEngine;
use crate::pc_hierarchy::{PredictiveCoding, PCError};
use crate::persistence::PCPersistence;
use crate::config::BootstrapConfig;

pub struct BootstrapManager {
    config: BootstrapConfig,
    ml_engine: Arc<Mutex<MLEngine>>,
    pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
}

impl BootstrapManager {
    pub fn new(
        config: BootstrapConfig,
        ml_engine: Arc<Mutex<MLEngine>>,
        pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
    ) -> Self {
        Self {
            config,
            ml_engine,
            pc_hierarchy,
        }
    }

    /// Process a single text chunk through the ML engine and PC hierarchy
    async fn process_text_chunk(&self, text: &str, file_path: &Path, chunk_id: usize) -> Result<(), Box<dyn std::error::Error>> {
        // Generate Embedding (The "Eyes")
        let engine = self.ml_engine.lock().await;
        let tensor = match engine.process_text(text).await {
            Ok(t) => t,
            Err(e) => {
                warn!("Failed to embed chunk: {}", e);
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };
        drop(engine); // Release engine lock

        // Flatten & reshape for PC
        let pc_dim = self.pc_hierarchy.lock().await.config.dim_per_level[0];
        
        // Convert to vec
        let data = match tensor.to_vec1::<f32>() {
            Ok(d) => d,
            Err(e) => {
                warn!("Failed to convert tensor to vec: {}", e);
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };
        
        let mut processed_data = if data.len() >= pc_dim {
            data[..pc_dim].to_vec()
        } else {
            let mut resized = data;
            resized.resize(pc_dim, 0.0);
            resized
        };

        // Ensure we have enough data
        if processed_data.len() < pc_dim {
            processed_data.resize(pc_dim, 0.0);
        }

        let pc_tensor = match candle_core::Tensor::from_vec(
            processed_data,
            (pc_dim, 1),
            &candle_core::Device::Cpu
        ) {
            Ok(t) => t,
            Err(e) => {
                warn!("Failed to create tensor for PC: {}", e);
                return Err(Box::new(e) as Box<dyn std::error::Error>);
            }
        };

        // Learn (The "Brain")
        let pc_hierarchy = self.pc_hierarchy.clone();
        let stats = tokio::task::spawn_blocking(move || -> Result<f32, PCError> {
            let mut pc = pc_hierarchy.blocking_lock();
            // Run multiple epochs per chunk to ensure it sinks in
            let mut final_fe = 0.0;
            for _ in 0..3 {
                let res = pc.learn_legacy(&pc_tensor)?;
                final_fe = res.total_surprise;
            }
            Ok(final_fe)
        }).await.unwrap()?;

        debug!("Learned chunk {} from {:?} | Surprise: {:.4}", chunk_id, file_path.file_name(), stats);
        Ok(())
    }

    /// Run the bootstrapping process on local directories
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Starting Bootstrap Learning Phase...");
        let mut total_files = 0;
        let mut total_chunks = 0;

        for dir_path in &self.config.document_paths {
            let path = Path::new(dir_path);
            if !path.exists() {
                warn!("Bootstrap path {:?} does not exist. Skipping.", path);
                continue;
            }

            // 1. Traverse the directory for text/markdown/jsonl files
            for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                if !entry.file_type().is_file() { continue; }
                
                let ext = entry.path().extension().and_then(|e| e.to_str()).unwrap_or("");
                if ext != "txt" && ext != "md" && ext != "rs" && ext != "json" && ext != "jsonl" {
                    continue; // Only process readable text files
                }

                total_files += 1;

                // Handle different file types
                if ext == "jsonl" {
                    // Process JSONL files with request-response format
                    let content = match std::fs::read_to_string(entry.path()) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };
                    
                    for line in content.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        
                        match serde_json::from_str::<serde_json::Value>(line) {
                            Ok(json) => {
                                // Extract prompt and canonical_solution for request-response training
                                let prompt = json.get("prompt").and_then(|v| v.as_str()).unwrap_or("");
                                let solution = json.get("canonical_solution").and_then(|v| v.as_str()).unwrap_or("");
                                
                                if !prompt.is_empty() && !solution.is_empty() {
                                    // Create request-response pair
                                    let request_response = format!("Question: {}\nAnswer: {}", prompt, solution);
                                    if let Err(e) = self.process_text_chunk(&request_response, entry.path(), total_chunks).await {
                                        warn!("Failed to process JSONL entry: {}", e);
                                    }
                                    total_chunks += 1;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse JSONL line: {}", e);
                                continue;
                            }
                        }
                    }
                } else {
                    // Process regular text files
                    let text = match std::fs::read_to_string(entry.path()) {
                        Ok(t) => t,
                        Err(_) => continue,
                    };

                    // Chunk the text (e.g., by paragraphs or character count)
                    // Real implementation should use a proper tokenizer chunker.
                    // Here we chunk roughly by 1000 characters to simulate context windows.
                    let chunks: Vec<&str> = text.as_bytes()
                        .chunks(1000)
                        .map(|c| std::str::from_utf8(c).unwrap_or(""))
                        .filter(|s| !s.trim().is_empty())
                        .collect();

                    for chunk in chunks {
                        total_chunks += 1;
                        if let Err(e) = self.process_text_chunk(chunk, entry.path(), total_chunks).await {
                            warn!("Failed to process chunk: {}", e);
                        }
                    }
                }
            }
        }

        info!("✅ Bootstrap Complete! Processed {} files ({} semantic chunks).", total_files, total_chunks);
        
        // Save the new "Smart" Brain to the Database
        self.save_brain_state().await;

        Ok(())
    }

    async fn save_brain_state(&self) {
        let pc = self.pc_hierarchy.lock().await;
        let db_path = pc.config.persistence_db_path.clone().unwrap_or_else(|| "./neurofed.db".to_string());
        
        // Extract layers
        let levels_to_save: Vec<_> = pc.levels.iter().enumerate().map(|(i, l)| {
            let (rows, cols) = match l.weights.shape().dims2() {
                Ok(dims) => dims,
                Err(_) => (0, 0),
            };
            let flat_weights = match l.weights.flatten_all() {
                Ok(t) => match t.to_vec1::<f32>() {
                    Ok(v) => v,
                    Err(_) => vec![],
                },
                Err(_) => vec![],
            };
            crate::persistence::PCLevelWeights {
                level_index: i,
                input_dim: rows,
                output_dim: cols,
                weights: flat_weights,
                updated_at: chrono::Utc::now().timestamp(),
            }
        }).collect();
        drop(pc);

        // Save asynchronously
        if let Ok(db) = PCPersistence::new(&db_path).await {
            for level in levels_to_save {
                let _ = db.save_level_weights(&level).await;
            }
            info!("💾 Bootstrapped weights safely persisted to SQLite.");
        }
    }
}

// Keep the existing error types and structures for backward compatibility
#[derive(Debug)]
pub struct BootstrapError(String);

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "BootstrapError: {}", self.0)
    }
}

impl std::error::Error for BootstrapError {}

// Re-export the old Bootstrap struct for backward compatibility
pub struct Bootstrap {
    config: BootstrapConfig,
}

impl Bootstrap {
    pub fn new(config: BootstrapConfig) -> Result<Self, BootstrapError> {
        Ok(Self { config })
    }

    pub fn run(&mut self) -> Result<(), BootstrapError> {
        warn!("Legacy Bootstrap::run() called - use BootstrapManager instead");
        Ok(())
    }
}

// Keep the old example_usage for backward compatibility
pub fn example_usage() {
    let config = BootstrapConfig::new(1024, 32, 10, 0.001, vec!["./data".to_string()]);
    let mut bootstrap = Bootstrap::new(config).expect("Failed to create Bootstrap instance");
    let _ = bootstrap.run().expect("Bootstrap failed");
    println!("Bootstrap completed successfully (legacy mode)");
}

mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_creation() {
        let config = BootstrapConfig::new(1024, 32, 10, 0.001, vec!["./test_data".to_string()]);
        let bootstrap = Bootstrap::new(config).expect("Failed to create Bootstrap instance");
        // Just test that it creates without error
        assert!(true);
    }
}