use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use candle_core::Tensor;
use sha2::{Digest, Sha256};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use crate::ml_engine::MLEngine;
use crate::pc_decoder::ThoughtDecoder;
use crate::pc_hierarchy::PredictiveCoding;
use crate::persistence::PCPersistence;
use crate::types::{CognitiveDictionary, ThoughtOp};

// NEW: For Parquet support
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::RowAccessor;


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

    pub async fn run_full_bootstrap(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.study_documents().await?;
        self.run_synthetic_training().await?;
        Ok(())
    }

    pub async fn study_documents(&self) -> Result<(), Box<dyn std::error::Error>> {
        if self.config.document_paths.is_empty() {
            info!("📚 No document paths configured for study. Skipping.");
            return Ok(());
        }

        let mut all_files = Vec::new();
        for path_str in &self.config.document_paths {
            let path = Path::new(path_str);
            if path.is_dir() {
                for entry in std::fs::read_dir(path)? {
                    let entry = entry?;
                    let file_path = entry.path();
                    if file_path.is_file() { all_files.push(file_path); }
                }
            } else if path.is_file() {
                all_files.push(path.to_path_buf());
            }
        }

        let bar = ProgressBar::new(all_files.len() as u64);
        bar.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("#>-"));

        let engine = self.ml_engine.read().await;
        let mut pc = self.pc_hierarchy.write().await;

        for file_path in all_files {
            let file_name = file_path.file_name().unwrap_or_default().to_string_lossy();
            bar.set_message(format!("Studying: {}", file_name));
            tracing::debug!("📖 Processing file: {}", file_name);

            let ext = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
            
            // Handle different file types
            let text_chunks: Vec<String> = match ext {
                "parquet" => extract_rows_from_parquet(&file_path),
                "pdf" => extract_text_from_pdf(&file_path).map(|s| vec![s]).unwrap_or_default(),
                "epub" => extract_text_from_epub(&file_path).map(|s| vec![s]).unwrap_or_default(),
                "txt" | "md" | "jsonl" => {
                    std::fs::read_to_string(&file_path)
                        .map(|s| vec![s])
                        .unwrap_or_default()
                }
                _ => {
                    warn!("Unsupported file type: {}. Skipping.", ext);
                    vec![]
                }
            };

            tracing::debug!("📖 Extracted {} text chunks from {}", text_chunks.len(), file_name);
            
            for (idx, raw_text) in text_chunks.iter().enumerate() {
                let paragraphs = chunk_text(&raw_text);
                tracing::debug!("📖 Chunk {} has {} paragraphs", idx, paragraphs.len());
                
                for (paragraph_idx, chunk) in paragraphs.iter().enumerate() {
                    if chunk.len() > 50 {
                        let preview = if chunk.len() > 100 {
                            let end = 100.min(chunk.len());
                            format!("{}...", &chunk[..end])
                        } else {
                            chunk.clone()
                        };
                        tracing::debug!("📖 Learning paragraph {} ({} chars): {}",
                            paragraph_idx, chunk.len(), preview);
                        
                        let sequence_tensor = engine.process_text_sequence(&chunk).await?;
                        
                        // Clear temporal state between chunks to prevent "thought bleeding"
                        for level in pc.levels.iter_mut() {
                            level.beliefs = level.beliefs.zeros_like()?;
                            level.prev_beliefs = level.prev_beliefs.zeros_like()?;
                        }
                        
                        if let Ok(stats) = pc.learn_sequence(&sequence_tensor, None) {
                            // Calculate study efficiency: How much Level 0 noise was converted into higher-level concepts
                            let n_levels = stats.level_surprises.len();
                            if n_levels >= 2 {
                                let top_l = n_levels.saturating_sub(2);
                                let abstraction_ratio = 1.0 - (stats.level_surprises[top_l] / stats.level_surprises[0].max(1.0));
                                
                                // Efficiency: How much Level 0 noise was converted into Level 2 concepts
                                if abstraction_ratio < 0.2 {
                                    warn!("📉 Low Study Efficiency ({:.1}%). Ideas are too complex for {} layers.", abstraction_ratio * 100.0, n_levels);
                                } else {
                                    tracing::debug!("📊 Study Efficiency: {:.1}% (abstraction ratio)", abstraction_ratio * 100.0);
                                }
                            }
                        }
                    }
                }
            }
            bar.inc(1);
        }

        bar.finish_with_message("✅ Study session complete.");
        Ok(())
    }

    pub async fn run_synthetic_training(&self) -> Result<(), Box<dyn std::error::Error>> {
        info!("🚀 Training Thought Decoder...");
        let synthetic_data = self.generate_synthetic_decoder_dataset().await?;
        let mut decoder = self.thought_decoder.write().await;

        let max_epochs = self.config.max_epochs.max(100);
        let lr = self.config.learning_rate.max(0.01) as f64;
        
        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;
            for (belief, seq) in &synthetic_data {
                let loss = decoder.train_step(belief, seq, lr)?;
                total_loss += loss;
            }
            if epoch % 20 == 0 {
                 info!("Epoch {}: Loss = {:.4}", epoch, total_loss / synthetic_data.len() as f32);
            }
        }
        Ok(())
    }

    async fn generate_synthetic_decoder_dataset(&self) -> Result<Vec<(Tensor, Vec<u32>)>, Box<dyn std::error::Error>> {
        let engine = self.ml_engine.read().await;
        let dict = self.dict.read().await;
        let mut pc = self.pc_hierarchy.write().await;
        let mut dataset = Vec::new();

        let scenarios = vec![
            ("Solve 2x = 10", vec![dict.op_to_id[&ThoughtOp::Define], dict.op_to_id[&ThoughtOp::Compute], dict.op_to_id[&ThoughtOp::EOF]]),
        ];

        for (query, seq) in scenarios {
            let emb = engine.process_text(query).await?;
            pc.learn(&emb, None)?;
            let belief = pc.levels.last().unwrap().beliefs.flatten_all()?;
            dataset.push((belief, seq));
        }
        Ok(dataset)
    }
    
    /// Parses a single file, computes its hash, and returns its text chunks if it needs studying.
    pub async fn process_and_check_file(&self, file_path: &Path, persistence: &PCPersistence) -> Result<Option<Vec<String>>, Box<dyn std::error::Error + Send + Sync>> {
        let ext = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let content_bytes = std::fs::read(file_path)?;
        
        let mut hasher = Sha256::new();
        hasher.update(&content_bytes);
        let content_hash = format!("{:x}", hasher.finalize());
        
        if persistence.is_document_studied(&file_path.to_string_lossy(), &content_hash).await? {
            tracing::debug!("Skipping already studied file: {:?}", file_path);
            return Ok(None);
        }
        
        let text_chunks: Vec<String> = match ext {
            "parquet" => {
                // For parquet files, we need to write to a temp file first
                use tempfile::NamedTempFile;
                match NamedTempFile::new() {
                    Ok(temp_file) => {
                        if let Err(e) = std::fs::write(temp_file.path(), &content_bytes) {
                            error!("Failed to write parquet to temp file: {}", e);
                            vec![]
                        } else {
                            extract_rows_from_parquet(temp_file.path())
                        }
                    }
                    Err(e) => {
                        error!("Failed to create temp file for parquet: {}", e);
                        vec![]
                    }
                }
            }
            "pdf" => {
                // For PDF files, write to temp file
                use tempfile::NamedTempFile;
                match NamedTempFile::new() {
                    Ok(temp_file) => {
                        if let Err(e) = std::fs::write(temp_file.path(), &content_bytes) {
                            error!("Failed to write PDF to temp file: {}", e);
                            vec![]
                        } else {
                            match extract_text_from_pdf(temp_file.path()) {
                                Ok(text) => vec![text],
                                Err(e) => {
                                    error!("Failed to extract text from PDF: {}", e);
                                    vec![]
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Failed to create temp file for PDF: {}", e);
                        vec![]
                    }
                }
            }
            "epub" => {
                // EPUB extraction not yet implemented
                vec![]
            }
            "txt" | "md" | "jsonl" => {
                match String::from_utf8(content_bytes) {
                    Ok(text) => vec![text],
                    Err(e) => {
                        error!("Failed to decode text file as UTF-8: {}", e);
                        vec![]
                    }
                }
            }
            _ => vec![],
        };
        
        persistence.mark_document_as_studied(&file_path.to_string_lossy(), &content_hash).await?;
        Ok(Some(text_chunks))
    }

    pub async fn study_file_chunks(&self, text_chunks: Vec<String>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let engine = self.ml_engine.read().await;
        let mut pc = self.pc_hierarchy.write().await;

        for raw_text in text_chunks {
            let paragraphs = chunk_text(&raw_text);
            
            for chunk in paragraphs {
                if chunk.len() > 50 {
                    let sequence_tensor = engine.process_text_sequence(&chunk).await?;
                    
                    // Clear temporal state between chunks to prevent "thought bleeding"
                    for level in pc.levels.iter_mut() {
                        level.beliefs = level.beliefs.zeros_like()?;
                        level.prev_beliefs = level.prev_beliefs.zeros_like()?;
                    }
                    
                    if let Ok(stats) = pc.learn_sequence(&sequence_tensor, None) {
                        // Calculate study efficiency: How much Level 0 noise was converted into higher-level concepts
                        let n_levels = stats.level_surprises.len();
                        if n_levels >= 2 {
                            let top_l = n_levels.saturating_sub(2);
                            let abstraction_ratio = 1.0 - (stats.level_surprises[top_l] / stats.level_surprises[0].max(1.0));
                            
                            // Efficiency: How much Level 0 noise was converted into Level 2 concepts
                            if abstraction_ratio < 0.2 {
                                warn!("📉 Low Study Efficiency ({:.1}%). Ideas are too complex for {} layers.", abstraction_ratio * 100.0, n_levels);
                            } else {
                                tracing::debug!("📊 Study Efficiency: {:.1}% (abstraction ratio)", abstraction_ratio * 100.0);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// --- Helper Functions ---

/// Specifically designed to extract Question/Answer pairs from GSM8K Parquet files
fn extract_rows_from_parquet(path: &Path) -> Vec<String> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => { error!("Failed to open parquet: {}", e); return vec![]; }
    };

    let reader = match SerializedFileReader::new(file) {
        Ok(r) => r,
        Err(e) => { error!("Failed to read parquet metadata: {}", e); return vec![]; }
    };

    let mut results = Vec::new();
    let iter = reader.get_row_iter(None).unwrap();

    for row in iter {
        if let Ok(r) = row {
            // GSM8K Standard: Question is col 0, Answer is col 1
            let question = r.get_string(0).map(|s| s.as_str()).unwrap_or("");
            let answer = r.get_string(1).map(|s| s.as_str()).unwrap_or("");
            
            if !question.is_empty() {
                results.push(format!("Question: {}\nAnswer: {}", question, answer));
            }
        }
    }
    results
}

fn extract_text_from_pdf(path: &Path) -> Result<String, String> {
    pdf_extract::extract_text(path).map_err(|e| e.to_string())
}

fn extract_text_from_epub(_path: &Path) -> Result<String, String> {
    // Simple epub extraction - for now, just return empty string
    // TODO: Implement proper epub parsing
    Ok("".to_string())
}

fn chunk_text(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|s| s.trim().replace("\n", " "))
        .filter(|s| !s.is_empty())
        .collect()
}

// Removed unused strip_html_tags function
