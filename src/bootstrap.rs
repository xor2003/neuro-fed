use std::fs::File;
use std::io::BufRead;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use candle_core::Tensor;
use sha2::{Digest, Sha256};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use walkdir::WalkDir;
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
    pub shutdown_signal: Arc<AtomicBool>, // <--- NEW
}

impl BootstrapManager {
    pub fn new(
        ml_engine: Arc<RwLock<MLEngine>>,
        thought_decoder: Arc<RwLock<ThoughtDecoder>>,
        dict: Arc<RwLock<CognitiveDictionary>>,
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        config: crate::config::BootstrapConfig,
    ) -> Self {
        Self {
            ml_engine,
            thought_decoder,
            dict,
            pc_hierarchy,
            config,
            shutdown_signal: Arc::new(AtomicBool::new(false)), // <--- NEW
        }
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
            
            // --- 🔴 MODIFIED BLOCK START ---
            if path.is_dir() {
                // Use WalkDir to find all files recursively
                for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                    if entry.file_type().is_file() {
                        all_files.push(entry.path().to_path_buf());
                    }
                }
            } else if path.is_file() {
                // Keep the logic to handle single file paths
                all_files.push(path.to_path_buf());
            }
            // --- 🔴 MODIFIED BLOCK END ---
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
                "pdf" => {
                    match std::fs::read(&file_path) {
                        Ok(bytes) => extract_text_from_pdf(&bytes).map(|s| vec![s]).unwrap_or_default(),
                        Err(_) => vec![],
                    }
                }
                "epub" => extract_text_from_epub(&file_path).map(|s| vec![s]).unwrap_or_default(),
                "txt" | "md" => {
                    std::fs::read_to_string(&file_path)
                        .map(|s| vec![s])
                        .unwrap_or_default()
                }
                "jsonl" => extract_lines_from_jsonl(&file_path),
                _ => {
                    warn!("Unsupported file type: {}. Skipping.", ext);
                    vec![]
                }
            };

            tracing::debug!("📖 Extracted {} text chunks from {}", text_chunks.len(), file_name);
            
            for (idx, raw_text) in text_chunks.iter().enumerate() {
                // Log position information based on file type
                match ext {
                    "jsonl" => {
                        tracing::info!("📝 Studying JSONL line {}", idx + 1);
                    }
                    "pdf" => {
                        tracing::info!("📄 Studying PDF page {}", idx + 1);
                    }
                    "epub" => {
                        tracing::info!("📖 Studying EPUB page {}", idx + 1);
                    }
                    "parquet" => {
                        tracing::info!("📊 Studying Parquet row {}", idx + 1);
                    }
                    _ => {
                        tracing::info!("📄 Studying text chunk {}", idx + 1);
                    }
                }
                
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
        let mut lr = self.config.learning_rate.max(0.01) as f64;
        
        // 🚀 FIX #11: Adaptive Learning Rate
        let mut best_loss = f32::INFINITY;
        let mut patience = 0;
        const PATIENCE_LIMIT: usize = 10;
        const LR_DECAY_FACTOR: f64 = 0.5;
        const MIN_LR: f64 = 1e-5;
        
        for epoch in 0..max_epochs {
            let mut total_loss = 0.0;
            for (belief, seq) in &synthetic_data {
                let loss = decoder.train_step(belief, seq, lr)?;
                total_loss += loss;
            }
            let avg_loss = total_loss / synthetic_data.len() as f32;
            
            // Adaptive learning rate logic
            if avg_loss < best_loss - 1e-4 {
                // Improvement
                best_loss = avg_loss;
                patience = 0;
            } else {
                patience += 1;
                if patience >= PATIENCE_LIMIT && lr > MIN_LR {
                    // Reduce learning rate
                    lr = (lr * LR_DECAY_FACTOR).max(MIN_LR);
                    info!("📉 Loss plateaued at epoch {}, reducing learning rate to {:.6}", epoch, lr);
                    patience = 0; // reset patience after LR reduction
                }
            }
            
            if epoch % 20 == 0 {
                 info!("Epoch {}: Loss = {:.4}, LR = {:.6}", epoch, avg_loss, lr);
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
        
        // Comment out this check temporarily for debugging
        // if persistence.is_document_studied(&file_path.to_string_lossy(), &content_hash).await? {
        //     tracing::debug!("Skipping already studied file: {:?}", file_path);
        //     return Ok(None);
        // }
        
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
                // Extract text directly from bytes
                match extract_text_from_pdf(&content_bytes) {
                    Ok(text) => vec![text],
                    Err(e) => {
                        error!("Failed to extract text from PDF: {}", e);
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
        let engine_lock = self.ml_engine.clone();
        let main_pc_lock = self.pc_hierarchy.clone();
        let shutdown = self.shutdown_signal.clone();

        // 1. Get base configuration and initial weights to clone
        let base_weights = main_pc_lock.read().await.get_level_weights()?;
        let pc_config = main_pc_lock.read().await.config.clone();
        let device = main_pc_lock.read().await.device.clone();

        // 2. Determine how many pieces to split the book into (based on actual Rayon threads)
        let num_threads = rayon::current_num_threads().max(1);
        let chunk_size = (text_chunks.len() / num_threads).max(1);
        
        tracing::info!("📚 Splitting document into {} parallel segments across {} CPU cores...",
            (text_chunks.len() as f32 / chunk_size as f32).ceil(), num_threads);

        let segments: Vec<Vec<String>> = text_chunks.chunks(chunk_size).map(|c| c.to_vec()).collect();

        // 3. 🔴 ESCAPE TOKIO: Move CPU-heavy math into Rayon threads
        // We use spawn_blocking so Tokio doesn't freeze while Rayon does the math.
        let shutdown_clone = shutdown.clone();
        let all_learned_weights = tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;

            segments.into_par_iter().filter_map(|segment| {
                if shutdown_clone.load(Ordering::Relaxed) { return None; }

                // We need a tiny local async runtime to call process_text_sequence
                // because we are currently inside a synchronous Rayon thread.
                let rt = tokio::runtime::Builder::new_current_thread().build().unwrap();
                
                // Create an isolated clone of the PC Brain for this specific CPU core
                let mut local_pc = PredictiveCoding::new_with_device(pc_config.clone(), device.clone()).ok()?;
                local_pc.load_level_weights(base_weights.clone()).ok()?;

                for raw_text in segment {
                    if shutdown_clone.load(Ordering::Relaxed) { return None; }
                    let paragraphs = chunk_text(&raw_text);
                    
                    for p in paragraphs {
                        if p.len() > 50 {
                            // Run the async embedder inside our tiny sync runtime
                            let sequence_tensor = rt.block_on(async {
                                engine_lock.read().await.process_text_sequence(&p).await
                            }).ok()?;
                            
                            // Reset temporal state for new paragraph
                            for level in local_pc.levels.iter_mut() {
                                level.beliefs = level.beliefs.zeros_like().ok()?;
                                level.prev_beliefs = level.prev_beliefs.zeros_like().ok()?;
                            }
                            
                            let _ = local_pc.learn_sequence(&sequence_tensor, None);
                        }
                    }
                }
                
                // Return the brain weights that THIS specific core learned
                local_pc.get_level_weights().ok()
            }).collect::<Vec<_>>()
        }).await?;

        if shutdown.load(Ordering::Relaxed) {
            info!("🛑 Study interrupted by shutdown signal. Discarding parallel weights.");
            return Ok(());
        }

        // 4. 🔴 KNOWLEDGE MERGE: Federated Averaging
        // We take the brains from all CPU cores and mathematically average them.
        tracing::info!("🧠 Merging learned knowledge from {} parallel cores...", all_learned_weights.len());
        let mut main_pc = main_pc_lock.write().await;
        
        if !all_learned_weights.is_empty() {
            let num_models = all_learned_weights.len() as f32;
            let n_levels = main_pc.levels.len();

            for l in 0..n_levels {
                // Get the total number of elements in this level's weight matrix
                let elem_count = main_pc.levels[l].weights.flatten_all()?.elem_count();
                let mut avg_weights_l: Vec<f32> = vec![0.0; elem_count];
                
                // Sum all weights from all parallel threads
                for model_weights in &all_learned_weights {
                    if let Some(level_weights) = model_weights.get(l) {
                        for (avg_w, new_w) in avg_weights_l.iter_mut().zip(level_weights.weights.iter()) {
                            *avg_w += *new_w;
                        }
                    }
                }

                // Divide by N to get the true mathematical average
                for avg_w in &mut avg_weights_l {
                    *avg_w /= num_models;
                }

                // Apply the averaged knowledge back to the main Brain
                main_pc.levels[l].set_weights_from_vec(avg_weights_l)?;
            }
            tracing::info!("✅ Parallel knowledge merged successfully.");
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

    // Get total row count from metadata
    let metadata = reader.metadata();
    let total_rows = metadata.file_metadata().num_rows();
    tracing::info!("📊 Parquet file {} has {} total rows", path.display(), total_rows);

    let mut results = Vec::new();
    let iter = reader.get_row_iter(None).unwrap();

    for (row_idx, row) in iter.enumerate() {
        if let Ok(r) = row {
            // GSM8K Standard: Question is col 0, Answer is col 1
            let question = r.get_string(0).map(|s| s.as_str()).unwrap_or("");
            let answer = r.get_string(1).map(|s| s.as_str()).unwrap_or("");
            
            if !question.is_empty() {
                results.push(format!("Question: {}\nAnswer: {}", question, answer));
                
                // Log progress every 100 rows
                if row_idx % 100 == 0 {
                    tracing::debug!("📊 Processing parquet row {} / {}", row_idx + 1, total_rows);
                    tracing::info!("📊 Studying Parquet row {}", row_idx + 1);
                }
            }
        }
    }
    
    tracing::info!("📊 Finished processing {} rows from {}", results.len(), path.display());
    results
}

fn extract_lines_from_jsonl(path: &Path) -> Vec<String> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => { error!("Failed to open JSONL: {}", e); return vec![]; }
    };
    
    let mut lines = Vec::new();
    let reader = std::io::BufReader::new(file);
    
    // Count total lines first
    let mut line_count = 0;
    for line in reader.lines() {
        match line {
            Ok(l) => {
                lines.push(l);
                line_count += 1;
                
                // Log progress every 100 lines
                if line_count % 100 == 0 {
                    tracing::debug!("📝 Processing JSONL line {}", line_count);
                    tracing::info!("📝 Studying JSONL line {}", line_count);
                }
            }
            Err(e) => {
                error!("Error reading JSONL line: {}", e);
            }
        }
    }
    
    tracing::info!("📝 JSONL file {} has {} total lines", path.display(), line_count);
    lines
}

fn extract_text_from_pdf(bytes: &[u8]) -> Result<String, String> {
    let doc = lopdf::Document::load_mem(bytes).map_err(|e| e.to_string())?;
    let mut full_text = String::new();
    
    let pages = doc.get_pages();
    let total_pages = pages.len();
    tracing::info!("📄 PDF has {} total pages", total_pages);
    
    for (page_num, _) in pages.into_iter() {
        if let Ok(text) = doc.extract_text(&[page_num]) {
            full_text.push_str(&text);
            full_text.push('\n');
            
            // Log progress every 10 pages
            if page_num % 10 == 0 {
                tracing::debug!("📄 Processing PDF page {} / {}", page_num + 1, total_pages);
                tracing::info!("📄 Studying PDF page {}", page_num + 1);
            }
        }
    }
    
    if full_text.trim().is_empty() {
        return Err("No text found in PDF".to_string());
    }
    Ok(full_text)
}

fn strip_html_tags(html: &str) -> String {
    let mut text = String::new();
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => (),
        }
    }
    text
}

fn extract_text_from_epub(path: &Path) -> Result<String, String> {
    // Read the EPUB file into bytes
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let cursor = std::io::Cursor::new(bytes);
    let mut doc = epub::doc::EpubDoc::from_reader(cursor).map_err(|e| e.to_string())?;
    let mut content = String::new();

    // Iterate through all chapters
    let total_pages = doc.get_num_pages();
    for page_num in 0..total_pages {
        if let Some((curr_content, _)) = doc.get_current_str() {
            content.push_str(&strip_html_tags(&curr_content));
            content.push('\n');
            
            // Log progress every 10 pages
            if page_num % 10 == 0 {
                tracing::debug!("📖 Processing EPUB page {}/{}", page_num + 1, total_pages);
                tracing::info!("📖 Studying EPUB page {}", page_num + 1);
            }
        }
        doc.go_next();
    }
    
    if content.trim().is_empty() {
        return Err("No text content found in EPUB".to_string());
    }
    
    tracing::info!("📖 EPUB file {} processed ({} pages)", path.display(), total_pages);
    Ok(content)
}

fn chunk_text(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(|s| s.trim().replace("\n", " "))
        .filter(|s| !s.is_empty())
        .collect()
}

// Removed unused strip_html_tags function
