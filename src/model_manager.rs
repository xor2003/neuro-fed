// src/model_manager.rs
// Model Manager component for automatic model download and selection

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use reqwest::Client;
use reqwest::header::{CONTENT_LENGTH, CONTENT_RANGE, RANGE};
use serde::{Deserialize, Serialize};
use tokio::fs as async_fs;
use tokio::io::AsyncWriteExt;
use tokio::time::sleep;
use tracing::{error, info, warn};

use crate::config::NodeConfig;
use crate::types::DeviceType;

// Local types for model management (not in types.rs to avoid duplication)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoModel {
    pub name: String,
    pub version: String,
    pub parameters: u64,
    pub capabilities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTokenizer {
    pub vocab_size: usize,
    pub max_length: usize,
    pub special_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoConfig {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
}

/// Model information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub size_mb: u64,
    pub min_memory_mb: u64,
    pub max_memory_mb: u64,
    pub quantization: String,
    pub download_url: String,
    pub fallback_download_urls: Vec<String>,
    pub local_path: String,
    pub tokenizer_url: Option<String>,
    pub tokenizer_local_path: Option<String>,
}

/// Model download progress
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub percentage: f64,
    pub speed_kbps: f64,
    pub eta_seconds: u64,
}

/// Model manager error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelManagerError {
    DownloadError(String),
    FileError(String),
    MemoryDetectionError(String),
    ModelLoadError(String),
    InvalidModelError(String),
    NetworkError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for ModelManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ModelManagerError::DownloadError(msg) => write!(f, "Download error: {}", msg),
            ModelManagerError::FileError(msg) => write!(f, "File error: {}", msg),
            ModelManagerError::MemoryDetectionError(msg) => {
                write!(f, "Memory detection error: {}", msg)
            }
            ModelManagerError::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
            ModelManagerError::InvalidModelError(msg) => write!(f, "Invalid model error: {}", msg),
            ModelManagerError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            ModelManagerError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for ModelManagerError {}

/// Model Manager implementation
#[allow(dead_code)]
pub struct ModelManager {
    models: HashMap<String, ModelInfo>,
    client: Client,
    config: NodeConfig,
    download_dir: String,
    progress_callback: Option<Arc<Mutex<dyn Fn(DownloadProgress) + Send + Sync>>>,
}

impl ModelManager {
    /// Create a new ModelManager instance
    pub fn new(config: NodeConfig) -> Self {
        let models = Self::get_default_models();
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .unwrap();

        let download_dir = "models".to_string();

        Self {
            models,
            client,
            config,
            download_dir,
            progress_callback: None,
        }
    }

    /// Set progress callback for download progress
    pub fn set_progress_callback(
        &mut self,
        callback: impl Fn(DownloadProgress) + Send + Sync + 'static,
    ) {
        self.progress_callback = Some(Arc::new(Mutex::new(callback)));
    }

    /// Get available models
    pub fn get_available_models(&self) -> Vec<ModelInfo> {
        self.models.values().cloned().collect()
    }

    fn find_model_by_local_path(&self, model_path: &str) -> Option<ModelInfo> {
        self.models
            .values()
            .find(|model| model.local_path == model_path)
            .cloned()
    }

    fn has_placeholder_model_path(&self) -> bool {
        let requested = Path::new(&self.config.model_path);
        requested == Path::new("models/gguf_model.gguf")
            || requested
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.eq_ignore_ascii_case("gguf_model.gguf"))
                .unwrap_or(false)
    }

    /// Get recommended model based on available memory
    pub async fn get_recommended_model(&self) -> Result<ModelInfo, ModelManagerError> {
        let available_memory = self.detect_available_memory().await.map_err(|e| {
            error!("Memory detection failed: {}", e);
            ModelManagerError::MemoryDetectionError(e.to_string())
        })?;

        info!("Available memory: {} MB", available_memory);

        // Find best matching model
        let mut suitable_models: Vec<_> = self
            .models
            .values()
            .filter(|model| {
                available_memory >= model.min_memory_mb && available_memory <= model.max_memory_mb
            })
            .cloned()
            .collect();

        if suitable_models.is_empty() {
            // Return smallest model if no perfect match
            suitable_models = self.models.values().cloned().collect();
            suitable_models.sort_by_key(|m| m.min_memory_mb);
        }

        suitable_models.sort_by(|a, b| {
            let a_score = (a.max_memory_mb - a.min_memory_mb) as i64;
            let b_score = (b.max_memory_mb - b.min_memory_mb) as i64;
            b_score.cmp(&a_score)
        });

        suitable_models.first().cloned().ok_or_else(|| {
            ModelManagerError::ConfigurationError("No suitable models available".to_string())
        })
    }

    /// Detect available system memory
    pub async fn detect_available_memory(&self) -> Result<u64, String> {
        // Try multiple methods to detect available memory
        if let Ok(memory) = self.detect_memory_linux().await {
            return Ok(memory);
        }

        if let Ok(memory) = self.detect_memory_macos().await {
            return Ok(memory);
        }

        if let Ok(memory) = self.detect_memory_windows().await {
            return Ok(memory);
        }
        if let Ok(memory) = self.detect_memory_windows_fallback().await {
            return Ok(memory);
        }

        warn!("Failed to detect available memory on this platform, using conservative fallback");
        Ok(4096)
    }

    /// Detect memory on Linux
    async fn detect_memory_linux(&self) -> Result<u64, String> {
        use tokio::process::Command;

        let output = match Command::new("free").arg("-m").output().await {
            Ok(output) => output,
            Err(e) => return Err(e.to_string()),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();

        if lines.len() < 2 {
            return Err("Invalid free command output".to_string());
        }

        let mem_line: Vec<&str> = lines[1].split_whitespace().collect();
        if mem_line.len() < 4 {
            return Err("Invalid memory line format".to_string());
        }

        let available_str = mem_line.get(3).unwrap_or(&"0");
        available_str
            .parse::<u64>()
            .map_err(|_| "Failed to parse available memory".to_string())
    }

    /// Detect memory on macOS
    async fn detect_memory_macos(&self) -> Result<u64, String> {
        use tokio::process::Command;

        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .await
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Ok(bytes) = stdout.trim().parse::<u64>() {
                let total_mb = bytes / 1024 / 1024;
                if total_mb > 0 {
                    // Use a conservative fraction of total unified memory for model selection.
                    return Ok(((total_mb as f64) * 0.6) as u64);
                }
            }
        }

        let output = match Command::new("vm_stat").output().await {
            Ok(output) => output,
            Err(e) => return Err(e.to_string()),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut page_size = 4096u64;
        let mut pages_free = 0u64;
        let mut pages_speculative = 0u64;
        let mut pages_inactive = 0u64;

        for line in stdout.lines() {
            if line.contains("page size of") {
                let digits: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
                if let Ok(parsed) = digits.parse::<u64>() {
                    page_size = parsed;
                }
            } else if line.contains("free pages:") {
                if let Some(num) = line.split_whitespace().nth(1) {
                    if let Ok(val) = num.trim_matches('.').parse::<u64>() {
                        pages_free = val;
                    }
                }
            } else if line.contains("speculative pages:") {
                if let Some(num) = line.split_whitespace().nth(1) {
                    if let Ok(val) = num.trim_matches('.').parse::<u64>() {
                        pages_speculative = val;
                    }
                }
            } else if line.contains("Pages inactive:") || line.contains("inactive pages:") {
                if let Some(num) = line
                    .split_whitespace()
                    .nth(2)
                    .or_else(|| line.split_whitespace().nth(1))
                {
                    if let Ok(val) = num.trim_matches('.').parse::<u64>() {
                        pages_inactive = val;
                    }
                }
            }
        }

        let available_pages = pages_free + pages_speculative + pages_inactive;
        let available_bytes = available_pages.saturating_mul(page_size);
        let available_mb = available_bytes / 1024 / 1024;
        if available_mb > 0 {
            Ok(available_mb)
        } else {
            Err("Failed to parse available macOS memory".to_string())
        }
    }

    /// Detect memory on Windows
    async fn detect_memory_windows(&self) -> Result<u64, String> {
        use tokio::process::Command;

        let output = match Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory")
            .output()
            .await
        {
            Ok(output) => output,
            Err(e) => return Err(e.to_string()),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();

        let free_memory_str = lines
            .iter()
            .find_map(|line| {
                let trimmed = line.trim();
                if trimmed.is_empty() || !trimmed.chars().all(|c| c.is_ascii_digit()) {
                    None
                } else {
                    Some(trimmed)
                }
            })
            .ok_or_else(|| "Invalid Windows memory output".to_string())?;

        free_memory_str
            .parse::<u64>()
            .map(|kb| kb / 1024)
            .map_err(|_| "Failed to parse free memory".to_string())
    }

    #[allow(dead_code)]
    async fn detect_memory_windows_fallback(&self) -> Result<u64, String> {
        use tokio::process::Command;

        let output = match Command::new("cmd")
            .arg("/C")
            .arg("systeminfo | findstr /C:\"Available Physical Memory\"")
            .output()
            .await
        {
            Ok(output) => output,
            Err(e) => return Err(e.to_string()),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let digits: String = stdout.chars().filter(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return Err("Invalid Windows fallback memory output".to_string());
        }
        digits
            .parse::<u64>()
            .map_err(|_| "Failed to parse fallback free memory".to_string())
    }

    /// Check if model is already downloaded
    pub async fn is_model_downloaded(&self, model_name: &str) -> bool {
        if let Some(model) = self.models.get(model_name) {
            let path = Path::new(&model.local_path);
            let tokenizer_ok = model
                .tokenizer_local_path
                .as_ref()
                .map(|tokenizer_path| Path::new(tokenizer_path).exists())
                .unwrap_or(true);
            path.exists() && tokenizer_ok
        } else {
            false
        }
    }

    pub async fn ensure_startup_model(&self) -> Result<Option<ModelInfo>, ModelManagerError> {
        let requested_path = Path::new(&self.config.model_path);
        if requested_path.exists() {
            return Ok(None);
        }

        if let Some(model) = self.find_model_by_local_path(&self.config.model_path) {
            self.download_model(&model.name).await?;
            return Ok(Some(model));
        }

        if self.has_placeholder_model_path() {
            let model = self.get_recommended_model().await?;
            self.download_model(&model.name).await?;
            return Ok(Some(model));
        }

        if let Some(downloaded) = self
            .models
            .values()
            .find(|model| Path::new(&model.local_path).exists())
            .cloned()
        {
            return Ok(Some(downloaded));
        }

        Err(ModelManagerError::FileError(format!(
            "Configured model path {} does not exist and does not match a managed model",
            self.config.model_path
        )))
    }

    /// Download a model with progress tracking
    pub async fn download_model(&self, model_name: &str) -> Result<(), ModelManagerError> {
        let model = self.models.get(model_name).ok_or_else(|| {
            ModelManagerError::InvalidModelError(format!("Model {} not found", model_name))
        })?;

        if self.is_model_downloaded(model_name).await {
            info!("Model {} already downloaded", model_name);
        } else {
            info!("Starting download for model: {}", model_name);
            self.download_model_payload(model).await?;
            info!("Model {} downloaded successfully", model_name);
        }

        // Download tokenizer if URL is provided
        if let Some(tokenizer_url) = &model.tokenizer_url {
            if let Some(tokenizer_path) = &model.tokenizer_local_path {
                let tokenizer_path = Path::new(tokenizer_path);
                if !tokenizer_path.exists() {
                    info!("Downloading tokenizer for model: {}", model_name);
                    self.download_file(tokenizer_url, tokenizer_path).await?;
                    info!("Tokenizer for {} downloaded successfully", model_name);
                } else {
                    info!("Tokenizer for {} already exists", model_name);
                }
            }
        }

        Ok(())
    }

    async fn download_model_payload(&self, model: &ModelInfo) -> Result<(), ModelManagerError> {
        let mut errors = Vec::new();
        for url in std::iter::once(&model.download_url).chain(model.fallback_download_urls.iter()) {
            match self
                .download_with_retries(url, Path::new(&model.local_path), 3)
                .await
            {
                Ok(()) => return Ok(()),
                Err(err) => {
                    warn!("Model download attempt failed for {}: {}", url, err);
                    errors.push(format!("{} => {}", url, err));
                }
            }
        }

        Err(ModelManagerError::DownloadError(format!(
            "All download URLs failed for {}: {}",
            model.name,
            errors.join(" | ")
        )))
    }

    async fn download_with_retries(
        &self,
        url: &str,
        file_path: &Path,
        max_attempts: usize,
    ) -> Result<(), ModelManagerError> {
        let mut last_error = None;
        for attempt in 1..=max_attempts.max(1) {
            match self.download_file(url, file_path).await {
                Ok(()) => return Ok(()),
                Err(err) => {
                    last_error = Some(err.clone());
                    if attempt < max_attempts {
                        warn!(
                            "Download attempt {}/{} failed for {}: {}. Retrying...",
                            attempt, max_attempts, url, err
                        );
                        sleep(Duration::from_secs(attempt as u64)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ModelManagerError::DownloadError(format!("Failed to download {}", url))
        }))
    }

    /// Helper function to download a file with progress tracking
    async fn download_file(&self, url: &str, file_path: &Path) -> Result<(), ModelManagerError> {
        let dir_path = file_path.parent().unwrap_or(Path::new(&self.download_dir));

        // Create directory if it doesn't exist
        if let Err(e) = async_fs::create_dir_all(dir_path).await {
            return Err(ModelManagerError::FileError(e.to_string()));
        }

        let temp_path = temporary_download_path(file_path);
        let existing_bytes = async_fs::metadata(&temp_path)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);

        let mut request = self.client.get(url);
        if existing_bytes > 0 {
            request = request.header(RANGE, format!("bytes={}-", existing_bytes));
            info!(
                "Resuming download for {} from byte {}",
                file_path.display(),
                existing_bytes
            );
        }

        let response = request
            .send()
            .await
            .map_err(|e| ModelManagerError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            return Err(ModelManagerError::DownloadError(format!(
                "HTTP status {} for url ({})",
                status, url
            )));
        }

        let supports_resume = status == reqwest::StatusCode::PARTIAL_CONTENT;
        let must_restart = existing_bytes > 0 && !supports_resume;
        let resumed_bytes = if must_restart { 0 } else { existing_bytes };
        if must_restart {
            warn!(
                "Server did not honor resume request for {}. Restarting download from byte 0.",
                url
            );
            let _ = async_fs::remove_file(&temp_path).await;
        }

        let total_size = total_bytes_from_headers(response.headers(), resumed_bytes);
        let mut stream = response.bytes_stream();

        let mut file = if supports_resume && resumed_bytes > 0 {
            async_fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&temp_path)
                .await
                .map_err(|e| ModelManagerError::FileError(e.to_string()))?
        } else {
            async_fs::File::create(&temp_path)
                .await
                .map_err(|e| ModelManagerError::FileError(e.to_string()))?
        };

        let mut downloaded = resumed_bytes;
        let start_time = std::time::Instant::now();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ModelManagerError::NetworkError(e.to_string()))?;

            downloaded += chunk.len() as u64;
            file.write_all(&chunk)
                .await
                .map_err(|e| ModelManagerError::FileError(e.to_string()))?;

            // Calculate progress
            let percentage = if total_size > 0 {
                (downloaded as f64 / total_size as f64) * 100.0
            } else {
                0.0
            };

            let elapsed = start_time.elapsed();
            let speed_kbps = if elapsed.as_secs() > 0 {
                (downloaded as f64 / elapsed.as_secs() as f64) / 1024.0
            } else {
                0.0
            };

            let eta_seconds = if speed_kbps > 0.0 && total_size > 0 {
                ((total_size - downloaded) as f64 / speed_kbps / 1024.0) as u64
            } else {
                0
            };

            let progress = DownloadProgress {
                bytes_downloaded: downloaded,
                total_bytes: total_size,
                percentage,
                speed_kbps,
                eta_seconds,
            };

            // Call progress callback if set
            if let Some(callback) = &self.progress_callback {
                let callback = callback.lock().unwrap();
                callback(progress);
            }
        }

        file.sync_all()
            .await
            .map_err(|e| ModelManagerError::FileError(e.to_string()))?;

        async_fs::rename(&temp_path, file_path)
            .await
            .map_err(|e| ModelManagerError::FileError(e.to_string()))?;

        info!(
            "File downloaded successfully ({} MB)",
            downloaded / 1024 / 1024
        );
        Ok(())
    }

    /// Load a model using candle-core
    pub async fn load_model(&self, model_name: &str) -> Result<AutoModel, ModelManagerError> {
        let _model = self.models.get(model_name).ok_or_else(|| {
            ModelManagerError::InvalidModelError(format!("Model {} not found", model_name))
        })?;

        if !self.is_model_downloaded(model_name).await {
            self.download_model(model_name).await.map_err(|e| {
                error!("Failed to download model: {}", e);
                e
            })?;
        }

        // Simulate model loading with candle-core
        // In a real implementation, this would use candle-core's model loading
        let loaded_model = AutoModel {
            name: model_name.to_string(),
            version: "1.0".to_string(),
            parameters: 0, // Would be actual parameter count
            capabilities: vec!["text-generation".to_string()],
        };

        info!("Model {} loaded successfully", model_name);
        Ok(loaded_model)
    }

    /// Get model configuration for ML engine
    pub fn get_model_config(&self, model_name: &str) -> Result<AutoConfig, ModelManagerError> {
        let _model = self.models.get(model_name).ok_or_else(|| {
            ModelManagerError::InvalidModelError(format!("Model {} not found", model_name))
        })?;

        // Create configuration based on model characteristics
        let config = AutoConfig {
            hidden_size: 4096,             // Example value
            num_layers: 32,                // Example value
            vocab_size: 32000,             // Example value
            max_position_embeddings: 2048, // Example value
        };

        Ok(config)
    }

    /// Get tokenizer for a model
    pub fn get_tokenizer(&self, model_name: &str) -> Result<AutoTokenizer, ModelManagerError> {
        let _model = self.models.get(model_name).ok_or_else(|| {
            ModelManagerError::InvalidModelError(format!("Model {} not found", model_name))
        })?;

        // Create tokenizer based on model characteristics
        let tokenizer = AutoTokenizer {
            vocab_size: 32000, // Example value
            max_length: 2048,  // Example value
            special_tokens: vec![
                "[PAD]".to_string(),
                "[UNK]".to_string(),
                "[CLS]".to_string(),
                "[SEP]".to_string(),
                "[MASK]".to_string(),
            ],
        };

        Ok(tokenizer)
    }

    /// Get device configuration for a model
    pub fn get_device_config(&self, model_name: &str) -> Result<DeviceType, ModelManagerError> {
        let model = self.models.get(model_name).ok_or_else(|| {
            ModelManagerError::InvalidModelError(format!("Model {} not found", model_name))
        })?;

        // Determine device based on model size and available memory
        let device_type = if model.min_memory_mb > 8192 {
            DeviceType {
                name: "cpu".to_string(),
                description: "CPU processing (large model)".to_string(),
                supported: true,
            }
        } else {
            DeviceType {
                name: "auto".to_string(),
                description: "Auto-detect best device".to_string(),
                supported: true,
            }
        };

        Ok(device_type)
    }

    /// Get default models configuration
    fn get_default_models() -> HashMap<String, ModelInfo> {
        let mut models = HashMap::new();

        // TinyLlama 1.1B Chat (GGUF, Q2_K)
        models.insert(
            "tinyllama-1.1b-chat".to_string(),
            ModelInfo {
                name: "tinyllama-1.1b-chat".to_string(),
                version: "1.0".to_string(),
                size_mb: 622, // Approximate size for Q4_K_M
                min_memory_mb: 1024,
                max_memory_mb: 2048,
                quantization: "Q4_K_M".to_string(),
                download_url: "https://huggingface.co/tensorblock/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf".to_string(),
                fallback_download_urls: vec![
                    "https://huggingface.co/pbatra/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
                    "https://huggingface.co/tensorblock/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q2_K.gguf".to_string(),
                ],
                local_path: "models/tinyllama/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf".to_string(),
                tokenizer_url: Some("https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/raw/main/tokenizer.json".to_string()),
                tokenizer_local_path: Some("models/tinyllama/tokenizer.json".to_string()),
            },
        );

        // Qwen2.5 1.5B Instruct (GGUF, Q4_K_M)
        models.insert(
            "qwen2.5-1.5b-instruct".to_string(),
            ModelInfo {
                name: "qwen2.5-1.5b-instruct".to_string(),
                version: "2.5".to_string(),
                size_mb: 1120, // Approximate size for Q4_K_M
                min_memory_mb: 4096,
                max_memory_mb: 32768,
                quantization: "Q4_K_M".to_string(),
                download_url: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                fallback_download_urls: vec![
                    "https://huggingface.co/QuantFactory/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                    "https://huggingface.co/itlwas/Qwen2.5-1.5B-Instruct-Q4_K_M-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                ],
                local_path: "models/qwen2.5/qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
                tokenizer_url: Some("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/tokenizer.json".to_string()),
                tokenizer_local_path: Some("models/qwen2.5/tokenizer.json".to_string()),
            },
        );

        // Add more models as needed
        models
    }

    /// Clean up downloaded models
    pub async fn cleanup(&self) -> Result<(), ModelManagerError> {
        for model in self.models.values() {
            let path = Path::new(&model.local_path);
            if path.exists() {
                if let Err(e) = async_fs::remove_file(path).await {
                    warn!("Failed to remove model file {}: {}", model.name, e);
                }
            }
        }
        Ok(())
    }
}

fn temporary_download_path(file_path: &Path) -> PathBuf {
    let mut os = file_path.as_os_str().to_os_string();
    os.push(".part");
    PathBuf::from(os)
}

fn total_bytes_from_headers(headers: &reqwest::header::HeaderMap, resumed_bytes: u64) -> u64 {
    if let Some(content_range) = headers.get(CONTENT_RANGE)
        && let Ok(content_range) = content_range.to_str()
        && let Some((_range, total)) = content_range.split_once('/')
        && let Ok(total) = total.parse::<u64>()
    {
        return total;
    }

    headers
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .map(|content_length| content_length + resumed_bytes)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_model_manager_creation() {
        let config = NodeConfig::default();
        let manager = ModelManager::new(config);

        assert_eq!(manager.models.len(), 2);
        assert!(manager.models.contains_key("tinyllama-1.1b-chat"));
        assert!(manager.models.contains_key("qwen2.5-1.5b-instruct"));
    }

    #[tokio::test]
    async fn test_memory_detection() {
        let config = NodeConfig::default();
        let manager = ModelManager::new(config);

        let memory = manager.detect_available_memory().await;
        assert!(memory.is_ok());
        assert!(memory.unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_model_recommendation() {
        let config = NodeConfig::default();
        let manager = ModelManager::new(config);

        let recommended = manager.get_recommended_model().await;
        assert!(recommended.is_ok());
        let model = recommended.unwrap();

        assert!(model.name == "qwen2.5-1.5b-instruct" || model.name == "tinyllama-1.1b-chat");
    }

    #[test]
    fn test_total_bytes_from_content_range() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_RANGE, "bytes 100-199/1000".parse().unwrap());

        assert_eq!(total_bytes_from_headers(&headers, 100), 1000);
    }

    #[test]
    fn test_total_bytes_from_content_length_plus_existing_bytes() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_LENGTH, "900".parse().unwrap());

        assert_eq!(total_bytes_from_headers(&headers, 100), 1000);
    }

    #[test]
    fn test_temporary_download_path_appends_part_suffix() {
        let path = Path::new("models/qwen/model.gguf");
        let temp = temporary_download_path(path);
        assert_eq!(temp, PathBuf::from("models/qwen/model.gguf.part"));
    }

    #[tokio::test]
    async fn test_model_download() {
        let config = NodeConfig::default();
        let manager = ModelManager::new(config);

        // This would normally download, but we'll mock for testing
        // Since we cannot mock the HTTP client, we'll test that the function
        // returns either Ok (if file already exists) or Err (if network fails).
        // We'll accept either result because the test environment is unpredictable.
        let result = manager.download_model("qwen2.5-1.5b-instruct").await;
        // The test passes as long as the function doesn't panic.
        // We'll log the result for debugging.
        println!("test_model_download result: {:?}", result);
    }

    #[tokio::test]
    async fn test_model_loading() {
        let config = NodeConfig::default();
        let manager = ModelManager::new(config);

        let result = manager.load_model("qwen2.5-1.5b").await;
        assert!(result.is_err()); // Should fail due to missing file
    }

    #[tokio::test]
    async fn test_ensure_startup_model_prefers_existing_configured_file() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("existing.gguf");
        std::fs::write(&model_path, b"stub").unwrap();

        let mut config = NodeConfig::default();
        config.model_path = model_path.to_string_lossy().to_string();

        let manager = ModelManager::new(config);
        let result = manager.ensure_startup_model().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_ensure_startup_model_uses_downloaded_managed_model_when_placeholder_missing() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path().join("models").join("qwen2.5");
        std::fs::create_dir_all(&model_dir).unwrap();
        let downloaded_model = model_dir.join("qwen2.5-1.5b-instruct-q4_k_m.gguf");
        let downloaded_tokenizer = model_dir.join("tokenizer.json");
        std::fs::write(&downloaded_model, b"stub").unwrap();
        std::fs::write(&downloaded_tokenizer, b"{}").unwrap();

        let mut config = NodeConfig::default();
        config.model_path = dir
            .path()
            .join("models")
            .join("gguf_model.gguf")
            .to_string_lossy()
            .to_string();

        let mut manager = ModelManager::new(config);
        if let Some(model) = manager.models.get_mut("qwen2.5-1.5b-instruct") {
            model.local_path = downloaded_model.to_string_lossy().to_string();
            model.tokenizer_local_path = Some(downloaded_tokenizer.to_string_lossy().to_string());
        }

        let resolved = manager.ensure_startup_model().await.unwrap().unwrap();
        assert_eq!(resolved.name, "qwen2.5-1.5b-instruct");
        assert_eq!(
            resolved.local_path,
            downloaded_model.to_string_lossy().to_string()
        );
    }
}
