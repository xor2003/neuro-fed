// src/brain_manager.rs
// Brain manager for uploading/downloading PC-brains via Nostr Blossom protocol.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::blossom_client::{BlossomClient, BlossomError, BrainMetadata};
use crate::nostr_federation::{NostrFederation, NostrError};

/// Errors that can occur in brain management.
#[derive(Debug, thiserror::Error)]
pub enum BrainManagerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Blossom client error: {0}")]
    Blossom(#[from] BlossomError),
    #[error("Nostr federation error: {0}")]
    Nostr(#[from] NostrError),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Safetensors error: {0}")]
    Safetensors(String),
    #[error("Invalid brain: {0}")]
    InvalidBrain(String),
    #[error("Base model mismatch: expected {expected}, got {actual}")]
    BaseModelMismatch { expected: String, actual: String },
    #[error("Brain not found: {0}")]
    BrainNotFound(String),
}

/// Configuration for brain sharing (imported from config module).
use crate::config::BrainSharingConfig;

/// Tracks a downloaded/uploaded brain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainRecord {
    /// Brain ID (SHA256 of weights).
    pub brain_id: String,
    /// Base model ID.
    pub base_model_id: String,
    /// Local path to the brain file (safetensors).
    pub local_path: PathBuf,
    /// Metadata.
    pub metadata: BrainMetadata,
    /// Timestamp of when this brain was added.
    pub added_at: u64,
    /// Whether this brain is currently loaded in the PC hierarchy.
    pub loaded: bool,
}

/// Manages brain sharing and compatibility.
pub struct BrainManager {
    config: BrainSharingConfig,
    blossom_client: BlossomClient,
    nostr_federation: Arc<NostrFederation>,
    /// Map from brain ID to BrainRecord.
    brains: HashMap<String, BrainRecord>,
    /// Current loaded brain ID (if any).
    loaded_brain_id: Option<String>,
}

impl BrainManager {
    /// Create a new BrainManager.
    pub fn new(
        config: BrainSharingConfig,
        nostr_federation: Arc<NostrFederation>,
    ) -> Result<Self, BrainManagerError> {
        // Ensure directories exist.
        std::fs::create_dir_all(&config.brain_storage_dir)
            .map_err(|e| BrainManagerError::Io(e))?;
        std::fs::create_dir_all(&config.cache_dir)
            .map_err(|e| BrainManagerError::Io(e))?;

        let blossom_client = BlossomClient::new(config.relay_urls.clone(), &config.cache_dir);

        let manager = Self {
            config,
            blossom_client,
            nostr_federation,
            brains: HashMap::new(),
            loaded_brain_id: None,
        };

        // Load existing brain records from disk (if any).
        // For simplicity, we skip for now.
        Ok(manager)
    }

    /// Save the current PC‑brain to a safetensors file.
    /// Returns the brain ID (hash) and the local path.
    pub async fn save_brain(
        &self,
        weights: &HashMap<String, Vec<f32>>, // placeholder for actual weights
        description: &str,
        tags: Vec<String>,
    ) -> Result<(String, PathBuf), BrainManagerError> {
        info!("Saving brain with description: {}", description);

        // 1. Serialize weights to safetensors.
        // This is a placeholder; we need to integrate with candle tensors.
        let serialized = self.serialize_weights_to_safetensors(weights)?;

        // 2. Compute SHA256 hash.
        let brain_id = BlossomClient::compute_file_hash_from_bytes(&serialized)?;

        // 3. Save to local storage.
        let filename = format!("{}.safetensors", brain_id);
        let local_path = self.config.brain_storage_dir.join(&filename);
        std::fs::write(&local_path, &serialized)
            .map_err(|e| BrainManagerError::Io(e))?;

        // 4. Create metadata.
        let metadata = BrainMetadata {
            brain_id: brain_id.clone(),
            base_model_id: self.config.base_model_id.clone(),
            version: "0.1.0".to_string(),
            description: description.to_string(),
            size: serialized.len() as u64,
            sha256: brain_id.clone(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            tags,
            license: Some("MIT".to_string()),
            author: "unknown".to_string(), // TODO: get from Nostr identity
        };

        // 5. Store metadata as JSON sidecar.
        let metadata_path = local_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(metadata_path, metadata_json)
            .map_err(BrainManagerError::Io)?;

        info!("Brain saved: {}", brain_id);
        Ok((brain_id, local_path))
    }

    /// Load a brain from a local file into the PC hierarchy.
    /// Returns the raw bytes of the brain file.
    pub async fn load_brain(
        &mut self,
        brain_id: &str,
    ) -> Result<Vec<u8>, BrainManagerError> {
        info!("Loading brain {}", brain_id);

        let record = self
            .brains
            .get(brain_id)
            .ok_or_else(|| BrainManagerError::BrainNotFound(brain_id.to_string()))?;

        // Validate base model compatibility.
        if record.base_model_id != self.config.base_model_id {
            return Err(BrainManagerError::BaseModelMismatch {
                expected: self.config.base_model_id.clone(),
                actual: record.base_model_id.clone(),
            });
        }

        // Load safetensors bytes.
        let brain_bytes = BlossomClient::load_brain(&record.local_path, Some(brain_id)).await?;

        // Update loaded brain ID.
        self.loaded_brain_id = Some(brain_id.to_string());

        // TODO: Integrate with PredictiveCoding hierarchy (pass weights).
        info!("Brain loaded successfully.");
        Ok(brain_bytes)
    }

    /// Share (upload) a brain to the Nostr network.
    pub async fn share_brain(&self, brain_id: &str) -> Result<String, BrainManagerError> {
        info!("Sharing brain {}", brain_id);

        let record = self
            .brains
            .get(brain_id)
            .ok_or_else(|| BrainManagerError::BrainNotFound(brain_id.to_string()))?;

        // Upload via Blossom client.
        let (_uploaded_brain_id, _event_id) = self
            .blossom_client
            .upload_brain(&record.local_path, record.metadata.clone())
            .await?;

        // Publish NIP‑94 event.
        let metadata_json = serde_json::to_string(&record.metadata)?;
        let file_url = format!("https://example.com/brains/{}.safetensors", brain_id); // placeholder
        let published_event_id = self
            .nostr_federation
            .publish_brain_event(brain_id, &metadata_json, &file_url)
            .await?;

        info!("Brain shared, event ID: {}", published_event_id);
        Ok(published_event_id)
    }

    /// Import a brain from the Nostr network (download and add to local storage).
    pub async fn import_brain(
        &mut self,
        brain_id: &str,
        expected_base_model: Option<&str>,
    ) -> Result<BrainRecord, BrainManagerError> {
        info!("Importing brain {}", brain_id);

        // Download via Blossom client.
        let downloaded_path = self
            .blossom_client
            .download_brain(brain_id, expected_base_model)
            .await?;

        // Compute hash to verify.
        let computed_hash_result = BlossomClient::compute_file_hash(&downloaded_path).await;
        let computed_hash = computed_hash_result?;
        if computed_hash != brain_id {
            return Err(BrainManagerError::InvalidBrain(format!(
                "Hash mismatch: expected {}, got {}",
                brain_id, computed_hash
            )));
        }

        // Load metadata (should be sidecar or embedded in NIP‑94 event).
        // For now, we create a minimal metadata.
        let metadata = BrainMetadata {
            brain_id: brain_id.to_string(),
            base_model_id: expected_base_model
                .unwrap_or(&self.config.base_model_id)
                .to_string(),
            version: "unknown".to_string(),
            description: "Imported brain".to_string(),
            size: std::fs::metadata(&downloaded_path)
                .map_err(BrainManagerError::Io)?
                .len(),
            sha256: brain_id.to_string(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            tags: vec!["imported".to_string()],
            license: None,
            author: "unknown".to_string(),
        };

        // Move to brain storage directory.
        let dest_path = self
            .config
            .brain_storage_dir
            .join(format!("{}.safetensors", brain_id));
        if dest_path != downloaded_path {
            std::fs::copy(&downloaded_path, &dest_path).map_err(BrainManagerError::Io)?;
        }

        let record = BrainRecord {
            brain_id: brain_id.to_string(),
            base_model_id: metadata.base_model_id.clone(),
            local_path: dest_path,
            metadata,
            added_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            loaded: false,
        };

        self.brains.insert(brain_id.to_string(), record.clone());
        info!("Brain imported successfully.");
        Ok(record)
    }

    /// List all brains known to this manager.
    pub fn list_brains(&self) -> Vec<BrainRecord> {
        self.brains.values().cloned().collect()
    }

    /// Get the currently loaded brain ID.
    pub fn loaded_brain_id(&self) -> Option<&str> {
        self.loaded_brain_id.as_deref()
    }

    /// Check compatibility of a brain with this node's base model.
    pub fn check_compatibility(&self, brain_id: &str) -> Result<bool, BrainManagerError> {
        let record = self
            .brains
            .get(brain_id)
            .ok_or_else(|| BrainManagerError::BrainNotFound(brain_id.to_string()))?;
        Ok(record.base_model_id == self.config.base_model_id)
    }

    /// Remove a brain from local storage.
    pub fn remove_brain(&mut self, brain_id: &str) -> Result<(), BrainManagerError> {
        let record = self
            .brains
            .get(brain_id)
            .ok_or_else(|| BrainManagerError::BrainNotFound(brain_id.to_string()))?;

        // Delete the brain file.
        std::fs::remove_file(&record.local_path).map_err(BrainManagerError::Io)?;

        // Delete metadata sidecar if exists.
        let metadata_path = record.local_path.with_extension("json");
        let _ = std::fs::remove_file(metadata_path);

        self.brains.remove(brain_id);
        if self.loaded_brain_id.as_deref() == Some(brain_id) {
            self.loaded_brain_id = None;
        }

        info!("Brain removed: {}", brain_id);
        Ok(())
    }

    /// Placeholder: serialize weights to safetensors.
    fn serialize_weights_to_safetensors(
        &self,
        _weights: &HashMap<String, Vec<f32>>,
    ) -> Result<Vec<u8>, BrainManagerError> {
        // TODO: Implement using safetensors crate.
        // For now, return dummy bytes.
        Ok(vec![0u8; 1024])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_brain_manager_creation() {
        let config = BrainSharingConfig::default();
        let nostr_config = crate::config::NostrConfig::default();
        let nostr_federation = Arc::new(crate::nostr_federation::NostrFederation::new(nostr_config));
        let manager = BrainManager::new(config, nostr_federation);
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_save_and_load_brain() {
        // This is a placeholder test; real implementation would need actual weights.
        assert!(true);
    }
}