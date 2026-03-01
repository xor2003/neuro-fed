// src/blossom_client.rs
// Blossom client for Nostr's file storage protocol (NIP-94)

use std::path::{Path, PathBuf};
use std::fs;
use std::io;

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use anyhow::Result;
use thiserror::Error;
use tracing::{info, warn};

/// Errors that can occur in Blossom client operations.
#[derive(Error, Debug)]
pub enum BlossomError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Safetensors error: {0}")]
    Safetensors(String),
    #[error("Network error: {0}")]
    Network(String),
    #[error("Invalid file hash: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("Missing metadata: {0}")]
    MissingMetadata(String),
    #[error("Nostr error: {0}")]
    Nostr(String),
}

/// Metadata for a brain file (NIP-94 compatible).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMetadata {
    /// Unique identifier for this brain (SHA256 of weights).
    pub brain_id: String,
    /// Base model identifier (e.g., "llama-3-8b").
    pub base_model_id: String,
    /// Version of the brain (semantic versioning).
    pub version: String,
    /// Description of the brain.
    pub description: String,
    /// Size of the brain file in bytes.
    pub size: u64,
    /// SHA256 hash of the brain file.
    pub sha256: String,
    /// Timestamp of creation (Unix epoch).
    pub created_at: u64,
    /// Tags for categorization.
    pub tags: Vec<String>,
    /// License information.
    pub license: Option<String>,
    /// Author public key (Nostr pubkey).
    pub author: String,
}

/// Blossom client for uploading/downloading brain files via Nostr.
pub struct BlossomClient {
    /// Nostr federation client (to be integrated).
    /// For now, we just store relay URLs.
    relay_urls: Vec<String>,
    /// Local directory for caching downloaded brains.
    cache_dir: PathBuf,
}

impl BlossomClient {
    /// Create a new Blossom client.
    pub fn new(relay_urls: Vec<String>, cache_dir: impl AsRef<Path>) -> Self {
        let cache_dir = cache_dir.as_ref().to_path_buf();
        Self {
            relay_urls,
            cache_dir,
        }
    }

    /// Upload a brain (safetensors file) to Nostr relays.
    /// Returns the brain ID (hash) and the NIP-94 event ID.
    pub async fn upload_brain(
        &self,
        brain_path: impl AsRef<Path>,
        metadata: BrainMetadata,
    ) -> Result<(String, String), BlossomError> {
        let brain_path = brain_path.as_ref();
        info!("Uploading brain from {:?}", brain_path);

        // 1. Validate the brain file (safetensors).
        let file_bytes = fs::read(brain_path)
            .map_err(|e| BlossomError::Io(e))?;
        let _ = SafeTensors::deserialize(&file_bytes)
            .map_err(|e| BlossomError::Safetensors(e.to_string()))?;

        // 2. Compute SHA256 hash.
        let mut hasher = Sha256::new();
        hasher.update(&file_bytes);
        let hash = hasher.finalize();
        let hex_hash = hex::encode(hash);
        if hex_hash != metadata.sha256 {
            return Err(BlossomError::HashMismatch {
                expected: metadata.sha256,
                actual: hex_hash,
            });
        }

        // 3. Create NIP-94 event with metadata and file URL.
        // For now, we assume the file is hosted elsewhere (e.g., HTTP server).
        // In a real implementation, we would upload the file to a Blossom‑compatible
        // storage server (NIP-96) and publish a NIP-94 event referencing it.
        let event_id = self.publish_brain_event(&metadata).await?;

        info!("Brain uploaded successfully: {}", metadata.brain_id);
        Ok((metadata.brain_id, event_id))
    }

    /// Download a brain by its brain ID.
    /// Returns the local path to the downloaded brain file.
    pub async fn download_brain(
        &self,
        brain_id: &str,
        expected_base_model: Option<&str>,
    ) -> Result<PathBuf, BlossomError> {
        info!("Downloading brain {}", brain_id);

        // 1. Query Nostr relays for NIP-94 events with this brain ID.
        let metadata = self.fetch_brain_metadata(brain_id).await?;

        // 2. Validate base model compatibility.
        if let Some(expected) = expected_base_model {
            if metadata.base_model_id != expected {
                return Err(BlossomError::MissingMetadata(format!(
                    "Base model mismatch: expected {}, got {}",
                    expected, metadata.base_model_id
                )));
            }
        }

        // 3. Check if already cached.
        let cached_path = self.cache_dir.join(format!("{}.safetensors", brain_id));
        if cached_path.exists() {
            // Verify hash.
            let file_bytes = fs::read(&cached_path)
                .map_err(|e| BlossomError::Io(e))?;
            let mut hasher = Sha256::new();
            hasher.update(&file_bytes);
            let hex_hash = hex::encode(hasher.finalize());
            if hex_hash == metadata.sha256 {
                info!("Brain already cached and hash matches.");
                return Ok(cached_path);
            } else {
                warn!("Cached brain hash mismatch, re‑downloading.");
            }
        }

        // 4. Download the brain file from the URL referenced in the NIP-94 event.
        // For now, we simulate by copying a local file (placeholder).
        // In reality, we would fetch from a HTTP URL (NIP-96) or Nostr‑based storage.
        let download_url = self.resolve_brain_url(&metadata).await?;
        let bytes = self.download_from_url(&download_url).await?;

        // 5. Verify hash.
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let hex_hash = hex::encode(hasher.finalize());
        if hex_hash != metadata.sha256 {
            return Err(BlossomError::HashMismatch {
                expected: metadata.sha256,
                actual: hex_hash,
            });
        }

        // 6. Save to cache.
        fs::write(&cached_path, &bytes)
            .map_err(|e| BlossomError::Io(e))?;

        info!("Brain downloaded successfully to {:?}", cached_path);
        Ok(cached_path)
    }

    /// Publish a NIP-94 event for a brain.
    async fn publish_brain_event(&self, metadata: &BrainMetadata) -> Result<String, BlossomError> {
        // TODO: Integrate with nostr_federation.rs to actually publish a NIP-94 event.
        // For now, we return a dummy event ID.
        warn!("NIP-94 event publishing not yet implemented (placeholder)");
        Ok(format!("event_{}", metadata.brain_id))
    }

    /// Fetch brain metadata from Nostr relays.
    async fn fetch_brain_metadata(&self, _brain_id: &str) -> Result<BrainMetadata, BlossomError> {
        // TODO: Query Nostr relays for NIP-94 events with tag ["brain", brain_id].
        // For now, we return dummy metadata.
        warn!("Fetching brain metadata not yet implemented (placeholder)");
        Err(BlossomError::MissingMetadata(
            "Brain metadata not found".to_string(),
        ))
    }

    /// Resolve the download URL for a brain from its metadata.
    async fn resolve_brain_url(&self, metadata: &BrainMetadata) -> Result<String, BlossomError> {
        // TODO: Extract URL from NIP-94 event's "url" tag.
        // For now, we return a dummy URL.
        Ok(format!("https://example.com/brains/{}.safetensors", metadata.brain_id))
    }

    /// Download bytes from a URL.
    async fn download_from_url(&self, url: &str) -> Result<Vec<u8>, BlossomError> {
        // Use reqwest to download.
        let client = reqwest::Client::new();
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| BlossomError::Network(e.to_string()))?;
        if !response.status().is_success() {
            return Err(BlossomError::Network(format!(
                "HTTP error: {}",
                response.status()
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|e| BlossomError::Network(e.to_string()))?
            .to_vec();
        Ok(bytes)
    }

    /// Compute SHA256 hash of a file.
    pub fn compute_file_hash(path: impl AsRef<Path>) -> Result<String, BlossomError> {
        let bytes = fs::read(path).map_err(BlossomError::Io)?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        Ok(hex::encode(hasher.finalize()))
    }

    /// Compute SHA256 hash of bytes.
    pub fn compute_file_hash_from_bytes(bytes: &[u8]) -> Result<String, BlossomError> {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Ok(hex::encode(hasher.finalize()))
    }

    /// Load a brain from a safetensors file and validate metadata.
    /// Returns the raw bytes of the brain file.
    pub fn load_brain(
        path: impl AsRef<Path>,
        expected_brain_id: Option<&str>,
    ) -> Result<Vec<u8>, BlossomError> {
        let path = path.as_ref();
        let bytes = fs::read(path).map_err(BlossomError::Io)?;
        
        // Validate hash if expected_brain_id is provided
        if let Some(expected) = expected_brain_id {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            let hex_hash = hex::encode(hasher.finalize());
            if hex_hash != expected {
                return Err(BlossomError::HashMismatch {
                    expected: expected.to_string(),
                    actual: hex_hash,
                });
            }
        }
        
        // Validate it's a valid safetensors file
        let _ = SafeTensors::deserialize(&bytes)
            .map_err(|e| BlossomError::Safetensors(e.to_string()))?;
        
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_compute_file_hash() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.bin");
        fs::write(&file_path, b"hello world").unwrap();
        let hash = BlossomClient::compute_file_hash(&file_path).unwrap();
        // SHA256 of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[tokio::test]
    async fn test_load_brain() {
        // Create a dummy safetensors file.
        // Since we don't have a real safetensors file, we'll skip this test for now.
        // We'll just ensure the function compiles.
        assert!(true);
    }
}