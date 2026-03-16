// src/brain_manager.rs
// Brain manager for exporting, importing, and merging PC-brains (Federated Averaging).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::pc_hierarchy::PredictiveCoding;
use crate::persistence::PCLevelWeights;

/// Errors that can occur in brain management.
#[derive(Debug, thiserror::Error)]
pub enum BrainManagerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Invalid brain: {0}")]
    InvalidBrain(String),
    #[error("Base model mismatch: expected {expected}, got {actual}")]
    BaseModelMismatch { expected: String, actual: String },
    #[error("Brain not found: {0}")]
    BrainNotFound(String),
}

use crate::config::BrainSharingConfig;

/// A bundled export of a brain's weights and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainBundle {
    pub brain_id: String,
    pub base_model_id: String,
    pub embedding_dim: usize,
    pub n_levels: usize,
    pub levels: Vec<PCLevelWeights>,
    pub created_at: u64,
}

/// Manages brain sharing and compatibility via standard files.
pub struct BrainManager {
    config: BrainSharingConfig,
    /// Map from brain ID to local path.
    pub brains: HashMap<String, PathBuf>,
}

impl BrainManager {
    /// Create a new BrainManager.
    pub fn new(
        config: BrainSharingConfig,
        _nostr: Arc<crate::nostr_federation::NostrFederation>,
    ) -> Result<Self, BrainManagerError> {
        std::fs::create_dir_all(&config.brain_storage_dir)?;
        Ok(Self {
            config,
            brains: HashMap::new(),
        })
    }

    /// 🔴 EXPORT: Save the current PC‑brain to a `.neurobrain` file.
    pub async fn export_local_brain(
        &mut self,
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        description: &str,
    ) -> Result<PathBuf, BrainManagerError> {
        info!("Exporting brain: {}", description);

        let pc = pc_hierarchy.read().await;
        let levels = pc
            .get_level_weights()
            .map_err(|e| BrainManagerError::InvalidBrain(e.to_string()))?;
        let embedding_dim = pc.config.dim_per_level.first().cloned().unwrap_or(0);
        let n_levels = pc.config.n_levels;
        drop(pc);

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let brain_id = format!("brain_{}", timestamp);

        let bundle = BrainBundle {
            brain_id: brain_id.clone(),
            base_model_id: self.config.base_model_id.clone(),
            embedding_dim,
            n_levels,
            levels,
            created_at: timestamp,
        };

        let serialized = bincode::serialize(&bundle)
            .map_err(|e| BrainManagerError::Serialization(e.to_string()))?;

        let filename = format!("{}.neurobrain", brain_id);
        let local_path = self.config.brain_storage_dir.join(&filename);

        std::fs::write(&local_path, &serialized)?;
        self.brains.insert(brain_id.clone(), local_path.clone());

        info!("✅ Brain successfully exported to: {:?}", local_path);
        Ok(local_path)
    }

    /// 🔴 IMPORT & MERGE (Federated Averaging): Averages an external brain into yours.
    pub async fn import_and_merge_brain(
        &self,
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        import_path: &PathBuf,
    ) -> Result<(), BrainManagerError> {
        info!("Importing brain from {:?}", import_path);

        let bytes = std::fs::read(import_path)?;
        let bundle: BrainBundle = bincode::deserialize(&bytes)
            .map_err(|e| BrainManagerError::Serialization(e.to_string()))?;

        let mut pc = pc_hierarchy.write().await;

        // 1. Compatibility Checks!
        let local_dim = *pc.config.dim_per_level.first().unwrap_or(&0);
        if bundle.embedding_dim != local_dim {
            return Err(BrainManagerError::BaseModelMismatch {
                expected: format!("dim {}", local_dim),
                actual: format!("dim {}", bundle.embedding_dim),
            });
        }

        if bundle.n_levels != pc.config.n_levels {
            return Err(BrainManagerError::InvalidBrain(format!(
                "Level mismatch: local has {} levels, imported has {}.",
                pc.config.n_levels, bundle.n_levels
            )));
        }

        // 2. Federated Averaging (Merge)
        let local_weights = pc
            .get_level_weights()
            .map_err(|e| BrainManagerError::InvalidBrain(e.to_string()))?;

        for (l, imported_level) in bundle.levels.iter().enumerate() {
            if let Some(local_level) = local_weights.get(l) {
                // Ensure matrix sizes actually match
                if local_level.weights.len() != imported_level.weights.len() {
                    return Err(BrainManagerError::InvalidBrain(
                        "Weight matrix size mismatch inside level".to_string(),
                    ));
                }

                let mut merged_weights = vec![0.0; local_level.weights.len()];
                for i in 0..local_level.weights.len() {
                    // Mathematic average of the two brains
                    merged_weights[i] = (local_level.weights[i] + imported_level.weights[i]) / 2.0;
                }

                // Write the merged knowledge back into the live network
                pc.levels[l]
                    .set_weights_from_vec(merged_weights)
                    .map_err(|e| BrainManagerError::InvalidBrain(e.to_string()))?;
            }
        }

        info!("✅ Successfully merged external brain into local hierarchy!");
        Ok(())
    }
}

#[cfg(test)]
mod brain_sharing_tests {
    use super::*;
    use crate::config::NostrConfig;
    use crate::pc_hierarchy::{PCConfig, PredictiveCoding};
    use candle_core::Device;

    #[tokio::test]
    async fn test_brain_export_and_federated_merge() -> Result<(), Box<dyn std::error::Error>> {
        let config = BrainSharingConfig::default();
        let nostr = Arc::new(crate::nostr_federation::NostrFederation::new(
            NostrConfig::default(),
        ));
        let mut manager = BrainManager::new(config.clone(), nostr)?;

        let pc_config = PCConfig::new(2, vec![4, 2]);
        let device = Device::Cpu;

        // Brain A (Our Local Brain)
        let pc_a = PredictiveCoding::new_with_device(pc_config.clone(), device.clone())?;
        let arc_a = Arc::new(RwLock::new(pc_a));

        // Brain B (Another person's brain)
        let mut pc_b = PredictiveCoding::new_with_device(pc_config.clone(), device.clone())?;

        // Modify Brain B so we can track the merge mathematically
        let modified_weights = vec![10.0; 8]; // 4x2 matrix
        pc_b.levels[0].set_weights_from_vec(modified_weights.clone())?;
        let arc_b = Arc::new(RwLock::new(pc_b));

        // 1. Export Brain B to a file
        let export_path = manager
            .export_local_brain(arc_b.clone(), "test_brain")
            .await?;

        // Capture Brain A's original weights for comparison
        let orig_a = arc_a.read().await.levels[0].weights.to_vec2::<f32>()?[0][0];

        // 2. Import Brain B and merge into Brain A
        manager
            .import_and_merge_brain(arc_a.clone(), &export_path)
            .await?;

        // 3. Verify Federated Averaging Math
        let new_a = arc_a.read().await.levels[0].weights.to_vec2::<f32>()?[0][0];

        // The new weight should be exactly the average of (orig_a + 10.0) / 2.0
        let expected = (orig_a + 10.0) / 2.0;
        assert!(
            (new_a - expected).abs() < 0.001,
            "Merge failed! Expected {}, got {}",
            expected,
            new_a
        );

        // Cleanup
        let _ = std::fs::remove_file(export_path);
        Ok(())
    }

    #[tokio::test]
    async fn test_brain_rejects_incompatible_dimensions() -> Result<(), Box<dyn std::error::Error>>
    {
        let config = BrainSharingConfig::default();
        let nostr = Arc::new(crate::nostr_federation::NostrFederation::new(
            NostrConfig::default(),
        ));
        let mut manager = BrainManager::new(config.clone(), nostr)?;

        // Brain A uses Llama (2048 dim)
        let pc_a =
            PredictiveCoding::new_with_device(PCConfig::new(2, vec![2048, 1024]), Device::Cpu)?;
        // Brain B uses Mistral (4096 dim)
        let pc_b =
            PredictiveCoding::new_with_device(PCConfig::new(2, vec![4096, 1024]), Device::Cpu)?;

        let arc_a = Arc::new(RwLock::new(pc_a));
        let arc_b = Arc::new(RwLock::new(pc_b));

        // 1. Export Brain B to a file
        let export_path = manager.export_local_brain(arc_b, "bad_brain").await?;

        // Attempting to merge a 4096 brain into a 2048 node MUST fail safely
        let result = manager.import_and_merge_brain(arc_a, &export_path).await;

        assert!(
            result.is_err(),
            "Manager allowed merging incompatible dimensions!"
        );
        if let Err(BrainManagerError::BaseModelMismatch { expected, actual }) = result {
            assert!(expected.contains("2048"));
            assert!(actual.contains("4096"));
        } else {
            panic!("Wrong error type returned.");
        }

        let _ = std::fs::remove_file(export_path);
        Ok(())
    }
}
