// src/persistence.rs
// Pure Rust Redb database and state persistence for PC weights

use crate::openai_proxy::calibration::CalibrationStore;
use crate::types::{CognitiveDictionary, InvestigationNote, WorkflowMemoryNote};
use candle_core::Error as CandleError;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("Database error: {0}")]
    DatabaseError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Candle error: {0}")]
    CandleError(#[from] CandleError),
}

/// Represents a stored PC level weight matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCLevelWeights {
    pub level_index: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    pub pubkey: String,
    pub reputation_score: f64,
    pub zaps_received: i64,
    pub last_seen: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaHistory {
    pub id: String,
    pub author_pubkey: String,
    pub free_energy_drop: f64,
    pub applied_locally: bool,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheEntryDB {
    pub id: i64,
    pub prompt_hash: String,
    pub prompt_text: String,
    pub response_json: String,
    pub embedding: Vec<u8>,
    pub access_count: i64,
    pub last_accessed: i64,
}

// Redb Tables Definition
const PC_LEVEL_WEIGHTS: TableDefinition<u64, &[u8]> = TableDefinition::new("pc_level_weights");
const PEERS: TableDefinition<&str, &[u8]> = TableDefinition::new("peers");
const DELTA_HISTORY: TableDefinition<&str, &[u8]> = TableDefinition::new("delta_history");
const SEMANTIC_CACHE: TableDefinition<u64, &[u8]> = TableDefinition::new("semantic_cache");
const STUDIED_DOCUMENTS: TableDefinition<&str, &[u8]> = TableDefinition::new("studied_documents");
const THOUGHT_DECODER: TableDefinition<u64, &[u8]> = TableDefinition::new("thought_decoder");
const COGNITIVE_DICTIONARY: TableDefinition<u64, &[u8]> =
    TableDefinition::new("cognitive_dictionary");
const CALIBRATION_STORE: TableDefinition<u64, &[u8]> = TableDefinition::new("calibration_store");
const INVESTIGATION_NOTES: TableDefinition<u64, &[u8]> = TableDefinition::new("investigation_notes");
const WORKFLOW_MEMORY_NOTES: TableDefinition<u64, &[u8]> = TableDefinition::new("workflow_memory_notes");

/// Database manager for PC persistence using pure Rust Redb
pub struct PCPersistence {
    db: Arc<Database>,
}

impl PCPersistence {
    pub async fn new(db_path: &str) -> Result<Self, PersistenceError> {
        let path = std::path::Path::new(db_path);
        if let Some(parent) = path.parent() {
            if !parent.exists() && !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(PersistenceError::IoError)?;
            }
        }

        let db =
            Database::create(path).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;

        // Initialize tables
        let write_txn = db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let _ = write_txn.open_table(PC_LEVEL_WEIGHTS);
            let _ = write_txn.open_table(PEERS);
            let _ = write_txn.open_table(DELTA_HISTORY);
            let _ = write_txn.open_table(SEMANTIC_CACHE);
            let _ = write_txn.open_table(STUDIED_DOCUMENTS);
            let _ = write_txn.open_table(THOUGHT_DECODER);
            let _ = write_txn.open_table(COGNITIVE_DICTIONARY);
            let _ = write_txn.open_table(CALIBRATION_STORE);
            let _ = write_txn.open_table(INVESTIGATION_NOTES);
            let _ = write_txn.open_table(WORKFLOW_MEMORY_NOTES);
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;

        info!(
            "PC persistence database initialized at {} using Redb",
            db_path
        );
        Ok(Self { db: Arc::new(db) })
    }

    // ========== Weights Methods ==========

    pub async fn save_level_weights(&self, level: &PCLevelWeights) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(&level)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(PC_LEVEL_WEIGHTS)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(level.level_index as u64, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;

        debug!("Saved weights for level {}", level.level_index);
        Ok(())
    }

    pub async fn load_level_weights(
        &self,
        level_index: usize,
    ) -> Result<Option<PCLevelWeights>, PersistenceError> {
        let read_txn = self
            .db
            .as_ref()
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table: redb::ReadOnlyTable<u64, &[u8]> = match read_txn.open_table(PC_LEVEL_WEIGHTS) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        if let Some(data) = table
            .get(level_index as u64)
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let w: PCLevelWeights = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            Ok(Some(w))
        } else {
            Ok(None)
        }
    }

    pub async fn has_any_weights(&self) -> Result<bool, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        if let Ok(table) = read_txn.open_table(PC_LEVEL_WEIGHTS) {
            Ok(table
                .iter()
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
                .next()
                .is_some())
        } else {
            Ok(false)
        }
    }

    pub async fn load_all_levels(&self) -> Result<Vec<PCLevelWeights>, PersistenceError> {
        let read_txn = self
            .db
            .as_ref()
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table: redb::ReadOnlyTable<u64, &[u8]> = match read_txn.open_table(PC_LEVEL_WEIGHTS) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut levels = Vec::new();
        for result in table
            .iter()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let (_, data) = result.map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            let w: PCLevelWeights = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            levels.push(w);
        }

        levels.sort_by_key(|l| l.level_index);
        Ok(levels)
    }

    pub async fn clear_all(&self) -> Result<(), PersistenceError> {
        let write_txn = self
            .db
            .as_ref()
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(PC_LEVEL_WEIGHTS)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            // Clear all entries using retain_in with full range
            table
                .retain_in(0u64..u64::MAX, |_, _| false)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // ========== Peer Methods ==========

    pub async fn save_peer(&self, peer: &Peer) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(&peer)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(PEERS)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(peer.pubkey.as_str(), serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;

        debug!("Saved peer {}", peer.pubkey);
        Ok(())
    }

    pub async fn load_peer(&self, pubkey: &str) -> Result<Option<Peer>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(PEERS) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        if let Some(data) = table
            .get(pubkey)
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let peer: Peer = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            Ok(Some(peer))
        } else {
            Ok(None)
        }
    }

    // ========== Thought Decoder Methods ==========

    // 🔴 FIX: Signature updated to save the fused w_gate_stack
    pub async fn save_decoder(
        &self,
        w_gate_stack: &[f32],
        w_vocab: &[f32],
    ) -> Result<(), PersistenceError> {
        let gate_blob = bincode::serialize(&w_gate_stack)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let vocab_blob = bincode::serialize(&w_vocab)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;

        let tuple = (gate_blob, vocab_blob);
        let serialized = bincode::serialize(&tuple)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;

        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(THOUGHT_DECODER)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(1u64, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // 🔴 FIX: Signature updated to return the fused matrix
    pub async fn load_decoder(&self) -> Result<Option<(Vec<f32>, Vec<f32>)>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(THOUGHT_DECODER) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };

        if let Some(data) = table
            .get(1u64)
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            // Support legacy 3-tuple fallback if users upgrade without deleting DB
            if let Ok((gate_blob, vocab_blob)) =
                bincode::deserialize::<(Vec<u8>, Vec<u8>)>(data.value())
            {
                let gate = bincode::deserialize(&gate_blob).unwrap_or_default();
                let vocab = bincode::deserialize(&vocab_blob).unwrap_or_default();
                return Ok(Some((gate, vocab)));
            } else if let Ok((_u, _h, _v_blob)) =
                bincode::deserialize::<(Vec<u8>, Vec<u8>, Vec<u8>)>(data.value())
            {
                // If old DB schema, return None so it gets reset cleanly
                tracing::warn!("Old decoder schema detected. Wiping it to allow clean upgrade.");
                return Ok(None);
            }
            Ok(None)
        } else {
            Ok(None)
        }
    }

    // ========== Cognitive Dictionary Methods ==========

    pub async fn save_dictionary(
        &self,
        dict: &CognitiveDictionary,
    ) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(dict)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(COGNITIVE_DICTIONARY)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(1u64, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_dictionary(&self) -> Result<Option<CognitiveDictionary>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(COGNITIVE_DICTIONARY) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        if let Some(data) = table
            .get(1u64)
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let dict: CognitiveDictionary = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            Ok(Some(dict))
        } else {
            Ok(None)
        }
    }

    // ========== Calibration Store Methods ==========

    pub async fn save_calibration_store(
        &self,
        store: &CalibrationStore,
    ) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(store)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(CALIBRATION_STORE)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(1u64, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_calibration_store(
        &self,
    ) -> Result<Option<CalibrationStore>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(CALIBRATION_STORE) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        if let Some(data) = table
            .get(1u64)
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let store: CalibrationStore = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            Ok(Some(store))
        } else {
            Ok(None)
        }
    }

    // ========== Studied Document Tracking ==========

    pub async fn is_document_studied(
        &self,
        path: &str,
        content_hash: &str,
    ) -> Result<bool, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(STUDIED_DOCUMENTS) {
            Ok(t) => t,
            Err(_) => return Ok(false),
        };

        if let Some(data) = table
            .get(path)
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let stored_hash = String::from_utf8(data.value().to_vec()).unwrap_or_default();
            Ok(stored_hash == content_hash)
        } else {
            Ok(false)
        }
    }

    pub async fn mark_document_as_studied(
        &self,
        path: &str,
        content_hash: &str,
    ) -> Result<(), PersistenceError> {
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(STUDIED_DOCUMENTS)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(path, content_hash.as_bytes())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // ========== Semantic Cache Storage Methods ==========

    pub async fn save_semantic_cache_entry(
        &self,
        entry: &SemanticCacheEntryDB,
    ) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(entry)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(SEMANTIC_CACHE)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(entry.id as u64, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_semantic_cache_entries(
        &self,
    ) -> Result<Vec<SemanticCacheEntryDB>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(SEMANTIC_CACHE) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut entries = Vec::new();
        for result in table
            .iter()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let (_, data) = result.map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            let entry: SemanticCacheEntryDB = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            entries.push(entry);
        }

        entries.sort_by_key(|e| std::cmp::Reverse(e.last_accessed));
        Ok(entries)
    }

    pub async fn clear_semantic_cache(&self) -> Result<(), PersistenceError> {
        let write_txn = self
            .db
            .as_ref()
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(SEMANTIC_CACHE)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            // Clear all entries using retain_in with full range
            table
                .retain_in(0u64..u64::MAX, |_, _| false)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn save_investigation_note(
        &self,
        note: &InvestigationNote,
    ) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(note)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(INVESTIGATION_NOTES)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(note.id, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_investigation_notes(
        &self,
    ) -> Result<Vec<InvestigationNote>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(INVESTIGATION_NOTES) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut notes = Vec::new();
        for result in table
            .iter()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let (_, data) = result.map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            let note: InvestigationNote = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            notes.push(note);
        }

        notes.sort_by_key(|note| std::cmp::Reverse(note.updated_at));
        Ok(notes)
    }

    pub async fn save_workflow_memory_note(
        &self,
        note: &WorkflowMemoryNote,
    ) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(note)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self
            .db
            .begin_write()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(WORKFLOW_MEMORY_NOTES)
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table
                .insert(note.id, serialized.as_slice())
                .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_workflow_memory_notes(
        &self,
    ) -> Result<Vec<WorkflowMemoryNote>, PersistenceError> {
        let read_txn = self
            .db
            .begin_read()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(WORKFLOW_MEMORY_NOTES) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };

        let mut notes = Vec::new();
        for result in table
            .iter()
            .map_err(|e| PersistenceError::DatabaseError(e.to_string()))?
        {
            let (_, data) = result.map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            let note: WorkflowMemoryNote = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            notes.push(note);
        }

        notes.sort_by_key(|note| std::cmp::Reverse(note.updated_at));
        Ok(notes)
    }
}

#[cfg(test)]
mod persistence_integrity_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    // Helper to get a unique temp DB path for tests
    fn temp_db_path() -> String {
        std::env::temp_dir()
            .join(format!(
                "neurofed_test_db_{}.redb",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
            .to_string_lossy()
            .to_string()
    }

    #[tokio::test]
    async fn test_brain_weights_exact_serialization() {
        let db_path = temp_db_path();
        let persistence = PCPersistence::new(&db_path)
            .await
            .expect("Failed to init DB");

        // 1. Create a mock layer of the Brain
        let mock_weights = vec![0.123, -0.456, 0.789, 1.0, -1.0, 0.0];
        let original_level = PCLevelWeights {
            level_index: 2,
            input_dim: 3,
            output_dim: 2,
            weights: mock_weights.clone(),
            updated_at: 1234567890,
        };

        // 2. Save it to disk
        persistence
            .save_level_weights(&original_level)
            .await
            .expect("Save failed");

        // 3. Load it back
        let loaded_level = persistence
            .load_level_weights(2)
            .await
            .expect("Load failed")
            .expect("Level not found in DB");

        // 4. Prove bit-for-bit accuracy (no float precision loss during serialization)
        assert_eq!(original_level.level_index, loaded_level.level_index);
        assert_eq!(original_level.input_dim, loaded_level.input_dim);
        assert_eq!(original_level.output_dim, loaded_level.output_dim);

        for (orig, loaded) in original_level
            .weights
            .iter()
            .zip(loaded_level.weights.iter())
        {
            assert_eq!(
                orig, loaded,
                "Weight corruption detected during DB save/load!"
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_semantic_cache_storage() {
        let db_path = temp_db_path();
        let persistence = PCPersistence::new(&db_path).await.unwrap();

        let cache_entry = SemanticCacheEntryDB {
            id: 42,
            prompt_hash: "abcdef123".to_string(),
            prompt_text: "{\"test\":\"data\"}".to_string(),
            response_json: "{\"response\":\"ok\"}".to_string(),
            embedding: vec![0, 1, 2, 3], // Mock byte serialization
            access_count: 5,
            last_accessed: 999999,
        };

        persistence
            .save_semantic_cache_entry(&cache_entry)
            .await
            .unwrap();
        let loaded = persistence.load_semantic_cache_entries().await.unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].prompt_hash, "abcdef123");
        assert_eq!(loaded[0].access_count, 5);

        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_investigation_note_storage() {
        let db_path = temp_db_path();
        let persistence = PCPersistence::new(&db_path).await.unwrap();

        let note = InvestigationNote {
            id: 7,
            query: "investigate architecture drift".into(),
            goal: "find runtime and doc drift".into(),
            summary: "Runtime path is narrower than docs imply.".into(),
            evidence_summary: "Compared main startup path with documented modules.".into(),
            open_questions: vec!["Which module should be integrated next?".into()],
            plan_steps: vec!["inspect runtime".into(), "compare docs".into()],
            constraints: vec!["Collect evidence before concluding".into()],
            assumptions: vec!["Some modules are placeholders".into()],
            embedding: vec![0.1, 0.2, 0.3],
            updated_at: 42,
        };

        persistence.save_investigation_note(&note).await.unwrap();
        let loaded = persistence.load_investigation_notes().await.unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], note);

        let _ = std::fs::remove_file(db_path);
    }

    #[tokio::test]
    async fn test_workflow_memory_note_storage() {
        let db_path = temp_db_path();
        let persistence = PCPersistence::new(&db_path).await.unwrap();

        let note = WorkflowMemoryNote {
            id: 9,
            intent: crate::types::AssistantIntent::CodeTask,
            query: "fix parser bug".into(),
            goal: "fix parser bug".into(),
            summary: "Adjusted parser edge-case handling.".into(),
            deliverables: vec!["change plan".into(), "verification summary".into()],
            verification_checks: vec!["run cargo build".into()],
            verification_summary: "cargo build passed".into(),
            constraints: vec!["preserve behavior".into()],
            assumptions: vec!["parser has existing tests".into()],
            embedding: vec![0.2, 0.3, 0.4],
            updated_at: 77,
        };

        persistence.save_workflow_memory_note(&note).await.unwrap();
        let loaded = persistence.load_workflow_memory_notes().await.unwrap();

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], note);

        let _ = std::fs::remove_file(db_path);
    }
}
