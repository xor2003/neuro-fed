// src/persistence.rs
// Pure Rust Redb database and state persistence for PC weights

use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tracing::{info, debug};
use candle_core::Error as CandleError;

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
        
        let db = Database::create(path).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        
        // Initialize tables
        let write_txn = db.begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let _ = write_txn.open_table(PC_LEVEL_WEIGHTS);
            let _ = write_txn.open_table(PEERS);
            let _ = write_txn.open_table(DELTA_HISTORY);
            let _ = write_txn.open_table(SEMANTIC_CACHE);
            let _ = write_txn.open_table(STUDIED_DOCUMENTS);
            let _ = write_txn.open_table(THOUGHT_DECODER);
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        
        info!("PC persistence database initialized at {} using Redb", db_path);
        Ok(Self { db: Arc::new(db) })
    }
    
    // ========== Weights Methods ==========

    pub async fn save_level_weights(&self, level: &PCLevelWeights) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(&level).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self.db.begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(PC_LEVEL_WEIGHTS).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table.insert(level.level_index as u64, serialized.as_slice()).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        
        debug!("Saved weights for level {}", level.level_index);
        Ok(())
    }
    
    pub async fn load_level_weights(&self, level_index: usize) -> Result<Option<PCLevelWeights>, PersistenceError> {
        let read_txn = self.db.as_ref().begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table: redb::ReadOnlyTable<u64, &[u8]> = match read_txn.open_table(PC_LEVEL_WEIGHTS) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        
        if let Some(data) = table.get(level_index as u64).map_err(|e| PersistenceError::DatabaseError(e.to_string()))? {
            let w: PCLevelWeights = bincode::deserialize(data.value()).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            Ok(Some(w))
        } else {
            Ok(None)
        }
    }

    pub async fn has_any_weights(&self) -> Result<bool, PersistenceError> {
        let read_txn = self.db.begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        if let Ok(table) = read_txn.open_table(PC_LEVEL_WEIGHTS) {
            Ok(table.iter().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?.next().is_some())
        } else {
            Ok(false)
        }
    }
    
    pub async fn load_all_levels(&self) -> Result<Vec<PCLevelWeights>, PersistenceError> {
        let read_txn = self.db.as_ref().begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table: redb::ReadOnlyTable<u64, &[u8]> = match read_txn.open_table(PC_LEVEL_WEIGHTS) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };
        
        let mut levels = Vec::new();
        for result in table.iter().map_err(|e| PersistenceError::DatabaseError(e.to_string()))? {
            let (_, data) = result.map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            let w: PCLevelWeights = bincode::deserialize(data.value()).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            levels.push(w);
        }
        
        levels.sort_by_key(|l| l.level_index);
        Ok(levels)
    }

    pub async fn clear_all(&self) -> Result<(), PersistenceError> {
        let write_txn = self.db.as_ref().begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(PC_LEVEL_WEIGHTS).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            // Clear all entries using retain_in with full range
            table.retain_in(0u64..u64::MAX, |_, _| false).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // ========== Peer Methods ==========

    pub async fn save_peer(&self, peer: &Peer) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(&peer).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self.db.begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(PEERS).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table.insert(peer.pubkey.as_str(), serialized.as_slice()).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        
        debug!("Saved peer {}", peer.pubkey);
        Ok(())
    }

    pub async fn load_peer(&self, pubkey: &str) -> Result<Option<Peer>, PersistenceError> {
        let read_txn = self.db.begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(PEERS) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        
        if let Some(data) = table.get(pubkey).map_err(|e| PersistenceError::DatabaseError(e.to_string()))? {
            let peer: Peer = bincode::deserialize(data.value()).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            Ok(Some(peer))
        } else {
            Ok(None)
        }
    }

    // ========== Thought Decoder Methods ==========

    pub async fn save_decoder(&self, w_update: &[f32], w_hidden: &[f32], w_vocab: &[f32]) -> Result<(), PersistenceError> {
        let u_blob = bincode::serialize(&w_update).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let h_blob = bincode::serialize(&w_hidden).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let v_blob = bincode::serialize(&w_vocab).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        
        let tuple = (u_blob, h_blob, v_blob);
        let serialized = bincode::serialize(&tuple).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        
        let write_txn = self.db.begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(THOUGHT_DECODER).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table.insert(1u64, serialized.as_slice()).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_decoder(&self) -> Result<Option<(Vec<f32>, Vec<f32>, Vec<f32>)>, PersistenceError> {
        let read_txn = self.db.begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(THOUGHT_DECODER) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        };
        
        if let Some(data) = table.get(1u64).map_err(|e| PersistenceError::DatabaseError(e.to_string()))? {
            let (u_blob, h_blob, v_blob): (Vec<u8>, Vec<u8>, Vec<u8>) = bincode::deserialize(data.value())
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
                
            let u = bincode::deserialize(&u_blob).unwrap_or_default();
            let h = bincode::deserialize(&h_blob).unwrap_or_default();
            let v = bincode::deserialize(&v_blob).unwrap_or_default();
            Ok(Some((u, h, v)))
        } else {
            Ok(None)
        }
    }

    // ========== Studied Document Tracking ==========

    pub async fn is_document_studied(&self, path: &str, content_hash: &str) -> Result<bool, PersistenceError> {
        let read_txn = self.db.begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(STUDIED_DOCUMENTS) {
            Ok(t) => t,
            Err(_) => return Ok(false),
        };
        
        if let Some(data) = table.get(path).map_err(|e| PersistenceError::DatabaseError(e.to_string()))? {
            let stored_hash = String::from_utf8(data.value().to_vec()).unwrap_or_default();
            Ok(stored_hash == content_hash)
        } else {
            Ok(false)
        }
    }

    pub async fn mark_document_as_studied(&self, path: &str, content_hash: &str) -> Result<(), PersistenceError> {
        let write_txn = self.db.begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(STUDIED_DOCUMENTS).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table.insert(path, content_hash.as_bytes()).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    // ========== Semantic Cache Storage Methods ==========

    pub async fn save_semantic_cache_entry(&self, entry: &SemanticCacheEntryDB) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(entry).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        let write_txn = self.db.begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(SEMANTIC_CACHE).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            table.insert(entry.id as u64, serialized.as_slice()).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }

    pub async fn load_semantic_cache_entries(&self) -> Result<Vec<SemanticCacheEntryDB>, PersistenceError> {
        let read_txn = self.db.begin_read().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        let table = match read_txn.open_table(SEMANTIC_CACHE) {
            Ok(t) => t,
            Err(_) => return Ok(Vec::new()),
        };
        
        let mut entries = Vec::new();
        for result in table.iter().map_err(|e| PersistenceError::DatabaseError(e.to_string()))? {
            let (_, data) = result.map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            let entry: SemanticCacheEntryDB = bincode::deserialize(data.value()).map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            entries.push(entry);
        }
        
        entries.sort_by_key(|e| std::cmp::Reverse(e.last_accessed));
        Ok(entries)
    }

    pub async fn clear_semantic_cache(&self) -> Result<(), PersistenceError> {
        let write_txn = self.db.as_ref().begin_write().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        {
            let mut table = write_txn.open_table(SEMANTIC_CACHE).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
            // Clear all entries using retain_in with full range
            table.retain_in(0u64..u64::MAX, |_, _| false).map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        }
        write_txn.commit().map_err(|e| PersistenceError::DatabaseError(e.to_string()))?;
        Ok(())
    }
}