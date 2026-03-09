// src/persistence.rs
// SQLite database and state persistence for PC weights

use sqlx::{Sqlite, Pool, Row};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, debug};
use candle_core::Error as CandleError;
use chrono;

#[derive(Debug, Error)]
pub enum PersistenceError {
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
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
    /// Flattened weight matrix (row-major) as f32 vector
    pub weights: Vec<f32>,
    pub updated_at: i64,
}

/// Represents a Nostr peer in the trust graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Peer {
    pub pubkey: String,
    pub reputation_score: f64,
    pub zaps_received: i64,
    pub last_seen: i64,
}

/// Represents a delta update in the history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaHistory {
    pub id: String,
    pub author_pubkey: String,
    pub free_energy_drop: f64,
    pub applied_locally: bool,
    pub timestamp: i64,
}

/// Represents a semantic cache entry for persistence
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

/// Database manager for PC persistence
pub struct PCPersistence {
    pool: Pool<Sqlite>,
}

impl PCPersistence {
    /// Initialize database at given path (creates if not exists)
    pub async fn new(db_path: &str) -> Result<Self, PersistenceError> {
        use sqlx::sqlite::{SqlitePoolOptions, SqliteConnectOptions};
        use std::str::FromStr;
        
        // For in-memory database, use special handling
        let options = if db_path == ":memory:" {
            // For in-memory databases with connection pooling, we need to use a shared cache
            // so all connections in the pool share the same in-memory database
            SqliteConnectOptions::new()
                .filename("file::memory:?cache=shared")
        } else {
            // Ensure parent directory exists for file-based databases
            if let Some(parent) = std::path::Path::new(db_path).parent() {
                if !parent.exists() && parent.to_string_lossy() != "" {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| PersistenceError::IoError(e))?;
                }
            }
            // Use sqlite: prefix without // to handle paths correctly
            let connection_string = format!("sqlite:{}", db_path);
            SqliteConnectOptions::from_str(&connection_string)
                .map_err(PersistenceError::DatabaseError)?
                .create_if_missing(true)
        };
        
        let pool = SqlitePoolOptions::new()
            .connect_with(options)
            .await?;
        
        // Enable WAL mode for better concurrency (only for file-based databases)
        if db_path != ":memory:" {
            sqlx::query("PRAGMA journal_mode=WAL;")
                .execute(&pool)
                .await?;
            sqlx::query("PRAGMA synchronous=NORMAL;")
                .execute(&pool)
                .await?;
        }
        
        // Create tables if they don't exist
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS pc_level_weights (
                level_index INTEGER PRIMARY KEY,
                input_dim INTEGER NOT NULL,
                output_dim INTEGER NOT NULL,
                weights BLOB NOT NULL,
                updated_at INTEGER NOT NULL
            )
            "#
        )
        .execute(&pool)
        .await?;
        
        // Create trust graph table for Nostr peers
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS peers (
                pubkey TEXT PRIMARY KEY,
                reputation_score REAL DEFAULT 0.5,
                zaps_received INTEGER DEFAULT 0,
                last_seen INTEGER
            )
            "#
        )
        .execute(&pool)
        .await?;
        
        // Create delta history table for Nostr gossip
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS delta_history (
                id TEXT PRIMARY KEY,
                author_pubkey TEXT,
                free_energy_drop REAL,
                applied_locally BOOLEAN,
                timestamp INTEGER
            )
            "#
        )
        .execute(&pool)
        .await?;
        
        // Create semantic cache storage table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash TEXT UNIQUE,
                prompt_text TEXT,
                response_json TEXT,
                embedding BLOB,
                access_count INTEGER DEFAULT 1,
                last_accessed INTEGER
            )
            "#
        )
        .execute(&pool)
        .await?;
        
        info!("PC persistence database initialized at {} with WAL mode", db_path);
        Ok(Self { pool })
    }
    
    /// Store or update weights for a specific level
    pub async fn save_level_weights(&self, level: &PCLevelWeights) -> Result<(), PersistenceError> {
        let serialized = bincode::serialize(&level.weights)
            .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
        
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO pc_level_weights 
            (level_index, input_dim, output_dim, weights, updated_at)
            VALUES (?, ?, ?, ?, ?)
            "#
        )
        .bind(level.level_index as i64)
        .bind(level.input_dim as i64)
        .bind(level.output_dim as i64)
        .bind(serialized)
        .bind(level.updated_at)
        .execute(&self.pool)
        .await?;
        
        info!("Saved weights for level {}", level.level_index);
        Ok(())
    }
    
    /// Load weights for a specific level
    pub async fn load_level_weights(&self, level_index: usize) -> Result<Option<PCLevelWeights>, PersistenceError> {
        let row = sqlx::query(
            r#"
            SELECT level_index, input_dim, output_dim, weights, updated_at
            FROM pc_level_weights
            WHERE level_index = ?
            "#
        )
        .bind(level_index as i64)
        .fetch_optional(&self.pool)
        .await?;
        
        match row {
            Some(row) => {
                let input_dim = row.get::<i64, _>("input_dim") as usize;
                let output_dim = row.get::<i64, _>("output_dim") as usize;
                let weights_blob: Vec<u8> = row.get("weights");
                let weights = bincode::deserialize(&weights_blob)
                    .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
                let updated_at = row.get("updated_at");
                
                Ok(Some(PCLevelWeights {
                    level_index,
                    input_dim,
                    output_dim,
                    weights,
                    updated_at,
                }))
            }
            None => Ok(None),
        }
    }
    
    /// Load all levels
    pub async fn load_all_levels(&self) -> Result<Vec<PCLevelWeights>, PersistenceError> {
        let rows = sqlx::query(
            r#"
            SELECT level_index, input_dim, output_dim, weights, updated_at
            FROM pc_level_weights
            ORDER BY level_index
            "#
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut levels = Vec::new();
        for row in rows {
            let level_index = row.get::<i64, _>("level_index") as usize;
            let input_dim = row.get::<i64, _>("input_dim") as usize;
            let output_dim = row.get::<i64, _>("output_dim") as usize;
            let weights_blob: Vec<u8> = row.get("weights");
            let weights = bincode::deserialize(&weights_blob)
                .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
            let updated_at = row.get("updated_at");
            
            levels.push(PCLevelWeights {
                level_index,
                input_dim,
                output_dim,
                weights,
                updated_at,
            });
        }
        
        Ok(levels)
    }
    
    /// Delete weights for a specific level
    pub async fn delete_level_weights(&self, level_index: usize) -> Result<(), PersistenceError> {
        sqlx::query(
            r#"
            DELETE FROM pc_level_weights WHERE level_index = ?
            "#
        )
        .bind(level_index as i64)
        .execute(&self.pool)
        .await?;
        
        info!("Deleted weights for level {}", level_index);
        Ok(())
    }
    
    /// Clear all stored weights
    pub async fn clear_all(&self) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM pc_level_weights")
            .execute(&self.pool)
            .await?;
        
        info!("Cleared all PC weights");
        Ok(())
    }

    // ========== Trust Graph (Peers) Methods ==========

    /// Save or update a peer in the trust graph
    pub async fn save_peer(&self, peer: &Peer) -> Result<(), PersistenceError> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO peers
            (pubkey, reputation_score, zaps_received, last_seen)
            VALUES (?, ?, ?, ?)
            "#
        )
        .bind(&peer.pubkey)
        .bind(peer.reputation_score)
        .bind(peer.zaps_received)
        .bind(peer.last_seen)
        .execute(&self.pool)
        .await?;
        
        info!("Saved peer: {}", peer.pubkey);
        Ok(())
    }

    /// Load a peer by public key
    pub async fn load_peer(&self, pubkey: &str) -> Result<Option<Peer>, PersistenceError> {
        let row = sqlx::query(
            r#"
            SELECT pubkey, reputation_score, zaps_received, last_seen
            FROM peers
            WHERE pubkey = ?
            "#
        )
        .bind(pubkey)
        .fetch_optional(&self.pool)
        .await?;
        
        match row {
            Some(row) => {
                Ok(Some(Peer {
                    pubkey: row.get("pubkey"),
                    reputation_score: row.get("reputation_score"),
                    zaps_received: row.get("zaps_received"),
                    last_seen: row.get("last_seen"),
                }))
            }
            None => Ok(None),
        }
    }

    /// Load all peers, optionally sorted by reputation
    pub async fn load_all_peers(&self, sort_by_reputation: bool) -> Result<Vec<Peer>, PersistenceError> {
        let query = if sort_by_reputation {
            "SELECT pubkey, reputation_score, zaps_received, last_seen FROM peers ORDER BY reputation_score DESC"
        } else {
            "SELECT pubkey, reputation_score, zaps_received, last_seen FROM peers"
        };
        
        let rows = sqlx::query(query)
            .fetch_all(&self.pool)
            .await?;
        
        let mut peers = Vec::new();
        for row in rows {
            peers.push(Peer {
                pubkey: row.get("pubkey"),
                reputation_score: row.get("reputation_score"),
                zaps_received: row.get("zaps_received"),
                last_seen: row.get("last_seen"),
            });
        }
        
        Ok(peers)
    }

    /// Update a peer's reputation score
    pub async fn update_peer_reputation(&self, pubkey: &str, reputation_score: f64) -> Result<(), PersistenceError> {
        sqlx::query(
            r#"
            UPDATE peers
            SET reputation_score = ?, last_seen = ?
            WHERE pubkey = ?
            "#
        )
        .bind(reputation_score)
        .bind(chrono::Utc::now().timestamp())
        .bind(pubkey)
        .execute(&self.pool)
        .await?;
        
        info!("Updated reputation for peer {}: {}", pubkey, reputation_score);
        Ok(())
    }

    /// Delete a peer from the trust graph
    pub async fn delete_peer(&self, pubkey: &str) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM peers WHERE pubkey = ?")
            .bind(pubkey)
            .execute(&self.pool)
            .await?;
        
        info!("Deleted peer: {}", pubkey);
        Ok(())
    }

    /// Clear all peers
    pub async fn clear_peers(&self) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM peers")
            .execute(&self.pool)
            .await?;
        
        info!("Cleared all peers");
        Ok(())
    }

    // ========== Delta History Methods ==========

    /// Save a delta history entry
    pub async fn save_delta(&self, delta: &DeltaHistory) -> Result<(), PersistenceError> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO delta_history
            (id, author_pubkey, free_energy_drop, applied_locally, timestamp)
            VALUES (?, ?, ?, ?, ?)
            "#
        )
        .bind(&delta.id)
        .bind(&delta.author_pubkey)
        .bind(delta.free_energy_drop)
        .bind(delta.applied_locally)
        .bind(delta.timestamp)
        .execute(&self.pool)
        .await?;
        
        info!("Saved delta: {} from {}", delta.id, delta.author_pubkey);
        Ok(())
    }

    /// Load a delta by ID
    pub async fn load_delta(&self, id: &str) -> Result<Option<DeltaHistory>, PersistenceError> {
        let row = sqlx::query(
            r#"
            SELECT id, author_pubkey, free_energy_drop, applied_locally, timestamp
            FROM delta_history
            WHERE id = ?
            "#
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        
        match row {
            Some(row) => {
                Ok(Some(DeltaHistory {
                    id: row.get("id"),
                    author_pubkey: row.get("author_pubkey"),
                    free_energy_drop: row.get("free_energy_drop"),
                    applied_locally: row.get("applied_locally"),
                    timestamp: row.get("timestamp"),
                }))
            }
            None => Ok(None),
        }
    }

    /// Load recent deltas, optionally filtered by author
    pub async fn load_recent_deltas(&self, limit: i64, author_pubkey: Option<&str>) -> Result<Vec<DeltaHistory>, PersistenceError> {
        let query = match author_pubkey {
            Some(_author) => {
                "SELECT id, author_pubkey, free_energy_drop, applied_locally, timestamp FROM delta_history WHERE author_pubkey = ? ORDER BY timestamp DESC LIMIT ?"
            }
            None => {
                "SELECT id, author_pubkey, free_energy_drop, applied_locally, timestamp FROM delta_history ORDER BY timestamp DESC LIMIT ?"
            }
        };
        
        let mut query_builder = sqlx::query(query);
        
        if let Some(author) = author_pubkey {
            query_builder = query_builder.bind(author);
        }
        
        query_builder = query_builder.bind(limit);
        
        let rows = query_builder.fetch_all(&self.pool).await?;
        
        let mut deltas = Vec::new();
        for row in rows {
            deltas.push(DeltaHistory {
                id: row.get("id"),
                author_pubkey: row.get("author_pubkey"),
                free_energy_drop: row.get("free_energy_drop"),
                applied_locally: row.get("applied_locally"),
                timestamp: row.get("timestamp"),
            });
        }
        
        Ok(deltas)
    }

    /// Delete a delta by ID
    pub async fn delete_delta(&self, id: &str) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM delta_history WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        
        info!("Deleted delta: {}", id);
        Ok(())
    }

    /// Clear all delta history
    pub async fn clear_deltas(&self) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM delta_history")
            .execute(&self.pool)
            .await?;
        
        info!("Cleared all delta history");
        Ok(())
    }

    // ========== Semantic Cache Storage Methods ==========

    /// Save a semantic cache entry to persistent storage
    pub async fn save_semantic_cache_entry(&self, entry: &SemanticCacheEntryDB) -> Result<(), PersistenceError> {
        sqlx::query(
            r#"
            INSERT OR REPLACE INTO semantic_cache
            (id, prompt_hash, prompt_text, response_json, embedding, access_count, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            "#
        )
        .bind(entry.id)
        .bind(&entry.prompt_hash)
        .bind(&entry.prompt_text)
        .bind(&entry.response_json)
        .bind(&entry.embedding)
        .bind(entry.access_count)
        .bind(entry.last_accessed)
        .execute(&self.pool)
        .await?;
        
        debug!("Saved semantic cache entry: {}", entry.prompt_hash);
        Ok(())
    }

    /// Load all semantic cache entries
    pub async fn load_semantic_cache_entries(&self) -> Result<Vec<SemanticCacheEntryDB>, PersistenceError> {
        let rows = sqlx::query(
            r#"
            SELECT id, prompt_hash, prompt_text, response_json, embedding, access_count, last_accessed
            FROM semantic_cache
            ORDER BY last_accessed DESC
            "#
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut entries = Vec::new();
        for row in rows {
            entries.push(SemanticCacheEntryDB {
                id: row.get("id"),
                prompt_hash: row.get("prompt_hash"),
                prompt_text: row.get("prompt_text"),
                response_json: row.get("response_json"),
                embedding: row.get("embedding"),
                access_count: row.get("access_count"),
                last_accessed: row.get("last_accessed"),
            });
        }
        
        Ok(entries)
    }

    /// Delete a semantic cache entry by ID
    pub async fn delete_semantic_cache_entry(&self, id: i64) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM semantic_cache WHERE id = ?")
            .bind(id)
            .execute(&self.pool)
            .await?;
        
        debug!("Deleted semantic cache entry: {}", id);
        Ok(())
    }

    /// Clear all semantic cache entries
    pub async fn clear_semantic_cache(&self) -> Result<(), PersistenceError> {
        sqlx::query("DELETE FROM semantic_cache")
            .execute(&self.pool)
            .await?;
        
        info!("Cleared all semantic cache entries");
        Ok(())
    }
}

/// Helper to convert candle Tensor to flattened vector
pub fn tensor_to_vec(tensor: &candle_core::Tensor) -> Result<Vec<f32>, PersistenceError> {
    let shape = tensor.shape();
    if shape.dims().len() != 2 {
        return Err(PersistenceError::SerializationError(
            "Expected 2D tensor".to_string()
        ));
    }
    let rows = shape.dims()[0];
    let cols = shape.dims()[1];
    let flat = tensor.flatten_all()?;
    let vec: Vec<f32> = flat.to_vec1()?;
    if vec.len() != rows * cols {
        return Err(PersistenceError::SerializationError(
            format!("Tensor size mismatch: expected {}, got {}", rows * cols, vec.len())
        ));
    }
    Ok(vec)
}

/// Helper to create tensor from flattened vector and shape
pub fn vec_to_tensor(vec: Vec<f32>, rows: usize, cols: usize, device: &candle_core::Device) -> Result<candle_core::Tensor, PersistenceError> {
    if vec.len() != rows * cols {
        return Err(PersistenceError::SerializationError(
            format!("Vector size mismatch: expected {}, got {}", rows * cols, vec.len())
        ));
    }
    let tensor = candle_core::Tensor::from_vec(vec, (rows, cols), device)
        .map_err(|e| PersistenceError::SerializationError(e.to_string()))?;
    Ok(tensor)
}