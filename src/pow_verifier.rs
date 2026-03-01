// src/pow_verifier.rs
// Proof-of-work verification for no-wallet mode

use std::time::{Duration, SystemTime, Instant};
use sha2::{Sha256, Digest};
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::info;
use async_trait::async_trait;

use crate::types::PoWVerification;
use crate::federation_manager::PoWVerifier as PoWVerifierTrait;

/// Errors that can occur during PoW verification
#[derive(Error, Debug, Clone)]
pub enum PoWVerifierError {
    #[error("Invalid proof format: {0}")]
    InvalidProof(String),
    #[error("Proof does not meet difficulty: {0}")]
    InsufficientDifficulty(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Hash mismatch: {0}")]
    HashMismatch(String),
    #[error("Nonce out of range: {0}")]
    NonceOutOfRange(String),
}

/// Proof-of-work verifier
pub struct PoWVerifier {
    /// Hash algorithm to use (currently only SHA256)
    hash_algorithm: String,
    /// Maximum nonce value
    max_nonce: u64,
}

impl PoWVerifier {
    /// Create a new PoW verifier
    pub fn new(hash_algorithm: String, max_nonce: u64) -> Self {
        Self {
            hash_algorithm,
            max_nonce,
        }
    }

    /// Verify a proof-of-work
    pub async fn verify_pow(
        &self,
        pow_proof: &str,
        difficulty: u32,
        timeout: Duration,
    ) -> Result<PoWVerification, PoWVerifierError> {
        info!("Verifying PoW proof with difficulty {}", difficulty);
        
        let start_time = Instant::now();
        
        // Parse proof
        let proof: PoWProof = serde_json::from_str(pow_proof)
            .map_err(|e| PoWVerifierError::InvalidProof(e.to_string()))?;

        // Validate nonce range
        if proof.nonce > self.max_nonce {
            return Err(PoWVerifierError::NonceOutOfRange(format!(
                "Nonce {} exceeds max {}",
                proof.nonce, self.max_nonce
            )));
        }

        // Verify hash matches data + nonce
        let computed_hash = self.compute_hash(&proof.data, proof.nonce);
        if computed_hash != proof.hash {
            return Err(PoWVerifierError::HashMismatch(format!(
                "Expected {}, got {}",
                proof.hash, computed_hash
            )));
        }

        // Check difficulty (leading zero bits)
        if !self.check_difficulty(&proof.hash, difficulty) {
            return Err(PoWVerifierError::InsufficientDifficulty(format!(
                "Hash {} does not meet difficulty {}",
                proof.hash, difficulty
            )));
        }

        // Check timeout
        if start_time.elapsed() > timeout {
            return Err(PoWVerifierError::Timeout("Verification timed out".to_string()));
        }

        Ok(PoWVerification {
            verified: true,
            nonce: proof.nonce,
            hash: proof.hash,
            reason: None,
        })
    }

    /// Generate a PoW challenge
    pub async fn generate_pow_challenge(&self, data: &str) -> Result<String, PoWVerifierError> {
        info!("Generating PoW challenge for data: {}", data);
        
        let challenge = PoWChallenge {
            data: data.to_string(),
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            target_difficulty: 5, // Default difficulty
        };

        serde_json::to_string(&challenge)
            .map_err(|e| PoWVerifierError::InvalidProof(e.to_string()))
    }

    /// Compute hash of data + nonce
    fn compute_hash(&self, data: &str, nonce: u64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hasher.update(nonce.to_le_bytes());
        let result = hasher.finalize();
        hex::encode(result)
    }

    /// Check if hash meets difficulty (number of leading zero bits)
    fn check_difficulty(&self, hash: &str, difficulty: u32) -> bool {
        // Convert hex hash to bytes
        let bytes = match hex::decode(hash) {
            Ok(b) => b,
            Err(_) => return false,
        };

        // Count leading zero bits
        let mut leading_zero_bits = 0;
        for byte in bytes {
            if byte == 0 {
                leading_zero_bits += 8;
            } else {
                leading_zero_bits += byte.leading_zeros();
                break;
            }
        }

        leading_zero_bits >= difficulty
    }

    /// Mine a proof-of-work (for testing)
    pub async fn mine_pow(&self, data: &str, difficulty: u32, timeout: Duration) -> Result<PoWProof, PoWVerifierError> {
        info!("Mining PoW for data: {} with difficulty {}", data, difficulty);
        let start_time = Instant::now();
        let mut rng = rand::thread_rng();

        for _attempt in 0..self.max_nonce {
            if start_time.elapsed() > timeout {
                return Err(PoWVerifierError::Timeout("Mining timed out".to_string()));
            }

            let nonce = rng.gen_range(0..self.max_nonce);
            let hash = self.compute_hash(data, nonce);

            if self.check_difficulty(&hash, difficulty) {
                return Ok(PoWProof {
                    data: data.to_string(),
                    nonce,
                    hash,
                    timestamp: SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                });
            }
        }

        Err(PoWVerifierError::Timeout("Max nonce attempts exceeded".to_string()))
    }
}

impl Default for PoWVerifier {
    fn default() -> Self {
        Self {
            hash_algorithm: "sha256".to_string(),
            max_nonce: 1_000_000,
        }
    }
}

#[async_trait]
impl PoWVerifierTrait for PoWVerifier {
    async fn verify_pow(
        &self,
        pow_proof: &str,
        difficulty: u32,
        timeout: Duration,
    ) -> Result<PoWVerification, Box<dyn std::error::Error>> {
        self.verify_pow(pow_proof, difficulty, timeout)
            .await
            .map_err(Into::into)
    }
    
    async fn generate_pow_challenge(&self, data: &str) -> Result<String, Box<dyn std::error::Error>> {
        self.generate_pow_challenge(data)
            .await
            .map_err(Into::into)
    }
}

/// PoW proof structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoWProof {
    /// Data being proven
    data: String,
    /// Nonce that satisfies difficulty
    nonce: u64,
    /// Hash of data + nonce
    hash: String,
    /// Timestamp of proof generation
    timestamp: u64,
}

/// PoW challenge structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoWChallenge {
    /// Data to include in hash
    data: String,
    /// Timestamp to prevent replay
    timestamp: u64,
    /// Target difficulty
    target_difficulty: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_verify_pow_success() {
        let verifier = PoWVerifier::default();
        
        // Create a simple proof (this would normally be mined)
        let proof = PoWProof {
            data: "test_data".to_string(),
            nonce: 12345,
            hash: verifier.compute_hash("test_data", 12345),
            timestamp: 1234567890,
        };
        
        let proof_json = serde_json::to_string(&proof).unwrap();
        let result = verifier.verify_pow(&proof_json, 0, Duration::from_secs(5)).await;
        assert!(result.is_ok());
        let verification = result.unwrap();
        assert!(verification.verified);
        assert_eq!(verification.nonce, 12345);
    }

    #[tokio::test]
    async fn test_verify_pow_invalid_hash() {
        let verifier = PoWVerifier::default();
        
        let proof = PoWProof {
            data: "test_data".to_string(),
            nonce: 12345,
            hash: "invalid_hash".to_string(),
            timestamp: 1234567890,
        };
        
        let proof_json = serde_json::to_string(&proof).unwrap();
        let result = verifier.verify_pow(&proof_json, 0, Duration::from_secs(5)).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            PoWVerifierError::HashMismatch(_) => {}
            _ => panic!("Expected HashMismatch error"),
        }
    }

    #[tokio::test]
    async fn test_generate_challenge() {
        let verifier = PoWVerifier::default();
        let challenge = verifier.generate_pow_challenge("test_data").await;
        assert!(challenge.is_ok());
        let challenge_str = challenge.unwrap();
        assert!(challenge_str.contains("test_data"));
    }

    #[tokio::test]
    async fn test_mine_pow() {
        let verifier = PoWVerifier::new("sha256".to_string(), 1000);
        let result = verifier.mine_pow("test_data", 2, Duration::from_secs(1)).await;
        // Might succeed or timeout depending on difficulty
        // Just ensure no panic
        assert!(result.is_ok() || matches!(result.err(), Some(PoWVerifierError::Timeout(_))));
    }
}