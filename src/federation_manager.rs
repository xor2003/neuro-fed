// src/federation_manager.rs
// Federation manager for wallet vs. no-wallet federation modes

use std::sync::Arc;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, debug};
use async_trait::async_trait;

use crate::nostr_federation::{NostrFederation, NostrEvent, EventKind};
use crate::types::{FederationRequest, FederationResponse, PaymentVerification, PoWVerification};

/// Federation strategy selection
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FederationStrategy {
    /// Wallet mode with Nostr payments (zaps)
    WalletMode {
        min_sats: u64,
        required_confirmations: u32,
    },
    /// No-wallet mode with proof-of-work
    NoWalletMode {
        difficulty: u32,
        timeout_seconds: u64,
    },
}

impl Default for FederationStrategy {
    fn default() -> Self {
        FederationStrategy::WalletMode {
            min_sats: 1000,
            required_confirmations: 1,
        }
    }
}

/// Federation manager configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationManagerConfig {
    pub strategy: FederationStrategy,
    pub enable_fallback: bool,
    pub max_retries: u32,
    pub request_timeout_seconds: u64,
}

impl Default for FederationManagerConfig {
    fn default() -> Self {
        Self {
            strategy: FederationStrategy::default(),
            enable_fallback: true,
            max_retries: 3,
            request_timeout_seconds: 30,
        }
    }
}

/// Federation manager errors
#[derive(Error, Debug, Clone)]
pub enum FederationError {
    #[error("Payment verification failed: {0}")]
    PaymentVerificationFailed(String),
    #[error("Proof-of-work verification failed: {0}")]
    PoWVerificationFailed(String),
    #[error("Federation request timeout")]
    Timeout,
    #[error("Invalid federation request: {0}")]
    InvalidRequest(String),
    #[error("Nostr federation error: {0}")]
    NostrError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Federation manager
pub struct FederationManager {
    config: FederationManagerConfig,
    nostr_federation: Arc<NostrFederation>,
    payment_verifier: Option<Arc<dyn PaymentVerifier>>,
    pow_verifier: Option<Arc<dyn PoWVerifier>>,
}

impl FederationManager {
    /// Create a new federation manager
    pub fn new(
        config: FederationManagerConfig,
        nostr_federation: Arc<NostrFederation>,
        payment_verifier: Option<Arc<dyn PaymentVerifier>>,
        pow_verifier: Option<Arc<dyn PoWVerifier>>,
    ) -> Self {
        Self {
            config,
            nostr_federation,
            payment_verifier,
            pow_verifier,
        }
    }

    /// Process a federation request
    pub async fn process_federation_request(
        &self,
        request: FederationRequest,
    ) -> Result<FederationResponse, FederationError> {
        info!("Processing federation request: {:?}", request.request_type);
        
        // Verify based on strategy
        match &self.config.strategy {
            FederationStrategy::WalletMode { min_sats, required_confirmations } => {
                self.verify_payment(&request, *min_sats, *required_confirmations).await?;
            }
            FederationStrategy::NoWalletMode { difficulty, timeout_seconds } => {
                self.verify_pow(&request, *difficulty, *timeout_seconds).await?;
            }
        }

        // Process the request (delegate to Nostr federation)
        let response = self.send_federation_response(&request).await?;
        
        Ok(response)
    }

    /// Verify payment for wallet mode
    async fn verify_payment(
        &self,
        request: &FederationRequest,
        min_sats: u64,
        required_confirmations: u32,
    ) -> Result<(), FederationError> {
        debug!("Verifying payment for request {}", request.id);
        
        let verifier = self.payment_verifier.as_ref()
            .ok_or_else(|| FederationError::ConfigError("Payment verifier not configured".to_string()))?;
        
        let verification = verifier.verify_zap(
            &request.payment_proof,
            min_sats,
            required_confirmations,
        ).await
        .map_err(|e| FederationError::PaymentVerificationFailed(e.to_string()))?;
        
        if verification.verified {
            info!("Payment verified: {} sats", verification.amount_sats);
            Ok(())
        } else {
            Err(FederationError::PaymentVerificationFailed(
                format!("Payment verification failed: {:?}", verification.reason)
            ))
        }
    }

    /// Verify proof-of-work for no-wallet mode
    async fn verify_pow(
        &self,
        request: &FederationRequest,
        difficulty: u32,
        timeout_seconds: u64,
    ) -> Result<(), FederationError> {
        debug!("Verifying PoW for request {}", request.id);
        
        let verifier = self.pow_verifier.as_ref()
            .ok_or_else(|| FederationError::ConfigError("PoW verifier not configured".to_string()))?;
        
        let verification = verifier.verify_pow(
            &request.pow_proof,
            difficulty,
            Duration::from_secs(timeout_seconds),
        ).await
        .map_err(|e| FederationError::PoWVerificationFailed(e.to_string()))?;
        
        if verification.verified {
            info!("PoW verified: nonce={}, hash={}", verification.nonce, verification.hash);
            Ok(())
        } else {
            Err(FederationError::PoWVerificationFailed(
                format!("PoW verification failed: {:?}", verification.reason)
            ))
        }
    }

    /// Send federation response via Nostr
    async fn send_federation_response(
        &self,
        request: &FederationRequest,
    ) -> Result<FederationResponse, FederationError> {
        let event = NostrEvent {
            id: format!("fed-resp-{}", request.id),
            content: serde_json::to_string(&request).map_err(|e| {
                FederationError::InvalidRequest(format!("Serialization error: {}", e))
            })?,
            kind: EventKind::SystemMessage,
            timestamp: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.nostr_federation.publish_event(event).await
            .map_err(|e| FederationError::NostrError(e.to_string()))?;

        Ok(FederationResponse {
            id: request.id.clone(),
            success: true,
            message: "Federation request processed successfully".to_string(),
            timestamp: SystemTime::now(),
            metadata: Default::default(),
        })
    }

    /// Get current federation strategy
    pub fn strategy(&self) -> &FederationStrategy {
        &self.config.strategy
    }

    /// Switch federation strategy
    pub fn switch_strategy(&mut self, new_strategy: FederationStrategy) {
        info!("Switching federation strategy to {:?}", new_strategy);
        self.config.strategy = new_strategy;
    }
}

/// Trait for payment verification
#[async_trait]
pub trait PaymentVerifier: Send + Sync {
    async fn verify_zap(
        &self,
        payment_proof: &str,
        min_sats: u64,
        required_confirmations: u32,
    ) -> Result<PaymentVerification, Box<dyn std::error::Error>>;
    
    async fn check_balance(&self, pubkey: &str) -> Result<u64, Box<dyn std::error::Error>>;
}

/// Trait for proof-of-work verification
#[async_trait]
pub trait PoWVerifier: Send + Sync {
    async fn verify_pow(
        &self,
        pow_proof: &str,
        difficulty: u32,
        timeout: Duration,
    ) -> Result<PoWVerification, Box<dyn std::error::Error>>;
    
    async fn generate_pow_challenge(&self, data: &str) -> Result<String, Box<dyn std::error::Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use crate::nostr_federation::{NostrConfig, NostrFederation};
    use crate::types::{PaymentVerification, PoWVerification};
    use async_trait::async_trait;

    struct MockPaymentVerifier;
    #[async_trait]
    impl PaymentVerifier for MockPaymentVerifier {
        async fn verify_zap(
            &self,
            _payment_proof: &str,
            _min_sats: u64,
            _required_confirmations: u32,
        ) -> Result<PaymentVerification, Box<dyn std::error::Error>> {
            Ok(PaymentVerification {
                verified: true,
                amount_sats: 1000,
                reason: None,
            })
        }
        
        async fn check_balance(&self, _pubkey: &str) -> Result<u64, Box<dyn std::error::Error>> {
            Ok(5000)
        }
    }

    struct MockPoWVerifier;
    #[async_trait]
    impl PoWVerifier for MockPoWVerifier {
        async fn verify_pow(
            &self,
            _pow_proof: &str,
            _difficulty: u32,
            _timeout: Duration,
        ) -> Result<PoWVerification, Box<dyn std::error::Error>> {
            Ok(PoWVerification {
                verified: true,
                nonce: 12345,
                hash: "abcdef".to_string(),
                reason: None,
            })
        }
        
        async fn generate_pow_challenge(&self, _data: &str) -> Result<String, Box<dyn std::error::Error>> {
            Ok("challenge".to_string())
        }
    }

    #[tokio::test]
    async fn test_wallet_mode_federation() {
        let nostr_config = NostrConfig::default();
        let nostr_fed = Arc::new(NostrFederation::new(nostr_config));
        let payment_verifier = Arc::new(MockPaymentVerifier);
        
        let config = FederationManagerConfig {
            strategy: FederationStrategy::WalletMode {
                min_sats: 100,
                required_confirmations: 1,
            },
            ..Default::default()
        };
        
        let manager = FederationManager::new(
            config,
            nostr_fed,
            Some(payment_verifier),
            None,
        );
        
        let request = FederationRequest {
            id: "test-1".to_string(),
            request_type: "test".to_string(),
            payment_proof: "zap-proof".to_string(),
            pow_proof: "".to_string(),
            timestamp: SystemTime::now(),
            metadata: Default::default(),
        };
        
        let result = manager.process_federation_request(request).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_no_wallet_mode_federation() {
        let nostr_config = NostrConfig::default();
        let nostr_fed = Arc::new(NostrFederation::new(nostr_config));
        let pow_verifier = Arc::new(MockPoWVerifier);
        
        let config = FederationManagerConfig {
            strategy: FederationStrategy::NoWalletMode {
                difficulty: 5,
                timeout_seconds: 10,
            },
            ..Default::default()
        };
        
        let manager = FederationManager::new(
            config,
            nostr_fed,
            None,
            Some(pow_verifier),
        );
        
        let request = FederationRequest {
            id: "test-2".to_string(),
            request_type: "test".to_string(),
            payment_proof: "".to_string(),
            pow_proof: "pow-proof".to_string(),
            timestamp: SystemTime::now(),
            metadata: Default::default(),
        };
        
        let result = manager.process_federation_request(request).await;
        assert!(result.is_ok());
    }
}