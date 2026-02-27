# Nostr Federation Component Technical Specifications

## Overview
`nostr_federation.rs` implements decentralized federation using the Nostr protocol for gossip-based collective intelligence. It handles publishing/signing prediction-error deltas and subscribing to trusted nodes for CPC-style decentralized Bayesian updates.

## Architecture

### Core Data Structures
```rust
// Public API
pub struct NostrFederation {
    config: NostrConfig,
    client: NostrClient,
    trust_manager: TrustManager,
    delta_processor: DeltaProcessor,
    event_handler: EventHandler,
    zap_handler: ZapHandler,
}

#[derive(Debug, Clone)]
pub struct NostrConfig {
    relays: Vec<String>,
    pubkey: String,
    private_key: String,
    kind_pc_error_delta: u16,
    kind_pc_belief_snapshot: u16,
    trust_pubkeys: Vec<String>,
    zap_relay: Option<String>,
    blossom_enabled: bool,
    max_delta_size: usize,
    compression_level: u8,
}

pub struct TrustManager {
    pubkeys: HashMap<String, TrustLevel>,
    reputation: HashMap<String, f32>,
    last_seen: HashMap<String, DateTime<Utc>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrustLevel {
    Untrusted,
    Trusted,
    HighlyTrusted,
}

pub struct DeltaProcessor {
    pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
    compression: CompressionConfig,
    quantization: QuantizationConfig,
}

#[derive(Debug, Clone)]
pub struct CompressionConfig {
    level: u8,
    algorithm: CompressionAlgorithm,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompressionAlgorithm {
    Zstd,
    LZ4,
    Brotli,
}

pub struct QuantizationConfig {
    bits: u8,
    method: QuantizationMethod,
    scale_factors: HashMap<String, f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationMethod {
    Linear,
    Log,
    Symmetric,
}

pub struct ZapHandler {
    lightning_client: LightningClient,
    zap_threshold: f32,
    zap_amount_msat: u64,
}

pub struct EventHandler {
    event_handlers: HashMap<u16, Box<dyn Fn(NostrEvent) -> Result<(), EventHandlerError>>>,
}

#[derive(Debug, Clone)]
pub struct NostrEvent {
    id: String,
    kind: u16,
    content: String,
    tags: Vec<NostrTag>,
    pubkey: String,
    created_at: DateTime<Utc>,
    relay: String,
}

#[derive(Debug, Clone)]
pub struct NostrTag {
    kind: String,
    value: String,
}

#[derive(Debug, Clone)]
pub struct PCErrorDelta {
    pubkey: String,
    timestamp: DateTime<Utc>,
    level_errors: Vec<LevelError>,
    weight_updates: Vec<WeightUpdate>,
    free_energy: f32,
    surprise_score: f32,
    compression_info: CompressionInfo,
}

#[derive(Debug, Clone)]
pub struct LevelError {
    level: usize,
    error_vector: Vec<f32>,
    precision: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct WeightUpdate {
    level: usize,
    delta_weights: Vec<f32>,
    update_count: usize,
}

#[derive(Debug, Clone)]
pub struct CompressionInfo {
    algorithm: CompressionAlgorithm,
    original_size: usize,
    compressed_size: usize,
    compression_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct PCBeliefSnapshot {
    pubkey: String,
    timestamp: DateTime<Utc>,
    beliefs: Vec<Array2<f32>>,
    free_energy: f32,
    trust_score: f32,
}
```

### Event Kind Definitions
```rust
impl NostrFederation {
    pub fn new_event_kinds() -> (u16, u16) {
        // Custom event kinds for NeuroPC
        const PC_ERROR_DELTA: u16 = 8700;
        const PC_BELIEF_SNAPSHOT: u16 = 8710;
        
        (PC_ERROR_DELTA, PC_BELIEF_SNAPSHOT)
    }
}
```

## Core Functionality

### Publishing Deltas
```rust
impl NostrFederation {
    pub async fn publish_delta(&self, delta: &PCErrorDelta) -> Result<String, FederationError> {
        // Serialize delta to JSON
        let json_content = serde_json::to_string(delta)?;
        
        // Compress content
        let compressed = self.compress_content(&json_content)?;
        
        // Create Nostr event
        let event = NostrEvent {
            id: Self::generate_event_id(),
            kind: self.config.kind_pc_error_delta,
            content: base64::encode(&compressed),
            tags: self.create_delta_tags(delta),
            pubkey: self.config.pubkey.clone(),
            created_at: Utc::now(),
            relay: self.select_relay(),
        };
        
        // Sign and publish
        let signed_event = self.client.sign_event(&event)?;
        let event_id = self.client.publish_event(&signed_event).await?;
        
        Ok(event_id)
    }
    
    fn compress_content(&self, content: &str) -> Result<Vec<u8>, FederationError> {
        match self.config.compression_algorithm {
            CompressionAlgorithm::Zstd => {
                let compressor = zstd::block::Compressor::new(self.config.compression_level)?;
                compressor.compress(content.as_bytes())
            }
            CompressionAlgorithm::LZ4 => {
                let mut encoder = lz4::block::Encoder::new(Vec::new())?;
                encoder.write_all(content.as_bytes())?;
                let (compressed, result) = encoder.finish();
                result?;
                Ok(compressed)
            }
            CompressionAlgorithm::Brotli => {
                let mut buffer = Vec::new();
                let mut brotli_encoder = brotli::enc::BrotliEncoder::new();
                brotli_encoder.set_quality(self.config.compression_level as i32);
                brotli_encoder.compress(content.as_bytes(), &mut buffer)?;
                Ok(buffer)
            }
        }
    }
    
    fn create_delta_tags(&self, delta: &PCErrorDelta) -> Vec<NostrTag> {
        vec![
            NostrTag {
                kind: "d".to_string(),
                value: format!("level_errors:{}", delta.level_errors.len()),
            },
            NostrTag {
                kind: "d".to_string(),
                value: format!("weight_updates:{}", delta.weight_updates.len()),
            },
            NostrTag {
                kind: "d".to_string(),
                value: format!("fe:{:.2}", delta.free_energy),
            },
            NostrTag {
                kind: "d".to_string(),
                value: format!("surprise:{:.2}", delta.surprise_score),
            },
            NostrTag {
                kind: "r".to_string(),
                value: self.config.relays.join(","),
            },
        ]
    }
}
```

### Subscribing to Events
```rust
impl NostrFederation {
    pub async fn start_subscription(&self) -> Result<(), FederationError> {
        // Subscribe to custom event kinds
        let kinds = vec![self.config.kind_pc_error_delta, self.config.kind_pc_belief_snapshot];
        
        for relay in &self.config.relays {
            self.client.subscribe_to_kinds(relay, &kinds).await?;
        }
        
        // Set up event handler
        self.event_handler.register_handler(
            self.config.kind_pc_error_delta,
            Box::new(self.handle_pc_error_delta)
        );
        
        self.event_handler.register_handler(
            self.config.kind_pc_belief_snapshot,
            Box::new(self.handle_pc_belief_snapshot)
        );
        
        Ok(())
    }
    
    async fn handle_pc_error_delta(&self, event: NostrEvent) -> Result<(), EventHandlerError> {
        // Verify event signature
        if !self.client.verify_event(&event)? {
            return Err(EventHandlerError::InvalidSignature);
        }
        
        // Check trust level
        let trust_level = self.trust_manager.get_trust_level(&event.pubkey);
        if trust_level < TrustLevel::Trusted {
            return Err(EventHandlerError::UntrustedSource);
        }
        
        // Deserialize and decompress
        let compressed = base64::decode(&event.content)?;
        let json_content = self.decompress_content(&compressed)?;
        let delta: PCErrorDelta = serde_json::from_str(&json_content)?;
        
        // Process delta
        self.process_incoming_delta(&delta).await?;
        
        Ok(())
    }
    
    async fn process_incoming_delta(&self, delta: &PCErrorDelta) -> Result<(), FederationError> {
        // Apply CPC-style decentralized Bayesian update
        let mut pc_hierarchy = self.delta_processor.pc_hierarchy.lock().await;
        
        for level_error in &delta.level_errors {
            // Apply error to corresponding level
            if let Some(level) = pc_hierarchy.levels.get_mut(level_error.level) {
                // Weighted error application
                let weighted_error = &level_error.error_vector * delta.surprise_score;
                level.beliefs = &level.beliefs + &weighted_error;
            }
        }
        
        // Update trust reputation based on utility
        self.trust_manager.update_reputation(
            &delta.pubkey,
            delta.surprise_score
        );
        
        // Request zap if delta was highly useful
        if delta.surprise_score > self.config.zap_threshold {
            self.request_zap(&delta.pubkey, delta.surprise_score).await?;
        }
        
        Ok(())
    }
}
```

## Trust Management

### Trust Level Evaluation
```rust
impl TrustManager {
    pub fn get_trust_level(&self, pubkey: &str) -> TrustLevel {
        if let Some(level) = self.pubkeys.get(pubkey) {
            level.clone()
        } else if self.config.trust_pubkeys.contains(pubkey) {
            TrustLevel::Trusted
        } else {
            TrustLevel::Untrusted
        }
    }
    
    pub fn update_reputation(&mut self, pubkey: &str, utility_score: f32) {
        let current_rep = *self.reputation.get(pubkey).unwrap_or(&0.0);
        let new_rep = current_rep * 0.9 + utility_score * 0.1; // Exponential moving average
        
        self.reputation.insert(pubkey.to_string(), new_rep);
        self.last_seen.insert(pubkey.to_string(), Utc::now());
        
        // Update trust level based on reputation
        if new_rep > 0.8 {
            self.pubkeys.insert(pubkey.to_string(), TrustLevel::HighlyTrusted);
        } else if new_rep > 0.5 {
            self.pubkeys.insert(pubkey.to_string(), TrustLevel::Trusted);
        } else {
            self.pubkeys.insert(pubkey.to_string(), TrustLevel::Untrusted);
        }
    }
    
    pub fn get_reliable_pubkeys(&self) -> Vec<String> {
        self.reputation.iter()
            .filter(|(_, rep)| **rep > 0.5)
            .map(|(pubkey, _)| pubkey.clone())
            .collect()
    }
}
```

## Delta Processing

### Compression and Quantization
```rust
impl DeltaProcessor {
    pub fn compress_and_quantize(&self, delta: &PCErrorDelta) -> Result<Vec<u8>, FederationError> {
        // Serialize to JSON
        let json_content = serde_json::to_string(delta)?;
        
        // Quantize numerical values
        let quantized_content = self.quantize_content(&json_content)?;
        
        // Compress
        let compressed = self.compress_content(&quantized_content)?;
        
        Ok(compressed)
    }
    
    fn quantize_content(&self, content: &str) -> Result<String, FederationError> {
        // Simple quantization for demonstration
        let mut result = content.to_string();
        
        // Quantize floating point numbers to specified bits
        let re = Regex::new(r"-?[0-9]+\.[0-9]+")?;
        result = re.replace_all(&result, |cap: &Captures| {
            let num: f64 = cap.get(0).unwrap().as_str().parse().unwrap();
            let quantized = (num * (1 << self.config.quantization.bits) as f64).round() 
                          / (1 << self.config.quantization.bits) as f64;
            format!("{:.4}", quantized)
        }).to_string();
        
        Ok(result)
    }
}
```

## Zap Handling

### Lightning Integration
```rust
impl ZapHandler {
    pub async fn request_zap(&self, pubkey: &str, utility_score: f32) -> Result<(), FederationError> {
        // Calculate zap amount based on utility
        let zap_amount = (self.zap_amount_msat as f32 * utility_score) as u64;
        
        // Create lightning invoice
        let memo = format!("NeuroPC delta utility: {:.2}", utility_score);
        let invoice = self.lightning_client.create_invoice(zap_amount, &memo).await?;
        
        // Create zap request event
        let zap_event = NostrEvent {
            id: Self::generate_event_id(),
            kind: 9000, // Standard zap event kind
            content: invoice.payment_request.clone(),
            tags: vec![
                NostrTag {
                    kind: "d".to_string(),
                    value: "type:pc_delta_zap".to_string(),
                },
                NostrTag {
                    kind: "p".to_string(),
                    value: pubkey.to_string(),
                },
                NostrTag {
                    kind: "s".to_string(),
                    value: format!("{}", utility_score),
                },
            ],
            pubkey: self.lightning_client.get_pubkey(),
            created_at: Utc::now(),
            relay: self.config.zap_relay.clone().unwrap_or_default(),
        };
        
        // Publish zap request
        self.lightning_client.publish_event(&zap_event).await?;
        
        Ok(())
    }
}
```

## Blossom Integration

### Large Delta Handling
```rust
impl NostrFederation {
    pub async fn publish_large_delta(&self, delta: &PCErrorDelta) -> Result<Vec<String>, FederationError> {
        if self.config.blossom_enabled && delta.estimated_size() > self.config.max_delta_size {
            // Use Blossom for large deltas
            let blossom_id = self.publish_via_blossom(delta).await?;
            Ok(vec![blossom_id])
        } else {
            // Use standard publishing
            let event_id = self.publish_delta(delta)?;
            Ok(vec![event_id])
        }
    }
    
    async fn publish_via_blossom(&self, delta: &PCErrorDelta) -> Result<String, FederationError> {
        // Compress and split into chunks
        let compressed = self.compress_content(&serde_json::to_string(delta)?)?;
        let chunks = self.split_into_chunks(&compressed)?;
        
        // Publish chunks via Blossom
        let blossom_ids = self.publish_blossom_chunks(&chunks).await?;
        
        // Create manifest event
        let manifest = self.create_blossom_manifest(&blossom_ids, delta)?;
        let manifest_id = self.client.publish_event(&manifest).await?;
        
        Ok(manifest_id)
    }
}
```

## Configuration Examples

### Basic Federation Configuration
```rust
let basic_config = NostrConfig {
    relays: vec![
        "wss://relay.primal.net".to_string(),
        "wss://relay.damus.io".to_string(),
    ],
    pubkey: "your-nostr-pubkey".to_string(),
    private_key: "your-nostr-private-key".to_string(),
    kind_pc_error_delta: 8700,
    kind_pc_belief_snapshot: 8710,
    trust_pubkeys: vec![],
    zap_relay: Some("wss://relay.primal.net".to_string()),
    blossom_enabled: false,
    max_delta_size: 10 * 1024, // 10 KB
    compression_level: 6,
    compression_algorithm: CompressionAlgorithm::Zstd,
    quantization: QuantizationConfig {
        bits: 8,
        method: QuantizationMethod::Linear,
        scale_factors: HashMap::new(),
    },
};
```

### Advanced Configuration with Trust Settings
```rust
let advanced_config = NostrConfig {
    relays: vec![
        "wss://relay.primal.net",
        "wss://relay.damus.io",
        "wss://nostr-pub.wellorder.net",
    ],
    pubkey: "your-nostr-pubkey".to_string(),
    private_key: "your-nostr-private-key".to_string(),
    kind_pc_error_delta: 8700,
    kind_pc_belief_snapshot: 8710,
    trust_pubkeys: vec![
        "trusted-pubkey-1".to_string(),
        "trusted-pubkey-2".to_string(),
    ],
    zap_relay: Some("wss://relay.primal.net".to_string()),
    blossom_enabled: true,
    max_delta_size: 50 * 1024, // 50 KB
    compression_level: 9,
    compression_algorithm: CompressionAlgorithm::Brotli,
    quantization: QuantizationConfig {
        bits: 16,
        method: QuantizationMethod::Log,
        scale_factors: HashMap::from([
            ("level_errors".to_string(), 1.0),
            ("weight_updates".to_string(), 0.1),
        ]),
    },
};
```

## Error Handling

### Custom Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum FederationError {
    #[error("Nostr client error: {0}")]
    NostrClientError(String),
    
    #[error("Compression failed: {0}")]
    CompressionError(String),
    
    #[error("Serialization failed: {0}")]
    SerializationError(String),
    
    #[error("Invalid event format")]
    InvalidEventFormat,
    
    #[error("Untrusted source: {0}")]
    UntrustedSource(String),
    
    #[error("Zap request failed: {0}")]
    ZapRequestFailed(String),
    
    #[error("Blossom integration error: {0}")]
    BlossomError(String),
}

#[derive(Debug, thiserror::Error)]
pub enum EventHandlerError {
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("Untrusted source")]
    UntrustedSource,
    
    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),
    
    #[error("Processing timeout")]
    ProcessingTimeout,
}
```

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_delta_compression() {
        let processor = DeltaProcessor::default();
        let delta = create_test_delta();
        
        let compressed = processor.compress_and_quantize(&delta).unwrap();
        assert!(compressed.len() < serde_json::to_string(&delta).unwrap().len());
    }
    
    #[test]
    fn test_trust_evaluation() {
        let mut trust_manager = TrustManager::default();
        
        trust_manager.update_reputation("pubkey-1", 0.9);
        trust_manager.update_reputation("pubkey-2", 0.3);
        
        assert_eq!(trust_manager.get_trust_level("pubkey-1"), TrustLevel::HighlyTrusted);
        assert_eq!(trust_manager.get_trust_level("pubkey-2"), TrustLevel::Untrusted);
    }
}
```

### Integration Tests
- Test with actual Nostr relays and event publishing
- Verify delta processing and trust updates
- Test zap request flow with mock Lightning client
- Benchmark compression and quantization performance

## Dependencies

### Required
- `nostr-rs` or `nostr-sdk` - Nostr protocol implementation
- `serde = { version = "1.0", features = ["derive"] }` - Serialization
- `serde_json = "1.0"` - JSON serialization
- `thiserror = "1.0"` - Error handling
- `chrono = "0.4"` - Timestamp handling
- `tokio = { version = "1.0", features = ["full"] }` - Async runtime

### Optional
- `zstd = "0.11"` - Zstandard compression
- `lz4 = "1.0"` - LZ4 compression
- `brotli = "3.0"` - Brotli compression
- `lightning = "0.1"` - Lightning Network integration
- `blossom = "0.1"` - Blossom protocol integration
- `tracing = "0.1"` - Structured logging

## Security Considerations

- Validate all incoming event signatures
- Implement rate limiting to prevent spam
- Use secure random number generation for event IDs
- Sanitize all event content before processing
- Implement proper error handling to prevent information leakage

This specification provides a complete blueprint for implementing the Nostr federation component with all necessary protocol integration, trust management, and compression features needed for the Phase 1 development.