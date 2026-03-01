# Federation Modes: Wallet vs. No-Wallet

## Overview

NeuroFed Node supports two federation strategies to accommodate users with different preferences:

1. **Wallet Mode**: Uses Nostr payments (zaps) for verification
2. **No-Wallet Mode**: Uses proof-of-work (PoW) for verification

This document describes the architecture, configuration, and usage of both federation modes.

## Architecture

### Federation Strategy Enum

The core of the federation system is the `FederationStrategy` enum:

```rust
pub enum FederationStrategy {
    WalletMode,
    NoWalletMode,
}
```

### Federation Manager

The `FederationManager` struct orchestrates federation operations:

```rust
pub struct FederationManager {
    config: FederationManagerConfig,
    payment_verifier: Option<Arc<dyn PaymentVerifier>>,
    pow_verifier: Option<Arc<dyn PoWVerifier>>,
    nostr_federation: Arc<NostrFederation>,
    brain_manager: Arc<BrainManager>,
}
```

### Verification Traits

Two trait interfaces define the verification mechanisms:

```rust
#[async_trait]
pub trait PaymentVerifier: Send + Sync {
    async fn verify_zap(&self, amount_sats: u64, proof: &str) -> Result<(), Box<dyn std::error::Error>>;
    async fn check_balance(&self, pubkey: &str) -> Result<u64, Box<dyn std::error::Error>>;
}

#[async_trait]
pub trait PoWVerifier: Send + Sync {
    async fn verify_pow(&self, data: &str, proof: &str, difficulty: u32) -> Result<(), Box<dyn std::error::Error>>;
    async fn generate_pow_challenge(&self, data: &str) -> Result<String, Box<dyn std::error::Error>>;
}
```

## Wallet Mode

### Overview

Wallet mode uses Nostr payments (zaps) to verify federation requests. This approach:

- Requires users to have a Nostr wallet
- Uses real economic incentives
- Provides stronger Sybil resistance
- Integrates with the existing Nostr ecosystem

### Components

#### PaymentVerifier

The `PaymentVerifier` struct handles payment verification:

```rust
pub struct PaymentVerifier {
    relays: Vec<String>,
    public_key: String,
    private_key: Option<String>,
}
```

**Key Methods:**
- `verify_zap()`: Verifies a zap payment meets minimum amount
- `check_balance()`: Checks a user's balance
- `verify_signature()`: Validates cryptographic signatures
- `get_confirmations()`: Checks payment confirmations

#### Configuration

Wallet configuration in `config.rs`:

```rust
pub struct WalletConfig {
    pub private_key: Option<String>,
    pub relays: Vec<String>,
    pub min_zap_amount_sats: u64,
    pub required_confirmations: u32,
}
```

**Default Values:**
- `min_zap_amount_sats`: 1000 (0.00001 BTC)
- `required_confirmations`: 3

### Usage Example

```rust
use neuro_fed_node::payment_verifier::PaymentVerifier;
use neuro_fed_node::federation_manager::{FederationManager, FederationManagerConfig};

let payment_verifier = Arc::new(PaymentVerifier::new(
    vec!["wss://relay.damus.io".to_string()],
    "npub1...".to_string(),
    Some("nsec1...".to_string()),
));

let config = FederationManagerConfig {
    strategy: FederationStrategy::WalletMode,
    wallet_config: WalletConfig {
        private_key: Some("nsec1...".to_string()),
        relays: vec!["wss://relay.damus.io".to_string()],
        min_zap_amount_sats: 1000,
        required_confirmations: 3,
    },
    pow_config: PoWConfig::default(),
};

let manager = FederationManager::new(
    config,
    Some(payment_verifier),
    None,
    nostr_federation,
    brain_manager,
);
```

## No-Wallet Mode

### Overview

No-wallet mode uses proof-of-work (PoW) for verification. This approach:

- Requires no cryptocurrency or wallet
- Uses computational work for verification
- Accessible to users without financial resources
- Provides basic Sybil resistance through computational cost

### Components

#### PoWVerifier

The `PoWVerifier` struct handles proof-of-work verification:

```rust
pub struct PoWVerifier {
    hash_algorithm: String,
    max_nonce: u64,
}
```

**Key Methods:**
- `verify_pow()`: Verifies a PoW solution meets difficulty target
- `generate_pow_challenge()`: Creates a challenge for mining
- `mine_pow()`: Mines a PoW solution (for testing/benchmarking)
- `check_difficulty()`: Validates hash meets difficulty requirement

#### Configuration

PoW configuration in `config.rs`:

```rust
pub struct PoWConfig {
    pub difficulty: u32,
    pub timeout_seconds: u64,
    pub hash_algorithm: String,
    pub max_nonce: u64,
}
```

**Default Values:**
- `difficulty`: 4 (requires 4 leading zero bits)
- `timeout_seconds`: 30
- `hash_algorithm`: "sha256".to_string()
- `max_nonce`: 1_000_000

### Usage Example

```rust
use neuro_fed_node::pow_verifier::PoWVerifier;
use neuro_fed_node::federation_manager::{FederationManager, FederationManagerConfig};

let pow_verifier = Arc::new(PoWVerifier::new(
    "sha256".to_string(),
    1_000_000,
));

let config = FederationManagerConfig {
    strategy: FederationStrategy::NoWalletMode,
    wallet_config: WalletConfig::default(),
    pow_config: PoWConfig {
        difficulty: 4,
        timeout_seconds: 30,
        hash_algorithm: "sha256".to_string(),
        max_nonce: 1_000_000,
    },
};

let manager = FederationManager::new(
    config,
    None,
    Some(pow_verifier),
    nostr_federation,
    brain_manager,
);
```

## Configuration

### FederationConfig

The main federation configuration structure:

```rust
pub struct FederationConfig {
    pub strategy: String,
    pub wallet_config: WalletConfig,
    pub pow_config: PoWConfig,
}
```

**Default Strategy:** `"wallet"` (WalletMode)

### NodeConfig Integration

Federation configuration is integrated into the main node configuration:

```rust
pub struct NodeConfig {
    // ... other fields
    pub federation_config: FederationConfig,
}
```

**Default Configuration:**

```toml
[federation]
strategy = "wallet"  # or "no_wallet"

[federation.wallet]
private_key = ""  # optional, for sending zaps
relays = ["wss://relay.damus.io"]
min_zap_amount_sats = 1000
required_confirmations = 3

[federation.pow]
difficulty = 4
timeout_seconds = 30
hash_algorithm = "sha256"
max_nonce = 1000000
```

## Integration with Nostr Federation

The `NostrFederation` struct has been extended to support both modes:

```rust
pub struct NostrFederation {
    config: NostrConfig,
    federation_strategy: FederationStrategy,
    // ... other fields
}
```

**New Methods:**
- `new_with_strategy()`: Creates instance with specific strategy
- Updated event processing to check federation strategy

## Error Handling

### FederationError

Common error type for federation operations:

```rust
pub enum FederationError {
    PaymentVerificationError(String),
    PoWVerificationError(String),
    FederationStrategyError(String),
    InvalidRequest(String),
    InsufficientFunds(u64, u64),  // required, actual
    InvalidProof(String),
    Timeout,
}
```

### PaymentVerifierError

Wallet-specific errors:

```rust
pub enum PaymentVerifierError {
    InvalidSignature,
    InsufficientAmount(u64, u64),  // required, actual
    RelayError(String),
    ParseError(String),
    Timeout,
}
```

### PoWVerifierError

PoW-specific errors:

```rust
pub enum PoWVerifierError {
    InvalidHash,
    InsufficientDifficulty(u32, u32),  // required, actual
    Timeout,
    InvalidProofFormat,
    MiningFailed,
}
```

## Usage Examples

### Basic Usage with Wallet Mode

```rust
use neuro_fed_node::{
    FederationManager, FederationManagerConfig, FederationStrategy,
    payment_verifier::PaymentVerifier,
    nostr_federation::NostrFederation,
    brain_manager::BrainManager,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = NodeConfig::load_from_file("config.toml")?;
    
    // Create verifiers based on strategy
    let payment_verifier = match config.federation_config.strategy.as_str() {
        "wallet" => Some(Arc::new(PaymentVerifier::new(
            config.federation_config.wallet_config.relays.clone(),
            "npub1...".to_string(),
            config.federation_config.wallet_config.private_key.clone(),
        )) as Arc<dyn PaymentVerifier>),
        _ => None,
    };
    
    let pow_verifier = match config.federation_config.strategy.as_str() {
        "no_wallet" => Some(Arc::new(PoWVerifier::new(
            config.federation_config.pow_config.hash_algorithm.clone(),
            config.federation_config.pow_config.max_nonce,
        )) as Arc<dyn PoWVerifier>),
        _ => None,
    };
    
    // Create federation manager
    let federation_manager = FederationManager::new(
        FederationManagerConfig {
            strategy: match config.federation_config.strategy.as_str() {
                "wallet" => FederationStrategy::WalletMode,
                "no_wallet" => FederationStrategy::NoWalletMode,
                _ => FederationStrategy::WalletMode,
            },
            wallet_config: config.federation_config.wallet_config.clone(),
            pow_config: config.federation_config.pow_config.clone(),
        },
        payment_verifier,
        pow_verifier,
        nostr_federation,
        brain_manager,
    );
    
    // Process federation requests
    let request = FederationRequest {
        brain_id: "test-brain".to_string(),
        operation: "share".to_string(),
        timestamp: Utc::now().timestamp(),
    };
    
    let response = federation_manager.process_federation_request(&request).await?;
    println!("Federation response: {:?}", response);
    
    Ok(())
}
```

### Switching Strategies at Runtime

```rust
// Switch from wallet to no-wallet mode
federation_manager.switch_strategy(FederationStrategy::NoWalletMode);

// Update configuration accordingly
let new_config = FederationManagerConfig {
    strategy: FederationStrategy::NoWalletMode,
    wallet_config: WalletConfig::default(),
    pow_config: PoWConfig {
        difficulty: 6,  // Higher difficulty
        timeout_seconds: 60,
        hash_algorithm: "sha256".to_string(),
        max_nonce: 10_000_000,
    },
};

// Recreate manager with new configuration
```

## Testing

### Unit Tests

Both federation modes have comprehensive unit tests:

**Wallet Mode Tests:**
- `test_verify_zap_success()`: Valid zap verification
- `test_verify_zap_insufficient_amount()`: Rejects insufficient payments
- `test_check_balance()`: Balance checking functionality

**No-Wallet Mode Tests:**
- `test_verify_pow_success()`: Valid PoW verification
- `test_verify_pow_invalid_hash()`: Rejects invalid hashes
- `test_generate_challenge()`: Challenge generation
- `test_mine_pow()`: PoW mining functionality

**Integration Tests:**
- `test_wallet_mode_federation()`: Full wallet mode workflow
- `test_no_wallet_mode_federation()`: Full no-wallet mode workflow

### Running Tests

```bash
# Run all federation tests
cargo test federation_manager::tests
cargo test payment_verifier::tests
cargo test pow_verifier::tests

# Run specific test
cargo test test_wallet_mode_federation
cargo test test_no_wallet_mode_federation
```

## Performance Considerations

### Wallet Mode
- **Network Dependent**: Requires relay connectivity
- **Async Operations**: Payment verification is async
- **Cryptographic Overhead**: Signature verification adds CPU cost
- **Confirmation Wait**: May need to wait for confirmations

### No-Wallet Mode
- **CPU Intensive**: PoW mining requires computational resources
- **Difficulty Scaling**: Adjust difficulty based on security needs
- **Memory Efficient**: Minimal memory footprint
- **Deterministic**: No network dependencies

## Security Considerations

### Wallet Mode Security
1. **Signature Verification**: All zaps must be cryptographically signed
2. **Minimum Amount**: Enforces economic cost for Sybil resistance
3. **Confirmation Requirements**: Prevents double-spend attacks
4. **Relay Trust**: Depends on honest relay behavior

### No-Wallet Mode Security
1. **Difficulty Adjustment**: Must be high enough to deter spam
2. **Nonce Range**: Large enough to prevent brute force
3. **Hash Algorithm**: Use cryptographically secure hashes (SHA-256)
4. **Timeouts**: Prevent resource exhaustion attacks

## Migration Guide

### From Single Mode to Dual Mode

1. **Update Configuration**: Add `federation_config` section to your config
2. **Choose Strategy**: Select `"wallet"` or `"no_wallet"`
3. **Update Code**: Use `FederationManager` instead of direct `NostrFederation`
4. **Test Verification**: Verify both modes work correctly

### Configuration Migration

**Old configuration:**
```toml
[nostr]
relays = ["wss://relay.damus.io"]
private_key = "nsec1..."
```

**New configuration:**
```toml
[nostr]
relays = ["wss://relay.damus.io"]
private_key = "nsec1..."

[federation]
strategy = "wallet"

[federation.wallet]
private_key = "nsec1..."
relays = ["wss://relay.damus.io"]
min_zap_amount_sats = 1000
required_confirmations = 3

[federation.pow]
difficulty = 4
timeout_seconds = 30
hash_algorithm = "sha256"
max_nonce = 1000000
```

## Troubleshooting

### Common Issues

1. **Payment Verification Fails**
   - Check relay connectivity
   - Verify Nostr keys are correct
   - Ensure sufficient zap amount
   - Check signature format

2. **PoW Verification Fails**
   - Verify difficulty setting
   - Check hash algorithm compatibility
   - Ensure nonce range is sufficient
   - Validate proof format

3. **Strategy Switching Issues**
   - Verify both verifiers are properly initialized
   - Check configuration consistency
   - Ensure proper error handling

4. **Performance Problems**
   - Adjust PoW difficulty for your hardware
   - Use async operations for network calls
   - Implement caching where appropriate

### Debugging

Enable debug logging:
```rust
env_logger::Builder::from_default_env()
    .filter_level(log::LevelFilter::Debug)
    .init();
```

Check logs for:
- Verification attempts and results
- Strategy selection
- Configuration loading
- Error details

## Future Extensions

### Planned Features
1. **Hybrid Mode**: Combine wallet and PoW verification
2. **Dynamic Strategy Switching**: Auto-switch based on network conditions
3. **Multi-Signature Support**: Enhanced wallet security
4. **Proof-of-Stake**: Alternative to PoW
5. **Reputation System**: Combine with verification

### API Stability
The federation API is designed for extensibility. Future changes will maintain backward compatibility through versioned traits and configuration.

## Conclusion

The dual federation mode system provides flexibility for users with different preferences and resources. Wallet mode offers strong economic security for users with cryptocurrency access, while no-wallet mode ensures accessibility for users without financial resources. Both modes integrate seamlessly with the existing NeuroFed architecture and provide robust Sybil resistance for decentralized federation.