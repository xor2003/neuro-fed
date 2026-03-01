# Privacy Networks Integration

## Overview

The NeuroFed Node includes support for multiple privacy networks to enhance anonymity, censorship resistance, and decentralized communication. This document describes the privacy network integration architecture, configuration, and usage.

## Supported Networks

### 1. Yggdrasil Mesh Network
Yggdrasil is an encrypted IPv6 mesh networking protocol that provides decentralized communication without central infrastructure. It creates a self-organizing, scalable mesh network where each node acts as both client and router.

**Features:**
- End-to-end encrypted IPv6 networking
- Self-organizing mesh topology
- No central servers or infrastructure
- NAT traversal capabilities
- Low latency and high throughput

**Use Cases:**
- Decentralized peer-to-peer communication
- Censorship-resistant networking
- Private mesh networks for communities

### 2. Tor Onion Services
Tor provides anonymity through onion routing, where traffic is encrypted and relayed through multiple nodes before reaching its destination.

**Features:**
- Strong anonymity guarantees
- Hidden services (.onion addresses)
- Traffic analysis resistance
- Mature ecosystem with wide adoption

**Use Cases:**
- Anonymous communication
- Hidden services for private nodes
- Censorship circumvention

### 3. I2P (Invisible Internet Project)
I2P is an anonymous overlay network that uses garlic routing (a variant of onion routing) to provide strong privacy protections.

**Features:**
- Garlic routing for enhanced privacy
- Distributed network database
- Tunnel-based communication
- Resistance to traffic analysis

**Use Cases:**
- Anonymous messaging and file sharing
- Private web browsing (eepSites)
- Censorship-resistant applications

## Architecture

### PrivacyNetworkManager
The `PrivacyNetworkManager` is the central component that manages all privacy network connections. It provides:

- **Network Selection**: Automatic or manual selection of privacy networks
- **Fallback Mechanisms**: Automatic switching between networks based on latency and availability
- **Connection Management**: Unified interface for connecting/disconnecting
- **Status Monitoring**: Real-time network status and statistics

### Network Clients
Each privacy network has a dedicated client implementation:

- `YggdrasilClient`: Manages Yggdrasil mesh network connections
- `TorClient`: Handles Tor onion service creation and communication
- `I2PClient`: Manages I2P tunnel creation and messaging

All clients implement the `PrivacyNetworkClient` trait, providing a consistent interface for network operations.

## Configuration

### PrivacyNetworkConfig
The privacy network configuration is part of the main `NodeConfig`:

```rust
pub struct PrivacyNetworkConfig {
    /// Default network to use (Yggdrasil, Tor, I2P, or Direct)
    pub default_network: PrivacyNetwork,
    
    /// Enable automatic fallback between networks
    pub enable_fallback: bool,
    
    /// Maximum latency in milliseconds before switching networks
    pub max_latency_ms: u64,
    
    /// Enable anonymity features
    pub enable_anonymity: bool,
    
    /// Yggdrasil-specific configuration
    pub yggdrasil: YggdrasilConfig,
    
    /// Tor-specific configuration
    pub tor: TorConfig,
    
    /// I2P-specific configuration
    pub i2p: I2PConfig,
}
```

### Example Configuration (TOML)
```toml
[privacy]
default_network = "Yggdrasil"
enable_fallback = true
max_latency_ms = 1000
enable_anonymity = true

[privacy.yggdrasil]
node_address = "localhost:9001"
peers = [
    "tcp://[200:1234:5678::1]:9001",
    "tcp://[200:abcd:ef01::2]:9001"
]
enable_ipv6 = true
encryption_key = ""

[privacy.tor]
control_port = 9051
socks_port = 9050
enable_hidden_services = true
hidden_service_dir = "/var/lib/tor/hidden_service"
control_password = ""
use_bridges = false
bridges = []

[privacy.i2p]
router_address = "127.0.0.1:7656"
sam_port = 7656
enable_tunnels = true
destination_key = ""
session_name = "neurofed"
max_tunnels = 3
tunnel_length = 3
encrypted_leasesets = true
```

## Usage

### Basic Usage
```rust
use neuro_fed_node::privacy_networks::{PrivacyNetworkManager, PrivacyNetworkConfig};

// Create configuration
let config = PrivacyNetworkConfig::default();

// Initialize manager
let mut manager = PrivacyNetworkManager::new(config);

// Initialize network clients
manager.initialize().await?;

// Connect to default network
manager.connect().await?;

// Get current status
let status = manager.get_status().await;
println!("Network status: {:?}", status);

// Send data through network
manager.send(b"Hello, world!", "destination.address").await?;

// Receive data
let data = manager.receive().await?;

// Switch networks
manager.switch_network(PrivacyNetwork::Tor).await?;

// Disconnect
manager.disconnect().await?;
```

### Network Selection
The system supports automatic network selection based on:
1. **Latency**: Automatically switches if latency exceeds threshold
2. **Availability**: Falls back to alternative networks if current fails
3. **Configuration**: Manual selection via `switch_network()`

### Error Handling
All network operations return `Result<(), PrivacyNetworkError>` with detailed error information:
- `ConnectionFailed`: Network connection issues
- `ConfigurationError`: Invalid configuration
- `Timeout`: Network timeout
- `NotSupported`: Feature not supported by current network
- `AuthenticationFailed`: Authentication failures

## Integration with Main Application

The privacy network manager is integrated into the main NeuroFed Node application:

```rust
// In main.rs
let mut privacy_manager = PrivacyNetworkManager::new(config.privacy_config.clone());
match privacy_manager.initialize().await {
    Ok(_) => {
        info!("Privacy network manager initialized successfully");
        match privacy_manager.connect().await {
            Ok(_) => info!("Connected to privacy network: {:?}", privacy_manager.current_network),
            Err(e) => warn!("Failed to connect to privacy network: {}", e),
        }
    }
    Err(e) => warn!("Failed to initialize privacy network manager: {}", e),
}
```

## Testing

### Unit Tests
Each component includes comprehensive unit tests:
- Network client creation and configuration
- Connection/disconnection lifecycle
- Data sending/receiving (simulated)
- Error handling

### Integration Tests
Integration tests verify:
- Network manager initialization
- Automatic fallback between networks
- Configuration validation
- Performance under simulated network conditions

### Running Tests
```bash
# Run all privacy network tests
cargo test privacy_networks -- --nocapture

# Run specific network tests
cargo test yggdrasil_client
cargo test tor_client
cargo test i2p_client
```

## Dependencies

The privacy network feature requires the following dependencies:

```toml
[dependencies]
yggdrasil = "0.5.0"
tor-client = "0.1.0"
i2p = "0.2.0"
url = "2.5.0"
base64 = "0.22.0"
```

These dependencies are optional and can be enabled via the `privacy` feature flag:

```toml
[features]
privacy = ["yggdrasil", "tor-client", "i2p"]
```

## Performance Considerations

### Latency
- **Yggdrasil**: ~50ms (direct mesh connections)
- **Tor**: ~100ms (multiple relay hops)
- **I2P**: ~150ms (garlic routing overhead)

### Bandwidth
- **Yggdrasil**: High bandwidth (direct connections)
- **Tor**: Moderate bandwidth (relay capacity limited)
- **I2P**: Lower bandwidth (tunnel overhead)

### Resource Usage
- **Yggdrasil**: Moderate memory, low CPU
- **Tor**: High memory, moderate CPU
- **I2P**: Moderate memory, moderate CPU

## Security Considerations

### Anonymity Guarantees
- **Yggdrasil**: Provides encryption but not anonymity (IP visible to peers)
- **Tor**: Strong anonymity through onion routing
- **I2P**: Strong anonymity through garlic routing

### Threat Model
The privacy networks protect against:
- Network surveillance
- Traffic analysis
- Censorship
- IP address tracking

### Limitations
- Network latency may impact real-time applications
- Some networks require external daemons (Tor, I2P)
- Configuration complexity varies by network

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Verify network daemons are running
   - Check firewall settings
   - Validate configuration parameters

2. **High Latency**
   - Consider switching to a different network
   - Adjust `max_latency_ms` configuration
   - Add more peers (Yggdrasil) or relays (Tor)

3. **Authentication Errors**
   - Verify credentials and keys
   - Check file permissions for hidden service directories
   - Ensure proper I2P router configuration

### Logging
Enable debug logging for detailed network diagnostics:
```rust
RUST_LOG=neuro_fed_node=debug,privacy_networks=debug
```

## Future Enhancements

Planned improvements:
1. **Network Metrics**: Detailed performance statistics
2. **Adaptive Routing**: Dynamic network selection based on application needs
3. **Multi-path Routing**: Simultaneous use of multiple networks
4. **Stealth Improvements**: Enhanced anonymity features
5. **Mobile Support**: Optimized for mobile devices and intermittent connectivity

## References

- [Yggdrasil Documentation](https://yggdrasil-network.github.io/)
- [Tor Project](https://www.torproject.org/)
- [I2P Documentation](https://geti2p.net/)
- [NeuroFed Architecture](../architecture.md)