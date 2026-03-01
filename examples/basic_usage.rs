//! Basic usage example for the enhanced ML Engine with ModelManager integration and Federation Modes.
//!
//! This example demonstrates:
//! 1. Creating a ModelManager with default configuration
//! 2. Initializing the MLEngine with ModelManager integration
//! 3. Processing text to generate embeddings
//! 4. Retrieving model information
//! 5. Working with different device types
//! 6. Brain sharing functionality
//! 7. Federation modes (Wallet vs. No-Wallet)

use neuro_fed_node::config::{BrainSharingConfig, NodeConfig};
use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::model_manager::ModelManager;
use neuro_fed_node::types::DeviceType;
use neuro_fed_node::nostr_federation::{NostrFederation, NostrConfig};
use neuro_fed_node::brain_manager::BrainManager;
use neuro_fed_node::federation_manager::{FederationManager, FederationManagerConfig, FederationStrategy};
use neuro_fed_node::payment_verifier::PaymentVerifier;
use neuro_fed_node::pow_verifier::PoWVerifier;
use neuro_fed_node::types::FederationRequest;
use std::sync::Arc;
use tokio;
use std::collections::HashMap;
use std::time::SystemTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NeuroFed Node ML Engine Basic Usage Example ===\n");

    // 1. Create ModelManager with default configuration
    println!("1. Creating ModelManager...");
    let config = NodeConfig::default();
    let manager = ModelManager::new(config);
    
    // Show available memory (if detection works)
    match manager.detect_available_memory().await {
        Ok(memory_mb) => println!("   Available system memory: {} MB", memory_mb),
        Err(e) => println!("   Memory detection failed: {}", e),
    }
    
    // Get recommended model
    match manager.get_recommended_model().await {
        Ok(model) => println!("   Recommended model: {} ({} MB)", model.name, model.size_mb),
        Err(e) => println!("   Model recommendation failed: {}", e),
    }

    // 2. Create MLEngine with ModelManager integration
    println!("\n2. Creating MLEngine with ModelManager integration...");
    let engine = match MLEngine::new_with_manager(Arc::new(manager), "llama-3-8b-instruct").await {
        Ok(engine) => {
            println!("   Engine created successfully");
            engine
        }
        Err(e) => {
            println!("   Failed to create engine with ModelManager: {:?}", e);
            println!("   Falling back to direct initialization...");
            
            // Fallback to direct initialization
            MLEngine::new("models/llama-3-8b-instruct", DeviceType { name: "cpu".to_string(), description: "CPU".to_string(), supported: true })?
        }
    };

    // 3. Get model information
    println!("\n3. Model information:");
    let info = engine.get_model_info();
    for (key, value) in info {
        println!("   {}: {}", key, value);
    }

    // 4. Process some example texts
    println!("\n4. Processing example texts...");
    
    let texts = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming healthcare, finance, and education.",
        "Rust provides memory safety without garbage collection through its ownership system.",
        "Federated learning enables machine learning on decentralized data while preserving privacy.",
        "Predictive coding is a neuroscience-inspired algorithm for unsupervised learning.",
    ];

    for (i, text) in texts.iter().enumerate() {
        println!("\n   Text {}: '{}...'", i + 1, &text[..30].trim_end());
        
        match engine.process_text(text).await {
            Ok(embedding) => {
                let shape = embedding.shape();
                println!("   Embedding shape: {:?}", shape);
                
                // Extract a few values for demonstration
                if let Ok(values) = embedding.flatten_all()?.to_vec1::<f32>() {
                    if values.len() >= 3 {
                        println!("   First 3 values: [{:.4}, {:.4}, {:.4}]", 
                                 values[0], values[1], values[2]);
                    }
                }
            }
            Err(e) => {
                println!("   Error processing text: {:?}", e);
            }
        }
    }

    // 5. Demonstrate caching
    println!("\n5. Demonstrating caching...");
    
    let test_text = "This text will be processed twice to demonstrate caching.";
    
    // First processing (not cached)
    let start = std::time::Instant::now();
    let _embedding1 = engine.process_text(test_text).await?;
    let first_duration = start.elapsed();
    
    // Second processing (should be cached)
    let start = std::time::Instant::now();
    let _embedding2 = engine.process_text(test_text).await?;
    let second_duration = start.elapsed();
    
    println!("   First processing: {:?}", first_duration);
    println!("   Second processing (cached): {:?}", second_duration);
    
    if second_duration < first_duration {
        println!("   Caching provided {:.1}x speedup", 
                 first_duration.as_secs_f64() / second_duration.as_secs_f64());
    }

    // 6. Clear cache and process again
    println!("\n6. Clearing cache...");
    engine.clear_cache();
    
    let start = std::time::Instant::now();
    let _embedding3 = engine.process_text(test_text).await?;
    let third_duration = start.elapsed();
    
    println!("   Processing after cache clear: {:?}", third_duration);

    // 7. Demonstrate different device types (if supported)
    println!("\n7. Device type demonstration:");
    
    let device_types = [
        DeviceType { name: "cpu".to_string(), description: "CPU".to_string(), supported: true },
        DeviceType { name: "cuda".to_string(), description: "CUDA GPU".to_string(), supported: false },
        DeviceType { name: "metal".to_string(), description: "Metal GPU".to_string(), supported: false },
        DeviceType { name: "vulkan".to_string(), description: "Vulkan GPU".to_string(), supported: false },
        DeviceType { name: "auto".to_string(), description: "Best Available".to_string(), supported: true },
    ];
    
    for device_type in device_types.iter() {
        match MLEngine::new("models/qwen2.5-1.5b-instruct", device_type.clone()) {
            Ok(_engine) => {
                println!("   {}: Supported", device_type.description);
            }
            Err(_) => {
                println!("   {}: Not available", device_type.description);
            }
        }
    }

    // 8. Demonstrate brain sharing functionality
    println!("\n8. Brain sharing demonstration:");
    
    // Create a brain manager configuration
    let brain_sharing_config = BrainSharingConfig {
        enabled: true,
        relay_urls: vec![
            "wss://relay.damus.io".to_string(),
            "wss://nostr.wine".to_string(),
        ],
        brain_storage_dir: "./brains".into(),
        cache_dir: "./brain_cache".into(),
        base_model_id: "llama-3-8b-instruct".to_string(),
        allow_untrusted_authors: false,
        max_brain_size: 1024 * 1024 * 1024, // 1GB
    };
    
    // Create Nostr federation (placeholder)
    let nostr_config = NostrConfig::default();
    let nostr_federation = NostrFederation::new(nostr_config);
    let nostr_federation_arc = Arc::new(nostr_federation);
    
    // Create brain manager
    match BrainManager::new(brain_sharing_config, nostr_federation_arc.clone()) {
        Ok(mut brain_manager) => {
            println!("   Brain manager created successfully");
            
            // Create dummy weights for demonstration
            let mut dummy_weights = std::collections::HashMap::new();
            dummy_weights.insert("layer1".to_string(), vec![0.1, 0.2, 0.3]);
            dummy_weights.insert("layer2".to_string(), vec![0.4, 0.5, 0.6]);
            
            // Try to save a brain
            match brain_manager.save_brain(&dummy_weights, "Test brain", vec!["test".to_string()]).await {
                Ok((brain_id, path)) => {
                    println!("   Brain saved successfully:");
                    println!("     Brain ID: {}", brain_id);
                    println!("     Path: {:?}", path);
                    
                    // Try to share the brain
                    match brain_manager.share_brain(&brain_id).await {
                        Ok(event_id) => {
                            println!("   Brain shared successfully:");
                            println!("     Event ID: {}", event_id);
                        }
                        Err(e) => {
                            println!("   Brain sharing failed (expected for demo): {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("   Brain saving failed (expected for demo): {}", e);
                }
            }
            
            // Try to import a brain (this will fail since we don't have real brains)
            match brain_manager.import_brain("dummy_brain_id", Some("llama-3-8b-instruct")).await {
                Ok(path) => {
                    println!("   Brain imported successfully: {:?}", path);
                }
                Err(e) => {
                    println!("   Brain import failed (expected): {}", e);
                }
            }
        }
        Err(e) => {
            println!("   Brain manager creation failed: {}", e);
        }
    }

    // 9. Demonstrate Federation Modes (Wallet vs. No-Wallet)
    println!("\n9. Federation Modes Demonstration:");
    
    // Create federation configuration
    let federation_config = neuro_fed_node::config::FederationConfig {
        strategy: "wallet".to_string(), // or "no_wallet"
        wallet: neuro_fed_node::config::WalletConfig {
            private_key: "nsec1...".to_string(), // Example key
            payment_relays: vec!["wss://relay.damus.io".to_string()],
            min_sats: 1000,
            required_confirmations: 3,
            enable_auto_zap: false,
        },
        pow: neuro_fed_node::config::PoWConfig {
            difficulty: 4,
            timeout_seconds: 30,
            hash_algorithm: "sha256".to_string(),
            enable_dynamic_difficulty: false,
            max_nonce: 1_000_000,
        },
        enable_fallback: true,
        max_retries: 3,
        request_timeout_seconds: 30,
    };
    
    // Create FederationManagerConfig
    let federation_manager_config = FederationManagerConfig {
        strategy: FederationStrategy::WalletMode {
            min_sats: 1000,
            required_confirmations: 3,
        },
        enable_fallback: true,
        max_retries: 3,
        request_timeout_seconds: 30,
    };
    
    // Create verifiers based on strategy
    let payment_verifier = match &federation_manager_config.strategy {
        FederationStrategy::WalletMode { min_sats, required_confirmations } => {
            println!("   Creating PaymentVerifier for Wallet Mode...");
            println!("     Minimum sats: {}, Required confirmations: {}", min_sats, required_confirmations);
            Some(Arc::new(PaymentVerifier::new(
                federation_config.wallet.payment_relays.clone(),
                "npub1examplepublickey".to_string(),
                Some(federation_config.wallet.private_key.clone()),
            )) as Arc<dyn neuro_fed_node::federation_manager::PaymentVerifier>)
        }
        FederationStrategy::NoWalletMode { difficulty, timeout_seconds } => {
            println!("   No-Wallet Mode selected (difficulty: {}, timeout: {}s)", difficulty, timeout_seconds);
            None
        }
    };
    
    let pow_verifier = match &federation_manager_config.strategy {
        FederationStrategy::NoWalletMode { difficulty, timeout_seconds } => {
            println!("   Creating PoWVerifier for No-Wallet Mode...");
            Some(Arc::new(PoWVerifier::new(
                federation_config.pow.hash_algorithm.clone(),
                federation_config.pow.max_nonce,
            )) as Arc<dyn neuro_fed_node::federation_manager::PoWVerifier>)
        }
        FederationStrategy::WalletMode { .. } => None,
    };
    
    // Create brain manager for federation (need to create a new config since brain_sharing_config was moved)
    let brain_sharing_config2 = BrainSharingConfig {
        enabled: true,
        relay_urls: vec![
            "wss://relay.damus.io".to_string(),
            "wss://nostr.wine".to_string(),
        ],
        brain_storage_dir: "./brains".into(),
        cache_dir: "./brain_cache".into(),
        base_model_id: "llama-3-8b-instruct".to_string(),
        allow_untrusted_authors: false,
        max_brain_size: 1024 * 1024 * 1024, // 1GB
    };
    
    let brain_manager = match BrainManager::new(brain_sharing_config2, nostr_federation_arc.clone()) {
        Ok(bm) => Arc::new(bm),
        Err(e) => {
            println!("   Failed to create brain manager for federation: {}", e);
            return Ok(());
        }
    };
    
    // Create FederationManager
    println!("   Creating FederationManager...");
    let mut federation_manager = FederationManager::new(
        federation_manager_config,
        nostr_federation_arc.clone(),
        payment_verifier,
        pow_verifier,
    );
    
    // Demonstrate processing a federation request
    println!("   Demonstrating federation request processing...");
    let federation_request = FederationRequest {
        id: "test-request-123".to_string(),
        request_type: "share".to_string(),
        payment_proof: "test-payment-proof".to_string(),
        pow_proof: "".to_string(),
        timestamp: SystemTime::now(),
        metadata: HashMap::from([
            ("brain_id".to_string(), "test-brain-123".to_string()),
            ("requester_pubkey".to_string(), "npub1requester".to_string()),
        ]),
    };
    
    match federation_manager.process_federation_request(federation_request.clone()).await {
        Ok(response) => {
            println!("   Federation request processed successfully:");
            println!("     Response ID: {}", response.id);
            println!("     Success: {}", response.success);
            println!("     Message: {}", response.message);
            
            // Note: send_federation_response is private, so we can't call it directly
            println!("   Federation response would be sent via Nostr federation");
        }
        Err(e) => {
            println!("   Federation request processing failed: {}", e);
            println!("   This is expected in demo mode without actual verification.");
        }
    }
    
    // Demonstrate switching federation strategies
    println!("\n   Demonstrating strategy switching...");
    let new_strategy = FederationStrategy::NoWalletMode {
        difficulty: 6,
        timeout_seconds: 60,
    };
    federation_manager.switch_strategy(new_strategy);
    println!("     Switched to No-Wallet mode successfully");
    
    // Create a new federation request for No-Wallet mode
    let pow_request = FederationRequest {
        id: "test-request-pow".to_string(),
        request_type: "import".to_string(),
        payment_proof: "".to_string(),
        pow_proof: "0000abcdef1234567890".to_string(), // Example PoW proof
        timestamp: SystemTime::now(),
        metadata: HashMap::from([
            ("brain_id".to_string(), "test-brain-pow".to_string()),
            ("difficulty".to_string(), "4".to_string()),
        ]),
    };
    
    println!("   Processing No-Wallet federation request...");
    match federation_manager.process_federation_request(pow_request.clone()).await {
        Ok(response) => println!("     No-Wallet request processed: {:?}", response),
        Err(e) => println!("     No-Wallet request failed (expected): {}", e),
    }

    // 8. Privacy Network Integration Example
    println!("\n8. Privacy Network Integration Example...");
    use neuro_fed_node::privacy_networks::{PrivacyNetworkManager, PrivacyNetworkConfig, PrivacyNetwork};
    
    // Create privacy network configuration
    let privacy_config = PrivacyNetworkConfig::default();
    println!("   Created default privacy network configuration");
    println!("   Default network: {:?}", privacy_config.default_network);
    println!("   Enable fallback: {}", privacy_config.enable_fallback);
    println!("   Max latency: {} ms", privacy_config.max_latency_ms);
    
    // Initialize privacy network manager
    let mut privacy_manager = PrivacyNetworkManager::new(privacy_config);
    match privacy_manager.initialize().await {
        Ok(_) => println!("   Privacy network manager initialized successfully"),
        Err(e) => println!("   Privacy network manager initialization failed: {}", e),
    }
    
    // Connect to default network
    match privacy_manager.connect().await {
        Ok(_) => println!("   Connected to privacy network: {:?}", privacy_manager.current_network()),
        Err(e) => println!("   Connection failed: {}", e),
    }
    
    // Get network status
    let status = privacy_manager.get_status().await;
    println!("   Network status: {:?}", status);
    
    // Demonstrate network switching
    println!("   Demonstrating network switching...");
    match privacy_manager.switch_network(PrivacyNetwork::Tor).await {
        Ok(_) => println!("   Switched to Tor network"),
        Err(e) => println!("   Network switch failed: {}", e),
    }
    
    // Get latency
    match privacy_manager.get_latency().await {
        Ok(latency) => println!("   Network latency: {:?}", latency),
        Err(e) => println!("   Latency measurement failed: {}", e),
    }
    
    // Disconnect
    match privacy_manager.disconnect().await {
        Ok(_) => println!("   Disconnected from privacy network"),
        Err(e) => println!("   Disconnection failed: {}", e),
    }

    println!("\n=== Example completed successfully ===");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_example_creation() {
        // Simple test to verify the example compiles and basic functionality works
        let config = NodeConfig::default();
        let manager = ModelManager::new(config);
        
        // This should not panic
        let _ = manager.detect_available_memory().await;
        
        // Create engine (may fail if no models are available, which is okay for test)
        let _ = MLEngine::new_with_manager(
            Arc::new(manager),
            "llama-3-8b-instruct"
        ).await;
    }
}