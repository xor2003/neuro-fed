// src/main.rs
// Main application entry point

use neuro_fed_node::bootstrap::{Bootstrap, BootstrapConfig, LlamaContext};
use neuro_fed_node::pc_hierarchy::PCConfig;
use neuro_fed_node::PredictiveCoding;
use neuro_fed_node::config::NodeConfig;
use neuro_fed_node::nostr_federation::NostrFederation;
use neuro_fed_node::brain_manager::BrainManager;
use neuro_fed_node::federation_manager::{FederationManager, FederationManagerConfig, FederationStrategy};
use neuro_fed_node::payment_verifier::PaymentVerifier;
use neuro_fed_node::pow_verifier::PoWVerifier;
use neuro_fed_node::privacy_networks::PrivacyNetworkManager;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{info, warn, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    setup_logging();

    // Create components
    let _llama_ctx = LlamaContext::new("models/gguf_model.gguf", 2048);
    let mut pc_hierarchy = PredictiveCoding::new(PCConfig::new(3, vec![2048, 1024, 512]))?;
    let mut bootstrap = Bootstrap::new(BootstrapConfig::new(
        "models/gguf_model.gguf".to_string(),
        2048,
        vec!["./data".to_string()],
    ))
    .expect("Failed to create Bootstrap instance");

    info!("NeuroPC Node initialized successfully");

    // Brain sharing integration
    let config = NodeConfig::default();
    if config.brain_sharing_config.enabled {
        info!("Brain sharing enabled, initializing brain manager...");
        let nostr_federation = NostrFederation::new(config.nostr_config.clone());
        let nostr_federation_arc = Arc::new(nostr_federation);
        match BrainManager::new(config.brain_sharing_config.clone(), nostr_federation_arc) {
            Ok(brain_manager) => {
                info!("Brain manager initialized successfully.");
                // Example: list brains
                let brains = brain_manager.list_brains();
                info!("Known brains: {}", brains.len());
                // TODO: integrate brain sharing into main loop
            }
            Err(e) => warn!("Failed to initialize brain manager: {}", e),
        }
    } else {
        info!("Brain sharing disabled (see brain_sharing_config.enabled).");
    }

    // Federation manager integration
    info!("Initializing federation manager with strategy: {:?}", config.federation_config.strategy);
    let nostr_federation = Arc::new(NostrFederation::new(config.nostr_config.clone()));
    
    let federation_manager = match config.federation_config.strategy.as_str() {
        "wallet" => {
            let payment_verifier = Arc::new(PaymentVerifier::new(
                config.federation_config.wallet.payment_relays.clone(),
                config.nostr_config.public_key.clone(),
                Some(config.nostr_config.private_key.clone()),
            ));
            FederationManager::new(
                FederationManagerConfig {
                    strategy: FederationStrategy::WalletMode {
                        min_sats: config.federation_config.wallet.min_sats,
                        required_confirmations: config.federation_config.wallet.required_confirmations,
                    },
                    enable_fallback: config.federation_config.enable_fallback,
                    max_retries: config.federation_config.max_retries,
                    request_timeout_seconds: config.federation_config.request_timeout_seconds,
                },
                nostr_federation,
                Some(payment_verifier),
                None,
            )
        }
        "no_wallet" => {
            let pow_verifier = Arc::new(PoWVerifier::new(
                config.federation_config.pow.hash_algorithm.clone(),
                config.federation_config.pow.max_nonce,
            ));
            FederationManager::new(
                FederationManagerConfig {
                    strategy: FederationStrategy::NoWalletMode {
                        difficulty: config.federation_config.pow.difficulty,
                        timeout_seconds: config.federation_config.pow.timeout_seconds,
                    },
                    enable_fallback: config.federation_config.enable_fallback,
                    max_retries: config.federation_config.max_retries,
                    request_timeout_seconds: config.federation_config.request_timeout_seconds,
                },
                nostr_federation,
                None,
                Some(pow_verifier),
            )
        }
        _ => {
            warn!("Unknown federation strategy '{}', defaulting to wallet mode", config.federation_config.strategy);
            let payment_verifier = Arc::new(PaymentVerifier::default());
            FederationManager::new(
                FederationManagerConfig::default(),
                nostr_federation,
                Some(payment_verifier),
                None,
            )
        }
    };

    info!("Federation manager initialized with strategy: {:?}", federation_manager.strategy());

    // Privacy network integration
    info!("Initializing privacy network manager...");
    let mut privacy_manager = PrivacyNetworkManager::new(config.privacy_config.clone());
    match privacy_manager.initialize().await {
        Ok(_) => {
            info!("Privacy network manager initialized successfully");
            match privacy_manager.connect().await {
                Ok(_) => info!("Connected to privacy network: {:?}", privacy_manager.current_network()),
                Err(e) => warn!("Failed to connect to privacy network: {}", e),
            }
        }
        Err(e) => warn!("Failed to initialize privacy network manager: {}", e),
    }

    // Simple demo loop
    let mut counter = 0;
    let mut interval = interval(Duration::from_millis(1000));

    loop {
        interval.tick().await;
        counter += 1;

        if counter % 5 == 0 {
            info!("Running inference...");
            let input = ndarray::Array2::ones((1, 2048));
            let _result = pc_hierarchy.infer(&input, 10);
        }

        if counter == 10 {
            info!("Starting bootstrap...");
            let _bootstrap_result = bootstrap.run();
        }

        if counter >= 20 {
            info!("Demo complete, shutting down...");
            break;
        }
    }

    Ok(())
}

fn setup_logging() {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();
}
