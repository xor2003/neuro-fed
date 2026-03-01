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
use candle_core::{Tensor, Device};
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{info, warn, Level};
use tracing_subscriber;

/// Simple metrics for the dashboard
#[derive(Default)]
struct SimpleMetrics {
    inference_count: u64,
    learning_count: u64,
    free_energy: f32,
}

/// Spawn a background task that periodically prints a runtime metrics dashboard
fn spawn_metrics_dashboard(
    pc_hierarchy: Arc<tokio::sync::Mutex<PredictiveCoding>>,
    metrics: Arc<tokio::sync::Mutex<SimpleMetrics>>,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(TokioDuration::from_secs(5)); // Update every 5 seconds
        
        loop {
            ticker.tick().await;
            
            let pc = pc_hierarchy.lock().await;
            let metrics = metrics.lock().await;
            
            // Clear screen (optional, gives it a top/htop feel)
            print!("\x1B[2J\x1B[1;1H");

            println!("=========================================================");
            println!("🧠 NEUROFED NODE STATUS                 🟢 ONLINE");
            println!("=========================================================\n");
            
            println!("[ COGNITION & PC HIERARCHY ]");
            println!("  Current Free Energy (Surprise): {:.4}", pc.free_energy);
            println!("  Hierarchy Depth:                {} levels", pc.levels.len());
            println!("  Total Inference Cycles:         {}", metrics.inference_count);
            println!("  Total Learning Cycles:          {}", metrics.learning_count);
            println!("  Free Energy (Latest):           {:.4}\n", metrics.free_energy);
            
            println!("[ FEDERATION & NETWORK ]");
            println!("  Privacy Network:                {:?}", "Not implemented in dashboard");
            println!("  Federation Strategy:            {:?}", "Not implemented in dashboard");
            println!("=========================================================");
        }
    });
}

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

    // Create shared state for metrics dashboard
    let pc_hierarchy_arc = Arc::new(tokio::sync::Mutex::new(pc_hierarchy));
    
    // Create a simple metrics struct (in a real implementation, you'd use ProxyMetrics from openai_proxy)
    // For now, we'll just track some basic stats
    let metrics_arc = Arc::new(tokio::sync::Mutex::new(SimpleMetrics::default()));
    
    // Spawn the metrics dashboard
    spawn_metrics_dashboard(pc_hierarchy_arc.clone(), metrics_arc.clone());
    
    // Simple demo loop with dashboard integration
    let mut counter = 0;
    let mut interval = interval(Duration::from_millis(1000));

    loop {
        interval.tick().await;
        counter += 1;

        if counter % 5 == 0 {
            info!("Running inference...");
            let input = Tensor::ones((1, 2048), candle_core::DType::F32, &Device::Cpu)
                .expect("Failed to create input tensor");
            
            // Update metrics
            {
                let mut metrics = metrics_arc.lock().await;
                metrics.inference_count += 1;
            }
            
            // Run inference (blocking, but that's okay for demo)
            let mut pc = pc_hierarchy_arc.lock().await;
            let result = pc.infer(&input, 10);
            if let Ok(stats) = result {
                let mut metrics = metrics_arc.lock().await;
                metrics.free_energy = *stats.free_energy_history.last().unwrap_or(&0.0);
            }
        }

        if counter == 10 {
            info!("Starting bootstrap...");
            let _bootstrap_result = bootstrap.run();
        }

        if counter >= 30 {
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
