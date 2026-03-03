// src/main.rs
// Main application entry point with CLI

use clap::{Parser, Subcommand};
use neuro_fed_node::bootstrap::{Bootstrap, LlamaContext};
use neuro_fed_node::config::NodeConfig;
use neuro_fed_node::pc_hierarchy;
use neuro_fed_node::PredictiveCoding;
use neuro_fed_node::nostr_federation::NostrFederation;
use neuro_fed_node::brain_manager::BrainManager;
use neuro_fed_node::federation_manager::{FederationManager, FederationManagerConfig, FederationStrategy};
use neuro_fed_node::payment_verifier::PaymentVerifier;
use neuro_fed_node::pow_verifier::PoWVerifier;
use neuro_fed_node::privacy_networks::PrivacyNetworkManager;
use neuro_fed_node::openai_proxy::OpenAiProxy;
use candle_core::{Tensor, Device};
use reqwest;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::{interval, Duration as TokioDuration};
use tracing::{info, warn, Level};
use tracing_subscriber;

/// CLI arguments
#[derive(Parser)]
#[command(name = "neurofed")]
#[command(about = "NeuroFed Node - Decentralized Federated AGI System")]
#[command(version = "0.1.0")]
#[command(arg_required_else_help(false))]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start both proxy daemon and interactive chat (default)
    Default {
        /// Port for the daemon
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
        
        /// Host for the daemon
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    
    /// Start only the NeuroFed daemon (OpenAI proxy server)
    Daemon {
        /// Port to listen on
        #[arg(short, long, default_value_t = 8080)]
        port: u16,
        
        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
    },
    
    /// Start interactive chat with the NeuroFed brain
    Chat {
        /// URL of the NeuroFed daemon
        #[arg(short, long, default_value = "http://127.0.0.1:8080")]
        url: String,
    },
    
    /// Run the full NeuroFed node with all components
    Run {
        /// Enable brain sharing
        #[arg(long)]
        brain_sharing: bool,
        
        /// Enable privacy networks
        #[arg(long)]
        privacy: bool,
    },
}

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
            
            // Print separator and timestamp instead of clearing screen
            let now = chrono::Utc::now();
            println!("\n--- Dashboard Update: {} ---", now.format("%H:%M:%S"));
            println!("🧠 NEUROFED NODE STATUS");
            println!("=========================================================");
            
            println!("[ COGNITION & PC HIERARCHY ]");
            println!("  Current Free Energy (Surprise): {:.4}", pc.free_energy);
            println!("  Hierarchy Depth:                {} levels", pc.levels.len());
            println!("  Total Inference Cycles:         {}", metrics.inference_count);
            println!("  Total Learning Cycles:          {}", metrics.learning_count);
            println!("  Free Energy (Latest):           {:.4}", metrics.free_energy);
            
            println!("[ FEDERATION & NETWORK ]");
            println!("  Privacy Network:                {:?}", "Not implemented in dashboard");
            println!("  Federation Strategy:            {:?}", "Not implemented in dashboard");
            println!("=========================================================\n");
        }
    });
}

/// Start the OpenAI proxy server (daemon mode)
async fn start_daemon(port: u16, host: String) -> Result<(), Box<dyn Error>> {
    info!("Starting NeuroFed daemon on {}:{}", host, port);
    
    // Create components for OpenAI proxy
    let config = NodeConfig::load_or_default();
    let proxy_config = config.proxy_config.clone();
    
    // Create ML Engine
    let device_type = neuro_fed_node::types::DeviceType {
        name: "CPU".to_string(),
        description: "CPU device".to_string(),
        supported: true,
    };
    let local_engine = Arc::new(tokio::sync::Mutex::new(
        neuro_fed_node::ml_engine::MLEngine::new(&config.model_path, device_type)?
    ));
    
    // Create Predictive Coding hierarchy
    let pc_config = pc_hierarchy::PCConfig::new(3, vec![512, 256, 128]);
    let pc_hierarchy = Arc::new(tokio::sync::Mutex::new(
        PredictiveCoding::new(pc_config)?
    ));
    
    info!("NeuroPC Node initialized successfully");

    // Create OpenAI proxy
    let proxy: OpenAiProxy = OpenAiProxy::new(
        config,
        proxy_config,
        local_engine,
        pc_hierarchy,
    );

    info!("Starting OpenAI proxy server on {}:{}", host, port);
    proxy.start(port).await?;
    
    Ok(())
}

/// Start both daemon and chat in parallel
async fn start_both(port: u16, host: String) -> Result<(), Box<dyn Error>> {
    use tokio::task;
    
    // Start daemon in a separate task
    let daemon_handle = task::spawn(async move {
        if let Err(e) = start_daemon(port, host).await {
            eprintln!("Daemon error: {}", e);
        }
    });
    
    // Give daemon a moment to start
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    
    // Start chat in the main task
    let url = format!("http://{}:{}", "127.0.0.1", port);
    println!("Starting chat client connecting to {}", url);
    if let Err(e) = start_chat(url).await {
        eprintln!("Chat error: {}", e);
    }
    
    // Wait for daemon (it will run forever)
    daemon_handle.await?;
    
    Ok(())
}

/// Start interactive chat mode
async fn start_chat(url: String) -> Result<(), Box<dyn Error>> {
    println!("🧠 NeuroFed Chat Client");
    println!("Connecting to daemon at {}...", url);
    println!("Type your messages (type 'quit' or 'exit' to quit):");
    
    let client = reqwest::Client::new();
    let chat_url = format!("{}/v1/chat/completions", url);
    
    // Simple chat loop
    let mut rl = match rustyline::DefaultEditor::new() {
        Ok(editor) => editor,
        Err(_) => {
            eprintln!("Warning: Could not initialize readline, using simple input mode");
            // Create a simple editor without history
            rustyline::Editor::<(), rustyline::history::FileHistory>::new().unwrap()
        }
    };
    
    let mut message_history: Vec<serde_json::Value> = Vec::new();
    
    loop {
        let readline = rl.readline("> ");
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                
                if line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
                    println!("Goodbye!");
                    break;
                }
                
                if line.eq_ignore_ascii_case("clear") {
                    message_history.clear();
                    println!("Conversation cleared.");
                    continue;
                }
                
                if line.eq_ignore_ascii_case("history") {
                    println!("Message history ({} messages):", message_history.len());
                    for (i, msg) in message_history.iter().enumerate() {
                        println!("  {}. {}", i + 1, msg["content"].as_str().unwrap_or("[no content]"));
                    }
                    continue;
                }
                
                // Add user message to history
                let user_message = serde_json::json!({
                    "role": "user",
                    "content": line
                });
                message_history.push(user_message.clone());
                
                // Prepare request
                let request = serde_json::json!({
                    "model": "neurofed",
                    "messages": message_history,
                    "stream": false,
                    "max_tokens": 1000
                });
                
                println!("🤔 Thinking...");
                
                // Send request to daemon
                let start_time = std::time::Instant::now();
                let response = match client.post(&chat_url)
                    .json(&request)
                    .send()
                    .await
                {
                    Ok(resp) => resp,
                    Err(e) => {
                        eprintln!("Error connecting to daemon: {}", e);
                        println!("Make sure the daemon is running with 'neurofed daemon'");
                        continue;
                    }
                };
                
                let elapsed = start_time.elapsed();
                
                if response.status().is_success() {
                    let response_json: serde_json::Value = match response.json().await {
                        Ok(json) => json,
                        Err(e) => {
                            eprintln!("Error parsing response: {}", e);
                            continue;
                        }
                    };
                    
                    // Extract assistant message
                    let choices = response_json["choices"].as_array();
                    let assistant_message = choices.and_then(|c| c.first())
                        .and_then(|c| c["message"].as_object())
                        .cloned();
                    
                    if let Some(msg) = assistant_message {
                        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
                        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("assistant");
                        
                        // Add assistant message to history
                        message_history.push(serde_json::json!({
                            "role": role,
                            "content": content
                        }));
                        
                        // Determine color based on model source
                        let model = response_json.get("model")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown");
                        
                        // Get neurofed source if available
                        let neurofed_source = response_json.get("_neurofed_source")
                            .and_then(|s| s.as_str())
                            .unwrap_or("unknown");
                        
                        // ANSI color codes
                        let green = "\x1b[32m";
                        let yellow = "\x1b[33m";
                        let red = "\x1b[31m";
                        let reset = "\x1b[0m";
                        
                        let (color, source) = match neurofed_source {
                            "pc" => (green, "PC Model"),
                            "remote" => (yellow, "OpenAI Remote"),
                            "local" => (red, "Local Model"),
                            "embedding" => (red, "Embedding"),
                            _ => {
                                // Fallback to model-based detection
                                if model.contains("neurofed") || model.contains("pc") {
                                    (green, "PC Model")
                                } else if model.contains("gpt-3.5") || model.contains("gpt-4") {
                                    (yellow, "OpenAI Remote")
                                } else {
                                    (red, "Local Model")
                                }
                            }
                        };
                        
                        // Print colored response
                        println!("{}🧠 {} (Source: {}){}", color, content, source, reset);
                        println!("📊 Model: {}, Response time: {:.2}s, History: {} messages", model, elapsed.as_secs_f32(), message_history.len());
                    } else {
                        println!("⚠️  No valid response from daemon");
                        println!("📊 Response time: {:.2}s", elapsed.as_secs_f32());
                    }
                } else {
                    let status = response.status();
                    let error_text = response.text().await.unwrap_or_else(|_| "No error details".to_string());
                    eprintln!("Error from daemon ({}): {}", status, error_text);
                    println!("📊 Response time: {:.2}s", elapsed.as_secs_f32());
                }
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("Interrupted (Ctrl+C)");
                break;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("EOF (Ctrl+D)");
                break;
            }
            Err(err) => {
                eprintln!("Error reading line: {}", err);
                break;
            }
        }
    }
    
    Ok(())
}

/// Run full NeuroFed node (original behavior)
async fn run_full_node(brain_sharing: bool, privacy: bool) -> Result<(), Box<dyn Error>> {
    // Brain sharing integration
    let config = NodeConfig::load_or_default();
    
    // Create components using the single GGUF model path
    let model_path = config.model_path.clone();
    let _llama_ctx = LlamaContext::new(&model_path, config.context_size);
    // Convert config::PCConfig to pc_hierarchy::PCConfig
    let pc_config: crate::pc_hierarchy::PCConfig = config.pc_config.clone().into();
    let pc_hierarchy = PredictiveCoding::new(pc_config)?;
    
    // Create bootstrap config with the shared model path
    let bootstrap_config = config.bootstrap_config.clone();
    // Note: bootstrap_config no longer has model_path field
    let _bootstrap = Bootstrap::new(bootstrap_config)
        .expect("Failed to create Bootstrap instance");

    info!("NeuroPC Node initialized successfully");
    if brain_sharing && config.brain_sharing_config.enabled {
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
    if privacy {
        info!("Initializing privacy network manager...");
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
            
            // Update metrics
            let mut metrics = metrics_arc.lock().await;
            metrics.inference_count += 1;
            
            // Run inference
            let mut pc = pc_hierarchy_arc.lock().await;
            // Get input dimension from PC hierarchy config (first level dimension)
            let input_dim = pc.config.dim_per_level[0];
            let input = Tensor::ones((input_dim,), candle_core::DType::F32, &Device::Cpu)
                .expect("Failed to create input tensor");
            
            match pc.infer(&input, 10) {
                Ok(stats) => {
                    metrics.free_energy = stats.total_surprise;
                    info!("Inference completed, free energy: {:.4}", stats.total_surprise);
                }
                Err(e) => warn!("Inference failed: {}", e),
            }
        }
    }
}

fn setup_logging() {
    // Use try_init instead of init to avoid panic if already initialized
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .try_init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    setup_logging();
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Default { port, host }) => {
            start_both(port, host).await
        }
        Some(Commands::Daemon { port, host }) => {
            start_daemon(port, host).await
        }
        Some(Commands::Chat { url }) => {
            start_chat(url).await
        }
        Some(Commands::Run { brain_sharing, privacy }) => {
            run_full_node(brain_sharing, privacy).await
        }
        None => {
            // Default behavior: start both proxy and chat
            start_both(8080, "127.0.0.1".to_string()).await
        }
    }
}