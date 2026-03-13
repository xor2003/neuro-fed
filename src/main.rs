// src/main.rs
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{mpsc, RwLock};
use anyhow::Result;
use thread_priority::*; // <--- NEW IMPORT
use neuro_fed_node::{
    config::{self, NodeConfig}, ml_engine::MLEngine, pc_hierarchy::PredictiveCoding,
    types::{DeviceType, Episode, StudyState},
    persistence::PCPersistence,
    sleep_phase::SleepManager,
    node_loop::NodeLoop,
    openai_proxy::{OpenAiProxy, create_router},
    openai_proxy::components::ProxyConfig,
    pc_decoder::ThoughtDecoder,
    types::CognitiveDictionary,
    metrics,
    ui,
    bootstrap::BootstrapManager,
    federation::nostr_federation::NostrFederation,
    brain_manager::BrainManager,
    federation_manager::{FederationManager, FederationManagerConfig, FederationStrategy},
    payment_verifier::PaymentVerifier,
    pow_verifier::PoWVerifier,
};
use std::collections::VecDeque;
use candle_core::Device;
use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::mpsc::channel;
use std::time::Duration;
use walkdir::WalkDir;

/// Динамический выбор устройства на основе конфигурации
fn select_device(config: &NodeConfig) -> Device {
    if config.ml_config.use_gpu {
        // 1. Try NVIDIA (CUDA)
        if let Ok(dev) = Device::new_cuda(0) {
            return dev;
        }

        // 3. Try Apple Silicon (Metal)
        if let Ok(dev) = Device::new_metal(0) {
            return dev;
        }

        tracing::warn!("GPU requested but no compatible NVIDIA/AMD/Metal device found. Falling back to CPU.");
        Device::Cpu
    } else {
        Device::Cpu
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let config = NodeConfig::load_or_default();
    let filter = tracing_subscriber::EnvFilter::new(config.log_level.clone());
    tracing_subscriber::fmt().with_env_filter(filter).init();
    metrics::init_metrics();
    tracing::info!("🧠 NeuroFed Node - Final Production Build - Starting...");

    // --- 🔴 NEW: Smart N-1 CPU Allocation and Priority ---
    let total_cpus = std::thread::available_parallelism()?.get();
    
    // Calculate N - reserved cores (ensure we have at least 1)
    let target_cpus = total_cpus.saturating_sub(config.ml_config.reserved_cores).max(1);
    let low_priority = config.ml_config.low_priority_learning;

    rayon::ThreadPoolBuilder::new()
        .num_threads(target_cpus)
        .start_handler(move |_thread_idx| {
            // Drop OS thread priority to lowest so it doesn't lag the user's PC
            if low_priority {
                if let Err(e) = set_current_thread_priority(ThreadPriority::Min) {
                    tracing::debug!("Could not set minimum thread priority: {:?}", e);
                }
            }
        })
        .build_global()
        .unwrap();

    tracing::info!(
        "🧠 Brain scaling: Using {} out of {} CPU cores (Reserved: {}, Low Priority: {}).",
        target_cpus, total_cpus, config.ml_config.reserved_cores, low_priority
    );
    // -----------------------------------------------------

    // Create global shutdown signal for graceful termination
    let stop_signal = Arc::new(AtomicBool::new(false));
    let _stop_signal_for_bootstrap = stop_signal.clone(); // Clone for the manager
    
    // Create global shutdown signal for graceful termination
    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_for_bootstrap = stop_signal.clone(); // Clone for the manager
    
    // 1. Инициализация Базы Данных и Персистентности
    let persistence = Arc::new(PCPersistence::new(config.pc_config.persistence_db_path.as_deref().unwrap_or("./neurofed.db")).await?);
    
    // 2. Динамический выбор устройства
    let device = select_device(&config);
    tracing::info!("Using device: {:?}", device);
    
    // 3. Инициализация основных AI-компонентов с tokio::sync::RwLock
    let device_type = DeviceType {
        name: config.ml_config.device_type.clone(),
        description: format!("Device: {}", config.ml_config.device_type),
        supported: config.ml_config.use_gpu,
    };
    let ml_engine = Arc::new(RwLock::new(MLEngine::new(&config.model_path, device_type)?));
    let embedding_dim = ml_engine.read().await.embedding_dim();

    let mut pc_config = config.pc_config.clone();
    
    // --- 🔴 NEW: Configuration Sanitizing Logic ---
    // Ensure dim_per_level array matches n_levels to prevent crashes
    let n_levels = pc_config.n_levels;
    let dims_len = pc_config.dim_per_level.len();
    
    if dims_len < n_levels {
        tracing::warn!(
            "PC config mismatch: n_levels is {} but only {} dimensions are specified. Padding with default values.",
            n_levels, dims_len
        );
        let last_dim = *pc_config.dim_per_level.last().unwrap_or(&512);
        pc_config.dim_per_level.resize(n_levels, last_dim);
    } else if dims_len > n_levels {
        tracing::warn!(
            "PC config mismatch: n_levels is {} but {} dimensions are specified. Truncating.",
            n_levels, dims_len
        );
        pc_config.dim_per_level.truncate(n_levels);
    }
    
    // Ограничение роста уровней (Capping) для стабильности памяти
    let max_dim = 4096; // Жесткий предел
    if !pc_config.dim_per_level.is_empty() {
        // Make sure the first level dimension matches the ML engine's embedding dimension
        pc_config.dim_per_level[0] = embedding_dim.min(max_dim);
        // Применяем лимит ко всем уровням
        for dim in pc_config.dim_per_level.iter_mut() {
            *dim = (*dim).min(max_dim);
        }
    }
    
    // Check database for existing weights and adjust config if needed
    let db_missing = !std::path::Path::new(config.pc_config.persistence_db_path.as_deref().unwrap_or("./neurofed.db")).exists();
    let has_weights = if !db_missing {
        persistence.has_any_weights().await.unwrap_or(false)
    } else {
        false
    };
    
    if has_weights && !db_missing {
        match persistence.load_all_levels().await {
            Ok(saved_weights) => {
                if !saved_weights.is_empty() {
                    let saved_levels_count = saved_weights.len();
                    if saved_levels_count != pc_config.n_levels {
                        tracing::warn!("DB has {} levels but config has {}. Overriding config to match DB.",
                            saved_levels_count, pc_config.n_levels);
                        pc_config.n_levels = saved_levels_count;
                        // Update dim_per_level accordingly - keep existing dimensions or pad with defaults
                        while pc_config.dim_per_level.len() < saved_levels_count {
                            pc_config.dim_per_level.push(512); // default dimension
                        }
                        pc_config.dim_per_level.truncate(saved_levels_count);
                    }
                }
            }
            Err(e) => tracing::warn!("Failed to load weights from database during config check: {}", e),
        }
    }
    
    let pc_hierarchy = Arc::new(RwLock::new(PredictiveCoding::new(pc_config.clone())?));
    
    // 🧬 KNOWLEDGE INJECTION: If starting fresh, steal intelligence from the LLM
    if db_missing || !has_weights {
        tracing::info!("🧬 Attempting to inject pre-trained LLM genetics into PC Level 0...");
        let engine = ml_engine.read().await;
        
        let target_rows = pc_config.dim_per_level[0];
        let target_cols = if pc_config.n_levels > 1 { pc_config.dim_per_level[1] } else { target_rows };
        
        // Extract weights from an early feed-forward block to act as the primary feature extractor
        if let Ok(knowledge) = engine.extract_pretrained_weights("blk.0.ffn_down.weight", target_rows, target_cols) {
            if let Ok(mut knowledge_dev) = knowledge.to_device(&device) {
                // 🔴 FIX: Make contiguous and Scale weights to prevent Gradient Explosion!
                knowledge_dev = knowledge_dev.contiguous().unwrap_or(knowledge_dev);
                
                if let Ok(sqr) = knowledge_dev.sqr() {
                    if let Ok(sum_t) = sqr.sum_all() {
                        if let Ok(sum_sq) = sum_t.to_scalar::<f32>() {
                            let elements = (target_rows * target_cols) as f32;
                            let rms = (sum_sq / elements).sqrt();
                            let target_rms = (1.0 / target_rows as f32).sqrt();
                            
                            // Scale the variance so it safely fits what the Predictive Coding layer expects
                            if rms > 0.0 {
                                let scale = (target_rms / rms) as f64;
                                if let Ok(scaled) = knowledge_dev.affine(scale, 0.0) {
                                    knowledge_dev = scaled;
                                    tracing::info!("⚖️ Scaled injected weights by {:.4} to match PC initialization variance (preventing explosion)", scale);
                                }
                            }
                        }
                    }
                }

                pc_hierarchy.write().await.levels[0].weights = knowledge_dev;
                tracing::info!("✅ Successfully injected LLM knowledge matrix into PC Memory!");
            }
        } else {
            tracing::info!("⚠️ Could not inject LLM weights, starting Tabula Rasa.");
        }
    }

    let cognitive_dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
    let top_belief_dim = *pc_hierarchy.read().await.config.dim_per_level.last().unwrap_or(&embedding_dim);
    let thought_decoder = Arc::new(RwLock::new(ThoughtDecoder::new(
        top_belief_dim,
        cognitive_dict.read().await.len(),
        &device
    )?));

    // 🔴 Create shared StudyState for tracking document study progress
    let study_state = Arc::new(RwLock::new(StudyState::default()));
    tracing::info!("📊 StudyState initialized for UI progress tracking");

    // 3. Создание каналов для Gossip и Событий
    let (gossip_tx, _gossip_rx) = mpsc::channel(100); // gossip_rx пойдет в NostrFederation
    pc_hierarchy.write().await.gossip_sender = Some(gossip_tx);

    let (_user_input_tx, user_input_rx) = mpsc::channel(100);
    let (_file_events_tx, file_events_rx) = mpsc::channel(100);
    let (_nostr_events_tx, nostr_events_rx) = mpsc::channel(100);

    // 4. Инициализация Менеджеров и Циклов с АСИНХРОННЫМИ локами
    let episodic_memory = Arc::new(RwLock::new(VecDeque::<Episode>::new()));
    let sleep_manager = Arc::new(SleepManager::new(
        pc_hierarchy.clone(),
        thought_decoder.clone(),
        cognitive_dict.clone(),
        episodic_memory.clone()
    ));

    let mut node_loop = NodeLoop::new_with_pc_hierarchy(
        user_input_rx, file_events_rx, nostr_events_rx,
        pc_hierarchy.clone(), Some(sleep_manager), Some(episodic_memory.clone()), None
    );

    tokio::spawn(async move {
        node_loop.start().await.expect("Node loop failed");
    });
    tracing::info!("✅ Node Event Loop started.");

    // 5. Запуск HTTP-сервера (OpenAI Proxy)
    let proxy_config = ProxyConfig {
        openai_api_key: config.proxy_config.openai_api_key.clone(),
        openai_model: config.proxy_config.openai_model.clone(),
        ollama_url: config.proxy_config.ollama_base_url.clone(),
        fallback_url: config.proxy_config.openai_base_url.clone(),
        enable_cache: config.proxy_config.semantic_cache_enabled,
        cache_size: config.proxy_config.max_cache_size,
        timeout_seconds: 30,
    };
    let proxy = Arc::new(OpenAiProxy::new(
        config.clone(), proxy_config, ml_engine.clone(), pc_hierarchy.clone(), embedding_dim, thought_decoder.clone(), cognitive_dict.clone(),
        study_state.clone(),
    ));

    // Initialize Nostr federation and brain sharing (config-driven)
    let nostr_federation = Arc::new(NostrFederation::new(config.nostr_config.clone()));
    let federation_strategy = if config.federation_config.strategy == "wallet" {
        FederationStrategy::WalletMode {
            min_sats: config.federation_config.wallet.min_sats,
            required_confirmations: config.federation_config.wallet.required_confirmations,
        }
    } else {
        FederationStrategy::NoWalletMode {
            difficulty: config.federation_config.pow.difficulty,
            timeout_seconds: config.federation_config.pow.timeout_seconds,
        }
    };
    let federation_manager_config = FederationManagerConfig {
        strategy: federation_strategy,
        enable_fallback: config.federation_config.enable_fallback,
        max_retries: config.federation_config.max_retries,
        request_timeout_seconds: config.federation_config.request_timeout_seconds,
    };
    let payment_verifier = Arc::new(PaymentVerifier::new(
        config.federation_config.wallet.payment_relays.clone(),
        config.nostr_config.public_key.clone(),
        Some(config.federation_config.wallet.private_key.clone()),
    ));
    let pow_verifier = Arc::new(PoWVerifier::new(
        config.federation_config.pow.hash_algorithm.clone(),
        config.federation_config.pow.max_nonce,
    ));
    let _federation_manager = FederationManager::new(
        federation_manager_config,
        nostr_federation.clone(),
        Some(payment_verifier),
        Some(pow_verifier),
    );
    let _brain_manager = if config.brain_sharing_config.enabled {
        match BrainManager::new(config.brain_sharing_config.clone(), nostr_federation.clone()) {
            Ok(mgr) => Some(mgr),
            Err(e) => {
                tracing::warn!("BrainManager init failed: {}", e);
                None
            }
        }
    } else {
        None
    };
    let app = create_router(proxy.clone()).merge(ui::create_router(proxy));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    tracing::info!("🚀 NeuroFed API (OpenAI Compatible) listening on 0.0.0.0:8080");
    let db_path = config.pc_config.persistence_db_path.as_deref().unwrap_or("./neurofed.db");
    // Clone for shutdown handler
    let pc_hierarchy_for_shutdown = pc_hierarchy.clone();
    let persistence_for_shutdown = persistence.clone();
    let stop_signal_for_shutdown = stop_signal.clone(); // <--- Capture for shutdown handler
    let shutdown = async move {
        tracing::info!("[DEBUG] Shutdown handler initialized and waiting for Ctrl+C...");
        
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("CTRL+C received, starting graceful shutdown...");
            stop_signal_for_shutdown.store(true, Ordering::SeqCst); // <--- TRIGGER IT
            
            // Immediate test log
            tracing::info!("[DEBUG] Ctrl+C handler triggered, starting save operations");
            
            // 1. Save PC weights
            tracing::info!("Saving PC weights...");
            match pc_hierarchy_for_shutdown.read().await.get_level_weights() {
                Ok(weights) => {
                    let total_levels = weights.len();
                    let mut saved_count = 0;
                    let mut error_count = 0;
                    
                    tracing::info!("Found {} PC levels to save", total_levels);
                    
                    for level in weights {
                        tracing::info!("Saving level {} weights ({}x{} matrix)...",
                            level.level_index, level.input_dim, level.output_dim);
                        
                        if let Err(e) = persistence_for_shutdown.save_level_weights(&level).await {
                            tracing::error!("Failed to save level {} weights: {}", level.level_index, e);
                            error_count += 1;
                        } else {
                            tracing::info!("Successfully saved level {} weights", level.level_index);
                            saved_count += 1;
                        }
                    }
                    
                    if error_count == 0 {
                        tracing::info!("✅ Successfully saved all {} PC weight levels", saved_count);
                    } else {
                        tracing::warn!("⚠️ Saved {}/{} PC weight levels, {} failed", saved_count, total_levels, error_count);
                    }
                }
                Err(e) => tracing::error!("❌ Failed to extract PC weights: {}", e),
            }
            
            // 2. TODO: Save semantic cache if enabled
            tracing::info!("Semantic cache saving not yet implemented");
            
            // 3. TODO: Save trust graph peers
            tracing::info!("Trust graph peers saving not yet implemented");
            
            // 4. TODO: Save delta history
            tracing::info!("Delta history saving not yet implemented");
            
            tracing::info!("Shutdown preparation complete. Terminating process...");
            
            // Small delay to ensure logs are flushed
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            tracing::info!("[DEBUG] Final log before exit");
            exit(0);
        } else {
            tracing::warn!("Ctrl+C await returned error");
        }
    };
    let db_missing = !std::path::Path::new(db_path).exists();
    let has_weights = persistence.has_any_weights().await.unwrap_or(false);
    
    // Load existing weights if available
    if has_weights && !db_missing {
        tracing::info!("Loading existing PC weights from database...");
        match persistence.load_all_levels().await {
            Ok(saved_weights) => {
                if !saved_weights.is_empty() {
                    let weights_clone = saved_weights.clone();
                    match pc_hierarchy.write().await.load_level_weights(saved_weights) {
                        Ok(()) => tracing::info!("Successfully loaded {} level weights from persistence", weights_clone.len()),
                        Err(e) => tracing::warn!("Failed to load PC weights: {}", e),
                    }
                } else {
                    tracing::info!("No weights found in database (empty result)");
                }
            }
            Err(e) => tracing::warn!("Failed to load weights from database: {}", e),
        }
    }
    
    if config.bootstrap_on_start && (db_missing || !has_weights) {
        let mut bootstrap = BootstrapManager::new(
            ml_engine.clone(),
            thought_decoder.clone(),
            cognitive_dict.clone(),
            pc_hierarchy.clone(),
            config.bootstrap_config.clone(),
            study_state.clone(), // <--- ADD STUDY STATE
        );
        bootstrap.shutdown_signal = stop_signal_for_bootstrap; // <--- WIRE IT UP

        tokio::spawn(async move {
            tracing::info!("🚀 Running synthetic bootstrap training...");
            if let Err(e) = bootstrap.run_synthetic_training().await {
                tracing::warn!("Bootstrap synthetic training failed: {}", e);
            }
        });
    } else if config.bootstrap_on_start && !db_missing && has_weights {
        tracing::info!("Bootstrap skipped: existing DB with weights found at {}", db_path);
    } else if config.bootstrap_on_start && !db_missing && !has_weights {
        tracing::info!("Bootstrap scheduled: DB exists but contains no weights at {}", db_path);
    }

    // Start persistent event-driven study system with notify watcher
    if !config.bootstrap_config.document_paths.is_empty() {
        let watch_path = config.bootstrap_config.document_paths[0].clone();
        let watch_path_for_log = watch_path.clone();
        let persistence_clone = persistence.clone();
        let ml_engine_clone = ml_engine.clone();
        let thought_decoder_clone = thought_decoder.clone();
        let cognitive_dict_clone = cognitive_dict.clone();
        let pc_hierarchy_clone = pc_hierarchy.clone();
        let bootstrap_config_clone = config.bootstrap_config.clone();
        
        let study_state_clone = study_state.clone();
        tokio::spawn(async move {
            if let Err(e) = start_file_watcher(
                watch_path,
                persistence_clone,
                ml_engine_clone,
                thought_decoder_clone,
                cognitive_dict_clone,
                pc_hierarchy_clone,
                bootstrap_config_clone,
                study_state_clone,
            ).await {
                tracing::error!("File watcher failed: {}", e);
            }
        });
        tracing::info!("📁 Started persistent event-driven study system watching: {}", watch_path_for_log);
    } else {
        tracing::info!("📁 No document paths configured for persistent study system");
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    Ok(())
}

/// Starts a persistent file watcher that monitors the study directory for new or modified files
async fn start_file_watcher(
    watch_path: String,
    persistence: Arc<PCPersistence>,
    ml_engine: Arc<RwLock<MLEngine>>,
    thought_decoder: Arc<RwLock<ThoughtDecoder>>,
    cognitive_dict: Arc<RwLock<CognitiveDictionary>>,
    pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
    bootstrap_config: config::BootstrapConfig,
    study_state: Arc<RwLock<StudyState>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use notify::event::{CreateKind, ModifyKind, EventKind};
    use std::path::Path;
    
    let (tx, mut rx) = channel(100);
    
    // Create watcher
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            if let Ok(event) = res {
                let _ = tx.blocking_send(event);
            }
        },
        notify::Config::default(),
    )?;
    
    let path = Path::new(&watch_path);
    if !path.exists() {
        tracing::warn!("Watch path does not exist, creating: {}", watch_path);
        std::fs::create_dir_all(path)?;
    }
    
    watcher.watch(path, RecursiveMode::Recursive)?;
    tracing::info!("🔍 File watcher started for: {}", watch_path);
    
    // Create bootstrap manager for processing files
    let bootstrap = BootstrapManager::new(
        ml_engine,
        thought_decoder,
        cognitive_dict,
        pc_hierarchy,
        bootstrap_config,
        study_state.clone(), // <--- ADD STUDY STATE
    );
    
    // Process existing files first
    tracing::info!("📚 Processing existing files in study directory (recursively)...");
    
    // --- 🔴 MODIFIED BLOCK START ---
    if path.is_dir() {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let file_path = entry.path();
                tracing::debug!("Processing existing file: {:?}", file_path);
                if let Ok(Some(chunks)) = bootstrap.process_and_check_file(&file_path, &persistence).await {
                    if let Err(e) = bootstrap.study_file_chunks(&file_path, chunks).await {
                        tracing::warn!("Failed to study file {:?}: {}", file_path, e);
                    }
                }
            }
        }
    }
    // --- 🔴 MODIFIED BLOCK END ---
    
    // Watch for new events
    while let Some(event) = rx.recv().await {
        match event.kind {
            EventKind::Create(CreateKind::File) |
            EventKind::Modify(ModifyKind::Data(_) | ModifyKind::Name(_)) => {
                for path in event.paths {
                    if path.is_file() {
                        tracing::info!("📄 Detected new/modified file: {:?}", path);
                        
                        // Small delay to ensure file is fully written
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        
                        if let Ok(Some(chunks)) = bootstrap.process_and_check_file(&path, &persistence).await {
                            if let Err(e) = bootstrap.study_file_chunks(&path, chunks).await {
                                tracing::warn!("Failed to study file {:?}: {}", path, e);
                            } else {
                                tracing::info!("✅ Successfully studied file: {:?}", path);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod main_architecture_tests {
    use super::*;
    use neuro_fed_node::config::NodeConfig;
    use neuro_fed_node::types::DeviceType;
    
    #[tokio::test]
    async fn test_knowledge_injection_compatibility() -> anyhow::Result<()> {
        let config = NodeConfig::default();
        if !std::path::Path::new(&config.model_path).exists() {
            println!("Skipping injection test: model file missing.");
            return Ok(());
        }
        
        let device = Device::Cpu;
        let device_type = DeviceType { name: "cpu".into(), ..Default::default() };
        let engine = MLEngine::new(&config.model_path, device_type)?;
        
        // Emulate config creation
        let pc_config = neuro_fed_node::config::PCConfig::new(2, vec![2048, 1024]);
        let mut pc = PredictiveCoding::new(pc_config)?;
        
        let initial_weights = pc.levels[0].weights.to_vec2::<f32>()?;
        
        // Perform Injection
        let knowledge = engine.extract_pretrained_weights("blk.0.ffn_down.weight", 2048, 1024)?;
        pc.levels[0].weights = knowledge.to_device(&device)?;
        
        let injected_weights = pc.levels[0].weights.to_vec2::<f32>()?;
        
        // Ensure weights actually changed and dimensions were respected
        assert_ne!(initial_weights, injected_weights, "Knowledge injection failed to overwrite random weights");
        assert_eq!(injected_weights.len(), 2048);
        assert_eq!(injected_weights[0].len(), 1024);
        
        Ok(())
    }
}
