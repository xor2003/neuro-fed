// src/main.rs
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use anyhow::Result;
use neuro_fed_node::{
    config::NodeConfig, ml_engine::MLEngine, pc_hierarchy::PredictiveCoding,
    types::{DeviceType, Episode},
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

/// Динамический выбор устройства на основе конфигурации
fn select_device(config: &NodeConfig) -> Device {
    if config.ml_config.use_gpu {
        // Пробуем CUDA, затем Metal, затем CPU
        Device::new_cuda(0)
            .or_else(|_| Device::new_metal(0))
            .unwrap_or(Device::Cpu)
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
    // Ограничение роста уровней (Capping) для стабильности памяти
    let max_dim = 4096; // Жесткий предел
    if !pc_config.dim_per_level.is_empty() {
        pc_config.dim_per_level[0] = embedding_dim.min(max_dim);
        // Применяем лимит ко всем уровням
        for dim in pc_config.dim_per_level.iter_mut() {
            *dim = (*dim).min(max_dim);
        }
    }
    let pc_hierarchy = Arc::new(RwLock::new(PredictiveCoding::new(pc_config)?));

    let cognitive_dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
    let top_belief_dim = *pc_hierarchy.read().await.config.dim_per_level.last().unwrap_or(&embedding_dim);
    let thought_decoder = Arc::new(RwLock::new(ThoughtDecoder::new(
        top_belief_dim,
        cognitive_dict.read().await.len(),
        &device
    )?));

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
    let shutdown = async {
        if tokio::signal::ctrl_c().await.is_ok() {
            tracing::info!("CTRL+C received, shutting down HTTP server...");
        }
    };
    let db_path = config.pc_config.persistence_db_path.as_deref().unwrap_or("./neurofed.db");
    let db_missing = !std::path::Path::new(db_path).exists();
    let has_weights = persistence.has_any_weights().await.unwrap_or(false);
    if config.bootstrap_on_start && (db_missing || !has_weights) {
        let bootstrap = BootstrapManager::new(ml_engine, thought_decoder, cognitive_dict, pc_hierarchy, config.bootstrap_config.clone());
        tokio::spawn(async move {
            if let Err(e) = bootstrap.run_synthetic_training().await {
                tracing::warn!("Bootstrap training failed: {}", e);
            }
        });
    } else if config.bootstrap_on_start && !db_missing && has_weights {
        tracing::info!("Bootstrap skipped: existing DB with weights found at {}", db_path);
    } else if config.bootstrap_on_start && !db_missing && !has_weights {
        tracing::info!("Bootstrap scheduled: DB exists but contains no weights at {}", db_path);
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await?;

    Ok(())
}
