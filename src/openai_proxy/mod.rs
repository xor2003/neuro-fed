use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, Duration};
use tokio::sync::RwLock;
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
    Router,
    routing::post,
};
use serde::Serialize;
use tracing::{info, warn};
use chrono::Utc;

pub mod metrics;
pub mod types;
pub mod components;
pub mod client;
pub mod calibration;
pub mod streaming;

use crate::ml_engine::MLEngine;
use crate::pc_hierarchy::PredictiveCoding;
use crate::pc_decoder::ThoughtDecoder;
use crate::types::{CognitiveDictionary, StructuredState, Episode};
use crate::config::NodeConfig;
use crate::openai_proxy::metrics::ProxyMetrics;
use crate::openai_proxy::types::{ProxyError, OpenAiRequest, OpenAiResponse, Message, Choice, Usage};
use crate::openai_proxy::components::ProxyConfig;
use crate::semantic_cache::SemanticCache;
use crate::openai_proxy::calibration::CalibrationStore;
use crate::openai_proxy::client::BackendClient;

const REMOTE_PRICE_PER_1K_TOKENS_USD: f64 = 0.002;
const LOCAL_PRICE_PER_1K_TOKENS_USD: f64 = 0.0;

/// Main OpenAI proxy struct with integrated Thought Decoder
pub struct OpenAiProxy {
    pub config: NodeConfig,
    pub proxy_config: ProxyConfig,
    pub local_engine: Arc<RwLock<MLEngine>>,
    pub pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
    pub embedding_dim: usize,
    pub thought_decoder: Arc<RwLock<ThoughtDecoder>>,
    pub cognitive_dict: Arc<RwLock<CognitiveDictionary>>,
    pub metrics: Arc<RwLock<ProxyMetrics>>,
    pub cache: Option<Arc<RwLock<SemanticCache>>>,
    
    episodic_memory: Arc<RwLock<VecDeque<Episode>>>,
    calibration: Arc<RwLock<CalibrationStore>>,
    pub ui_state: Arc<RwLock<UiState>>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct UiState {
    pub status: String,
    pub steps: Vec<String>,
    pub last_response: Option<String>,
    pub last_source: Option<String>,
    pub estimated_remote_cost_usd: f64,
    pub last_saved_usd: f64,
    pub saved_total_usd: f64,
    pub progress_current: u32,
    pub progress_total: u32,
    pub progress_percent: f32,
    pub last_updated: i64,
}

impl OpenAiProxy {
    pub fn new(
        config: NodeConfig,
        proxy_config: ProxyConfig,
        local_engine: Arc<RwLock<MLEngine>>,
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        embedding_dim: usize,
        thought_decoder: Arc<RwLock<ThoughtDecoder>>,
        cognitive_dict: Arc<RwLock<CognitiveDictionary>>,
    ) -> Self {
        let metrics = Arc::new(RwLock::new(ProxyMetrics::default()));
        OpenAiProxy {
            config,
            proxy_config,
            local_engine,
            pc_hierarchy,
            embedding_dim,
            thought_decoder,
            cognitive_dict,
            metrics,
            cache: None,
            episodic_memory: Arc::new(RwLock::new(VecDeque::new())),
            calibration: Arc::new(RwLock::new(CalibrationStore::default())),
            ui_state: Arc::new(RwLock::new(UiState::default())),
        }
    }

    /// Main handler with iterative reasoning, calibration, and verification
    pub async fn handle_chat_completion(
        &self,
        req: OpenAiRequest,
    ) -> Result<OpenAiResponse, ProxyError> {
        let mut req = req;
        if req.max_tokens.is_none() {
            req.max_tokens = Some(self.config.context_size);
        }
        if self.config.ml_config.max_batch_size > 0 && req.messages.len() > self.config.ml_config.max_batch_size {
            let start = req.messages.len().saturating_sub(self.config.ml_config.max_batch_size);
            req.messages = req.messages[start..].to_vec();
        }
        let estimated_prompt_tokens = estimate_tokens_from_messages(&req.messages);
        {
            let mut ui = self.ui_state.write().await;
            ui.status = "processing".to_string();
            ui.steps.clear();
            ui.progress_current = 0;
            ui.progress_total = 0;
            ui.progress_percent = 0.0;
            ui.last_updated = Utc::now().timestamp();
        }
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_requests += 1;
        }
        let start_time = Instant::now();

        let state = self.extract_structured_state(&req).await;
        let mut final_text = String::new();
        let thought_trajectory: Vec<u32> = Vec::new();
        let mut sequence_tensors_vec = Vec::new();
        let mut initial_novelty = 0.0;
        let mut raw_confidence = 0.0;
        let mut last_source = "local_pc";
        let mut pc_error: Option<String> = None;
        let mut remote_error: Option<String> = None;
        let mut local_error: Option<String> = None;

        // Step 1: PC-only attempt
        {
            let mut ui = self.ui_state.write().await;
            ui.status = "PC reasoning".to_string();
            ui.steps.push("PC reasoning".to_string());
            ui.progress_total = 3;
            ui.progress_current = 1;
            ui.progress_percent = 33.0;
            ui.last_updated = Utc::now().timestamp();
        }
        
        match self.local_engine.read().await.process_text_sequence(&state.raw_query).await {
            Ok(query_seq) => {
                let seq_len = query_seq.dims()[0];
                for i in 0..seq_len {
                    if let Ok(vec) = query_seq.narrow(0, i, 1)
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<f32>()) {
                        sequence_tensors_vec.push(vec);
                    }
                }
                
                // 🔴 FIX: PARALLEL INFERENCE
                // Instead of locking the main brain, we clone its state into a local thread
                let pc_config = self.pc_hierarchy.read().await.config.clone();
                let pc_device = self.pc_hierarchy.read().await.device.clone();
                let pc_weights = self.pc_hierarchy.read().await.get_level_weights().unwrap_or_default();
                let embedding_dim = self.embedding_dim;
                
                let (belief_opt, stats_opt, err_msg) = tokio::task::spawn_blocking(move || {
                    let mut local_pc = match PredictiveCoding::new_with_device(pc_config, pc_device) {
                        Ok(pc) => pc,
                        Err(e) => return (None, None, Some(format!("Failed to init PC clone: {}", e))),
                    };
                    if let Err(e) = local_pc.load_level_weights(pc_weights) {
                        return (None, None, Some(format!("Failed to load weights into PC clone: {}", e)));
                    }
                    
                    match local_pc.infer_sequence(&query_seq, 5) {
                        Ok(stats) => {
                            let expected_dims =[embedding_dim, 1];
                            let belief = if local_pc.levels.first().map(|l| l.beliefs.dims()) == Some(expected_dims.as_slice()) {
                                local_pc.levels.first().unwrap().beliefs.clone()
                            } else {
                                local_pc.levels.last().unwrap().beliefs.clone()
                            };
                            (Some(belief), Some(stats), None)
                        }
                        Err(e) => (None, None, Some(format!("PC inference failed: {}", e))),
                    }
                }).await.unwrap_or((None, None, Some("Inference task panicked".to_string())));

                if let Some(err) = err_msg {
                    pc_error = Some(err.clone());
                    let mut ui = self.ui_state.write().await;
                    ui.steps.push(err);
                    ui.last_updated = Utc::now().timestamp();
                } else if let (Some(belief), Some(pc_stats)) = (belief_opt, stats_opt) {
                    initial_novelty = pc_stats.novelty_score;
                    raw_confidence = pc_stats.confidence_score;
                    
                    match self.local_engine.read().await.decode_belief_with_confidence(&belief) {
                        Ok((decoded, avg_logit, max_logit)) if !decoded.trim().is_empty() => {
                            // Threshold for babbling vs knowing
                            if max_logit < 5.0 {
                                let msg = format!("PC output low confidence (max={:.4}) — trying remote LLM", max_logit);
                                tracing::info!("{}", msg);
                                pc_error = Some(msg.clone());
                                let mut ui = self.ui_state.write().await;
                                ui.steps.push(msg);
                                ui.last_updated = Utc::now().timestamp();
                                final_text.clear();
                            } else {
                                final_text = decoded;
                                last_source = "local_pc";
                            }
                        }
                        Ok(_) => { pc_error = Some("PC produced empty response".to_string()); }
                        Err(e) => { pc_error = Some(format!("PC decode failed: {}", e)); }
                    }
                }
            }
            Err(e) => { pc_error = Some(format!("PC embedding failed: {}", e)); }
        }

        // Step 2: Remote LLM attempt
        if final_text.trim().is_empty() {
            let mut ui = self.ui_state.write().await;
            ui.status = "Remote LLM request".to_string();
            ui.steps.push("Remote LLM request".to_string());
            ui.progress_current = 2;
            ui.last_updated = Utc::now().timestamp();
            drop(ui);
            match self.forward_to_remote(&req).await {
                Ok(resp) => {
                    if let Some(choice) = resp.choices.first() {
                        let raw = choice.message.content.to_string();
                        if !raw.trim().is_empty() {
                            final_text = raw;
                            last_source = "remote_llm";
                        } else {
                            remote_error = Some("Remote LLM returned empty response".to_string());
                        }
                    } else {
                        remote_error = Some("Remote LLM returned no choices".to_string());
                    }
                }
                Err(e) => { remote_error = Some(format!("Remote LLM failed: {}", e)); }
            }
        }

        // Step 3: Local LLM attempt
        if final_text.trim().is_empty() {
            let mut ui = self.ui_state.write().await;
            ui.status = "Local LLM request".to_string();
            ui.steps.push("Local LLM request".to_string());
            ui.progress_current = 3;
            ui.last_updated = Utc::now().timestamp();
            drop(ui);
            match self.forward_to_local(&req).await {
                Ok(resp) => {
                    if let Some(choice) = resp.choices.first() {
                        let raw = choice.message.content.to_string();
                        if !raw.trim().is_empty() {
                            final_text = raw;
                            last_source = "local_llm";
                        } else {
                            local_error = Some("Local LLM returned empty response".to_string());
                        }
                    } else {
                        local_error = Some("Local LLM returned no choices".to_string());
                    }
                }
                Err(e) => { local_error = Some(format!("Local LLM failed: {}", e)); }
            }
        }

        if final_text.trim().is_empty() {
            let mut details = Vec::new();
            if let Some(err) = pc_error { details.push(format!("PC: {}", err)); }
            if let Some(err) = remote_error { details.push(format!("Remote LLM: {}", err)); }
            if let Some(err) = local_error { details.push(format!("Local LLM: {}", err)); }
            final_text = format!("No response from PC, remote LLM, or local LLM. {}", details.join(" | "));
        }

        let verification_result: Result<String, String> = Ok("Verification skipped".to_string());
        let success = !final_text.starts_with("No response");

        if success && self.config.proxy_config.pc_learning_enabled {
            let learn_text = format!("User: {}\nAssistant: {}", state.raw_query, final_text);
            match self.local_engine.read().await.process_text_sequence(&learn_text).await {
                Ok(seq) => {
                    let mut pc = self.pc_hierarchy.write().await;
                    match pc.learn_sequence(&seq, None) {
                        Ok(_) => {
                            let mut metrics = self.metrics.write().await;
                            metrics.pc_learning_calls += 1;
                        }
                        Err(e) => { tracing::warn!("PC learning failed: {}", e); }
                    }
                }
                Err(e) => { tracing::warn!("PC learning embedding failed: {}", e); }
            }
        }
        
        self.calibration.write().await.record_outcome(raw_confidence, success);

        self.episodic_memory.write().await.push_back(Episode {
            raw_query: state.raw_query.clone(),
            query_sequence: sequence_tensors_vec,
            novelty: initial_novelty,
            confidence: raw_confidence,
            generated_code: final_text.clone(),
            thought_sequence: thought_trajectory,
            success,
        });

        let estimated_completion_tokens = estimate_tokens_from_text(&final_text);
        let total_tokens = estimated_prompt_tokens + estimated_completion_tokens;
        let actual_cost = if last_source == "remote_llm" { (total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD } else { 0.0 };
        let saved = ((total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD - actual_cost).max(0.0);

        {
            let mut ui = self.ui_state.write().await;
            ui.status = if success { "done".to_string() } else { "error".to_string() };
            ui.last_response = Some(final_text.clone());
            ui.last_source = Some(last_source.to_string());
            ui.last_saved_usd = saved;
            ui.saved_total_usd += saved;
            ui.progress_current = 3;
            ui.progress_percent = 100.0;
            ui.last_updated = Utc::now().timestamp();
        }

        let response = OpenAiResponse {
            id: format!("agent-{}", Utc::now().timestamp()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: "neurofed-response".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message { role: "assistant".to_string(), content: serde_json::json!(final_text), name: None },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Usage::default(),
            neurofed_source: Some(last_source.to_string()),
        };
        
        self.update_metrics_success(start_time.elapsed(), &response).await;
        Ok(response)
    }

    async fn extract_structured_state(&self, req: &OpenAiRequest) -> StructuredState {
        let raw_query = req.messages.last().unwrap().content.to_string();
        StructuredState { goal: raw_query.clone(), entities: HashMap::new(), constraints: Vec::new(), assumptions: Vec::new(), tests: "".to_string(), raw_query }
    }

    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let mut req = req.clone();
        if !self.config.proxy_config.ollama_model.is_empty() {
            req.model = self.config.proxy_config.ollama_model.clone();
        }
        let client = BackendClient::new(
            self.proxy_config.ollama_url.clone(),
            self.proxy_config.fallback_url.clone(),
            self.proxy_config.timeout_seconds,
            self.proxy_config.openai_api_key.clone()
        );
        client.send_to_ollama(&req).await
    }

    async fn forward_to_remote(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let models = self.remote_models();
        let client = BackendClient::new(
            self.proxy_config.ollama_url.clone(),
            self.proxy_config.fallback_url.clone(),
            self.proxy_config.timeout_seconds,
            self.proxy_config.openai_api_key.clone()
        );
        let mut last_err: Option<ProxyError> = None;
        for model in models {
            let mut req = req.clone();
            req.model = model;
            match client.send_to_fallback(&req).await {
                Ok(resp) => return Ok(resp),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| ProxyError::BackendError("remote models failed".to_string())))
    }

    async fn forward_to_local(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        self.forward_to_ollama(req).await
    }

    fn remote_models(&self) -> Vec<String> {
        self.proxy_config.openai_model.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).map(|s| s.strip_prefix("openrouter/").unwrap_or(&s).to_string()).collect()
    }

    async fn update_metrics_success(&self, elapsed: Duration, _response: &OpenAiResponse) {
        let mut metrics = self.metrics.write().await;
        metrics.total_processing_time_ms += elapsed.as_millis() as u64;
    }
}

pub async fn handle_chat_completion_endpoint(
    State(proxy): State<Arc<OpenAiProxy>>,
    Json(req): Json<OpenAiRequest>,
) -> impl IntoResponse {
    match proxy.handle_chat_completion(req).await {
        Ok(response) => Json::<OpenAiResponse>(response).into_response(),
        Err(e) => Json::<OpenAiResponse>(OpenAiResponse::error(&e.to_string())).into_response()
    }
}

pub fn create_router(proxy: Arc<OpenAiProxy>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion_endpoint))
        .with_state(proxy)
}

impl Default for OpenAiProxy {
    fn default() -> Self {
        panic!("OpenAiProxy cannot be default-initialized")
    }
}

fn estimate_tokens_from_messages(messages: &[Message]) -> usize {
    let mut chars = 0usize;
    for msg in messages {
        if let Some(s) = msg.content.as_str() { chars += s.chars().count(); } else { chars += msg.content.to_string().chars().count(); }
    }
    (chars / 4).max(1)
}

fn estimate_tokens_from_text(text: &str) -> usize {
    (text.chars().count() / 4).max(1)
}
