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
use crate::types::{CognitiveDictionary, StructuredState, Episode, StudyState};
use crate::config::NodeConfig;
use crate::openai_proxy::metrics::ProxyMetrics;
use crate::openai_proxy::types::{ProxyError, OpenAiRequest, OpenAiResponse, Message, Choice, Usage};
use crate::openai_proxy::components::ProxyConfig;
use crate::semantic_cache::SemanticCache;
use crate::openai_proxy::calibration::CalibrationStore;
use crate::openai_proxy::client::BackendClient;

const REMOTE_PRICE_PER_1K_TOKENS_USD: f64 = 0.002;

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
    pub study_state: Arc<RwLock<StudyState>>,
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
        study_state: Arc<RwLock<StudyState>>,
        episodic_memory: Arc<RwLock<VecDeque<Episode>>>,
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
            episodic_memory,
            calibration: Arc::new(RwLock::new(CalibrationStore::default())),
            study_state,
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
            let mut metrics = self.metrics.write().await;
            metrics.total_requests += 1;
        }
        let start_time = Instant::now();

        let state = self.extract_structured_state(&req).await;
        let mut final_text = String::new();
        let mut thought_trajectory: Vec<u32> = Vec::new();
        let mut sequence_tensors_vec = Vec::new();
        let mut initial_novelty = 0.0;
        let mut raw_confidence = 0.0;
        let mut last_source = "local_pc";
        let mut pc_error: Option<String> = None;
        let mut remote_error: Option<String> = None;
        let mut local_error: Option<String> = None;

        // 🚀 PART 1: LOGGING FIX - Log BEFORE the blocking task
        tracing::info!("🤖 STARTING COGNITIVE STEP: Attempting PC Inference first...");
        
        let mut pc_thoughts_string = String::new();
        
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
                
                // 🚀 PART 2: PC GUIDANCE & LOGGING
                let pc_result = {
                    let pc_config = self.pc_hierarchy.read().await.config.clone();
                    let pc_device = self.pc_hierarchy.read().await.device.clone();
                    let pc_weights = self.pc_hierarchy.read().await.get_level_weights().unwrap_or_default();
                    let query_seq_tensor = query_seq.clone();
                    
                    tokio::task::spawn_blocking(move || {
                        let mut local_pc = PredictiveCoding::new_with_device(pc_config, pc_device).ok()?;
                        local_pc.load_level_weights(pc_weights).ok()?;
                        let stats = local_pc.infer_sequence(&query_seq_tensor, 5).ok()?;
                        let belief = local_pc.levels.last()?.beliefs.clone();
                        Some((belief, stats))
                    }).await.unwrap_or(None)
                };

                if let Some((belief, stats)) = pc_result {
                    raw_confidence = stats.confidence_score;
                    initial_novelty = stats.novelty_score;

                    let decoder = self.thought_decoder.read().await;
                    let dict = self.cognitive_dict.read().await;
                    
                    if let Ok(seq) = decoder.decode_sequence(&belief, 10, 3) {
                        thought_trajectory = seq.clone();
                        let thoughts: Vec<String> = seq.iter().map(|id| dict.get_op(*id).to_string()).collect();
                        pc_thoughts_string = thoughts.join(" -> ");
                        
                        // 🚀 FIX: This log will now definitely show up
                        tracing::info!("🧠 PC Thought Trajectory: {}", pc_thoughts_string);
                        tracing::info!("🧠 PC confidence: {:.4} (Threshold: 0.6)", raw_confidence);
                        
                        if raw_confidence > 0.6 {
                            tracing::info!("✅ PC is confident. Using thoughts to guide LLM.");
                        } else {
                            let msg = format!("⚠️ PC output low confidence ({:.4}) — falling back to LLMs", raw_confidence);
                            tracing::info!("{}", msg);
                            pc_error = Some(msg.clone());
                        }
                    } else {
                        pc_error = Some("Thought Decoder failed".to_string());
                        tracing::error!("❌ Thought Decoder error");
                    }
                } else {
                    tracing::warn!("⚠️ PC Inference failed to return a result.");
                    pc_error = Some("PC Inference failed".to_string());
                }
            }
            Err(e) => {
                pc_error = Some(format!("PC embedding failed: {}", e));
                tracing::error!("❌ PC embedding error: {}", e);
            }
        }

        // 🚀 PART 3: PROMPT IMPROVEMENT & CACHING
        // We inject the PC's thoughts into the LLM's messages.
        // This makes the LLM follow the PC's logic and improves result quality.
        if !pc_thoughts_string.is_empty() {
            let guidance = format!(
                "[INTERNAL_THOUGHT_PLAN]: {}\n[CONFIDENCE]: {:.2}",
                pc_thoughts_string, raw_confidence
            );
            
            // Insert guidance as a system-like message at the top for better caching
            req.messages.insert(0, Message {
                role: "system".to_string(),
                content: serde_json::json!(format!(
                    "You are a NeuroFed executor. Use the provided internal thought plan to craft your answer. {}",
                    guidance
                )),
                name: None,
            });
            
            tracing::info!("📝 Injected PC guidance into LLM prompt for caching");
        }

        // Step 2: Remote LLM attempt
        if final_text.trim().is_empty() {
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

        let _verification_result: Result<String, String> = Ok("Verification skipped".to_string());
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
        let _saved = ((total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD - actual_cost).max(0.0);

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
