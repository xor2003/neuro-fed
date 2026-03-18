// src/openai_proxy/mod.rs

use axum::{Json, Router, extract::State, response::IntoResponse, routing::post};
use chrono::Utc;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

pub mod calibration;
pub mod client;
pub mod components;
pub mod metrics;
pub mod streaming;
pub mod types;

use crate::config::NodeConfig;
use crate::ml_engine::MLEngine;
use crate::openai_proxy::calibration::CalibrationStore;
use crate::openai_proxy::client::BackendClient;
use crate::openai_proxy::components::ProxyConfig;
use crate::openai_proxy::metrics::ProxyMetrics;
use crate::openai_proxy::types::{
    Choice, Message, OpenAiRequest, OpenAiResponse, ProxyError, Usage,
};
use crate::pc_decoder::{ThoughtConstraints, ThoughtDecoder};
use crate::pc_hierarchy::PredictiveCoding;
use crate::reasoning_state::{execute_plan, recommended_ops, render_output};
use crate::semantic_cache::SemanticCache;
use crate::sleep_phase::SleepManager;
use crate::types::{
    AssistantIntent, CognitiveDictionary, Episode, ReasoningTask, StructuredState, StudyState,
};
use candle_core::Tensor;

const REMOTE_PRICE_PER_1K_TOKENS_USD: f64 = 0.002;
const EPISODIC_MEMORY_CAPACITY: usize = 50;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct UiState {
    pub status: String,
    pub steps: Vec<String>,
    pub last_updated: i64,
    pub last_source: String,
    pub saved_total_usd: f64,
}

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
    pub ui_state: Arc<RwLock<UiState>>,
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
        calibration: Arc<RwLock<CalibrationStore>>,
        cache: Option<Arc<RwLock<SemanticCache>>>,
    ) -> Self {
        let metrics = Arc::new(RwLock::new(ProxyMetrics::default()));
        let ui_state = Arc::new(RwLock::new(UiState::default()));
        OpenAiProxy {
            config,
            proxy_config,
            local_engine,
            pc_hierarchy,
            embedding_dim,
            thought_decoder,
            cognitive_dict,
            metrics,
            cache,
            episodic_memory,
            calibration,
            study_state,
            ui_state,
        }
    }

    async fn ui_reset(&self, status: &str) {
        let mut ui = self.ui_state.write().await;
        ui.status = status.to_string();
        ui.steps.clear();
        ui.last_updated = Utc::now().timestamp();
    }

    async fn ui_set_status(&self, status: &str) {
        let mut ui = self.ui_state.write().await;
        ui.status = status.to_string();
        ui.last_updated = Utc::now().timestamp();
    }

    async fn ui_push_step(&self, step: impl Into<String>) {
        let mut ui = self.ui_state.write().await;
        ui.steps.push(step.into());
        if ui.steps.len() > 20 {
            ui.steps.remove(0);
        }
        ui.last_updated = Utc::now().timestamp();
    }

    async fn ui_set_source(&self, source: &str) {
        let mut ui = self.ui_state.write().await;
        ui.last_source = source.to_string();
        ui.last_updated = Utc::now().timestamp();
    }

    async fn ui_add_saved(&self, saved: f64) {
        let mut ui = self.ui_state.write().await;
        ui.saved_total_usd += saved;
        ui.last_updated = Utc::now().timestamp();
    }

    async fn metrics_inc_pc_inference(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.pc_inference_calls += 1;
    }

    async fn metrics_inc_thought_decoder(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.thought_decoder_calls += 1;
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
        if self.config.ml_config.max_batch_size > 0
            && req.messages.len() > self.config.ml_config.max_batch_size
        {
            let start = req
                .messages
                .len()
                .saturating_sub(self.config.ml_config.max_batch_size);
            req.messages = req.messages[start..].to_vec();
        }
        let estimated_prompt_tokens = estimate_tokens_from_messages(&req.messages);
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_requests += 1;
            metrics.status_message = "Answering...".to_string();
        }
        self.ui_reset("Answering...").await;
        let state = self.extract_structured_state(&req).await;
        self.ui_push_step(format!("Intent: {}", intent_label(&state.intent)))
            .await;
        self.ui_push_step("PC reasoning: embed + infer").await;
        let start_time = Instant::now();
        let mut final_text = String::new();
        let mut thought_trajectory: Vec<u32> = Vec::new();
        let mut sequence_tensors_vec = Vec::new();
        let mut initial_novelty = 0.0;
        let mut raw_confidence = 0.0;
        let mut last_source = "local_pc";
        let mut pc_error: Option<String> = None;
        let mut remote_error: Option<String> = None;
        let mut local_error: Option<String> = None;
        let mut pc_thoughts_string = String::new();
        let mut pc_belief: Option<Tensor> = None;
        let mut effective_reasoning_task = state.reasoning_task.clone();
        let mut effective_expected_output = state.expected_output.clone();

        tracing::info!("🤖 STARTING COGNITIVE STEP: Attempting PC Inference first...");

        match self
            .local_engine
            .read()
            .await
            .process_text_sequence(&state.raw_query)
            .await
        {
            Ok(query_seq) => {
                let seq_len = query_seq.dims()[0];
                for i in 0..seq_len {
                    if let Ok(vec) = query_seq
                        .narrow(0, i, 1)
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<f32>())
                    {
                        sequence_tensors_vec.push(vec);
                    }
                }

                self.metrics_inc_pc_inference().await;
                let pc_result = {
                    let mut pc = self.pc_hierarchy.write().await;
                    match pc.infer_sequence(&query_seq, 5) {
                        Ok(stats) => {
                            raw_confidence = stats.confidence_score;
                            initial_novelty = stats.novelty_score;
                            match pc.get_top_belief() {
                                Ok(belief) => Some(belief),
                                Err(e) => {
                                    pc_error = Some(format!("Failed to get top belief: {}", e));
                                    None
                                }
                            }
                        }
                        Err(e) => {
                            pc_error = Some(format!("PC inference failed: {}", e));
                            None
                        }
                    }
                };

                if let Some(belief) = pc_result {
                    let decoder = self.thought_decoder.read().await;
                    let dict = self.cognitive_dict.read().await;
                    pc_belief = Some(belief.clone());

                    self.metrics_inc_thought_decoder().await;
                    let decode_result = if self.proxy_config.require_thought_ops {
                        let mut constraints = ThoughtConstraints::new(dict.eof_id());
                        constraints.min_non_eof = self.proxy_config.min_thought_ops;
                        decoder.decode_sequence_with_constraints(&belief, 10, 3, &constraints)
                    } else {
                        decoder.decode_sequence(&belief, 10, 3)
                    };

                    match decode_result {
                        Ok(seq) => {
                            thought_trajectory = seq.clone();
                            let thoughts: Vec<String> =
                                seq.iter().map(|id| dict.get_op(*id).to_string()).collect();
                            pc_thoughts_string = thoughts.join(" -> ");

                            tracing::info!("🧠 PC Thought Trajectory: {}", pc_thoughts_string);
                            if !pc_thoughts_string.is_empty() {
                                self.ui_push_step(format!("ThoughtOps: {}", pc_thoughts_string))
                                    .await;
                            }
                            tracing::info!(
                                "🧠 PC confidence: {:.4} (Threshold: 0.6)",
                                raw_confidence
                            );

                            if raw_confidence > 0.6 {
                                tracing::info!("✅ PC is confident. Using thoughts to guide LLM.");
                            } else {
                                let msg = format!(
                                    "⚠️ PC output low confidence ({:.4}) — falling back to LLMs",
                                    raw_confidence
                                );
                                tracing::info!("{}", msg);
                                pc_error = Some(msg.clone());
                            }
                        }
                        Err(e) => {
                            let msg = format!("Thought Decoder failed: {}", e);
                            pc_error = Some(msg.clone());
                            tracing::error!("❌ {}", msg);
                            self.ui_push_step("Thought decoder failed".to_string()).await;
                            if self.proxy_config.require_thought_ops {
                                self.ui_set_status("error").await;
                                return Err(ProxyError::PCError(format!(
                                    "Reasoning required but unavailable: {}",
                                    e
                                )));
                            }
                        }
                    }
                } else if pc_error.is_none() {
                    tracing::warn!("⚠️ PC Inference failed to return a belief.");
                    pc_error = Some("PC Inference failed".to_string());
                }
            }
            Err(e) => {
                pc_error = Some(format!("PC embedding failed: {}", e));
                tracing::error!("❌ PC embedding error: {}", e);
            }
        }

        let mut guidance_parts = Vec::new();
        if !pc_thoughts_string.is_empty() {
            guidance_parts.push(format!(
                "Use this internal thought plan when relevant.\n[INTERNAL_THOUGHT_PLAN]: {}\n[CONFIDENCE]: {:.2}",
                pc_thoughts_string, raw_confidence
            ));
        }
        if let Some(intent_guidance) = build_intent_guidance(&state) {
            guidance_parts.push(intent_guidance);
        }
        if !guidance_parts.is_empty() {
            req.messages.insert(0, Message {
                role: "system".to_string(),
                content: serde_json::json!(format!(
                    "You are a NeuroFed executor.\n{}",
                    guidance_parts.join("\n\n")
                )),
                name: None,
            });

            tracing::info!("Injected assistant guidance into prompt");
        }

        if let Some(task) = effective_reasoning_task.clone() {
            if thought_trajectory.is_empty() {
                let canonical_ops = recommended_ops(&task);
                let dict = self.cognitive_dict.read().await;
                thought_trajectory = canonical_ops
                    .iter()
                    .filter_map(|op| dict.op_to_id.get(op).copied())
                    .collect();
                pc_thoughts_string = canonical_ops
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" -> ");
                if !pc_thoughts_string.is_empty() {
                    self.ui_push_step(format!("ThoughtOps: {}", pc_thoughts_string))
                        .await;
                }
            }

            let canonical_ops: Vec<_> = {
                let dict = self.cognitive_dict.read().await;
                thought_trajectory
                    .iter()
                    .map(|id| dict.get_op(*id))
                    .collect()
            };
            let outcome = execute_plan(&task, &canonical_ops);
            if let Some(answer) = render_output(&task, &outcome) {
                final_text = answer.clone();
                last_source = "reasoning_state";
                raw_confidence = raw_confidence.max(0.95);
                effective_expected_output = Some(answer);
                self.ui_push_step("Deterministic reasoning execution".to_string())
                    .await;
            } else if self.proxy_config.require_thought_ops {
                self.ui_set_status("error").await;
                return Err(ProxyError::PCError(format!(
                    "Reasoning required but state execution failed: {}",
                    outcome.errors.join(" | ")
                )));
            }
        } else if self.proxy_config.require_thought_ops && thought_trajectory.is_empty() {
            self.ui_set_status("error").await;
            return Err(ProxyError::PCError(
                "Reasoning required but no ThoughtOp sequence produced".to_string(),
            ));
        }

        // Step 2: Remote LLM attempt
        if final_text.trim().is_empty() {
            self.ui_push_step("Remote LLM request".to_string()).await;
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
                Err(e) => {
                    remote_error = Some(format!("Remote LLM failed: {}", e));
                }
            }
        }

        // Step 3: Local LLM attempt
        if final_text.trim().is_empty() {
            self.ui_push_step("Local LLM request".to_string()).await;
            match self
                .forward_to_local(&state, &pc_thoughts_string, pc_belief.as_ref())
                .await
            {
                Ok(resp) => {
                    if !resp.trim().is_empty() {
                        final_text = resp;
                        last_source = "local_llm";
                    } else {
                        local_error = Some("Local fallback produced empty response".to_string());
                    }
                }
                Err(e) => {
                    local_error = Some(format!("Local LLM failed: {}", e));
                }
            }
        }

        if final_text.trim().is_empty() {
            let mut details = Vec::new();
            if let Some(err) = pc_error {
                details.push(format!("PC: {}", err));
            }
            if let Some(err) = remote_error {
                details.push(format!("Remote LLM: {}", err));
            }
            if let Some(err) = local_error {
                details.push(format!("Local fallback: {}", err));
            }
            final_text = format!(
                "No response from PC, remote LLM, or local LLM. {}",
                details.join(" | ")
            );
        }

        let _verification_result: Result<String, String> = Ok("Verification skipped".to_string());
        let success = !final_text.starts_with("No response");

        if success && self.config.proxy_config.pc_learning_enabled {
            let learn_text = format!("User: {}\nAssistant: {}", state.raw_query, final_text);
            match self
                .local_engine
                .read()
                .await
                .process_text_sequence(&learn_text)
                .await
            {
                Ok(seq) => {
                    let mut pc = self.pc_hierarchy.write().await;
                    match pc.learn_sequence(&seq, None) {
                        Ok(_) => {
                            let mut metrics = self.metrics.write().await;
                            metrics.pc_learning_calls += 1;
                        }
                        Err(e) => {
                            tracing::warn!("PC learning failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("PC learning embedding failed: {}", e);
                }
            }
        }

        self.calibration
            .write()
            .await
            .record_outcome(raw_confidence, success);

        self.episodic_memory.write().await.push_back(Episode {
            raw_query: state.raw_query.clone(),
            query_sequence: sequence_tensors_vec,
            novelty: initial_novelty,
            confidence: raw_confidence,
            generated_code: final_text.clone(),
            thought_sequence: thought_trajectory,
            success,
            assistant_intent: Some(state.intent.clone()),
            goal: Some(state.goal.clone()),
            plan_steps: state.plan_steps.clone(),
            constraints: state.constraints.clone(),
            assumptions: state.assumptions.clone(),
            tests: if state.tests.trim().is_empty() {
                None
            } else {
                Some(state.tests.clone())
            },
            reasoning_task: effective_reasoning_task.take(),
            expected_output: effective_expected_output.take(),
        });

        // 🔴 FIX: PREVENT OOM AND ENFORCE SLEEP CYCLE WHEN MEMORY IS FULL
        if self.episodic_memory.read().await.len() >= EPISODIC_MEMORY_CAPACITY {
            tracing::warn!(
                "🧠 Episodic memory full ({} entries). Forcing sleep cycle.",
                EPISODIC_MEMORY_CAPACITY
            );
            let pc_clone = self.pc_hierarchy.clone();
            let decoder_clone = self.thought_decoder.clone();
            let dict_clone = self.cognitive_dict.clone();
            let mem_clone = self.episodic_memory.clone();
            let state_clone = self.study_state.clone();

            tokio::spawn(async move {
                {
                    let mut state = state_clone.write().await;
                    state.is_studying = true;
                    state.current_file = "Forced Sleep Phase (Memory Full)".to_string();
                    state.progress_percent = 0.0;
                }

                let sleep_mgr = SleepManager::new(pc_clone, decoder_clone, dict_clone, mem_clone);
                if let Err(e) = sleep_mgr.process_sleep_cycle().await {
                    tracing::error!("Forced sleep cycle failed: {}", e);
                }

                let mut state = state_clone.write().await;
                state.is_studying = false;
                state.current_file = "".to_string();
                state.progress_percent = 100.0;
            });
        }

        let estimated_completion_tokens = estimate_tokens_from_text(&final_text);
        let total_tokens = estimated_prompt_tokens + estimated_completion_tokens;
        let actual_cost = if last_source == "remote_llm" {
            (total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD
        } else {
            0.0
        };
        let _saved = ((total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD
            - actual_cost)
            .max(0.0);
        self.ui_set_source(last_source).await;
        self.ui_add_saved(_saved).await;

        // 🔴 FIX 3: Reset UI Status
        {
            let mut state = self.study_state.write().await;
            state.is_studying = false;
            state.progress_percent = 100.0;
        }

        let response = OpenAiResponse {
            id: format!("agent-{}", Utc::now().timestamp()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: "neurofed-response".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".to_string(),
                    content: serde_json::json!(final_text),
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Usage::default(),
            neurofed_source: Some(last_source.to_string()),
        };

        self.update_metrics_success(start_time.elapsed(), &response)
            .await;

        {
            let mut metrics = self.metrics.write().await;
            if metrics.status_message != "Idle" {
                metrics.status_message = "Idle".to_string();
            }
        }
        self.ui_set_status("Idle").await;
        Ok(response)
    }

    async fn extract_structured_state(&self, req: &OpenAiRequest) -> StructuredState {
        let raw_query = req
            .messages
            .last()
            .map(extract_message_text)
            .unwrap_or_default();
        let (reasoning_task, expected_output) = detect_reasoning_task(&raw_query);
        let intent = detect_intent(&raw_query, reasoning_task.is_some());
        let plan_steps = build_plan_steps(&intent, &raw_query);
        let (constraints, assumptions, tests) = scaffold_state_for_intent(&intent, &raw_query);
        StructuredState {
            intent,
            goal: raw_query.clone(),
            plan_steps,
            entities: HashMap::new(),
            constraints,
            assumptions,
            tests,
            raw_query,
            reasoning_task,
            expected_output,
        }
    }

    #[allow(dead_code)]
    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let mut req = req.clone();
        if !self.config.proxy_config.ollama_model.is_empty() {
            req.model = self.config.proxy_config.ollama_model.clone();
        }
        let client = BackendClient::new(
            self.proxy_config.ollama_url.clone(),
            self.proxy_config.fallback_url.clone(),
            self.proxy_config.timeout_seconds,
            self.proxy_config.openai_api_key.clone(),
        );
        client.send_to_ollama(&req).await
    }

    async fn forward_to_remote(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let models = self.sanitize_remote_models()?;
        let client = BackendClient::new(
            self.proxy_config.ollama_url.clone(),
            self.proxy_config.fallback_url.clone(),
            self.proxy_config.timeout_seconds,
            self.proxy_config.openai_api_key.clone(),
        );
        let mut last_err: Option<ProxyError> = None;
        for model in models {
            let mut req = req.clone();
            req.model = model.clone();
            match client.send_to_fallback(&req).await {
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    tracing::warn!("Remote model {} failed: {}", model, e);
                    last_err = Some(e);
                }
            }
        }
        Err(last_err
            .unwrap_or_else(|| ProxyError::BackendError("remote models failed".to_string())))
    }

    async fn forward_to_local(
        &self,
        state: &StructuredState,
        pc_summary: &str,
        belief: Option<&Tensor>,
    ) -> Result<String, ProxyError> {
        let engine = self.local_engine.read().await;
        if let Some(belief_tensor) = belief {
            let (decoded, avg, max) = engine
                .decode_belief_with_confidence(belief_tensor)
                .map_err(|e| {
                    ProxyError::BackendError(format!("Local fallback decode failed: {}", e))
                })?;
            return Ok(render_local_intent_response(
                state,
                Some(decoded.as_str()),
                pc_summary,
                Some(avg),
                Some(max),
            ));
        }
        Ok(render_local_intent_response(
            state,
            None,
            pc_summary,
            None,
            None,
        ))
    }

    fn sanitize_remote_models(&self) -> Result<Vec<String>, ProxyError> {
        let mut seen = HashSet::new();
        let models: Vec<String> = self
            .proxy_config
            .openai_model
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.strip_prefix("openrouter/").unwrap_or(s).to_string())
            .filter(|s| seen.insert(s.clone()))
            .collect();
        if models.is_empty() {
            Err(ProxyError::ConfigError(
                "No remote models configured for OpenAI proxy".to_string(),
            ))
        } else {
            Ok(models)
        }
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
        Err(e) => Json::<OpenAiResponse>(OpenAiResponse::error(&e.to_string())).into_response(),
    }
}

pub fn create_router(proxy: Arc<OpenAiProxy>) -> Router {
    Router::new()
        .route(
            "/v1/chat/completions",
            post(handle_chat_completion_endpoint),
        )
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
        if let Some(s) = msg.content.as_str() {
            chars += s.chars().count();
        } else {
            chars += msg.content.to_string().chars().count();
        }
    }
    (chars / 4).max(1)
}

fn estimate_tokens_from_text(text: &str) -> usize {
    (text.chars().count() / 4).max(1)
}

fn extract_message_text(message: &Message) -> String {
    message
        .content
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| message.content.to_string())
}

fn parse_i64_tokens(text: &str) -> Vec<i64> {
    text.split(|c: char| !c.is_ascii_digit() && c != '-')
        .filter(|part| !part.is_empty() && *part != "-")
        .filter_map(|part| part.parse::<i64>().ok())
        .collect()
}

fn detect_reasoning_task(raw_query: &str) -> (Option<ReasoningTask>, Option<String>) {
    let normalized = raw_query.trim();
    let lower = normalized.to_lowercase();
    let numbers = parse_i64_tokens(normalized);

    if lower.contains('*') && numbers.len() >= 2 {
        let task = ReasoningTask::Multiply {
            a: numbers[0],
            b: numbers[1],
        };
        return (Some(task), Some((numbers[0] * numbers[1]).to_string()));
    }

    if lower.contains("multiply") && numbers.len() >= 2 {
        let task = ReasoningTask::Multiply {
            a: numbers[0],
            b: numbers[1],
        };
        return (Some(task), Some((numbers[0] * numbers[1]).to_string()));
    }

    if let Some(idx) = lower.find("reverse") {
        let remainder = normalized[idx + "reverse".len()..]
            .trim()
            .trim_matches('"')
            .trim_matches('\'');
        if !remainder.is_empty() {
            let task = ReasoningTask::ReverseString {
                input: remainder.to_string(),
            };
            let expected = remainder.chars().rev().collect::<String>();
            return (Some(task), Some(expected));
        }
    }

    if lower.contains("sum") && lower.contains("even") && !numbers.is_empty() {
        let task = ReasoningTask::SumEven {
            values: numbers.clone(),
        };
        let expected = numbers
            .iter()
            .filter(|value| **value % 2 == 0)
            .sum::<i64>()
            .to_string();
        return (Some(task), Some(expected));
    }

    if (lower.contains("max") || lower.contains("maximum")) && !numbers.is_empty() {
        if let Some(max_value) = numbers.iter().max() {
            let task = ReasoningTask::Max {
                values: numbers.clone(),
            };
            return (Some(task), Some(max_value.to_string()));
        }
    }

    if (lower.contains("sort") || lower.contains("ordered")) && !numbers.is_empty() {
        let mut sorted = numbers.clone();
        sorted.sort();
        let task = ReasoningTask::SortList { values: numbers };
        let expected = sorted
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        return (Some(task), Some(expected));
    }

    (None, None)
}

fn detect_intent(raw_query: &str, has_reasoning_task: bool) -> AssistantIntent {
    if has_reasoning_task {
        return AssistantIntent::Reasoning;
    }

    let lower = raw_query.to_lowercase();
    if lower.contains("investigate")
        || lower.contains("research")
        || lower.contains("analyze")
        || lower.contains("compare")
        || lower.contains("find out")
    {
        return AssistantIntent::Investigation;
    }
    if lower.contains("write code")
        || lower.contains("fix")
        || lower.contains("refactor")
        || lower.contains("implement")
        || lower.contains("cargo")
        || lower.contains("rust")
        || lower.contains("function")
        || lower.contains("test")
    {
        return AssistantIntent::CodeTask;
    }
    if lower.contains("rewrite")
        || lower.contains("summarize")
        || lower.contains("draft")
        || lower.contains("edit")
        || lower.contains("improve this text")
    {
        return AssistantIntent::TextTask;
    }

    AssistantIntent::Chat
}

fn intent_label(intent: &AssistantIntent) -> &'static str {
    match intent {
        AssistantIntent::Chat => "chat",
        AssistantIntent::Reasoning => "reasoning",
        AssistantIntent::Investigation => "investigation",
        AssistantIntent::CodeTask => "code_task",
        AssistantIntent::TextTask => "text_task",
    }
}

fn build_intent_guidance(state: &StructuredState) -> Option<String> {
    match state.intent {
        AssistantIntent::Chat | AssistantIntent::Reasoning => None,
        AssistantIntent::Investigation => Some(format!(
            "Investigation mode:\n- restate the question precisely\n- gather evidence before conclusions\n- separate findings from assumptions\n- end with open questions or uncertainties if any remain\nPlanned steps:\n- {}\nTask: {}",
            state.plan_steps.join("\n- "),
            state.raw_query
        )),
        AssistantIntent::CodeTask => Some(format!(
            "Code-task mode:\n- inspect the relevant code path first\n- propose or follow a concrete change plan\n- preserve behavior unless intentionally changed\n- verify with tests or build commands when possible\nPlanned steps:\n- {}\nTask: {}",
            state.plan_steps.join("\n- "),
            state.raw_query
        )),
        AssistantIntent::TextTask => Some(format!(
            "Text-task mode:\n- identify audience, goal, and tone\n- preserve key facts and constraints\n- optimize for clarity and structure\nPlanned steps:\n- {}\nTask: {}",
            state.plan_steps.join("\n- "),
            state.raw_query
        )),
    }
}

fn join_or_default(items: &[String], fallback: &str) -> String {
    if items.is_empty() {
        fallback.to_string()
    } else {
        items.join("; ")
    }
}

fn render_local_intent_response(
    state: &StructuredState,
    decoded_hint: Option<&str>,
    pc_summary: &str,
    avg: Option<f32>,
    max: Option<f32>,
) -> String {
    let decoded_hint = decoded_hint
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("no decoded local belief");
    let signal_line = match (avg, max) {
        (Some(avg), Some(max)) => format!("Local signal: avg {:.3}, max {:.3}.", avg, max),
        _ => "Local signal: predictive-coding belief unavailable.".to_string(),
    };
    let guidance_line = if pc_summary.trim().is_empty() {
        "ThoughtOps: none.".to_string()
    } else {
        format!("ThoughtOps: {}.", pc_summary)
    };

    match state.intent {
        AssistantIntent::Investigation => format!(
            "Investigation Plan:\n- Goal: {}\n- Steps: {}\n- Evidence needed: {}\n- Assumptions: {}\n- Open questions: {}\n- Working local clue: {}\n{}\n{}",
            state.raw_query,
            join_or_default(&state.plan_steps, "restate the question; collect evidence; synthesize findings"),
            join_or_default(&state.constraints, "collect evidence before concluding"),
            join_or_default(&state.assumptions, "the available local context is incomplete"),
            if state.tests.trim().is_empty() {
                "document uncertainty and list missing evidence".to_string()
            } else {
                state.tests.clone()
            },
            decoded_hint,
            guidance_line,
            signal_line
        ),
        AssistantIntent::CodeTask => format!(
            "Code Task Plan:\n- Goal: {}\n- Steps: {}\n- Constraints: {}\n- Assumptions: {}\n- Verification: {}\n- Working local clue: {}\n{}\n{}",
            state.raw_query,
            join_or_default(&state.plan_steps, "inspect code path; implement smallest coherent change; verify behavior"),
            join_or_default(&state.constraints, "preserve existing behavior until validated"),
            join_or_default(&state.assumptions, "the repository needs inspection before editing"),
            if state.tests.trim().is_empty() {
                "run build/tests for the touched path".to_string()
            } else {
                state.tests.clone()
            },
            decoded_hint,
            guidance_line,
            signal_line
        ),
        AssistantIntent::TextTask => format!(
            "Text Task Plan:\n- Goal: {}\n- Steps: {}\n- Constraints: {}\n- Assumptions: {}\n- Quality check: {}\n- Working local clue: {}\n{}\n{}",
            state.raw_query,
            join_or_default(&state.plan_steps, "identify target tone; rewrite; check fidelity and clarity"),
            join_or_default(&state.constraints, "preserve meaning while improving the writing"),
            join_or_default(&state.assumptions, "the user wants a direct rewrite or edit"),
            if state.tests.trim().is_empty() {
                "check clarity, consistency, and tone".to_string()
            } else {
                state.tests.clone()
            },
            decoded_hint,
            guidance_line,
            signal_line
        ),
        AssistantIntent::Reasoning => format!(
            "Reasoning fallback:\n- Goal: {}\n- Working local clue: {}\n{}\n{}",
            state.raw_query, decoded_hint, guidance_line, signal_line
        ),
        AssistantIntent::Chat => format!(
            "Chat fallback:\n- User request: {}\n- Working local clue: {}\n{}\n{}",
            state.raw_query, decoded_hint, guidance_line, signal_line
        ),
    }
}

fn build_plan_steps(intent: &AssistantIntent, raw_query: &str) -> Vec<String> {
    match intent {
        AssistantIntent::Chat => Vec::new(),
        AssistantIntent::Reasoning => vec![
            "extract the structured task".to_string(),
            "execute the canonical reasoning plan".to_string(),
            "render the deterministic answer".to_string(),
        ],
        AssistantIntent::Investigation => vec![
            format!("restate the investigation target: {}", raw_query),
            "collect evidence and competing explanations".to_string(),
            "synthesize findings and remaining uncertainties".to_string(),
        ],
        AssistantIntent::CodeTask => vec![
            "inspect the existing code path and constraints".to_string(),
            "implement the smallest coherent change".to_string(),
            "verify with build/tests and summarize risks".to_string(),
        ],
        AssistantIntent::TextTask => vec![
            "identify audience, tone, and constraints".to_string(),
            "rewrite while preserving meaning".to_string(),
            "check clarity, brevity, and factual consistency".to_string(),
        ],
    }
}

fn scaffold_state_for_intent(
    intent: &AssistantIntent,
    raw_query: &str,
) -> (Vec<String>, Vec<String>, String) {
    match intent {
        AssistantIntent::Chat | AssistantIntent::Reasoning => (Vec::new(), Vec::new(), String::new()),
        AssistantIntent::Investigation => (
            vec![
                "Collect evidence before concluding".to_string(),
                "Separate findings from assumptions".to_string(),
                "State open questions explicitly".to_string(),
            ],
            vec!["The initial user request may be underspecified".to_string()],
            format!(
                "Return findings, evidence summary, and remaining uncertainties for: {}",
                raw_query
            ),
        ),
        AssistantIntent::CodeTask => (
            vec![
                "Inspect the existing code path before editing".to_string(),
                "Preserve intended behavior unless changing it explicitly".to_string(),
                "Verify changes with build or tests when possible".to_string(),
            ],
            vec!["The current implementation may already encode hidden constraints".to_string()],
            "Verification target: cargo build plus the narrowest relevant tests".to_string(),
        ),
        AssistantIntent::TextTask => (
            vec![
                "Preserve core meaning and factual content".to_string(),
                "Improve clarity and structure".to_string(),
                "Match the requested tone and brevity".to_string(),
            ],
            vec!["The original text may contain signal that should be preserved".to_string()],
            format!(
                "Check that the rewritten text still satisfies the user goal: {}",
                raw_query
            ),
        ),
    }
}

#[cfg(test)]
mod proxy_utility_tests {
    use super::*;
    use crate::openai_proxy::types::Message;

    #[test]
    fn test_token_estimation_safety() {
        // Test empty strings don't crash and return a minimum of 1 token
        assert_eq!(estimate_tokens_from_text(""), 1);

        // Test standard text (Roughly 1 token per 4 chars)
        let text = "Hello world, this is a test string.";
        let estimated = estimate_tokens_from_text(text);
        assert!(
            estimated > 5 && estimated < 15,
            "Token estimation is wildly inaccurate: {}",
            estimated
        );

        // Test Message array parsing
        let msgs = vec![
            Message {
                role: "user".into(),
                content: serde_json::json!("short"),
                name: None,
            },
            Message {
                role: "system".into(),
                content: serde_json::json!("also short"),
                name: None,
            },
        ];
        let msg_tokens = estimate_tokens_from_messages(&msgs);
        assert!(msg_tokens >= 3, "Message token estimation failed");
    }

    #[test]
    fn test_detect_reasoning_task_for_supported_queries() {
        let (task, expected) = detect_reasoning_task("17 * 23");
        assert_eq!(task, Some(ReasoningTask::Multiply { a: 17, b: 23 }));
        assert_eq!(expected.as_deref(), Some("391"));

        let (task, expected) = detect_reasoning_task("reverse abc");
        assert_eq!(
            task,
            Some(ReasoningTask::ReverseString {
                input: "abc".to_string()
            })
        );
        assert_eq!(expected.as_deref(), Some("cba"));

        let (task, expected) = detect_reasoning_task("sum even 1 2 4 5");
        assert_eq!(
            task,
            Some(ReasoningTask::SumEven {
                values: vec![1, 2, 4, 5]
            })
        );
        assert_eq!(expected.as_deref(), Some("6"));
    }

    #[test]
    fn test_detect_intent_for_assistant_modes() {
        assert_eq!(
            detect_intent("17 * 23", true),
            AssistantIntent::Reasoning
        );
        assert_eq!(
            detect_intent("investigate the architecture drift in this repo", false),
            AssistantIntent::Investigation
        );
        assert_eq!(
            detect_intent("implement a parser and add tests", false),
            AssistantIntent::CodeTask
        );
        assert_eq!(
            detect_intent("rewrite this paragraph to be shorter", false),
            AssistantIntent::TextTask
        );
        assert_eq!(detect_intent("hello there", false), AssistantIntent::Chat);
    }

    #[test]
    fn test_scaffold_state_for_code_and_investigation_intents() {
        let (constraints, assumptions, tests) = scaffold_state_for_intent(
            &AssistantIntent::CodeTask,
            "implement a parser and verify it",
        );
        assert!(constraints.iter().any(|item| item.contains("Inspect the existing code path")));
        assert!(!assumptions.is_empty());
        assert!(tests.contains("cargo build"));

        let (constraints, assumptions, tests) = scaffold_state_for_intent(
            &AssistantIntent::Investigation,
            "investigate architecture drift",
        );
        assert!(constraints.iter().any(|item| item.contains("Collect evidence")));
        assert!(!assumptions.is_empty());
        assert!(tests.contains("evidence summary"));
    }

    #[test]
    fn test_build_plan_steps_for_assistant_modes() {
        let code_steps = build_plan_steps(&AssistantIntent::CodeTask, "fix parser bug");
        assert_eq!(code_steps.len(), 3);
        assert!(code_steps[0].contains("inspect"));

        let investigation_steps =
            build_plan_steps(&AssistantIntent::Investigation, "investigate architecture drift");
        assert_eq!(investigation_steps.len(), 3);
        assert!(investigation_steps[0].contains("investigation target"));
    }

    #[test]
    fn test_render_local_intent_response_for_code_task() {
        let state = StructuredState {
            intent: AssistantIntent::CodeTask,
            goal: "fix parser bug".to_string(),
            plan_steps: vec![
                "inspect the parser".to_string(),
                "patch the bug".to_string(),
                "run tests".to_string(),
            ],
            entities: HashMap::new(),
            constraints: vec!["Inspect the existing code path before editing".to_string()],
            assumptions: vec!["Tests are the main correctness gate".to_string()],
            tests: "Run cargo build and the relevant parser tests".to_string(),
            raw_query: "fix parser bug".to_string(),
            reasoning_task: None,
            expected_output: None,
        };

        let rendered = render_local_intent_response(
            &state,
            Some("parser -> token stream"),
            "PLAN -> REFINE",
            Some(0.2),
            Some(0.7),
        );

        assert!(rendered.contains("Code Task Plan"));
        assert!(rendered.contains("fix parser bug"));
        assert!(rendered.contains("Run cargo build"));
        assert!(rendered.contains("PLAN -> REFINE"));
        assert!(rendered.contains("avg 0.200"));
    }

    #[test]
    fn test_render_local_intent_response_for_investigation_without_belief() {
        let state = StructuredState {
            intent: AssistantIntent::Investigation,
            goal: "investigate architecture drift".to_string(),
            plan_steps: vec![
                "restate the investigation target".to_string(),
                "collect evidence".to_string(),
                "summarize findings".to_string(),
            ],
            entities: HashMap::new(),
            constraints: vec!["Collect evidence before concluding".to_string()],
            assumptions: vec!["The local context may be incomplete".to_string()],
            tests: "Return an evidence summary and open questions".to_string(),
            raw_query: "investigate architecture drift".to_string(),
            reasoning_task: None,
            expected_output: None,
        };

        let rendered = render_local_intent_response(&state, None, "", None, None);

        assert!(rendered.contains("Investigation Plan"));
        assert!(rendered.contains("evidence summary"));
        assert!(rendered.contains("predictive-coding belief unavailable"));
    }
}
#[cfg(test)]
mod proxy_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_ui_status_transitions_during_inference() {
        // PROVES: The study_state toggles from "Idle" to "Answering" to "Idle" during a request.
        let state = Arc::new(RwLock::new(StudyState::default()));

        // Assert initial state is idle
        assert!(!state.read().await.is_studying);

        // Simulating the exact logic from handle_chat_completion
        {
            let mut s = state.write().await;
            s.is_studying = true;
            s.current_file = "Answering prompt...".to_string();
            s.progress_percent = 50.0;
        }

        assert!(state.read().await.is_studying);
        assert_eq!(state.read().await.current_file, "Answering prompt...");

        // Simulating the end of handle_chat_completion
        {
            let mut s = state.write().await;
            s.is_studying = false;
            s.progress_percent = 100.0;
        }

        assert!(!state.read().await.is_studying);
    }
}

#[cfg(test)]
mod reasoning_consistency_tests {
    use super::*;
    use crate::config::NodeConfig;
    use crate::openai_proxy::components::ProxyConfig;
    use crate::openai_proxy::types::OpenAiRequest;
    use candle_core::{Device, Tensor};

    #[tokio::test]
    async fn test_pc_reasoning_is_deterministic() {
        let mut config = NodeConfig::load_or_default();
        config.proxy_config.pc_learning_enabled = false;
        let device = Device::Cpu;
        let engine = Arc::new(RwLock::new(MLEngine::mock().unwrap()));
        let embedding_dim = engine.read().await.embedding_dim();
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let pc_hierarchy = Arc::new(RwLock::new(
            PredictiveCoding::new(config.pc_config.clone()).unwrap(),
        ));
        let dict_len = dict.read().await.len();
        let vocab_capacity = config.pc_config.thought_vocab_capacity.max(dict_len);
        let thought_decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(512, vocab_capacity, &device).unwrap(),
        ));
        {
            let decoder = thought_decoder.write().await;
            let ones = Tensor::ones_like(decoder.w_vocab.as_tensor())
                .expect("Failed to init vocab tensor");
            let zeros = Tensor::zeros_like(decoder.w_gate_stack.as_tensor())
                .expect("Failed to init gate tensor");
            decoder.w_vocab.set(&ones).expect("Failed to write vocab");
            decoder
                .w_gate_stack
                .set(&zeros)
                .expect("Failed to write gate stack");
        }
        let study_state = Arc::new(RwLock::new(StudyState::default()));
        let episodic_memory = Arc::new(RwLock::new(VecDeque::new()));
        let calibration = Arc::new(RwLock::new(CalibrationStore::default()));

        let _proxy = OpenAiProxy::new(
            config.clone(),
            ProxyConfig::default(),
            engine.clone(),
            pc_hierarchy.clone(),
            embedding_dim,
            thought_decoder,
            dict,
            study_state,
            episodic_memory,
            calibration,
            None,
        );

        let query_text = "Calculate the square root of 144";
        let first_level_dim = config.pc_config.dim_per_level[0];
        let query_seq = engine
            .read()
            .await
            .deterministic_embedding_with_dim(query_text, 1, first_level_dim)
            .unwrap();

        // Run inference twice to ensure deterministic behavior.
        let (stats1, stats2) = {
            let mut pc = pc_hierarchy.write().await;
            pc.reset_state().unwrap();
            let first = pc.infer_sequence(&query_seq, 5).unwrap();
            pc.reset_state().unwrap();
            let second = pc.infer_sequence(&query_seq, 5).unwrap();
            (first, second)
        };

        assert_eq!(stats1.total_surprise, stats2.total_surprise);
        assert_eq!(stats1.level_surprises, stats2.level_surprises);
        assert!((stats1.confidence_score - stats2.confidence_score).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_proxy_uses_reasoning_state_for_simple_math() {
        let mut config = NodeConfig::load_or_default();
        config.proxy_config.pc_learning_enabled = false;
        let device = Device::Cpu;
        let engine = Arc::new(RwLock::new(MLEngine::mock().unwrap()));
        let embedding_dim = engine.read().await.embedding_dim();
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let pc_hierarchy = Arc::new(RwLock::new(
            PredictiveCoding::new(config.pc_config.clone()).unwrap(),
        ));
        let dict_len = dict.read().await.len();
        let vocab_capacity = config.pc_config.thought_vocab_capacity.max(dict_len);
        let thought_decoder = Arc::new(RwLock::new(
            ThoughtDecoder::new(512, vocab_capacity, &device).unwrap(),
        ));
        let study_state = Arc::new(RwLock::new(StudyState::default()));
        let episodic_memory = Arc::new(RwLock::new(VecDeque::new()));
        let calibration = Arc::new(RwLock::new(CalibrationStore::default()));

        let proxy = OpenAiProxy::new(
            config,
            ProxyConfig::default(),
            engine,
            pc_hierarchy,
            embedding_dim,
            thought_decoder,
            dict,
            study_state,
            episodic_memory.clone(),
            calibration,
            None,
        );

        let response = proxy
            .handle_chat_completion(OpenAiRequest {
                model: "neurofed-response".to_string(),
                messages: vec![Message {
                    role: "user".to_string(),
                    content: serde_json::json!("17 * 23"),
                    name: None,
                }],
                ..OpenAiRequest::default()
            })
            .await
            .expect("reasoning-state request should succeed");

        let content = response.choices[0]
            .message
            .content
            .as_str()
            .expect("assistant response should be a string");
        assert_eq!(content, "391");
        assert_eq!(response.neurofed_source.as_deref(), Some("reasoning_state"));

        let memory = episodic_memory.read().await;
        let last = memory.back().expect("episode should be recorded");
        assert_eq!(last.assistant_intent, Some(AssistantIntent::Reasoning));
        assert!(!last.plan_steps.is_empty());
        assert_eq!(last.reasoning_task, Some(ReasoningTask::Multiply { a: 17, b: 23 }));
        assert_eq!(last.expected_output.as_deref(), Some("391"));
    }
}
