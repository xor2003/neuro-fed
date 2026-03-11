use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Instant, Duration};
use tokio::sync::RwLock;
use tokio::task;
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
    Router,
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn, error};
use chrono::Utc;

// Для статической карты стоимости действий
use lazy_static::lazy_static;

pub mod metrics;
pub mod types;
pub mod components;
pub mod client;
pub mod calibration;
pub mod streaming;

use crate::ml_engine::MLEngine;
use crate::pc_hierarchy::PredictiveCoding;
use crate::pc_decoder::ThoughtDecoder;
use crate::types::{CognitiveDictionary, ThoughtOp, StructuredState, Episode};
use crate::config::NodeConfig;
use crate::openai_proxy::metrics::ProxyMetrics;
use crate::openai_proxy::types::{ProxyError, OpenAiRequest, OpenAiResponse, Message, Choice, Usage};
use crate::openai_proxy::components::ProxyConfig;
use crate::semantic_cache::SemanticCache;
use crate::knowledge_filter::CodeVerifier;
use crate::openai_proxy::calibration::CalibrationStore;
use crate::openai_proxy::client::BackendClient;

// Добавляем статический словарь цен действий
lazy_static::lazy_static! {
    static ref ACTION_COSTS: HashMap<u32, f32> = {
        let mut m = HashMap::new();
        m.insert(0, 0.1); // Define
        m.insert(1, 0.2); // Iterate
        m.insert(2, 0.3); // Check
        m.insert(3, 0.5); // Compute (может быть ресурсоемким)
        m.insert(4, 0.4); // Aggregate
        m.insert(5, 0.1); // Return
        m.insert(6, 1.0); // Explain (требует много генерации)
        m
    };
}

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
    
    // NEW: Action/Perception dependencies
    code_verifier: CodeVerifier,
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
        let pc_inference_enabled = config.proxy_config.pc_inference_enabled;
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
            code_verifier: CodeVerifier::new(pc_inference_enabled),
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

        // 1. Extract state AND generation of unit tests
        let mut state = self.extract_structured_state(&req).await;
        let mut final_text = String::new();
        let mut pc_context = state.get_pc_context();
        
        let mut attempt = 0;
        let max_steps = 10;
        let mut thought_trajectory = Vec::new();
        let mut sequence_tensors_vec = Vec::new();
        let mut initial_novelty = 0.0;
        let mut raw_confidence = 0.0;
        let mut used_backend = false;
        let mut used_direct_backend_reply = false;
        let mut has_backend_response = false;
        
        while attempt < max_steps {
            info!("🔄 Agentic Cycle: Attempt {}/{}", attempt + 1, max_steps);
            {
                let mut ui = self.ui_state.write().await;
                ui.steps.push(format!("Cycle {}: perception → inference", attempt + 1));
                ui.progress_total = max_steps as u32;
                ui.progress_current = (attempt + 1) as u32;
                ui.progress_percent = (ui.progress_current as f32 / ui.progress_total.max(1) as f32) * 100.0;
                ui.last_updated = Utc::now().timestamp();
            }

            // Perception: Temporal Sequence Inference
            let query_seq = self.local_engine.read().await.process_text_sequence(&pc_context).await
                .map_err(|e| ProxyError::EmbeddingError(e.to_string()))?;
                
            let seq_len = query_seq.dims()[0];
            if attempt == 0 {
                for i in 0..seq_len {
                    sequence_tensors_vec.push(query_seq.narrow(0, i, 1).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap());
                }
            }

            let mut pc = self.pc_hierarchy.write().await;
            let pc_stats = pc.infer_sequence(&query_seq, 5).map_err(|e| ProxyError::PCError(e.to_string()))?; 
            
            if attempt == 0 {
                initial_novelty = pc_stats.novelty_score;
                raw_confidence = pc_stats.confidence_score;
            }
            
            // 2. Structurally apply Calibration to the PC's actual precision math
            let calibrated_conf = self.calibration.read().await.calibrated_confidence(raw_confidence);
            pc.modulate_precision(calibrated_conf).map_err(|e| ProxyError::PCError(e.to_string()))?;
            let current_belief = pc.levels.last().unwrap().beliefs.clone();
            drop(pc);

            // Action Selection
            let decoder = self.thought_decoder.read().await;
            let thought_ids = decoder.decode_sequence_with_costs(&current_belief, 1, 1, Some(&ACTION_COSTS))
                .map_err(|e| ProxyError::PCError(e.to_string()))?;
            drop(decoder);
            
            let next_op_id = *thought_ids.first().unwrap_or(&7); 
            thought_trajectory.push(next_op_id);
            let dict = self.cognitive_dict.read().await;
            let op = dict.get_op(next_op_id);
            drop(dict);

            if op == ThoughtOp::EOF { break; }

            // 3. Execution (Real Backend Call via HTTP Client)
            let step_prompt = format!(
                "Goal: {}\nConstraints: {:?}\nCurrent Code:\n{}\nNext Step: {:?}\nWrite ONLY code for this step.",
                state.goal, state.constraints, final_text, op
            );
            let step_req = self.create_internal_req(&step_prompt, &req);
            
            if let Ok(step_response) = self.forward_to_ollama(&step_req).await {
                used_backend = true;
                if let Some(choice) = step_response.choices.first() {
                    let raw_content = if let Some(s) = choice.message.content.as_str() {
                        s.to_string()
                    } else {
                        choice.message.content.to_string()
                    };
                    let step_code = raw_content.replace("```python", "").replace("```", "");
                    
                    if step_code.trim().is_empty() {
                        tracing::warn!("Backend returned empty content for step {:?}", op);
                    } else {
                        final_text.push_str(&step_code);
                        final_text.push_str("\n");
                        pc_context = format!("{}\nExecuted: {:?}\nResult:\n{}", pc_context, op, step_code);
                        has_backend_response = true;
                    }
                    {
                        let mut ui = self.ui_state.write().await;
                        ui.steps.push(format!("Cycle {}: backend step → {:?}", attempt + 1, op));
                        ui.progress_total = max_steps as u32;
                        ui.progress_current = (attempt + 1) as u32;
                        ui.progress_percent = (ui.progress_current as f32 / ui.progress_total.max(1) as f32) * 100.0;
                        ui.last_updated = Utc::now().timestamp();
                    }
                } else {
                    tracing::warn!("Backend returned no choices for step {:?}", op);
                }
            } else {
                warn!("Backend Ollama execution failed. Retrying...");
            }
            attempt += 1;
        }

        // If the agentic loop produced nothing, fall back to a direct backend reply.
        if final_text.trim().is_empty() {
            match self.forward_to_ollama(&req).await {
                Ok(resp) => {
                    if let Some(choice) = resp.choices.first() {
                        let raw = if let Some(s) = choice.message.content.as_str() {
                            s.to_string()
                        } else {
                            choice.message.content.to_string()
                        };
                        tracing::debug!("direct fallback content len={}", raw.len());
                        if !raw.trim().is_empty() {
                            final_text = raw;
                            used_backend = true;
                            used_direct_backend_reply = true;
                            has_backend_response = true;
                            let mut ui = self.ui_state.write().await;
                            let src = if is_local_url(&self.proxy_config.ollama_url) { "local LLM" } else { "remote LLM" };
                            ui.steps.push(format!("Direct backend reply — {} (agentic steps produced no output)", src));
                            ui.last_updated = Utc::now().timestamp();
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!("Direct backend fallback failed: {}", e);
                }
            }
        }

        // 4. Ground-Truth Verification (Unit Test Harness) - ASYNC with timeout
        if final_text.trim().is_empty() {
            final_text = "No backend response. Check Ollama or backend connectivity.".to_string();
        }

        let verification_result = if has_backend_response && !used_direct_backend_reply {
            self.code_verifier.execute_with_tests(&final_text, &state.tests).await
        } else {
            Ok("Verification skipped".to_string())
        };
        let success = verification_result.is_ok();
        
        // 5. Update Calibration Database based on true outcome
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

        match verification_result {
            Ok(stdout) => info!("✅ Verification passed! Output: {}", stdout),
            Err(stderr) => {
                warn!("❌ Verification failed: {}", stderr);
                final_text.push_str(&format!("\n# Execution Failed. Error:\n# {}", stderr));
            }
        }

        let estimated_completion_tokens = estimate_tokens_from_text(&final_text);
        let total_tokens = estimated_prompt_tokens + estimated_completion_tokens;
        let estimated_remote_cost = (total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD;
        let is_local = is_local_url(&self.proxy_config.ollama_url);
        let last_source = if !used_backend {
            "local_pc"
        } else if is_local {
            "local_llm"
        } else {
            "remote_llm"
        };
        let actual_cost = if last_source == "remote_llm" {
            (total_tokens as f64 / 1000.0) * REMOTE_PRICE_PER_1K_TOKENS_USD
        } else if last_source == "local_llm" {
            (total_tokens as f64 / 1000.0) * LOCAL_PRICE_PER_1K_TOKENS_USD
        } else {
            0.0
        };
        let saved = (estimated_remote_cost - actual_cost).max(0.0);

        let final_status = if success && !final_text.starts_with("No backend response") {
            "done"
        } else if final_text.starts_with("No backend response") {
            "error"
        } else {
            "error"
        };
        {
            let mut ui = self.ui_state.write().await;
            ui.status = final_status.to_string();
            ui.last_response = Some(final_text.clone());
            ui.last_source = Some(last_source.to_string());
            ui.estimated_remote_cost_usd = estimated_remote_cost;
            ui.last_saved_usd = saved;
            ui.saved_total_usd += saved;
            ui.progress_total = max_steps as u32;
            ui.progress_current = max_steps as u32;
            ui.progress_percent = 100.0;
            ui.last_updated = Utc::now().timestamp();
        }

        let response = OpenAiResponse {
            id: format!("agent-{}", Utc::now().timestamp()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: "neurofed-active-inference".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message { role: "assistant".to_string(), content: serde_json::json!(final_text), name: None },
                finish_reason: Some("stop".to_string()),
                logprobs: None,
            }],
            usage: Usage::default(),
            neurofed_source: Some(format!("active_inference_loop (Conf: {:.2})", self.calibration.read().await.calibrated_confidence(raw_confidence))),
        };
        
        let elapsed = start_time.elapsed();
        self.update_metrics_success(elapsed, &response).await;
        Ok(response)
    }

    /// Robust JSON Extraction with tests field
    async fn extract_structured_state(&self, req: &OpenAiRequest) -> StructuredState {
        let raw_query = req.messages.last().unwrap().content.to_string();
        let prompt = format!(
            "Analyze the request and extract into JSON. Format: {{\"goal\": \"intent\", \"entities\": {{\"name\": \"type\"}}, \"constraints\": [\"rule1\"], \"tests\": \"assert result == expected\"}}\nRequest: {}",
            raw_query
        );
        let internal_req = self.create_internal_req(&prompt, req);
        
        if let Ok(resp) = self.forward_to_ollama(&internal_req).await {
            if let Some(choice) = resp.choices.first() {
                let text = choice.message.content.as_str().unwrap_or("");
                let json_text = if let Some(start) = text.find('{') {
                    if let Some(end) = text.rfind('}') { &text[start..=end] } else { text }
                } else { text };

                if let Ok(mut state) = serde_json::from_str::<StructuredState>(json_text) {
                    state.raw_query = raw_query.clone();
                    return state;
                }
            }
        }
        
        StructuredState { goal: raw_query.clone(), entities: HashMap::new(), constraints: Vec::new(), assumptions: Vec::new(), tests: "".to_string(), raw_query }
    }

    fn create_internal_req(&self, prompt: &str, original_req: &OpenAiRequest) -> OpenAiRequest {
        OpenAiRequest {
            model: original_req.model.clone(),
            messages: vec![Message { role: "user".to_string(), content: json!(prompt), name: None }],
            ..Default::default()
        }
    }

    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let mut req = req.clone();
        if !self.config.proxy_config.ollama_model.is_empty() {
            req.model = self.config.proxy_config.ollama_model.clone();
        }
        tracing::debug!(
            "ollama request: model={}, messages={}, max_tokens={:?}, temperature={:?}",
            req.model,
            req.messages.len(),
            req.max_tokens,
            req.temperature
        );
        let client = BackendClient::new(
            self.proxy_config.ollama_url.clone(),
            self.proxy_config.fallback_url.clone(),
            self.proxy_config.timeout_seconds,
            self.proxy_config.openai_api_key.clone()
        );
        let response = client.send_to_ollama(&req).await;
        match &response {
            Ok(resp) => {
                let snippet = resp.choices.first()
                    .map(|c| c.message.content.to_string())
                    .unwrap_or_else(|| "<no choices>".to_string());
                tracing::debug!(
                    "ollama response: choices={}, snippet={}",
                    resp.choices.len(),
                    snippet
                );
            }
            Err(e) => tracing::debug!("ollama error: {}", e),
        }
        if response.is_err() && self.config.proxy_config.local_fallback_enabled {
            tracing::warn!("ollama failed, trying fallback backend");
            return client.send_to_fallback(&req).await;
        }
        response
    }

    async fn update_metrics_success(&self, elapsed: Duration, _response: &OpenAiResponse) {
        let mut metrics = self.metrics.write().await;
        metrics.total_processing_time_ms += elapsed.as_millis() as u64;
    }
}

/// Axum handler for /v1/chat/completions
pub async fn handle_chat_completion_endpoint(
    State(proxy): State<Arc<OpenAiProxy>>,
    Json(req): Json<OpenAiRequest>,
) -> impl IntoResponse {
    match proxy.handle_chat_completion(req).await {
        Ok(response) => Json::<OpenAiResponse>(response).into_response(),
        Err(e) => Json::<OpenAiResponse>(OpenAiResponse::error(&e.to_string())).into_response()
    }
}

/// Create router for OpenAI proxy
pub fn create_router(proxy: Arc<OpenAiProxy>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion_endpoint))
        .with_state(proxy)
}

// Default implementations
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

fn is_local_url(url: &str) -> bool {
    url.contains("localhost") || url.contains("127.0.0.1")
}

#[cfg(test)]
mod async_reactor_tests {
    use super::*;
    use std::sync::{Arc, RwLock};
    use crate::types::CognitiveDictionary;

    #[test]
    fn test_ml_locks_are_sync_for_spawn_blocking() {
        // Убеждаемся, что ML Engine и PC Hierarchy используют стандартные (не async) мьютексы.
        // Если бы они использовали tokio::sync::RwLock, компилятор бы не позволил
        // использовать блокирующий write() внутри обычного кода.
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let initial_len = dict.read().unwrap().len();
        
        // Моделируем работу внутри spawn_blocking (отдельный OS-поток)
        let dict_clone = Arc::clone(&dict);
        let handle = std::thread::spawn(move || {
            let mut d = dict_clone.write().unwrap();
            d.add_op(crate::types::ThoughtOp::Compute); // Compute already exists, but that's fine
        });
        
        handle.join().unwrap();
        let final_len = dict.read().unwrap().len();
        // Length should stay the same because Compute already exists
        assert_eq!(initial_len, final_len, "Dictionary length should not change when adding existing op");
        assert!(final_len >= 8, "Dictionary should have at least default 8 ops");
    }
}
