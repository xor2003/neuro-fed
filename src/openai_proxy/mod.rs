use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Instant, Duration};
use tokio::sync::Mutex;
use axum::{
    extract::State,
    response::IntoResponse,
    Json,
    Router,
    routing::post,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{info, warn, error, debug};
use candle_core::{Tensor, Device};
use chrono::Utc;

pub mod metrics;
pub mod types;
pub mod components;

use crate::ml_engine::MLEngine;
use crate::pc_hierarchy::PredictiveCoding;
use crate::pc_decoder::ThoughtDecoder;
use crate::types::{CognitiveDictionary, ThoughtOp, StructuredState};
use crate::config::NodeConfig;
use crate::openai_proxy::metrics::ProxyMetrics;
use crate::openai_proxy::types::{ProxyError, OpenAiRequest, OpenAiResponse, Message, Choice, Usage};
use crate::openai_proxy::components::ProxyConfig;
use crate::semantic_cache::SemanticCache;

/// Main OpenAI proxy struct with integrated Thought Decoder
pub struct OpenAiProxy {
    pub config: NodeConfig,
    pub proxy_config: ProxyConfig,
    pub local_engine: Arc<Mutex<MLEngine>>,
    pub pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
    pub embedding_dim: usize,
    pub thought_decoder: Arc<Mutex<ThoughtDecoder>>,
    pub cognitive_dict: Arc<Mutex<CognitiveDictionary>>,
    pub metrics: Arc<Mutex<ProxyMetrics>>,
    pub cache: Option<Arc<Mutex<SemanticCache>>>,
}

impl OpenAiProxy {
    pub fn new(
        config: NodeConfig,
        proxy_config: ProxyConfig,
        local_engine: Arc<Mutex<MLEngine>>,
        pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
        embedding_dim: usize,
        thought_decoder: Arc<Mutex<ThoughtDecoder>>,
        cognitive_dict: Arc<Mutex<CognitiveDictionary>>,
    ) -> Self {
        let metrics = Arc::new(Mutex::new(ProxyMetrics::default()));
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
        }
    }

    /// Main handler with iterative reasoning and verification
    pub async fn handle_chat_completion(
        &self,
        req: OpenAiRequest,
    ) -> Result<OpenAiResponse, ProxyError> {
        self.metrics.lock().await.total_requests += 1;
        let start_time = Instant::now();

        let mut state = self.extract_structured_state(&req).await;
        
        let max_revisions = 3;
        let mut final_thought_ids = Vec::new();
        let mut last_known_error = String::new();
        let mut plan_has_flaw = false;

        for attempt in 0..max_revisions {
            info!("🔄 Reasoning Cycle: Attempt {}/{}", attempt + 1, max_revisions);

            let pc_context = state.get_pc_context();
            let query_emb = self.local_engine.lock().await.process_text(&pc_context).await
                .map_err(|e| ProxyError::EmbeddingError(e.to_string()))?;
            
            let mut pc = self.pc_hierarchy.lock().await;
            pc.infer(&query_emb, 15)
                .map_err(|e| ProxyError::PCError(e.to_string()))?;
            let anchor_belief = pc.levels.last().unwrap().beliefs.clone();
            drop(pc);

            let decoder = self.thought_decoder.lock().await;
            let thought_ids = decoder.decode_sequence(&anchor_belief, 10, 3)
                .map_err(|e| ProxyError::PCError(e.to_string()))?;
            drop(decoder);

            let dict = self.cognitive_dict.lock().await;
            let plan_strings: Vec<String> = thought_ids.iter().map(|&id| dict.get_op(id).to_string()).collect();
            drop(dict);

            info!("📋 Сгенерирован План: {:?}", plan_strings);

            let verification_result = self.verify_plan_against_constraints(&plan_strings, &state.constraints).await;

            if verification_result.is_valid {
                info!("✅ План верифицирован!");
                final_thought_ids = thought_ids;
                plan_has_flaw = false;
                break;
            } else {
                warn!("❌ Ошибка в плане: {}", verification_result.reason);
                last_known_error = verification_result.reason.clone();
                state.assumptions.push(format!("Avoid this error: {}", last_known_error));
                
                if attempt == max_revisions - 1 {
                    warn!("⚠️ Достигнут лимит ревизий. Передаем LLM сломанный план с предупреждением.");
                    final_thought_ids = thought_ids;
                    plan_has_flaw = true;
                }
            }
        }
        
        let mut final_text = String::new();
        let dict = self.cognitive_dict.lock().await;
        
        for id in final_thought_ids {
            let op = dict.get_op(id);
            if op == ThoughtOp::EOF { break; }

            let warning_block = if plan_has_flaw {
                format!("\nWARNING: The logical plan has a detected flaw: {}. Please manually correct this while writing the code.", last_known_error)
            } else { String::new() };

            let step_prompt = format!(
                "TASK STRUCTURE:\nGoal: {}\nConstraints: {:?}\nVariables: {:?}\n\nCURRENT CODE:\n```\n{}\n```\n\nNEXT LOGICAL STEP: {:?}{}\n\nWrite ONLY the exact code/text for this specific logical step.",
                state.goal, state.constraints, state.entities, final_text, op, warning_block
            );

            let step_req = self.create_internal_req(&step_prompt, &req);
            if let Ok(step_response) = self.forward_to_ollama(&step_req).await {
                if let Some(choice) = step_response.choices.first() {
                    let step_code = choice.message.content.as_str().unwrap_or("");
                    final_text.push_str(step_code);
                    final_text.push_str("\n");
                }
            }
        }
        
        let response = OpenAiResponse {
            id: format!("pc-{}", Utc::now().timestamp()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: "neurofed-hybrid-reasoner".to_string(),
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
            usage: Usage { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
            neurofed_source: Some("pc_iterative_reasoning".to_string()),
        };
        
        let elapsed = start_time.elapsed();
        self.update_metrics_success(elapsed, &response).await;
        Ok(response)
    }

    /// Robust JSON Extraction
    async fn extract_structured_state(&self, req: &OpenAiRequest) -> StructuredState {
        let raw_query = req.messages.last().unwrap().content.to_string();
        let prompt = format!(
            "Analyze the request and extract into JSON. Format: {{\"goal\": \"intent\", \"entities\": {{\"name\": \"type\"}}, \"constraints\": [\"rule1\"]}}\nRequest: {}",
            raw_query
        );
        let internal_req = self.create_internal_req(&prompt, req);
        
        if let Ok(resp) = self.forward_to_ollama(&internal_req).await {
            if let Some(choice) = resp.choices.first() {
                let text = choice.message.content.as_str().unwrap_or("");
                
                let json_text = if let Some(start) = text.find('{') {
                    if let Some(end) = text.rfind('}') {
                        &text[start..=end]
                    } else { text }
                } else { text };

                if let Ok(mut state) = serde_json::from_str::<StructuredState>(json_text) {
                    state.raw_query = raw_query.clone();
                    return state;
                } else {
                    warn!("Не удалось распарсить JSON из: {}", json_text);
                }
            }
        }
        
        StructuredState { 
            goal: raw_query.clone(), 
            entities: HashMap::new(), 
            constraints: Vec::new(), 
            assumptions: Vec::new(), 
            raw_query 
        }
    }

    /// Placeholder for plan verification (can be expanded)
    async fn verify_plan_against_constraints(&self, plan: &[String], constraints: &[String]) -> PlanVerificationResult {
        let plan_joined = plan.join(" ");
        for constraint in constraints {
            if constraint.contains("forbidden") && plan_joined.contains(&constraint["forbidden ".len()..]) {
                return PlanVerificationResult { is_valid: false, reason: format!("Constraint violation: {}", constraint) };
            }
        }
        PlanVerificationResult { is_valid: true, reason: String::new() }
    }

    fn create_internal_req(&self, prompt: &str, original_req: &OpenAiRequest) -> OpenAiRequest {
        OpenAiRequest {
            model: original_req.model.clone(),
            messages: vec![Message {
                role: "user".to_string(),
                content: json!(prompt),
                name: None,
            }],
            ..Default::default()
        }
    }

    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        Err(ProxyError::BackendError("Not implemented".to_string()))
    }

    async fn update_metrics_success(&self, elapsed: Duration, response: &OpenAiResponse) {
        let mut metrics = self.metrics.lock().await;
        metrics.total_processing_time_ms += elapsed.as_millis() as u64;
    }
}

/// Axum handler for /v1/chat/completions
pub async fn handle_chat_completion(
    State(proxy): State<Arc<OpenAiProxy>>,
    Json(req): Json<OpenAiRequest>,
) -> impl IntoResponse {
    match proxy.handle_chat_completion(req).await {
        Ok(response) => Json::<OpenAiResponse>(response).into_response(),
        Err(e) => {
            error!("Proxy error: {}", e);
            Json::<OpenAiResponse>(OpenAiResponse::error(&e.to_string())).into_response()
        }
    }
}

/// Create router for OpenAI proxy
pub fn create_router(proxy: Arc<OpenAiProxy>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_chat_completion))
        .with_state(proxy)
}

// Default implementations
impl Default for OpenAiProxy {
    fn default() -> Self {
        panic!("OpenAiProxy cannot be default-initialized")
    }
}

/// Helper structs for new reasoning flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanVerificationResult {
    pub is_valid: bool,
    pub reason: String,
}
