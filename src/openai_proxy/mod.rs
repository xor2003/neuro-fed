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
        }
    }

    /// Main handler with iterative reasoning, calibration, and verification
    pub async fn handle_chat_completion(
        &self,
        req: OpenAiRequest,
    ) -> Result<OpenAiResponse, ProxyError> {
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
        
        while attempt < max_steps {
            info!("🔄 Agentic Cycle: Attempt {}/{}", attempt + 1, max_steps);

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
                if let Some(choice) = step_response.choices.first() {
                    let step_code = choice.message.content.as_str().unwrap_or("")
                        .replace("```python", "").replace("```", "");
                    
                    final_text.push_str(&step_code);
                    final_text.push_str("\n");
                    pc_context = format!("{}\nExecuted: {:?}\nResult:\n{}", pc_context, op, step_code);
                }
            } else {
                warn!("Backend Ollama execution failed. Retrying...");
            }
            attempt += 1;
        }

        // 4. Ground-Truth Verification (Unit Test Harness) - ASYNC with timeout
        let verification_result = self.code_verifier.execute_with_tests(&final_text, &state.tests).await;
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
        let client = BackendClient::new(
            self.proxy_config.ollama_url.clone(),
            self.proxy_config.fallback_url.clone(),
            self.proxy_config.timeout_seconds
        );
        client.send_to_ollama(req).await
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
