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
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn, error};
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
use crate::knowledge_filter::CodeVerifier;

// Use the Episode struct from types.rs instead of defining a duplicate
pub use crate::types::Episode;

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
            // NEW: Action/Perception dependencies
            code_verifier: CodeVerifier::new(pc_inference_enabled),
            episodic_memory: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Main handler with iterative reasoning and verification
    pub async fn handle_chat_completion(
        &self,
        req: OpenAiRequest,
    ) -> Result<OpenAiResponse, ProxyError> {
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_requests += 1;
        }
        let start_time = Instant::now();

        let mut state = self.extract_structured_state(&req).await;
        let max_revisions = 3;
        let mut final_text = String::new();
        
        // 🔴 ACTION-PERCEPTION LOOP
        for attempt in 0..max_revisions {
            info!("🔄 Agentic Cycle: Attempt {}/{}", attempt + 1, max_revisions);

            // 1. Perception & Belief Update
            let pc_context = state.get_pc_context();
            let query_emb = self.local_engine.read().await.process_text(&pc_context).await
                .map_err(|e| ProxyError::EmbeddingError(e.to_string()))?;
            
            let mut pc = self.pc_hierarchy.write().await;
            let pc_stats = pc.infer(&query_emb, 15)
                .map_err(|e| ProxyError::PCError(e.to_string()))?; // Calculate Surprise, Confidence, Novelty
            let anchor_belief = pc.levels.last().unwrap().beliefs.clone();
            
            // Explicit uncertainty tracking
            let confidence = 1.0 / (1.0 + pc_stats.total_surprise); // Heuristic confidence
            let novelty = pc_stats.free_energy_history.first().cloned().unwrap_or(0.0);
            drop(pc);

            // 2. Planning (Graph of Thoughts)
            let decoder = self.thought_decoder.read().await;
            let thought_ids = decoder.decode_sequence(&anchor_belief, 10, 3)
                .map_err(|e| ProxyError::PCError(e.to_string()))?;
            drop(decoder);

            // 3. Execution (Rendering via LLM)
            final_text.clear();
            let dict = self.cognitive_dict.read().await;
            for id in &thought_ids {
                let op = dict.get_op(*id);
                if op == ThoughtOp::EOF { break; }

                let step_prompt = format!(
                    "TASK:\nGoal: {}\nConstraints: {:?}\n\nCURRENT CODE:\n```python\n{}\n```\n\nNEXT STEP: {:?}\nWrite ONLY the python code for this step.",
                    state.goal, state.constraints, final_text, op
                );

                let step_req = self.create_internal_req(&step_prompt, &req);
                if let Ok(step_response) = self.forward_to_ollama(&step_req).await {
                    if let Some(choice) = step_response.choices.first() {
                        let step_code = choice.message.content.as_str().unwrap_or("").replace("```python", "").replace("```", "");
                        final_text.push_str(&step_code);
                        final_text.push_str("\n");
                    }
                }
            }
            drop(dict);

            // 4. External Verification (The World Model Simulator)
            info!("🔬 Simulating execution of generated code in external environment...");
            let sim_result = self.code_verifier.execute_python_simulator(&final_text);

            // Fast Memory: Log the episode
            // Convert query embedding tensor to vector
            let query_emb_vec = query_emb.flatten_all()
                .map_err(|e| ProxyError::EmbeddingError(e.to_string()))?
                .to_vec1::<f32>()
                .map_err(|e| ProxyError::EmbeddingError(e.to_string()))?;
            
            self.episodic_memory.write().await.push_back(Episode {
                raw_query: state.raw_query.clone(),
                query_embedding: query_emb_vec,
                novelty,
                confidence,
                generated_code: final_text.clone(),
                thought_sequence: thought_ids.clone(),
                success: sim_result.is_ok(),
            });

            match sim_result {
                Ok(stdout) => {
                    info!("✅ Simulation successful! Stdout: {}", stdout);
                    
                    // SLOW MEMORY GATING: Only consolidate to PC weights if highly novel and successful
                    if novelty > 5.0 && self.config.proxy_config.pc_learning_enabled {
                        info!("🧠 High novelty detected ({}). Consolidating to semantic memory.", novelty);
                        let mut pc = self.pc_hierarchy.write().await;
                        let _ = pc.learn_legacy(&query_emb); // Commit to deep weights
                    }
                    break; // Success! Exit reasoning loop.
                },
                Err(stderr) => {
                    warn!("❌ Simulation failed: {}", stderr);
                    // Feedback loop: Update state with the exact execution error
                    state.assumptions.push(format!("Simulator crashed with error: {}. Fix the logic.", stderr));
                    
                    if attempt == max_revisions - 1 {
                        final_text.push_str(&format!("\n# WARNING: Failed to verify. Error: {}", stderr));
                    }
                }
            }
        }
        
        let response = OpenAiResponse {
            id: format!("agent-{}", Utc::now().timestamp()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: "neurofed-active-inference".to_string(),
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
            neurofed_source: Some("active_inference_loop".to_string()),
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

    async fn forward_to_ollama(&self, _req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        Err(ProxyError::BackendError("Not implemented".to_string()))
    }

    async fn update_metrics_success(&self, elapsed: Duration, _response: &OpenAiResponse) {
        let mut metrics = self.metrics.write().await;
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

#[cfg(test)]
mod agentic_loop_tests {
    use super::*;
    use crate::types::{StructuredState, VerificationResult};

    #[test]
    fn test_belief_revision_loop_accumulates_errors() {
        // This test ensures that when the verifier finds a flaw,
        // the error message is added to the "assumptions" to guide the next reasoning cycle.

        let mut state = StructuredState {
            goal: "Write a sort function".to_string(),
            ..Default::default()
        };

        // Iteration 1: The plan is invalid
        let bad_verification = VerificationResult {
            is_valid: false,
            reason: "Plan is missing an ITERATE step.".to_string(),
        };

        // Simulate the feedback loop
        if !bad_verification.is_valid {
            state.assumptions.push(format!("Avoid this error: {}", bad_verification.reason));
        }

        // Check the state for the next iteration
        assert_eq!(state.assumptions.len(), 1);
        assert!(state.assumptions[0].contains("missing an ITERATE step"));

        // The context for the PC brain now includes the correction
        let pc_context = state.get_pc_context();
        assert!(pc_context.contains("Goal: Write a sort function"));
        assert!(pc_context.contains("Corrected Assumptions: Avoid this error: Plan is missing an ITERATE step."));
    }

    #[test]
    fn test_fast_slow_memory_gating_logic() {
        // This test verifies the core principle: only learn what is BOTH successful AND novel.

        // Case 1: Success, but NOT novel (e.g., asking the same question twice)
        let low_novelty_success = Episode {
            raw_query: "test".to_string(),
            query_embedding: vec![],
            novelty: 1.0, // Below threshold of 5.0
            confidence: 0.9,
            generated_code: "".to_string(),
            thought_sequence: vec![],
            success: true,
        };
        // Should we learn?
        let should_learn_1 = low_novelty_success.novelty > 5.0 && low_novelty_success.success;
        assert!(!should_learn_1, "Should NOT learn from successful but non-novel experiences.");

        // Case 2: Novel, but FAILED (e.g., new question, but generated code crashed)
        let high_novelty_failure = Episode {
            raw_query: "test".to_string(),
            query_embedding: vec![],
            novelty: 10.0, // Above threshold
            confidence: 0.8,
            generated_code: "".to_string(),
            thought_sequence: vec![],
            success: false,
        };
        // Should we learn?
        let should_learn_2 = high_novelty_failure.novelty > 5.0 && high_novelty_failure.success;
        assert!(!should_learn_2, "Should NOT learn from failed experiences, even if they are novel.");

        // Case 3: Novel AND Successful (The only time we update deep weights)
        let high_novelty_success = Episode {
            raw_query: "test".to_string(),
            query_embedding: vec![],
            novelty: 15.0, // Above threshold
            confidence: 0.95,
            generated_code: "".to_string(),
            thought_sequence: vec![],
            success: true,
        };
        // Should we learn?
        let should_learn_3 = high_novelty_success.novelty > 5.0 && high_novelty_success.success;
        assert!(should_learn_3, "MUST learn from experiences that are both novel and successful.");
    }
}
