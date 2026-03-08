// src/openai_proxy/mod.rs
pub mod types;
pub mod metrics;
pub mod streaming;
pub mod components;
pub mod client;

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

use crate::ml_engine::MLEngine;
use crate::pc_hierarchy::PredictiveCoding;
use crate::pc_decoder::ThoughtDecoder;
use crate::types::{CognitiveDictionary, ThoughtOp, WorkingMemory};
use crate::config::NodeConfig;

pub use types::*;
pub use metrics::ProxyMetrics;
pub use streaming::*;
pub use components::*;
pub use client::*;

/// Main OpenAI proxy struct with integrated Thought Decoder
pub struct OpenAiProxy {
    pub config: NodeConfig,
    pub proxy_config: ProxyConfig,
    pub local_engine: Arc<Mutex<MLEngine>>,
    pub pc_hierarchy: Arc<Mutex<PredictiveCoding>>,
    pub embedding_dim: usize,
    pub thought_decoder: Arc<Mutex<ThoughtDecoder>>,
    pub cognitive_dict: Arc<Mutex<CognitiveDictionary>>,
    pub metrics: Arc<Mutex<crate::openai_proxy::metrics::ProxyMetrics>>,
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
        let metrics = Arc::new(Mutex::new(crate::openai_proxy::metrics::ProxyMetrics::default()));
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

    /// Main handler for chat completions with new reasoning architecture
    pub async fn handle_chat_completion(
        &self,
        req: OpenAiRequest,
    ) -> Result<OpenAiResponse, ProxyError> {
        self.metrics.lock().await.total_requests += 1;
        let start_time = Instant::now();

        // 1. Extract Working Memory (placeholder)
        let wm = self.extract_working_memory(&req).await;

        // 2. Get Belief from PC
        let query_emb = self.local_engine.lock().await.process_text(&wm.raw_query).await
            .map_err(|e| ProxyError::EmbeddingError(e.to_string()))?;
        let mut pc = self.pc_hierarchy.lock().await;
        pc.infer(&query_emb, 15)
            .map_err(|e| ProxyError::PCError(e.to_string()))?;
        let anchor_belief = pc.levels.last().unwrap().beliefs.clone();
        drop(pc);

        // 3. Thought Decoder generates plan
        let decoder = self.thought_decoder.lock().await;
        let thought_ids = decoder.decode_sequence(&anchor_belief, 10)
            .map_err(|e| ProxyError::PCError(e.to_string()))?;
        drop(decoder);
        
        // 4. Step-by-step rendering via LLM
        let mut final_text = String::new();
        let dict = self.cognitive_dict.lock().await;
        
        for id in thought_ids {
            let op = dict.get_op(id);
            if op == ThoughtOp::EOF { break; }

            info!("🛠️ Rendering step: {:?}", op);
            
            let step_prompt = format!(
                "CONTEXT:\nLanguage: {}\nEntities: {:?}\nUser Query: {}\n\nCURRENT PROGRESS:\n```\n{}\n```\n\nNEXT LOGICAL STEP: {:?}\n\nTASK: Write ONLY the code/text for this specific logical step. Do not write the whole function. Do not add explanations.",
                wm.language, wm.entities, wm.raw_query, final_text, op
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
        
        // Build final response
        let response = OpenAiResponse {
            model: "neurofed-v2-reasoner".to_string(),
            choices: vec![Choice {
                message: Message { 
                    role: "assistant".to_string(), 
                    content: json!(final_text),
                    ..Default::default()
                },
                ..Default::default()
            }],
            neurofed_source: Some("pc_rendered_by_llm".to_string()),
            ..Default::default()
        };
        
        let elapsed = start_time.elapsed();
        self.metrics.lock().await.total_processing_time_ms += elapsed.as_millis() as u64;
        Ok(response)
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

    async fn extract_working_memory(&self, req: &OpenAiRequest) -> WorkingMemory {
        // TODO: Call LLM to extract entities
        WorkingMemory {
            language: "Python".to_string(),
            entities: HashMap::new(),
            constraints: Vec::new(),
            raw_query: req.messages.last().unwrap().content.to_string(),
        }
    }

    async fn forward_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        // Placeholder: forward to local Ollama or fallback
        Err(ProxyError::BackendError("Not implemented".to_string()))
    }
}

/// Axum handler for /v1/chat/completions
pub async fn handle_chat_completion(
    State(proxy): State<Arc<OpenAiProxy>>,
    Json(req): Json<OpenAiRequest>,
) -> impl IntoResponse {
    match proxy.handle_chat_completion(req).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => {
            error!("Proxy error: {}", e);
            Json(OpenAiResponse::error(&e.to_string())).into_response()
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