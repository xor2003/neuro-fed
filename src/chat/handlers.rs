// src/chat/handlers.rs
use axum::{extract::State, Json, http::StatusCode, response::IntoResponse};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, error, debug, warn};
use crate::openai_proxy::OpenAiProxy;
use crate::openai_proxy::types::{OpenAiRequest, OpenAiResponse, Choice, Message, Usage};
use crate::types::ThoughtOp;
use chrono::Utc;

pub async fn handle_chat_completion(
    State(proxy): State<Arc<OpenAiProxy>>,
    Json(req): Json<OpenAiRequest>,
) -> impl IntoResponse {
    proxy.metrics.lock().await.total_requests += 1;
    let start_time = Instant::now();

    // 1. Check for tool calls - Tool Bypass Mode
    if (req.tools.is_some() || req.tool_calls.is_some()) && proxy.backend_config.tool_bypass_enabled {
        proxy.metrics.lock().await.tool_bypass_requests += 1;
        debug!("Tool bypass mode activated, forwarding directly to backend");
        match proxy.forward_to_backend(&req).await {
            Ok(response) => {
                proxy.update_metrics_success(start_time.elapsed(), &response).await;
                return Ok(Json(response));
            }
            Err(e) => {
                error!("Tool bypass forwarding failed: {}", e);
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
        }
    }
    
    // 2. Semantic caching check
    if proxy.backend_config.semantic_cache_enabled {
        if let Some(cached_response) = proxy.check_semantic_cache(&req).await {
            proxy.metrics.lock().await.cache_hits += 1;
            proxy.metrics.lock().await.semantic_similarity_hits += 1;
            info!("Semantic cache hit, returning cached response");
            proxy.update_metrics_success(start_time.elapsed(), &cached_response).await;
            return Ok(Json(cached_response));
        }
    }

    // 3. System 2 Thinking (Cognitive Planning) with Belief Revision Loop
    if proxy.backend_config.pc_inference_enabled {
        proxy.metrics.lock().await.pc_inference_calls += 1;
        
        let mut state = proxy.extract_structured_state(&req).await;
        let max_revisions = 3;
        let mut final_thought_ids = Vec::new();

        for attempt in 0..max_revisions {
            tracing::info!("🔄 Reasoning Cycle: Attempt {}/{}", attempt + 1, max_revisions);

            let pc_context = state.get_pc_context();
            let query_emb = match proxy.local_engine.lock().await.process_text(&pc_context).await {
                Ok(t) => t,
                Err(_) => break,
            };
            
            let anchor_belief = {
                let mut pc = proxy.pc_hierarchy.lock().await;
                match pc.infer(&query_emb, 15) {
                    Ok(_) => pc.levels.last().unwrap().beliefs.clone(),
                    Err(_) => break,
                }
            };
            
            let thought_ids = {
                let decoder = proxy.thought_decoder.lock().await;
                match decoder.decode_sequence(&anchor_belief, 10) {
                    Ok(ids) => ids,
                    Err(_) => Vec::new(),
                }
            };

            let dict = proxy.cognitive_dict.lock().await;
            let plan_strings: Vec<String> = thought_ids.iter()
                .map(|&id| dict.get_op(id).to_string())
                .collect();
            drop(dict);

            let verification_result = proxy.verify_plan_against_constraints(&plan_strings, &state.constraints).await;

            if verification_result.is_valid {
                final_thought_ids = thought_ids;
                break;
            } else {
                state.assumptions.push(format!("Avoid this error: {}", verification_result.reason));
                if attempt == max_revisions - 1 {
                    final_thought_ids = thought_ids;
                }
            }
        }
        
        if !final_thought_ids.is_empty() {
            let mut final_text = String::new();
            let dict = proxy.cognitive_dict.lock().await;
            
            for id in final_thought_ids {
                let op = dict.get_op(id);
                if op == ThoughtOp::EOF { break; }

                let step_prompt = format!(
                    "Goal: {}\nConstraints: {:?}\nNEXT STEP: {:?}\nCURRENT: {}",
                    state.goal, state.constraints, op, final_text
                );

                let step_req = proxy.create_internal_req(&step_prompt);
                if let Ok(step_response) = proxy.forward_to_ollama(&step_req).await {
                    if let Some(choice) = step_response.choices.first() {
                        let step_code = crate::openai_proxy::streaming::extract_all_text(&choice.message.content);
                        final_text.push_str(&step_code);
                        final_text.push_str("\n");
                    }
                }
            }
            
            if !final_text.trim().is_empty() {
                let response = OpenAiResponse {
                    id: format!("pc-{}", Utc::now().timestamp()),
                    object: "chat.completion".to_string(),
                    created: Utc::now().timestamp(),
                    model: "neurofed-hybrid-reasoner".to_string(),
                    choices: vec![Choice {
                        index: 0,
                        message: Message { role: "assistant".to_string(), content: serde_json::json!(final_text), ..Default::default() },
                        finish_reason: Some("stop".to_string()),
                        ..Default::default()
                    }],
                    usage: Usage::default(),
                    neurofed_source: Some("pc_iterative_reasoning".to_string()),
                };
                
                proxy.update_metrics_success(start_time.elapsed(), &response).await;
                return Ok(Json(response));
            }
        }
    }
    
    // 4. Forward to appropriate backend
    proxy.metrics.lock().await.cache_misses += 1;
    match proxy.forward_to_backend(&req).await {
        Ok(response) => {
            if proxy.backend_config.pc_learning_enabled {
                let _ = proxy.learn_from_response(&req, &response).await;
            }
            if proxy.backend_config.semantic_cache_enabled {
                let _ = proxy.update_semantic_cache(&req, &response).await;
            }
            proxy.update_metrics_success(start_time.elapsed(), &response).await;
            Ok(Json(response))
        }
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR)
    }
}

pub async fn handle_completion(State(proxy): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>) -> impl IntoResponse {
    match proxy.forward_to_backend(&req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn handle_embeddings(State(proxy): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>) -> impl IntoResponse {
    match proxy.forward_to_backend(&req).await {
        Ok(resp) => Ok(Json(resp)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn handle_metrics(State(proxy): State<Arc<OpenAiProxy>>) -> impl IntoResponse {
    Json(proxy.metrics.lock().await.clone())
}

pub async fn handle_memory(State(proxy): State<Arc<OpenAiProxy>>) -> impl IntoResponse {
    let pc = proxy.pc_hierarchy.lock().await;
    let memory_size = pc.levels.last().map(|l| l.memory.len()).unwrap_or(0);
    Json(serde_json::json!({ "memory_size": memory_size }))
}

pub async fn handle_generic_endpoint() -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}
