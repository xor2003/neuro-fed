// src/chat/handlers.rs
use axum::{extract::State, Json, http::StatusCode, response::IntoResponse};
use std::sync::Arc;
use crate::openai_proxy::OpenAiProxy;
use crate::openai_proxy::types::OpenAiRequest;

pub async fn handle_chat_completion(
    State(proxy): State<Arc<OpenAiProxy>>,
    Json(req): Json<OpenAiRequest>,
) -> impl IntoResponse {
    match proxy.handle_chat_completion(req).await {
        Ok(response) => Ok(Json(response)),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

pub async fn handle_completion(State(proxy): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>) -> impl IntoResponse {
    let _ = (proxy, req);
    StatusCode::NOT_IMPLEMENTED
}

pub async fn handle_embeddings(State(proxy): State<Arc<OpenAiProxy>>, Json(req): Json<OpenAiRequest>) -> impl IntoResponse {
    let _ = (proxy, req);
    StatusCode::NOT_IMPLEMENTED
}

pub async fn handle_metrics(State(proxy): State<Arc<OpenAiProxy>>) -> impl IntoResponse {
    Json(proxy.metrics.read().await.clone())
}

pub async fn handle_memory(State(proxy): State<Arc<OpenAiProxy>>) -> impl IntoResponse {
    let pc = proxy.pc_hierarchy.read().await;
    let levels = pc.levels.len();
    let belief_history_len = pc.belief_history.len();
    Json(serde_json::json!({ "levels": levels, "belief_history": belief_history_len }))
}

pub async fn handle_generic_endpoint() -> impl IntoResponse {
    StatusCode::NOT_IMPLEMENTED
}
