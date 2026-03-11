// src/chat/mod.rs
pub mod handlers;

use axum::{routing, Router};
use axum::extract::State;
use axum::Json;
use std::sync::Arc;
use crate::openai_proxy::OpenAiProxy;

pub fn create_router(state: Arc<OpenAiProxy>) -> Router {
    Router::new()
        .route("/v1/models", routing::get(|State(_): State<Arc<OpenAiProxy>>| async move {
            Ok::<Json<serde_json::Value>, axum::http::StatusCode>(Json(serde_json::json!({ "data": [] })))
        }))
        .route("/v1/chat/completions", routing::post(handlers::handle_chat_completion))
        .route("/v1/completions", routing::post(handlers::handle_completion))
        .route("/v1/embeddings", routing::post(handlers::handle_embeddings))
        .route("/v1/metrics", routing::get(handlers::handle_metrics))
        .route("/v1/memory", routing::get(handlers::handle_memory))
        .route("/{*path}", routing::any(handlers::handle_generic_endpoint))
        .with_state(state)
}
