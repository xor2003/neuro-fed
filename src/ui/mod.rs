use std::process::Command;
use std::sync::Arc;

use axum::{
    extract::State,
    http::{header, HeaderMap},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};

use crate::openai_proxy::{OpenAiProxy, UiState};

const INDEX_HTML: &str = include_str!("../../ui/index.html");
const APP_JS: &str = include_str!("../../ui/app.js");
const STYLES_CSS: &str = include_str!("../../ui/styles.css");

pub fn create_router(proxy: Arc<OpenAiProxy>) -> Router {
    Router::new()
        .route("/", get(ui_index))
        .route("/ui", get(ui_index))
        .route("/ui/app.js", get(ui_app_js))
        .route("/ui/styles.css", get(ui_styles))
        .route("/ui/metrics", get(ui_metrics))
        .route("/ui/state", get(ui_state))
        .route("/ui/stats", get(ui_stats))
        .with_state(proxy)
}

async fn ui_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn ui_app_js() -> Response {
    static_response(APP_JS, "application/javascript; charset=utf-8")
}

async fn ui_styles() -> Response {
    static_response(STYLES_CSS, "text/css; charset=utf-8")
}

async fn ui_metrics(State(proxy): State<Arc<OpenAiProxy>>) -> Json<crate::openai_proxy::metrics::ProxyMetrics> {
    Json(proxy.metrics.read().await.clone())
}

async fn ui_state(State(proxy): State<Arc<OpenAiProxy>>) -> Json<UiState> {
    Json(proxy.ui_state.read().await.clone())
}

#[derive(serde::Serialize)]
struct UiStats {
    db_size_bytes: u64,
    memory_bytes: u64,
    cpu_usage: f32,
}

async fn ui_stats(State(proxy): State<Arc<OpenAiProxy>>) -> Json<UiStats> {
    let db_path = proxy
        .config
        .pc_config
        .persistence_db_path
        .clone()
        .unwrap_or_else(|| "./neurofed.db".to_string());
    let db_size_bytes = std::fs::metadata(db_path).map(|m| m.len()).unwrap_or(0);

    // Get memory usage using `free` command (simplified approach)
    let memory_bytes = match Command::new("free").arg("-b").output() {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Parse the "used" memory from the first line after "Mem:"
            stdout
                .lines()
                .find(|line| line.starts_with("Mem:"))
                .and_then(|line| {
                    line.split_whitespace()
                        .nth(2) // used memory is the 3rd field
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(0)
        }
        Err(_) => 0,
    };

    // CPU usage is complex to get without sysinfo; return 0.0 as placeholder
    let cpu_usage = 0.0;

    Json(UiStats {
        db_size_bytes,
        memory_bytes,
        cpu_usage,
    })
}

fn static_response(body: &'static str, content_type: &'static str) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    (headers, body).into_response()
}
