use std::process::Command;
use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};

use crate::openai_proxy::OpenAiProxy;

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
        .route("/ui/stats", get(ui_stats))
        .route("/ui/introspection", get(get_brain_introspection))
        .route("/ui/brain_stats", get(get_brain_statistics))
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

async fn ui_metrics(
    State(proxy): State<Arc<OpenAiProxy>>,
) -> Json<crate::openai_proxy::metrics::ProxyMetrics> {
    // 1. Get lock-free metrics from the global METRICS store
    let _telemetry = crate::metrics::METRICS.get_snapshot();

    // 2. Get StudyState with a TRY lock to prevent UI freezing
    let study = match proxy.study_state.try_read() {
        Ok(s) => s.clone(),
        Err(_) => crate::types::StudyState::default(), // Fallback if busy
    };

    // 3. Try to get proxy metrics with a TRY lock
    let proxy_metrics = match proxy.metrics.try_read() {
        Ok(m) => m.clone(),
        Err(_) => crate::openai_proxy::metrics::ProxyMetrics::default(), // Fallback if busy
    };

    // Create the summary string for the last completed task
    let summary = if let Some(last_task) = &study.last_task {
        format!(
            "Studied {} ({} paragraphs) in {:.2}s",
            last_task.file_name, last_task.paragraphs_processed, last_task.duration_seconds
        )
    } else {
        "No documents studied yet.".to_string()
    };

    // Combine data from all sources into a single response
    let mut combined_metrics = crate::openai_proxy::metrics::ProxyMetrics {
        total_requests: proxy_metrics.total_requests,
        total_processing_time_ms: proxy_metrics.total_processing_time_ms,
        cache_hits: proxy_metrics.cache_hits,
        cache_misses: proxy_metrics.cache_misses,
        pc_inference_calls: proxy_metrics.pc_inference_calls,
        pc_learning_calls: proxy_metrics.pc_learning_calls,
        thought_decoder_calls: proxy_metrics.thought_decoder_calls,
        errors: proxy_metrics.errors,
        status_message: proxy_metrics.status_message.clone(),

        // Populate the new fields from the study state
        is_studying: study.is_studying,
        study_progress: study.progress_percent,
        current_study_file: study.current_file.clone(),
        last_study_summary: summary,
    };

    if combined_metrics.status_message.is_empty() {
        if study.is_studying {
            combined_metrics.status_message = "Studying...".to_string();
        } else {
            combined_metrics.status_message = "Idle".to_string();
        }
    }

    Json(combined_metrics)
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

async fn get_brain_introspection(State(proxy): State<Arc<OpenAiProxy>>) -> Json<serde_json::Value> {
    // Try to get read locks on both PC hierarchy and ML engine
    let pc = match proxy.pc_hierarchy.try_read() {
        Ok(pc) => pc,
        Err(_) => {
            return Json(serde_json::json!({
                "status": "error",
                "message": "PC hierarchy is busy (locked)"
            }));
        }
    };

    let engine = match proxy.local_engine.try_read() {
        Ok(engine) => engine,
        Err(_) => {
            return Json(serde_json::json!({
                "status": "error",
                "message": "ML engine is busy (locked)"
            }));
        }
    };

    // 1. Get the current belief at the top of the hierarchy
    if let Ok(top_belief) = pc.get_top_belief() {
        // 2. Decode what this math vector means in English words!
        if let Ok((word, avg_mag, max_mag)) = engine.decode_belief_with_confidence(&top_belief) {
            return Json(serde_json::json!({
                "status": "success",
                "current_dominant_concept": word,
                "activation_magnitude": max_mag,
                "average_activation": avg_mag,
                "total_levels": pc.levels.len(),
                "free_energy": pc.free_energy,
            }));
        }
    }

    Json(serde_json::json!({"status": "idle or empty"}))
}

async fn get_brain_statistics(State(proxy): State<Arc<OpenAiProxy>>) -> Json<serde_json::Value> {
    // Try to get read lock on PC hierarchy
    let pc = match proxy.pc_hierarchy.try_read() {
        Ok(pc) => pc,
        Err(_) => {
            return Json(serde_json::json!({
                "status": "error",
                "message": "PC hierarchy is busy (locked)"
            }));
        }
    };

    let mut level_stats = Vec::new();

    // Analyze each level
    for (i, level) in pc.levels.iter().enumerate() {
        // Get weights matrix
        if let Ok(weights_vec) = level.weights.to_vec2::<f32>() {
            let rows = weights_vec.len();
            let cols = if rows > 0 { weights_vec[0].len() } else { 0 };

            // Calculate statistics
            let mut dead_neurons = 0;
            let mut hyper_active_neurons = 0;
            let mut total_abs = 0.0;
            let mut max_abs = 0.0;
            let mut min_abs = f32::MAX;

            // Analyze each column (neuron)
            for col in 0..cols {
                let mut col_sum = 0.0;
                let mut col_max = 0.0;

                for row in 0..rows {
                    let val = weights_vec[row][col].abs();
                    col_sum += val;
                    if val > col_max {
                        col_max = val;
                    }
                }

                let col_avg = col_sum / rows as f32;
                total_abs += col_sum;

                if col_avg < 0.001 {
                    dead_neurons += 1;
                } else if col_max > 1.0 {
                    hyper_active_neurons += 1;
                }

                if col_max > max_abs {
                    max_abs = col_max;
                }
                if col_avg < min_abs && col_avg > 0.0 {
                    min_abs = col_avg;
                }
            }

            let avg_activation = if cols > 0 {
                total_abs / (rows * cols) as f32
            } else {
                0.0
            };

            level_stats.push(serde_json::json!({
                "level": i,
                "dimensions": format!("{}x{}", rows, cols),
                "dead_neurons": dead_neurons,
                "hyper_active_neurons": hyper_active_neurons,
                "dead_neuron_percentage": if cols > 0 { (dead_neurons as f32 / cols as f32) * 100.0 } else { 0.0 },
                "avg_activation": avg_activation,
                "max_activation": max_abs,
                "min_nonzero_activation": if min_abs == f32::MAX { 0.0 } else { min_abs },
            }));
        }
    }

    Json(serde_json::json!({
        "status": "success",
        "total_levels": pc.levels.len(),
        "free_energy": pc.free_energy,
        "levels": level_stats,
        "summary": {
            "total_dead_neurons": level_stats.iter().map(|s| s["dead_neurons"].as_u64().unwrap_or(0)).sum::<u64>(),
            "total_hyper_active_neurons": level_stats.iter().map(|s| s["hyper_active_neurons"].as_u64().unwrap_or(0)).sum::<u64>(),
        }
    }))
}

fn static_response(body: &'static str, content_type: &'static str) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, content_type.parse().unwrap());
    (headers, body).into_response()
}
