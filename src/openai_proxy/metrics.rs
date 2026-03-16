// src/openai_proxy/metrics.rs

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ProxyMetrics {
    pub total_requests: u64,
    pub total_processing_time_ms: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub pc_inference_calls: u64,
    pub pc_learning_calls: u64,
    pub thought_decoder_calls: u64,
    pub errors: u64,

    pub status_message: String,
    // --- 🔴 ADD THESE NEW FIELDS ---
    pub is_studying: bool,
    pub study_progress: f64,
    pub current_study_file: String,
    pub last_study_summary: String,
}

impl ProxyMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn average_processing_time_ms(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.total_processing_time_ms as f64 / self.total_requests as f64
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
}
