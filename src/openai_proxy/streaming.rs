// src/openai_proxy/streaming.rs
use futures::stream::Stream;
use serde_json::Value;

/// Streaming response chunk
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamChoice {
    pub index: u32,
    pub delta: Value,
    pub finish_reason: Option<String>,
}

/// Stream manager for OpenAI-compatible streaming responses
pub struct StreamManager {
    // Placeholder for future implementation
}

impl StreamManager {
    pub fn new() -> Self {
        Self {}
    }

    pub fn create_stream(&self) -> impl Stream<Item = StreamChunk> {
        // Placeholder: returns empty stream
        futures::stream::empty()
    }
}
