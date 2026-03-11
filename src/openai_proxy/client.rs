// src/openai_proxy/client.rs
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;
use crate::openai_proxy::types::{OpenAiRequest, OpenAiResponse, ProxyError};

/// HTTP client for forwarding requests to backend LLM services
pub struct BackendClient {
    client: Client,
    ollama_url: String,
    fallback_url: String,
    timeout: Duration,
    api_key: Option<String>,
}

impl BackendClient {
    pub fn new(ollama_url: String, fallback_url: String, timeout_seconds: u64, api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .unwrap();
        Self {
            client,
            ollama_url,
            fallback_url,
            timeout: Duration::from_secs(timeout_seconds),
            api_key,
        }
    }
    
    pub async fn send_to_ollama(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let url = format!("{}/v1/chat/completions", self.ollama_url);
        tracing::debug!("sending POST {}", url);
        let response = self.client
            .post(&url)
            .json(req)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| ProxyError::BackendError(format!("Ollama request failed: {}", e)))?;
        let status = response.status();
        let response_text = response.text().await
            .map_err(|e| ProxyError::InvalidResponse(format!("Failed to read Ollama response: {}", e)))?;
        tracing::debug!("ollama raw response status={} body={}", status, response_text);
        if !status.is_success() {
            return Err(ProxyError::BackendError(format!("Ollama returned status: {}", status)));
        }
        let response_body = serde_json::from_str::<OpenAiResponse>(&response_text)
            .map_err(|e| ProxyError::InvalidResponse(format!("Failed to parse Ollama response: {}", e)))?;
        
        Ok(response_body)
    }
    
    pub async fn send_to_fallback(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let url = format!("{}/v1/chat/completions", self.fallback_url);
        tracing::debug!("sending POST {}", url);
        let mut builder = self.client
            .post(&url)
            .json(req)
            .timeout(self.timeout);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        let response = builder
            .send()
            .await
            .map_err(|e| ProxyError::BackendError(format!("Fallback request failed: {}", e)))?;
        let status = response.status();
        let response_text = response.text().await
            .map_err(|e| ProxyError::InvalidResponse(format!("Failed to read fallback response: {}", e)))?;
        tracing::debug!("fallback raw response status={} body={}", status, response_text);
        if !status.is_success() {
            return Err(ProxyError::BackendError(format!("Fallback returned status: {}", status)));
        }
        let response_body = serde_json::from_str::<OpenAiResponse>(&response_text)
            .map_err(|e| ProxyError::InvalidResponse(format!("Failed to parse fallback response: {}", e)))?;
        
        Ok(response_body)
    }
}
