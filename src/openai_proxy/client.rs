// src/openai_proxy/client.rs
use reqwest::Client;
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
    fn build_chat_url(base: &str) -> String {
        let base = base.trim_end_matches('/');
        if base.ends_with("/v1") {
            format!("{}/chat/completions", base)
        } else {
            format!("{}/v1/chat/completions", base)
        }
    }

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
        let url = Self::build_chat_url(&self.ollama_url);
        tracing::info!("🦙 Sending request to Ollama: {}", url);
        
        tracing::info!("📤 Outgoing Ollama Request: {}", serde_json::to_string_pretty(&req).unwrap_or_default());
        
        let response = self.client
            .post(&url)
            .json(req)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("❌ Ollama network error: {}", e);
                ProxyError::BackendError(format!("Ollama request failed: {}", e))
            })?;
        let status = response.status();
        let response_text = response.text().await
            .map_err(|e| ProxyError::InvalidResponse(format!("Failed to read Ollama response: {}", e)))?;
        let response_text = response_text.trim_start().to_string();
        
        if !status.is_success() {
            tracing::error!("❌ Ollama API failed! Status: {}", status);
            tracing::error!("❌ Ollama Body: {}", response_text);
            let snippet = response_text.chars().take(500).collect::<String>();
            return Err(ProxyError::BackendError(format!(
                "Ollama returned status: {} body={}",
                status, snippet
            )));
        }
        let response_body = serde_json::from_str::<OpenAiResponse>(&response_text)
            .map_err(|e| {
                tracing::error!("Failed to parse Ollama JSON response. Raw body: {}", response_text.chars().take(500).collect::<String>());
                ProxyError::InvalidResponse(format!("Failed to parse Ollama response: {}", e))
            })?;
        
        tracing::info!("✅ Ollama request successful");
        Ok(response_body)
    }
    
    pub async fn send_to_fallback(&self, req: &OpenAiRequest) -> Result<OpenAiResponse, ProxyError> {
        let url = Self::build_chat_url(&self.fallback_url);
        
        tracing::info!("🌐 Sending request to remote API: {}", url);
        
        let mut builder = self.client
            .post(&url)
            .timeout(self.timeout)
            .header("HTTP-Referer", "http://neurofed.local")
            .header("X-Title", "NeuroFed-Node-v1")
            .header("X-OpenRouter-Cache", "true");

        if let Some(key) = &self.api_key {
            if !key.is_empty() {
                builder = builder.bearer_auth(key);
            } else {
                tracing::warn!("Remote API URL provided, but API key is empty!");
            }
        } else {
            tracing::warn!("No API key configured for remote fallback!");
        }

        let final_req = req.clone();
        if url.contains("localhost") || url.contains("127.0.0.1") {
            tracing::debug!("Ollama endpoint detected, caching should work automatically");
        }

        tracing::info!("📤 Outgoing Remote Request: {}", serde_json::to_string_pretty(&final_req).unwrap_or_default());

        let response = builder.json(&final_req).send().await
            .map_err(|e| {
                tracing::error!("❌ Network Error: check your internet or TLS features! {:?}", e);
                ProxyError::BackendError(format!("Network error: {}", e))
            })?;
            
        let status = response.status();
        let response_text = response.text().await
            .map_err(|e| ProxyError::InvalidResponse(format!("Failed to read fallback response: {}", e)))?;
            
        if !status.is_success() {
            tracing::error!("❌ Remote API failed! Status: {}", status);
            tracing::error!("❌ Remote API Body: {}", response_text);
            return Err(ProxyError::BackendError(format!("Status: {}", status)));
        }
        
        // 🔴 FIX 2: Print the SUCCESSFUL remote response so we can see what the LLM said!
        tracing::info!("📥 Remote API Response: {}", response_text);
        
        let response_body = serde_json::from_str::<OpenAiResponse>(&response_text)
            .map_err(|e| {
                tracing::error!("Failed to parse JSON response from remote. Raw body: {}", response_text.chars().take(500).collect::<String>());
                ProxyError::InvalidResponse(format!("JSON Parse error: {}", e))
            })?;
        
        Ok(response_body)
    }
}