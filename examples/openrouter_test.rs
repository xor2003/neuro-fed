//! OpenAI Smart Proxy OpenRouter test
//!
//! This example tests the OpenAI proxy with OpenRouter API

use neuro_fed_node::{
    openai_proxy::{OpenAiProxy, OpenAiRequest, Message},
    config::{NodeConfig, BackendConfig},
    ml_engine::MLEngine,
    pc_hierarchy::{PredictiveCoding, PCConfig},
    types::DeviceType,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use serde_json::json;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== OpenAI Smart Proxy OpenRouter Test ===\n");

    // 1. Create configuration with OpenRouter URL
    println!("1. Creating configuration with OpenRouter URL...");
    let config = NodeConfig::default();
    
    let backend_config = BackendConfig {
        openai_api_key: Some("test-key-123".to_string()), // Test key
        openai_base_url: "https://openrouter.ai/api/v1".to_string(),
        ollama_base_url: "http://localhost:11434".to_string(),
        ollama_model: "tinyllama".to_string(),
        local_fallback_enabled: true,
        tool_bypass_enabled: true,
        semantic_cache_enabled: true,
        semantic_similarity_threshold: 0.8,
        pc_inference_enabled: true,
        pc_learning_enabled: true,
        max_cache_size: 100,
    };

    // 2. Initialize ML Engine
    println!("2. Initializing ML Engine...");
    let device_type = DeviceType {
        name: "CPU".to_string(),
        description: "CPU device".to_string(),
        supported: true,
    };
    
    // Create a mock engine
    let local_engine = Arc::new(Mutex::new(
        MLEngine::new("models/minimal-model.gguf", device_type.clone())
            .unwrap_or_else(|_| {
                println!("   Using mock ML Engine (no actual model loaded)");
                let fallback_device_type = DeviceType {
                    name: "CPU".to_string(),
                    description: "CPU device".to_string(),
                    supported: true,
                };
                MLEngine::new("models/minimal-model.gguf", fallback_device_type).unwrap()
            })
    ));

    // 3. Initialize Predictive Coding hierarchy
    println!("3. Initializing Predictive Coding hierarchy...");
    let pc_config = PCConfig::new(3, vec![512, 256, 128]);
    let pc_hierarchy = Arc::new(Mutex::new(
        PredictiveCoding::new(pc_config)
            .expect("Failed to create Predictive Coding hierarchy")
    ));

    // 4. Create OpenAI Proxy
    println!("4. Creating OpenAI Smart Proxy...");
    let proxy = OpenAiProxy::new(
        config,
        backend_config,
        local_engine,
        pc_hierarchy,
    );

    // 5. Start the proxy server
    println!("5. Starting proxy server on port 8080...");
    let proxy_clone = proxy.clone();
    let server_task = tokio::spawn(async move {
        println!("   Starting server...");
        if let Err(e) = proxy_clone.start(8080).await {
            eprintln!("Failed to start proxy server: {}", e);
        }
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    println!("   Server should be running on http://localhost:8080");

    // 6. Test with a real request using HTTP client
    println!("6. Testing proxy with OpenRouter request...");
    
    // Create HTTP client for making requests
    let client = reqwest::Client::new();
    
    let request_body = json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "api_key": "test-key-123"
    });
    
    println!("   Sending request to proxy...");
    match client.post("http://localhost:8080/v1/chat/completions")
        .header("Content-Type", "application/json")
        .json(&request_body)
        .send()
        .await {
            Ok(response) => {
                let status = response.status();
                println!("   Response status: {}", status);
                if status.is_success() {
                    match response.text().await {
                        Ok(body) => {
                            println!("   SUCCESS: Got response from proxy!");
                            println!("   Response body (first 500 chars): {}", &body[..body.len().min(500)]);
                        }
                        Err(e) => {
                            println!("   ERROR: Failed to read response body: {}", e);
                        }
                    }
                } else {
                    let body = response.text().await.unwrap_or_default();
                    println!("   ERROR: Proxy returned error status: {}", status);
                    println!("   Error body: {}", body);
                }
            }
            Err(e) => {
                println!("   ERROR: Failed to send request to proxy: {}", e);
                println!("   This might be expected with an invalid API key or connection issue");
            }
        }

    // 7. Check metrics via the metrics endpoint
    println!("7. Checking proxy metrics...");
    match client.get("http://localhost:8080/v1/metrics").send().await {
        Ok(response) => {
            if response.status().is_success() {
                match response.text().await {
                    Ok(body) => println!("   Metrics: {}", body),
                    Err(e) => println!("   Failed to read metrics: {}", e),
                }
            } else {
                println!("   Failed to get metrics: {}", response.status());
            }
        }
        Err(e) => println!("   Failed to request metrics: {}", e),
    }

    // 8. Clean up
    println!("8. Cleaning up...");
    
    // Stop the server
    server_task.abort();
    
    println!("\n=== Test completed ===");
    Ok(())
}