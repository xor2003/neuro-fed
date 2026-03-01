//! OpenAI Smart Proxy usage example
//!
//! This example demonstrates:
//! 1. Setting up the OpenAI Smart Proxy with ML Engine and Predictive Coding
//! 2. Starting the proxy server
//! 3. Making requests through the proxy
//! 4. Checking metrics and cache statistics

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
    println!("=== OpenAI Smart Proxy Usage Example ===\n");

    // 1. Create configuration
    println!("1. Creating configuration...");
    let config = NodeConfig::default();
    
    let backend_config = BackendConfig {
        openai_api_key: Some("your-openai-api-key-here".to_string()), // Replace with actual key
        openai_base_url: "https://api.openai.com".to_string(),
        ollama_base_url: "http://localhost:11434".to_string(),
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
    
    // Note: In a real scenario, you would load an actual model
    // For this example, we'll create a mock engine
    let local_engine = Arc::new(Mutex::new(
        MLEngine::new("models/example-model.gguf", device_type.clone())
            .unwrap_or_else(|_| {
                println!("   Using mock ML Engine (no actual model loaded)");
                // Create a minimal mock engine for demonstration
                // In production, you would load a real model
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

    // 5. Start the proxy server (in a separate task)
    println!("5. Starting proxy server on port 8080...");
    let proxy_clone = proxy.clone();
    tokio::spawn(async move {
        if let Err(e) = proxy_clone.start(8080).await {
            eprintln!("Failed to start proxy server: {}", e);
        }
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // 6. Demonstrate proxy features using public API
    println!("6. Demonstrating proxy features...");
    
    // Create HTTP client for making requests
    let client = reqwest::Client::new();
    
    // Example request payload
    let request_payload = json!({
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "max_tokens": 100,
        "temperature": 0.7
    });

    println!("   Example request payload created");
    println!("   Proxy server running on http://localhost:8080");
    println!("   You can test it with:");
    println!("   curl -X POST http://localhost:8080/v1/chat/completions \\");
    println!("     -H \"Content-Type: application/json\" \\");
    println!("     -d '{{\"model\": \"gpt-4\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'");
    
    // 7. Check metrics (public API)
    println!("7. Checking proxy metrics...");
    
    // Get cache stats
    let (cache_size, max_size) = proxy.get_cache_stats().await;
    println!("   Cache stats: {}/{} entries", cache_size, max_size);
    
    // Get metrics
    let metrics = proxy.get_metrics().await;
    println!("   Proxy metrics:");
    println!("     Total requests: {}", metrics.total_requests);
    println!("     Cache hits: {}", metrics.cache_hits);
    println!("     Cache misses: {}", metrics.cache_misses);
    println!("     Tool bypass requests: {}", metrics.tool_bypass_requests);
    println!("     PC inference calls: {}", metrics.pc_inference_calls);
    println!("     PC learning calls: {}", metrics.pc_learning_calls);
    println!("     Semantic similarity hits: {}", metrics.semantic_similarity_hits);
    println!("     Total tokens saved: {}", metrics.total_tokens_saved);
    println!("     Average response time: {:.2}ms", metrics.average_response_time_ms);
    
    // 8. Demonstrate tool bypass scenario
    println!("8. Demonstrating tool bypass scenario...");
    
    // Create a request with function call (tool)
    let mut tool_args = HashMap::new();
    tool_args.insert("city".to_string(), "Paris".to_string());
    
    let request_with_tools = OpenAiRequest {
        model: "gpt-4".to_string(),
        messages: vec![
            Message {
                role: "user".to_string(),
                content: json!("Get the weather in Paris"),
                name: None,
            },
        ],
        max_tokens: Some(100),
        temperature: Some(0.7),
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        stream: None,
        n: None,
        echo: None,
        logit_bias: None,
        function_call: Some(neuro_fed_node::types::FunctionCall {
            name: "get_weather".to_string(),
            arguments: tool_args,
        }),
        tools: None,
        tool_calls: None,
        usage: None,
    };
    
    println!("   Created request with function call (tool)");
    println!("   With tool_bypass_enabled=true, this request would bypass local processing");
    println!("   and be forwarded directly to the backend API");
    
    // 9. Cleanup
    println!("9. Cleaning up...");
    
    // Clear cache
    proxy.clear_cache().await;
    println!("   Cache cleared");
    
    // Reset metrics
    proxy.reset_metrics().await;
    println!("   Metrics reset");
    
    let final_metrics = proxy.get_metrics().await;
    println!("   Final total requests: {}", final_metrics.total_requests);
    
    println!("\n=== Example completed successfully ===");
    println!("\nSummary:");
    println!("- OpenAI Smart Proxy server started on port 8080");
    println!("- Tool bypass enabled: Requests with tools will bypass local processing");
    println!("- Semantic caching enabled: Similar requests will return cached responses");
    println!("- Predictive Coding inference enabled: Can generate responses locally");
    println!("- Predictive Coding learning enabled: Learns from API responses");
    println!("- Multiple backend support: OpenAI API and Ollama");
    println!("- Comprehensive metrics collection");
    
    println!("\nNext steps:");
    println!("1. Configure your OpenAI API key in the backend_config");
    println!("2. Run: cargo run --example openai_proxy_usage");
    println!("3. Test with: curl -X POST http://localhost:8080/v1/chat/completions \\");
    println!("   -H \"Content-Type: application/json\" \\");
    println!("   -d '{{\"model\": \"gpt-4\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'");
    println!("4. Check metrics: curl http://localhost:8080/v1/metrics");
    
    Ok(())
}