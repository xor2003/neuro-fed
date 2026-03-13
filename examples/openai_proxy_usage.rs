use neuro_fed_node::config::NodeConfig;
use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::pc_hierarchy::{PredictiveCoding, PCConfig};
use neuro_fed_node::pc_decoder::ThoughtDecoder;
use neuro_fed_node::openai_proxy::{OpenAiProxy, create_router};
use neuro_fed_node::openai_proxy::components::ProxyConfig;
use neuro_fed_node::types::{DeviceType, CognitiveDictionary, StudyState};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize configuration
    println!("1. Initializing configuration...");
    let config = NodeConfig::default();
    let proxy_config = ProxyConfig::default();
    
    // 2. Initialize ML Engine
    println!("2. Initializing ML Engine...");
    let device_type = DeviceType {
        name: "CPU".to_string(),
        description: "CPU device".to_string(),
        supported: true,
    };
    
    // Create a mock engine
    let local_engine = Arc::new(RwLock::new(
        MLEngine::new("models/example-model.gguf", device_type.clone())
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
    let pc_hierarchy = Arc::new(RwLock::new(
        PredictiveCoding::new(pc_config)
            .expect("Failed to create Predictive Coding hierarchy")
    ));

    // Initialize Cognitive Components
    let cognitive_dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
    let vocab_size = cognitive_dict.read().await.len();
    let thought_decoder = Arc::new(RwLock::new(
        ThoughtDecoder::new(512, vocab_size, &candle_core::Device::Cpu)?
    ));

    // 4. Create OpenAI Proxy
    println!("4. Creating OpenAI Smart Proxy...");
    let study_state = Arc::new(RwLock::new(StudyState::default()));
    let proxy = Arc::new(OpenAiProxy::new(
        config,
        proxy_config,
        local_engine,
        pc_hierarchy,
        512, // embedding_dim from PC config
        thought_decoder,
        cognitive_dict,
        study_state,
    ));

    // 5. Start the proxy server (in a separate task)
    println!("5. Starting proxy server on port 8080...");
    let proxy_clone = Arc::clone(&proxy);
    tokio::spawn(async move {
        let app = create_router(proxy_clone);
        let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // 6. Demonstrate proxy features using public API
    println!("6. Demonstrating proxy features...");
    
    // Create HTTP client for making requests
    let client = reqwest::Client::new();
    
    // Example: Chat Completion
    println!("   Testing chat completion endpoint...");
    let chat_req = serde_json::json!({
        "model": "neurofed-v2",
        "messages": [
            {"role": "user", "content": "How do I implement a binary search in Rust?"}
        ]
    });
    
    // In a real environment, this would forward to Ollama/OpenAI
    // For this example, we just show the request structure
    println!("   Request: {}", serde_json::to_string_pretty(&chat_req)?);

    // Example: Metrics
    println!("   Checking proxy metrics...");
    let metrics_url = "http://localhost:8080/v1/metrics";
    match client.get(metrics_url).send().await {
        Ok(resp) => {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                println!("   Metrics: {}", serde_json::to_string_pretty(&json)?);
            }
        }
        Err(_) => println!("   Note: Server might not have fully started or backend unavailable."),
    }

    println!("OpenAI Proxy Usage demonstration complete.");
    Ok(())
}
