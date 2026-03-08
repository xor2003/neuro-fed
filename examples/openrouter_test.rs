use neuro_fed_node::config::{NodeConfig, BackendConfig};
use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::pc_hierarchy::{PredictiveCoding, PCConfig, ThoughtDecoder};
use neuro_fed_node::openai_proxy::OpenAiProxy;
use neuro_fed_node::types::{DeviceType, CognitiveDictionary};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize configuration
    println!("1. Initializing configuration...");
    let mut config = NodeConfig::default();
    
    // Check for OpenRouter API key in environment
    let api_key = env::var("OPENROUTER_API_KEY").ok();
    if api_key.is_none() {
        println!("   Warning: OPENROUTER_API_KEY not set. Using dummy key.");
    }
    
    let mut backend_config = BackendConfig::default();
    backend_config.openai_api_key = api_key.or(Some("sk-or-v1-dummy-key".to_string()));
    backend_config.openai_base_url = "https://openrouter.ai/api/v1".to_string();
    backend_config.ollama_model = "google/palm-2-chat-bison".to_string();
    backend_config.pc_inference_enabled = true;
    
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

    // Initialize Cognitive Components
    let cognitive_dict = Arc::new(Mutex::new(CognitiveDictionary::default()));
    let vocab_size = cognitive_dict.lock().await.len();
    let thought_decoder = Arc::new(Mutex::new(
        ThoughtDecoder::new(512, vocab_size, &candle_core::Device::Cpu)?
    ));

    // 4. Create OpenAI Proxy
    println!("4. Creating OpenAI Smart Proxy...");
    let proxy = OpenAiProxy::new(
        config,
        backend_config,
        local_engine,
        pc_hierarchy,
        512, // embedding_dim from PC config
        thought_decoder,
        cognitive_dict,
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
    // We'll just demonstrate the setup here. In a real test, you'd use a client to hit http://localhost:8080/v1/chat/completions
    
    println!("OpenRouter Test setup complete.");
    
    // Cleanup
    server_task.abort();
    
    Ok(())
}
