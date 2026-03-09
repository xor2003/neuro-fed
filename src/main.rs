// NeuroFed Node - Decentralized Federated AGI System
// Minimal working binary entry point

use neuro_fed_node::{
    config::NodeConfig,
    ml_engine::MLEngine,
    pc_hierarchy::{PredictiveCoding, PCConfig},
    types::DeviceType,
};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 NeuroFed Node - Starting...");

    // Load configuration
    let config = NodeConfig::default();
    println!("✅ Configuration loaded");

    // Initialize ML Engine
    let device_type = DeviceType {
        name: "cpu".to_string(),
        description: "CPU".to_string(),
        supported: true,
    };
    
    println!("🔄 Initializing ML Engine...");
    let ml_engine = match MLEngine::new("models/tinyllama.Q2_K.gguf", device_type.clone()) {
        Ok(engine) => {
            println!("✅ ML Engine initialized");
            engine
        }
        Err(e) => {
            println!("⚠️  ML Engine initialization failed: {}", e);
            println!("   Using fallback minimal engine...");
            // Create a minimal fallback engine
            MLEngine::new("models/minimal.gguf", device_type.clone())?
        }
    };
    
    let ml_engine = Arc::new(Mutex::new(ml_engine));

    // Get embedding dimension
    let embedding_dim = ml_engine.lock().await.embedding_dim();
    println!("📏 Embedding dimension: {}", embedding_dim);

    // Initialize PC Hierarchy
    println!("🔄 Initializing Predictive Coding hierarchy...");
    let pc_config = PCConfig::new(
        3, // n_levels
        vec![embedding_dim, embedding_dim / 2, embedding_dim / 4], // dim_per_level
    );
    
    let pc_hierarchy = match PredictiveCoding::new(pc_config) {
        Ok(pc) => {
            println!("✅ PC Hierarchy initialized");
            pc
        }
        Err(e) => {
            println!("❌ PC Hierarchy initialization failed: {}", e);
            return Err(e.into());
        }
    };
    
    let pc_hierarchy = Arc::new(Mutex::new(pc_hierarchy));

    // Test basic functionality
    println!("\n🧪 Testing basic functionality...");
    
    // Test text processing
    let test_text = "Hello, NeuroFed!";
    println!("📝 Processing text: '{}'", test_text);
    
    match ml_engine.lock().await.process_text(test_text).await {
        Ok(tensor) => {
            println!("✅ Text processed successfully");
            println!("   Tensor shape: {:?}", tensor.shape());
            
            // Test PC inference
            println!("🔄 Running PC inference...");
            match pc_hierarchy.lock().await.infer(&tensor, 5) {
                Ok(stats) => {
                    println!("✅ Inference completed");
                    println!("   Total surprise: {:.4}", stats.total_surprise);
                    println!("   Free energy history: {:?}", stats.free_energy_history);
                }
                Err(e) => println!("❌ Inference failed: {}", e),
            }
        }
        Err(e) => println!("❌ Text processing failed: {}", e),
    }

    println!("\n🚀 NeuroFed Node is ready!");
    println!("📊 Available components:");
    println!("   - ML Engine with embedding dimension: {}", embedding_dim);
    println!("   - PC Hierarchy with {} levels", pc_hierarchy.lock().await.config.n_levels);
    println!("   - Device: {}", device_type.name);
    
    // Keep the node running (in a real implementation, this would start servers/loops)
    println!("\n⏳ Node running (press Ctrl+C to exit)...");
    tokio::signal::ctrl_c().await?;
    println!("👋 Shutting down...");

    Ok(())
}