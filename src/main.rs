// NeuroFed Node - Decentralized Federated AGI System
// Minimal working binary entry point

use neuro_fed_node::{
    config::NodeConfig,
    ml_engine::MLEngine,
    pc_hierarchy::{PredictiveCoding},
    types::DeviceType,
};
use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 NeuroFed Node - Starting...");

    // Load configuration (use config file or default)
    let config = NodeConfig::load_or_default();
    println!("✅ Configuration loaded (model_path: {})", config.model_path);

    // Initialize ML Engine
    let device_type = DeviceType {
        name: config.ml_config.device_type.clone(),
        description: format!("Device: {}", config.ml_config.device_type),
        supported: config.ml_config.use_gpu,
    };
    
    println!("🔄 Initializing ML Engine...");
    let ml_engine = match MLEngine::new(&config.model_path, device_type.clone()) {
        Ok(engine) => {
            println!("✅ ML Engine initialized");
            engine
        }
        Err(e) => {
            println!("⚠️  ML Engine initialization failed: {}", e);
            eprintln!("   Model path: {}", config.model_path);
            return Err(e.into());
        }
    };
    
    let ml_engine = Arc::new(RwLock::new(ml_engine));

    // Get embedding dimension
    let embedding_dim = ml_engine.read().await.embedding_dim();
    println!("📏 Embedding dimension: {}", embedding_dim);

    // Initialize PC Hierarchy with unified config
    println!("🔄 Initializing Predictive Coding hierarchy...");
    
    // Build PCConfig from config file with dynamic dimensions
    let mut pc_config = config.pc_config.clone();
    let n_levels = pc_config.n_levels;
    let factor = pc_config.hidden_dim_factor;
    pc_config.dim_per_level = (0..n_levels)
        .map(|i| (embedding_dim as f32 * factor.powi(i as i32)).max(1.0) as usize)
        .collect();
    
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
    
    let pc_hierarchy = Arc::new(RwLock::new(pc_hierarchy));

    // Test basic functionality
    println!("\n🧪 Testing basic functionality...");
    
    // Test text processing
    let test_text = "Hello, NeuroFed!";
    println!("📝 Processing text: '{}'", test_text);
    
    match ml_engine.read().await.process_text(test_text).await {
        Ok(tensor) => {
            println!("✅ Text processed successfully");
            println!("   Tensor shape: {:?}", tensor.shape());
            
            // Test PC inference
            println!("🔄 Running PC inference...");
            match pc_hierarchy.write().await.infer(&tensor, 5) {
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
    println!("   - PC Hierarchy with {} levels", pc_hierarchy.read().await.config.n_levels);
    println!("   - Device: {}", device_type.name);
    
    // Keep the node running (in a real implementation, this would start servers/loops)
    println!("\n⏳ Node running (press Ctrl+C to exit)...");
    tokio::signal::ctrl_c().await?;
    println!("👋 Shutting down...");

    Ok(())
}