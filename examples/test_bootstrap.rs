use neuro_fed_node::bootstrap::BootstrapManager;
use neuro_fed_node::config::BootstrapConfig;
use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::pc_hierarchy::{PredictiveCoding, PCConfig};
use neuro_fed_node::types::DeviceType;
use candle_core::Device;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing bootstrap system with human-eval data...");
    
    // Create a simple test configuration
    let config = BootstrapConfig {
        embedding_dim: 768,
        batch_size: 1,
        max_epochs: 1,
        learning_rate: 0.01,
        document_paths: vec!["human-eval/data/example_problem.jsonl".to_string()],
    };
    
    // Create ML Engine (dummy for testing)
    let device_type = DeviceType {
        name: "CPU".to_string(),
        description: "CPU device".to_string(),
        supported: true,
    };
    let ml_engine = Arc::new(Mutex::new(MLEngine::new("models/tinyllama.Q2_K.gguf", device_type)?));
    
    // Create Predictive Coding hierarchy
    let mut pc_config = PCConfig::new(3, vec![768, 512, 256]);
    pc_config.persistence_db_path = Some("brains/test_bootstrap.db".to_string());
    pc_config.mu_pc_scaling = true;
    pc_config.convergence_threshold = 0.001;
    pc_config.learning_rate = 0.01;
    
    let device = Device::Cpu;
    let pc = Arc::new(Mutex::new(PredictiveCoding::new_with_device(pc_config, &device)?));
    
    // Create BootstrapManager
    let bootstrapper = BootstrapManager::new(
        config,
        ml_engine,
        pc,
    );
    
    println!("Running bootstrap...");
    match bootstrapper.run().await {
        Ok(_) => {
            println!("Bootstrap completed successfully!");
            Ok(())
        }
        Err(e) => {
            println!("Bootstrap failed: {}", e);
            Err(e)
        }
    }
}