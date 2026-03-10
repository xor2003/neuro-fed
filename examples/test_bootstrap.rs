use neuro_fed_node::bootstrap::BootstrapManager;
use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::pc_decoder::ThoughtDecoder;
use neuro_fed_node::pc_hierarchy::{PredictiveCoding, PCConfig};
use neuro_fed_node::types::{DeviceType, CognitiveDictionary};
use candle_core::Device;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("Testing bootstrap system with synthetic training...");
    
    // Create ML Engine (dummy for testing)
    let device_type = DeviceType {
        name: "CPU".to_string(),
        description: "CPU device".to_string(),
        supported: true,
    };
    let ml_engine = Arc::new(RwLock::new(MLEngine::new("models/tinyllama.Q2_K.gguf", device_type)?));
    
    // Create Cognitive Dictionary
    let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
    
    // Create PC Hierarchy
    let pc_config = PCConfig::new(3, vec![512, 256, 128]);
    let pc_hierarchy = Arc::new(RwLock::new(PredictiveCoding::new(pc_config)?));

    // Create Thought Decoder
    let belief_dim = 512;
    let vocab_size = dict.read().await.len();
    let thought_decoder = Arc::new(RwLock::new(
        ThoughtDecoder::new(belief_dim, vocab_size, &Device::Cpu)?
    ));
    
    // Create BootstrapManager
    let bootstrapper = BootstrapManager::new(
        ml_engine,
        thought_decoder,
        dict,
        pc_hierarchy,
    );
    
    println!("Running bootstrap synthetic training...");
    match bootstrapper.run_synthetic_training().await {
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
