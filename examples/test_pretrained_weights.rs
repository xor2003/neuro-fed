//! Test program for verifying pre-trained weight injection into PC hierarchy
//! This test loads a GGUF model, extracts layer weights, injects them into PC hierarchy,
//! and tests if the PC brain can generate meaningful text.

use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::pc_hierarchy::{PredictiveCoding, PCConfig};
use neuro_fed_node::types::DeviceType;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("=== Testing Pre-trained Weight Injection ===");
    
    // 1. Load MLEngine with GGUF model
    let model_path = "models/tinyllama.Q2_K.gguf";
    let tokenizer_path = "models/tokenizer.json";
    
    println!("Loading MLEngine from {}...", model_path);
    let device_type = DeviceType {
        name: "cpu".to_string(),
        description: "CPU".to_string(),
        supported: true,
    };
    let ml_engine = MLEngine::new_with_tokenizer(model_path, tokenizer_path, device_type)?;
    
    println!("MLEngine loaded successfully:");
    println!("  - Embedding dimension: {}", ml_engine.embedding_dim());
    println!("  - Vocabulary size: {}", ml_engine.get_model_info().get("vocab_size").unwrap());
    
    // 2. List available layer weights in GGUF
    println!("\nListing available layer weights in GGUF...");
    match ml_engine.list_layer_weights() {
        Ok(weights) => {
            println!("Found {} layer weights:", weights.len());
            for (i, weight_name) in weights.iter().enumerate().take(10) {
                println!("  {}. {}", i + 1, weight_name);
            }
            if weights.len() > 10 {
                println!("  ... and {} more", weights.len() - 10);
            }
        }
        Err(e) => {
            println!("Failed to list layer weights: {}", e);
        }
    }
    
    // 3. Create PC hierarchy with dimensions matching the GGUF model
    // For TinyLlama, embedding_dim is 2048
    let embedding_dim = ml_engine.embedding_dim();
    let pc_config = PCConfig::new(
        3, // 3 levels
        vec![embedding_dim, embedding_dim / 2, embedding_dim / 4], // Decreasing dimensions
    )
    .with_learning_rate(0.01)
    .with_convergence_threshold(0.001);
    
    println!("\nCreating PC hierarchy with dimensions: {:?}", pc_config.dim_per_level);
    let mut pc = PredictiveCoding::new(pc_config)?;
    
    // 4. Inject pre-trained weights from GGUF
    println!("\nInjecting pre-trained weights from GGUF into PC hierarchy...");
    match pc.inject_pretrained_weights(&ml_engine) {
        Ok(_) => println!("Successfully injected pre-trained weights!"),
        Err(e) => println!("Failed to inject pre-trained weights: {}", e),
    }
    
    // 5. Test text processing and decoding
    println!("\n=== Testing Text Processing ===");
    let test_text = "Hello, world!";
    println!("Processing text: '{}'", test_text);
    
    let embedding = ml_engine.process_text(test_text).await?;
    println!("Generated embedding with shape: {:?}", embedding.shape());
    
    // 6. Run PC inference
    println!("\nRunning PC inference...");
    let stats = pc.infer(&embedding, 10)?;
    println!("Inference completed:");
    println!("  - Total surprise: {:.4}", stats.total_surprise);
    println!("  - Free energy history: {:?}", stats.free_energy_history);
    
    // 7. Get beliefs from bottom level (level 0) and decode them
    println!("\nDecoding beliefs from PC hierarchy (bottom level)...");
    let bottom_beliefs = pc.get_beliefs(0)?;
    println!("Bottom level beliefs shape: {:?}", bottom_beliefs.shape());
    
    // Decode the beliefs back to text
    match ml_engine.decode_belief(bottom_beliefs) {
        Ok(text) => println!("Decoded text: '{}'", text),
        Err(e) => println!("Failed to decode beliefs: {}", e),
    }
    
    // 8. Test dream (generative) function - need to create a seed with correct dimension
    println!("\n=== Testing Dream (Generative) Function ===");
    // Create a seed with the top level dimension (512)
    let device = candle_core::Device::Cpu;
    let seed = candle_core::Tensor::randn(0f32, 1.0, (1, 512), &device)?;
    match pc.dream(&seed) {
        Ok(generated) => {
            println!("Dream generated tensor with shape: {:?}", generated.shape());
            
            // Try to decode the generated tensor (should be bottom level dimension 2048)
            match ml_engine.decode_belief(&generated) {
                Ok(text) => println!("Dream decoded text: '{}'", text),
                Err(e) => println!("Failed to decode dream output: {}", e),
            }
        }
        Err(e) => println!("Dream failed: {}", e),
    }
    
    // 9. Test learning with pre-trained weights
    println!("\n=== Testing Learning with Pre-trained Weights ===");
    let learning_stats = pc.learn_legacy(&embedding)?;
    println!("Learning completed:");
    println!("  - Free energy drop: {:.4}", 
        learning_stats.free_energy_history.first().unwrap_or(&0.0) - 
        learning_stats.free_energy_history.last().unwrap_or(&0.0));
    
    // 10. Export memory to see weights
    println!("\n=== Exporting PC Memory ===");
    match pc.export_memory() {
        Ok(memory) => {
            if let Some(levels) = memory["levels"].as_array() {
                println!("PC hierarchy has {} levels", levels.len());
                for (i, level) in levels.iter().enumerate() {
                    if let Some(weights_shape) = level["weights_shape"].as_array() {
                        if weights_shape.len() >= 2 {
                            println!("  Level {}: weights shape {}x{}",
                                i, weights_shape[0], weights_shape[1]);
                        }
                    }
                }
            } else {
                println!("No levels found in memory export");
            }
        }
        Err(e) => println!("Failed to export memory: {}", e),
    }
    
    println!("\n=== Test Complete ===");
    Ok(())
}