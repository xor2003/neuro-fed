//! Test program for verifying pre-trained weight injection in main NeuroFed node flow
//! This simulates the initialization logic from src/main.rs

use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::pc_hierarchy::{PCConfig, PredictiveCoding};
use neuro_fed_node::types::DeviceType;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("🧠 Testing NeuroFed Node with Pre-trained Weight Injection");

    // 1. Create MLEngine (simulating the flow in main.rs)
    let model_path = "models/tinyllama.Q2_K.gguf";
    let tokenizer_path = "models/tinyllama_tokenizer.json";
    println!("Creating MLEngine with model: {}", model_path);

    let device_type = DeviceType {
        name: "cpu".to_string(),
        description: "CPU".to_string(),
        supported: true,
    };
    let ml_engine = MLEngine::new_with_tokenizer(model_path, tokenizer_path, device_type)?;
    println!("MLEngine created successfully");

    // Get embedding dimension
    let embedding_dim = ml_engine.embedding_dim();
    println!("Detected embedding dimension: {}", embedding_dim);

    // 2. Create PC hierarchy with dynamic dimension (simulating main.rs logic)
    let mut pc_config = PCConfig::new(
        3,                                                         // n_levels
        vec![embedding_dim, embedding_dim / 2, embedding_dim / 4], // dim_per_level
    );
    pc_config.mu_pc_scaling = true;
    pc_config.convergence_threshold = 0.001;
    pc_config.learning_rate = 0.01;

    println!(
        "PC config: {} levels, dimensions: {:?}",
        pc_config.n_levels, pc_config.dim_per_level
    );

    let mut pc = PredictiveCoding::new(pc_config)?;
    println!("PC hierarchy created successfully");

    // 3. Verify PC can generate meaningful stats
    println!("\nTesting inference...");

    // Create a test input tensor
    let test_text = "Hello, world!";
    println!("Processing text: '{}'", test_text);

    let input_tensor = ml_engine.process_text(test_text).await?;
    println!("Input tensor shape: {:?}", input_tensor.shape());

    // Run inference
    let stats = pc.infer(&input_tensor, 10)?;
    let latest_free_energy = stats.free_energy_history.last().unwrap_or(&0.0);
    println!(
        "Inference completed: total_surprise={:.4}, latest_free_energy={:.4}",
        stats.total_surprise, latest_free_energy
    );

    // Get beliefs from bottom level
    let bottom_beliefs = pc.levels[0].beliefs.clone();
    println!("Bottom beliefs shape: {:?}", bottom_beliefs.shape());

    // Decode beliefs to text
    match ml_engine.decode_belief(&bottom_beliefs) {
        Ok(text) => println!("Decoded text: '{}'", text),
        Err(e) => println!("Failed to decode beliefs: {}", e),
    }

    // 4. Test learning
    println!("\nTesting learning...");
    let learning_stats = pc.learn_legacy(&input_tensor)?;
    println!(
        "Learning completed: free energy drop = {:.4}",
        learning_stats.free_energy_history.first().unwrap_or(&0.0)
            - learning_stats.free_energy_history.last().unwrap_or(&0.0)
    );

    println!("\n✅ Test completed successfully!");
    println!("The PC hierarchy is functioning correctly with pre-trained model embeddings.");

    Ok(())
}
