//! Performance benchmark for contiguous tensor memory optimizations
//! This benchmark tests the speed improvement from adding .contiguous() calls
//! to tensor operations in the PC hierarchy.

use neuro_fed_node::{
    bootstrap::BootstrapManager,
    config::{BootstrapConfig, NodeConfig},
    ml_engine::MLEngine,
    pc_hierarchy::{PredictiveCoding, PCConfig},
    persistence::PCPersistence,
    types::DeviceType,
};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Benchmark for Contiguous Tensor Optimizations ===");
    println!("Testing study session speed improvements...");
    
    // Create a minimal configuration
    let config = NodeConfig::default();
    
    // Create ML engine with dummy model path
    let ml_engine = MLEngine::new_with_tokenizer(
        "models/tinyllama.Q2_K.gguf",  // model path
        "models/tinyllama_tokenizer.json",  // tokenizer path
        DeviceType {
            name: "CPU".to_string(),
            description: "CPU device".to_string(),
            supported: true,
        },
    )?;
    
    // Create PC hierarchy with 3 levels
    let pc_config = PCConfig::default();
    let mut pc = PredictiveCoding::new(pc_config)?;
    
    // Create bootstrap manager
    let bootstrap_config = BootstrapConfig {
        embedding_dim: 512,
        batch_size: 8,
        max_epochs: 10,
        learning_rate: 0.01,
        document_paths: vec!["study/books/Using Asyncio in Python Full.pdf".to_string()],
    };
    
    let bootstrap_manager = BootstrapManager::new(
        bootstrap_config,
        ml_engine,
        pc,
    );
    
    // Create persistence
    let persistence = PCPersistence::new(":memory:").await?;
    
    // Benchmark 1: Study a single file chunk
    println!("\n--- Benchmark 1: Study single file chunk ---");
    let test_chunks = vec![
        "This is a test chunk to measure performance of the study system. ".repeat(10),
        "Another test chunk with different content to ensure caching doesn't affect results. ".repeat(10),
    ];
    
    let start = Instant::now();
    bootstrap_manager.study_file_chunks(test_chunks).await?;
    let duration = start.elapsed();
    println!("Study duration: {:?}", duration);
    
    // Benchmark 2: Process and check file (includes hash computation)
    println!("\n--- Benchmark 2: Process and check file ---");
    let test_path = std::path::Path::new("study/books/Using Asyncio in Python Full.pdf");
    
    let start = Instant::now();
    let result = bootstrap_manager.process_and_check_file(test_path, &persistence).await?;
    let duration = start.elapsed();
    println!("File processing duration: {:?}", duration);
    println!("Result: {:?}", result.is_some());
    
    // Benchmark 3: Synthetic training (PC learning)
    println!("\n--- Benchmark 3: Synthetic training (PC learning) ---");
    let start = Instant::now();
    bootstrap_manager.run_synthetic_training().await?;
    let duration = start.elapsed();
    println!("Synthetic training duration: {:?}", duration);
    
    // Summary
    println!("\n=== Performance Summary ===");
    println!("All benchmarks completed successfully.");
    println!("The contiguous tensor optimizations should provide:");
    println!("1. Faster matmul operations in PC hierarchy");
    println!("2. Better CPU cache utilization");
    println!("3. SIMD activation for tensor operations");
    println!("4. Overall study session speed improvement");
    
    Ok(())
}