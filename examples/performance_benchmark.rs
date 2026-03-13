//! Performance benchmark for contiguous tensor memory optimizations
//! This benchmark tests the speed improvement from adding .contiguous() calls
//! to tensor operations in the PC hierarchy.

use neuro_fed_node::{
    pc_hierarchy::{PredictiveCoding, PCConfig},
};
use candle_core::{Device, Tensor};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Performance Benchmark for Contiguous Tensor Optimizations ===");
    println!("Testing inference speed improvements...");
    
    // Create PC hierarchy with 2 levels (small for fast testing)
    let pc_config = PCConfig::new(2, vec![64, 32]);
    let device = Device::Cpu;
    let mut pc = PredictiveCoding::new_with_device(pc_config, device)?;
    
    // Create a random input tensor
    let input_values: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let input = Tensor::from_vec(input_values, (64, 1), &Device::Cpu)?;
    
    // Benchmark 1: Inference (multiple steps)
    println!("\n--- Benchmark 1: Inference (20 steps) ---");
    let start = Instant::now();
    for _ in 0..10 {
        pc.infer(&input, 20)?;
    }
    let duration = start.elapsed();
    println!("10 inference calls duration: {:?}", duration);
    println!("Average per inference: {:?}", duration / 10);
    
    // Benchmark 2: Learning
    println!("\n--- Benchmark 2: Learning ---");
    let start = Instant::now();
    for _ in 0..10 {
        pc.learn(&input, None)?;
    }
    let duration = start.elapsed();
    println!("10 learning calls duration: {:?}", duration);
    println!("Average per learning: {:?}", duration / 10);
    
    // Benchmark 3: Combined inference + learning
    println!("\n--- Benchmark 3: Combined inference + learning ---");
    let start = Instant::now();
    for _ in 0..5 {
        pc.infer(&input, 20)?;
        pc.learn(&input, None)?;
    }
    let duration = start.elapsed();
    println!("5 combined calls duration: {:?}", duration);
    println!("Average per combined: {:?}", duration / 5);
    
    println!("\n=== Benchmark complete ===");
    Ok(())
}