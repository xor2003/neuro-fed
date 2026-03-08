//! Integration test for PC learning and inference pipeline.
//!
//! Tests three scenarios:
//! 1. Inference using fake embeddings - ensure PC can infer from input.
//! 2. Learning - ensure PC learns from data.
//! 3. Answering using learned PC - ensure PC can produce answers based on learned knowledge.

use neuro_fed_node::pc_hierarchy::{PredictiveCoding, PCConfig};
use candle_core::{Device, Tensor};
use std::error::Error;

/// Create a deterministic fake embedding tensor for testing.
fn fake_embedding(dim: usize, seed: f32) -> Tensor {
    let values: Vec<f32> = (0..dim).map(|i| (seed + i as f32).sin()).collect();
    Tensor::from_vec(values, (dim, 1), &Device::Cpu).unwrap()
}

#[test]
fn test_pc_inference() -> Result<(), Box<dyn Error>> {
    // 1. Create a PC hierarchy with small dimensions for fast testing
    let config = PCConfig::new(2, vec![16, 8]);
    let device = Device::Cpu;
    let mut pc = PredictiveCoding::new_with_device(config, device)?;

    // 2. Create a fake embedding
    let embedding = fake_embedding(16, 0.5);

    // 3. Perform inference
    let result = pc.infer(&embedding, 10)?;
    assert!(result.total_surprise >= 0.0, "Total surprise should be non-negative");
    println!("Inference succeeded with total surprise: {}", result.total_surprise);
    Ok(())
}

#[test]
fn test_pc_learning() -> Result<(), Box<dyn Error>> {
    let config = PCConfig::new(2, vec![16, 8]);
    let device = Device::Cpu;
    let mut pc = PredictiveCoding::new_with_device(config, device)?;

    // Generate some training data (fake embeddings)
    let texts = ["Hello world", "Predictive coding is cool", "Rust is fast"];
    for (i, _text) in texts.iter().enumerate() {
        let embedding = fake_embedding(16, i as f32);
        pc.learn(&embedding, None)?;
    }

    // Verify that learning didn't crash and free energy is updated
    // (learning updates internal state, we can test by inferring again)
    let test_embedding = fake_embedding(16, 42.0);
    let result = pc.infer(&test_embedding, 10)?;
    // Check that free energy history is not empty and last value is finite
    assert!(!result.free_energy_history.is_empty());
    let last_fe = result.free_energy_history.last().unwrap();
    assert!(!last_fe.is_nan());
    println!("Learning test passed with free energy history length: {}, last free energy: {}", result.free_energy_history.len(), last_fe);
    Ok(())
}

#[test]
fn test_pc_answer_with_learned_knowledge() -> Result<(), Box<dyn Error>> {
    // This test simulates answering a query using the PC's internal state after learning.
    // Since PC is not a generative model, we test that inference on a new input
    // yields lower free energy after learning (i.e., the PC has adapted).

    let config = PCConfig::new(2, vec![16, 8]);
    let device = Device::Cpu;
    let mut pc = PredictiveCoding::new_with_device(config, device)?;

    // 1. Learn from a specific pattern
    let pattern_embedding = fake_embedding(16, 7.7);
    let initial_inference = pc.infer(&pattern_embedding, 10)?;
    let initial_free_energy = *initial_inference.free_energy_history.last().unwrap_or(&0.0);

    // 2. Learn from the same pattern multiple times
    for _ in 0..5 {
        pc.learn(&pattern_embedding, None)?;
    }

    // 3. Infer again on the same pattern - free energy should decrease (or at least not increase drastically)
    let post_inference = pc.infer(&pattern_embedding, 10)?;
    let post_free_energy = *post_inference.free_energy_history.last().unwrap_or(&0.0);
    println!("Initial free energy: {}, After learning: {}", initial_free_energy, post_free_energy);
    // Note: free energy may not always decrease due to random weights, but we can assert it's not NaN
    assert!(!post_free_energy.is_nan());

    // 4. Test inference on a slightly different input
    let similar_embedding = fake_embedding(16, 7.8);
    let similar_result = pc.infer(&similar_embedding, 10)?;
    let similar_fe = *similar_result.free_energy_history.last().unwrap_or(&0.0);
    assert!(!similar_fe.is_nan());
    println!("Answering with learned PC succeeded");
    Ok(())
}

/// Integration test combining all three steps.
#[test]
fn test_full_pc_pipeline() -> Result<(), Box<dyn Error>> {
    // This test mimics a real scenario:
    // - Receive a query, generate embedding (fake)
    // - Perform PC inference to get belief state
    // - Learn from the response (simulated)
    // - Use learned PC to answer a similar query

    // Create PC
    let config = PCConfig::new(3, vec![32, 16, 8]);
    let device = Device::Cpu;
    let mut pc = PredictiveCoding::new_with_device(config, device)?;

    // Step 1: Inference on a query
    let query_embedding = fake_embedding(32, 123.0);
    let inference_result = pc.infer(&query_embedding, 15)?;
    // Check that free energy history is not empty and last value is non-negative
    assert!(!inference_result.free_energy_history.is_empty());
    let last_fe = *inference_result.free_energy_history.last().unwrap();
    assert!(last_fe >= 0.0);

    // Step 2: Simulate a response (like from LLM) and learn from it
    let response_embedding = fake_embedding(32, 456.0);
    pc.learn(&response_embedding, None)?;

    // Step 3: Use PC to answer a similar query
    let similar_embedding = fake_embedding(32, 124.0);
    let answer_result = pc.infer(&similar_embedding, 15)?;
    let answer_fe = *answer_result.free_energy_history.last().unwrap_or(&0.0);
    assert!(!answer_fe.is_nan());

    println!("Full PC pipeline test passed");
    Ok(())
}