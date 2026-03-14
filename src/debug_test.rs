use candle_core::{Device, Tensor};
use crate::pc_hierarchy::{PredictiveCoding, PCConfig};

#[test]
fn debug_free_energy() -> Result<(), crate::pc_hierarchy::PCError> {
    let device = Device::Cpu;
    let mut config = PCConfig::new(2, vec![4, 2]);
    config.selective_update = true;
    config.surprise_threshold = 0.1;
    
    let mut pc = PredictiveCoding::new(config)?;
    let input = Tensor::randn(0f32, 1.0, (4, 1), &device)?;
    
    println!("Running infer...");
    let stats = pc.infer(&input, pc.config.inference_steps)?;
    
    println!("Free energy history: {:?}", stats.free_energy_history);
    println!("Total surprise: {}", stats.total_surprise);
    println!("High surprise indices: {:?}", stats.high_surprise_indices);
    println!("Surprise threshold: {}", pc.config.surprise_threshold);
    
    // Check if any free energy exceeds threshold
    for (i, &fe) in stats.free_energy_history.iter().enumerate() {
        if fe > pc.config.surprise_threshold {
            println!("Step {}: free energy {} > threshold {}", i, fe, pc.config.surprise_threshold);
        }
    }
    
    println!("\nNow running learn...");
    let initial_weights_l0 = pc.levels[0].weights.to_vec2::<f32>()?;
    println!("Initial weights shape: {}x{}", initial_weights_l0.len(), initial_weights_l0[0].len());
    
    let stats2 = pc.learn(&input, None)?;
    println!("Learn stats - high surprise indices: {:?}", stats2.high_surprise_indices);
    
    let new_weights_l0 = pc.levels[0].weights.to_vec2::<f32>()?;
    
    if initial_weights_l0 == new_weights_l0 {
        println!("ERROR: Weights did NOT update!");
        // Print first few values
        println!("Initial[0][0] = {}, New[0][0] = {}", initial_weights_l0[0][0], new_weights_l0[0][0]);
    } else {
        println!("SUCCESS: Weights updated.");
    }
    
    Ok(())
}