use neuro_fed_node::{PCConfig, PredictiveCoding};
use candle_core::{Device, Tensor};

#[test]
fn test_minimal_pc_energy_decreases() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut config = PCConfig::new(2, vec![4, 2]);
    config.minimal_pc_mode = true;
    config.use_amortized_init = false;
    config.learning_rate = 0.2;

    let mut pc = PredictiveCoding::new_with_device(config, device.clone())?;

    // Force deterministic behavior: zero weights -> predictions start at 0
    pc.levels[0].weights = Tensor::zeros((4, 2), candle_core::DType::F32, &device)?;

    let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4, 1), &device)?;
    let stats = pc.infer(&input, 5)?;

    assert_eq!(stats.free_energy_history.len(), 5);
    let first = stats.free_energy_history.first().copied().unwrap_or(0.0);
    let last = stats.free_energy_history.last().copied().unwrap_or(0.0);
    assert!(first.is_finite() && last.is_finite());
    assert!(
        last <= first,
        "Energy should decrease in minimal PC mode. First: {first}, Last: {last}"
    );

    Ok(())
}
