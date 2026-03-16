//! Integration test for PC persistence (save/load weights).
//!
//! Tests that weights are saved to database on learning and can be loaded back.

use candle_core::{Device, Tensor};
use neuro_fed_node::pc_hierarchy::{PCConfig, PredictiveCoding};
use neuro_fed_node::persistence::PCPersistence;
use std::error::Error;

/// Create a deterministic fake embedding tensor for testing.
fn fake_embedding(dim: usize, seed: f32) -> Tensor {
    let values: Vec<f32> = (0..dim).map(|i| (seed + i as f32).sin()).collect();
    Tensor::from_vec(values, (dim, 1), &Device::Cpu).unwrap()
}

#[tokio::test]
async fn test_persistence_save_load() -> Result<(), Box<dyn Error>> {
    // Use in-memory SQLite database for isolation
    let db_path = ":memory:".to_string();

    // Create a PC hierarchy with small dimensions for fast testing
    let config = PCConfig::new(2, vec![16, 8]);
    let device = Device::Cpu;
    let mut pc = PredictiveCoding::new_with_device(config, device)?;

    // Perform some learning to change weights from random initialization
    let embedding = fake_embedding(16, 0.5);
    pc.learn(&embedding, None)?;

    // Save weights to database using the new get_level_weights() method
    let persistence = PCPersistence::new(&db_path).await?;

    // Extract current weights from PC hierarchy using the new method
    let levels_to_save = pc.get_level_weights()?;
    assert!(
        !levels_to_save.is_empty(),
        "Should have extracted some levels"
    );

    for level in &levels_to_save {
        persistence.save_level_weights(level).await?;
    }

    // Create a new PC hierarchy with same config
    let mut pc2 = PredictiveCoding::new_with_device(PCConfig::new(2, vec![16, 8]), Device::Cpu)?;

    // Load weights from database
    let loaded_levels = persistence.load_all_levels().await?;
    assert!(!loaded_levels.is_empty(), "Should have loaded some levels");
    assert_eq!(
        loaded_levels.len(),
        levels_to_save.len(),
        "Should have same number of levels"
    );

    // Apply loaded weights to pc2 using the new load_level_weights() method
    pc2.load_level_weights(loaded_levels)?;

    // Verify that weights are approximately equal (they should be exactly equal)
    for (orig_level, loaded_level) in pc.levels.iter().zip(pc2.levels.iter()) {
        let orig_weights: Vec<f32> = orig_level.weights.flatten_all()?.to_vec1()?;
        let loaded_weights: Vec<f32> = loaded_level.weights.flatten_all()?.to_vec1()?;
        assert_eq!(orig_weights.len(), loaded_weights.len());
        for (i, (a, b)) in orig_weights.iter().zip(loaded_weights.iter()).enumerate() {
            let diff = f32::abs(*a - *b);
            if diff > 1e-6_f32 {
                panic!(
                    "Weight mismatch at index {}: original {}, loaded {}, diff={}",
                    i, a, b, diff
                );
            }
        }
    }

    println!("Persistence test passed: weights saved and loaded correctly");
    Ok(())
}

#[tokio::test]
async fn test_persistence_empty_load() -> Result<(), Box<dyn Error>> {
    // Fresh in-memory database should have no saved levels
    // Use a unique in-memory database with shared cache to allow connection pooling
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let db_path = format!("file:memdb_{}?mode=memory&cache=shared", timestamp);
    let persistence = PCPersistence::new(&db_path).await?;
    let loaded_levels = persistence.load_all_levels().await?;
    assert!(
        loaded_levels.is_empty(),
        "Fresh database should have no levels"
    );
    Ok(())
}
