use candle_core::{Device, Tensor};
use neuro_fed_node::{PCConfig, PredictiveCoding};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

fn embed_text_to_vec4(text: &str) -> [f32; 4] {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let hash = hasher.finalize();
    let mut out = [0.0f32; 4];
    for i in 0..4 {
        let idx = i * 8;
        let chunk = &hash[idx..idx + 8];
        let mut buf = [0u8; 8];
        buf.copy_from_slice(chunk);
        let val = u64::from_le_bytes(buf);
        out[i] = (val as f64 / u64::MAX as f64) as f32;
    }
    out
}

#[test]
fn test_minimal_pc_dataset_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("study/minimal_pc/data/minimal_pc_sum.jsonl");
    let raw = fs::read_to_string(path)?;
    let mut lines = 0;

    let device = Device::Cpu;
    let mut config = PCConfig::new(2, vec![4, 2]);
    config.minimal_pc_mode = true;
    config.use_amortized_init = false;
    config.learning_rate = 0.1;
    let mut pc = PredictiveCoding::new_with_device(config, device.clone())?;

    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)?;
        let text = value.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let vec4 = embed_text_to_vec4(text);
        let input = Tensor::from_vec(vec4.to_vec(), (4, 1), &device)?;
        let stats = pc.infer(&input, 5)?;
        assert!(!stats.free_energy_history.is_empty());
        let first = stats.free_energy_history.first().copied().unwrap_or(0.0);
        let last = stats.free_energy_history.last().copied().unwrap_or(0.0);
        assert!(first.is_finite() && last.is_finite());
        assert!(
            last <= first,
            "Energy should decrease or stay flat. First: {first}, Last: {last}"
        );
        lines += 1;
    }

    assert!(lines > 0, "Minimal dataset should have at least one line.");
    Ok(())
}
