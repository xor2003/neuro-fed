use serde_json::Value;
use std::fs;
use std::path::Path;

#[test]
fn test_minimal_pc_dataset_format_smoke() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("study/minimal_pc/data/minimal_pc_sum.jsonl");
    let raw = fs::read_to_string(path)?;
    let mut count = 0;

    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)?;
        let text = value.get("text").and_then(|v| v.as_str()).unwrap_or("");
        assert!(
            text.contains("Input:") && text.contains("Output:"),
            "Minimal dataset line missing Input/Output markers."
        );
        count += 1;
    }

    assert!(count > 0, "Minimal dataset should not be empty.");
    Ok(())
}
