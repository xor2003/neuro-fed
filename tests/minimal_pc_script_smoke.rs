use std::process::Command;

#[test]
fn test_minimal_pc_script_smoke() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("RUN_MINIMAL_PC_SCRIPT_SMOKE").ok().as_deref() != Some("1") {
        eprintln!("Skipping minimal PC script smoke (set RUN_MINIMAL_PC_SCRIPT_SMOKE=1 to run).");
        return Ok(());
    }
    let status = Command::new("bash")
        .arg("scripts/run_minimal_pc.sh")
        .status()?;
    assert!(status.success(), "run_minimal_pc.sh failed");
    Ok(())
}
