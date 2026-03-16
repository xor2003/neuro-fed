use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use serde_json::Value;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Collect learning loss/trajectory summaries for specified datasets without running the entire node.
#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    /// Selected study files (HumanEval + GSM8K) to learn on
    #[arg(long, value_delimiter = ',')]
    study_paths: Vec<String>,

    /// Output CSV file for aggregated metrics
    #[arg(short, long, default_value = "learning_feedback.csv")]
    output: PathBuf,

    /// Reuse existing detail.log without re-running learning
    #[arg(long)]
    skip_run: bool,
}

fn parse_detail_log(path: &PathBuf) -> Result<Vec<LearningRecord>> {
    let raw = fs::read_to_string(path).context("reading detail.log")?;
    let mut results = Vec::new();
    for block in raw.split("---\n") {
        if !block.contains("Bootstrap learning") {
            continue;
        }
        let mut task_id = None;
        let mut loss = None;
        let mut trajectory = None;
        for line in block.lines() {
            if line.starts_with("Question:") {
                if let Some(json) = line.splitn(2, ':').nth(1) {
                    if let Ok(value) = serde_json::from_str::<Value>(json) {
                        task_id = value
                            .get("task_id")
                            .and_then(|v| v.as_str())
                            .map(str::to_string);
                    }
                }
            }
            if line.trim_start().starts_with("Loss:") {
                loss = Some(
                    line.split("Loss:")
                        .nth(1)
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                );
            }
            if line.trim_start().starts_with("Trajectory:") {
                trajectory = Some(
                    line.split("Trajectory:")
                        .nth(1)
                        .unwrap_or_default()
                        .trim()
                        .to_string(),
                );
            }
        }
        if let (Some(task_id), Some(loss)) = (task_id, loss) {
            results.push(LearningRecord {
                task_id,
                loss,
                trajectory,
            });
        }
    }
    Ok(results)
}

fn export_csv(records: &[LearningRecord], output: &PathBuf) -> Result<()> {
    let mut wtr = csv::Writer::from_path(output)?;
    wtr.write_record(&["task_id", "loss", "trajectory"])?;
    for record in records {
        wtr.write_record(&[
            &record.task_id,
            &record.loss,
            record.trajectory.as_deref().unwrap_or_default(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

#[derive(Debug)]
struct LearningRecord {
    task_id: String,
    loss: String,
    trajectory: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.skip_run {
        println!(
            "Running learning benchmark for paths: {:?}",
            args.study_paths
        );
        let config_path = env::temp_dir().join("learning_benchmark.toml");
        let config = format!(
            "[bootstrap_config]\ndocument_paths = {}\n",
            serde_json::to_string(&args.study_paths)?
        );
        fs::write(&config_path, config)?;

        let mut cmd = Command::new("cargo");
        cmd.args([
            "run",
            "--bin",
            "neuro-fed-node",
            "--",
            "--config",
            config_path.to_str().unwrap(),
            "--study",
            "limited",
        ]);
        // For now we just run existing binary; in future replace with targeted logic.
        let status = cmd.status().context("launching neuro-fed-node")?;
        if !status.success() {
            anyhow::bail!("learning run failed")
        }
    }

    let log_path = PathBuf::from("detail.log");
    if !log_path.exists() {
        anyhow::bail!("detail.log not found; run learning first")
    }

    let records = parse_detail_log(&log_path)?;
    export_csv(&records, &args.output)?;
    println!(
        "Exported {} rows to {}",
        records.len(),
        args.output.display()
    );
    Ok(())
}
