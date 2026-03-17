use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use serde_json::Value;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use neuro_fed_node::reasoning_state::execute_plan;
use neuro_fed_node::pc_decoder::ThoughtDecoder;
use neuro_fed_node::sleep_phase::SleepManager;
use neuro_fed_node::types::{CognitiveDictionary, Episode, ReasoningTask, ThoughtOp};
use neuro_fed_node::{PCConfig, PredictiveCoding};
use candle_core::Device;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
use sha2::{Digest, Sha256};

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

    /// Run reasoning-state checks (ThoughtOp -> state) without full learning
    #[arg(long)]
    reasoning_check: bool,

    /// Output CSV file for reasoning-state checks
    #[arg(long, default_value = "reasoning_feedback.csv")]
    reasoning_output: PathBuf,

    /// Run a minimal reasoning replay cycle to exercise learning + generation paths
    #[arg(long)]
    reasoning_replay: bool,

    /// Optional JSONL files with reasoning episodes (ThoughtOps + expected output)
    #[arg(long, value_delimiter = ',')]
    reasoning_jsonl: Vec<String>,

    /// Enable minimal PC mode for stable, simple inference + learning
    #[arg(long)]
    minimal_pc: bool,
}

fn parse_detail_log(path: &PathBuf) -> Result<Vec<LearningRecord>> {
    let raw = fs::read_to_string(path).context("reading detail.log")?;
    let mut results = Vec::new();
    for block in raw.split("---\n") {
        let is_bootstrap = block.contains("Bootstrap learning");
        let is_sleep = block.contains("Sleep Summary") || block.contains("Input Question:");
        if !(is_bootstrap || is_sleep) {
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
            if line.trim_start().starts_with("Combined loss:") {
                loss = Some(
                    line.split("Combined loss:")
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
            if line.trim_start().starts_with("Input Question:") && task_id.is_none() {
                if let Some(raw) = line.splitn(2, ':').nth(1) {
                    let cleaned = raw.trim();
                    if !cleaned.is_empty() {
                        task_id = Some(format!("Sleep:{}", cleaned));
                    }
                }
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

fn run_reasoning_checks(output: &PathBuf) -> Result<()> {
    let mut wtr = Writer::from_path(output)?;
    wtr.write_record(&["case_id", "success", "errors"])?;

    let cases: Vec<(&str, ReasoningTask, Vec<ThoughtOp>)> = vec![
        (
            "mul_17_23",
            ReasoningTask::Multiply { a: 17, b: 23 },
            vec![
                ThoughtOp::Plan,
                ThoughtOp::Decompose,
                ThoughtOp::Initialize,
                ThoughtOp::Compute,
                ThoughtOp::Refine,
                ThoughtOp::Return,
            ],
        ),
        (
            "reverse_abc",
            ReasoningTask::ReverseString {
                input: "abc".to_string(),
            },
            vec![
                ThoughtOp::Plan,
                ThoughtOp::Initialize,
                ThoughtOp::Iterate,
                ThoughtOp::Refine,
                ThoughtOp::Return,
            ],
        ),
        (
            "sum_even",
            ReasoningTask::SumEven {
                values: vec![1, 2, 4, 5],
            },
            vec![
                ThoughtOp::Plan,
                ThoughtOp::Initialize,
                ThoughtOp::Iterate,
                ThoughtOp::Check,
                ThoughtOp::Return,
            ],
        ),
        (
            "max_list",
            ReasoningTask::Max {
                values: vec![3, 9, 2, 7],
            },
            vec![
                ThoughtOp::Plan,
                ThoughtOp::Initialize,
                ThoughtOp::Iterate,
                ThoughtOp::Return,
            ],
        ),
        (
            "sort_list",
            ReasoningTask::SortList {
                values: vec![3, 1, 2],
            },
            vec![
                ThoughtOp::Plan,
                ThoughtOp::Initialize,
                ThoughtOp::Aggregate,
                ThoughtOp::Return,
            ],
        ),
        (
            "missing_compute",
            ReasoningTask::Multiply { a: 7, b: 6 },
            vec![ThoughtOp::Plan, ThoughtOp::Initialize, ThoughtOp::Return],
        ),
    ];

    let mut failures = Vec::new();
    for (case_id, task, ops) in cases {
        let outcome = execute_plan(&task, &ops);
        wtr.write_record(&[
            case_id,
            &outcome.success.to_string(),
            &outcome.errors.join(" | "),
        ])?;
        if !outcome.success && case_id != "missing_compute" {
            failures.push(case_id.to_string());
        }
        if outcome.success && case_id == "missing_compute" {
            failures.push(case_id.to_string());
        }
    }
    wtr.flush()?;

    if !failures.is_empty() {
        anyhow::bail!("reasoning checks failed: {:?}", failures);
    }

    Ok(())
}

#[derive(Debug)]
struct ReasoningEpisodeSpec {
    raw_query: String,
    task: ReasoningTask,
    ops: Vec<ThoughtOp>,
    expected_output: Option<String>,
}

fn dummy_sequence(seed: &str) -> Vec<Vec<f32>> {
    let mut hasher = Sha256::new();
    hasher.update(seed.as_bytes());
    let hash = hasher.finalize();
    let vals = hash[0..4]
        .iter()
        .map(|b| (*b as f32) / 255.0)
        .collect::<Vec<_>>();
    vec![vals]
}

fn parse_op_token(token: &str) -> Option<ThoughtOp> {
    let norm = token
        .trim()
        .to_uppercase()
        .replace(' ', "_")
        .replace('-', "_");
    match norm.as_str() {
        "DEFINE_FUNCTION" | "DEFINE" => Some(ThoughtOp::Define),
        "INITIALIZE_VARIABLE" | "INITIALIZE" => Some(ThoughtOp::Initialize),
        "VALIDATE_INPUT" | "VALIDATE" => Some(ThoughtOp::Validate),
        "ITERATE_COLLECTION" | "ITERATE" => Some(ThoughtOp::Iterate),
        "CHECK_CONDITION" | "CHECK" => Some(ThoughtOp::Check),
        "DECIDE_BRANCH" | "DECIDE" => Some(ThoughtOp::Decide),
        "COMPUTE_MATH" | "COMPUTE" => Some(ThoughtOp::Compute),
        "AGGREGATE_RESULTS" | "AGGREGATE" => Some(ThoughtOp::Aggregate),
        "HANDLE_ERROR" | "HANDLEERROR" => Some(ThoughtOp::HandleError),
        "RETURN_VALUE" | "RETURN" => Some(ThoughtOp::Return),
        "EXPLAIN" => Some(ThoughtOp::Explain),
        "PLAN" => Some(ThoughtOp::Plan),
        "DECOMPOSE" => Some(ThoughtOp::Decompose),
        "REFINE" => Some(ThoughtOp::Refine),
        "SYMPY_EVAL" => Some(ThoughtOp::SympyEval),
        "Z3_SOLVE" => Some(ThoughtOp::Z3Solve),
        "EOF" => Some(ThoughtOp::EOF),
        _ => None,
    }
}

fn default_ops_for(task: &ReasoningTask) -> Vec<ThoughtOp> {
    match task {
        ReasoningTask::Multiply { .. } => vec![
            ThoughtOp::Plan,
            ThoughtOp::Decompose,
            ThoughtOp::Initialize,
            ThoughtOp::Compute,
            ThoughtOp::Refine,
            ThoughtOp::Return,
            ThoughtOp::EOF,
        ],
        ReasoningTask::ReverseString { .. } => vec![
            ThoughtOp::Plan,
            ThoughtOp::Initialize,
            ThoughtOp::Iterate,
            ThoughtOp::Return,
            ThoughtOp::EOF,
        ],
        ReasoningTask::SumEven { .. } => vec![
            ThoughtOp::Plan,
            ThoughtOp::Initialize,
            ThoughtOp::Iterate,
            ThoughtOp::Return,
            ThoughtOp::EOF,
        ],
        ReasoningTask::Max { .. } => vec![
            ThoughtOp::Plan,
            ThoughtOp::Initialize,
            ThoughtOp::Iterate,
            ThoughtOp::Return,
            ThoughtOp::EOF,
        ],
        ReasoningTask::SortList { .. } => vec![
            ThoughtOp::Plan,
            ThoughtOp::Initialize,
            ThoughtOp::Aggregate,
            ThoughtOp::Return,
            ThoughtOp::EOF,
        ],
        ReasoningTask::SympyEval { .. } => vec![ThoughtOp::SympyEval, ThoughtOp::EOF],
        ReasoningTask::Z3Solve { .. } => vec![ThoughtOp::Z3Solve, ThoughtOp::EOF],
    }
}

fn parse_reasoning_jsonl(paths: &[String]) -> Result<Vec<ReasoningEpisodeSpec>> {
    let mut specs = Vec::new();
    for path in paths {
        let content = fs::read_to_string(path).with_context(|| format!("reading {}", path))?;
        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line).with_context(|| {
                format!("parsing jsonl {} line {}", path, line_idx + 1)
            })?;
            let task_type = value.get("task").and_then(|v| v.as_str()).unwrap_or("");
            let task = match task_type {
                "multiply" => {
                    let a = value.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                    let b = value.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                    ReasoningTask::Multiply { a, b }
                }
                "reverse_string" => {
                    let input = value
                        .get("input")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    ReasoningTask::ReverseString { input }
                }
                "sum_even" => {
                    let values = value
                        .get("values")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_i64())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    ReasoningTask::SumEven { values }
                }
                "max" => {
                    let values = value
                        .get("values")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_i64())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    ReasoningTask::Max { values }
                }
                "sort_list" => {
                    let values = value
                        .get("values")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_i64())
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    ReasoningTask::SortList { values }
                }
                "sympy_eval" => {
                    let expression = value
                        .get("expression")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let operation = value
                        .get("operation")
                        .and_then(|v| v.as_str())
                        .unwrap_or("simplify")
                        .to_string();
                    let expected = value
                        .get("expected")
                        .or_else(|| value.get("expected_output"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());
                    ReasoningTask::SympyEval {
                        expression,
                        operation,
                        expected,
                    }
                }
                "z3_solve" => {
                    let var = value
                        .get("var")
                        .and_then(|v| v.as_str())
                        .unwrap_or("x")
                        .to_string();
                    let constraints = value
                        .get("constraints")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let expected = value
                        .get("expected")
                        .or_else(|| value.get("expected_output"))
                        .and_then(|v| v.as_i64());
                    ReasoningTask::Z3Solve {
                        var,
                        constraints,
                        expected,
                    }
                }
                other => {
                    eprintln!("Skipping unknown task '{}' in {}", other, path);
                    continue;
                }
            };

            let raw_query = value
                .get("raw_query")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{:?}", task));

            let ops = value
                .get("ops")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .filter_map(parse_op_token)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| default_ops_for(&task));

            let expected_output = value
                .get("expected_output")
                .or_else(|| value.get("expected"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            let expected_from_task = match &task {
                ReasoningTask::SympyEval { expected, .. } => expected.clone(),
                ReasoningTask::Z3Solve { expected, .. } => expected.map(|v| v.to_string()),
                _ => None,
            };

            specs.push(ReasoningEpisodeSpec {
                raw_query,
                task,
                ops,
                expected_output: expected_output.or(expected_from_task),
            });
        }
    }
    Ok(specs)
}

fn run_reasoning_replay(specs: Option<Vec<ReasoningEpisodeSpec>>) -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let pc_config = PCConfig::new(2, vec![4, 2]);
        let pc = PredictiveCoding::new(pc_config.clone())?;
        let pc_hierarchy = Arc::new(RwLock::new(pc));

        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let dict_len = dict.read().await.len();
        let vocab_capacity = pc_config.thought_vocab_capacity.max(dict_len);
        let decoder = Arc::new(RwLock::new(ThoughtDecoder::new(
            2,
            vocab_capacity,
            &Device::Cpu,
        )?));

        let mut episodes = VecDeque::new();
        let specs = specs.unwrap_or_else(|| {
            vec![
                ReasoningEpisodeSpec {
                    raw_query: "17 * 23".into(),
                    task: ReasoningTask::Multiply { a: 17, b: 23 },
                    ops: default_ops_for(&ReasoningTask::Multiply { a: 17, b: 23 }),
                    expected_output: Some("391".into()),
                },
                ReasoningEpisodeSpec {
                    raw_query: "reverse abc".into(),
                    task: ReasoningTask::ReverseString {
                        input: "abc".into(),
                    },
                    ops: default_ops_for(&ReasoningTask::ReverseString {
                        input: "abc".into(),
                    }),
                    expected_output: Some("cba".into()),
                },
                ReasoningEpisodeSpec {
                    raw_query: "sum even 1 2 4 5".into(),
                    task: ReasoningTask::SumEven {
                        values: vec![1, 2, 4, 5],
                    },
                    ops: default_ops_for(&ReasoningTask::SumEven {
                        values: vec![1, 2, 4, 5],
                    }),
                    expected_output: Some("6".into()),
                },
                ReasoningEpisodeSpec {
                    raw_query: "sort 3 1 2".into(),
                    task: ReasoningTask::SortList {
                        values: vec![3, 1, 2],
                    },
                    ops: default_ops_for(&ReasoningTask::SortList {
                        values: vec![3, 1, 2],
                    }),
                    expected_output: Some("1 2 3".into()),
                },
            ]
        });

        let dict_guard = dict.read().await;
        for mut spec in specs {
            if !matches!(spec.ops.last(), Some(ThoughtOp::EOF)) {
                spec.ops.push(ThoughtOp::EOF);
            }
            let mut op_ids = Vec::new();
            for op in spec.ops.iter() {
                if let Some(id) = dict_guard.op_to_id.get(op) {
                    op_ids.push(*id);
                }
            }
            if op_ids.is_empty() {
                continue;
            }
            episodes.push_back(Episode {
                raw_query: spec.raw_query.clone(),
                query_sequence: dummy_sequence(&spec.raw_query),
                novelty: 2.5,
                confidence: 0.9,
                generated_code: spec
                    .expected_output
                    .clone()
                    .unwrap_or_else(|| "<missing>".to_string()),
                thought_sequence: op_ids,
                success: true,
                reasoning_task: Some(spec.task.clone()),
                expected_output: spec.expected_output.clone(),
            });
        }
        drop(dict_guard);

        let episodic_memory = Arc::new(RwLock::new(episodes));
        let sleep_mgr = SleepManager::new(
            pc_hierarchy,
            decoder,
            dict,
            episodic_memory.clone(),
        );
        sleep_mgr
            .process_sleep_cycle()
            .await
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok::<(), anyhow::Error>(())
    })?;
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

    if args.reasoning_check {
        run_reasoning_checks(&args.reasoning_output)?;
        if args.skip_run && args.study_paths.is_empty() {
            println!("Reasoning checks complete; skipping learning log parsing.");
            return Ok(());
        }
    }

    if args.reasoning_replay {
        let specs = if args.reasoning_jsonl.is_empty() {
            None
        } else {
            Some(parse_reasoning_jsonl(&args.reasoning_jsonl)?)
        };
        run_reasoning_replay(specs)?;
    }
    if !args.skip_run {
        println!(
            "Running learning benchmark for paths: {:?}",
            args.study_paths
        );
        let config_path = env::temp_dir().join("learning_benchmark.toml");
        let mut config = format!(
            "[bootstrap_config]\ndocument_paths = {}\n",
            serde_json::to_string(&args.study_paths)?
        );
        if args.minimal_pc {
            config.push_str("\n[pc_config]\nminimal_pc_mode = true\n");
        }
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
