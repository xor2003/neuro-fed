use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use neuro_fed_node::config::NodeConfig;
use neuro_fed_node::ml_engine::MLEngine;
use neuro_fed_node::openai_proxy::calibration::CalibrationStore;
use neuro_fed_node::openai_proxy::components::ProxyConfig;
use neuro_fed_node::openai_proxy::types::{Message, OpenAiRequest};
use neuro_fed_node::openai_proxy::OpenAiProxy;
use serde_json::Value;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use neuro_fed_node::reasoning_state::{
    execute_plan, recommended_ops, render_output, state_error, text_error,
};
use neuro_fed_node::pc_decoder::ThoughtDecoder;
use neuro_fed_node::sleep_phase::SleepManager;
use neuro_fed_node::types::{CognitiveDictionary, Episode, ReasoningTask, StudyState, ThoughtOp};
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

fn expected_sections_for_intent(intent: &str) -> &'static [&'static str] {
    match intent {
        "Investigation" => &["Goal:", "Plan:", "Findings:", "Evidence:", "Open Questions:"],
        "CodeTask" => &[
            "Goal:",
            "Plan:",
            "Deliverables:",
            "Implementation:",
            "Verification:",
            "Risks:",
        ],
        "TextTask" => &[
            "Goal:",
            "Plan:",
            "Deliverables:",
            "Rewritten Text:",
            "Quality Check:",
        ],
        _ => &[],
    }
}

fn structured_section_score(intent: &str, answer: &str) -> usize {
    let normalized = answer.replace("\r\n", "\n");
    expected_sections_for_intent(intent)
        .iter()
        .filter(|section| normalized.contains(**section))
        .count()
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
        let mut intent = None;
        let mut answer = None;
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
            if line.trim_start().starts_with("Intent:") {
                intent = line
                    .splitn(2, ':')
                    .nth(1)
                    .map(str::trim)
                    .map(str::to_string);
            }
            if line.trim_start().starts_with("Answer:") {
                answer = line
                    .splitn(2, ':')
                    .nth(1)
                    .map(str::trim)
                    .map(str::to_string);
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
            let intent_label = intent.unwrap_or_else(|| "Unknown".to_string());
            let section_score = answer
                .as_deref()
                .map(|value| structured_section_score(&intent_label, value))
                .unwrap_or(0);
            results.push(LearningRecord {
                task_id,
                intent: intent_label,
                loss,
                trajectory,
                structured_section_score: section_score,
            });
        }
    }
    Ok(results)
}

fn export_csv(records: &[LearningRecord], output: &PathBuf) -> Result<()> {
    let mut wtr = csv::Writer::from_path(output)?;
    wtr.write_record(&["task_id", "intent", "loss", "trajectory", "structured_section_score"])?;
    for record in records {
        wtr.write_record(&[
            &record.task_id,
            &record.intent,
            &record.loss,
            record.trajectory.as_deref().unwrap_or_default(),
            &record.structured_section_score.to_string(),
        ])?;
    }
    wtr.flush()?;
    Ok(())
}

fn run_reasoning_checks(output: &PathBuf) -> Result<()> {
    let mut wtr = Writer::from_path(output)?;
    wtr.write_record(&[
        "case_id",
        "check_kind",
        "success",
        "source",
        "expected_output",
        "actual_output",
        "state_error",
        "text_error",
        "fallback_used",
        "errors",
    ])?;

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
        let actual_output = render_output(&task, &outcome).unwrap_or_default();
        let expected_output = expected_output_for(&task);
        let state_error_value = outcome.errors.len().to_string();
        let text_error_value = if expected_output.is_empty() {
            "0".to_string()
        } else {
            text_error(&expected_output, &actual_output).to_string()
        };
        wtr.write_record(&[
            case_id,
            "state_engine",
            &outcome.success.to_string(),
            "reasoning_state",
            &expected_output,
            &actual_output,
            &state_error_value,
            &text_error_value,
            "false",
            &outcome.errors.join(" | "),
        ])?;
        if !outcome.success && case_id != "missing_compute" {
            failures.push(case_id.to_string());
        }
        if outcome.success && case_id == "missing_compute" {
            failures.push(case_id.to_string());
        }
    }

    let rt = tokio::runtime::Runtime::new()?;
    let proxy_failures = rt.block_on(async {
        let mut proxy_failures = Vec::new();
        let mut config = NodeConfig::load_or_default();
        config.proxy_config.pc_learning_enabled = false;
        config.proxy_config.require_thought_ops = false;
        let mut proxy_config = ProxyConfig::default();
        proxy_config.require_thought_ops = false;
        let engine = Arc::new(RwLock::new(MLEngine::mock()?));
        let embedding_dim = engine.read().await.embedding_dim();
        let dict = Arc::new(RwLock::new(CognitiveDictionary::default()));
        let pc_hierarchy = Arc::new(RwLock::new(
            PredictiveCoding::new(config.pc_config.clone())?,
        ));
        let dict_len = dict.read().await.len();
        let vocab_capacity = config.pc_config.thought_vocab_capacity.max(dict_len);
        let thought_decoder = Arc::new(RwLock::new(ThoughtDecoder::new(
            512,
            vocab_capacity,
            &Device::Cpu,
        )?));
        let study_state = Arc::new(RwLock::new(StudyState::default()));
        let episodic_memory = Arc::new(RwLock::new(VecDeque::new()));
        let calibration = Arc::new(RwLock::new(CalibrationStore::default()));
        let proxy = OpenAiProxy::new(
            config,
            proxy_config,
            engine,
            pc_hierarchy,
            embedding_dim,
            thought_decoder,
            dict,
            study_state,
            episodic_memory,
            calibration,
            None,
            None,
        );

        let proxy_cases = vec![
            ("proxy_mul_17_23", "17 * 23", "391"),
            ("proxy_reverse_abc", "reverse abc", "cba"),
            ("proxy_sum_even", "sum even 1 2 4 5", "6"),
            ("proxy_sort_list", "sort 3 1 2", "1 2 3"),
        ];

        for (case_id, query, expected) in proxy_cases {
            let response = proxy
                .handle_chat_completion(OpenAiRequest {
                    model: "neurofed-response".to_string(),
                    messages: vec![Message {
                        role: "user".to_string(),
                        content: serde_json::json!(query),
                        name: None,
                    }],
                    ..OpenAiRequest::default()
                })
                .await;

            match response {
                Ok(resp) => {
                    let actual_output = resp.choices.first()
                        .and_then(|choice| choice.message.content.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let source = resp.neurofed_source.unwrap_or_else(|| "unknown".to_string());
                    let fallback_used = (source != "reasoning_state").to_string();
                    let success = actual_output == expected && source == "reasoning_state";
                    wtr.write_record(&[
                        case_id,
                        "proxy_path",
                        &success.to_string(),
                        &source,
                        expected,
                        &actual_output,
                        "0",
                        &text_error(expected, &actual_output).to_string(),
                        &fallback_used,
                        "",
                    ])?;
                    if !success {
                        proxy_failures.push(case_id.to_string());
                    }
                }
                Err(err) => {
                    wtr.write_record(&[
                        case_id,
                        "proxy_path",
                        "false",
                        "error",
                        expected,
                        "",
                        "1",
                        "1",
                        "true",
                        &err.to_string(),
                    ])?;
                    proxy_failures.push(case_id.to_string());
                }
            }
        }

        let structured_cases = vec![
            (
                "proxy_investigation_structure",
                "investigate architecture drift in this repo",
                "Investigation",
                vec!["Goal:", "Findings:", "Evidence:"],
            ),
            (
                "proxy_code_structure",
                "implement a parser and add tests",
                "CodeTask",
                vec!["Goal:", "Implementation:", "Verification:"],
            ),
            (
                "proxy_text_structure",
                "rewrite this paragraph to be shorter",
                "TextTask",
                vec!["Goal:", "Rewritten Text:", "Quality Check:"],
            ),
        ];

        for (case_id, query, intent, sections) in structured_cases {
            let response = proxy
                .handle_chat_completion(OpenAiRequest {
                    model: "neurofed-response".to_string(),
                    messages: vec![Message {
                        role: "user".to_string(),
                        content: serde_json::json!(query),
                        name: None,
                    }],
                    ..OpenAiRequest::default()
                })
                .await;

            match response {
                Ok(resp) => {
                    let actual_output = resp.choices.first()
                        .and_then(|choice| choice.message.content.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let section_hits = sections.iter().filter(|section| actual_output.contains(**section)).count();
                    let success = section_hits == sections.len();
                    wtr.write_record(&[
                        case_id,
                        "proxy_structure",
                        &success.to_string(),
                        intent,
                        &sections.join(" | "),
                        &actual_output,
                        "0",
                        &(sections.len() - section_hits).to_string(),
                        "false",
                        if success { "" } else { "missing structured sections" },
                    ])?;
                    if !success {
                        proxy_failures.push(case_id.to_string());
                    }
                }
                Err(err) => {
                    wtr.write_record(&[
                        case_id,
                        "proxy_structure",
                        "false",
                        intent,
                        &sections.join(" | "),
                        "",
                        "1",
                        "1",
                        "true",
                        &err.to_string(),
                    ])?;
                    proxy_failures.push(case_id.to_string());
                }
            }
        }

        Ok::<Vec<String>, anyhow::Error>(proxy_failures)
    })?;
    failures.extend(proxy_failures);
    wtr.flush()?;

    if !failures.is_empty() {
        anyhow::bail!("reasoning checks failed: {:?}", failures);
    }

    Ok(())
}

fn expected_output_for(task: &ReasoningTask) -> String {
    let ops = recommended_ops(task);
    let (_, outcome) = state_error(task, &ops);
    render_output(task, &outcome).unwrap_or_default()
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
    recommended_ops(task)
}

fn parse_i64_values(value: &Value, key: &str) -> Vec<i64> {
    value
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect::<Vec<_>>())
        .unwrap_or_default()
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        other => Some(other.to_string()),
    }
}

fn default_raw_query_for_task(task: &ReasoningTask) -> String {
    match task {
        ReasoningTask::Multiply { a, b } => format!("multiply {} and {}", a, b),
        ReasoningTask::ReverseString { input } => format!("reverse {}", input),
        ReasoningTask::SumEven { values } => format!(
            "sum even {}",
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ),
        ReasoningTask::Max { values } => format!(
            "max {}",
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ),
        ReasoningTask::SortList { values } => format!(
            "sort {}",
            values
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ),
        ReasoningTask::SympyEval {
            expression,
            operation,
            ..
        } => format!("sympy {} {}", operation, expression),
        ReasoningTask::Z3Solve {
            var,
            constraints,
            ..
        } => format!("solve {} with {}", var, constraints.join(", ")),
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
                    let values = parse_i64_values(&value, "values");
                    ReasoningTask::SumEven { values }
                }
                "max" => {
                    let values = parse_i64_values(&value, "values");
                    ReasoningTask::Max { values }
                }
                "sort_list" => {
                    let values = parse_i64_values(&value, "values");
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
                .or_else(|| value.get("query"))
                .or_else(|| value.get("problem"))
                .or_else(|| value.get("instruction"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| default_raw_query_for_task(&task));

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
                .and_then(value_to_string);

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
                assistant_intent: Some(match spec.task {
                    ReasoningTask::Multiply { .. }
                    | ReasoningTask::ReverseString { .. }
                    | ReasoningTask::SumEven { .. }
                    | ReasoningTask::Max { .. }
                    | ReasoningTask::SortList { .. }
                    | ReasoningTask::SympyEval { .. }
                    | ReasoningTask::Z3Solve { .. } => neuro_fed_node::types::AssistantIntent::Reasoning,
                }),
                goal: Some(spec.raw_query.clone()),
                plan_steps: spec.ops.iter().map(ToString::to_string).collect(),
                deliverables: Vec::new(),
                verification_checks: Vec::new(),
                constraints: Vec::new(),
                assumptions: Vec::new(),
                tests: spec.expected_output.clone(),
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

#[cfg(test)]
mod section_tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_temp_jsonl(contents: &str) -> Result<PathBuf> {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let path = env::temp_dir().join(format!("reasoning-replay-{}.jsonl", unique));
        fs::write(&path, contents)?;
        Ok(path)
    }

    #[test]
    fn test_parse_reasoning_jsonl_supports_aliases_and_defaults() -> Result<()> {
        let path = write_temp_jsonl(
            r#"{"task":"multiply","a":17,"b":23,"query":"solve this product","expected_output":391}
{"task":"sort_list","values":[3,1,2]}
"#,
        )?;

        let specs = parse_reasoning_jsonl(&[path.to_string_lossy().to_string()])?;
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].raw_query, "solve this product");
        assert_eq!(specs[0].expected_output.as_deref(), Some("391"));
        assert_eq!(
            specs[0].ops,
            recommended_ops(&ReasoningTask::Multiply { a: 17, b: 23 })
        );
        assert_eq!(specs[1].raw_query, "sort 3 1 2");
        assert_eq!(
            specs[1].ops,
            recommended_ops(&ReasoningTask::SortList {
                values: vec![3, 1, 2]
            })
        );

        let _ = fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn test_parse_reasoning_jsonl_parses_tool_tasks_and_ops() -> Result<()> {
        let path = write_temp_jsonl(
            r#"{"task":"sympy_eval","expression":"2*x + 2*x","operation":"simplify","expected":"4*x","ops":["sympy_eval","eof"]}
{"task":"z3_solve","var":"x","constraints":["x > 1","x < 3"],"expected":2,"problem":"find x","ops":["z3_solve"]}
"#,
        )?;

        let specs = parse_reasoning_jsonl(&[path.to_string_lossy().to_string()])?;
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].raw_query, "sympy simplify 2*x + 2*x");
        assert_eq!(specs[0].expected_output.as_deref(), Some("4*x"));
        assert_eq!(specs[0].ops, vec![ThoughtOp::SympyEval, ThoughtOp::EOF]);
        assert_eq!(specs[1].raw_query, "find x");
        assert_eq!(specs[1].expected_output.as_deref(), Some("2"));
        assert_eq!(specs[1].ops, vec![ThoughtOp::Z3Solve]);

        let _ = fs::remove_file(path);
        Ok(())
    }
}

#[derive(Debug)]
struct LearningRecord {
    task_id: String,
    intent: String,
    loss: String,
    trajectory: Option<String>,
    structured_section_score: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_section_score_for_code_task() {
        let score = structured_section_score(
            "CodeTask",
            "Goal:\nfix parser\n\nPlan:\n- inspect\n\nDeliverables:\n- summary\n\nImplementation:\nchanged parser\n\nVerification:\nran cargo build\n\nRisks:\n- edge cases",
        );
        assert_eq!(score, 6);
    }

    #[test]
    fn test_structured_section_score_for_unknown_intent_is_zero() {
        assert_eq!(structured_section_score("Unknown", "Goal:\nwhatever"), 0);
    }
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
