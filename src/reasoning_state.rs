use crate::reasoning_tools::{sympy_eval, z3_solve_int};
use crate::types::{ReasoningTask, ThoughtOp};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub enum StateValue {
    Int(i64),
    Str(String),
    List(Vec<i64>),
}

#[derive(Debug, Default)]
pub struct StateEngine {
    state: HashMap<String, StateValue>,
    errors: Vec<String>,
}

#[derive(Debug)]
pub struct ReasoningOutcome {
    pub success: bool,
    pub errors: Vec<String>,
    pub state: HashMap<String, StateValue>,
}

impl StateEngine {
    pub fn new() -> Self {
        Self {
            state: HashMap::new(),
            errors: Vec::new(),
        }
    }

    pub fn apply_op(&mut self, task: &ReasoningTask, op: &ThoughtOp) {
        match task {
            ReasoningTask::Multiply { a, b } => match op {
                ThoughtOp::Initialize => {
                    self.state.insert("a".to_string(), StateValue::Int(*a));
                    self.state.insert("b".to_string(), StateValue::Int(*b));
                    self.state.insert("result".to_string(), StateValue::Int(0));
                }
                ThoughtOp::Compute => {
                    if let (Some(StateValue::Int(a)), Some(StateValue::Int(b))) =
                        (self.state.get("a"), self.state.get("b"))
                    {
                        self.state
                            .insert("result".to_string(), StateValue::Int(a * b));
                    } else {
                        self.errors
                            .push("Compute before Initialize in Multiply".to_string());
                    }
                }
                _ => {}
            },
            ReasoningTask::ReverseString { input } => match op {
                ThoughtOp::Initialize => {
                    self.state
                        .insert("input".to_string(), StateValue::Str(input.clone()));
                    self.state
                        .insert("output".to_string(), StateValue::Str(String::new()));
                }
                ThoughtOp::Iterate => {
                    if let Some(StateValue::Str(value)) = self.state.get("input") {
                        let reversed = value.chars().rev().collect::<String>();
                        self.state
                            .insert("output".to_string(), StateValue::Str(reversed));
                    } else {
                        self.errors
                            .push("Iterate before Initialize in ReverseString".to_string());
                    }
                }
                _ => {}
            },
            ReasoningTask::SumEven { values } => match op {
                ThoughtOp::Initialize => {
                    self.state
                        .insert("values".to_string(), StateValue::List(values.clone()));
                    self.state.insert("sum".to_string(), StateValue::Int(0));
                }
                ThoughtOp::Iterate => {
                    if let Some(StateValue::List(list)) = self.state.get("values") {
                        let sum = list.iter().filter(|v| *v % 2 == 0).sum::<i64>();
                        self.state.insert("sum".to_string(), StateValue::Int(sum));
                    } else {
                        self.errors
                            .push("Iterate before Initialize in SumEven".to_string());
                    }
                }
                _ => {}
            },
            ReasoningTask::Max { values } => match op {
                ThoughtOp::Initialize => {
                    self.state
                        .insert("values".to_string(), StateValue::List(values.clone()));
                }
                ThoughtOp::Iterate => {
                    if let Some(StateValue::List(list)) = self.state.get("values") {
                        if let Some(max) = list.iter().max() {
                            self.state.insert("max".to_string(), StateValue::Int(*max));
                        }
                    } else {
                        self.errors
                            .push("Iterate before Initialize in Max".to_string());
                    }
                }
                _ => {}
            },
            ReasoningTask::SortList { values } => match op {
                ThoughtOp::Initialize => {
                    self.state
                        .insert("values".to_string(), StateValue::List(values.clone()));
                }
                ThoughtOp::Aggregate => {
                    if let Some(StateValue::List(list)) = self.state.get("values") {
                        let mut sorted = list.clone();
                        sorted.sort();
                        self.state
                            .insert("sorted".to_string(), StateValue::List(sorted));
                    } else {
                        self.errors
                            .push("Aggregate before Initialize in SortList".to_string());
                    }
                }
                _ => {}
            },
            ReasoningTask::SympyEval {
                expression,
                operation,
                ..
            } => match op {
                ThoughtOp::SympyEval => match sympy_eval(operation, expression) {
                    Ok(result) => {
                        self.state
                            .insert("result".to_string(), StateValue::Str(result));
                    }
                    Err(e) => self.errors.push(format!("SympyEval error: {}", e)),
                },
                _ => {}
            },
            ReasoningTask::Z3Solve {
                var, constraints, ..
            } => match op {
                ThoughtOp::Z3Solve => {
                    let refs: Vec<&str> = constraints.iter().map(String::as_str).collect();
                    match z3_solve_int(var, &refs) {
                        Ok(value) => {
                            self.state.insert(var.to_string(), StateValue::Int(value));
                        }
                        Err(e) => self.errors.push(format!("Z3Solve error: {}", e)),
                    }
                }
                _ => {}
            },
        }
    }
}

pub fn required_ops(task: &ReasoningTask) -> Vec<ThoughtOp> {
    match task {
        ReasoningTask::Multiply { .. } => vec![ThoughtOp::Initialize, ThoughtOp::Compute],
        ReasoningTask::ReverseString { .. } => vec![ThoughtOp::Initialize, ThoughtOp::Iterate],
        ReasoningTask::SumEven { .. } => vec![ThoughtOp::Initialize, ThoughtOp::Iterate],
        ReasoningTask::Max { .. } => vec![ThoughtOp::Initialize, ThoughtOp::Iterate],
        ReasoningTask::SortList { .. } => vec![ThoughtOp::Initialize, ThoughtOp::Aggregate],
        ReasoningTask::SympyEval { .. } => vec![ThoughtOp::SympyEval],
        ReasoningTask::Z3Solve { .. } => vec![ThoughtOp::Z3Solve],
    }
}

pub fn recommended_ops(task: &ReasoningTask) -> Vec<ThoughtOp> {
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
            ThoughtOp::Check,
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

pub fn render_output(task: &ReasoningTask, outcome: &ReasoningOutcome) -> Option<String> {
    if !outcome.success {
        return None;
    }

    match task {
        ReasoningTask::Multiply { .. } => match outcome.state.get("result") {
            Some(StateValue::Int(v)) => Some(v.to_string()),
            _ => None,
        },
        ReasoningTask::ReverseString { .. } => match outcome.state.get("output") {
            Some(StateValue::Str(v)) => Some(v.clone()),
            _ => None,
        },
        ReasoningTask::SumEven { .. } => match outcome.state.get("sum") {
            Some(StateValue::Int(v)) => Some(v.to_string()),
            _ => None,
        },
        ReasoningTask::Max { .. } => match outcome.state.get("max") {
            Some(StateValue::Int(v)) => Some(v.to_string()),
            _ => None,
        },
        ReasoningTask::SortList { .. } => match outcome.state.get("sorted") {
            Some(StateValue::List(v)) => Some(
                v.iter()
                    .map(|item| item.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            _ => None,
        },
        ReasoningTask::SympyEval { .. } => match outcome.state.get("result") {
            Some(StateValue::Str(v)) => Some(v.clone()),
            _ => None,
        },
        ReasoningTask::Z3Solve { var, .. } => match outcome.state.get(var) {
            Some(StateValue::Int(v)) => Some(v.to_string()),
            _ => None,
        },
    }
}

fn expected_state(task: &ReasoningTask) -> HashMap<String, StateValue> {
    let mut state = HashMap::new();
    match task {
        ReasoningTask::Multiply { a, b } => {
            state.insert("a".to_string(), StateValue::Int(*a));
            state.insert("b".to_string(), StateValue::Int(*b));
            state.insert("result".to_string(), StateValue::Int(a * b));
        }
        ReasoningTask::ReverseString { input } => {
            state.insert("input".to_string(), StateValue::Str(input.clone()));
            state.insert(
                "output".to_string(),
                StateValue::Str(input.chars().rev().collect()),
            );
        }
        ReasoningTask::SumEven { values } => {
            state.insert("values".to_string(), StateValue::List(values.clone()));
            let sum = values.iter().filter(|v| *v % 2 == 0).sum::<i64>();
            state.insert("sum".to_string(), StateValue::Int(sum));
        }
        ReasoningTask::Max { values } => {
            state.insert("values".to_string(), StateValue::List(values.clone()));
            if let Some(max) = values.iter().max() {
                state.insert("max".to_string(), StateValue::Int(*max));
            }
        }
        ReasoningTask::SortList { values } => {
            state.insert("values".to_string(), StateValue::List(values.clone()));
            let mut sorted = values.clone();
            sorted.sort();
            state.insert("sorted".to_string(), StateValue::List(sorted));
        }
        ReasoningTask::SympyEval {
            expression,
            operation,
            expected,
        } => {
            state.insert(
                "expression".to_string(),
                StateValue::Str(expression.clone()),
            );
            state.insert("operation".to_string(), StateValue::Str(operation.clone()));
            if let Some(value) = expected {
                state.insert("result".to_string(), StateValue::Str(value.clone()));
            }
        }
        ReasoningTask::Z3Solve {
            var,
            constraints,
            expected,
        } => {
            state.insert(
                "constraints".to_string(),
                StateValue::Str(constraints.join("; ")),
            );
            if let Some(value) = expected {
                state.insert(var.clone(), StateValue::Int(*value));
            }
        }
    }
    state
}

pub fn execute_plan(task: &ReasoningTask, ops: &[ThoughtOp]) -> ReasoningOutcome {
    let mut engine = StateEngine::new();
    let required = required_ops(task);

    for op in ops {
        engine.apply_op(task, op);
    }

    for required_op in required.iter() {
        if !ops.contains(required_op) {
            engine
                .errors
                .push(format!("Missing required op: {}", required_op));
        }
    }

    let expected = expected_state(task);
    for (key, expected_value) in expected {
        match engine.state.get(&key) {
            Some(actual) if actual == &expected_value => {}
            Some(actual) => engine.errors.push(format!(
                "State mismatch for {}: expected {:?}, got {:?}",
                key, expected_value, actual
            )),
            None => engine.errors.push(format!("Missing state key {}", key)),
        }
    }

    let success = engine.errors.is_empty();
    ReasoningOutcome {
        success,
        errors: engine.errors,
        state: engine.state,
    }
}

pub fn state_error(task: &ReasoningTask, ops: &[ThoughtOp]) -> (f32, ReasoningOutcome) {
    let outcome = execute_plan(task, ops);
    let error = outcome.errors.len() as f32;
    (error, outcome)
}

pub fn text_error(expected: &str, actual: &str) -> f32 {
    let expected_norm = expected.trim();
    let actual_norm = actual.trim();
    if expected_norm.is_empty() {
        return 0.0;
    }
    if expected_norm == actual_norm {
        0.0
    } else {
        1.0
    }
}
