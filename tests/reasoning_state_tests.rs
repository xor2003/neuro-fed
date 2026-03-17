use neuro_fed_node::reasoning_state::{execute_plan, StateValue};
use neuro_fed_node::types::{ReasoningTask, ThoughtOp};

#[test]
fn test_multiply_requires_compute() {
    let task = ReasoningTask::Multiply { a: 6, b: 7 };
    let ops = vec![ThoughtOp::Initialize, ThoughtOp::Return];
    let outcome = execute_plan(&task, &ops);

    assert!(!outcome.success, "Expected failure without Compute op");
    assert!(
        outcome
            .errors
            .iter()
            .any(|e| e.contains("Missing required op")),
        "Expected missing required op error"
    );
}

#[test]
fn test_multiply_state_is_correct() {
    let task = ReasoningTask::Multiply { a: 6, b: 7 };
    let ops = vec![
        ThoughtOp::Initialize,
        ThoughtOp::Compute,
        ThoughtOp::Return,
    ];
    let outcome = execute_plan(&task, &ops);

    assert!(outcome.success, "Expected successful reasoning outcome");
    assert_eq!(
        outcome.state.get("result"),
        Some(&StateValue::Int(42))
    );
}

#[test]
fn test_reverse_string_state_is_correct() {
    let task = ReasoningTask::ReverseString {
        input: "hello".to_string(),
    };
    let ops = vec![
        ThoughtOp::Initialize,
        ThoughtOp::Iterate,
        ThoughtOp::Return,
    ];
    let outcome = execute_plan(&task, &ops);

    assert!(outcome.success, "Expected successful reasoning outcome");
    assert_eq!(
        outcome.state.get("output"),
        Some(&StateValue::Str("olleh".to_string()))
    );
}
