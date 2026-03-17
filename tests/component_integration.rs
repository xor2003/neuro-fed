use neuro_fed_node::reasoning_state::required_ops;
use neuro_fed_node::reasoning_tools::z3_solve_int;
use neuro_fed_node::types::{CognitiveDictionary, ReasoningTask, ThoughtOp};

#[test]
fn test_dictionary_includes_tool_ops() {
    let dict = CognitiveDictionary::default();
    assert!(dict.op_to_id.contains_key(&ThoughtOp::SympyEval));
    assert!(dict.op_to_id.contains_key(&ThoughtOp::Z3Solve));
}

#[test]
fn test_required_ops_include_tool_ops() {
    let sym = ReasoningTask::SympyEval {
        expression: "2+2".to_string(),
        operation: "simplify".to_string(),
        expected: Some("4".to_string()),
    };
    let z3 = ReasoningTask::Z3Solve {
        var: "x".to_string(),
        constraints: vec!["x > 0".to_string()],
        expected: Some(1),
    };

    let sym_ops = required_ops(&sym);
    let z3_ops = required_ops(&z3);
    assert!(sym_ops.contains(&ThoughtOp::SympyEval));
    assert!(z3_ops.contains(&ThoughtOp::Z3Solve));
}

#[test]
#[cfg(not(feature = "z3-tools"))]
fn test_z3_disabled_returns_error() {
    let err = z3_solve_int("x", &["x > 0"]).unwrap_err();
    assert!(
        err.to_string().contains("z3-tools"),
        "Expected z3-tools disabled error, got: {}",
        err
    );
}
