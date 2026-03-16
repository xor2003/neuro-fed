// src/types.rs
// Common types used across the NeuroFed Node
// NOTE: Configuration types (NodeConfig, PCConfig, etc.) are now centralized in config.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInput {
    pub id: String,
    pub content: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCommand {
    pub id: String,
    pub command: String,
    pub args: Vec<String>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEvent {
    pub path: PathBuf,
    pub event_type: FileEventType,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileEventType {
    Created,
    Modified,
    Deleted,
    Renamed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeResponse {
    pub id: String,
    pub content: String,
    pub confidence: f32,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeError {
    pub id: String,
    pub message: String,
    pub error_type: ErrorType,
    pub timestamp: SystemTime,
    pub context: HashMap<String, String>,
}

impl std::fmt::Display for NodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "NodeError[{}]: {} (type: {:?})",
            self.id, self.message, self.error_type
        )
    }
}

impl std::error::Error for NodeError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    ValidationError,
    ProcessingError,
    NetworkError,
    StorageError,
    ConfigurationError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NostrEvent {
    pub id: String,
    pub content: String,
    pub author: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

// 🔴 FIX: 100% OpenAI-Compliant Tool Schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    // OpenAI sends arguments as a raw JSON formatted string, not a HashMap
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    // OpenAI uses a JSON Schema object for parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // Usually "function"
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // Usually "function"
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyStats {
    pub requests_total: u64,
    pub requests_successful: u64,
    pub requests_failed: u64,
    pub average_response_time: f32,
    pub last_reset: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeviceType {
    pub name: String,
    pub description: String,
    pub supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationRequest {
    pub id: String,
    pub request_type: String,
    pub payment_proof: String,
    pub pow_proof: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationResponse {
    pub id: String,
    pub success: bool,
    pub message: String,
    pub timestamp: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentVerification {
    pub verified: bool,
    pub amount_sats: u64,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoWVerification {
    pub verified: bool,
    pub nonce: u64,
    pub hash: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FederationError {
    PaymentVerificationFailed(String),
    PoWVerificationFailed(String),
    Timeout(String),
    InvalidRequest(String),
    NostrError(String),
    ConfigError(String),
}

impl std::fmt::Display for FederationError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for FederationError {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MLError {
    ModelLoadError(String),
    RequestError(String),
    SerializationError(String),
    InvalidResponse(String),
    ConfigurationError(String),
    TokenizationError(String),
    TensorError(String),
    FileError(String),
}

impl std::fmt::Display for MLError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
impl std::error::Error for MLError {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ThoughtOp {
    Define,
    Initialize, // <-- НОВОЕ
    Validate,   // <-- НОВОЕ
    Iterate,
    Check,
    Decide, // <-- НОВОЕ
    Compute,
    Aggregate,
    HandleError, // <-- НОВОЕ
    Return,
    Explain,
    EOF,
    Dynamic(String), // ALLOWS INFINITE NEW CHUNKS (e.g., "Iterate_Check")
}

impl std::fmt::Display for ThoughtOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThoughtOp::Define => write!(f, "DEFINE_FUNCTION"),
            ThoughtOp::Initialize => write!(f, "INITIALIZE_VARIABLE"),
            ThoughtOp::Validate => write!(f, "VALIDATE_INPUT"),
            ThoughtOp::Iterate => write!(f, "ITERATE_COLLECTION"),
            ThoughtOp::Check => write!(f, "CHECK_CONDITION"),
            ThoughtOp::Decide => write!(f, "DECIDE_BRANCH"),
            ThoughtOp::Compute => write!(f, "COMPUTE_MATH"),
            ThoughtOp::Aggregate => write!(f, "AGGREGATE_RESULTS"),
            ThoughtOp::HandleError => write!(f, "HANDLE_ERROR"),
            ThoughtOp::Return => write!(f, "RETURN_VALUE"),
            ThoughtOp::Explain => write!(f, "EXPLAIN"),
            ThoughtOp::EOF => write!(f, "EOF"),
            ThoughtOp::Dynamic(s) => write!(f, "{}", s),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkingMemory {
    pub language: String,
    pub entities: HashMap<String, String>,
    pub constraints: Vec<String>,
    pub raw_query: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructuredState {
    pub goal: String,
    pub entities: HashMap<String, String>,
    pub constraints: Vec<String>,
    pub assumptions: Vec<String>,
    pub tests: String, // NEW: Task-specific assertions/unit tests
    pub raw_query: String,
}

impl StructuredState {
    pub fn get_pc_context(&self) -> String {
        let mut ctx = format!("Goal: {}", self.goal);
        if !self.assumptions.is_empty() {
            ctx.push_str("\nCorrected Assumptions: ");
            ctx.push_str(&self.assumptions.join("; "));
        }
        ctx
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub raw_query: String,
    pub query_sequence: Vec<Vec<f32>>, // NEW: Retains temporal sequence structure [seq_len, dim]
    pub novelty: f32,
    pub confidence: f32,
    pub generated_code: String,
    pub thought_sequence: Vec<u32>,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDictionary {
    pub id_to_op: HashMap<u32, ThoughtOp>,
    pub op_to_id: HashMap<ThoughtOp, u32>,
    pub next_id: u32,
}

impl Default for CognitiveDictionary {
    fn default() -> Self {
        let mut dict = Self {
            id_to_op: HashMap::new(),
            op_to_id: HashMap::new(),
            next_id: 0,
        };
        dict.add_op(ThoughtOp::Define);
        dict.add_op(ThoughtOp::Initialize);
        dict.add_op(ThoughtOp::Validate);
        dict.add_op(ThoughtOp::Iterate);
        dict.add_op(ThoughtOp::Check);
        dict.add_op(ThoughtOp::Decide);
        dict.add_op(ThoughtOp::Compute);
        dict.add_op(ThoughtOp::Aggregate);
        dict.add_op(ThoughtOp::HandleError);
        dict.add_op(ThoughtOp::Return);
        dict.add_op(ThoughtOp::Explain);
        dict.add_op(ThoughtOp::EOF);
        dict
    }
}

impl CognitiveDictionary {
    pub fn add_op(&mut self, op: ThoughtOp) -> u32 {
        if let Some(&id) = self.op_to_id.get(&op) {
            return id;
        }
        let id = self.next_id;
        self.id_to_op.insert(id, op.clone());
        self.op_to_id.insert(op, id);
        self.next_id += 1;
        id
    }

    pub fn get_op(&self, id: u32) -> ThoughtOp {
        self.id_to_op.get(&id).cloned().unwrap_or(ThoughtOp::EOF)
    }

    pub fn len(&self) -> usize {
        self.id_to_op.len()
    }

    /// CHUNK DISCOVERY: Finds frequent pairs of thoughts and combines them
    pub fn discover_chunks(&mut self, episodes: &[Episode]) -> usize {
        let mut bigrams: HashMap<(u32, u32), usize> = HashMap::new();
        for ep in episodes {
            if !ep.success {
                continue;
            }
            for window in ep.thought_sequence.windows(2) {
                *bigrams.entry((window[0], window[1])).or_insert(0) += 1;
            }
        }

        let mut new_chunks = 0;
        for ((id1, id2), count) in bigrams {
            if count >= 3 {
                // Threshold for chunking
                // Skip chunking if either thought is EOF (terminal marker)
                if self.get_op(id1) == ThoughtOp::EOF || self.get_op(id2) == ThoughtOp::EOF {
                    continue;
                }
                let name1 = self.get_op(id1).to_string();
                let name2 = self.get_op(id2).to_string();
                let new_concept = format!("{}_{}", name1, name2);

                let dynamic_op = ThoughtOp::Dynamic(new_concept);
                if !self.op_to_id.contains_key(&dynamic_op) {
                    tracing::info!(
                        "✨ Chunk Discovery: Created new thought pattern: {}",
                        dynamic_op
                    );
                    self.add_op(dynamic_op);
                    new_chunks += 1;
                }
            }
        }
        new_chunks
    }
}

/// Holds statistics about the last completed study task.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LastStudyTask {
    pub file_name: String,
    pub paragraphs_processed: u64,
    pub duration_seconds: f64,
}

/// A thread-safe, shared state for tracking document study progress.
#[derive(Debug, Clone, Default)]
pub struct StudyState {
    pub is_studying: bool,
    pub current_file: String,
    pub progress_percent: f64,
    pub last_task: Option<LastStudyTask>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_dictionary_initialization() {
        let dict = CognitiveDictionary::default();
        assert_eq!(dict.len(), 12, "Dictionary should have 12 core ops");
        assert!(dict.op_to_id.contains_key(&ThoughtOp::Define));
    }
}

#[cfg(test)]
mod cognitive_dictionary_tests {
    use super::*;

    #[test]
    fn test_chunk_discovery_creates_new_ops() {
        let mut dict = CognitiveDictionary::default();
        let define_id = dict.op_to_id[&ThoughtOp::Define];
        let check_id = dict.op_to_id[&ThoughtOp::Check];

        // Create 3 successful episodes with the sequence [Define, Check]
        let mut episodes = Vec::new();
        for _ in 0..4 {
            episodes.push(Episode {
                raw_query: "test".into(),
                query_sequence: vec![], // Changed from query_embedding
                novelty: 0.0,
                confidence: 1.0,
                generated_code: "".into(),
                thought_sequence: vec![define_id, check_id, dict.op_to_id[&ThoughtOp::EOF]],
                success: true,
            });
        }

        let initial_len = dict.len();
        let chunks_added = dict.discover_chunks(&episodes);

        // Should discover 1 new chunk: DEFINE_FUNCTION_CHECK_CONDITION (Check+EOF is skipped because EOF is terminal)
        assert_eq!(
            chunks_added, 1,
            "Should discover 1 new chunk (Define+Check) because EOF chunks are skipped"
        );
        assert_eq!(dict.len(), initial_len + 1);

        let new_op1 = ThoughtOp::Dynamic("DEFINE_FUNCTION_CHECK_CONDITION".into());
        assert!(
            dict.op_to_id.contains_key(&new_op1),
            "The combined chunk must exist in the dictionary"
        );
        // The second chunk (CHECK_CONDITION_EOF) is skipped because EOF chunks are not created
    }
}

#[cfg(test)]
mod types_architecture_tests {
    use super::*;

    #[test]
    fn test_structured_state_generates_correct_pc_context() {
        let state = StructuredState {
            goal: "Write a binary search tree".to_string(),
            entities: HashMap::new(),
            constraints: vec!["O(log n)".to_string()],
            assumptions: vec![
                "The array is already sorted".to_string(),
                "Failed previously on empty array".to_string(),
            ],
            tests: "assert bst([1,2,3], 2) == 1".to_string(),
            raw_query: "Provide a BST".to_string(),
        };

        let context = state.get_pc_context();

        // The PC context must include both the goal and the corrective assumptions
        // to ensure it learns from its past failures.
        assert!(context.contains("Goal: Write a binary search tree"));
        assert!(context.contains(
            "Corrected Assumptions: The array is already sorted; Failed previously on empty array"
        ));
    }
}

#[cfg(test)]
mod openai_compatibility_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_definition_serialization() {
        // This test ensures we generate the JSON schema that OpenAI/OpenRouter expects
        let tool = Tool {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get the current weather".to_string()),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })),
            },
        };

        let serialized = serde_json::to_value(&tool).unwrap();

        // Assertions for OpenAI spec requirements
        assert_eq!(serialized["type"], "function");
        assert_eq!(serialized["function"]["name"], "get_weather");
        assert!(serialized["function"]["parameters"].is_object());
    }

    #[test]
    fn test_tool_call_deserialization() {
        // This test simulates a response coming back from an LLM (like DeepSeek or Llama)
        // It ensures we can handle 'arguments' as a raw string.
        let raw_json = json!({
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "solve_math",
                "arguments": "{\"query\": \"sqrt(144)\"}"
            }
        });

        let tool_call: ToolCall = serde_json::from_value(raw_json).unwrap();

        assert_eq!(tool_call.id, "call_123");
        assert_eq!(tool_call.function.name, "solve_math");

        // CRITICAL: Ensure arguments is a String, not a Map
        assert_eq!(tool_call.function.arguments, "{\"query\": \"sqrt(144)\"}");
    }

    #[test]
    fn test_optional_function_fields() {
        // This test ensures that if an IDE sends a tool without a description
        // or parameters, we don't crash (OpenAI allows these to be optional).
        let raw_json = json!({
            "type": "function",
            "function": {
                "name": "simple_ping"
            }
        });

        let res: Result<Tool, _> = serde_json::from_value(raw_json);
        assert!(
            res.is_ok(),
            "Should successfully parse even without optional fields"
        );

        let tool = res.unwrap();
        assert!(tool.function.description.is_none());
        assert!(tool.function.parameters.is_none());
    }

    #[test]
    fn test_continue_extension_compatibility_regression() {
        // This reproduces the exact error reported: missing field 'arguments'
        // in a ToolCall-like structure.
        let raw_json = r#"{
            "id": "abc",
            "type": "function",
            "function": {
                "name": "test",
                "arguments": "{}"
            }
        }"#;

        let res: Result<ToolCall, _> = serde_json::from_str(raw_json);

        assert!(
            res.is_ok(),
            "Regression check failed: ToolCall failed to parse JSON with arguments string. Error: {:?}",
            res.err()
        );
    }
}
