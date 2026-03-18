# OpenAI Smart Proxy Documentation

## Overview

The OpenAI Smart Proxy is a transparent proxy server that sits between clients and OpenAI-compatible APIs (OpenAI API, Ollama, etc.). It provides enhanced capabilities including:

1. **Tool Calling Bypass**: Automatically detects and bypasses tool/function calls to external APIs
2. **Semantic Caching**: Caches responses based on semantic similarity of requests
3. **Predictive Coding Integration**: Uses hierarchical predictive coding for inference and learning
4. **Deterministic Reasoning Tasks**: Routes supported arithmetic/string/list tasks through the internal reasoning state engine before backend fallback
5. **Multiple Backend Support**: Routes requests to OpenAI API, Ollama, or local fallback
6. **Metrics Collection**: Comprehensive monitoring of proxy performance and usage
7. **Intent-Aware Guidance**: Distinguishes chat, reasoning, investigation, code, and text tasks and injects mode-specific execution guidance into the request path

## Architecture

### Core Components

```rust
pub struct OpenAiProxy {
    config: NodeConfig,
    backend_config: BackendConfig,
    local_engine: Arc<tokio::sync::Mutex<MLEngine>>,
    pc_hierarchy: Arc<tokio::sync::Mutex<PredictiveCoding>>,
    client: Client,
    semantic_cache: Arc<tokio::sync::Mutex<HashMap<String, SemanticCacheEntry>>>,
    metrics: Arc<tokio::sync::Mutex<ProxyMetrics>>,
    stats: Arc<tokio::sync::Mutex<ProxyStats>>,
}
```

### Request Flow

1. **Request Reception**: HTTP POST request to `/v1/chat/completions`
2. **Tool Detection**: Check if request contains `tools`, `tool_calls`, or `function_call`
3. **Tool Bypass**: If tools detected and bypass enabled, forward directly to backend
4. **Semantic Cache Check**: Generate embedding, check for semantically similar cached responses
5. **PC Inference**: Decode ThoughtOps and confidence from predictive coding
6. **Reasoning-State Execution**: For supported tasks such as multiply, reverse, sum-even, max, and sort-list, execute a deterministic state plan and return the rendered result
7. **Intent Routing**: Detect whether the request is primarily chat, investigation, code, or text work
8. **Backend Forwarding**: If deterministic reasoning does not apply, forward to configured backend (OpenAI API or Ollama) with intent-aware planning guidance
9. **PC Learning**: If PC learning enabled, learn from the response
10. **Cache Update**: Update semantic cache with new response
11. **Metrics Update**: Update performance metrics

## Key Features

### 1. Tool Calling Bypass

**Purpose**: Tool/function calls require specific API responses that may not be available locally. The proxy detects these requests and bypasses local processing.

**Detection Logic**:
- Checks for `tools` field in request
- Checks for `tool_calls` field in request  
- Checks for `function_call` field in request

**Configuration**:
```toml
[backend_config]
tool_bypass_enabled = true
```

### 2. Semantic Caching

**Purpose**: Cache responses based on semantic similarity rather than exact string matching.

**How it works**:
1. Convert request to text representation
2. Generate embedding using ML Engine
3. Compare with cached embeddings using cosine similarity
4. Return cached response if similarity > threshold (default: 0.8)

**Cache Entry**:
```rust
pub struct SemanticCacheEntry {
    pub embedding: Vec<f32>,
    pub response: OpenAiResponse,
    pub timestamp: SystemTime,
    pub access_count: u64,
}
```

**Eviction Policy**: LRU-like based on access count and timestamp

### 3. Predictive Coding Integration

**Inference Mode**:
- Uses hierarchical predictive coding to generate responses
- Operates on request embeddings
- Returns inferred response if confidence is high

**Learning Mode**:
- Learns from actual API responses
- Updates PC hierarchy weights
- Improves future inference accuracy

### 4. Deterministic Reasoning Tasks

The proxy now detects a small set of structured tasks directly from the last user message and executes them through the shared `reasoning_state` engine instead of waiting on a remote or local language-model fallback.

Supported task families:
- multiplication such as `17 * 23` or `multiply 17 and 23`
- reverse-string requests such as `reverse abc`
- even-sum requests such as `sum even 1 2 4 5`
- maximum selection over integer lists
- sort-list requests over integer lists

For these tasks the proxy:
1. extracts a `ReasoningTask`
2. uses canonical `ThoughtOp` plans from the shared reasoning module
3. executes the state engine
4. renders a deterministic answer
5. stores `reasoning_task` and `expected_output` in episodic memory so sleep/replay training can score state and text quality later

### 5. Benchmark Gate

The reasoning benchmark now validates two layers instead of only one:

1. `state_engine`:
   verifies canonical `ThoughtOp` plans produce the expected state and rendered output
2. `proxy_path`:
   instantiates the OpenAI proxy with a mock local engine and verifies supported queries are answered from `_neurofed_source = reasoning_state` rather than falling back to remote or local text generation

Use:

```bash
cargo run --bin learning_benchmark -- --reasoning-check --skip-run
```

The generated CSV now includes:
- `check_kind`
- `source`
- `expected_output`
- `actual_output`
- `state_error`
- `text_error`
- `fallback_used`

This is intended to catch regressions where deterministic assistant tasks silently stop using the internal reasoning engine.

### 6. Intent Routing

The proxy now assigns each request to a lightweight assistant mode:
- `chat`
- `reasoning`
- `investigation`
- `code_task`
- `text_task`

Current use:
- deterministic reasoning requests are routed to the internal state engine
- non-deterministic investigation/code/text requests receive a mode-specific planning prompt before remote or local fallback

This is a routing layer, not yet a full planner/executor. Its purpose is to keep the assistant behavior from collapsing into a single generic chat mode while the larger planner system is being built.

### 7. Structured Assistant State

`StructuredState` is no longer populated with empty placeholders for non-chat tasks.

For investigation, code, and text requests the proxy now pre-fills:
- `plan_steps`
- `constraints`
- `assumptions`
- `tests`

Examples:
- investigation: evidence-first constraints and an explicit uncertainty/evidence check
- code tasks: inspect-before-edit and build/test verification expectations
- text tasks: preserve meaning while improving clarity/tone

This scaffolding is intentionally lightweight. It gives later planner/executor work a stable state representation without pretending the full task orchestration system already exists.

### 8. Planner Scaffold

For non-chat tasks the proxy now generates an explicit ordered plan inside `StructuredState.plan_steps`.

Examples:
- investigation: restate target -> collect evidence -> synthesize findings
- code task: inspect code path -> implement smallest coherent change -> verify and summarize risks
- text task: identify audience/tone -> rewrite -> check clarity and consistency

This is still a scaffold rather than a full executor, but it changes the assistant contract in an important way:
- planning is now explicit state
- prompt guidance is derived from that state
- later executor work can reuse the same plan representation instead of inventing a new one

### 9. Intent-Aware Local Fallback

When remote generation is unavailable, the proxy no longer falls back to a generic "rephrase the query" response.

The local fallback now renders a structured assistant response from `StructuredState`:
- investigation requests return goal, plan, evidence needs, assumptions, and open questions
- code tasks return goal, plan, constraints, assumptions, and verification steps
- text tasks return goal, rewrite plan, constraints, assumptions, and quality checks
- chat and reasoning still return a simpler fallback, but now include ThoughtOps and local signal metadata when available

This matters because the assistant path now preserves task shape even when generation quality is weak:
- investigation remains evidence-oriented
- code remains verification-oriented
- text work remains constraint-oriented

That makes the fallback path compatible with future planner/executor work instead of collapsing everything back into generic chat.

### 10. Structured Episodic Memory

Assistant episodes now retain more than raw query, ThoughtOps, and final text.

The proxy records the planner scaffold into episodic memory:
- `assistant_intent`
- `goal`
- `plan_steps`
- `constraints`
- `assumptions`
- `tests`
- optional `reasoning_task` and `expected_output`

This is consumed by the sleep/replay path so learning logs include task shape, not just the final answer.

Practical effect:
- investigation episodes preserve evidence-oriented plans
- code episodes preserve verification intent
- text episodes preserve rewrite constraints
- reasoning episodes keep deterministic targets for state/text loss checks

This is the first step toward replaying full assistant behavior instead of replaying only a response string.

### 11. Investigation Memory

The proxy now keeps a dedicated investigation-note layer on top of episodic memory.

When an investigation request succeeds:
- the proxy builds an `InvestigationNote`
- stores a semantic embedding for the query
- persists the note through the main Redb persistence layer when available

Each note captures:
- query and goal
- compact summary
- evidence summary
- open questions
- plan steps
- constraints and assumptions

For later investigation requests, the proxy:
1. embeds the new query
2. retrieves the most similar stored investigation notes
3. injects those prior findings back into the investigation guidance path

This gives the assistant a basic evidence memory:
- repeated investigations can reuse prior findings
- open questions survive across sessions when persistence is enabled
- the retrieval path is narrow and evidence-oriented rather than generic chat history replay

**Configuration**:
```toml
[backend_config]
pc_inference_enabled = true
pc_learning_enabled = true
```

### 4. Multiple Backend Support

**Supported Backends**:
1. **OpenAI API**: Standard OpenAI API endpoints
2. **Ollama**: Local Ollama instance for open-source models
3. **Local Fallback**: ML Engine for local inference (future)

**Routing Logic**:
- Primary: OpenAI API (if API key provided)
- Fallback: Ollama (if local_fallback_enabled)
- Future: Local ML Engine

### 5. Metrics Collection

**ProxyMetrics**:
```rust
pub struct ProxyMetrics {
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub tool_bypass_requests: u64,
    pub pc_inference_calls: u64,
    pub pc_learning_calls: u64,
    pub semantic_similarity_hits: u64,
    pub total_tokens_saved: u64,
    pub average_response_time_ms: f64,
    pub last_updated: SystemTime,
}
```

**ProxyStats**:
```rust
pub struct ProxyStats {
    pub requests_total: u64,
    pub requests_successful: u64,
    pub requests_failed: u64,
    pub average_response_time: f32,
    pub last_reset: SystemTime,
}
```

## Configuration

### BackendConfig

```rust
pub struct BackendConfig {
    pub openai_api_key: Option<String>,
    pub openai_base_url: String,
    pub ollama_base_url: String,
    pub local_fallback_enabled: bool,
    pub tool_bypass_enabled: bool,
    pub semantic_cache_enabled: bool,
    pub semantic_similarity_threshold: f32,
    pub pc_inference_enabled: bool,
    pub pc_learning_enabled: bool,
    pub max_cache_size: usize,
}
```

### Default Configuration

```rust
impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            openai_api_key: None,
            openai_base_url: "https://api.openai.com".to_string(),
            ollama_base_url: "http://localhost:11434".to_string(),
            local_fallback_enabled: true,
            tool_bypass_enabled: true,
            semantic_cache_enabled: true,
            semantic_similarity_threshold: 0.8,
            pc_inference_enabled: true,
            pc_learning_enabled: true,
            max_cache_size: 1000,
        }
    }
}
```

## API Endpoints

### `/v1/chat/completions`
- **Method**: POST
- **Description**: Main chat completion endpoint with enhanced routing
- **Request**: `OpenAiRequest`
- **Response**: `OpenAiResponse`

### `/v1/completions`
- **Method**: POST  
- **Description**: Legacy completions endpoint
- **Request**: `OpenAiRequest`
- **Response**: `OpenAiResponse`

### `/v1/embeddings`
- **Method**: POST
- **Description**: Embeddings generation using local ML Engine
- **Request**: `OpenAiRequest`
- **Response**: OpenAI-compatible embeddings response

### `/v1/models`
- **Method**: GET
- **Description**: List available models
- **Response**: List of model objects

### `/v1/metrics`
- **Method**: GET
- **Description**: Get proxy performance metrics
- **Response**: `ProxyMetrics`

## Usage Examples

### Basic Usage

```rust
use neuro_fed_node::config::{NodeConfig, BackendConfig};
use neuro_fed_node::openai_proxy::OpenAiProxy;
use std::sync::Arc;
use tokio::sync::Mutex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = NodeConfig::default();
    let backend_config = BackendConfig::default();
    
    // Initialize ML Engine
    let device_type = DeviceType {
        name: "CPU".to_string(),
        description: "CPU device".to_string(),
        supported: true,
    };
    let local_engine = Arc::new(Mutex::new(MLEngine::new("model.gguf", device_type)?));
    
    // Initialize Predictive Coding hierarchy
    let pc_config = PCConfig::new(3, vec![512, 256, 128]);
    let pc_hierarchy = Arc::new(Mutex::new(PredictiveCoding::new(pc_config)?));
    
    // Create and start proxy
    let proxy = OpenAiProxy::new(config, backend_config, local_engine, pc_hierarchy);
    proxy.start(8080).await?;
    
    Ok(())
}
```

### Making Requests

```rust
use reqwest::Client;
use serde_json::json;

let client = Client::new();
let response = client
    .post("http://localhost:8080/v1/chat/completions")
    .json(&json!({
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "max_tokens": 100
    }))
    .send()
    .await?
    .json::<serde_json::Value>()
    .await?;
```

## Metrics Dashboard

The proxy collects comprehensive metrics that can be exposed via:

1. **HTTP Endpoint**: `/v1/metrics` returns JSON metrics
2. **Prometheus**: Future integration for monitoring systems
3. **Logging**: Structured logs with request/response details

### Key Metrics

- **Cache Hit Rate**: `cache_hits / (cache_hits + cache_misses)`
- **Tool Bypass Rate**: Percentage of requests with tool calls
- **PC Inference Success Rate**: Successful inferences vs attempts
- **Average Response Time**: Moving average of response times
- **Token Savings**: Estimated tokens saved via caching/inference

## Performance Considerations

### Memory Usage
- Semantic cache stores embeddings (vectors) and responses
- Default max cache size: 1000 entries
- Each entry: ~embedding_dim * 4 bytes + response size

### Latency
- Embedding generation: ~10-50ms (depends on model)
- Cosine similarity: O(n) for cache size
- PC inference: Depends on hierarchy depth and dimensions

### Scalability
- Uses `tokio::sync::Mutex` for thread-safe shared state
- Async/await for non-blocking I/O
- Connection pooling via `reqwest::Client`

## Error Handling

The proxy uses comprehensive error handling with `ProxyError` enum:

```rust
pub enum ProxyError {
    ModelLoadError(String),
    RequestError(String),
    SerializationError(String),
    InvalidResponse(String),
    CacheError(String),
    ConfigError(String),
    PCError(String),
    EmbeddingError(String),
    BackendError(String),
}
```

All errors are logged with appropriate severity levels and returned as HTTP 500 errors.

## Testing

### Unit Tests
- Tool detection logic
- Semantic cache operations
- Cosine similarity calculations
- Request/response serialization
- Metrics collection

### Integration Tests
- End-to-end request flow
- Backend forwarding (OpenAI/Ollama)
- Cache hit/miss scenarios
- Error handling

## Future Enhancements

1. **Dynamic Routing**: AI-based backend selection
2. **Adaptive Caching**: Self-tuning similarity thresholds
3. **Multi-modal Support**: Image/text embeddings
4. **Distributed Cache**: Redis/Memcached integration
5. **Advanced Metrics**: Prometheus exporter, Grafana dashboards
6. **Rate Limiting**: Per-user/API key rate limits
7. **Request Batching**: Combine similar requests
8. **Response Compression**: Reduce bandwidth usage

## Configuration Examples

### Basic Configuration

```toml
[backend_config]
openai_api_key = "sk-..."
openai_base_url = "https://api.openai.com"
ollama_base_url = "http://localhost:11434"
local_fallback_enabled = true
tool_bypass_enabled = true
semantic_cache_enabled = true
semantic_similarity_threshold = 0.85
pc_inference_enabled = true
pc_learning_enabled = true
max_cache_size = 500
```

### Advanced Configuration

```toml
[backend_config]
openai_api_key = "sk-..."
openai_base_url = "https://api.openai.com"
ollama_base_url = "http://localhost:11434"
local_fallback_enabled = false  # Disable local fallback
tool_bypass_enabled = true
semantic_cache_enabled = true
semantic_similarity_threshold = 0.9  # Higher threshold = more exact matches
pc_inference_enabled = false  # Disable PC inference
pc_learning_enabled = true   # But keep learning from responses
max_cache_size = 1000       # Larger cache
```

## Monitoring and Observability

### Logging Levels
- `error`: Critical failures
- `warn`: Non-critical issues
- `info`: Request/response summaries
- `debug`: Detailed processing steps
- `trace`: Verbose debugging

### Health Checks
- `/health`: Basic health endpoint
- `/ready`: Readiness probe
- `/metrics`: Performance metrics

## Security Considerations

1. **API Key Protection**: OpenAI API keys stored in configuration
2. **Request Validation**: Validate all incoming requests
3. **Rate Limiting**: Prevent abuse
4. **Input Sanitization**: Protect against injection attacks
5. **TLS/SSL**: Secure communication (future)

## Deployment

### Docker
```dockerfile
FROM rust:latest
WORKDIR /app
COPY . .
RUN cargo build --release
EXPOSE 8080
CMD ["./target/release/neuro-fed-node"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuro-fed-proxy
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: proxy
        image: neurofed/proxy:latest
        ports:
        - containerPort: 8080
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
```

## Troubleshooting

### Common Issues

1. **Cache Not Working**: Check `semantic_cache_enabled` and embedding model
2. **PC Inference Failing**: Verify PC hierarchy initialization
3. **High Memory Usage**: Reduce `max_cache_size`
4. **Slow Responses**: Check backend connectivity and ML Engine performance
5. **Tool Calls Not Bypassed**: Ensure `tool_bypass_enabled` is true

### Debugging
- Enable debug logging: `RUST_LOG=debug`
- Check metrics endpoint: `curl http://localhost:8080/v1/metrics`
- Monitor cache stats: Check `/v1/metrics` for hit rates

## Conclusion

The OpenAI Smart Proxy provides a powerful, extensible layer between clients and AI APIs. By combining semantic caching, predictive coding, and intelligent routing, it reduces API costs, improves response times, and enables local inference capabilities.
