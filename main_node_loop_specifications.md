# Main.rs + Node Loop Component Technical Specifications

## Overview
`main.rs` and `node_loop.rs` implement the core application entry point and async processing loop. It orchestrates all components: user input, file watching, Nostr events, PC inference, and response generation.

## Architecture

### Core Data Structures
```rust
// Public API
pub struct NeuroPCNode {
    config: NodeConfig,
    llama_ctx: LlamaContext,
    pc_hierarchy: PredictiveCoding,
    nostr_federation: NostrFederation,
    bootstrap: Bootstrap,
    node_loop: NodeLoop,
    web_ui: Option<WebUI>,
    shutdown_signal: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct NodeConfig {
    data_paths: Vec<String>,
    model_path: String,
    context_size: usize,
    inference_steps: usize,
    learning_rate: f32,
    surprise_threshold: f32,
    selective_update: bool,
    web_ui_enabled: bool,
    web_ui_port: u16,
    log_level: String,
    bootstrap_on_start: bool,
}

pub struct NodeLoop {
    rx_user_input: Receiver<UserInput>,
    rx_file_events: Receiver<FileEvent>,
    rx_nostr_events: Receiver<NostrEvent>,
    tx_responses: Sender<NodeResponse>,
    shutdown_signal: Arc<AtomicBool>,
    config: NodeLoopConfig,
}

#[derive(Debug, Clone)]
pub struct NodeLoopConfig {
    max_batch_size: usize,
    input_timeout: Duration,
    file_watch_interval: Duration,
    nostr_event_timeout: Duration,
    response_queue_size: usize,
}

#[derive(Debug, Clone)]
pub enum UserInput {
    Text(String),
    File(PathBuf),
    Command(NodeCommand),
    InteractiveMode,
}

#[derive(Debug, Clone)]
pub enum NodeCommand {
    Bootstrap,
    ResetHierarchy,
    SaveState,
    LoadState,
    Status,
    TrustReport,
    ZapReport,
}

#[derive(Debug, Clone)]
pub enum FileEvent {
    Created(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
}

#[derive(Debug, Clone)]
pub enum NodeResponse {
    Text(String),
    File(PathBuf),
    Status(NodeStatus),
    Error(NodeError),
    EventProcessed,
}

#[derive(Debug, Clone)]
pub struct NodeStatus {
    free_energy: f32,
    surprise_score: f32,
    trust_level: f32,
    zap_balance: u64,
    connected_relays: usize,
    processed_events: usize,
    learned_patterns: usize,
}

#[derive(Debug, Clone)]
pub enum NodeError {
    ProcessingError(String),
    FileError(String),
    NetworkError(String),
    ModelError(String),
}

pub struct WebUI {
    server: actix_web::HttpServer,
    app: actix_web::App<NeuroPCNodeState>,
    port: u16,
}

pub struct NeuroPCNodeState {
    node: Arc<Mutex<NeuroPCNode>>,
}
```

### Main Application Flow
```rust
// main.rs
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    setup_logging();
    
    // Load configuration
    let config = NodeConfig::load_from_file("config.toml")?;
    
    // Create NeuroPCNode instance
    let mut node = NeuroPCNode::new(config).await?;
    
    // Bootstrap if configured
    if node.config.bootstrap_on_start {
        node.bootstrap_and_initialize().await?;
    }
    
    // Start node loop
    node.start().await?;
    
    Ok(())
}

// NeuroPCNode implementation
impl NeuroPCNode {
    pub async fn new(config: NodeConfig) -> Result<Self, NodeError> {
        // Initialize components
        let llama_ctx = LlamaContext::new(&config.model_path, config.context_size)?;
        let pc_hierarchy = PredictiveCoding::new(PCConfig::from_node_config(&config));
        let nostr_federation = NostrFederation::new(NostrConfig::from_node_config(&config))?;
        let bootstrap = Bootstrap::new(BootstrapConfig::from_node_config(&config));
        let node_loop = NodeLoop::new(NodeLoopConfig::from_node_config(&config));
        
        let web_ui = if config.web_ui_enabled {
            Some(WebUI::new(config.web_ui_port)?)
        } else {
            None
        };
        
        Ok(NeuroPCNode {
            config,
            llama_ctx,
            pc_hierarchy,
            nostr_federation,
            bootstrap,
            node_loop,
            web_ui,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        })
    }
    
    pub async fn start(&mut self) -> Result<(), NodeError> {
        // Start web UI if enabled
        if let Some(ref mut web_ui) = self.web_ui {
            web_ui.start().await?;
        }
        
        // Start Nostr federation
        self.nostr_federation.start_subscription().await?;
        
        // Start node loop
        self.node_loop.start(self.shutdown_signal.clone()).await?;
        
        // Main application loop
        self.run_main_loop().await?;
        
        Ok(())
    }
    
    async fn run_main_loop(&mut self) -> Result<(), NodeError> {
        // Create channels for communication
        let (tx_user_input, rx_user_input) = mpsc::channel(100);
        let (tx_file_events, rx_file_events) = mpsc::channel(100);
        let (tx_nostr_events, rx_nostr_events) = mpsc::channel(100);
        let (tx_responses, rx_responses) = mpsc::channel(100);
        
        // Set up file watcher
        let mut file_watcher = FileWatcher::new(&self.config.data_paths);
        
        // Set up user input handler
        let mut user_input_handler = UserInputHandler::new(tx_user_input.clone());
        
        // Main processing loop
        loop {
            // Check for shutdown signal
            if self.shutdown_signal.load(atomic::Ordering::SeqCst) {
                break;
            }
            
            // Handle file events
            if let Some(event) = file_watcher.poll_event() {
                tx_file_events.send(event).await?;
            }
            
            // Handle user input
            if let Some(input) = user_input_handler.poll_input() {
                tx_user_input.send(input).await?;
            }
            
            // Handle Nostr events
            if let Some(event) = self.nostr_federation.poll_event() {
                tx_nostr_events.send(event).await?;
            }
            
            // Process queued events
            self.process_queued_events(
                &rx_user_input,
                &rx_file_events,
                &rx_nostr_events,
                &tx_responses
            ).await?;
            
            // Send responses
            self.send_responses(&rx_responses).await?;
            
            // Sleep briefly to prevent busy waiting
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        Ok(())
    }
}
```

## Event Processing

### Node Loop Implementation
```rust
impl NodeLoop {
    pub async fn start(&mut self, shutdown_signal: Arc<AtomicBool>) -> Result<(), NodeError> {
        // Start processing loop
        tokio::spawn(self.processing_loop(shutdown_signal));
        Ok(())
    }
    
    async fn processing_loop(&mut self, shutdown_signal: Arc<AtomicBo/home/xor/data/_rtksave/phone/sd/DCIMol>) {
        while !shutdown_signal.load(atomic::Ordering::SeqCst) {
            // Process events in batches
            let batch = self.collect_events().await;
            
            if !batch.is_empty() {
                self.process_batch(batch).await;
            }
            
            // Sleep briefly
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    
    async fn collect_events(&mut self) -> Vec<EventWrapper> {
        let mut events = Vec::new();
        
        // Collect user input events
        while let Ok(input) = self.rx_user_input.try_recv() {
            events.push(EventWrapper::UserInput(input));
        }
        
        // Collect file events
        while let Ok(file_event) = self.rx_file_events.try_recv() {
            events.push(EventWrapper::FileEvent(file_event));
        }
        
        // Collect Nostr events
        while let Ok(nostr_event) = self.rx_nostr_events.try_recv() {
            events.push(EventWrapper::NostrEvent(nostr_event));
        }
        
        events
    }
    
    async fn process_batch(&mut self, batch: Vec<EventWrapper>) {
        // Process events in order
        for event in batch {
            match event {
                EventWrapper::UserInput(input) => {
                    self.process_user_input(input).await;
                }
                EventWrapper::FileEvent(file_event) => {
                    self.process_file_event(file_event).await;
                }
                EventWrapper::NostrEvent(nostr_event) => {
                    self.process_nostr_event(nostr_event).await;
                }
            }
        }
    }
}

#[derive(Debug)]
enum EventWrapper {
    UserInput(UserInput),
    FileEvent(FileEvent),
    NostrEvent(NostrEvent),
}
```

### User Input Processing
```rust
impl NodeLoop {
    async fn process_user_input(&mut self, input: UserInput) {
        match input {
            UserInput::Text(text) => {
                self.process_text_input(text).await;
            }
            UserInput::File(path) => {
                self.process_file_input(path).await;
            }
            UserInput::Command(command) => {
                self.process_command(command).await;
            }
            UserInput::InteractiveMode => {
                self.enter_interactive_mode().await;
            }
        }
    }
    
    async fn process_text_input(&mut self, text: String) {
        // Embed text using llama FFI
        let embedding = self.llama_ctx.embed(&text).await?;
        
        // Process through PC hierarchy
        let stats = self.pc_hierarchy.learn(&embedding.data).await?;
        
        // Generate response
        let response = self.generate_response(&text, &stats).await;
        
        // Send response
        self.tx_responses.send(NodeResponse::Text(response)).await?;
        
        // Publish delta if surprise is high
        if stats.total_surprise > self.config.surprise_threshold {
            self.publish_delta(&text, &stats).await?;
        }
    }
    
    async fn generate_response(&self, text: &str, stats: &SurpriseStats) -> String {
        // Simple response generation based on surprise
        if stats.total_surprise > self.config.surprise_threshold * 2.0 {
            format!("That's surprising! I'm learning about: {}", text)
        } else if stats.total_surprise > self.config.surprise_threshold {
            format!("Interesting: {}", text)
        } else {
            format!("I understand: {}", text)
        }
    }
}
```

### File Event Processing
```rust
impl NodeLoop {
    async fn process_file_event(&mut self, event: FileEvent) {
        match event {
            FileEvent::Created(path) | FileEvent::Modified(path) => {
                self.process_file_content(&path).await;
            }
            FileEvent::Deleted(path) => {
                self.handle_file_deletion(&path).await;
            }
        }
    }
    
    async fn process_file_content(&mut self, path: &Path) {
        // Read file content
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                log::error!("Failed to read file {}: {}", path.display(), e);
                return;
            }
        };
        
        // Process content as text input
        self.process_text_input(content).await;
    }
    
    async fn handle_file_deletion(&mut self, path: &Path) {
        log::info!("File deleted: {}", path.display());
        // Optionally remove cached information about this file
    }
}
```

### Nostr Event Processing
```rust
impl NodeLoop {
    async fn process_nostr_event(&mut self, event: NostrEvent) {
        // Forward to Nostr federation for processing
        if let Err(e) = self.nostr_federation.handle_event(event).await {
            log::error!("Failed to process Nostr event: {}", e);
        }
    }
}
```

## Web UI Integration

### Actix Web Server
```rust
impl WebUI {
    pub fn new(port: u16) -> Result<Self, NodeError> {
        let server = HttpServer::new(|| {
            App::new()
                .app_data(web::Data::new(NeuroPCNodeState::default()))
                .service(web::resource("/").route(web::get().to(index)))
                .service(web::resource("/status").route(web::get().to(status)))
                .service(web::resource("/input").route(web::post().to(handle_input)))
                .service(web::resource("/trust").route(web::get().to(trust_report)))
                .service(web::resource("/zap").route(web::get().to(zap_report)))
        });
        
        Ok(WebUI { server, app, port })
    }
    
    pub async fn start(&mut self) -> Result<(), NodeError> {
        self.server.bind(format!("127.0.0.1:{}", self.port))?.run().await?;
        Ok(())
    }
}

async fn index(data: web::Data<NeuroPCNodeState>) -> impl Responder {
    let node = data.node.lock().await;
    HttpResponse::Ok().body(format!(
        "NeuroPC Node v0.1\nFree Energy: {:.2}\nSurprise: {:.2}",
        node.pc_hierarchy.free_energy,
        node.pc_hierarchy.surprise_score
    ))
}

async fn status(data: web::Data<NeuroPCNodeState>) -> impl Responder {
    let node = data.node.lock().await;
    let status = NodeStatus {
        free_energy: node.pc_hierarchy.free_energy,
        surprise_score: node.pc_hierarchy.surprise_score,
        trust_level: node.nostr_federation.trust_manager.get_average_trust(),
        zap_balance: node.nostr_federation.zap_handler.get_balance().await,
        connected_relays: node.nostr_federation.client.get_connected_relays().await,
        processed_events: node.node_loop.get_processed_events(),
        learned_patterns: node.pc_hierarchy.get_learned_patterns(),
    };
    
    HttpResponse::Ok().json(status)
}
```

## Configuration Examples

### Node Configuration
```rust
let node_config = NodeConfig {
    data_paths: vec![
        "/home/user/documents".to_string(),
        "/home/user/chat_history".to_string(),
    ],
    model_path: "models/llama-3.2-3B.Q4_K_M.gguf".to_string(),
    context_size: 2048,
    inference_steps: 20,
    learning_rate: 0.01,
    surprise_threshold: 1.0,
    selective_update: true,
    web_ui_enabled: true,
    web_ui_port: 8080,
    log_level: "info".to_string(),
    bootstrap_on_start: true,
};
```

### Node Loop Configuration
```rust
let loop_config = NodeLoopConfig {
    max_batch_size: 100,
    input_timeout: Duration::from_millis(100),
    file_watch_interval: Duration::from_millis(500),
    nostr_event_timeout: Duration::from_millis(200),
    response_queue_size: 1000,
};
```

## Error Handling

### Custom Error Types
```rust
#[derive(Debug, thiserror::Error)]
pub enum NodeError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    #[error("Model loading failed: {0}")]
    ModelError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("File I/O error: {0}")]
    FileError(String),
    
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    #[error("Web UI error: {0}")]
    WebUIError(String),
}
```

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_text_processing() {
        let mut node = NeuroPCNode::new(test_config()).await.unwrap();
        
        let response = node.process_text_input("test message").await.unwrap();
        assert!(response.starts_with("I understand") || response.starts_with("Interesting"));
    }
    
    #[tokio::test]
    async fn test_file_watching() {
        let mut node = NeuroPCNode::new(test_config()).await.unwrap();
        
        // Create test file
        let test_path = "/tmp/test_file.txt";
        std::fs::write(test_path, "test content").unwrap();
        
        // Wait for file event
        tokio::time::sleep(Duration::from_millis(1000)).await;
        
        // Verify file was processed
        assert!(node.node_loop.get_processed_files() > 0);
    }
}
```

### Integration Tests
- Test complete end-to-end processing flow
- Verify Nostr event handling and delta publishing
- Test web UI endpoints and responses
- Benchmark performance with various input sizes

## Dependencies

### Required
- `tokio = { version = "1.0", features = ["full"] }` - Async runtime
- `actix-web = "4.0"` - Web server for UI
- `notify = "5.0"` - File system watching
- `serde = { version = "1.0", features = ["derive"] }` - Serialization
- `serde_json = "1.0"` - JSON serialization
- `thiserror = "1.0"` - Error handling
- `chrono = "0.4"` - Timestamp handling

### Optional
- `tracing = "0.1"` - Structured logging
- `metrics = "0.20"` - Performance metrics
- `prometheus = "0.13"` - Prometheus metrics integration

## Performance Considerations

### Async Processing
- Use Tokio's async/await for non-blocking I/O
- Implement proper backpressure with bounded channels
- Use parallel processing for independent tasks

### Memory Management
```rust
impl NodeLoop {
    fn optimize_memory(&mut self) {
        // Pre-allocate channel buffers
        self.rx_user_input.reserve(100);
        self.rx_file_events.reserve(100);
        self.rx_nostr_events.reserve(100);
        
        // Use efficient data structures
        self.config.data_paths.shrink_to_fit();
    }
}
```

### Resource Cleanup
```rust
impl Drop for NeuroPCNode {
    fn drop(&mut self) {
        // Clean up resources
        self.shutdown_signal.store(true, atomic::Ordering::SeqCst);
        
        // Wait for graceful shutdown
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(self.node_loop.shutdown())
            .unwrap();
        
        // Save state if needed
        self.save_state().unwrap();
    }
}
```

This specification provides a complete blueprint for implementing the main application entry point and async processing loop, orchestrating all components of the NeuroPC Node system.