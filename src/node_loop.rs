// src/node_loop.rs
// Core node loop implementation for processing user input, file events, and Nostr events

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{info, error, debug, warn};

use crate::UserInput;
use crate::NodeCommand;
use crate::FileEvent;
use crate::NodeResponse;
use crate::NodeError;

pub struct NodeLoop {
    rx_user_input: mpsc::Receiver<UserInput>,
    rx_file_events: mpsc::Receiver<FileEvent>,
    rx_nostr_events: mpsc::Receiver<NostrEvent>,
    tx_responses: mpsc::Sender<NodeResponse>,
    shutdown_signal: Arc<AtomicBool>,
    config: NodeLoopConfig,
}

#[derive(Debug, Clone)]
pub struct NodeLoopConfig {
    pub max_batch_size: usize,
    pub input_timeout: Duration,
    pub file_watch_interval: Duration,
    pub nostr_event_timeout: Duration,
    pub response_queue_size: usize,
}

#[derive(Debug, Clone)]
pub enum UserInput {
    Text(String),
    File(std::path::PathBuf),
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
    Created(std::path::PathBuf),
    Modified(std::path::PathBuf),
    Deleted(std::path::PathBuf),
}

#[derive(Debug, Clone)]
pub enum NodeResponse {
    Text(String),
    File(std::path::PathBuf),
    Status(NodeStatus),
    Error(NodeError),
    EventProcessed,
}

#[derive(Debug, Clone)]
pub struct NodeStatus {
    pub free_energy: f32,
    pub surprise_score: f32,
    pub trust_level: f32,
    pub zap_balance: u64,
    pub connected_relays: usize,
    pub processed_events: usize,
    pub learned_patterns: usize,
}

#[derive(Debug, Clone)]
pub enum NodeError {
    ProcessingError(String),
    FileError(String),
    NetworkError(String),
    ModelError(String),
}

#[derive(Debug, Clone)]
pub struct NostrEvent {
    pub kind: String,
    pub content: String,
    pub pubkey: String,
    pub timestamp: i64,
}

impl NodeLoop {
    pub fn new(config: NodeLoopConfig) -> Self {
        let (tx_user_input, rx_user_input) = mpsc::channel(100);
        let (tx_file_events, rx_file_events) = mpsc::channel(100);
        let (tx_nostr_events, rx_nostr_events) = mpsc::channel(100);
        let (tx_responses, rx_responses) = mpsc::channel(100);
        
        NodeLoop {
            rx_user_input,
            rx_file_events,
            rx_nostr_events,
            tx_responses,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
            config,
        }
    }

    pub async fn start(&mut self, shutdown_signal: Arc<AtomicBool>) -> Result<(), NodeError> {
        self.shutdown_signal = shutdown_signal;
        
        // Start processing loop
        let mut processing_loop = self.process_events();
        
        // Start response handler
        let mut response_handler = self.handle_responses();
        
        // Run both loops concurrently
        tokio::select! {
            res = processing_loop => {
                if let Err(e) = res {
                    error!("Processing loop failed: {}", e);
                    return Err(e);
                }
            }
            res = response_handler => {
                if let Err(e) = res {
                    error!("Response handler failed: {}", e);
                    return Err(e);
                }
            }
        }
        
        Ok(())
    }

    async fn process_events(&mut self) -> Result<(), NodeError> {
        let mut event_interval = interval(Duration::from_millis(100));
        
        loop {
            tokio::select! {
                // Handle user input
                Some(input) = self.rx_user_input.recv() => {
                    self.handle_user_input(input).await?;
                }
                
                // Handle file events
                Some(event) = self.rx_file_events.recv() => {
                    self.handle_file_event(event).await?;
                }
                
                // Handle Nostr events
                Some(event) = self.rx_nostr_events.recv() => {
                    self.handle_nostr_event(event).await?;
                }
                
                // Periodic tasks
                _ = event_interval.tick() => {
                    self.periodic_tasks().await?;
                }
                
                // Check for shutdown
                _ = self.check_shutdown() => {
                    info!("Shutdown signal received, exiting processing loop...");
                    break;
                }
            }
        }
        
        Ok(())
    }

    async fn handle_user_input(&mut self, input: UserInput) -> Result<(), NodeError> {
        match input {
            UserInput::Text(text) => {
                info!("Processing user text input: {}", text);
                // Process text input through PC hierarchy
                self.process_text_input(text).await?;
            }
            UserInput::File(path) => {
                info!("Processing file input: {:?}", path);
                // Process file through PC hierarchy
                self.process_file_input(path).await?;
            }
            UserInput::Command(cmd) => {
                info!("Processing command: {:?}", cmd);
                self.process_command(cmd).await?;
            }
            UserInput::InteractiveMode => {
                info!("Entering interactive mode...");
                self.enter_interactive_mode().await?;
            }
        }
        
        Ok(())
    }

    async fn handle_file_event(&mut self, event: FileEvent) -> Result<(), NodeError> {
        match event {
            FileEvent::Created(path) => {
                info!("File created: {:?}", path);
                // Process new file
                self.process_new_file(path).await?;
            }
            FileEvent::Modified(path) => {
                info!("File modified: {:?}", path);
                // Process modified file
                self.process_modified_file(path).await?;
            }
            FileEvent::Deleted(path) => {
                info!("File deleted: {:?}", path);
                // Handle file deletion
                self.process_deleted_file(path).await?;
            }
        }
        
        Ok(())
    }

    async fn handle_nostr_event(&mut self, event: NostrEvent) -> Result<(), NodeError> {
        info!("Processing Nostr event: {} from {}", event.kind, event.pubkey);
        
        // Process Nostr event through PC hierarchy
        self.process_nostr_event(event).await?;
        
        Ok(())
    }

    async fn handle_responses(&mut self) -> Result<(), NodeError> {
        loop {
            // Mock implementation - would receive responses from processing
            // For now, just simulate some responses
            let response = NodeResponse::Text("Processing complete".to_string());
            
            if let Err(e) = self.tx_responses.send(response).await {
                error!("Failed to send response: {}", e);
            }
            
            // Sleep briefly to simulate processing
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn process_text_input(&mut self, text: String) -> Result<(), NodeError> {
        // Mock implementation - would process text through PC hierarchy
        info!("Processing text input: {} characters", text.len());
        
        // Simulate processing
        let response = NodeResponse::Text(format!("Processed {} characters of text", text.len()));
        
        if let Err(e) = self.tx_responses.send(response).await {
            error!("Failed to send text response: {}", e);
        }
        
        Ok(())
    }

    async fn process_file_input(&mut self, path: std::path::PathBuf) -> Result<(), NodeError> {
        // Mock implementation - would process file through PC hierarchy
        info!("Processing file input: {:?}", path);
        
        // Simulate processing
        let response = NodeResponse::Text(format!("Processed file: {:?}", path));
        
        if let Err(e) = self.tx_responses.send(response).await {
            error!("Failed to send file response: {}", e);
        }
        
        Ok(())
    }

    async fn process_command(&mut self, cmd: NodeCommand) -> Result<(), NodeError> {
        match cmd {
            NodeCommand::Bootstrap => {
                info!("Processing bootstrap command...");
                // Trigger bootstrap process
                let response = NodeResponse::Text("Bootstrap initiated".to_string());
                self.tx_responses.send(response).await?;
            }
            NodeCommand::ResetHierarchy => {
                info!("Processing reset hierarchy command...");
                let response = NodeResponse::Text("Hierarchy reset".to_string());
                self.tx_responses.send(response).await?;
            }
            NodeCommand::SaveState => {
                info!("Processing save state command...");
                let response = NodeResponse::Text("State saved".to_string());
                self.tx_responses.send(response).await?;
            }
            NodeCommand::LoadState => {
                info!("Processing load state command...");
                let response = NodeResponse::Text("State loaded".to_string());
                self.tx_responses.send(response).await?;
            }
            NodeCommand::Status => {
                info!("Processing status command...");
                let status = NodeStatus {
                    free_energy: 0.0,
                    surprise_score: 0.0,
                    trust_level: 0.0,
                    zap_balance: 0,
                    connected_relays: 0,
                    processed_events: 0,
                    learned_patterns: 0,
                };
                let response = NodeResponse::Status(status);
                self.tx_responses.send(response).await?;
            }
            NodeCommand::TrustReport => {
                info!("Processing trust report command...");
                let response = NodeResponse::Text("Trust report generated".to_string());
                self.tx_responses.send(response).await?;
            }
            NodeCommand::ZapReport => {
                info!("Processing zap report command...");
                let response = NodeResponse::Text("Zap report generated".to_string());
                self.tx_responses.send(response).await?;
            }
        }
        
        Ok(())
    }

    async fn enter_interactive_mode(&mut self) -> Result<(), NodeError> {
        info!("Entering interactive mode...");
        
        // Mock implementation - would enter interactive mode
        let response = NodeResponse::Text("Interactive mode activated".to_string());
        self.tx_responses.send(response).await?;
        
        Ok(())
    }

    async fn process_new_file(&mut self, path: std::path::PathBuf) -> Result<(), NodeError> {
        info!("Processing new file: {:?}", path);
        
        // Mock implementation - would process new file
        let response = NodeResponse::Text(format!("New file processed: {:?}", path));
        self.tx_responses.send(response).await?;
        
        Ok(())
    }

    async fn process_modified_file(&mut self, path: std::path::PathBuf) -> Result<(), NodeError> {
        info!("Processing modified file: {:?}", path);
        
        // Mock implementation - would process modified file
        let response = NodeResponse::Text(format!("Modified file processed: {:?}", path));
        self.tx_responses.send(response).await?;
        
        Ok(())
    }

    async fn process_deleted_file(&mut self, path: std::path::PathBuf) -> Result<(), NodeError> {
        info!("Processing deleted file: {:?}", path);
        
        // Mock implementation - would handle deleted file
        let response = NodeResponse::Text(format!("Deleted file processed: {:?}", path));
        self.tx_responses.send(response).await?;
        
        Ok(())
    }

    async fn process_nostr_event(&mut self, event: NostrEvent) -> Result<(), NodeError> {
        info!("Processing Nostr event: {} from {}", event.kind, event.pubkey);
        
        // Mock implementation - would process Nostr event
        let response = NodeResponse::Text(format!("Nostr event processed: {}", event.kind));
        self.tx_responses.send(response).await?;
        
        Ok(())
    }

    async fn periodic_tasks(&mut self) -> Result<(), NodeError> {
        // Mock implementation - would perform periodic tasks
        debug!("Running periodic tasks...");
        Ok(())
    }

    async fn check_for_new_files(&mut self) -> Result<(), NodeError> {
        // Mock implementation - would check for new files
        debug!("Checking for new files...");
        Ok(())
    }

    async fn process_pending_events(&mut self) -> Result<(), NodeError> {
        // Mock implementation - would process pending events
        debug!("Processing pending events...");
        Ok(())
    }

    fn check_shutdown(&self) -> Result<bool, NodeError> {
        if self.shutdown_signal.load(Ordering::Relaxed) {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn send_user_input(&mut self, input: UserInput) -> Result<(), NodeError> {
        if let Err(e) = self.rx_user_input.send(input).await {
            error!("Failed to send user input: {}", e);
            return Err(NodeError::ProcessingError(e.to_string()));
        }
        Ok(())
    }

    pub async fn send_file_event(&mut self, event: FileEvent) -> Result<(), NodeError> {
        if let Err(e) = self.rx_file_events.send(event).await {
            error!("Failed to send file event: {}", e);
            return Err(NodeError::ProcessingError(e.to_string()));
        }
        Ok(())
    }

    pub async fn send_nostr_event(&mut self, event: NostrEvent) -> Result<(), NodeError> {
        if let Err(e) = self.rx_nostr_events.send(event).await {
            error!("Failed to send Nostr event: {}", e);
            return Err(NodeError::ProcessingError(e.to_string()));
        }
        Ok(())
    }

    pub async fn receive_response(&mut self) -> Result<Option<NodeResponse>, NodeError> {
        match self.tx_responses.recv().await {
            Some(response) => Ok(Some(response)),
            None => Ok(None),
        }
    }
}

// Mock NostrEvent implementation
#[derive(Debug, Clone)]
pub struct NostrEvent {
    pub kind: String,
    pub content: String,
    pub pubkey: String,
    pub timestamp: i64,
}

// Example usage
#[cfg(test)]
pub fn example_usage() {
    // Create node loop
    let config = NodeLoopConfig {
        max_batch_size: 100,
        input_timeout: Duration::from_millis(500),
        file_watch_interval: Duration::from_secs(1),
        nostr_event_timeout: Duration::from_secs(30),
        response_queue_size: 100,
    };

    let mut node_loop = NodeLoop::new(config);
    
    // Create shutdown signal
    let shutdown_signal = Arc::new(AtomicBool::new(false));
    
    // Start node loop
    tokio::spawn(async move {
        if let Err(e) = node_loop.start(shutdown_signal.clone()).await {
            error!("Node loop failed: {}", e);
        }
    });
    
    // Send some test events
    let test_input = UserInput::Text("Hello, this is a test input".to_string());
    let test_file_event = FileEvent::Created(std::path::PathBuf::from("test.txt"));
    let test_nostr_event = NostrEvent {
        kind: "PCErrorDelta".to_string(),
        content: "Test delta content".to_string(),
        pubkey: "npub1test".to_string(),
        timestamp: 1234567890,
    };
    
    // Send events (in real code, this would be done by other components)
    node_loop.send_user_input(test_input).await.unwrap();
    node_loop.send_file_event(test_file_event).await.unwrap();
    node_loop.send_nostr_event(test_nostr_event).await.unwrap();
    
    // Wait a bit
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // Trigger shutdown
    shutdown_signal.store(true, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_loop_creation() {
        let config = NodeLoopConfig {
            max_batch_size: 100,
            input_timeout: Duration::from_millis(500),
            file_watch_interval: Duration::from_secs(1),
            nostr_event_timeout: Duration::from_secs(30),
            response_queue_size: 100,
        };

        let node_loop = NodeLoop::new(config);
        assert_eq!(node_loop.config.max_batch_size, 100);
    }

    #[test]
    fn test_node_loop_config() {
        let config = NodeLoopConfig {
            max_batch_size: 50,
            input_timeout: Duration::from_millis(200),
            file_watch_interval: Duration::from_secs(2),
            nostr_event_timeout: Duration::from_secs(10),
            response_queue_size: 50,
        };

        assert_eq!(config.max_batch_size, 50);
        assert_eq!(config.input_timeout, Duration::from_millis(200));
    }

    #[test]
    fn test_user_input_enum() {
        let text_input = UserInput::Text("test".to_string());
        let file_input = UserInput::File(std::path::PathBuf::from("test.txt"));
        let command = UserInput::Command(NodeCommand::Status);
        
        match text_input {
            UserInput::Text(t) => assert_eq!(t, "test"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_node_command_enum() {
        let cmd = NodeCommand::Bootstrap;
        match cmd {
            NodeCommand::Bootstrap => {}
            _ => panic!("Expected Bootstrap variant"),
        }
    }

    #[test]
    fn test_file_event_enum() {
        let event = FileEvent::Created(std::path::PathBuf::from("test.txt"));
        match event {
            FileEvent::Created(path) => {
                assert_eq!(path, std::path::PathBuf::from("test.txt"));
            }
            _ => panic!("Expected Created variant"),
        }
    }

    #[test]
    fn test_node_response_enum() {
        let response = NodeResponse::Text("test".to_string());
        match response {
            NodeResponse::Text(t) => assert_eq!(t, "test"),
            _ => panic!("Expected Text variant"),
        }
    }

    #[test]
    fn test_node_status_struct() {
        let status = NodeStatus {
            free_energy: 0.0,
            surprise_score: 0.0,
            trust_level: 0.0,
            zap_balance: 0,
            connected_relays: 0,
            processed_events: 0,
            learned_patterns: 0,
        };
        
        assert_eq!(status.free_energy, 0.0);
        assert_eq!(status.zap_balance, 0);
    }
}

// Example usage
#[cfg(test)]
pub fn example_usage() {
    // Create node loop
    let config = NodeLoopConfig {
        max_batch_size: 100,
        input_timeout: Duration::from_millis(500),
        file_watch_interval: Duration::from_secs(1),
        nostr_event_timeout: Duration::from_secs(30),
        response_queue_size: 100,
    };

    let mut node_loop = NodeLoop::new(config);
    
    // Create shutdown signal
    let shutdown_signal = Arc::new(AtomicBool::new(false));
    
    // Start node loop
    tokio::spawn(async move {
        if let Err(e) = node_loop.start(shutdown_signal.clone()).await {
            error!("Node loop failed: {}", e);
        }
    });
    
    // Send some test events
    let test_input = UserInput::Text("Hello, this is a test input".to_string());
    let test_file_event = FileEvent::Created(std::path::PathBuf::from("test.txt"));
    let test_nostr_event = NostrEvent {
        kind: "PCErrorDelta".to_string(),
        content: "Test delta content".to_string(),
        pubkey: "npub1test".to_string(),
        timestamp: 1234567890,
    };
    
    // Send events (in real code, this would be done by other components)
    node_loop.send_user_input(test_input).await.unwrap();
    node_loop.send_file_event(test_file_event).await.unwrap();
    node_loop.send_nostr_event(test_nostr_event).await.unwrap();
    
    // Wait a bit
    tokio::time::sleep(Duration::from_secs(1)).await;
    
    // Trigger shutdown
    shutdown_signal.store(true, Ordering::Relaxed);
}