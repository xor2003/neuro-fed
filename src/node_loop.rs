// src/node_loop.rs
// Core node loop implementation for processing user input, file events, and Nostr events

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;

use crate::types::{UserInput, FileEvent, NodeError, NostrEvent};

pub struct NodeLoop {
    rx_user_input: mpsc::Receiver<UserInput>,
    rx_file_events: mpsc::Receiver<FileEvent>,
    rx_nostr_events: mpsc::Receiver<NostrEvent>,
    stop_signal: Arc<AtomicBool>,
}

impl NodeLoop {
    /// Create a new node loop
    pub fn new(
        rx_user_input: mpsc::Receiver<UserInput>,
        rx_file_events: mpsc::Receiver<FileEvent>,
        rx_nostr_events: mpsc::Receiver<NostrEvent>,
    ) -> Self {
        Self {
            rx_user_input,
            rx_file_events,
            rx_nostr_events,
            stop_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the node loop
    pub async fn start(&mut self) -> Result<(), NodeError> {
        let mut user_input_interval = interval(Duration::from_millis(100));
        let mut file_event_interval = interval(Duration::from_millis(100));
        let mut nostr_event_interval = interval(Duration::from_millis(100));

        loop {
            // Check stop signal at the beginning of each iteration
            if self.stop_signal.load(Ordering::Relaxed) {
                break;
            }
            
            tokio::select! {
                _ = user_input_interval.tick() => {
                    if let Some(input) = self.rx_user_input.recv().await {
                        self.process_user_input(input).await?;
                    }
                }
                _ = file_event_interval.tick() => {
                    if let Some(event) = self.rx_file_events.recv().await {
                        self.process_file_event(event).await?;
                    }
                }
                _ = nostr_event_interval.tick() => {
                    if let Some(event) = self.rx_nostr_events.recv().await {
                        self.process_nostr_event(event).await?;
                    }
                }
                else => {
                    // Small sleep to prevent busy waiting
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }

        Ok(())
    }

    /// Process user input
    async fn process_user_input(&self, _input: UserInput) -> Result<(), NodeError> {
        // Implementation here
        Ok(())
    }

    /// Process file event
    async fn process_file_event(&self, _event: FileEvent) -> Result<(), NodeError> {
        // Implementation here
        Ok(())
    }

    /// Process Nostr event
    async fn process_nostr_event(&self, _event: NostrEvent) -> Result<(), NodeError> {
        // Implementation here
        Ok(())
    }

    /// Stop the node loop
    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::Relaxed);
    }
}

mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::mpsc;
    use tokio::time;

    #[tokio::test]
    async fn test_node_loop_creation() {
        let (tx_user_input, rx_user_input) = mpsc::channel(10);
        let (tx_file_events, rx_file_events) = mpsc::channel(10);
        let (tx_nostr_events, rx_nostr_events) = mpsc::channel(10);

        let node_loop = NodeLoop::new(rx_user_input, rx_file_events, rx_nostr_events);
        // Just verify creation succeeded
        assert!(true);
    }

    #[tokio::test]
    async fn test_node_loop_start_stop() {
        let (tx_user_input, rx_user_input) = mpsc::channel(10);
        let (tx_file_events, rx_file_events) = mpsc::channel(10);
        let (tx_nostr_events, rx_nostr_events) = mpsc::channel(10);

        let mut node_loop = NodeLoop::new(rx_user_input, rx_file_events, rx_nostr_events);
        let stop_signal = Arc::clone(&node_loop.stop_signal);
        
        let handle = tokio::spawn(async move {
            node_loop.start().await.unwrap();
        });

        // Drop the senders to close channels
        drop(tx_user_input);
        drop(tx_file_events);
        drop(tx_nostr_events);
        
        time::sleep(Duration::from_millis(100)).await;
        stop_signal.store(true, Ordering::Relaxed);
        
        // Add timeout to prevent hanging
        time::timeout(Duration::from_millis(500), handle).await.unwrap().unwrap();
    }
}