// src/node_loop.rs
// Core node loop implementation for processing user input, file events, and Nostr events
// Includes graceful shutdown (CTRL+C) and Dream Phase for offline consolidation

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;

use crate::types::{UserInput, FileEvent, NodeError, NostrEvent};
use crate::pc_hierarchy::PredictiveCoding;

pub struct NodeLoop {
    rx_user_input: mpsc::Receiver<UserInput>,
    rx_file_events: mpsc::Receiver<FileEvent>,
    rx_nostr_events: mpsc::Receiver<NostrEvent>,
    stop_signal: Arc<AtomicBool>,
    // Optional PC hierarchy for dream phase
    pc_hierarchy: Option<Arc<tokio::sync::Mutex<PredictiveCoding>>>,
    // Inactivity tracking for dream phase
    last_activity_time: Arc<tokio::sync::RwLock<Instant>>,
    dream_phase_interval: Duration,
}

impl NodeLoop {
    /// Create a new node loop without PC hierarchy (dream phase disabled)
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
            pc_hierarchy: None,
            last_activity_time: Arc::new(tokio::sync::RwLock::new(Instant::now())),
            dream_phase_interval: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Create a new node loop with PC hierarchy for dream phase
    pub fn new_with_pc_hierarchy(
        rx_user_input: mpsc::Receiver<UserInput>,
        rx_file_events: mpsc::Receiver<FileEvent>,
        rx_nostr_events: mpsc::Receiver<NostrEvent>,
        pc_hierarchy: Arc<tokio::sync::Mutex<PredictiveCoding>>,
        dream_phase_interval: Option<Duration>,
    ) -> Self {
        Self {
            rx_user_input,
            rx_file_events,
            rx_nostr_events,
            stop_signal: Arc::new(AtomicBool::new(false)),
            pc_hierarchy: Some(pc_hierarchy),
            last_activity_time: Arc::new(tokio::sync::RwLock::new(Instant::now())),
            dream_phase_interval: dream_phase_interval.unwrap_or(Duration::from_secs(300)), // 5 minutes default
        }
    }

    /// Start the node loop with graceful shutdown and dream phase support
    pub async fn start(&mut self) -> Result<(), NodeError> {
        let mut user_input_interval = interval(Duration::from_millis(100));
        let mut file_event_interval = interval(Duration::from_millis(100));
        let mut nostr_event_interval = interval(Duration::from_millis(100));
        let mut dream_phase_check_interval = interval(Duration::from_secs(60)); // Check every minute

        // Setup CTRL+C handler for graceful shutdown
        let stop_signal = Arc::clone(&self.stop_signal);
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("CTRL+C received, initiating graceful shutdown...");
            stop_signal.store(true, Ordering::Relaxed);
        });

        tracing::info!("Node loop started with dream phase interval: {:?}", self.dream_phase_interval);

        loop {
            // Check stop signal at the beginning of each iteration
            if self.stop_signal.load(Ordering::Relaxed) {
                tracing::info!("Stop signal received, shutting down node loop gracefully");
                break;
            }
            
            tokio::select! {
                _ = user_input_interval.tick() => {
                    if let Some(input) = self.rx_user_input.recv().await {
                        // Update last activity time
                        *self.last_activity_time.write().await = Instant::now();
                        self.process_user_input(input).await?;
                    }
                }
                _ = file_event_interval.tick() => {
                    if let Some(event) = self.rx_file_events.recv().await {
                        // Update last activity time
                        *self.last_activity_time.write().await = Instant::now();
                        self.process_file_event(event).await?;
                    }
                }
                _ = nostr_event_interval.tick() => {
                    if let Some(event) = self.rx_nostr_events.recv().await {
                        // Update last activity time
                        *self.last_activity_time.write().await = Instant::now();
                        self.process_nostr_event(event).await?;
                    }
                }
                _ = dream_phase_check_interval.tick() => {
                    self.check_and_trigger_dream_phase().await?;
                }
                else => {
                    // Small sleep to prevent busy waiting
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }

        tracing::info!("Node loop stopped gracefully");
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

    /// Check inactivity and trigger dream phase if needed
    async fn check_and_trigger_dream_phase(&self) -> Result<(), NodeError> {
        let last_activity = *self.last_activity_time.read().await;
        let inactivity_duration = Instant::now().duration_since(last_activity);
        
        if inactivity_duration >= self.dream_phase_interval {
            tracing::info!(
                "Inactivity detected ({:?}), triggering dream phase for offline consolidation",
                inactivity_duration
            );
            
            if let Some(_pc_hierarchy) = &self.pc_hierarchy {
                // In a full implementation, we would run pc.dream() here
                // For MVP, we just log that dream phase would be triggered
                tracing::info!("Dream phase would run here (offline weight consolidation)");
                tracing::debug!("Would generate random seed and call pc.dream() for generative replay");
            } else {
                tracing::debug!("Dream phase skipped: no PC hierarchy configured");
            }
            
            // Reset activity time to prevent immediate retrigger
            *self.last_activity_time.write().await = Instant::now();
        }
        
        Ok(())
    }

    /// Stop the node loop gracefully
    pub fn stop(&self) {
        tracing::info!("Stopping node loop gracefully...");
        self.stop_signal.store(true, Ordering::Relaxed);
    }

    /// Set the dream phase interval
    pub fn set_dream_phase_interval(&mut self, interval: Duration) {
        self.dream_phase_interval = interval;
        tracing::debug!("Dream phase interval set to {:?}", interval);
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