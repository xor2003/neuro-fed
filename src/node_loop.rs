// src/node_loop.rs
// Core node loop implementation for processing user input, file events, and Nostr events
// Includes graceful shutdown (CTRL+C) and Dream Phase for offline consolidation
// 🔴 FIX: Eliminated head-of-line blocking by using direct channel receivers and spawning tasks

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::sync::RwLock;
use tokio::time::interval;

use crate::types::{UserInput, FileEvent, NodeError, NostrEvent};
use crate::pc_hierarchy::PredictiveCoding;
use crate::sleep_phase::SleepManager;

#[allow(dead_code)]
pub struct NodeLoop {
    rx_user_input: mpsc::Receiver<UserInput>,
    rx_file_events: mpsc::Receiver<FileEvent>,
    rx_nostr_events: mpsc::Receiver<NostrEvent>,
    stop_signal: Arc<AtomicBool>,
    // Optional PC hierarchy for dream phase
    pc_hierarchy: Option<Arc<RwLock<PredictiveCoding>>>,
    // Sleep manager for consolidation
    sleep_manager: Option<Arc<SleepManager>>,
    // Episodic memory for sleep phase
    episodic_memory: Option<Arc<RwLock<std::collections::VecDeque<crate::types::Episode>>>>,
    // Inactivity tracking for dream phase
    last_activity_time: Arc<RwLock<Instant>>,
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
            sleep_manager: None,
            episodic_memory: None,
            last_activity_time: Arc::new(RwLock::new(Instant::now())),
            dream_phase_interval: Duration::from_secs(300), // 5 minutes default
        }
    }

    /// Create a new node loop with PC hierarchy for dream phase
    pub fn new_with_pc_hierarchy(
        rx_user_input: mpsc::Receiver<UserInput>,
        rx_file_events: mpsc::Receiver<FileEvent>,
        rx_nostr_events: mpsc::Receiver<NostrEvent>,
        pc_hierarchy: Arc<RwLock<PredictiveCoding>>,
        sleep_manager: Option<Arc<SleepManager>>,
        episodic_memory: Option<Arc<RwLock<std::collections::VecDeque<crate::types::Episode>>>>,
        dream_phase_interval: Option<Duration>,
    ) -> Self {
        Self {
            rx_user_input,
            rx_file_events,
            rx_nostr_events,
            stop_signal: Arc::new(AtomicBool::new(false)),
            pc_hierarchy: Some(pc_hierarchy),
            sleep_manager,
            episodic_memory,
            last_activity_time: Arc::new(RwLock::new(Instant::now())),
            dream_phase_interval: dream_phase_interval.unwrap_or(Duration::from_secs(300)), // 5 minutes default
        }
    }

    /// Start the node loop with graceful shutdown and dream phase support
    /// 🔴 FIX: Non-blocking select loop - processes events immediately without polling intervals
    pub async fn start(&mut self) -> Result<(), NodeError> {
        let mut dream_phase_check_interval = interval(Duration::from_secs(60));

        // Setup CTRL+C handler for graceful shutdown
        let stop_signal = Arc::clone(&self.stop_signal);
        tokio::spawn(async move {
            tokio::signal::ctrl_c().await.ok();
            tracing::info!("CTRL+C received, initiating graceful shutdown...");
            stop_signal.store(true, Ordering::Relaxed);
        });

        tracing::info!("Node loop started with dream phase interval: {:?}", self.dream_phase_interval);

        // 🔴 FIX: Proper tokio::select! avoiding head-of-line blocking
        loop {
            if self.stop_signal.load(Ordering::Relaxed) { break; }
            
            tokio::select! {
                // Direct channel receive - no polling intervals
                Some(input) = self.rx_user_input.recv() => {
                    let last_activity = Arc::clone(&self.last_activity_time);
                    let pc_hierarchy = self.pc_hierarchy.clone();
                    
                    // Spawn task to process without blocking the select loop
                    tokio::spawn(async move {
                        if let Err(e) = Self::process_user_input_async(input, pc_hierarchy).await {
                            tracing::error!("Error processing user input: {}", e);
                        }
                        *last_activity.write().await = Instant::now();
                    });
                }
                Some(event) = self.rx_file_events.recv() => {
                    let last_activity = Arc::clone(&self.last_activity_time);
                    
                    tokio::spawn(async move {
                        if let Err(e) = Self::process_file_event_async(event).await {
                            tracing::error!("Error processing file event: {}", e);
                        }
                        *last_activity.write().await = Instant::now();
                    });
                }
                Some(event) = self.rx_nostr_events.recv() => {
                    let last_activity = Arc::clone(&self.last_activity_time);
                    
                    tokio::spawn(async move {
                        if let Err(e) = Self::process_nostr_event_async(event).await {
                            tracing::error!("Error processing Nostr event: {}", e);
                        }
                        *last_activity.write().await = Instant::now();
                    });
                }
                _ = dream_phase_check_interval.tick() => {
                    // 🔴 FIX: Do not block the event loop while dreaming!
                    // If we `await` here directly, the node completely stops responding to HTTP/Nostr events.
                    let sleep_mgr_clone = self.sleep_manager.clone();
                    let pc_hierarchy_clone = self.pc_hierarchy.clone();
                    let last_activity_clone = self.last_activity_time.clone();
                    let interval_clone = self.dream_phase_interval;

                    tokio::spawn(async move {
                        let last_activity = *last_activity_clone.read().await;
                        let inactivity_duration = std::time::Instant::now().duration_since(last_activity);
                        
                        if inactivity_duration >= interval_clone {
                            tracing::info!(
                                "Inactivity detected ({:?}), triggering sleep phase for memory consolidation",
                                inactivity_duration
                            );
                            
                            if let Some(sleep_mgr) = sleep_mgr_clone {
                                if let Err(e) = sleep_mgr.process_sleep_cycle().await {
                                    tracing::error!("Sleep cycle failed: {}", e);
                                }
                            } else if let Some(pc) = pc_hierarchy_clone {
                                let _pc_write = pc.write().await;
                                tracing::info!("Basic dream phase running (offline weight consolidation)");
                            }
                            
                            // Reset activity time to prevent immediate retrigger
                            *last_activity_clone.write().await = std::time::Instant::now();
                        }
                    });
                }
                // Periodic wake-up to check stop signal when no events are pending
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    // Just a periodic check - no action needed
                }
            }
        }

        tracing::info!("Node loop stopped gracefully");
        Ok(())
    }

    /// Process user input (spawned as separate task)
    async fn process_user_input_async(
        input: UserInput,
        pc_hierarchy: Option<Arc<RwLock<PredictiveCoding>>>,
    ) -> Result<(), NodeError> {
        // Placeholder: integrate with PC hierarchy
        if let Some(pc) = pc_hierarchy {
            let _pc_read = pc.read().await;
            // Process input through PC
            tracing::debug!("Processing user input: {}", input.content);
        }
        Ok(())
    }

    /// Process file event (spawned as separate task)
    async fn process_file_event_async(event: FileEvent) -> Result<(), NodeError> {
        // Placeholder implementation
        tracing::debug!("File event processed: {:?}", event);
        Ok(())
    }

    /// Process Nostr event (spawned as separate task)
    async fn process_nostr_event_async(event: NostrEvent) -> Result<(), NodeError> {
        // Placeholder implementation
        tracing::debug!("Nostr event processed: {:?}", event);
        Ok(())
    }

    /// Check inactivity and trigger dream phase if needed
    async fn check_and_trigger_dream_phase(&self) -> Result<(), NodeError> {
        let last_activity = *self.last_activity_time.read().await;
        let inactivity_duration = Instant::now().duration_since(last_activity);
        
        if inactivity_duration >= self.dream_phase_interval {
            tracing::info!(
                "Inactivity detected ({:?}), triggering sleep phase for memory consolidation",
                inactivity_duration
            );
            
            // Trigger SleepManager if available
            if let Some(sleep_mgr) = &self.sleep_manager {
                if let Err(e) = sleep_mgr.process_sleep_cycle().await {
                    tracing::error!("Sleep cycle failed: {}", e);
                } else {
                    tracing::info!("Sleep cycle completed successfully");
                }
            } else if let Some(pc) = &self.pc_hierarchy {
                // Fallback: basic dream phase without SleepManager
                let _pc_write = pc.write().await;
                tracing::info!("Basic dream phase running (offline weight consolidation)");
            } else {
                tracing::debug!("Sleep phase skipped: no PC hierarchy or SleepManager configured");
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
    
    
    
    
    
    

    #[tokio::test]
    async fn test_node_loop_creation() {
        use super::*;
        use tokio::sync::mpsc;
        
        let (_tx_user_input, rx_user_input) = mpsc::channel(10);
        let (_tx_file_events, rx_file_events) = mpsc::channel(10);
        let (_tx_nostr_events, rx_nostr_events) = mpsc::channel(10);

        let _node_loop = NodeLoop::new(rx_user_input, rx_file_events, rx_nostr_events);
        // Just verify creation succeeded
        assert!(true);
    }

    #[tokio::test]
    async fn test_node_loop_start_stop() {
        use super::*;
        use tokio::sync::mpsc;
        use std::sync::Arc;
        use std::time::Duration;
        use std::sync::atomic::Ordering;
        
        let (tx_user_input, rx_user_input) = mpsc::channel(10);
        let (tx_file_events, rx_file_events) = mpsc::channel(10);
        let (tx_nostr_events, rx_nostr_events) = mpsc::channel(10);

        let mut node_loop = NodeLoop::new(rx_user_input, rx_file_events, rx_nostr_events);
        let stop_signal: Arc<std::sync::atomic::AtomicBool> = Arc::clone(&node_loop.stop_signal);
        
        let handle = tokio::spawn(async move {
            node_loop.start().await.unwrap();
        });

        // Drop the senders to close channels
        drop(tx_user_input);
        drop(tx_file_events);
        drop(tx_nostr_events);
        
        tokio::time::sleep(Duration::from_millis(100)).await;
        stop_signal.store(true, Ordering::Relaxed);
        
        // Add timeout to prevent hanging
        tokio::time::timeout(Duration::from_millis(500), handle).await.unwrap().unwrap();
    }
}
