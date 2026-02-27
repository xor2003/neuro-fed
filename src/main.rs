// src/main.rs
// Main application entry point

use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{info, error, debug, warn, Level};
use tracing_subscriber;
use crate::bootstrap::DistillationMethod;

mod llama_ffi;
mod pc_hierarchy;
mod bootstrap;

use llama_ffi::{LlamaContext, LlamaError};
use pc_hierarchy::{PredictiveCoding, PCConfig, PCError};
use bootstrap::{Bootstrap, BootstrapConfig, BootstrapError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize logging
    setup_logging();
    
    // Create components
    let llama_ctx = LlamaContext::new("models/gguf_model.gguf", 2048).await?;
    let pc_hierarchy = PredictiveCoding::new(PCConfig::new(3, vec![2048, 1024, 512]))?;
    let bootstrap = Bootstrap::new(BootstrapConfig::new("models/gguf_model.gguf".to_string(), 2048, vec!["./data".to_string()], None, 3, 100, 2048, DistillationMethod::LinearProjection, "./bootstrap_results".to_string()));
    
    info!("NeuroPC Node initialized successfully");
    
    // Simple demo loop
    let mut counter = 0;
    let mut interval = interval(Duration::from_millis(1000));
    
    loop {
        interval.tick().await;
        counter += 1;
        
        if counter % 5 == 0 {
            info!("Running inference...");
            let input = ndarray::Array2::ones((1, 2048));
            let _result = pc_hierarchy.infer(&input, 10);
        }
        
        if counter == 10 {
            info!("Starting bootstrap...");
            let _bootstrap_result = bootstrap.run().await;
        }
        
        if counter >= 20 {
            info!("Demo complete, shutting down...");
            break;
        }
    }
    
    Ok(())
}

fn setup_logging() {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set global default subscriber");
}