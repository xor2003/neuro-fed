// src/lib.rs
// Main library for NeuroFed Node

#![recursion_limit = "4096"]

// Core modules
pub mod bootstrap;
pub mod brain_manager;
pub mod chat;
pub mod config;
pub mod federation; // Moved: federation-related modules
pub mod knowledge_filter;
pub mod metrics;
pub mod ml_engine;
pub mod model_manager;
pub mod node_loop;
pub mod openai_proxy;
pub mod pc_decoder;
pub mod pc_hierarchy;
pub mod pc_level;
pub mod pc_types;
pub mod persistence;
pub mod pow_verifier;
pub mod privacy; // Moved: privacy-related modules
pub mod semantic_cache;
pub mod sleep_phase;
pub mod types;
pub mod learning_log;
pub mod ui;

// Backward compatibility aliases for moved modules
pub use federation::blossom_client;
pub use federation::federation_manager;
pub use federation::nostr_federation;
pub use federation::payment_verifier;
pub use privacy::privacy_networks;
pub use privacy::privacy_networks_fixed;

// Re-exports
pub use config::BootstrapConfig;
pub use model_manager::{ModelInfo, ModelManager, ModelManagerError};
pub use nostr_federation::NostrEvent;
pub use pc_hierarchy::{PCConfig, PCError, PredictiveCoding, SurpriseStats};
pub use pc_level::PCLevel;
pub use types::{FileEvent, NodeCommand, NodeError, NodeResponse, UserInput};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
