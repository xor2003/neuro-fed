// src/lib.rs
// Main library for NeuroFed Node

#![recursion_limit = "4096"]

// Core modules
pub mod bootstrap;
pub mod config;
pub mod federation;  // Moved: federation-related modules
pub mod privacy;     // Moved: privacy-related modules
pub mod knowledge_filter;
pub mod ml_engine;
pub mod model_manager;
pub mod node_loop;
pub mod openai_proxy;
pub mod brain_manager;
pub mod pc_hierarchy;
pub mod pc_types;
pub mod pc_level;
pub mod pc_decoder;
pub mod persistence;
pub mod semantic_cache;
pub mod sleep_phase;
pub mod types;
pub mod pow_verifier;

// Backward compatibility aliases for moved modules
pub use federation::blossom_client as blossom_client;
pub use federation::federation_manager as federation_manager;
pub use federation::nostr_federation as nostr_federation;
pub use federation::payment_verifier as payment_verifier;
pub use privacy::privacy_networks as privacy_networks;
pub use privacy::privacy_networks_fixed as privacy_networks_fixed;

// Re-exports
pub use config::BootstrapConfig;
pub use pc_hierarchy::{PredictiveCoding, PCConfig, PCError, SurpriseStats};
pub use model_manager::{ModelManager, ModelManagerError, ModelInfo};
pub use types::{UserInput, NodeCommand, FileEvent, NodeResponse, NodeError};
pub use pc_level::PCLevel;
pub use nostr_federation::NostrEvent;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
