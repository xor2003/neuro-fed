// src/lib.rs
// Main library for NeuroFed Node

#![recursion_limit = "4096"]

pub mod bootstrap;
pub mod config;
pub mod federation_manager;
pub mod knowledge_filter;
pub mod llama_ffi;
pub mod ml_engine;
pub mod model_manager;
pub mod node_loop;
pub mod nostr_federation;
pub mod openai_proxy;
pub mod blossom_client;
pub mod brain_manager;
pub mod pc_hierarchy;
pub mod persistence;
pub mod semantic_cache;
pub mod types;
pub mod payment_verifier;
pub mod pow_verifier;
pub mod privacy_networks;

pub use bootstrap::Bootstrap;
pub use config::BootstrapConfig;
pub use pc_hierarchy::PredictiveCoding;
pub use pc_hierarchy::PCConfig;
pub use llama_ffi::LlamaModel;
pub use model_manager::ModelManager;
pub use model_manager::ModelManagerError;
pub use model_manager::ModelInfo;
pub use bootstrap::BootstrapError;
pub use pc_hierarchy::PCError;
pub use types::{UserInput, NodeCommand, FileEvent, NodeResponse, NodeError, NostrEvent};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}