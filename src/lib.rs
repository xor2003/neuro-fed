// src/lib.rs
// Main library for NeuroFed Node

#![recursion_limit = "4096"]

pub mod bootstrap;
pub mod config;
pub mod llama_ffi;
pub mod ml_engine;
pub mod node_loop;
pub mod nostr_federation;
pub mod openai_proxy;
pub mod pc_hierarchy;
pub mod types;

pub use bootstrap::Bootstrap;
pub use bootstrap::BootstrapConfig;
pub use pc_hierarchy::PredictiveCoding;
pub use pc_hierarchy::PCConfig;
pub use llama_ffi::LlamaModel;
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