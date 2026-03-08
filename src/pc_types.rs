// src/pc_types.rs
use std::error::Error;
use std::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct PCError(pub String);

impl fmt::Display for PCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PCError: {}", self.0)
    }
}

impl Error for PCError {}

impl From<candle_core::Error> for PCError {
    fn from(err: candle_core::Error) -> Self {
        PCError(format!("Candle error: {}", err))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PCConfig {
    pub n_levels: usize,
    pub dim_per_level: Vec<usize>,
    pub learning_rate: f32,
    pub inference_steps: usize,
    pub surprise_threshold: f32,
    pub convergence_threshold: f32,
    pub selective_update: bool,
    pub mu_pc_scaling: bool,
    pub enable_precision_weighting: bool,
    pub free_energy_drop_threshold: f32,
    pub default_precision: f32,
    pub min_precision: f32,
    pub max_precision: f32,
    pub free_energy_history_size: usize,
    pub enable_code_verification: bool,
    pub enable_nostr_zap_tracking: bool,
    pub min_zaps_for_consensus: usize,
    pub persistence_db_path: Option<String>,
    pub hidden_dim_factor: f32,
}

impl PCConfig {
    pub fn new(n_levels: usize, dim_per_level: Vec<usize>) -> Self {
        PCConfig {
            n_levels,
            dim_per_level,
            learning_rate: 0.01,
            inference_steps: 20,
            surprise_threshold: 1.0,
            convergence_threshold: 0.01,
            selective_update: true,
            mu_pc_scaling: true,
            enable_precision_weighting: false,
            free_energy_drop_threshold: 0.5,
            default_precision: 0.3,
            min_precision: 0.1,
            max_precision: 1.0,
            free_energy_history_size: 10,
            enable_code_verification: false,
            enable_nostr_zap_tracking: false,
            min_zaps_for_consensus: 3,
            persistence_db_path: None,
            hidden_dim_factor: 0.5,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SurpriseStats {
    pub total_surprise: f32,
    pub free_energy_history: Vec<f32>,
    pub high_surprise_indices: Vec<usize>,
}
