// src/pc_types.rs
// PC-specific types and error definitions
// NOTE: PCConfig is now imported from config.rs (the single source of truth)

use std::error::Error;
use std::fmt;

// 🔴 Pull PCConfig securely from the single source of truth
pub use crate::config::PCConfig;

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

/// Upgraded stats: Explicit uncertainty and novelty tracking
#[derive(Debug, Clone, Default)]
pub struct SurpriseStats {
    pub total_surprise: f32,
    pub level_surprises: Vec<f32>, // Track which level is struggling
    pub free_energy_history: Vec<f32>,
    pub high_surprise_indices: Vec<usize>,

    // AI Architecture Improvements: Explicit Latent Uncertainty
    /// Novelty: How unexpected was this input initially? (Initial Free Energy)
    pub novelty_score: f32,
    /// Confidence: How stable did the belief become? (Inverse of final variance/energy)
    pub confidence_score: f32,
}
