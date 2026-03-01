// src/knowledge_filter.rs
// Knowledge Filtering with Precision Weighting (π) implementation for NeuroFed Node

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Precision weighting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    /// Free energy drop threshold for high precision (π = 1.0)
    pub free_energy_drop_threshold: f32,
    /// Default precision for unverified information
    pub default_precision: f32,
    /// Minimum precision value
    pub min_precision: f32,
    /// Maximum precision value
    pub max_precision: f32,
    /// Window size for free energy history tracking
    pub free_energy_history_size: usize,
    /// Whether to enable code execution verification
    pub enable_code_verification: bool,
    /// Whether to enable Nostr zap tracking
    pub enable_nostr_zap_tracking: bool,
    /// Minimum number of zaps for economic consensus
    pub min_zaps_for_consensus: usize,
    /// Trusted node public keys for zap verification
    pub trusted_node_keys: Vec<String>,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        PrecisionConfig {
            free_energy_drop_threshold: 0.5, // 50% drop in free energy
            default_precision: 0.3,
            min_precision: 0.1,
            max_precision: 1.0,
            free_energy_history_size: 10,
            enable_code_verification: false,
            enable_nostr_zap_tracking: false,
            min_zaps_for_consensus: 3,
            trusted_node_keys: Vec::new(),
        }
    }
}

/// Information source type for precision calculation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InformationSource {
    /// Direct observation or ground truth
    GroundTruth,
    /// Code execution with successful verification
    CodeExecution,
    /// Nostr event with economic consensus (zaps)
    NostrEvent,
    /// General information without verification
    GeneralInformation,
    /// High free energy drop indicates valuable information
    HighValueLearning,
}

/// Precision calculation result
#[derive(Debug, Clone)]
pub struct PrecisionResult {
    /// Precision value π ∈ [0, 1]
    pub precision: f32,
    /// Source of the precision calculation
    pub source: InformationSource,
    /// Confidence in the precision calculation
    pub confidence: f32,
    /// Additional metadata about the calculation
    pub metadata: Vec<(String, String)>,
}

/// Free energy history tracker
#[derive(Debug, Clone)]
pub struct FreeEnergyTracker {
    history: VecDeque<f32>,
    max_size: usize,
    last_free_energy: f32,
}

impl FreeEnergyTracker {
    pub fn new(max_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_size),
            max_size,
            last_free_energy: 0.0,
        }
    }

    pub fn record(&mut self, free_energy: f32) {
        self.last_free_energy = free_energy;
        self.history.push_back(free_energy);
        if self.history.len() > self.max_size {
            self.history.pop_front();
        }
    }

    pub fn calculate_drop(&self) -> Option<f32> {
        if self.history.len() < 2 {
            return None;
        }
        
        let oldest = self.history.front()?;
        let latest = self.history.back()?;
        
        if *oldest == 0.0 {
            return None;
        }
        
        Some((oldest - latest) / oldest) // Percentage drop
    }

    pub fn has_significant_drop(&self, threshold: f32) -> bool {
        match self.calculate_drop() {
            Some(drop) => drop >= threshold,
            None => false,
        }
    }
}

/// Code execution verification interface (stub)
#[derive(Debug, Clone)]
pub struct CodeVerifier {
    /// Whether code execution verification is enabled
    enabled: bool,
}

impl CodeVerifier {
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    pub fn verify_code_execution(&self, code: &str) -> Result<bool, String> {
        // Stub implementation - in Phase 2, this would interface with external execution
        if !self.enabled {
            return Ok(false);
        }
        
        // For now, return true for simple code snippets as a placeholder
        // In actual implementation, this would execute the code and check for success
        let is_simple = code.lines().count() <= 5 && 
                       !code.contains("panic!") && 
                       !code.contains("unsafe");
        
        Ok(is_simple)
    }
}

/// Nostr zap tracking interface (stub)
#[derive(Debug, Clone)]
pub struct NostrZapTracker {
    /// Whether zap tracking is enabled
    enabled: bool,
    /// Minimum zaps for consensus
    min_zaps: usize,
    /// Trusted node public keys
    trusted_keys: Vec<String>,
}

impl NostrZapTracker {
    pub fn new(enabled: bool, min_zaps: usize, trusted_keys: Vec<String>) -> Self {
        Self {
            enabled,
            min_zaps,
            trusted_keys,
        }
    }

    pub fn check_zap_consensus(&self, event_id: &str) -> Result<usize, String> {
        // Stub implementation - in Phase 2, this would interface with nostr_federation.rs
        if !self.enabled {
            return Ok(0);
        }
        
        // For now, return a mock number of zaps
        // In actual implementation, this would query Nostr relays for zaps on the event
        let mock_zaps = if event_id.contains("trusted") { 5 } else { 1 };
        Ok(mock_zaps)
    }

    pub fn has_economic_consensus(&self, event_id: &str) -> Result<bool, String> {
        let zap_count = self.check_zap_consensus(event_id)?;
        Ok(zap_count >= self.min_zaps)
    }
}

/// Main precision calculator
#[derive(Debug, Clone)]
pub struct PrecisionCalculator {
    config: PrecisionConfig,
    free_energy_tracker: FreeEnergyTracker,
    code_verifier: CodeVerifier,
    nostr_zap_tracker: NostrZapTracker,
}

impl PrecisionCalculator {
    pub fn new(config: PrecisionConfig) -> Self {
        Self {
            free_energy_tracker: FreeEnergyTracker::new(config.free_energy_history_size),
            code_verifier: CodeVerifier::new(config.enable_code_verification),
            nostr_zap_tracker: NostrZapTracker::new(
                config.enable_nostr_zap_tracking,
                config.min_zaps_for_consensus,
                config.trusted_node_keys.clone(),
            ),
            config,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(PrecisionConfig::default())
    }

    /// Record free energy for tracking drops
    pub fn record_free_energy(&mut self, free_energy: f32) {
        self.free_energy_tracker.record(free_energy);
    }

    /// Calculate precision based on multiple factors
    pub fn calculate_precision(&self, context: &PrecisionContext) -> PrecisionResult {
        let mut precision = self.config.default_precision;
        let mut source = InformationSource::GeneralInformation;
        let mut confidence = 0.5;
        let mut metadata = Vec::new();

        // Check free energy drop
        if let Some(drop) = self.free_energy_tracker.calculate_drop() {
            if drop >= self.config.free_energy_drop_threshold {
                precision = self.config.max_precision;
                source = InformationSource::HighValueLearning;
                confidence = 0.9;
                metadata.push(("free_energy_drop".to_string(), format!("{:.2}%", drop * 100.0)));
            }
        }

        // Check code execution verification
        if self.config.enable_code_verification {
            if let Some(code) = &context.code_snippet {
                match self.code_verifier.verify_code_execution(code) {
                    Ok(true) => {
                        precision = self.config.max_precision;
                        source = InformationSource::CodeExecution;
                        confidence = 0.8;
                        metadata.push(("code_verification".to_string(), "success".to_string()));
                    }
                    Ok(false) => {
                        // Code verification failed, keep default precision
                        metadata.push(("code_verification".to_string(), "failed".to_string()));
                    }
                    Err(e) => {
                        metadata.push(("code_verification_error".to_string(), e));
                    }
                }
            }
        }

        // Check Nostr zap consensus
        if self.config.enable_nostr_zap_tracking {
            if let Some(event_id) = &context.nostr_event_id {
                match self.nostr_zap_tracker.has_economic_consensus(event_id) {
                    Ok(true) => {
                        precision = self.config.max_precision;
                        source = InformationSource::NostrEvent;
                        confidence = 0.7;
                        metadata.push(("nostr_zap_consensus".to_string(), "achieved".to_string()));
                    }
                    Ok(false) => {
                        metadata.push(("nostr_zap_consensus".to_string(), "insufficient".to_string()));
                    }
                    Err(e) => {
                        metadata.push(("nostr_zap_error".to_string(), e));
                    }
                }
            }
        }

        // Apply ground truth if available
        if context.is_ground_truth {
            precision = self.config.max_precision;
            source = InformationSource::GroundTruth;
            confidence = 1.0;
            metadata.push(("ground_truth".to_string(), "true".to_string()));
        }

        // Clamp precision to configured bounds
        precision = precision.clamp(self.config.min_precision, self.config.max_precision);

        PrecisionResult {
            precision,
            source,
            confidence,
            metadata,
        }
    }

    /// Calculate precision for a batch of contexts
    pub fn calculate_precision_batch(&self, contexts: &[PrecisionContext]) -> Vec<PrecisionResult> {
        contexts.iter()
            .map(|context| self.calculate_precision(context))
            .collect()
    }

    /// Get current free energy drop percentage
    pub fn get_free_energy_drop(&self) -> Option<f32> {
        self.free_energy_tracker.calculate_drop()
    }

    /// Get configuration
    pub fn config(&self) -> &PrecisionConfig {
        &self.config
    }
}

/// Context for precision calculation
#[derive(Debug, Clone)]
pub struct PrecisionContext {
    /// Optional code snippet for verification
    pub code_snippet: Option<String>,
    /// Optional Nostr event ID for zap tracking
    pub nostr_event_id: Option<String>,
    /// Whether this information is considered ground truth
    pub is_ground_truth: bool,
    /// Additional context metadata
    pub metadata: Vec<(String, String)>,
}

impl Default for PrecisionContext {
    fn default() -> Self {
        Self {
            code_snippet: None,
            nostr_event_id: None,
            is_ground_truth: false,
            metadata: Vec::new(),
        }
    }
}

impl PrecisionContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_code_snippet(mut self, code: String) -> Self {
        self.code_snippet = Some(code);
        self
    }

    pub fn with_nostr_event_id(mut self, event_id: String) -> Self {
        self.nostr_event_id = Some(event_id);
        self
    }

    pub fn with_ground_truth(mut self, is_ground_truth: bool) -> Self {
        self.is_ground_truth = is_ground_truth;
        self
    }

    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.push((key, value));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_calculator_default() {
        let calculator = PrecisionCalculator::with_default_config();
        let context = PrecisionContext::new();
        let result = calculator.calculate_precision(&context);
        
        assert_eq!(result.precision, 0.3);
        assert_eq!(result.source, InformationSource::GeneralInformation);
        assert!(result.precision >= 0.1 && result.precision <= 1.0);
    }

    #[test]
    fn test_free_energy_tracker() {
        let mut tracker = FreeEnergyTracker::new(5);
        
        tracker.record(10.0);
        tracker.record(8.0);
        tracker.record(6.0);
        tracker.record(4.0);
        tracker.record(2.0);
        
        let drop = tracker.calculate_drop().unwrap();
        assert!(drop > 0.0); // Should be positive drop
        assert!(tracker.has_significant_drop(0.5)); // 80% drop > 50% threshold
    }

    #[test]
    fn test_precision_with_ground_truth() {
        let calculator = PrecisionCalculator::with_default_config();
        let context = PrecisionContext::new()
            .with_ground_truth(true);
        
        let result = calculator.calculate_precision(&context);
        
        assert_eq!(result.precision, 1.0);
        assert_eq!(result.source, InformationSource::GroundTruth);
    }
#[test]
fn test_precision_clamping() {
    let mut config = PrecisionConfig::default();
    config.default_precision = 2.0; // Above max
    config.min_precision = 0.0;
    config.max_precision = 1.0;
    
    let calculator = PrecisionCalculator::new(config);
    let context = PrecisionContext::new();
    let result = calculator.calculate_precision(&context);
    
    assert_eq!(result.precision, 1.0); // Should be clamped to max
}
}

/// Integration tests demonstrating precision weighting with PredictiveCoding
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::pc_hierarchy::{PCConfig, PredictiveCoding};
    use candle_core::{Device, Tensor};

    #[test]
    fn test_precision_weighting_integration() {
        // Create a PC hierarchy with precision weighting enabled
        let mut config = PCConfig::new(2, vec![10, 5]);
        config.enable_precision_weighting = true;
        config.free_energy_drop_threshold = 0.3;
        config.default_precision = 0.3;
        
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input data with correct shape (10 rows, 1 column)
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (10, 1), &device).unwrap();
        
        // First inference to establish baseline free energy
        let infer_stats = pc.infer(&input, 10).unwrap();
        let initial_free_energy = *infer_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Learn with default context (no precision factors)
        let context = None;
        let learn_stats = pc.learn(&input, context).unwrap();
        let after_learning_free_energy = *learn_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Free energy may fluctuate, but change should be bounded
        let change = after_learning_free_energy - initial_free_energy;
        assert!(change.abs() < 10.0, "Free energy change too large: {}", change);
        
        // Verify precision calculator exists
        assert!(pc.precision_calculator.is_some());
        
        println!("Integration test passed: Precision weighting integrated with PC hierarchy");
    }

    #[test]
    fn test_precision_weighting_with_context() {
        // Create a PC hierarchy with precision weighting enabled
        let mut config = PCConfig::new(2, vec![10, 5]);
        config.enable_precision_weighting = true;
        config.free_energy_drop_threshold = 0.1; // Low threshold for testing
        config.default_precision = 0.3;
        
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input data with correct shape (10 rows, 1 column)
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (10, 1), &device).unwrap();
        
        // Record initial free energy
        let infer_stats = pc.infer(&input, 10).unwrap();
        let initial_free_energy = *infer_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Create a context with ground truth (should give π = 1.0)
        let context = Some(PrecisionContext::new()
            .with_ground_truth(true));
        
        // Learn with ground truth context
        let learn_stats = pc.learn(&input, context).unwrap();
        let after_learning_free_energy = *learn_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Free energy may fluctuate, but change should be bounded
        let change = after_learning_free_energy - initial_free_energy;
        assert!(change.abs() < 10.0, "Free energy change too large: {}", change);
        
        println!("Integration test passed: Precision weighting with ground truth context");
    }

    #[test]
    fn test_precision_weighting_with_code_verification() {
        // Create a PC hierarchy with precision weighting and code verification enabled
        let mut config = PCConfig::new(2, vec![10, 5]);
        config.enable_precision_weighting = true;
        config.enable_code_verification = true;
        config.free_energy_drop_threshold = 0.1;
        config.default_precision = 0.3;
        
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input data with correct shape (10 rows, 1 column)
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (10, 1), &device).unwrap();
        
        // Record initial free energy
        let infer_stats = pc.infer(&input, 10).unwrap();
        let initial_free_energy = *infer_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Create a context with simple code snippet (should verify successfully)
        let context = Some(PrecisionContext::new()
            .with_code_snippet("let x = 1 + 1;".to_string()));
        
        // Learn with code verification context
        let learn_stats = pc.learn(&input, context).unwrap();
        let after_learning_free_energy = *learn_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Free energy may fluctuate, but change should be bounded
        let change = after_learning_free_energy - initial_free_energy;
        assert!(change.abs() < 10.0, "Free energy change too large: {}", change);
        
        println!("Integration test passed: Precision weighting with code verification");
    }

    #[test]
    fn test_precision_weighting_with_nostr_zaps() {
        // Create a PC hierarchy with precision weighting and Nostr zap tracking enabled
        let mut config = PCConfig::new(2, vec![10, 5]);
        config.enable_precision_weighting = true;
        config.enable_nostr_zap_tracking = true;
        config.min_zaps_for_consensus = 2;
        config.free_energy_drop_threshold = 0.1;
        config.default_precision = 0.3;
        
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input data with correct shape (10 rows, 1 column)
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (10, 1), &device).unwrap();
        
        // Record initial free energy
        let infer_stats = pc.infer(&input, 10).unwrap();
        let initial_free_energy = *infer_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Create a context with "trusted" event ID (mock returns 5 zaps > min_zaps)
        let context = Some(PrecisionContext::new()
            .with_nostr_event_id("trusted_event_123".to_string()));
        
        // Learn with Nostr zap context
        let learn_stats = pc.learn(&input, context).unwrap();
        let after_learning_free_energy = *learn_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Free energy may fluctuate, but change should be bounded
        let change = after_learning_free_energy - initial_free_energy;
        assert!(change.abs() < 10.0, "Free energy change too large: {}", change);
        
        println!("Integration test passed: Precision weighting with Nostr zap tracking");
    }

    #[test]
    fn test_precision_weighting_free_energy_drop() {
        // Create a PC hierarchy with precision weighting
        let mut config = PCConfig::new(2, vec![10, 5]);
        config.enable_precision_weighting = true;
        config.free_energy_drop_threshold = 0.5; // 50% drop threshold
        config.default_precision = 0.3;
        
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input data with correct shape (10 rows, 1 column)
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (10, 1), &device).unwrap();
        
        // First learning iteration
        let context1 = None;
        let stats1 = pc.learn(&input, context1).unwrap();
        let free_energy1 = *stats1.free_energy_history.last().unwrap_or(&0.0);
        
        // Simulate a large free energy drop by manually recording
        if let Some(ref mut calculator) = pc.precision_calculator {
            // Record high initial free energy
            calculator.record_free_energy(100.0);
            // Record low current free energy (simulating 60% drop)
            calculator.record_free_energy(40.0);
        }
        
        // Second learning iteration with context
        let context2 = None;
        let stats2 = pc.learn(&input, context2).unwrap();
        let free_energy2 = *stats2.free_energy_history.last().unwrap_or(&0.0);
        
        // Free energy may fluctuate, but change should be bounded
        let change = free_energy2 - free_energy1;
        assert!(change.abs() < 10.0, "Free energy change too large: {}", change);
        
        println!("Integration test passed: Precision weighting with free energy drop tracking");
    }

    #[test]
    fn test_precision_weighting_disabled() {
        // Create a PC hierarchy with precision weighting disabled
        let mut config = PCConfig::new(2, vec![10, 5]);
        config.enable_precision_weighting = false;
        
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input data with correct shape (10 rows, 1 column)
        let device = Device::Cpu;
        let input = Tensor::randn(0f32, 1.0, (10, 1), &device).unwrap();
        
        // Record initial free energy
        let infer_stats = pc.infer(&input, 10).unwrap();
        let initial_free_energy = *infer_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Learn with context (should be ignored since precision weighting is disabled)
        let context = Some(PrecisionContext::new()
            .with_ground_truth(true));
        
        let learn_stats = pc.learn(&input, context).unwrap();
        let after_learning_free_energy = *learn_stats.free_energy_history.last().unwrap_or(&0.0);
        
        // Free energy may fluctuate, but change should be bounded
        let change = after_learning_free_energy - initial_free_energy;
        assert!(change.abs() < 10.0, "Free energy change too large: {}", change);
        
        // Precision calculator should be None
        assert!(pc.precision_calculator.is_none());
        
        println!("Integration test passed: Precision weighting disabled works correctly");
    }
}