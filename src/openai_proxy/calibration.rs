//! Structural Uncertainty Calibration for NeuroFed
//! Transforms raw heuristic confidence into statistically calibrated confidence

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub total_attempts: u64,
    pub successes: u64,
}

impl CalibrationBin {
    pub fn empirical_accuracy(&self) -> f32 {
        if self.total_attempts == 0 {
            return 0.5;
        } // Unknown -> neutral
        self.successes as f32 / self.total_attempts as f32
    }
}

/// Stores empirical success rates for different decile bins of "Raw Confidence"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationStore {
    // 10 bins representing confidence[0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
    pub confidence_bins: Vec<CalibrationBin>,
}

impl Default for CalibrationStore {
    fn default() -> Self {
        Self {
            confidence_bins: vec![
                CalibrationBin {
                    total_attempts: 0,
                    successes: 0
                };
                10
            ],
        }
    }
}

impl CalibrationStore {
    fn get_bin_index(raw_confidence: f32) -> usize {
        let mut idx = (raw_confidence * 10.0).floor() as usize;
        if idx >= 10 {
            idx = 9;
        }
        idx
    }

    /// Records the actual ground-truth outcome of an episode
    pub fn record_outcome(&mut self, raw_confidence: f32, success: bool) {
        let idx = Self::get_bin_index(raw_confidence);
        self.confidence_bins[idx].total_attempts += 1;
        if success {
            self.confidence_bins[idx].successes += 1;
        }
    }

    /// Transforms raw heuristic confidence into statistically calibrated confidence
    pub fn calibrated_confidence(&self, raw_confidence: f32) -> f32 {
        let idx = Self::get_bin_index(raw_confidence);
        let bin = &self.confidence_bins[idx];

        // If we don't have enough data in this bin, fall back to raw confidence
        if bin.total_attempts < 5 {
            return raw_confidence;
        }

        // Return historical empirical probability of success
        bin.empirical_accuracy()
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::*;

    #[test]
    fn test_structural_calibration() {
        let mut store = CalibrationStore::default();

        // The PC thinks it is 95% confident (Raw conf = 0.95 -> Bin 9)
        // But in reality, it fails 80% of the time!
        for _ in 0..8 {
            store.record_outcome(0.95, false);
        } // 8 failures
        for _ in 0..2 {
            store.record_outcome(0.95, true);
        } // 2 successes

        // When queried for calibrated confidence...
        let calibrated = store.calibrated_confidence(0.95);

        // It should output ~0.2 (20% historical accuracy), NOT 0.95.
        assert!(
            (calibrated - 0.2).abs() < 0.01,
            "Calibration failed to correct overconfidence. Got {}",
            calibrated
        );
    }

    #[test]
    fn test_bin_index_calculation() {
        assert_eq!(CalibrationStore::get_bin_index(0.0), 0);
        assert_eq!(CalibrationStore::get_bin_index(0.05), 0);
        assert_eq!(CalibrationStore::get_bin_index(0.15), 1);
        assert_eq!(CalibrationStore::get_bin_index(0.95), 9);
        assert_eq!(CalibrationStore::get_bin_index(1.0), 9);
        assert_eq!(CalibrationStore::get_bin_index(1.5), 9); // Clamped
    }

    #[test]
    fn test_insufficient_data_fallback() {
        let store = CalibrationStore::default();
        // With no data, should return raw confidence
        assert!((store.calibrated_confidence(0.8) - 0.8).abs() < 0.001);
    }
}

#[cfg(test)]
mod active_calibration_tests {
    use super::*;

    #[test]
    fn test_calibration_store_corrects_overconfidence() {
        let mut store = CalibrationStore::default();

        // Scenario: The proxy is extremely confident (0.95), but it hallucinates frequently.
        // We log 1 success and 9 failures at the 0.95 confidence bin.
        store.record_outcome(0.95, true);
        for _ in 0..9 {
            store.record_outcome(0.95, false);
        }

        // When the proxy later asks "What is my *actual* confidence if the PC outputs 0.95?"
        let true_confidence = store.calibrated_confidence(0.95);

        // It should yield 10% (0.1), NOT 95%.
        assert!(
            (true_confidence - 0.1).abs() < 0.001,
            "Expected true confidence to be 0.1, got {}",
            true_confidence
        );
    }

    #[test]
    fn test_calibration_store_needs_minimum_sample_size() {
        let mut store = CalibrationStore::default();

        // Log only 2 attempts (1 success, 1 failure).
        store.record_outcome(0.85, true);
        store.record_outcome(0.85, false);

        let true_confidence = store.calibrated_confidence(0.85);

        // Because sample size < 5, it should fallback to the raw heuristic confidence to prevent knee-jerk bias.
        assert!(
            (true_confidence - 0.85).abs() < 0.001,
            "Should fallback to raw confidence on small sample sizes. Got {}",
            true_confidence
        );
    }
}
