// src/pc_hierarchy.rs
// Pure Predictive Coding (PC) implementation based on Rao-Ballard/Friston free-energy minimization

#![recursion_limit = "4096"]

use ndarray::{Array, Array2, Array3, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::Norm;
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct PCError(String);

impl fmt::Display for PCError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PCError: {}", self.0)
    }
}

impl Error for PCError {}


#[derive(Debug, Clone)]
pub struct PCConfig {
    pub n_levels: usize,
    pub dim_per_level: Vec<usize>,
    pub learning_rate: f32,
    pub inference_steps: usize,
    pub surprise_threshold: f32,
    pub selective_update: bool,
    pub muPC_scaling: bool,
}

impl PCConfig {
    pub fn new(n_levels: usize, dim_per_level: Vec<usize>) -> Self {
        PCConfig {
            n_levels,
            dim_per_level,
            learning_rate: 0.01,
            inference_steps: 20,
            surprise_threshold: 1.0,
            selective_update: true,
            muPC_scaling: false,
        }
    }

    pub fn with_muPC_scaling(mut self, enabled: bool) -> Self {
        self.muPC_scaling = enabled;
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }
}

#[derive(Debug, Clone)]
pub struct PCLevel {
    pub beliefs: Array2<f32>,
    pub predictions: Array2<f32>,
    pub errors: Array2<f32>,
    pub weights: Array2<f32>,
    pub precision: Array2<f32>,
}

impl PCLevel {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let beliefs = Array2::zeros((input_dim, 1));
        let predictions = Array2::zeros((input_dim, 1));
        let errors = Array2::zeros((input_dim, 1));
        let weights = Array2::zeros((input_dim, output_dim));
        let precision = Array2::ones((input_dim, 1));

        PCLevel {
            beliefs,
            predictions,
            errors,
            weights,
            precision,
        }
    }

    pub fn predict(&mut self) {
        // r_hat_l = U_l * r_{l+1}
        // Matrix multiplication: weights (input_dim x output_dim) * beliefs_next_level (output_dim x batch)
        let pre_activation = self.weights
            .dot(&self.beliefs_next_level())
            .mapv(|x| x); // Identity activation function

        self.predictions = pre_activation;
    }

    pub fn compute_errors(&mut self) {
        // epsilon_l = (r_l - r_hat_l) .* precision
        let raw_error = &self.beliefs - &self.predictions;
        self.errors = &raw_error * &self.precision;
    }

    fn beliefs_next_level(&self) -> Array2<f32> {
        // This would normally come from the level above
        // For now, return a zero array of appropriate shape
        Array2::zeros((self.weights.shape()[1], self.beliefs.shape()[1]))
    }

    pub fn update_weights(&mut self, eta: f32) {
        // Delta U_l = eta * epsilon_l * r_{l+1}^T
        let delta_weights = eta * &self.errors.dot(&self.beliefs_next_level().t());
        self.weights = &self.weights + &delta_weights;
    }
}

#[derive(Debug, Clone)]
pub struct SurpriseStats {
    pub total_surprise: f32,
    pub high_surprise_indices: Vec<usize>,
    pub free_energy_history: Vec<f32>,
}

impl Default for SurpriseStats {
    fn default() -> Self {
        SurpriseStats {
            total_surprise: 0.0,
            high_surprise_indices: Vec::new(),
            free_energy_history: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PredictiveCoding {
    pub levels: Vec<PCLevel>,
    pub config: PCConfig,
    pub surprise_threshold: f32,
    pub free_energy: f32,
}

impl PredictiveCoding {
    pub fn new(config: PCConfig) -> Result<Self, PCError> {
        if config.n_levels < 2 {
            return Err(PCError("Hierarchy must have at least 2 levels".to_string()));
        }

        let mut levels = Vec::new();
        
        // Create levels with decreasing dimensions
        for i in 0..config.n_levels {
            let input_dim = config.dim_per_level[i];
            let output_dim = if i < config.n_levels - 1 {
                config.dim_per_level[i + 1]
            } else {
                // Last level has no output
                1
            };
            
            let mut level = PCLevel::new(input_dim, output_dim);
            
            // Initialize weights with small random values
            if i < config.n_levels - 1 {
                let weights = Array2::random((input_dim, output_dim), Uniform::new(-0.1, 0.1).unwrap());
                level.weights = weights * 0.01; // Scale down
            }
            
            levels.push(level);
        }

        Ok(PredictiveCoding {
            levels,
            config: config.clone(),
            surprise_threshold: config.surprise_threshold,
            free_energy: 0.0,
        })
    }

    pub fn infer(&mut self, input: &Array2<f32>, steps: usize) -> Result<SurpriseStats, PCError> {
        if input.shape()[0] != self.config.dim_per_level[0] {
            return Err(PCError(format!("Input dimension {} does not match level 0 dimension {}", 
                input.shape()[0], self.config.dim_per_level[0])));
        }

        // Initialize bottom level with input
        self.levels[0].beliefs = input.clone();
        
        let mut stats = SurpriseStats::default();
        
        for step in 0..steps {
            // Upward pass: compute predictions and errors
            for l in 0..self.levels.len() - 1 {
                self.levels[l].predict();
                self.levels[l].compute_errors();
            }
            
            // Downward pass: update beliefs based on errors
            for l in (0..self.levels.len() - 1).rev() {
                // Belief update: r_l = r_l + eta * epsilon_l
                self.levels[l].beliefs = &self.levels[l].beliefs +
                    self.config.learning_rate * &self.levels[l].errors;
                
                // Propagate error upward to influence beliefs at next level: r_{l+1} += eta * U_l^T · epsilon_l
                if l < self.levels.len() - 2 {
                    let weight_transpose = self.levels[l].weights.t();
                    let belief_update = self.config.learning_rate * weight_transpose.dot(&self.levels[l].errors);
                    self.levels[l+1].beliefs = &self.levels[l+1].beliefs + &belief_update;
                }
            }
            
            // Track free energy and surprise
            let fe = self.compute_free_energy();
            stats.free_energy_history.push(fe);
            
            if fe > self.config.surprise_threshold {
                stats.high_surprise_indices.push(step);
            }
        }
        
        stats.total_surprise = stats.free_energy_history.iter().sum::<f32>();
        Ok(stats)
    }

    pub fn learn(&mut self, input: &Array2<f32>) -> Result<SurpriseStats, PCError> {
        // Perform inference to compute errors
        let stats = self.infer(input, self.config.inference_steps)?;
        
        // Update weights only for high-surprise components
        if self.config.selective_update {
            for (l, level) in self.levels.iter_mut().enumerate() {
                if stats.high_surprise_indices.is_empty() {
                    level.update_weights(self.config.learning_rate);
                }
            }
        } else {
            // Update all weights
            for level in self.levels.iter_mut() {
                level.update_weights(self.config.learning_rate);
            }
        }
        
        self.free_energy = stats.free_energy_history.last().unwrap_or(&0.0).clone();
        Ok(stats)
    }

    fn compute_free_energy(&self) -> f32 {
        // Simplified free energy computation
        // In a full implementation, this would involve KL divergence and other terms
        let mut fe = 0.0;
        
        for l in 0..self.levels.len() - 1 {
            let prediction_error = &self.levels[l].beliefs - &self.levels[l].predictions;
            let squared_error: f32 = prediction_error.mapv(|x| x*x).sum();
            fe += squared_error;
        }
        
        fe / self.levels.len() as f32
    }

    pub fn surprise(&mut self, input: &Array2<f32>) -> Result<f32, PCError> {
        let stats = self.infer(input, 1)?;
        Ok(stats.free_energy_history.last().unwrap_or(&0.0).clone())
    }

    pub fn get_beliefs(&self, level: usize) -> Result<&Array2<f32>, PCError> {
        if level >= self.levels.len() {
            return Err(PCError("Level index out of bounds".to_string()));
        }
        Ok(&self.levels[level].beliefs)
    }

    pub fn get_predictions(&self, level: usize) -> Result<&Array2<f32>, PCError> {
        if level >= self.levels.len() {
            return Err(PCError("Level index out of bounds".to_string()));
        }
        Ok(&self.levels[level].predictions)
    }

    pub fn get_errors(&self, level: usize) -> Result<&Array2<f32>, PCError> {
        if level >= self.levels.len() {
            return Err(PCError("Level index out of bounds".to_string()));
        }
        Ok(&self.levels[level].errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_hierarchy_creation() {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let pc = PredictiveCoding::new(config).unwrap();
        assert_eq!(pc.levels.len(), 3);
        assert_eq!(pc.levels[0].beliefs.shape()[0], 512);
        assert_eq!(pc.levels[1].beliefs.shape()[0], 256);
        assert_eq!(pc.levels[2].beliefs.shape()[0], 128);
    }

    #[test]
    fn test_inference() {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        // Create random input
        let input = Array2::random((512, 1), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0).unwrap());
        
        let stats = pc.infer(&input, 10).unwrap();
        assert_eq!(stats.free_energy_history.len(), 10);
        assert!(stats.total_surprise >= 0.0);
    }

    #[test]
    fn test_learning() {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        let input = Array2::random((512, 1), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0).unwrap());
        
        let stats = pc.learn(&input).unwrap();
        assert!(stats.free_energy_history.len() > 0);
    }

    #[test]
    fn test_surprise() {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let mut pc = PredictiveCoding::new(config).unwrap();
        
        let input = Array2::random((512, 1), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0).unwrap());
        let surprise = pc.surprise(&input).unwrap();
        assert!(surprise >= 0.0);
    }

    #[test]
    fn test_accessors() {
        let config = PCConfig::new(3, vec![512, 256, 128]);
        let pc = PredictiveCoding::new(config).unwrap();
        
        assert!(pc.get_beliefs(0).is_ok());
        assert!(pc.get_predictions(0).is_ok());
        assert!(pc.get_errors(0).is_ok());
        
        // Test out of bounds
        assert!(pc.get_beliefs(10).is_err());
    }

// Example usage
#[cfg(test)]
pub fn example_usage() {
    // Create a basic 3-level hierarchy
    let config = PCConfig::new(3, vec![512, 256, 128]);
    let mut pc = PredictiveCoding::new(config).unwrap();
    
    // Create random input
    let input = Array2::random((512, 1), ndarray_rand::rand_distr::Uniform::new(-1.0, 1.0).unwrap());
    
    // Perform inference
    let stats = pc.infer(&input, 20).unwrap();
    println!("Free energy history: {:?}", stats.free_energy_history);
    println!("Total surprise: {}", stats.total_surprise);
    
    // Learn from input
    let learning_stats = pc.learn(&input).unwrap();
    println!("Learning surprise: {}", learning_stats.total_surprise);
    
    // Get beliefs from level 1
    let beliefs = pc.get_beliefs(1).unwrap();
    println!("Level 1 beliefs shape: {:?}", beliefs.shape());
}
}