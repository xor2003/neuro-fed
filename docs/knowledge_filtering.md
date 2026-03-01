# Knowledge Filtering with Precision Weighting (π)

## Overview
Knowledge Filtering with Precision Weighting (π) is a key enhancement to the NeuroFed Node's Predictive Coding (PC) hierarchy that implements biologically-inspired precision weighting for federated AGI. This system modifies the standard PC learning rule to incorporate precision factors based on information quality, enabling the system to prioritize high-quality information during learning.

## Mathematical Foundation

### Standard PC Learning Rule
The standard Predictive Coding weight update rule is:

$$\Delta \mathbf{U}_l = \eta \cdot \boldsymbol{\varepsilon}_l \cdot \mathbf{r}_{l+1}^T$$

Where:
- $\eta$ is the learning rate
- $\boldsymbol{\varepsilon}_l$ is the prediction error at level $l$
- $\mathbf{r}_{l+1}$ is the belief vector at the next higher level

### Enhanced Learning Rule with Precision Weighting
The enhanced learning rule incorporates precision weighting $\pi$:

$$\Delta \mathbf{U}_l = \eta \cdot \boldsymbol{\varepsilon}_l \cdot \mathbf{r}_{l+1}^T \cdot \pi$$

Where $\pi \in [0, 1]$ is a precision factor that scales the weight updates based on information quality.

## Precision Calculation

The precision $\pi$ is calculated based on multiple factors:

### 1. Free Energy Drop Tracking
Large drops in free energy indicate high-quality, surprising information:

$$\pi_{\text{free\_energy}} = \begin{cases} 
1.0 & \text{if } \frac{F_{\text{prev}} - F_{\text{curr}}}{F_{\text{prev}}} \geq \theta_{\text{free\_energy}} \\
0.3 & \text{otherwise}
\end{cases}$$

Where:
- $F_{\text{prev}}$ is the previous free energy
- $F_{\text{curr}}$ is the current free energy
- $\theta_{\text{free\_energy}}$ is the threshold (default: 0.5 or 50% drop)

### 2. Ground Truth Verification
For code-related prompts, successful execution verification yields maximum precision:

$$\pi_{\text{ground\_truth}} = \begin{cases} 
1.0 & \text{if information is verified ground truth} \\
0.3 & \text{otherwise}
\end{cases}$$

### 3. Economic Consensus (Nostr Zaps)
Nostr events with sufficient zaps from trusted nodes indicate community validation:

$$\pi_{\text{zap}} = \begin{cases} 
1.0 & \text{if zap count} \geq N_{\text{min\_zaps}} \\
0.3 & \text{otherwise}
\end{cases}$$

### 4. Combined Precision
The final precision is the maximum of all applicable precision factors:

$$\pi = \max(\pi_{\text{free\_energy}}, \pi_{\text{ground\_truth}}, \pi_{\text{zap}}, \pi_{\text{default}})$$

Where $\pi_{\text{default}}$ is the default precision for unverified information (default: 0.3).

## Implementation Architecture

### Core Components

#### 1. `PrecisionCalculator`
The main struct that calculates precision values based on context:
- **Free Energy Tracker**: Maintains a sliding window of free energy history
- **Code Verifier**: Stub interface for code execution verification
- **Nostr Zap Tracker**: Stub interface for checking zap consensus
- **Configuration**: Precision thresholds and settings

#### 2. `PrecisionContext`
Context information for precision calculation:
- `code_snippet`: Optional code for verification
- `nostr_event_id`: Optional Nostr event ID for zap checking
- `is_ground_truth`: Boolean flag for ground truth information
- `metadata`: Additional context metadata

#### 3. `PrecisionResult`
Result of precision calculation:
- `precision`: Calculated $\pi$ value ∈ [0, 1]
- `source`: Information source type
- `confidence`: Confidence in the precision calculation
- `metadata`: Calculation metadata

### Integration with Predictive Coding

#### Configuration
The `PCConfig` struct has been extended with precision weighting parameters:
```rust
pub struct PCConfig {
    // ... existing fields ...
    pub enable_precision_weighting: bool,
    pub free_energy_drop_threshold: f32,
    pub default_precision: f32,
    pub min_precision: f32,
    pub max_precision: f32,
    pub free_energy_history_size: usize,
    pub enable_code_verification: bool,
    pub enable_nostr_zap_tracking: bool,
    pub min_zaps_for_consensus: usize,
    pub trusted_node_keys: Vec<String>,
}
```

#### Learning Method Enhancement
The `PredictiveCoding::learn` method now accepts an optional `PrecisionContext`:
```rust
pub fn learn(&mut self, input: &Array2<f32>, context: Option<PrecisionContext>) -> Result<SurpriseStats, PCError>
```

#### Weight Update Modification
The `PCLevel::update_weights` method now accepts an optional precision matrix:
```rust
pub fn update_weights(&mut self, eta: f32, next_level_beliefs: &Array2<f32>, precision: Option<&Array2<f32>>)
```

## Usage Examples

### Basic Usage with Precision Weighting
```rust
use neuro_fed_node::pc_hierarchy::{PCConfig, PredictiveCoding};
use neuro_fed_node::knowledge_filter::PrecisionContext;
use ndarray::Array2;

// Create PC hierarchy with precision weighting enabled
let mut config = PCConfig::new(3, vec![512, 256, 128]);
config.enable_precision_weighting = true;
config.free_energy_drop_threshold = 0.5;
config.default_precision = 0.3;

let mut pc = PredictiveCoding::new(config).unwrap();

// Create input
let input = Array2::random((512, 1), Uniform::new(-1.0, 1.0).unwrap());

// Learn with ground truth context (π = 1.0)
let context = PrecisionContext::new()
    .with_ground_truth(true);
    
let stats = pc.learn(&input, Some(context)).unwrap();
```

### Using Code Verification
```rust
// Enable code verification
let mut config = PCConfig::new(3, vec![512, 256, 128]);
config.enable_precision_weighting = true;
config.enable_code_verification = true;

let mut pc = PredictiveCoding::new(config).unwrap();

// Learn with code snippet context
let context = PrecisionContext::new()
    .with_code_snippet("let x = 1 + 1; println!(\"{}\", x);".to_string());
    
let stats = pc.learn(&input, Some(context)).unwrap();
```

### Using Nostr Zap Tracking
```rust
// Enable Nostr zap tracking
let mut config = PCConfig::new(3, vec![512, 256, 128]);
config.enable_precision_weighting = true;
config.enable_nostr_zap_tracking = true;
config.min_zaps_for_consensus = 3;
config.trusted_node_keys = vec![
    "npub1trustedkey123".to_string(),
    "npub1anothertrusted".to_string(),
];

let mut pc = PredictiveCoding::new(config).unwrap();

// Learn with Nostr event context
let context = PrecisionContext::new()
    .with_nostr_event_id("note1eventid123".to_string());
    
let stats = pc.learn(&input, Some(context)).unwrap();
```

## Configuration Defaults

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `enable_precision_weighting` | `false` | Enable precision weighting |
| `free_energy_drop_threshold` | `0.5` | 50% drop threshold for high precision |
| `default_precision` | `0.3` | Default precision for unverified information |
| `min_precision` | `0.1` | Minimum precision value |
| `max_precision` | `1.0` | Maximum precision value |
| `free_energy_history_size` | `10` | Window size for free energy tracking |
| `enable_code_verification` | `false` | Enable code execution verification |
| `enable_nostr_zap_tracking` | `false` | Enable Nostr zap tracking |
| `min_zaps_for_consensus` | `3` | Minimum zaps for economic consensus |

## Stub Interfaces for Future Implementation

### Code Verification Interface
The current implementation includes a stub `CodeVerifier` that returns mock results. In Phase 2, this will be extended to:
- Execute code snippets in a sandboxed environment
- Verify successful execution and correctness
- Return boolean success/failure

### Nostr Zap Tracking Interface
The current implementation includes a stub `NostrZapTracker` that returns mock zap counts. In Phase 2, this will be extended to:
- Query Nostr relays for zap events
- Verify zap signatures from trusted nodes
- Calculate consensus based on zap amounts and sources

## Testing

The implementation includes comprehensive unit and integration tests:

### Unit Tests
- `test_precision_calculator_default`: Tests default configuration
- `test_free_energy_tracker`: Tests free energy drop calculation
- `test_precision_with_ground_truth`: Tests ground truth precision
- `test_precision_clamping`: Tests precision value clamping

### Integration Tests
- `test_precision_weighting_integration`: Basic integration with PC hierarchy
- `test_precision_weighting_with_context`: Tests with ground truth context
- `test_precision_weighting_with_code_verification`: Tests with code verification
- `test_precision_weighting_with_nostr_zaps`: Tests with Nostr zap tracking
- `test_precision_weighting_free_energy_drop`: Tests free energy drop tracking
- `test_precision_weighting_disabled`: Tests disabled precision weighting

## Performance Considerations

1. **Free Energy Tracking**: Maintains a sliding window of size `free_energy_history_size` (default: 10)
2. **Precision Calculation**: O(1) complexity for most operations
3. **Memory Usage**: Minimal additional memory for precision calculator
4. **Backward Compatibility**: Legacy `learn_legacy` method maintains original behavior

## Future Enhancements

1. **Adaptive Precision**: Dynamically adjust precision based on learning progress
2. **Multi-source Integration**: Combine precision from multiple sources with Bayesian weighting
3. **Temporal Decay**: Apply time-based decay to precision values
4. **Cross-node Validation**: Use federated validation across nodes for precision calculation
5. **Economic Incentives**: Integrate with token economics for precision weighting

## References

1. Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B: Biological Sciences*.
2. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
3. Nostr Protocol: https://github.com/nostr-protocol/nostr
4. Free Energy Principle: https://en.wikipedia.org/wiki/Free_energy_principle