# Mathematical Documentation: Adapter Layer for LLM-to-PC Hierarchy Projection

## Overview
This document defines the mathematical framework for projecting Transformer embeddings into the Predictive Coding (PC) hierarchy during the bootstrap phase. The adapter layer bridges the gap between Transformer latent spaces (optimized for next-token prediction) and PC latent spaces (optimized for free-energy minimization).

## Mathematical Foundations

### 1. Transformer Embedding Space
A Transformer's hidden state at layer $l$ is a vector $\mathbf{h}_l \in \mathbb{R}^{d_{\text{model}}}$ where $d_{\text{model}}$ is typically 1024-4096 dimensions.

### 2. PC Hierarchy Structure
Each level $a$ in the PC hierarchy has:
- Belief vector: $\mathbf{r}_a \in \mathbb{R}^{d_a}$
- Prediction: $\hat{\mathbf{r}}_a = f(\mathbf{U}_a \cdot \mathbf{r}_{a+1})$
- Error: $\boldsymbol{\varepsilon}_a = \mathbf{r}_a - \hat{\mathbf{r}}_a$
- Weights: $\mathbf{U}_a \in \mathbb{R}^{d_a \times d_{a+1}}$

### 3. Adapter Layer Equation
The adapter layer projects Transformer embeddings into the bottom level of the PC hierarchy:

$$\mathbf{r}_0 = \mathbf{W}_{\text{adapter}} \cdot \mathbf{h}_l + \mathbf{b}_{\text{adapter}}$$

Where:
- $\mathbf{W}_{\text{adapter}} \in \mathbb{R}^{d_0 \times d_{\text{model}}}$ is the adapter weight matrix
- $\mathbf{b}_{\text{adapter}} \in \mathbb{R}^{d_0}$ is the adapter bias vector
- $\mathbf{r}_0$ is the initial belief at the bottom level of the PC hierarchy

### 4. Dimensionality Considerations
- If $d_0 < d_{\text{model}}$: The adapter performs dimensionality reduction
- If $d_0 > d_{\text{model}}$: The adapter performs dimensionality expansion
- If $d_0 = d_{\text{model}}$: The adapter performs linear transformation

### 5. Initialization Strategies

#### 5.1 Random Initialization
$$\mathbf{W}_{\text{adapter}} \sim \mathcal{N}(0, \sigma^2), \quad \mathbf{b}_{\text{adapter}} = \mathbf{0}$$

#### 5.2 Identity Initialization (when dimensions match)
$$\mathbf{W}_{\text{adapter}} = \mathbf{I}, \quad \mathbf{b}_{\text{adapter}} = \mathbf{0}$$

#### 5.3 Learned Initialization
Train the adapter on a small dataset to minimize:
$$\mathcal{L} = \|\mathbf{r}_0 - \mathbf{h}_l\|^2$$

### 6. Multi-Layer Adapter
For deeper hierarchies, use multiple adapter layers:

$$\mathbf{r}_0 = \mathbf{W}_n \cdot f(\mathbf{W}_{n-1} \cdot f(\cdots f(\mathbf{W}_1 \cdot \mathbf{h}_l + \mathbf{b}_1) \cdots ) + \mathbf{b}_{n-1}) + \mathbf{b}_n$$

Where $f$ is an activation function (e.g., ReLU, GELU).

### 7. Projection Quality Metrics

#### 7.1 Reconstruction Error
$$\text{RE} = \frac{\|\mathbf{r}_0 - \mathbf{h}_l\|_2}{\|\mathbf{h}_l\|_2}$$

#### 7.2 Correlation Coefficient
$$\rho = \frac{\text{Cov}(\mathbf{r}_0, \mathbf{h}_l)}{\sigma_{\mathbf{r}_0} \sigma_{\mathbf{h}_l}}$$

#### 7.3 KL Divergence (for probabilistic embeddings)
$$\text{KL}(P_{\mathbf{r}_0} \| P_{\mathbf{h}_l})$$

### 8. Implementation in Rust

```rust
// Adapter layer implementation
pub struct AdapterLayer {
    weights: Array2<f32>,
    bias: Array1<f32>,
    activation: Activation,
}

impl AdapterLayer {
    pub fn new(input_dim: usize, output_dim: usize, activation: Activation) -> Self {
        let weights = Array::random((output_dim, input_dim), rand_distr::Normal::new(0.0, 0.01).unwrap());
        let bias = Array::zeros(output_dim);
        Self { weights, bias, activation }
    }
    
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let linear = &self.weights.dot(input) + &self.bias;
        self.activation.apply(linear)
    }
}

// Usage in bootstrap
let adapter = AdapterLayer::new(model_embedding_dim, pc_bottom_dim, Activation::ReLU);
let pc_belief = adapter.forward(&transformer_embedding);
```

### 9. Training the Adapter

```rust
// Adapter training loop
for epoch in 0..max_epochs {
    let mut total_loss = 0.0;
    
    for (embedding, target) in training_data {
        let prediction = adapter.forward(&embedding);
        let loss = mean_squared_error(&prediction, &target);
        
        // Backpropagation
        let grad = compute_gradients(&prediction, &target);
        adapter.update_weights(&grad, learning_rate);
        
        total_loss += loss;
    }
    
    if total_loss < tolerance {
        break;
    }
}
```

### 10. Integration with PC Hierarchy

The adapter output $\mathbf{r}_0$ becomes the initial belief at the bottom level of the PC hierarchy. The hierarchy then processes this belief through:

1. **Prediction**: $\hat{\mathbf{r}}_0 = f(\mathbf{U}_0 \cdot \mathbf{r}_1)$
2. **Error Calculation**: $\boldsymbol{\varepsilon}_0 = \mathbf{r}_0 - \hat{\mathbf{r}}_0$
3. **Inference**: Iterative free-energy minimization
4. **Learning**: Local Hebbian updates to weights

### 11. Mathematical Properties

#### 11.1 Linearity
When using linear activation: $\mathbf{r}_0 = \mathbf{W}_{\text{adapter}} \cdot \mathbf{h}_l + \mathbf{b}_{\text{adapter}}$

#### 11.2 Non-linearity
With activation functions: $\mathbf{r}_0 = f(\mathbf{W}_{\text{adapter}} \cdot \mathbf{h}_l + \mathbf{b}_{\text{adapter}})$

#### 11.3 Differentiability
The adapter layer is differentiable, enabling gradient-based optimization during training.

### 12. Practical Considerations

#### 12.1 Computational Efficiency
- Use matrix multiplication optimizations (BLAS)
- Consider GPU acceleration for large models
- Implement batch processing for multiple embeddings

#### 12.2 Memory Management
- Pre-allocate tensors for adapter operations
- Use memory pools for repeated operations
- Implement proper cleanup of intermediate results

#### 12.3 Numerical Stability
- Use appropriate weight initialization
- Implement gradient clipping during training
- Monitor for exploding/vanishing gradients

### 13. References

1. Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature Neuroscience.

2. Friston, K. (2005). A theory of cortical responses. Philosophical Transactions of the Royal Society B.

3. Taniguchi, T. (2024-2025). Collective Predictive Coding: Decentralized Bayesian Inference for Emergent Intelligence.

4. Vaswani, A., et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems.

5. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

## Conclusion

The adapter layer provides a mathematically sound bridge between Transformer embeddings and PC hierarchies. By carefully considering dimensionality, initialization, and training strategies, the adapter enables effective bootstrapping of PC hierarchies from pre-trained LLMs while maintaining the biological plausibility of the predictive coding framework.