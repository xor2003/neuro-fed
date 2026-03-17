# 🧠 NeuroFed Node

**The first living cell of the decentralized, federated Global Brain.**

## Project Status

The repository currently builds as an experimental prototype. The code path that is concretely wired today is:

`main.rs` -> GGUF/tokenizer load in `ml_engine.rs` -> predictive-coding inference in `pc_hierarchy.rs`

Other subsystems such as the async node loop, brain sharing, federation, and proxy routing should be treated as roadmap work unless they are explicitly called out below as implemented. Read the current code as a prototype, not as a finished distributed node.

NeuroFed Node is a biologically plausible, offline-first AI system built on **Pure Hierarchical Predictive Coding (PC)**. It runs locally on your CPU or GPU, continuously learning from your interactions, and federates its knowledge over the Nostr network. 

It is not just another LLM wrapper—it is a continuous-learning cognitive engine that acts as a transparent proxy for your existing AI tools, creating a highly personalized "digital twin" that earns crypto (Zaps) by sharing conceptual breakthroughs with a decentralized network.

---

## 🌟 Short Description for Users

Imagine an AI that actually remembers what you taught it yesterday. 

NeuroFed sits quietly in the background of your computer as a **Smart Proxy**. You simply point your existing tools (Cursor, VS Code, Obsidian, AnythingLLM) to `http://localhost:8080/v1`. 

Under the hood, NeuroFed uses a tiny local LLM simply as "eyes and ears" to translate text into math. The real magic happens in the **Predictive Coding Hierarchy**—a fast, lightweight matrix that mimics the human cortex. When you teach it something new, it experiences "surprise" (Free Energy) and instantly rewires its local matrices to understand you better next time. 

It learns your codebase, your writing style, and your logic—100% privately. Then, it anonymously gossips these tiny "knowledge updates" (deltas) with other NeuroFed nodes across the globe, creating a decentralized superintelligence without ever sharing your raw data.

---

## ⚡ Killer Features

*   **🔌 The "Zero-Friction" Smart Proxy:** No new UIs to learn. Change your OpenAI Base URL to `localhost:8080` in your favorite app. NeuroFed intercepts, caches, learns from, and routes your requests automatically to save you API costs.
*   **🧠 "I Know Kung Fu" Brain Downloads:** Because NeuroFed separates *language* from *logic*, the logic weights are tiny (~50MB). You can instantly download a fully matured "Senior Rust Developer Brain" or "Medical Diagnostics Brain" shared by the community via Nostr Blossom.
*   **💸 Earn While You Think:** If your node discovers a highly efficient way to solve a problem, it broadcasts the "Error Delta" to the network. If other nodes find your delta useful, they automatically send Lightning Network micropayments (Zaps) to your wallet.
*   **🛌 Sleep & Dream Mode:** When you step away from your computer, your node goes to sleep. It replays the day's most surprising interactions, consolidates memories, and mathematically optimizes its worldview (minimizing global free energy).

## Current Architecture Snapshot

- `src/main.rs` is still a smoke-test style executable: it loads a local model, creates a PC hierarchy, runs one sample inference, then waits for `CTRL+C`.
- `src/ml_engine.rs` is the active frontend of the system. It embeds text by loading GGUF tensors directly and pooling token embeddings.
- `src/pc_hierarchy.rs`, `src/pc_level.rs`, and `src/pc_types.rs` are the closest thing to the current core.
- `src/config.rs` contains the main runtime config surface, while `src/types.rs` still carries overlapping legacy/common structs.
- `src/persistence.rs` and `src/model_manager.rs` contain substantial implementation, but they are not yet fully integrated into the runtime entrypoint.

## Major Architectural Risks

1. Split-brain domain model.
   `src/config.rs`, `src/types.rs`, and `src/pc_types.rs` define overlapping concepts such as config, errors, and PC-related types. That increases adapter code, silent defaulting, and long-term schema drift.

2. Runtime path is much narrower than the module graph.
   The current executable does not yet orchestrate the node loop, proxy, federation, persistence, and brain-sharing layers together. That makes architectural assumptions easy to overstate and hard to validate under real load.

3. Shared mutable state will serialize throughput.
   Core components are wrapped behind `Arc<Mutex<...>>`. That is acceptable for bootstrapping, but once request handling, background learning, and federation run concurrently, those locks become a latency and contention hotspot.

4. Initialization is brittle and side-effect heavy.
   Model startup depends on hard-coded local paths and may attempt fallback downloads for tokenizer assets. That makes reproducibility, offline behavior, and deployment predictability weaker than the README currently implies.

5. Event orchestration is polling-based and mostly unimplemented.
   `src/node_loop.rs` has the shape of a runtime loop, but the handlers are stubs and the loop currently polls intervals before awaiting channel receives. That is fine for scaffolding, but it is not yet a strong production concurrency design.

## TODO

- Integrate `src/node_loop.rs` into the runtime and replace its placeholder handlers with real orchestration logic.
- Integrate `src/brain_manager.rs` end-to-end before presenting brain sharing as an available workflow.
- Integrate federation and proxy routing into the executable path before documenting them as active runtime features.
- Collapse duplicated config/type definitions into one canonical source per concept.
- Make model initialization deterministic and explicitly offline-safe.
- Add integration tests for the real end-to-end path instead of relying on module presence as evidence of integration.

### Recent Enhancements
- **Early Exiting Inference:** PC inference now stops early when free energy drops below a configurable convergence threshold, saving compute time.
- **μPC Scaling for Deep Hierarchies:** Added μPC scaling to stabilize learning in deep hierarchies by scaling learning rates with input dimension.
- **Memory API:** Export PC hierarchy memory as human‑readable JSON for inspection and debugging.
- **Dreaming (Top‑Down Generation):** Generate bottom‑level representations from top‑level seeds via transposed weight projection.
- **Semantic Cache Persistence:** Semantic cache is now automatically saved to `./semantic_cache.json` and reloaded across restarts.

---

## 🥊 NeuroFed (Predictive Coding) vs. Transformers (LLMs)

| Feature | 🤖 Traditional Transformers (GPT-4, Llama) | 🧠 NeuroFed (Predictive Coding) |
| :--- | :--- | :--- |
| **Learning Speed** | Frozen in time. Requires millions of dollars and weeks of GPU fine-tuning to learn new concepts. | **Instant & Continuous.** Learns immediately from high-surprise events using local Hebbian updates. |
| **Size & Efficiency** | Bloated. Memorizes the entire internet, requiring massive VRAM and draining laptop batteries. | **Tiny & Agile.** Only stores compressed logic and personal context (~50MB). Runs smoothly on CPUs. |
| **Hallucinations** | High. They are advanced autocomplete engines guessing the most statistically likely next word. | **Mathematically Resistant.** Built to minimize internal contradictions (Free Energy). It verifies logic. |
| **Collaboration** | Isolated. You cannot easily merge the "knowledge" of two different instances of ChatGPT. | **Natively Collective.** Nodes naturally merge compressed matrix deltas (beliefs) via decentralized gossip. |
| **Data Privacy** | Your personal data is sent to corporate servers to be scraped for future training runs. | **100% Local.** Raw data never leaves your machine. Only abstract mathematical deltas are federated. |

---

## 🚀 How to Use (Quick Start)

**1. Install NeuroFed Node**
```bash
# Download and install the pre-compiled binary
curl -L https://neuro-fed.ai/install | sh
```

**2. Start the Node**
```bash
# Starts the ML Engine, Predictive Coding core, and Smart Proxy
neuro-fed-node start
```

**2.5. Quick GUI (Local)**
```bash
rm -f neurofed.db detail.log
cargo run --features web-ui --bin neuro-fed-node -- --config config.toml
```
Open: `http://localhost:8080/ui`
- **Send** = normal chat history
- **Ask Once** = single-shot query (does not persist in local storage)

If you want visible reasoning steps:
```toml
[proxy_config]
require_thought_ops = true
min_thought_ops = 2

[pc_config]
inference_steps = 8
thought_vocab_capacity = 256
```

**3. Point your tools to the Proxy**
Open your favorite AI application (e.g., Cursor IDE, Open WebUI, Obsidian) and change the API settings:
*   **Base URL:** `http://localhost:8080/v1`
*   **API Key:** `sk-neurofed` (or use your real OpenAI key as a fallback)

**4. Start Typing!**
NeuroFed will immediately begin intercepting requests. If it knows the answer, it responds instantly for free. If it's a complex coding task, it bypasses the request to the Base LLM, observes the correct answer, and *learns the association permanently*.

### ThoughtOps in the UI
The UI “Current Steps” panel now includes the **ThoughtOps** trajectory (e.g., `PLAN -> COMPUTE_MATH -> RETURN_VALUE`) when available.

### Seeded Demo Stories
To bootstrap reasoning quickly, use:
- `study/user_stories_seed.txt`
- `study/user_stories_thoughtops.jsonl` (explicit ThoughtOps traces)

---

## 📋 Full Feature List

### Core ML & AI
*   **Pure Rust ML Engine:** Built on HuggingFace's `candle-core`. Zero Python, zero C++ dependencies. Natively supports CPU, NVIDIA CUDA, and Apple Metal.
*   **Hardware Auto-Detection:** Automatically profiles your system and pulls the optimal quantized embedding model (e.g., Qwen 1.5B or Nomic Embed).
*   **Knowledge Filtering (Precision Weighting $\pi$):** Uses mathematical filters, ground-truth verification, and economic consensus to ensure the node only learns "genius-level" data and ignores hallucinations.

### Smart Proxy & Routing
*   **Tool Calling Bypass:** Automatically detects `tools` in JSON payloads and routes them to frontier models while caching the semantic intent.
*   **Semantic Caching:** Uses vector embeddings to match incoming questions with previously learned answers, returning results in milliseconds and saving API costs.
*   **Frontier Model Consensus (Arena Mode):** For high-complexity tasks, queries OpenAI, Anthropic, and local models simultaneously, compares their semantic outputs, and learns only from the mathematical consensus.

### Decentralization & Federation
*   **Dual Federation Modes:** 
    *   *Wallet Mode:* Uses Nostr Zaps (Lightning Network) for Sybil resistance and economic rewards.
    *   *No-Wallet Mode:* Uses Proof-of-Work (PoW) cryptographic challenges for users without crypto wallets.
*   **Decentralized Brain Sharing (NIP-94):** Upload and download full `.safetensors` PC-brains via the Nostr Blossom protocol.
*   **CRDT State Sync:** Utilizes conflict-free replicated data types (`automerge`) to seamlessly merge knowledge from thousands of peers without database corruption.

### Privacy & Security
*   **Web of Trust:** Strict cryptographic signature verification. Nodes only accept knowledge deltas from pubkeys in the user's NIP-02 Contact List or economically verified events.

---

## 📚 Scientific Foundations

NeuroFed is not a "black box" neural network. It is a direct implementation of decades of neuroscience research into how the biological cortex actually learns and reasons.

### 1. The Core Engine: Predictive Coding
The architecture is based on the **Rao-Ballard model**, which posits that the brain is a hierarchical inference engine that constantly generates top-down predictions and minimizes bottom-up prediction errors (Free Energy).
*   *Rao, R. P., & Ballard, D. H. (1999).* **[Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.](https://www.nature.com/articles/nn0199_79)** *Nature Neuroscience*.

### 2. The Learning Objective: Free Energy Principle
We minimize "Surprise" (thermodynamic Free Energy) rather than just "Loss". This allows the node to learn continuously from a single example without the catastrophic forgetting common in backpropagation-based systems.
*   *Friston, K. (2010).* **[The free-energy principle: a unified brain theory?](https://www.nature.com/articles/nrn2787)** *Nature Reviews Neuroscience*.

### 3. The Federation Protocol: Collective Predictive Coding (CPC)
NeuroFed implements Taniguchi's hypothesis that shared symbols (language/logic) emerge from decentralized Bayesian inference across a population of agents, rather than from a central training run.
*   *Taniguchi, T., et al. (2023).* **[Collective predictive coding hypothesis: symbol emergence as decentralized Bayesian inference.](https://www.frontiersin.org/articles/10.3389/frobt.2023.1233777/full)** *Frontiers in Robotics and AI*.

### 4. Scaling to Deep Hierarchies: μPC
To handle complex reasoning tasks, we utilize the "Micro Predictive Coding" (μPC) parameterization, allowing deep hierarchical networks to be trained stably using local learning rules.
*   *Innocenti, F., et al. (2025).* **[μPC: Scaling Predictive Coding to 100+ Layer Networks.](https://arxiv.org/abs/2505.13124)** *arXiv preprint*.
