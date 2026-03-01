# 🧠 NeuroFed Node

**The first living cell of the decentralized, federated Global Brain.**

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

**3. Point your tools to the Proxy**
Open your favorite AI application (e.g., Cursor IDE, Open WebUI, Obsidian) and change the API settings:
*   **Base URL:** `http://localhost:8080/v1`
*   **API Key:** `sk-neurofed` (or use your real OpenAI key as a fallback)

**4. Start Typing!**
NeuroFed will immediately begin intercepting requests. If it knows the answer, it responds instantly for free. If it's a complex coding task, it bypasses the request to the Base LLM, observes the correct answer, and *learns the association permanently*.

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
*   **Mesh Networking (Yggdrasil):** End-to-end encrypted IPv6 mesh networking for direct, NAT-bypassing peer communication.
*   **Darknet Routing (Tor / I2P):** Optional SOCKS5 proxy routing via `.onion` and `.b32.i2p` addresses to hide node IP addresses from corporate surveillance.
*   **Web of Trust:** Strict cryptographic signature verification. Nodes only accept knowledge deltas from pubkeys in the user's NIP-02 Contact List or economically verified events.
