# NeuroFed Node Development Guide

## Project Overview
NeuroFed (NeuroFed) Node is a decentralized federated AGI system based on pure hierarchical predictive coding. It implements a biologically plausible, fully decentralized, offline-first federated AGI system using Rust, candle framework, and Nostr protocol.

## Project Ideas / Goals
- Build a personal AI that can be used by other users and is not tied to any organization.
- Self-study via proxy use, local/remote books, YouTube subtitles, internet research, and fact checking.
- Strong coding capability with durable code knowledge; also strong text understanding and reasoning.
- Final package size under 500 MB.
- Use modern algorithms; support CPU/GPU/TPU when available; prioritize CPU cache locality.
- Avoid long-term degradation (stability of learned knowledge over time).
- TODO: Share database with other users (e.g., via Nostr). Support federated redundancy and multi-user desktop deployment to share compute resources and resist corporate centralization.
- TODO: Viral and eye-candy experience (memorable UI/UX and shareable story).
- License: GPL-3.0-or-later.
- Support fast and slow thinking modes.
- Primary target: x86_64_v3, 32 GB RAM.

## Current Status
- Treat the current codebase as an experimental single-process prototype, not a fully integrated node.
- The authoritative architecture is the code under `src/`, not the aspirational module list below.
- `cargo check` currently passes, but several top-level modules are only partially wired into the runtime.
- When reviewing or extending the system, distinguish between:
  - implemented runtime path: `main.rs` -> `ml_engine.rs` -> `pc_hierarchy.rs`
  - implemented but not fully composed infrastructure: `persistence.rs`, `model_manager.rs`
  - compatibility and placeholder surfaces: `types.rs`, some re-exports in `lib.rs`

## Directory Structure
```
neuro-pc-node/
├── src/
│   ├── main.rs              # Minimal executable path and smoke-test style startup
│   ├── lib.rs               # Public module graph and compatibility re-exports
│   ├── config.rs            # Primary runtime configuration types
│   ├── types.rs             # Legacy/common DTO-style types; not the single source of truth
│   ├── persistence.rs       # SQLite persistence for PC weights, peers, cache
│   ├── node_loop.rs         # Event loop skeleton
│   ├── ml_engine.rs         # GGUF/tokenizer loading and text embedding pipeline
│   ├── model_manager.rs     # Model selection and download logic
│   ├── pc_hierarchy.rs      # Predictive coding orchestration
│   ├── pc_level.rs          # Per-level PC update logic
│   ├── pc_types.rs          # Canonical PC config/error/stat types
│   ├── pc_decoder.rs        # Belief decoding logic
│   ├── bootstrap.rs         # Synthetic/bootstrap training utilities
│   ├── brain_manager.rs     # Brain sharing workflow
│   ├── semantic_cache.rs    # Semantic cache implementation
│   └── pow_verifier.rs      # PoW verification support
├── docs/
│   ├── architecture.md      # System architecture documentation
│   ├── equations.md         # Mathematical foundations
│   ├── api.md              # Public API documentation
│   └── installation.md     # Installation and setup guide
├── examples/
│   ├── basic_usage.rs
│   ├── federated_demo.rs
│   └── performance_bench.rs
├── ui/                    # Web UI assets (HTML/JS/CSS)
├── resources/
│   ├── default_config.toml
│   ├── models/             # GGUF model storage
│   └── schemas/            # Database schemas
├── scripts/
│   ├── build_release.sh
│   ├── cross_compile.sh
│   └── test_all.sh
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── release.yml
│   │   └── nightly.yml
│   └── ISSUE_TEMPLATE/
├── Cargo.toml              # Rust package configuration
├── README.md
├── LICENSE
└── .gitignore
```

## Development Setup

### Prerequisites
- Rust (stable 2026 or nightly)
- Git
- For development: Docker (optional)

### Installation
```bash
# Clone the repository
git clone https://github.com/neuro-pc/neuro-pc-node.git
cd neuro-pc-node

# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install dependencies
cargo install cargo-watch cargo-audit
```

### Environment Setup
```bash
# Set up development environment
export RUST_LOG=debug
export RUST_BACKTRACE=1

# For GPU development (if available)
export GPU_BACKEND=cuda  # or metal, vulkan, cpu
export GPU_DEVICE_ID=0
export GPU_MEMORY_FRACTION=0.8
```

## Development Workflow

### Code Structure
- **src/main.rs**: Application entry point and main loop
- **src/lib.rs**: Public API exports
- **src/config.rs**: Configuration management
- **src/types.rs**: Legacy/common data types; avoid adding new canonical config here
- **src/persistence.rs**: SQLite database and state persistence
- **src/node_loop.rs**: Async processing loop skeleton
- **src/ml_engine.rs**: ML Engine using candle framework for pure Rust CPU/GPU operations
- **src/pc_hierarchy.rs**: Pure Predictive Coding implementation

### Learning/Generation Quality Gate (Required)
For every code update, enforce the following before considering the step complete:
1. **Parse learning log**: run `cargo run --bin learning_benchmark -- --skip-run` and review the updated `learning_feedback.csv`.
2. **Investigate anomalies**: if losses, trajectories, or counts look worse than the previous step, inspect `detail.log` and explain the regression.
3. **Apply fixes**: do not proceed to the next change until learning/generation is at least stable or improved.
4. **Smoke coverage**: keep learning/generation unbroken by running lightweight tests when feasible (e.g., `cargo test --lib` and `cargo test --test integration_tests`).

### Improvement Workflow (No Confirmations)
Proceed through the current plan step-by-step without asking for confirmation:
1. Stabilize flaky learning-related tests (deterministic inputs and tolerances).
2. Expand learning log parsing to include replay/sleep-phase entries.
3. Add a JSONL reasoning dataset loader for replay.
After each step, run the Learning/Generation Quality Gate and only advance if metrics are stable or improved.

### Recursive Execution Rule
For every future plan, the final step must always be:
- create the next best plan from the new project state and continue on it

Operationally, each completed step should follow this loop:
1. implement the current best step
2. run the Learning/Generation Quality Gate
3. update the relevant docs
4. create a non-amended commit for the validated step
5. create the next best plan from the new state and continue without stopping

### Enormous Recursive Roadmap
Use this as the default long-horizon execution order for all future plans. Do not try to finish all of it in one step; always pick the highest-leverage next slice that can end in a working validated state.

1. Keep the current build, tests, and learning gate green at every step.
2. Preserve deterministic reasoning tasks as a non-regressing baseline.
3. Keep investigation, code, and text outputs structurally sectioned.
4. Score structured assistant sections in replay and benchmark logs.
5. Add heuristic evaluators for investigation findings/evidence quality.
6. Add heuristic evaluators for code implementation/verification/risk quality.
7. Add heuristic evaluators for text rewrite/quality-check fidelity.
8. Reuse prior investigation notes through persistent retrieval.
9. Reuse prior code workflow notes through persistent retrieval.
10. Reuse prior text workflow notes through persistent retrieval.
11. Strengthen code-task workflow contracts with touched-area summaries.
12. Strengthen code-task workflow contracts with explicit verification commands.
13. Strengthen code-task workflow contracts with residual-risk summaries.
14. Strengthen text-task workflow contracts with tone/length constraints.
15. Strengthen text-task workflow contracts with fidelity checks.
16. Strengthen investigation workflows with evidence/open-question persistence.
17. Convert successful assistant episodes into richer replay rows.
18. Add benchmark cases for structured investigation outputs.
19. Add benchmark cases for structured code-task outputs.
20. Add benchmark cases for structured text-task outputs.
21. Fail the gate when structured outputs collapse into generic chat.
22. Fail the gate when code verification disappears from code-task outputs.
23. Fail the gate when evidence/open-questions disappear from investigations.
24. Fail the gate when text quality checks disappear from rewrites.
25. Add planner state that can mark substeps pending/in-progress/completed.
26. Split planner state from execution state.
27. Allow plan revision when new evidence invalidates assumptions.
28. Make the assistant summarize why a plan changed.
29. Add a bounded software-development executor loop.
30. Let the assistant inspect code paths before proposing edits.
31. Let the assistant choose the narrowest relevant verification command.
32. Let the assistant store verified coding patterns for reuse.
33. Add a bounded text-editing executor loop.
34. Let the assistant store reusable text transformation patterns.
35. Add a bounded investigation executor loop.
36. Let the assistant carry unresolved questions across sessions.
37. Add user-preference memory for writing/help style.
38. Add reminder/task continuity for personal-assistant behavior.
39. Add durable project-specific architecture notes for coding assistance.
40. Expand dataset normalization to include structured assistant outputs.
41. Expand replay scoring to use section-level evaluator signals.
42. Expand sleep consolidation to learn from structured workflow episodes.
43. Add anti-cheat metrics for missing or trivial workflow structure.
44. Add evaluator summaries back into the stored workflow memories.
45. Add queueable background tasks for deferred study and follow-up.
46. Start integrating real executor behavior into `node_loop.rs`.
47. Reduce runtime/documentation drift after each integrated subsystem step.
48. Reduce type drift between config/types/pc_types while preserving behavior.
49. Prefer changes that compound verification, memory, and replay quality together.
50. End every completed plan by creating the next best plan from the new state and continuing automatically.

Selection rule:
- If several possible next steps exist, prefer the smallest step that improves at least one of:
  - output structure
  - evaluation quality
  - retrieval memory
  - replay/benchmark coverage
  - autonomous software-development usefulness

### Reasoning Replay JSONL Format
To exercise reasoning → state → output paths via `learning_benchmark --reasoning-replay`, provide JSONL with fields:
- `task`: one of `multiply`, `reverse_string`, `sum_even`, `max`, `sort_list`
- Task fields:
  - `multiply`: `a`, `b`
  - `reverse_string`: `input`
  - `sum_even`/`max`/`sort_list`: `values` (array of integers)
- Optional:
  - `ops`: array of ThoughtOps (e.g., `PLAN`, `DECOMPOSE`, `INITIALIZE_VARIABLE`, `COMPUTE_MATH`, `RETURN_VALUE`, `EOF`)
  - `expected_output`: string used for text-loss check
  - query aliases: `raw_query`, `query`, `problem`, or `instruction`
  - `expected` is also accepted as an alias for `expected_output`, including numeric values for solver tasks

If `ops` is omitted, the benchmark now fills in the canonical recommended ThoughtOp chain for the task automatically.
If no query field is provided, it synthesizes a stable default query string from the task payload so replay logs stay readable and comparable.

Example line:
```json
{"task":"multiply","a":17,"b":23,"ops":["PLAN","DECOMPOSE","INITIALIZE_VARIABLE","COMPUTE_MATH","REFINE","RETURN_VALUE","EOF"],"expected_output":"391"}
```

### Learning Data Pipeline
Use `scripts/generate_learning_dataset.py` to normalize multi-type JSONL into a single structured stream:
```bash
python scripts/generate_learning_dataset.py --input assistant.jsonl,reasoning.jsonl,code.jsonl,agent.jsonl --output merged_learning.jsonl
```
Expected input record types (`type` field):
- `assistant`: `user`, `assistant`
- `reasoning`: `problem`, `thoughts` (array), `solution`
- `code`: `instruction`, `code`, `tests`, optional `final`
- `agent`: `goal`, `tool_call`, `observation`, `next_action`

Additional reasoning task types for replay JSONL:
- `sympy_eval`: `expression`, `operation`, optional `expected`
- `z3_solve`: `var`, `constraints` (array of strings), optional `expected`

### Dataset Query/Adjust
Use `scripts/query_learning_dataset.py` to inspect and filter the merged JSONL:
```bash
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --stats
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --type reasoning --max-chars 3000 --output data/reasoning_filtered.jsonl
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --contains "toxicity|abuse" --output data/cleaned.jsonl
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --preset alpaca --output data/alpaca_filtered.jsonl
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --preset openassistant --min-score 0.6 --output data/oa_filtered.jsonl
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --preset reasoning --output data/reasoning_preset.jsonl --stats
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --max-chars 2500 --output data/merged_filtered.jsonl --stats
```

For reasoning training, the merged dataset is not sufficient by itself. Use the dedicated preparation step:
```bash
.venv/bin/python scripts/prepare_reasoning_dataset.py --input data/merged_learning.jsonl --output data/reasoning_ready.jsonl
```
This keeps reasoning/code/agent rows and only retains assistant rows that look reasoning-relevant, while attaching `reasoning_score` metadata and synthesizing minimal thought traces when safe.

If you want an OpenAI-compatible remote LLM to preprocess rows further, use:
```bash
.venv/bin/python scripts/llm_prepare_reasoning_dataset.py \
  --input data/merged_learning.jsonl \
  --output data/reasoning_ready_llm.jsonl \
  --base-url http://YOUR_OPENAI_COMPATIBLE_HOST/v1 \
  --api-key YOUR_KEY \
  --model YOUR_MODEL
```
Recommended flow:
1. run `prepare_reasoning_dataset.py` first for cheap local filtering
2. then run `llm_prepare_reasoning_dataset.py` on the merged or filtered dataset
3. keep `--min-heuristic-score` above 2 so the remote LLM only sees likely reasoning rows

If you want the local non-LLM path only:
```bash
.venv/bin/python scripts/llm_prepare_reasoning_dataset.py \
  --input data/merged_learning.jsonl \
  --output data/reasoning_ready_dryrun.jsonl \
  --dry-run
```

### Dataset Fetch (Required Sources)
Use `scripts/fetch_datasets.py` to download and convert the required datasets into raw JSONL:
```bash
.venv/bin/python scripts/fetch_datasets.py --datasets alpaca,dolly,openassistant,gsm8k,strategyqa,hotpotqa,codesearchnet,humaneval --limit 5000 --streaming
```
Notes:
- Requires a local venv with `datasets` + `huggingface_hub`:
  ```bash
  python3 -m venv .venv
  .venv/bin/pip install datasets huggingface_hub
  ```
- Set `HF_TOKEN` in `.env` for higher rate limits.
- For CodeSearchNet or The Stack, pass `--language python` (or `rust`, `go`, `java`, `javascript`).
- For The Stack subset, add `the_stack` to `--datasets` and set a small `--limit`.
- Agent datasets:
  - ToolBench: add `toolbench`
  - WebArena: add `webarena`

### End-to-End Conversion
```bash
.venv/bin/python scripts/fetch_datasets.py --datasets alpaca,dolly,openassistant,gsm8k,strategyqa,hotpotqa,codesearchnet,humaneval --limit 5000 --streaming
.venv/bin/python scripts/fetch_datasets.py --datasets toolbench,webarena --limit 200 --streaming
.venv/bin/python scripts/generate_learning_dataset.py --input data/raw/alpaca.jsonl,data/raw/dolly.jsonl,data/raw/openassistant.jsonl,data/raw/gsm8k.jsonl,data/raw/strategyqa.jsonl,data/raw/hotpotqa.jsonl,data/raw/codesearchnet.jsonl,data/raw/humaneval.jsonl,data/raw/toolbench.jsonl,data/raw/webarena.jsonl --output data/merged_learning.jsonl
.venv/bin/python scripts/prepare_reasoning_dataset.py --input data/merged_learning.jsonl --output data/reasoning_ready.jsonl
.venv/bin/python scripts/query_learning_dataset.py --input data/merged_learning.jsonl --max-chars 2500 --output data/merged_filtered.jsonl --stats
.venv/bin/python scripts/augment_reasoning_dataset.py --input data/merged_filtered.jsonl --output data/merged_augmented.jsonl
```

### Reasoning Augmentation (Cycle)
To automatically add simple reasoning traces to assistant rows:
```bash
.venv/bin/python scripts/augment_reasoning_dataset.py --input data/merged_learning.jsonl --output data/merged_learning_augmented.jsonl
.venv/bin/python scripts/augment_reasoning_dataset.py --input data/merged_filtered.jsonl --output data/merged_augmented.jsonl
```
This only augments simple arithmetic (`a + b`, `a - b`, `a * b`) when `thought` is missing.

### Reasoning Tooling
- Z3 integration lives in `src/reasoning_tools.rs` and is gated by feature flag `z3-tools`.
- SymPy checks use a Python subprocess (`python3 -c ...`); set `PYTHON` env var to override.
- **src/model_manager.rs**: Model detection, recommendation, and downloading
- **src/bootstrap.rs**: Bootstrap and synthetic training utilities
- **src/brain_manager.rs**: Brain sharing and import/export workflow

### Minimal PC Mode (Working Baseline)
Enable the minimal, stable predictive-coding loop (simple inference + learning rule):
```bash
# config.toml
[pc_config]
minimal_pc_mode = true
```
Use the small reasoning dataset:
```bash
study/minimal_pc/data/minimal_pc_sum.jsonl
```
Run a quick learning benchmark on the minimal set:
```bash
rm -f neurofed.db detail.log && \
  cargo run --bin learning_benchmark -- \
  --study-paths study/minimal_pc/data/minimal_pc_sum.jsonl
```
Or use the helper script:
```bash
scripts/run_minimal_pc.sh
```
Optional smoke test (disabled by default in CI/sandbox):
```bash
RUN_MINIMAL_PC_SCRIPT_SMOKE=1 cargo test --test minimal_pc_script_smoke
```

### Architectural Risks To Watch
- **Type drift across modules**: `config.rs`, `types.rs`, and `pc_types.rs` define overlapping concepts. New work should consolidate around one canonical type per concept instead of adding more adapters.
- **Runtime/documentation drift**: the binary currently exercises only a narrow startup path. Do not document subsystems as production-ready unless they are actually invoked from `main.rs` or an equivalent entrypoint.
- **Single-process lock contention**: `Arc<Mutex<...>>` around core ML and PC state is acceptable for the prototype, but it will serialize work and limit throughput once proxy and federation paths become active.
- **Blocking/external side effects during model init**: tokenizer/model fallback logic may trigger filesystem or network-dependent behavior at construction time. Keep initialization deterministic where possible.
- **Stubbed orchestration**: `node_loop.rs` currently proves lifecycle shape, not business behavior. Avoid building new assumptions on top of its placeholder handlers without implementing them first.

### TODO
- Integrate `node_loop.rs` into the actual runtime and replace placeholder handlers with real user/file/Nostr processing.
- Wire `brain_manager.rs` into the executable path and document the operational workflow only after end-to-end integration exists.
- Reduce compatibility/placeholder reliance in `types.rs` by moving callers to canonical types in `config.rs` and `pc_types.rs`.
- Promote currently standalone infrastructure such as persistence and model management into tested end-to-end flows.

### Building the Project
```bash
# Build in debug mode
cargo build

# Build in release mode
cargo build --release

# Build with web UI (Phase 3)
cargo build --features web-ui

# Build for specific target
cargo build --target x86_64-unknown-linux-gnu
```

### Running the Project
```bash
# Run in debug mode
cargo run

# Run with specific configuration
cargo run -- --config config.toml

# Run with web UI
cargo run --features web-ui

# Run tests
cargo test

# Run with specific test
cargo test ml_engine::tests::test_embedding_creation
```

### Quick GUI Run (Thoughtful Answers)
```bash
# Ensure config.toml has: web_ui_enabled=true, bootstrap_on_start=true,
# require_thought_ops=true, min_thought_ops=2, inference_steps=8
cargo run --features web-ui --bin neuro-fed-node -- --config config.toml
```
Then open:
- `http://localhost:8080/ui`
Ask a question and verify the response includes ThoughtOps and a coherent answer.
Use **Ask Once** for a single-shot query without storing chat history in local storage.

Seeded demo content (user stories):
- `study/user_stories_seed.txt`
- `study/user_stories_thoughtops.jsonl`

### Full-Mode Reasoning Knobs
Tune these for better “thinking” in full mode:
- `pc_config.inference_steps` (e.g., 8–16). Higher = more iterative reasoning.
- `proxy_config.require_thought_ops = true` and `min_thought_ops = 2`.
- If you see DB lock errors: `rm -f neurofed.db detail.log` before a fresh run.

### Code Quality
```bash
# Format code
cargo fmt

# Check code style
cargo clippy

# Run security audit
cargo audit

# Check for outdated dependencies
cargo outdated

# Generate documentation
cargo doc --open
```

### Investigate learning

Execute the following command to troubleshoot learning:
 rm -f neurofed.db detail.log && cargo build && timeout 180 target/debug/neuro-fed-node 2>&1 | tee output.log ; cat detail.log

### Guided replay benchmark

Use the dedicated learning benchmark binary and helper script to rerun specific HumanEval/GSM8K slices and collect plan vs canonical comparisons. Example workflow:
```bash
# Rerun only the targeted dataset, then export enriched CSV data
rm -f neurofed.db detail.log && \
  cargo build && \
  cargo run --bin learning_benchmark -- --study-paths study/human-eval/data/HumanEval.jsonl --output learning_feedback.csv --skip-run=false && \
  python scripts/collect_learning_feedback.py --log detail.log --output learning_feedback.csv
```
Adjust `--study-paths` (comma-separated) to focus on other subsets; guided replay will automatically trigger for HumanEval/48 and /72 when loss exceeds 150.

### Testing
```bash
# Run all tests
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests only
cargo test --test integration

# Run tests with specific features
cargo test --features web-ui

# Run tests in release mode
cargo test --release
```

### Linting and Static Analysis
```bash
# Run clippy with all features
cargo clippy --all-features -- -D warnings

# Run clippy with pedantic
cargo clippy -- -D clippy::pedantic

# Check for unsafe code
cargo grep unsafe

# Check for missing documentation
cargo doc --no-deps
```

## Development Guidelines

### Code Style
- Follow Rust standard conventions
- Use `rustfmt` for formatting
- Use `clippy` for linting
- Add documentation comments (`///`) for public APIs
- Add inline comments (`//`) for complex logic
- Keep files below 300 lines, to easy maintain them
- Prefer avoid lader effect of "if": do checks at the begging of function with early termination
- Follow SRP principle

### Error Handling
- Use `thiserror` for custom error types
- Implement `Display` and `Error` traits
- Use `Result<T, E>` for fallible operations
- Handle errors gracefully with meaningful messages

### Performance
- Use `candle-core` with hardware acceleration for tensor operations
- Implement proper memory management
- Use async/await for I/O operations
- Profile with `cargo flamegraph`

### Security
- Validate all input data
- Use secure random number generation
- Implement proper error handling
- Follow Rust security best practices

## Component Development

### Implementing New Components
1. Create new module in `src/` directory
2. Add module declaration to `src/lib.rs`
3. Implement core functionality
4. Add comprehensive tests
5. Update documentation
6. Add integration tests
7. Use TDD
8. Make sure high test coverage. Add coverage collection into test process

### Before Extending Architecture
1. Check whether a concept already exists in `config.rs`, `types.rs`, and `pc_types.rs`
2. Pick one canonical location for the concept and route callers there
3. Verify the runtime entrypoint actually composes the new component
4. Prefer explicit integration tests over adding more placeholder modules

### Adding Dependencies
1. Add to `[dependencies]` in Cargo.toml
2. Run `cargo update`
3. Test compatibility
4. Update documentation

### Feature Flags
- Use `#[cfg(feature = "feature_name")]` for conditional compilation
- Define features in `[features]` section of Cargo.toml
- Use `--features` flag when building

## Git Workflow

### Branching Strategy
- `main`: Production-ready code
- `develop`: Integration branch for new features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical bug fixes

### Commit Messages
```
feat(component): add new functionality
fix(component): resolve issue with description
docs(component): update documentation
refactor(component): improve code structure
perf(component): optimize performance
ci: update CI configuration
```

### Pull Requests
- Create from feature branch to develop
- Include comprehensive description
- Add tests for new functionality
- Ensure all checks pass
- Request review from team members

## Continuous Integration

### GitHub Actions
- **ci.yml**: Run tests and linting on all pushes
- **release.yml**: Build and publish releases
- **nightly.yml**: Test with nightly Rust

### CI Checks
- Rustfmt check
- Clippy linting
- Security audit
- Unit and integration tests
- Documentation generation

## Deployment

### Building Releases
```bash
# Build release binaries
cargo build --release

# Cross-compile for different platforms
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target aarch64-apple-darwin
cargo build --release --target x86_64-pc-windows-msvc

# Create installer packages
# (Scripts in scripts/ directory)
```

### Release Process
1. Update version in Cargo.toml
2. Update changelog
3. Tag release in git
4. Build release binaries
5. Create release on GitHub
6. Publish to package registries if applicable

## Documentation

### API Documentation
```bash
# Generate API docs
cargo doc --open

# Generate docs with private items
cargo doc --document-private-items --open
```

### User Documentation
- Update README.md with installation and usage instructions
- Update docs/ directory with comprehensive guides
- Include examples in examples/ directory

### Architecture Documentation
- Update architecture.md with system design
- Update equations.md with mathematical foundations
- Update api.md with public API documentation

## Performance Monitoring

### Profiling
```bash
# Generate flamegraph
cargo flamegraph

# Check memory usage
cargo memcheck

# Benchmark performance
cargo bench
```

### Metrics
- Use `metrics` crate for performance metrics
- Export to Prometheus for monitoring
- Include in web UI dashboard

## Security Considerations

### Code Security
- Use `cargo audit` regularly
- Follow Rust security best practices
- Validate all input data
- Use secure random number generation

### Data Security
- Encrypt sensitive data at rest
- Use secure communication protocols
- Implement proper access controls
- Regular security audits

## Community Guidelines

### Contributing
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Ensure all checks pass
5. Submit pull request
6. Address review comments

### Issue Reporting
- Use GitHub issues for bug reports
- Include detailed reproduction steps
- Add relevant logs and error messages
- Specify affected components

### Feature Requests
- Use GitHub issues for feature requests
- Include use cases and requirements
- Discuss with community
- Prioritize based on impact

This guide provides a comprehensive overview of the development process for NeuroFed Node, ensuring consistent, high-quality code and efficient collaboration.

## Development Best Practices

### Before Committing Code
1. Run `cargo check` to ensure no compilation errors
2. Run `cargo build` to verify the project builds correctly
3. Run `cargo test` to ensure all tests pass
4. Run `cargo clippy` to check for code style issues
5. Run `cargo fmt` to format the code
6. Run `cargo audit` to check for security vulnerabilities

### Development Workflow
1. Create a feature branch from `develop`
2. Make changes with comprehensive tests
3. Ensure all CI checks pass locally
4. Submit pull request to `develop`
5. Address review comments
6. Merge to `develop` and eventually to `main`

### Testing Strategy
- Unit tests for individual components
- Integration tests for component interactions
- End-to-end tests for complete workflows
- Performance tests for critical paths
- Security tests for vulnerabilities

### Code Review Guidelines
- Check for functionality correctness
- Verify code follows style guidelines
- Ensure comprehensive test coverage
- Review security implications
- Check for performance issues
- Verify documentation is up-to-date

## Predictive Coding Reasoning Roadmap (Phased)
High-level but engineer-actionable plan to make Predictive Coding the primary reasoning source (ThoughtOps become mandatory, not optional).

### Phase 1: Force Reasoning (Critical)
- Gate text output until at least one ThoughtOp has been emitted.
- Add two decoder modes: `MODE_REASONING` then `MODE_OUTPUT`.
- Test: `17 * 23` must emit `COMPUTE_MATH` before final answer.

### Phase 2: Make ThoughtOps Affect State (Very High)
- Introduce a State Engine that applies ThoughtOps to a real mutable state.
- Update loss: include `state_error` in addition to text error.
- Test: variable init/update yields correct state after the sequence.

### Phase 3: No-Shortcut Tasks (Very High)
- Train on tasks where direct output is impossible without ThoughtOps.
- Include arithmetic, symbolic transforms, and mini-programs.
- Dataset format must include `INPUT`, `THOUGHT`, `STATE`, `OUTPUT`.

### Phase 4: Split Reasoning vs Text (High)
- Separate reasoning tokens from text tokens (distinct channels).
- Enforce: ThoughtOps are not just decoded text.

### Phase 5: Reasoning-Weighted Loss (High)
- Total loss = `state_error + reasoning_error + text_error`.
- Penalize correct text with incorrect ThoughtOps/state.

### Phase 6: Multi-Step Reasoning (High)
- Allow variable-length ThoughtOp chains (N steps).
- Test with max/argmax-style tasks.

### Phase 7: Planning Layer (Medium)
- Add `PLAN / DECOMPOSE / REFINE` operations before execution ops.

### Phase 8: Anti-Cheat Metrics (High)
- Track `reasoning_usage_rate`, `state_accuracy`, `steps_per_task`.
- Penalize trivial or missing chains.

### Phase 9: Tool-Integrated Reasoning (Medium)
- Add external tool ops: `SYMPY_EVAL`, `Z3_SOLVE`.
- Loop: `THINK → ACT → OBSERVE → UPDATE`.
- Always compare tool result vs predicted state and backprop error.

### Required Tests
- ThoughtOp gating test (no output before reasoning).
- State Engine update test.
- Multi-step reasoning test.
- Tool-integrated validation test (SymPy/Z3).
