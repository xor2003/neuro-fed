# PLAN2: Deterministic Execution Plan for NeuroFed Node

## Purpose

This plan converts the most useful current ideas for this project into a deterministic implementation sequence. It is intentionally narrower and more operational than `PLAN.md`.

It focuses on work that is both:

- scientifically useful for a predictive-coding system in 2026
- directly actionable in the current Rust codebase under `src/`

It explicitly avoids adding generic "optimizers" into the proxy path. Proxy behavior may consume outputs from these improvements, but the proxy is not the implementation target.

## Determinism Rules

Every task in this plan must satisfy all of the following:

1. It has a concrete scope in the current repository.
2. It has a deterministic Definition of Done.
3. It has at least one explicit verification command.
4. It avoids hidden success criteria such as "feels better" or "looks smarter".
5. It leaves the runtime in a working state after completion.

## Priority Scale

- `P0`: blocks correctness, startup, or basic usability
- `P1`: high leverage for learning quality or architecture integrity
- `P2`: important capability expansion with bounded risk
- `P3`: useful after the higher-priority work is stable

## Complexity Scale

- `S`: 0.5 to 1 day
- `M`: 1 to 3 days
- `L`: 3 to 7 days
- `XL`: more than 1 week

## Research Inputs Chosen For This Plan

The following recent ideas are worth carrying into implementation planning:

1. `Meta-PCN` style stabilization
   Reason: directly addresses deep predictive-coding instability, exploding or vanishing prediction errors, and scaling beyond toy loops.

2. `Bidirectional Predictive Coding`
   Reason: fits this codebase better than pure top-down-only control because the project already mixes generation, analysis, and task execution.

3. `Desegregation of predictive processing`
   Reason: the code should not hard-code the assumption that prediction units and error units must be strictly separated into different populations or modules.

4. Lean context and memory discipline for agents
   Reason: useful for retrieval, memory, benchmarking, and local assistant quality, but should be applied in memory, retrieval, and orchestration layers, not shoved into the proxy.

## Non-Goals For PLAN2

- No rewrite of the project around a new ML stack.
- No claim of Apple NPU/ANE support unless there is a real backend and runtime verification.
- No OpenAI proxy token optimizer work.
- No vague "AGI improvements" without code-level acceptance criteria.

## Workstream Overview

This plan is organized into six deterministic workstreams:

1. backend and first-run usability
2. predictive-coding stability and observability
3. bidirectional reasoning path
4. memory and retrieval efficiency
5. benchmark and dataset hardening
6. documentation and operator clarity

---


## Workstream 2: Predictive-Coding Stability And Observability

### Task 2.1: Add meta-prediction-error instrumentation

- Priority: `P1`
- Complexity: `M`
- Why it matters:
  `Meta-PCN`-style stabilization is the most immediately useful research direction for this project because it gives measurable control over unstable error dynamics.
- Scope:
  - `src/pc_hierarchy.rs`
  - `src/pc_level.rs`
  - `src/pc_types.rs`
  - `src/bin/learning_benchmark.rs`
- Required changes:
  - track per-layer prediction error magnitude over time
  - compute secondary metrics:
    - error variance
    - error explosion events
    - error collapse events
  - export them into benchmark output or CSV
- Definition of Done:
  - benchmark output includes per-level prediction-error statistics
  - at least one regression test covers metric emission and formatting
  - metrics are deterministic for fixed inputs
- Verification:
  - `cargo run --bin learning_benchmark -- --skip-run`
  - inspect `learning_feedback.csv`
  - `cargo test --lib`

### Task 2.2: Add bounded stabilization controls to PC learning

- Priority: `P1`
- Complexity: `L`
- Why it matters:
  Observing instability is not enough; the hierarchy needs bounded controls to prevent divergence.
- Scope:
  - `src/pc_hierarchy.rs`
  - `src/pc_level.rs`
  - `src/config.rs`
  - `src/pc_types.rs`
- Required changes:
  - add configurable stabilization knobs:
    - prediction-error clipping
    - per-level learning-rate scaling
    - variance-aware weight regularization
    - optional precision floor and ceiling
  - default values must be conservative and deterministic
  - configuration must live in canonical config types, not ad-hoc duplicates
- Definition of Done:
  - all stabilization controls are configurable from runtime config
  - benchmark runs show no new exploding-loss regression on existing replay datasets
  - defaults preserve current behavior within an acceptable bounded drift
- Verification:
  - `cargo run --bin learning_benchmark -- --skip-run`
  - `cargo test --lib`
  - compare before and after `learning_feedback.csv`

### Task 2.3: Add learning-gate regression thresholds

- Priority: `P1`
- Complexity: `M`
- Why it matters:
  The project already has a learning gate culture, but it needs deterministic fail conditions instead of manual interpretation only.
- Scope:
  - `src/bin/learning_benchmark.rs`
  - `tests/`
  - optional helper script under `scripts/`
- Required changes:
  - define explicit benchmark regression thresholds for:
    - average loss
    - error explosion count
    - reasoning usage rate where applicable
  - fail the benchmark or test run when thresholds are violated
- Definition of Done:
  - there is at least one automated failure path for a controlled regression
  - threshold values are checked into the repo and documented
  - output explains which threshold failed
- Verification:
  - `cargo test --lib`
  - `cargo run --bin learning_benchmark -- --skip-run`

---

## Workstream 3: Bidirectional Reasoning Path

### Task 3.1: Separate top-down prediction and bottom-up evidence flow in code

- Priority: `P1`
- Complexity: `L`
- Why it matters:
  The bidirectional predictive-coding direction is useful here because the node has to both predict and interpret. The code should make those directions explicit instead of implicit.
- Scope:
  - `src/pc_hierarchy.rs`
  - `src/pc_decoder.rs`
  - `src/reasoning_state.rs`
  - `src/types.rs` or canonical replacement if needed
- Required changes:
  - represent downward prediction flow separately from upward error or evidence flow
  - keep interfaces explicit about which direction data is moving
  - avoid a single mixed update path where the semantics are hidden
- Definition of Done:
  - core update functions clearly distinguish prediction propagation from evidence propagation
  - tests cover both directions independently
  - no public API implies that bottom-up and top-down updates are interchangeable
- Verification:
  - `cargo test --lib`
  - targeted unit tests for directional update functions

### Task 3.2: Add a bounded bidirectional inference mode

- Priority: `P2`
- Complexity: `L`
- Why it matters:
  This creates a practical path to compare current behavior against a more biologically and algorithmically useful scheme without forcing a rewrite.
- Scope:
  - `src/pc_hierarchy.rs`
  - `src/config.rs`
  - `src/bin/learning_benchmark.rs`
- Required changes:
  - add a config-gated bidirectional inference mode
  - preserve the current simpler mode as baseline
  - benchmark both modes side by side
- Definition of Done:
  - two modes exist: baseline and bidirectional
  - both modes are runnable from config without code edits
  - benchmark output identifies which mode produced each result
- Verification:
  - `cargo run --bin learning_benchmark -- --skip-run`
  - compare logs for baseline versus bidirectional mode

### Task 3.3: Avoid hard-coded segregated error-unit assumptions

- Priority: `P2`
- Complexity: `M`
- Why it matters:
  Recent neuroscience evidence suggests strict prediction-unit versus error-unit segregation is too rigid as a universal assumption.
- Scope:
  - `src/pc_level.rs`
  - `src/pc_types.rs`
  - relevant tests
- Required changes:
  - ensure the internal representation allows mixed or shared pathways where needed
  - document which parts of the current model are engineering abstractions rather than biological claims
- Definition of Done:
  - no code comment or config field states that strict segregation is required unless it is actually enforced for a benchmark mode
  - at least one test covers a mixed-signal path
- Verification:
  - `cargo test --lib`
  - manual review of public comments and config docs

---

## Workstream 4: Memory And Retrieval Efficiency

### Task 4.1: Replace unbounded replay context with bounded summaries plus retrieval

- Priority: `P1`
- Complexity: `L`
- Why it matters:
  The most applicable token-efficiency ideas for this project are not prompt hacks. They are bounded state, selective retrieval, and compact memory reuse.
- Scope:
  - `src/semantic_cache.rs`
  - `src/persistence.rs`
  - `src/node_loop.rs`
  - `src/ui/mod.rs` if memory hits are surfaced there
- Required changes:
  - store compact episode summaries
  - retrieve top-k relevant memories instead of replaying broad history
  - keep selection deterministic for fixed embeddings and fixed k
- Definition of Done:
  - memory retrieval is bounded by explicit limits
  - an interaction does not rehydrate entire historical traces by default
  - retrieval logic can explain which memories were selected and why
- Verification:
  - `cargo test --lib`
  - add deterministic tests for retrieval ranking and truncation

### Task 4.2: Add structured compact memory records

- Priority: `P2`
- Complexity: `M`
- Why it matters:
  Compact structured memory is lower risk and higher value than free-form transcript accumulation.
- Scope:
  - `src/persistence.rs`
  - `src/types.rs` or canonical memory type location
  - migration logic if needed
- Required changes:
  - define a compact stored record with fields such as:
    - task kind
    - touched areas
    - evidence summary
    - verification summary
    - residual risk
  - keep text bounded per field
- Definition of Done:
  - a stable schema exists for compact memory records
  - records are written and read successfully
  - field lengths are bounded in code, not only by convention
- Verification:
  - `cargo test --lib`
  - persistence round-trip test

### Task 4.3: Add lazy-loading rules for heavy observations

- Priority: `P2`
- Complexity: `M`
- Why it matters:
  Heavy logs, traces, and raw documents should be fetched by slice, not blindly loaded into working memory.
- Scope:
  - `src/node_loop.rs`
  - `src/bootstrap.rs`
  - any log or dataset loading path that currently reads full payloads eagerly
- Required changes:
  - fetch large data in slices or chunks
  - summarize before promotion into active reasoning state
  - cap active observation size per step
- Definition of Done:
  - at least one heavy-input path is converted from eager full-load to bounded slice loading
  - active observation size limit is enforced in code
- Verification:
  - `cargo test --lib`
  - targeted test for chunked loading behavior

---

## Workstream 5: Benchmark And Dataset Hardening

### Task 5.1: Expand replay logs to include structured reasoning quality fields

- Priority: `P1`
- Complexity: `M`
- Why it matters:
  The roadmap already wants structured assistant behavior. The benchmark needs explicit fields, not just final loss numbers.
- Scope:
  - `src/bin/learning_benchmark.rs`
  - `learning_feedback.csv` generation
  - helper scripts under `scripts/` if needed
- Required changes:
  - record structured fields such as:
    - reasoning steps count
    - plan presence
    - verification presence
    - residual-risk presence
  - keep output machine-readable
- Definition of Done:
  - CSV or JSONL output includes these fields
  - tests verify the fields are emitted
- Verification:
  - `cargo run --bin learning_benchmark -- --skip-run`
  - inspect `learning_feedback.csv`

### Task 5.2: Add benchmark slices for new PC stabilization modes

- Priority: `P2`
- Complexity: `M`
- Why it matters:
  New stabilization logic needs a stable comparison harness or it will drift into anecdotal tuning.
- Scope:
  - `study/`
  - `src/bin/learning_benchmark.rs`
  - tests or scripts for selection
- Required changes:
  - add small deterministic benchmark subsets targeting:
    - multi-step arithmetic
    - state updates
    - reasoning replay
    - recovery from high-surprise examples
- Definition of Done:
  - benchmark slices are checked in
  - they run in a bounded local developer workflow
  - results are attributable to a named slice
- Verification:
  - run the benchmark on each new slice
  - confirm slice-specific output in logs or CSV

### Task 5.3: Add benchmark mode comparison output

- Priority: `P2`
- Complexity: `S`
- Why it matters:
  When baseline and bidirectional or stabilized modes coexist, side-by-side comparison becomes necessary.
- Scope:
  - `src/bin/learning_benchmark.rs`
- Required changes:
  - emit the active mode name and key config toggles in benchmark output
  - make CSV rows self-describing
- Definition of Done:
  - a single exported row identifies the model and reasoning mode that produced it
- Verification:
  - `cargo run --bin learning_benchmark -- --skip-run`

---

## Workstream 6: Documentation And Operator Clarity

### Task 6.1: Align documentation with actual runtime behavior

- Priority: `P1`
- Complexity: `M`
- Why it matters:
  This repository currently has some architecture ambition beyond what `main.rs` actually wires up. The docs should make runtime truth obvious.
- Scope:
  - `README.md`
  - `docs/architecture.md`
  - `AGENTS.md`
- Required changes:
  - separate "implemented now" from "planned"
  - document actual startup path and model/device behavior
  - document unsupported hardware claims clearly
- Definition of Done:
  - a new contributor can identify:
    - what runs today
    - what is experimental
    - what is not implemented
  - no document claims NPU support unless code and verification exist
- Verification:
  - manual review of docs against `src/main.rs`

### Task 6.2: Document deterministic development gates

- Priority: `P2`
- Complexity: `S`
- Why it matters:
  The project benefits from strict gates, but they should be easier to follow without interpretation gaps.
- Scope:
  - `AGENTS.md`
  - optional helper script docs
- Required changes:
  - list the exact commands for:
    - learning gate
    - smoke tests
    - backend verification
    - model startup verification
  - define what counts as pass or fail
- Definition of Done:
  - the gate commands are copy-pasteable
  - pass criteria are not ambiguous
- Verification:
  - manual review

---

## Execution Order

Implement in this order unless a blocking bug forces a local reorder:

1. Task 1.1
2. Task 1.2
3. Task 1.3
4. Task 2.1
5. Task 2.2
6. Task 2.3
7. Task 3.1
8. Task 5.3
9. Task 3.2
10. Task 3.3
11. Task 4.1
12. Task 4.2
13. Task 4.3
14. Task 5.1
15. Task 5.2
16. Task 6.1
17. Task 6.2

Rationale:

- first remove confusion and breakage in startup, device use, and model acquisition
- then stabilize the predictive-coding core before expanding architecture
- then improve memory efficiency and benchmark structure
- finally tighten docs around what is now true

## Minimum Deliverables Per Completed Task

Each completed task must leave behind all of the following:

1. code or docs committed in the repository
2. tests or explicit verification commands
3. a short note in commit message or task log describing:
   - what changed
   - what was verified
   - residual risk

## Definition Of Plan Success

`PLAN2.md` is considered successfully executed when all `P0` and `P1` items are complete and all of the following are true:

1. startup tells the user exactly which backend is active
2. first-run model download is resilient and low-noise
3. predictive-coding error dynamics are observable and bounded
4. bidirectional reasoning exists as an explicit, benchmarkable mode
5. memory and replay paths are bounded and retrieval-based rather than history-dump based
6. docs match runtime reality

## Explicit Deferred Items

These are intentionally not part of this plan unless later evidence changes priorities:

- Apple NPU/ANE backend work
- full multi-node federation redesign
- large UI redesign
- proxy-side token optimization features
- replacing candle with another ML runtime
