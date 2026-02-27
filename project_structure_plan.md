# NeuroFed Node Project Structure Plan

## Directory Layout
```
neuro-pc-node/
├── Cargo.toml              # Rust package configuration
├── src/
│   ├── main.rs              # Application entry point and main loop
│   ├── lib.rs               # Public API exports
│   ├── config.rs            # Configuration management
│   ├── persistence.rs       # SQLite database and state persistence
│   ├── node_loop.rs         # Main async processing loop
│   ├── ├── llama_ffi.rs     # FFI bindings to llama.cpp
│   ├── ├── pc_hierarchy.rs  # Pure Predictive Coding core
│   ├── ├── bootstrap.rs     # LLM distillation and initialization
│   ├── ├── nostr_federation.rs # Nostr protocol integration
│   ├── ├── web_ui/           # Optional web interface (Phase 3)
│   │   ├── mod.rs
│   │   └── handlers.rs
│   ├── ├── installer/        # One-click installer scripts
│   │   ├── mod.rs
│   │   └── scripts/
│   │       ├── install.sh
│   │       ├── install.ps1
│   │       └── install.bat
│   └── tests/
│       ├── integration.rs
│       └── unit/
│           ├── llama_ffi_tests.rs
│           └── pc_hierarchy_tests.rs
├── docs/
│   ├── architecture.md       # System architecture documentation
│   ├── equations.md          # Mathematical foundations
│   ├── api.md               # Public API documentation
│   └── installation.md      # Installation and setup guide
├── examples/
│   ├── basic_usage.rs
│   ├── federated_demo.rs
│   └── performance_bench.rs
├── resources/
│   ├── default_config.toml
│   ├── models/              # GGUF model storage
│   └── schemas/             # Database schemas
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
├── README.md
├── LICENSE
└── .gitignore
```

## Key Files and Their Purpose

### Core Components
- **llama_ffi.rs**: FFI bindings to llama.cpp for embeddings, decoding, and GGML operations
- **pc_hierarchy.rs**: Pure Predictive Coding implementation with Rao-Ballard equations
- **bootstrap.rs**: One-time distillation from frozen LLM to seed PC beliefs
- **nostr_federation.rs**: Nostr protocol integration for decentralized federation

### Application Structure
- **main.rs**: Async main loop handling user input, file watching, and Nostr events
- **config.rs**: TOML configuration management for learning parameters and trust settings
- **persistence.rs**: SQLite database for beliefs, delta history, and CRDT-style consistency

### Testing and Examples
- **tests/**: Integration and unit tests for all components
- **examples/**: Usage examples and performance benchmarks
- **docs/**: Comprehensive documentation including architecture, equations, and API

## Build Configuration

The Cargo.toml will include:
- Core dependencies: ndarray, nalgebra, tokio
- FFI to llama.cpp: libc
- Nostr protocol: nostr-sdk
- Configuration and persistence: config, sqlx
- Optional web UI: tauri (Phase 3)
- Release optimizations: LTO, codegen-units=1, panic=abort

## Development Workflow

1. **Phase 0**: Implement llama_ffi.rs + basic 3-level PC hierarchy
2. **Phase 1**: Add full hierarchy, selective learning, and Nostr federation
3. **Phase 2**: Implement trust clusters, zap rewards, and Blossom integration
4. **Phase 3**: Add web UI, μPC deeper hierarchies, and multi-modal support

This structure provides a clean separation of concerns while maintaining the minimal, single-binary philosophy of the project.