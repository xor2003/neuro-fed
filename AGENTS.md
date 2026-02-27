# NeuroFed Node Development Guide

## Project Overview
NeuroFed (NeuroFed) Node is a decentralized federated AGI system based on pure hierarchical predictive coding. It implements a biologically plausible, fully decentralized, offline-first federated AGI system using Rust, candle framework, and Nostr protocol.

## Directory Structure
```
neuro-pc-node/
├── src/
│   ├── main.rs              # Application entry point and main loop
│   ├── lib.rs               # Public API exports
│   ├── config.rs            # Configuration management
│   ├── persistence.rs       # SQLite database and state persistence
│   ├── node_loop.rs         # Main async processing loop
│   ├── ml_engine.rs         # ML Engine using candle framework for pure Rust CPU/GPU operations
│   ├── pc_hierarchy.rs      # Pure Predictive Coding core
│   ├── bootstrap.rs         # LLM distillation and initialization
│   ├── nostr_federation.rs  # Nostr protocol integration
│   ├── openai_proxy.rs      # OpenAI API transparent proxy with local fallback
│   ├── web_ui/              # Optional web interface (Phase 3)
│   │   ├── mod.rs
│   │   └── handlers.rs
│   ├── installer/           # One-click installer scripts
│   │   ├── mod.rs
│   │   └── scripts/
│   │       ├── install.sh
│   │       ├── install.ps1
│   │       └── install.bat
│   └── tests/
│       ├── integration.rs
│       └── unit/
│           ├── ml_engine_tests.rs
│           ├── pc_hierarchy_tests.rs
│           ├── bootstrap_tests.rs
│           ├── nostr_federation_tests.rs
│           ├── openai_proxy_tests.rs
│           └── node_loop_tests.rs
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
- **src/persistence.rs**: SQLite database and state persistence
- **src/node_loop.rs**: Main async processing loop
- **src/ml_engine.rs**: ML Engine using candle framework for pure Rust CPU/GPU operations
- **src/pc_hierarchy.rs**: Pure Predictive Coding implementation
- **src/bootstrap.rs**: LLM distillation and initialization
- **src/nostr_federation.rs**: Nostr protocol integration
- **src/openai_proxy.rs**: OpenAI API transparent proxy with local fallback

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