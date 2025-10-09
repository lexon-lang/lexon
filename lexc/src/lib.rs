//! # LEXON Compiler Library
//!
//! Stability policy: Starting with 1.0, the public API exposed via `lexc::prelude` is considered stable under SemVer. Items not re-exported in `prelude` are internal and may change. Experimental items are exposed under the optional `experimental` feature flag via `lexc::experimental`.
//!
//! - Stable surface: use `use lexc::prelude::*;`
//! - Experimental surface: `--features experimental` and `use lexc::experimental::*;`
//! - Tracing: enable `--features otel` and set env `LEXON_OTEL=1` to initialize OpenTelemetry on startup.
//!
//! LEXON is a domain-specific language (DSL) designed for LLM-native programming.
//! This library provides the complete compilation pipeline from source code to execution.
//!
//! ## Architecture Overview
//!
//! The compiler follows a multi-stage pipeline:
//! 1. **Parsing**: Source code → Concrete Syntax Tree (CST)
//! 2. **HIR Building**: CST → High-level Intermediate Representation (HIR)
//! 3. **Semantic Analysis**: HIR → Semantically analyzed HIR
//! 4. **LexIR Generation**: HIR → LEXON Intermediate Representation (LexIR)
//! 5. **Optimization**: LexIR → Optimized LexIR
//! 6. **Execution**: LexIR → Runtime execution
//!
//! ## Core Modules
//!
//! - [`hir`] - High-level Intermediate Representation definitions
//! - [`lexir`] - LEXON Intermediate Representation definitions
//! - [`executor`] - Runtime execution engine
//! - [`runtime`] - Runtime system and environment
//! - [`semantic`] - Semantic analysis and type checking
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use lexc::{build_hir_from_cst, convert_hir_to_lexir, ExecutionEnvironment};
//!
//! // Basic compilation pipeline
//! let hir = build_hir_from_cst(cst)?;
//! let lexir = convert_hir_to_lexir(&hir)?;
//! let mut executor = ExecutionEnvironment::new();
//! executor.execute_program(&lexir)?;
//! ```

// Core language modules
pub mod hir;
pub mod lexir;
pub mod schema;
pub mod symbols;

// Compilation pipeline modules
pub mod hir_builder;
pub mod hir_to_lexir;
pub mod linter;
pub mod optimizer;
pub mod resolver;
pub mod semantic;

// Execution modules
pub mod executor;
pub mod lex_executor;
#[path = "runtime/mod.rs"]
pub mod runtime;

// Supporting modules
pub mod ask_processor;
pub mod plugin;
pub mod telemetry;

// Prelude: re-export of the stable public API surface
pub mod prelude;

// Experimental namespace: only compiled when feature is enabled
#[cfg(feature = "experimental")]
pub mod experimental;

// Public API exports for compilation pipeline
pub use hir_builder::build_hir_from_cst;
pub use hir_to_lexir::convert_hir_to_lexir;

// Core type re-exports
pub use hir::HirNode;
pub use lexir::LexProgram;
pub use schema::JsonSchema;

// Runtime system re-exports
pub use runtime::{Runtime, RuntimeConfig, RuntimeError, RuntimeValue};

// Executor re-exports for CLI and external usage
pub use executor::{ExecutionEnvironment, ExecutorConfig};
pub use lex_executor::{HybridExecutor, LexExecutor};
