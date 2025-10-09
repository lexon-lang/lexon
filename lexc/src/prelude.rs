//! Prelude for the stable public API of the LEXON compiler/runtime.
//! Import this module to access the supported 1.x API surface.
//!
//! Example:
//! ```rust
//! use lexc::prelude::*;
//! ```

pub use crate::build_hir_from_cst;
pub use crate::convert_hir_to_lexir;

// Core type re-exports
pub use crate::hir::HirNode;
pub use crate::lexir::LexProgram;
pub use crate::schema::JsonSchema;

// Runtime system re-exports
pub use crate::runtime::{Runtime, RuntimeConfig, RuntimeError, RuntimeValue};

// Executor re-exports for CLI and external usage
pub use crate::executor::{ExecutionEnvironment, ExecutorConfig};
pub use crate::lex_executor::{HybridExecutor, LexExecutor};
