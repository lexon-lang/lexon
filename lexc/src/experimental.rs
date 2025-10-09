//! Experimental, unstable APIs.
//! This module is only compiled when built with `--features experimental`.
//! Backwards compatibility is not guaranteed across minor versions.
//!
//! Current experimental areas include:
//! - Multioutput helpers and related runtime plumbing
//! - Data pipeline helpers like `zip` and `flatten`
//! - Potential extended memory/indexing strategies
//!
//! Note: Many of these are DSL-level constructs implemented inside the runtime executor.
//! They are documented here for tracking; gating at the DSL level is planned.

// Re-export selected experimental markers or helpers if/when available.
// pub use crate::executor::experimental::*; // placeholder for future split
