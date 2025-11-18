// lexc/src/executor/csv_compat.rs
// Compatibility utilities for Polars API changes.

use polars::prelude::LazyCsvReader;

/// Extension trait adding a `with_has_header` builder method for
/// versions of Polars where it was renamed to `has_header`.
pub trait WithHasHeaderExt {
    fn with_has_header(self, has_header: bool) -> Self;
}

impl WithHasHeaderExt for LazyCsvReader {
    fn with_has_header(self, has_header: bool) -> Self {
        self.has_header(has_header)
    }
}