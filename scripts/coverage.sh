#!/usr/bin/env bash
set -euo pipefail

# Install cargo-llvm-cov if missing
if ! command -v cargo-llvm-cov >/dev/null 2>&1; then
  cargo install cargo-llvm-cov >/dev/null
fi

# Generate coverage for workspace
cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --lcov --output-path /tmp/lexon_coverage.lcov
echo "/tmp/lexon_coverage.lcov"


