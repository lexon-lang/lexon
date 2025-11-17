# Lexon fuzzing guide

This folder contains [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) targets that stress the parser/HIR builder/LexIR converter. The setup mirrors the instructions that were previously in the root docs.

## Targets

| Target            | Description                                                                |
|-------------------|----------------------------------------------------------------------------|
| `parser_cst`      | Feeds arbitrary UTF‑8 into the tree-sitter parser (CST only).              |
| `hir_builder`     | Parses the input and runs `build_hir_from_cst` to exercise the HIR layer.  |
| `hir_to_lexir`    | End-to-end CST → HIR → LexIR conversion (ignores semantic/execution).      |

All targets reuse the same helper to load the tree-sitter grammar via the `lexc` crate.

## Prerequisites

```bash
cargo install cargo-fuzz
```

`cargo fuzz` requires nightly; the command will download toolchains automatically.

## Running

From `v1.0.0-rc.1/`:

```bash
cargo fuzz list
# parser_cst
# hir_builder
# hir_to_lexir

cargo fuzz run parser_cst -- -max_total_time=60
cargo fuzz run hir_builder -- -max_total_time=60
cargo fuzz run hir_to_lexir -- -max_total_time=60
```

The `-max_total_time` flag keeps local runs bounded. Artifacts/minimized inputs are written under `fuzz/artifacts/<target>/`.

## Notes

- When a crash is found, add the minimized input to `fuzz/corpus/<target>/` and consider turning it into a proper unit/golden test.
- The targets intentionally ignore parser/HIR errors; the goal is to trigger panics or `unsafe` violations, not to enforce semantic correctness.
- You can pass `LEXON_LOG=debug` (or `LEXON_VERBOSE=1`) if you need runtime logs while reproducing a specific crash.
# Fuzzing guide (optional)

Targets a robust GA by fuzzing Lexon’s frontend and converter layers. We recommend `cargo-fuzz` (libFuzzer).

## Setup
```bash
cargo install cargo-fuzz
cd v1.0.0-rc.1
cargo fuzz init
```

Create three fuzz targets:
- `fuzz_targets/parser_cst.rs`: feeds random input into tree-sitter parser (CST).
- `fuzz_targets/hir_builder.rs`: takes UTF-8 input, parses to CST, then builds HIR.
- `fuzz_targets/hir_to_lexir.rs`: takes an existing HIR JSON corpus and converts to LexIR.

Example skeleton (parser_cst.rs):
```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use tree_sitter::Parser;
extern "C" { fn tree_sitter_lexon() -> tree_sitter::Language; }

fuzz_target!(|data: &[u8]| {
    if let Ok(txt) = std::str::from_utf8(data) {
        let mut p = Parser::new();
        unsafe { p.set_language(&tree_sitter_lexon()).ok(); }
        let _ = p.parse(txt, None);
    }
});
```

Run fuzz:
```bash
cargo fuzz run parser_cst -- -max_total_time=60
```

Artifacts and minimized cases will be stored under `fuzz/`. Integrate found cases into tests/goldens where meaningful.



