use std::path::PathBuf;

fn main() {
    let grammar_dir = PathBuf::from("..").join("tree-sitter-lexon").join("src");
    let parser_file = grammar_dir.join("parser.c");
    let scanner_file = grammar_dir.join("scanner.c");

    println!("cargo:rerun-if-changed={}", parser_file.display());
    if scanner_file.exists() {
        println!("cargo:rerun-if-changed={}", scanner_file.display());
    }
    println!("cargo:rerun-if-changed=build.rs");

    let mut build = cc::Build::new();
    build.file(&parser_file).include(&grammar_dir);
    if scanner_file.exists() {
        build.file(&scanner_file);
    }
    build.warnings(false);
    build.compile("tree_sitter_lexon_parser");
}
