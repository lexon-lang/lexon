#![no_main]

use libfuzzer_sys::fuzz_target;
use tree_sitter::Parser;

use lexc::hir_builder::build_hir_from_cst;
use lexc::hir_to_lexir::convert_hir_to_lexir;

extern "C" {
    fn tree_sitter_lexon() -> tree_sitter::Language;
}

fn setup_parser() -> Parser {
    let mut parser = Parser::new();
    let language = unsafe { tree_sitter_lexon() };
    let _ = parser.set_language(&language);
    parser
}

fuzz_target!(|data: &[u8]| {
    if let Ok(source) = std::str::from_utf8(data) {
        let mut parser = setup_parser();
        if let Some(tree) = parser.parse(source, None) {
            let root = tree.root_node();
            if let Ok(hir_nodes) = build_hir_from_cst(root, source) {
                let _ = convert_hir_to_lexir(&hir_nodes);
            }
        }
    }
});

