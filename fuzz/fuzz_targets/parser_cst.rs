#![no_main]

use libfuzzer_sys::fuzz_target;
use tree_sitter::Parser;

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
        let _ = parser.parse(source, None);
    }
});
