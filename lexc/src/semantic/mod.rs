// lexc/src/semantic/mod.rs
//
// Module for semantic analysis of Lexon programs

mod analyzer;
mod symbol_table;
mod type_resolver;

// Re-export the public API
pub use analyzer::{analyze_program, SemanticError};
pub use symbol_table::{DeclId, Declaration, SchemaFieldInfo, SymbolTable};
pub use type_resolver::{PrimitiveType, ResolvedType, TypeId};
