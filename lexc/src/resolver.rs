//! Name resolver for Lexon – builds the symbol table from HIR.
//!
//! Initial version (Sprint 8):
//!  * Creates modules based on `module` declarations.
//!  * Registers public functions and schemas in the table.
//!  * Processes imports: resolves base module and items if they exist.
//!  * Returns simple errors if symbols are not found.

use crate::hir::{HirImportDeclaration, HirNode};
use crate::symbols::{ModuleId, SymbolKind, SymbolTable, Visibility};
use std::collections::HashMap;

#[derive(Debug)]
pub enum ResolveError {
    SymbolNotFound {
        module_path: Vec<String>,
        name: String,
    },
    ModuleNotFound {
        module_path: Vec<String>,
    },
    TraitNotFound {
        trait_name: String,
    },
    ImplMissingMethod {
        trait_name: String,
        method_name: String,
    },
    ImplExtraMethod {
        trait_name: String,
        method_name: String,
    },
    DuplicateTypeParameter {
        item_name: String,
        param_name: String,
    },
    GenericArityMismatch {
        type_name: String,
        expected: usize,
        found: usize,
    },
}

pub type Result<T> = std::result::Result<T, ResolveError>;

/// Builds the symbol table for a HIR file.
///
/// For now assumes all nodes belong to a single file and,
/// optionally, to a module declared with `module path;`.
pub fn build_symbol_table(hir_nodes: &[HirNode]) -> (SymbolTable, Vec<ResolveError>) {
    let mut table = SymbolTable::new();
    let mut errors: Vec<ResolveError> = Vec::new();

    // Root module (id=0) is created implicitly when declaring any empty path.
    let mut current_module: ModuleId = table.declare_module(Vec::new());

    // First pass: detect module declaration (if any)
    for node in hir_nodes {
        if let HirNode::ModuleDeclaration(decl) = node {
            current_module = table.declare_module(decl.path.clone());
            break;
        }
    }

    // Second pass: register own symbols
    for node in hir_nodes {
        match node {
            HirNode::FunctionDefinition(func_def) => {
                let vis = match func_def.visibility {
                    crate::hir::HirVisibility::Public => Visibility::Public,
                    crate::hir::HirVisibility::Private => Visibility::Private,
                };
                table.declare_symbol(
                    current_module,
                    func_def.name.clone(),
                    SymbolKind::Function,
                    vis,
                );

                // Validate duplicate type parameters
                {
                    use std::collections::HashSet;
                    let mut seen: HashSet<&str> = HashSet::new();
                    for tp in &func_def.type_parameters {
                        if !seen.insert(tp) {
                            errors.push(ResolveError::DuplicateTypeParameter {
                                item_name: func_def.name.clone(),
                                param_name: tp.clone(),
                            });
                        }
                    }
                }
            }
            HirNode::SchemaDefinition(schema_def) => {
                let vis = match schema_def.visibility {
                    crate::hir::HirVisibility::Public => Visibility::Public,
                    crate::hir::HirVisibility::Private => Visibility::Private,
                };
                table.declare_symbol(
                    current_module,
                    schema_def.name.clone(),
                    SymbolKind::Schema,
                    vis,
                );

                // Validate duplicate type parameters
                {
                    use std::collections::HashSet;
                    let mut seen: HashSet<&str> = HashSet::new();
                    for tp in &schema_def.type_parameters {
                        if !seen.insert(tp) {
                            errors.push(ResolveError::DuplicateTypeParameter {
                                item_name: schema_def.name.clone(),
                                param_name: tp.clone(),
                            });
                        }
                    }
                }
            }
            HirNode::TraitDefinition(trait_def) => {
                let vis = match trait_def.visibility {
                    crate::hir::HirVisibility::Public => Visibility::Public,
                    crate::hir::HirVisibility::Private => Visibility::Private,
                };
                table.declare_symbol(
                    current_module,
                    trait_def.name.clone(),
                    SymbolKind::Trait,
                    vis,
                );

                // Validate duplicate type parameters
                {
                    use std::collections::HashSet;
                    let mut seen: HashSet<&str> = HashSet::new();
                    for tp in &trait_def.type_parameters {
                        if !seen.insert(tp) {
                            errors.push(ResolveError::DuplicateTypeParameter {
                                item_name: trait_def.name.clone(),
                                param_name: tp.clone(),
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // New pass: validate impl vs traits
    // 1. Build map of trait_name -> Vec<method signatures>
    let mut trait_map: HashMap<String, Vec<(String, Option<String>)>> = HashMap::new();
    for node in hir_nodes {
        if let HirNode::TraitDefinition(trait_def) = node {
            let sigs = trait_def
                .methods
                .iter()
                .map(|m| (m.name.clone(), m.return_type.clone()))
                .collect();
            trait_map.insert(trait_def.name.clone(), sigs);
        }
    }

    // 2. Validate each impl block
    for node in hir_nodes {
        if let HirNode::ImplBlock(impl_block) = node {
            let trait_name = &impl_block.target;
            match trait_map.get(trait_name) {
                Some(expected_methods) => {
                    // Build set for impl methods
                    use std::collections::HashSet;
                    let mut impl_set: HashSet<&str> = HashSet::new();
                    for m in &impl_block.methods {
                        impl_set.insert(m.name.as_str());
                    }
                    // Check missing methods
                    for (name, _ret) in expected_methods {
                        if !impl_set.contains(name.as_str()) {
                            errors.push(ResolveError::ImplMissingMethod {
                                trait_name: trait_name.clone(),
                                method_name: name.clone(),
                            });
                        }
                    }
                    // Check extra methods
                    for m in &impl_block.methods {
                        if !expected_methods.iter().any(|(n, _)| n == &m.name) {
                            errors.push(ResolveError::ImplExtraMethod {
                                trait_name: trait_name.clone(),
                                method_name: m.name.clone(),
                            });
                        }
                    }
                }
                None => {
                    errors.push(ResolveError::TraitNotFound {
                        trait_name: trait_name.clone(),
                    });
                }
            }
        }
    }

    // Third pass: process imports (simple check)
    for node in hir_nodes {
        if let HirNode::ImportDeclaration(import_decl) = node {
            if let Err(e) = process_import(import_decl, current_module, &table) {
                errors.push(e);
            }
        }
    }

    // Build type -> arity map from schemas and traits
    let mut arity_map: HashMap<String, usize> = HashMap::new();
    for node in hir_nodes {
        match node {
            HirNode::SchemaDefinition(schema_def) => {
                arity_map.insert(schema_def.name.clone(), schema_def.type_parameters.len());
            }
            HirNode::TraitDefinition(trait_def) => {
                arity_map.insert(trait_def.name.clone(), trait_def.type_parameters.len());
            }
            _ => {}
        }
    }

    // Helper to parse a type name string into (base, arg_count)
    fn parse_type(type_str: &str) -> (&str, usize) {
        if let Some(start) = type_str.find('<') {
            let base = &type_str[..start];
            let inner = &type_str[start + 1..type_str.len() - 1];
            let count = inner.split(',').filter(|s| !s.trim().is_empty()).count();
            (base, count)
        } else {
            (type_str, 0)
        }
    }

    // Fourth pass: validate generic type usage (variable decl and schema fields)
    for node in hir_nodes {
        match node {
            HirNode::VariableDeclaration(var_decl) => {
                if let Some(type_name) = &var_decl.type_name {
                    let (base, arg_count) = parse_type(type_name);
                    if let Some(&expected_arity) = arity_map.get(base) {
                        if arg_count != expected_arity {
                            errors.push(ResolveError::GenericArityMismatch {
                                type_name: base.to_string(),
                                expected: expected_arity,
                                found: arg_count,
                            });
                        }
                    }
                }
            }
            HirNode::SchemaDefinition(schema_def) => {
                for field in &schema_def.fields {
                    let (base, arg_count) = parse_type(&field.type_name);
                    if let Some(&expected_arity) = arity_map.get(base) {
                        if arg_count != expected_arity {
                            errors.push(ResolveError::GenericArityMismatch {
                                type_name: base.to_string(),
                                expected: expected_arity,
                                found: arg_count,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
    }

    (table, errors)
}

/// Process an import declaration.
fn process_import(
    import: &HirImportDeclaration,
    _current_module: ModuleId,
    table: &SymbolTable,
) -> Result<()> {
    // Check if the base module exists
    if table.module_id(&import.path).is_none() {
        return Err(ResolveError::ModuleNotFound {
            module_path: import.path.clone(),
        });
    }

    // If specific items are imported, check if they exist
    if !import.items.is_empty() {
        let module_id = table.module_id(&import.path).unwrap();
        for (item_name, _alias) in &import.items {
            if table.resolve_in_module(module_id, item_name).is_none() {
                return Err(ResolveError::SymbolNotFound {
                    module_path: import.path.clone(),
                    name: item_name.clone(),
                });
            }
        }
    }

    Ok(())
}
