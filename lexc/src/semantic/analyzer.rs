// lexc/src/semantic/analyzer.rs
//
// Semantic analysis implementation for Lexon programs

use super::symbol_table::{Declaration, SchemaFieldInfo, SymbolTable};
use super::type_resolver::TypeId;
use crate::hir::{
    HirAskExpression, HirFunctionDefinition, HirNode, HirSchemaDefinition, HirVariableDeclaration,
};

/// Errors that can occur during semantic analysis
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticError {
    NameAlreadyDeclared(String),
    NameNotFound(String),
    TypeNotFound(String),
    TypeMismatch { expected: String, found: String },
    SchemaFieldNotFound { schema: String, field: String },
    MissingRequiredField { schema: String, field: String },
    InvalidAskExpression(String),
    // More errors will be added as semantic analysis is expanded
}

/// Analyze a program's semantics and build a symbol table
pub fn analyze_program(program: &[HirNode]) -> Result<SymbolTable, Vec<SemanticError>> {
    let mut symbol_table = SymbolTable::new();
    let mut errors = Vec::new();

    // Initialize primitive types
    symbol_table.initialize_primitive_types();

    // First pass: Register all top-level declarations (for forward references)
    for node in program {
        match node {
            HirNode::SchemaDefinition(schema) => {
                register_schema(&mut symbol_table, schema, &mut errors);
            }
            HirNode::FunctionDefinition(func) => {
                register_function(&mut symbol_table, func, &mut errors);
            }
            _ => {} // Skip other nodes in the first pass
        }
    }

    // Second pass: Process schema fields after all schemas are registered
    for node in program {
        if let HirNode::SchemaDefinition(schema) = node {
            process_schema_fields(&mut symbol_table, schema, &mut errors);
        }
    }

    // Third pass: Analyze all declarations and expressions
    for node in program {
        match node {
            HirNode::VariableDeclaration(var_decl) => {
                analyze_variable_declaration(&mut symbol_table, var_decl, &mut errors);
            }
            HirNode::FunctionDefinition(func) => {
                analyze_function_body(&mut symbol_table, func, &mut errors);
            }
            _ => {} // Other nodes are handled in previous passes
        }
    }

    if errors.is_empty() {
        Ok(symbol_table)
    } else {
        Err(errors)
    }
}

// Helper functions for semantic analysis

fn register_schema(
    symbol_table: &mut SymbolTable,
    schema: &HirSchemaDefinition,
    errors: &mut Vec<SemanticError>,
) {
    // Just register the schema name in the first pass
    // Fields will be processed in the second pass after all schemas are registered

    let schema_decl = Declaration::Schema {
        name: schema.name.clone(),
        fields: Vec::new(), // Empty for now, will be filled in later
    };

    if let Err(e) = symbol_table.add_declaration(schema_decl) {
        errors.push(e);
    }
}

fn process_schema_fields(
    symbol_table: &mut SymbolTable,
    schema: &HirSchemaDefinition,
    errors: &mut Vec<SemanticError>,
) {
    // Find the schema declaration
    let schema_decl_id = match symbol_table.lookup_name(&schema.name) {
        Some(id) => id,
        None => {
            // This shouldn't happen if register_schema succeeded
            errors.push(SemanticError::NameNotFound(schema.name.clone()));
            return;
        }
    };

    // Create field info for each field
    let mut field_infos = Vec::new();

    for field in &schema.fields {
        // Resolve the field's type
        let type_id = match symbol_table.resolve_type_name(&field.type_name) {
            Ok(id) => id,
            Err(e) => {
                errors.push(e);
                continue;
            }
        };

        let field_info = SchemaFieldInfo {
            name: field.name.clone(),
            type_id,
            is_optional: field.is_optional,
            has_default: field.default_value.is_some(),
        };

        field_infos.push(field_info);
    }

    // Update the schema declaration with field info
    if !field_infos.is_empty() {
        if let Err(e) = symbol_table.update_schema_fields(schema_decl_id, field_infos) {
            errors.push(e);
        }
    }
}

fn register_function(
    symbol_table: &mut SymbolTable,
    func: &HirFunctionDefinition,
    errors: &mut Vec<SemanticError>,
) {
    // Resolve return type if specified
    let return_type_id = if let Some(type_name) = &func.return_type {
        match symbol_table.resolve_type_name(type_name) {
            Ok(id) => Some(id),
            Err(e) => {
                errors.push(e);
                None
            }
        }
    } else {
        None
    };

    let func_decl = Declaration::Function {
        name: func.name.clone(),
        return_type_id,
    };

    if let Err(e) = symbol_table.add_declaration(func_decl) {
        errors.push(e);
    }
}

fn analyze_variable_declaration(
    symbol_table: &mut SymbolTable,
    var_decl: &HirVariableDeclaration,
    errors: &mut Vec<SemanticError>,
) {
    // Resolve the variable's type
    let type_id = if let Some(type_name) = &var_decl.type_name {
        match symbol_table.resolve_type_name(type_name) {
            Ok(id) => Some(id),
            Err(e) => {
                errors.push(e);
                None
            }
        }
    } else {
        None
    };

    // Register the variable in the symbol table
    let var_decl = Declaration::Variable {
        name: var_decl.name.clone(),
        type_id,
        is_mutable: false, // Default to immutable, will be updated based on let/var
    };

    if let Err(e) = symbol_table.add_declaration(var_decl) {
        errors.push(e);
    }

    // Analyze the variable's initializer expression
    // TODO: Analyze expression & perform type checking
}

fn analyze_function_body(
    symbol_table: &mut SymbolTable,
    func: &HirFunctionDefinition,
    errors: &mut Vec<SemanticError>,
) {
    // Create a new scope for function body
    symbol_table.enter_scope();

    // Analyze each statement in the function body
    for stmt in &func.body {
        match stmt {
            HirNode::VariableDeclaration(var_decl) => {
                analyze_variable_declaration(symbol_table, var_decl, errors);
            }
            _ => {
                // Handle other statement types as they are added
            }
        }
    }

    // Exit the function scope
    symbol_table.exit_scope();
}

// Additional helpers for analyzing expressions (to be expanded)

#[allow(dead_code)]
fn analyze_ask_expression(
    symbol_table: &mut SymbolTable,
    ask_expr: &HirAskExpression,
    errors: &mut Vec<SemanticError>,
) -> Option<TypeId> {
    // Verify the referenced schema exists
    if let Some(schema_name) = &ask_expr.output_schema_name {
        match symbol_table.lookup_name(schema_name) {
            Some(decl_id) => {
                match symbol_table.get_declaration(decl_id) {
                    Some(Declaration::Schema { .. }) => {
                        // Schema exists, return its type
                        match symbol_table.resolve_type_name(schema_name) {
                            Ok(type_id) => Some(type_id),
                            Err(e) => {
                                errors.push(e);
                                None
                            }
                        }
                    }
                    _ => {
                        errors.push(SemanticError::TypeMismatch {
                            expected: "schema".to_string(),
                            found: "non-schema".to_string(),
                        });
                        None
                    }
                }
            }
            None => {
                errors.push(SemanticError::TypeNotFound(schema_name.clone()));
                None
            }
        }
    } else {
        errors.push(SemanticError::InvalidAskExpression(
            "Ask expression must specify an output schema".to_string(),
        ));
        None
    }
}
