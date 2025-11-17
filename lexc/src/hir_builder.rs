//! # HIR Builder - High-Level Intermediate Representation Construction
//!
//! This module provides the core functionality for converting Concrete Syntax Tree (CST) nodes
//! from Tree-sitter into High-Level Intermediate Representation (HIR) nodes. It serves as the
//! bridge between the parser and the semantic analysis phase of the Lexon compiler.
//!
//! ## Architecture Overview
//!
//! The HIR builder follows a recursive descent approach to convert CST nodes into HIR nodes:
//!
//! ```text
//! Tree-sitter CST → HIR Builder → HIR Nodes → Semantic Analysis
//! ```
//!
//! ## Key Components
//!
//! - **Expression Builder**: Converts CST expressions to HIR expressions
//! - **Statement Builder**: Converts CST statements to HIR statements
//! - **Type Inference**: Infers types from expressions when not explicitly provided
//! - **Error Handling**: Comprehensive error reporting with context
//! - **Ask Expression Support**: Specialized handling for LLM expressions
//! - **Anti-Hallucination Support**: Native support for ask_safe expressions
//!
//! ## Supported Language Constructs
//!
//! ### Expressions
//! - **Literals**: String, integer, float, boolean, array literals
//! - **LLM Expressions**: `ask()` and `ask_safe()` with full attribute support
//! - **Binary Operations**: Arithmetic, comparison, logical operations
//! - **Function Calls**: Regular and method calls with parameter inference
//! - **Data Operations**: `load`, `filter`, `select`, `take` operations
//! - **Control Flow**: `if`, `while`, `for`, `match` expressions
//!
//! ### Statements
//! - **Variable Declarations**: With type inference support
//! - **Function Definitions**: With parameter and return type analysis
//! - **Schema Definitions**: Type definitions with field validation
//! - **Module Declarations**: Module system support
//! - **Import Statements**: Dependency resolution
//! - **Trait Definitions**: Interface definitions
//!
//! ## Type Inference System
//!
//! The HIR builder includes a sophisticated type inference system that can:
//!
//! - Infer types from literal values
//! - Propagate types through binary expressions
//! - Handle function return type inference
//! - Support generic type parameters
//! - Provide fallback types for complex expressions
//!
//! ## Error Handling
//!
//! Comprehensive error handling with detailed context:
//!
//! - **Syntax Errors**: Malformed CST nodes
//! - **Type Errors**: Invalid type annotations
//! - **Missing Fields**: Required CST fields not present
//! - **Invalid Literals**: Malformed literal values
//! - **Unsupported Constructs**: Language features not yet implemented
//!
//! ## Usage Example
//!
//! ```rust
//! use lexon::hir_builder::build_hir_from_cst;
//! use tree_sitter::Parser;
//!
//! let mut parser = Parser::new();
//! parser.set_language(tree_sitter_lexon::language()).unwrap();
//!
//! let source_code = "let result = ask(\"What is 2 + 2?\")";
//! let tree = parser.parse(source_code, None).unwrap();
//! let hir_nodes = build_hir_from_cst(tree.root_node(), source_code)?;
//! ```
//!
//! ## Performance Considerations
//!
//! The HIR builder is designed for efficiency:
//!
//! - Single-pass conversion from CST to HIR
//! - Minimal memory allocation through careful use of references
//! - Lazy evaluation of complex type inference
//! - Optimized text extraction from CST nodes
//!
//! ## Anti-Hallucination Integration
//!
//! Native support for Lexon's anti-hallucination system:
//!
//! - **ask_safe expressions**: Automatic validation integration
//! - **Confidence scoring**: Built-in confidence metrics
//! - **Validation strategies**: Multiple validation approaches
//! - **Error recovery**: Graceful handling of validation failures

// ================================================================================================
// Dependencies and Imports
// ================================================================================================

use crate::ask_processor;
use crate::hir::{
    HirAskExpression, HirAssignment, HirAwait, HirBinaryExpression, HirBinaryOperator,
    HirDataFilter, HirDataLoad, HirDataSelect, HirDataTake, HirForIn, HirFunctionCall,
    HirFunctionDefinition, HirFunctionSignature, HirIf, HirImplBlock, HirImportDeclaration,
    HirLiteral, HirMatch, HirMatchArm, HirMethodCall, HirModuleDeclaration, HirNode, HirParameter,
    HirReturn, HirSchemaDefinition, HirSchemaField, HirTraitDefinition, HirTypeOf,
    HirVariableDeclaration, HirVisibility, HirWhile,
};
use tree_sitter::Node; // To use parse_ask_expression

#[derive(Debug, Clone, PartialEq)]
pub enum HirBuildError {
    UnsupportedNodeType(String),
    UnsupportedExpressionType(String),
    MissingField {
        node_type: String,
        field_name: String,
    },
    InvalidLiteralFormat {
        literal_kind: String,
        value: String,
    },
    CstParseError,
}

// Helper to extract text from a CST node
fn get_node_text(node: Node<'_>, source_code: &str) -> Option<String> {
    source_code
        .get(node.start_byte()..node.end_byte())
        .map(String::from)
}

// Helper to parse a generic_parameter_list CST node into Vec<String>
fn parse_generic_parameters(node: Node, source_code: &str) -> Vec<String> {
    let text = get_node_text(node, source_code).unwrap_or_default();
    if text.starts_with('<') && text.ends_with('>') {
        text[1..text.len() - 1]
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        Vec::new()
    }
}

/// Type inference helper function
/// Infers type from HIR expression for type inference support

fn parse_function_parameters(params_node: Node, source_code: &str) -> Vec<HirParameter> {
    let mut parameters = Vec::new();

    // Walk through parameter_list children
    let mut cursor = params_node.walk();
    for child in params_node.children(&mut cursor) {
        if child.kind() == "parameter" {
            let name_node = child.child_by_field_name("name");
            let type_node = child.child_by_field_name("type");

            if let (Some(name_n), Some(type_n)) = (name_node, type_node) {
                let param_name = get_node_text(name_n, source_code).unwrap_or_default();
                let param_type = get_node_text(type_n, source_code).unwrap_or_default();

                parameters.push(HirParameter {
                    name: param_name,
                    type_name: param_type,
                });
            }
        }
    }

    parameters
}

fn infer_type_from_expression(expr: &HirNode) -> Option<String> {
    match expr {
        HirNode::Literal(literal) => match literal {
            HirLiteral::String(_) | HirLiteral::MultiLineString(_) => Some("string".to_string()),
            HirLiteral::Integer(_) => Some("int".to_string()),
            HirLiteral::Float(_) => Some("float".to_string()),
            HirLiteral::Boolean(_) => Some("bool".to_string()),
            HirLiteral::Array(_) => Some("array".to_string()),
        },
        HirNode::Ask(_) => Some("string".to_string()), // Ask expressions return strings by default
        HirNode::FunctionCall(func_call) => {
            // Basic function type inference
            match func_call.function.as_str() {
                "load" => Some("Dataset".to_string()),
                "data_load" => Some("Dataset".to_string()),
                _ => Some("string".to_string()), // Default fallback
            }
        }
        HirNode::DataLoad(_) => Some("Dataset".to_string()),
        HirNode::DataFilter(_) => Some("Dataset".to_string()),
        HirNode::DataSelect(_) => Some("Dataset".to_string()),
        HirNode::DataTake(_) => Some("Dataset".to_string()),
        HirNode::Binary(binary_expr) => {
            // Infer type from binary expressions
            match binary_expr.operator {
                HirBinaryOperator::Add
                | HirBinaryOperator::Subtract
                | HirBinaryOperator::Multiply
                | HirBinaryOperator::Divide => {
                    // For arithmetic, try to infer from operands
                    if let Some(left_type) = infer_type_from_expression(&binary_expr.left) {
                        if left_type == "float" {
                            Some("float".to_string())
                        } else {
                            Some("int".to_string())
                        }
                    } else {
                        Some("int".to_string())
                    }
                }
                HirBinaryOperator::GreaterThan
                | HirBinaryOperator::LessThan
                | HirBinaryOperator::GreaterEqual
                | HirBinaryOperator::LessEqual
                | HirBinaryOperator::Equal
                | HirBinaryOperator::NotEqual
                | HirBinaryOperator::And
                | HirBinaryOperator::Or => Some("bool".to_string()),
            }
        }
        _ => None, // For complex expressions, we can't infer easily
    }
}

/// Helper function to get type name for variable declaration with type inference support
fn get_variable_type_name(
    type_node_opt: Option<Node>,
    value_expr: &HirNode,
    source_code: &str,
) -> Option<String> {
    if let Some(type_node) = type_node_opt {
        // Explicit type provided
        get_node_text(type_node, source_code)
    } else {
        // Type inference
        infer_type_from_expression(value_expr)
    }
}

// Determine if `child` node corresponds to an `arg` field inside `parent` using `field_name_for_child` (Tree-sitter 0.22).
fn is_arg(parent: &Node, child: &Node) -> bool {
    for i in 0..parent.child_count() {
        if let Some(c) = parent.child(i) {
            if c.id() == child.id() {
                return parent.field_name_for_child(i as u32) == Some("arg");
            }
        }
    }
    false
}

/// Parses a CST expression node into an HIR expression node.
fn build_hir_expression(expr_node: Node, source_code: &str) -> Result<HirNode, HirBuildError> {
    match expr_node.kind() {
        "ask_expression" => {
            match ask_processor::parse_ask_expression(expr_node, source_code) {
                Some(ask_data) => {
                    let hir_ask: HirAskExpression = ask_data.into();
                    Ok(HirNode::Ask(Box::new(hir_ask)))
                }
                None => {
                    // This indicates an internal error in ask_processor or an invalid ask_expression CST node
                    // that parse_ask_expression couldn't handle despite it being an 'ask_expression' kind.
                    eprintln!("Internal error: parse_ask_expression returned None for an ask_expression node: {:?}", expr_node.to_sexp());
                    Err(HirBuildError::UnsupportedExpressionType(
                        "Failed to process ask_expression".to_string(),
                    ))
                }
            }
        }
        // 🛡️ Support for ask_safe_expression - Anti-Hallucination System
        "ask_safe_expression" => {
            match ask_processor::parse_ask_safe_expression(expr_node, source_code) {
                Some(ask_safe_data) => {
                    let hir_ask_safe = ask_safe_data.into();
                    Ok(HirNode::AskSafe(Box::new(hir_ask_safe)))
                }
                None => {
                    eprintln!("Internal error: parse_ask_safe_expression returned None for an ask_safe_expression node: {:?}", expr_node.to_sexp());
                    Err(HirBuildError::UnsupportedExpressionType(
                        "Failed to process ask_safe_expression".to_string(),
                    ))
                }
            }
        }
        "string_literal" => {
            let text = get_node_text(expr_node, source_code).ok_or_else(|| {
                HirBuildError::InvalidLiteralFormat {
                    literal_kind: "string".to_string(),
                    value: "".to_string(),
                }
            })?;
            if text.starts_with("\"") && text.ends_with("\"") && text.len() >= 2 {
                Ok(HirNode::Literal(HirLiteral::String(
                    text[1..text.len() - 1].to_string(),
                )))
            } else {
                Err(HirBuildError::InvalidLiteralFormat {
                    literal_kind: "string".to_string(),
                    value: text,
                })
            }
        }
        "multiline_string_literal" => {
            let text = get_node_text(expr_node, source_code).ok_or_else(|| {
                HirBuildError::InvalidLiteralFormat {
                    literal_kind: "multiline_string".to_string(),
                    value: "".to_string(),
                }
            })?;
            if text.starts_with("\"\"\"") && text.ends_with("\"\"\"") && text.len() >= 6 {
                // Basic unquoting. Normalization (dedent, etc.) is now in ask_processor's get_literal_node_value_as_string.
                // If this function is called for multiline strings NOT in an ask_expression, we might need similar normalization here
                // or ensure get_literal_node_value_as_string is more general.
                // For now, just unquote.
                let inner_text = text[3..text.len() - 3].to_string();
                Ok(HirNode::Literal(HirLiteral::MultiLineString(inner_text)))
            } else {
                Err(HirBuildError::InvalidLiteralFormat {
                    literal_kind: "multiline_string".to_string(),
                    value: text,
                })
            }
        }
        "integer_literal" => {
            let text = get_node_text(expr_node, source_code).ok_or_else(|| {
                HirBuildError::InvalidLiteralFormat {
                    literal_kind: "integer".to_string(),
                    value: "".to_string(),
                }
            })?;
            text.parse::<i64>()
                .map(|val| HirNode::Literal(HirLiteral::Integer(val)))
                .map_err(|_| HirBuildError::InvalidLiteralFormat {
                    literal_kind: "integer".to_string(),
                    value: text,
                })
        }
        "float_literal" => {
            let text = get_node_text(expr_node, source_code).ok_or_else(|| {
                HirBuildError::InvalidLiteralFormat {
                    literal_kind: "float".to_string(),
                    value: "".to_string(),
                }
            })?;
            text.parse::<f64>()
                .map(|val| HirNode::Literal(HirLiteral::Float(val)))
                .map_err(|_| HirBuildError::InvalidLiteralFormat {
                    literal_kind: "float".to_string(),
                    value: text,
                })
        }
        "boolean_literal" => {
            let text = get_node_text(expr_node, source_code).ok_or_else(|| {
                HirBuildError::InvalidLiteralFormat {
                    literal_kind: "boolean".to_string(),
                    value: "".to_string(),
                }
            })?;
            match text.as_str() {
                "true" => Ok(HirNode::Literal(HirLiteral::Boolean(true))),
                "false" => Ok(HirNode::Literal(HirLiteral::Boolean(false))),
                _ => Err(HirBuildError::InvalidLiteralFormat {
                    literal_kind: "boolean".to_string(),
                    value: text,
                }),
            }
        }
        "array_literal" => {
            // Parse array elements: [1, 2, 3] or ["a", "b", "c"] or []
            let mut elements = Vec::new();
            let mut cursor = expr_node.walk();
            for child in expr_node.named_children(&mut cursor) {
                if child.kind() != "[" && child.kind() != "]" && child.kind() != "," {
                    // This is an element
                    elements.push(build_hir_expression(child, source_code)?);
                }
            }
            Ok(HirNode::Literal(HirLiteral::Array(elements)))
        }
        "method_call" => {
            // Extract target, method, arguments
            let target_node = expr_node.child_by_field_name("target").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "method_call".to_string(),
                    field_name: "target".to_string(),
                }
            })?;
            // Allow receiver to be any expression (identifier, literal, parenthesized expr, etc.)
            let target_expr = build_hir_expression(target_node, source_code)?;
            let method_node = expr_node.child_by_field_name("method").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "method_call".to_string(),
                    field_name: "method".to_string(),
                }
            })?;
            let method_name = get_node_text(method_node, source_code).unwrap_or_default();
            // Recoger argumentos usando field_name_for_child (Tree-sitter 0.22)
            let mut args = Vec::new();
            let mut mc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut mc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let hir_mc = HirMethodCall {
                target: Box::new(target_expr),
                method: method_name,
                args,
            };
            Ok(HirNode::MethodCall(Box::new(hir_mc)))
        }
        // 🛡️ Special anti-hallucination system functions as direct expressions
        "ask_ensemble_call" => {
            // Handle ask_ensemble_call as special function
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "ask_ensemble".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "ask_parallel_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "ask_parallel".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "ask_consensus_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "ask_consensus".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "ask_with_fallback_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "ask_with_fallback".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "memory_store_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "memory_store".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "memory_load_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "memory_load".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "enumerate_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "enumerate".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "range_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "range".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "map_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "map".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "filter_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "filter".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "reduce_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "reduce".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "read_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "read_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "write_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "write_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "save_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            if args.len() != 2 {
                return Err(HirBuildError::UnsupportedExpressionType(format!(
                    "save_file_call expects 2 arguments, got {}",
                    args.len()
                )));
            }
            let call = HirFunctionCall {
                function: "save_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "load_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            if args.len() != 1 {
                return Err(HirBuildError::UnsupportedExpressionType(format!(
                    "load_file_call expects 1 argument, got {}",
                    args.len()
                )));
            }
            let call = HirFunctionCall {
                function: "load_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "execute_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "execute".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        // 🔧 Sprint B: Global configuration functions
        "set_default_model_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "set_default_model".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "get_provider_default_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "get_provider_default".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        // 🛡️ Sprint C: Anti-hallucination validation functions
        "confidence_score_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "confidence_score".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "validate_response_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "validate_response".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        // 🧠 Sprint D: Memory Index / RAG Lite functions
        "memory_index_ingest_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "memory_index.ingest".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "memory_index_vector_search_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "memory_index.vector_search".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "auto_rag_context_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "auto_rag_context".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        // 📦 Multioutput functions - Multiple outputs system
        "ask_multioutput_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "ask_multioutput".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "save_binary_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "save_binary_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "load_binary_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "load_binary_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "get_multioutput_text_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "get_multioutput_text".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "get_multioutput_files_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "get_multioutput_files".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "get_multioutput_metadata_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "get_multioutput_metadata".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "save_multioutput_file_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "save_multioutput_file".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "load_csv_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "load_csv".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "save_json_call" => {
            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }
            let call = HirFunctionCall {
                function: "save_json".to_string(),
                args,
                type_arguments: Vec::new(),
            };
            Ok(HirNode::FunctionCall(Box::new(call)))
        }
        "function_call" => {
            // Function calls can be top-level statements
            let func_node = expr_node.child_by_field_name("function").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "function_call".to_string(),
                    field_name: "function".to_string(),
                }
            })?;
            // Try to capture fully-qualified name up to the opening parenthesis
            let start = func_node.start_byte();
            let src_bytes = source_code.as_bytes();
            let mut end = start;
            while end < src_bytes.len() {
                if src_bytes[end] as char == '(' {
                    break;
                }
                end += 1;
            }
            let func_span = if end > start {
                source_code[start..end].to_string()
            } else {
                get_node_text(func_node, source_code).unwrap_or_default()
            };
            let func_name = func_span.trim().to_string();

            let mut args = Vec::new();
            let mut fc_cursor = expr_node.walk();
            for ch in expr_node.named_children(&mut fc_cursor) {
                if is_arg(&expr_node, &ch) {
                    args.push(build_hir_expression(ch, source_code)?);
                }
            }

            // Check if this is a data function and convert to specific HIR node
            match func_name.as_str() {
                "data_load" => {
                    // data_load(path)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "data_load expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let path_expr = args.into_iter().next().unwrap();
                    // Extract the path string from the expression
                    let source_path = match path_expr {
                        HirNode::Literal(HirLiteral::String(s)) => s,
                        _ => {
                            return Err(HirBuildError::UnsupportedExpressionType(
                                "data_load path must be a string literal".to_string(),
                            ))
                        }
                    };
                    let data_load = HirDataLoad {
                        source: source_path,
                        schema_name: None,
                        options: Vec::new(),
                    };
                    Ok(HirNode::DataLoad(Box::new(data_load)))
                }
                "data_filter" => {
                    // data_filter(data, condition)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "data_filter expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let mut args_iter = args.into_iter();
                    let data_expr = args_iter.next().unwrap();
                    let condition_expr = args_iter.next().unwrap();
                    let data_filter = HirDataFilter {
                        input: Box::new(data_expr),
                        condition: Box::new(condition_expr),
                    };
                    Ok(HirNode::DataFilter(Box::new(data_filter)))
                }
                "data_select" => {
                    // data_select(data, fields)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "data_select expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let mut args_iter = args.into_iter();
                    let data_expr = args_iter.next().unwrap();
                    let fields_expr = args_iter.next().unwrap();
                    // For now, assume fields is a string literal containing field names
                    let fields = match fields_expr {
                        HirNode::Literal(HirLiteral::String(s)) => vec![s],
                        _ => {
                            return Err(HirBuildError::UnsupportedExpressionType(
                                "data_select fields must be a string literal".to_string(),
                            ))
                        }
                    };
                    let data_select = HirDataSelect {
                        input: Box::new(data_expr),
                        fields,
                    };
                    Ok(HirNode::DataSelect(Box::new(data_select)))
                }
                "data_take" => {
                    // data_take(data, count)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "data_take expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let mut args_iter = args.into_iter();
                    let data_expr = args_iter.next().unwrap();
                    let count_expr = args_iter.next().unwrap();
                    let data_take = HirDataTake {
                        input: Box::new(data_expr),
                        count: Box::new(count_expr),
                    };
                    Ok(HirNode::DataTake(Box::new(data_take)))
                }
                // 🛡️ Anti-Hallucination System Functions
                "ask_ensemble" => {
                    // ask_ensemble(prompts, strategy, [model])
                    if args.len() < 2 || args.len() > 3 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "ask_ensemble expects 2-3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "ask_parallel" => {
                    // ask_parallel(prompts, [model])
                    if args.is_empty() || args.len() > 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "ask_parallel expects 1-2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "ask_consensus" => {
                    // ask_consensus(prompts, models, [threshold])
                    if args.len() < 2 || args.len() > 3 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "ask_consensus expects 2-3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "ask_with_fallback" => {
                    // ask_with_fallback(prompts, fallback, [model])
                    if args.len() < 2 || args.len() > 3 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "ask_with_fallback expects 2-3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "memory_store" => {
                    // memory_store(key, value)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "memory_store expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "memory_load" => {
                    // memory_load(key)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "memory_load expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "enumerate" => {
                    // enumerate(iterable)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "enumerate expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "range" => {
                    // range(start, end, [step])
                    if args.len() < 2 || args.len() > 3 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "range expects 2-3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "map" => {
                    // map(function, iterable)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "map expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "filter" => {
                    // filter(function, iterable)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "filter expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "reduce" => {
                    // reduce(function, iterable, [initial_value])
                    if args.len() < 2 || args.len() > 3 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "reduce expects 2-3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "read_file" => {
                    // read_file(path)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "read_file expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "write_file" => {
                    // write_file(path, content)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "write_file expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "save_file" => {
                    // save_file(path, content)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "save_file expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    Ok(HirNode::FunctionCall(Box::new(HirFunctionCall {
                        function: "save_file".to_string(),
                        args,
                        type_arguments: Vec::new(),
                    })))
                }
                "load_file" => {
                    // load_file(path)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "load_file expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    Ok(HirNode::FunctionCall(Box::new(HirFunctionCall {
                        function: "load_file".to_string(),
                        args,
                        type_arguments: Vec::new(),
                    })))
                }
                "execute" => {
                    // execute(command)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "execute expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "set_default_model" => {
                    // set_default_model(model)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "set_default_model expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "get_provider_default" => {
                    // get_provider_default()
                    if !args.is_empty() {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "get_provider_default expects 0 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "confidence_score" => {
                    // confidence_score(response)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "confidence_score expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "validate_response" => {
                    // validate_response(response)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "validate_response expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "memory_index.ingest" => {
                    // memory_index.ingest(data)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "memory_index.ingest expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "memory_index.vector_search" => {
                    // memory_index.vector_search(query)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "memory_index.vector_search expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "auto_rag_context" => {
                    // auto_rag_context()
                    if !args.is_empty() {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "auto_rag_context expects 0 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "ask_multioutput" => {
                    // ask_multioutput(prompt, output_files)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "ask_multioutput expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "save_binary_file" => {
                    // save_binary_file(binary_file, path)
                    if args.len() != 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "save_binary_file expects 2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "load_binary_file" => {
                    // load_binary_file(path, [name])
                    if args.is_empty() || args.len() > 2 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "load_binary_file expects 1-2 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "get_multioutput_text" => {
                    // get_multioutput_text(multioutput)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "get_multioutput_text expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "get_multioutput_files" => {
                    // get_multioutput_files(multioutput)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "get_multioutput_files expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "get_multioutput_metadata" => {
                    // get_multioutput_metadata(multioutput)
                    if args.len() != 1 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "get_multioutput_metadata expects 1 argument, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                "save_multioutput_file" => {
                    // save_multioutput_file(multioutput, index, path)
                    if args.len() != 3 {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "save_multioutput_file expects 3 arguments, got {}",
                            args.len()
                        )));
                    }
                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments: Vec::new(),
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
                _ => {
                    // Regular function call
                    // Parse generic type arguments if present
                    let type_args_node = expr_node.child_by_field_name("type_args");
                    let type_arguments = if let Some(tnode) = type_args_node {
                        // reuse parse_generic_parameters but node may be same format
                        parse_generic_parameters(tnode, source_code)
                    } else {
                        Vec::new()
                    };

                    let call = HirFunctionCall {
                        function: func_name,
                        args,
                        type_arguments,
                    };
                    Ok(HirNode::FunctionCall(Box::new(call)))
                }
            }
        }
        "typeof_expression" => {
            let arg_node = expr_node.child_by_field_name("argument").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "typeof_expression".to_string(),
                    field_name: "argument".to_string(),
                }
            })?;
            let hir_arg = build_hir_expression(arg_node, source_code)?;
            let hir_typeof = HirTypeOf {
                argument: Box::new(hir_arg),
            };
            Ok(HirNode::TypeOf(Box::new(hir_typeof)))
        }
        "identifier" => {
            // Simple variable reference
            let name = get_node_text(expr_node, source_code).unwrap_or_default();
            Ok(HirNode::Identifier(name))
        }
        "binary_expression" => {
            // Parse binary expressions like x > 3, a + b, etc.
            let mut cursor = expr_node.walk();
            let children: Vec<_> = expr_node.children(&mut cursor).collect();

            if children.len() >= 3 {
                let left_expr = build_hir_expression(children[0], source_code)?;
                let operator = get_node_text(children[1], source_code).unwrap_or_default();
                let right_expr = build_hir_expression(children[2], source_code)?;

                let operator_enum = match operator.as_str() {
                    "+" => HirBinaryOperator::Add,
                    "-" => HirBinaryOperator::Subtract,
                    "*" => HirBinaryOperator::Multiply,
                    "/" => HirBinaryOperator::Divide,
                    ">" => HirBinaryOperator::GreaterThan,
                    "<" => HirBinaryOperator::LessThan,
                    ">=" => HirBinaryOperator::GreaterEqual,
                    "<=" => HirBinaryOperator::LessEqual,
                    "==" => HirBinaryOperator::Equal,
                    "!=" => HirBinaryOperator::NotEqual,
                    "&&" => HirBinaryOperator::And,
                    "||" => HirBinaryOperator::Or,
                    _ => {
                        return Err(HirBuildError::UnsupportedExpressionType(format!(
                            "Unsupported operator: {}",
                            operator
                        )))
                    }
                };

                let binary_expr = HirBinaryExpression {
                    left: Box::new(left_expr),
                    operator: operator_enum,
                    right: Box::new(right_expr),
                };
                Ok(HirNode::Binary(Box::new(binary_expr)))
            } else {
                Err(HirBuildError::UnsupportedExpressionType(
                    "Invalid binary_expression structure".to_string(),
                ))
            }
        }
        "assignment_expression" => {
            let left_node = expr_node.child_by_field_name("left").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "assignment_expression".to_string(),
                    field_name: "left".to_string(),
                }
            })?;
            let left_name = get_node_text(left_node, source_code).ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "identifier".to_string(),
                    field_name: "text".to_string(),
                }
            })?;
            let right_node = expr_node.child_by_field_name("right").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "assignment_expression".to_string(),
                    field_name: "right".to_string(),
                }
            })?;
            let right_expr = build_hir_expression(right_node, source_code)?;
            let assignment = HirAssignment {
                left: left_name,
                right: Box::new(right_expr),
            };
            Ok(HirNode::Assignment(Box::new(assignment)))
        }
        "await_expression" => {
            let expr_node = expr_node.child_by_field_name("expression").ok_or_else(|| {
                HirBuildError::MissingField {
                    node_type: "await_expression".to_string(),
                    field_name: "expression".to_string(),
                }
            })?;
            let inner_expr = build_hir_expression(expr_node, source_code)?;
            let await_expr = HirAwait {
                expression: Box::new(inner_expr),
            };
            Ok(HirNode::Await(Box::new(await_expr)))
        }
        _ => {
            eprintln!(
                "Unsupported expression node type in build_hir_expression: {}",
                expr_node.kind()
            );
            Err(HirBuildError::UnsupportedExpressionType(
                expr_node.kind().to_string(),
            ))
        }
    }
}

pub fn build_hir_from_cst(
    root_cst_node: Node,
    source_code: &str,
) -> Result<Vec<HirNode>, HirBuildError> {
    let mut hir_nodes = Vec::new();
    let mut cursor = root_cst_node.walk();

    for top_level_node in root_cst_node.children(&mut cursor) {
        if top_level_node.is_error() {
            // Gate noisy parser error-node logs behind VERBOSE
            if std::env::var("LEXON_VERBOSE").ok().as_deref() == Some("1") {
                eprintln!(
                    "Skipping CST error node during HIR build: {:?}",
                    top_level_node.to_sexp()
                );
            }
            continue;
        }

        match top_level_node.kind() {
            // 🔧 Sprint B: Support for modules and imports
            "module_declaration" => {
                let path_node = top_level_node.child_by_field_name("path").ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "module_declaration".to_string(),
                        field_name: "path".to_string(),
                    }
                })?;
                let module_path = get_node_text(path_node, source_code).ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "dotted_identifier".to_string(),
                        field_name: "text".to_string(),
                    }
                })?;

                let hir_module = HirModuleDeclaration {
                    path: module_path.split('.').map(|s| s.to_string()).collect(),
                };
                hir_nodes.push(HirNode::ModuleDeclaration(Box::new(hir_module)));
            }
            "import_statement" | "simple_import" | "from_import" => {
                // Normalize node to either a simple_import or from_import node
                let (node_kind, node_ref) = if top_level_node.kind() == "import_statement" {
                    // Find the child that is either simple_import or from_import
                    let mut it = top_level_node.walk();
                    let mut found = None;
                    for child in top_level_node.children(&mut it) {
                        if child.kind() == "simple_import" || child.kind() == "from_import" {
                            found = Some(child);
                            break;
                        }
                    }
                    let c = found.ok_or_else(|| HirBuildError::MissingField {
                        node_type: "import_statement".to_string(),
                        field_name: "child(simple|from)_import".to_string(),
                    })?;
                    (c.kind().to_string(), c)
                } else {
                    (top_level_node.kind().to_string(), top_level_node)
                };

                if node_kind == "simple_import" {
                    let path_node = node_ref.child_by_field_name("path").ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "simple_import".to_string(),
                            field_name: "path".to_string(),
                        }
                    })?;
                    let import_path = get_node_text(path_node, source_code).ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "dotted_identifier".to_string(),
                            field_name: "text".to_string(),
                        }
                    })?;
                    let alias = node_ref
                        .child_by_field_name("alias")
                        .and_then(|n| get_node_text(n, source_code));
                    // Some grammars may attach braced items to simple_import; parse them if present
                    let mut items = Vec::new();
                    if let Some(items_node) = node_ref.child_by_field_name("items") {
                        let mut items_cursor = items_node.walk();
                        for item_node in items_node.children(&mut items_cursor) {
                            if item_node.kind() == "import_item" {
                                let item_name = item_node
                                    .child(0)
                                    .and_then(|n| get_node_text(n, source_code))
                                    .unwrap_or_default();
                                let item_alias = if item_node.child_count() > 2 {
                                    item_node
                                        .child(2)
                                        .and_then(|n| get_node_text(n, source_code))
                                } else {
                                    None
                                };
                                items.push((item_name, item_alias));
                            }
                        }
                    }
                    let hir_import = if items.is_empty() {
                        HirImportDeclaration {
                            path: import_path.split('.').map(|s| s.to_string()).collect(),
                            items: Vec::new(),
                            alias,
                        }
                    } else {
                        // When items present, treat as from_import semantics; ignore module alias
                        HirImportDeclaration {
                            path: import_path.split('.').map(|s| s.to_string()).collect(),
                            items,
                            alias: None,
                        }
                    };
                    hir_nodes.push(HirNode::ImportDeclaration(Box::new(hir_import)));
                } else {
                    // from_import
                    let path_node = node_ref.child_by_field_name("path").ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "from_import".to_string(),
                            field_name: "path".to_string(),
                        }
                    })?;
                    let import_path = get_node_text(path_node, source_code).ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "dotted_identifier".to_string(),
                            field_name: "text".to_string(),
                        }
                    })?;
                    let mut items = Vec::new();
                    if let Some(items_node) = node_ref.child_by_field_name("items") {
                        let mut items_cursor = items_node.walk();
                        for item_node in items_node.children(&mut items_cursor) {
                            if item_node.kind() == "import_item" {
                                let item_name = item_node
                                    .child(0)
                                    .and_then(|n| get_node_text(n, source_code))
                                    .unwrap_or_default();
                                let item_alias = if item_node.child_count() > 2 {
                                    item_node
                                        .child(2)
                                        .and_then(|n| get_node_text(n, source_code))
                                } else {
                                    None
                                };
                                items.push((item_name, item_alias));
                            }
                        }
                    }
                    let hir_import = HirImportDeclaration {
                        path: import_path.split('.').map(|s| s.to_string()).collect(),
                        items,
                        alias: None,
                    };
                    hir_nodes.push(HirNode::ImportDeclaration(Box::new(hir_import)));
                }
            }
            "variable_declaration" => {
                let var_name_node =
                    top_level_node.child_by_field_name("name").ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "variable_declaration".to_string(),
                            field_name: "name".to_string(),
                        }
                    })?;
                let var_name = get_node_text(var_name_node, source_code).ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "identifier".to_string(),
                        field_name: "text".to_string(),
                    }
                })?; // Should not happen if field 'name' exists

                // Type inference support: type annotation is now optional
                let type_node_opt = top_level_node.child_by_field_name("type");

                let value_cst_node =
                    top_level_node.child_by_field_name("value").ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "variable_declaration".to_string(),
                            field_name: "value".to_string(),
                        }
                    })?;
                let hir_value_node = build_hir_expression(value_cst_node, source_code)?;

                // Get type name (explicit or inferred)
                let type_name_str =
                    get_variable_type_name(type_node_opt, &hir_value_node, source_code);

                let hir_var_decl = HirVariableDeclaration {
                    name: var_name,
                    type_name: type_name_str,
                    value: Box::new(hir_value_node),
                };
                hir_nodes.push(HirNode::VariableDeclaration(Box::new(hir_var_decl)));
            }
            "function_definition" => {
                // Parse function name
                let fn_name_node = top_level_node.child_by_field_name("name").ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "function_definition".to_string(),
                        field_name: "name".to_string(),
                    }
                })?;
                let fn_name = get_node_text(fn_name_node, source_code).ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "identifier".to_string(),
                        field_name: "text".to_string(),
                    }
                })?;

                // Generic parameters
                let type_params_node = top_level_node.child_by_field_name("type_params");
                let type_parameters = type_params_node
                    .map(|n| parse_generic_parameters(n, source_code))
                    .unwrap_or_default();

                // Optional return type
                let return_type_node_opt = top_level_node.child_by_field_name("return_type");
                let return_type_text =
                    return_type_node_opt.and_then(|n| get_node_text(n, source_code));

                // Body block
                let body_node = top_level_node.child_by_field_name("body").ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "function_definition".to_string(),
                        field_name: "body".to_string(),
                    }
                })?;

                // Extract HIR nodes from body (currently only variable_declaration)
                let mut body_hir_nodes = Vec::<HirNode>::new();
                let mut body_cursor = body_node.walk();
                for statement_node in body_node.children(&mut body_cursor) {
                    if statement_node.kind() == "variable_declaration" {
                        // Reuse logic for variable declarations by constructing HIR the same way
                        let var_name_node =
                            statement_node.child_by_field_name("name").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "variable_declaration".to_string(),
                                    field_name: "name".to_string(),
                                }
                            })?;
                        let var_name =
                            get_node_text(var_name_node, source_code).ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "identifier".to_string(),
                                    field_name: "text".to_string(),
                                }
                            })?;

                        let type_node_opt = statement_node.child_by_field_name("type");
                        let value_cst_node = statement_node
                            .child_by_field_name("value")
                            .ok_or_else(|| HirBuildError::MissingField {
                                node_type: "variable_declaration".to_string(),
                                field_name: "value".to_string(),
                            })?;
                        let hir_value_node = build_hir_expression(value_cst_node, source_code)?;
                        let type_name_str =
                            get_variable_type_name(type_node_opt, &hir_value_node, source_code);

                        let hir_var_decl = HirVariableDeclaration {
                            name: var_name,
                            type_name: type_name_str,
                            value: Box::new(hir_value_node),
                        };
                        body_hir_nodes.push(HirNode::VariableDeclaration(Box::new(hir_var_decl)));
                    } else if statement_node.kind() == "while_statement" {
                        // Parse while
                        let condition_node = statement_node
                            .child_by_field_name("condition")
                            .ok_or_else(|| HirBuildError::MissingField {
                                node_type: "while_statement".to_string(),
                                field_name: "condition".to_string(),
                            })?;
                        let condition_expr = build_hir_expression(condition_node, source_code)?;
                        let body_node =
                            statement_node.child_by_field_name("body").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "while_statement".to_string(),
                                    field_name: "body".to_string(),
                                }
                            })?;
                        let mut body_vec = Vec::new();
                        let mut body_cursor2 = body_node.walk();
                        for stmt in body_node.children(&mut body_cursor2) {
                            match stmt.kind() {
                                "break_statement" => body_vec.push(HirNode::Break),
                                "continue_statement" => body_vec.push(HirNode::Continue),
                                _ => {}
                            }
                        }
                        let hir_while = HirWhile {
                            condition: Box::new(condition_expr),
                            body: body_vec,
                        };
                        body_hir_nodes.push(HirNode::While(Box::new(hir_while)));
                    } else if statement_node.kind() == "if_statement" {
                        // Parse if statement
                        let condition_node = statement_node
                            .child_by_field_name("condition")
                            .ok_or_else(|| HirBuildError::MissingField {
                                node_type: "if_statement".to_string(),
                                field_name: "condition".to_string(),
                            })?;
                        let condition_expr = build_hir_expression(condition_node, source_code)?;

                        let consequence_node = statement_node
                            .child_by_field_name("consequence")
                            .ok_or_else(|| HirBuildError::MissingField {
                                node_type: "if_statement".to_string(),
                                field_name: "consequence".to_string(),
                            })?;

                        let mut then_body = Vec::new();
                        let mut then_cursor = consequence_node.walk();
                        for stmt in consequence_node.children(&mut then_cursor) {
                            if stmt.kind() == "expression_statement" {
                                if let Some(expr_child) = stmt.named_child(0) {
                                    let hir_expr = build_hir_expression(expr_child, source_code)?;
                                    then_body.push(hir_expr);
                                }
                            } else if stmt.kind() == "variable_declaration" {
                                // Handle variable declarations in if body
                                let var_name_node =
                                    stmt.child_by_field_name("name").ok_or_else(|| {
                                        HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "name".to_string(),
                                        }
                                    })?;
                                let var_name = get_node_text(var_name_node, source_code)
                                    .ok_or_else(|| HirBuildError::MissingField {
                                        node_type: "identifier".to_string(),
                                        field_name: "text".to_string(),
                                    })?;
                                let type_node =
                                    stmt.child_by_field_name("type").ok_or_else(|| {
                                        HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "type".to_string(),
                                        }
                                    })?;
                                let type_name_str = get_node_text(type_node, source_code);
                                let value_cst_node =
                                    stmt.child_by_field_name("value").ok_or_else(|| {
                                        HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "value".to_string(),
                                        }
                                    })?;
                                let hir_value_node =
                                    build_hir_expression(value_cst_node, source_code)?;
                                let hir_var_decl = HirVariableDeclaration {
                                    name: var_name,
                                    type_name: type_name_str,
                                    value: Box::new(hir_value_node),
                                };
                                then_body
                                    .push(HirNode::VariableDeclaration(Box::new(hir_var_decl)));
                            }
                        }

                        // Handle optional else block
                        let else_body = if let Some(alternative_node) =
                            statement_node.child_by_field_name("alternative")
                        {
                            let mut else_stmts = Vec::new();
                            let mut else_cursor = alternative_node.walk();
                            for stmt in alternative_node.children(&mut else_cursor) {
                                if stmt.kind() == "expression_statement" {
                                    if let Some(expr_child) = stmt.named_child(0) {
                                        let hir_expr =
                                            build_hir_expression(expr_child, source_code)?;
                                        else_stmts.push(hir_expr);
                                    }
                                } else if stmt.kind() == "variable_declaration" {
                                    // Handle variable declarations in else body
                                    let var_name_node = stmt
                                        .child_by_field_name("name")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "name".to_string(),
                                        })?;
                                    let var_name = get_node_text(var_name_node, source_code)
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "identifier".to_string(),
                                            field_name: "text".to_string(),
                                        })?;
                                    let type_node =
                                        stmt.child_by_field_name("type").ok_or_else(|| {
                                            HirBuildError::MissingField {
                                                node_type: "variable_declaration".to_string(),
                                                field_name: "type".to_string(),
                                            }
                                        })?;
                                    let type_name_str = get_node_text(type_node, source_code);
                                    let value_cst_node = stmt
                                        .child_by_field_name("value")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "value".to_string(),
                                        })?;
                                    let hir_value_node =
                                        build_hir_expression(value_cst_node, source_code)?;
                                    let hir_var_decl = HirVariableDeclaration {
                                        name: var_name,
                                        type_name: type_name_str,
                                        value: Box::new(hir_value_node),
                                    };
                                    else_stmts
                                        .push(HirNode::VariableDeclaration(Box::new(hir_var_decl)));
                                }
                            }
                            Some(else_stmts)
                        } else {
                            None
                        };

                        let hir_if = HirIf {
                            condition: Box::new(condition_expr),
                            then_body,
                            else_body,
                        };
                        body_hir_nodes.push(HirNode::If(Box::new(hir_if)));
                    } else if statement_node.kind() == "break_statement" {
                        body_hir_nodes.push(HirNode::Break);
                    } else if statement_node.kind() == "continue_statement" {
                        body_hir_nodes.push(HirNode::Continue);
                    } else if statement_node.kind() == "return_statement" {
                        // Parse return statement
                        let return_expr = if statement_node.child_count() > 1 {
                            // return_statement tiene: "return" + expression + ";"
                            // The expression is at index 1
                            if let Some(expr_node) = statement_node.child(1) {
                                if expr_node.kind() != ";" {
                                    Some(Box::new(build_hir_expression(expr_node, source_code)?))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        } else {
                            None
                        };
                        let hir_return = HirReturn {
                            expression: return_expr,
                        };
                        body_hir_nodes.push(HirNode::Return(Box::new(hir_return)));
                    } else if statement_node.kind() == "for_in_statement" {
                        // Parse for-in loop
                        let iter_node =
                            statement_node
                                .child_by_field_name("iterator")
                                .ok_or_else(|| HirBuildError::MissingField {
                                    node_type: "for_in_statement".to_string(),
                                    field_name: "iterator".to_string(),
                                })?;
                        let iterator_name =
                            get_node_text(iter_node, source_code).unwrap_or_default();
                        let iterable_node = statement_node
                            .child_by_field_name("iterable")
                            .ok_or_else(|| HirBuildError::MissingField {
                                node_type: "for_in_statement".to_string(),
                                field_name: "iterable".to_string(),
                            })?;
                        let iterable_expr = build_hir_expression(iterable_node, source_code)?;
                        let body_node =
                            statement_node.child_by_field_name("body").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "for_in_statement".to_string(),
                                    field_name: "body".to_string(),
                                }
                            })?;
                        let mut inner_body = Vec::new();
                        let mut inner_cursor = body_node.walk();
                        for stmt in body_node.children(&mut inner_cursor) {
                            match stmt.kind() {
                                "break_statement" => inner_body.push(HirNode::Break),
                                "continue_statement" => inner_body.push(HirNode::Continue),
                                _ => {}
                            }
                        }
                        let for_in_hir = HirForIn {
                            iterator: iterator_name,
                            iterable: Box::new(iterable_expr),
                            body: inner_body,
                        };
                        body_hir_nodes.push(HirNode::ForIn(Box::new(for_in_hir)));
                    } else if statement_node.kind() == "match_statement" {
                        // Parse match statement
                        let value_node =
                            statement_node.child_by_field_name("value").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "match_statement".to_string(),
                                    field_name: "value".to_string(),
                                }
                            })?;
                        let value_expr = build_hir_expression(value_node, source_code)?;

                        let mut arms = Vec::new();
                        let mut match_cursor = statement_node.walk();
                        for child in statement_node.children(&mut match_cursor) {
                            if child.kind() == "match_arm" {
                                let pattern_node = child
                                    .child_by_field_name("pattern")
                                    .ok_or_else(|| HirBuildError::MissingField {
                                        node_type: "match_arm".to_string(),
                                        field_name: "pattern".to_string(),
                                    })?;
                                let pattern_expr = build_hir_expression(pattern_node, source_code)?;

                                let mut body_stmts = Vec::new();
                                if let Some(body_node) = child.named_child(2) {
                                    if body_node.kind() == "block_statement" {
                                        let mut body_cursor = body_node.walk();
                                        for stmt in body_node.children(&mut body_cursor) {
                                            if stmt.kind() == "expression_statement" {
                                                if let Some(expr_child) = stmt.named_child(0) {
                                                    let hir_expr = build_hir_expression(
                                                        expr_child,
                                                        source_code,
                                                    )?;
                                                    body_stmts.push(hir_expr);
                                                }
                                            }
                                        }
                                    } else {
                                        let hir_expr =
                                            build_hir_expression(body_node, source_code)?;
                                        body_stmts.push(hir_expr);
                                    }
                                }

                                let match_arm = HirMatchArm {
                                    pattern: Box::new(pattern_expr),
                                    body: body_stmts,
                                };
                                arms.push(match_arm);
                            }
                        }

                        let hir_match = HirMatch {
                            value: Box::new(value_expr),
                            arms,
                        };
                        body_hir_nodes.push(HirNode::Match(Box::new(hir_match)));
                    }
                }

                // Parse function parameters
                let parameters =
                    if let Some(params_node) = top_level_node.child_by_field_name("parameters") {
                        parse_function_parameters(params_node, source_code)
                    } else {
                        Vec::new()
                    };

                // Check for async modifier
                let is_async = top_level_node
                    .child_by_field_name("async_modifier")
                    .is_some();

                let fn_hir = HirFunctionDefinition {
                    name: fn_name,
                    parameters,
                    type_parameters,
                    return_type: return_type_text,
                    body: body_hir_nodes,
                    visibility: HirVisibility::Private,
                    is_async,
                };
                hir_nodes.push(HirNode::FunctionDefinition(Box::new(fn_hir)));
            }
            "schema_definition" => {
                // Schema name
                let schema_name_node =
                    top_level_node.child_by_field_name("name").ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "schema_definition".to_string(),
                            field_name: "name".to_string(),
                        }
                    })?;
                let schema_name =
                    get_node_text(schema_name_node, source_code).ok_or_else(|| {
                        HirBuildError::MissingField {
                            node_type: "identifier".to_string(),
                            field_name: "text".to_string(),
                        }
                    })?;

                // Fields
                let mut schema_fields: Vec<HirSchemaField> = Vec::new();
                let mut schema_cursor = top_level_node.walk();
                for field_node in top_level_node.children(&mut schema_cursor) {
                    if field_node.kind() == "schema_field_definition" {
                        let field_name_node =
                            field_node.child_by_field_name("name").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "schema_field_definition".to_string(),
                                    field_name: "name".to_string(),
                                }
                            })?;
                        let field_name =
                            get_node_text(field_name_node, source_code).ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "identifier".to_string(),
                                    field_name: "text".to_string(),
                                }
                            })?;

                        let optional_marker = field_node.child_by_field_name("optional_marker");
                        let is_optional = optional_marker.is_some();

                        let field_type_node =
                            field_node.child_by_field_name("type").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "schema_field_definition".to_string(),
                                    field_name: "type".to_string(),
                                }
                            })?;
                        let type_name =
                            get_node_text(field_type_node, source_code).unwrap_or_default();

                        // Default value
                        let default_val_cst = field_node.child_by_field_name("default_value");
                        let default_hir = if let Some(def_node) = default_val_cst {
                            Some(Box::new(build_hir_expression(def_node, source_code)?))
                        } else {
                            None
                        };

                        let hir_field = HirSchemaField {
                            name: field_name,
                            type_name,
                            is_optional,
                            default_value: default_hir,
                        };
                        schema_fields.push(hir_field);
                    }
                }

                let type_params_node = top_level_node.child_by_field_name("type_params");
                let type_parameters = type_params_node
                    .map(|n| parse_generic_parameters(n, source_code))
                    .unwrap_or_default();

                let schema_hir = HirSchemaDefinition {
                    name: schema_name,
                    type_parameters,
                    fields: schema_fields,
                    visibility: HirVisibility::Private,
                };
                hir_nodes.push(HirNode::SchemaDefinition(Box::new(schema_hir)));
            }
            "trait_definition" => {
                // trait name
                let name_node = top_level_node.child_by_field_name("name").ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "trait_definition".to_string(),
                        field_name: "name".to_string(),
                    }
                })?;
                let trait_name = get_node_text(name_node, source_code).ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "identifier".to_string(),
                        field_name: "text".to_string(),
                    }
                })?;

                // methods signatures
                let mut methods = Vec::new();
                let mut trait_cursor = top_level_node.walk();
                for child in top_level_node.children(&mut trait_cursor) {
                    if child.kind() == "function_signature" {
                        let fname_node = child.child_by_field_name("name").ok_or_else(|| {
                            HirBuildError::MissingField {
                                node_type: "function_signature".to_string(),
                                field_name: "name".to_string(),
                            }
                        })?;
                        let fname = get_node_text(fname_node, source_code).unwrap_or_default();
                        let ret_type_node = child.child_by_field_name("return_type");
                        let ret_type = ret_type_node.and_then(|n| get_node_text(n, source_code));
                        // Parse function signature parameters
                        let sig_parameters =
                            if let Some(params_node) = child.child_by_field_name("parameters") {
                                parse_function_parameters(params_node, source_code)
                            } else {
                                Vec::new()
                            };
                        methods.push(HirFunctionSignature {
                            name: fname,
                            parameters: sig_parameters,
                            return_type: ret_type,
                        });
                    }
                }

                let type_params_node = top_level_node.child_by_field_name("type_params");
                let type_parameters = type_params_node
                    .map(|n| parse_generic_parameters(n, source_code))
                    .unwrap_or_default();

                let trait_hir = HirTraitDefinition {
                    name: trait_name,
                    type_parameters,
                    methods,
                    visibility: HirVisibility::Private,
                };
                hir_nodes.push(HirNode::TraitDefinition(Box::new(trait_hir)));
            }
            "impl_block" => {
                let target_node =
                    top_level_node
                        .child_by_field_name("target")
                        .ok_or_else(|| HirBuildError::MissingField {
                            node_type: "impl_block".to_string(),
                            field_name: "target".to_string(),
                        })?;
                let target_name = get_node_text(target_node, source_code).unwrap_or_default();

                let mut methods = Vec::new();
                let mut impl_cursor = top_level_node.walk();
                for child in top_level_node.children(&mut impl_cursor) {
                    if child.kind() == "function_definition" {
                        // Name
                        let fn_name_node = child.child_by_field_name("name").ok_or_else(|| {
                            HirBuildError::MissingField {
                                node_type: "function_definition".to_string(),
                                field_name: "name".to_string(),
                            }
                        })?;
                        let fn_name = get_node_text(fn_name_node, source_code).unwrap_or_default();

                        // Generics
                        let type_params_node = child.child_by_field_name("type_params");
                        let type_parameters = type_params_node
                            .map(|n| parse_generic_parameters(n, source_code))
                            .unwrap_or_default();

                        // Return type
                        let return_type_node_opt = child.child_by_field_name("return_type");
                        let return_type_text =
                            return_type_node_opt.and_then(|n| get_node_text(n, source_code));

                        // Body
                        let body_node = child.child_by_field_name("body").ok_or_else(|| {
                            HirBuildError::MissingField {
                                node_type: "function_definition".to_string(),
                                field_name: "body".to_string(),
                            }
                        })?;
                        let mut body_hir_nodes = Vec::<HirNode>::new();
                        let mut body_cursor = body_node.walk();
                        for statement_node in body_node.children(&mut body_cursor) {
                            match statement_node.kind() {
                                "variable_declaration" => {
                                    let var_name_node = statement_node
                                        .child_by_field_name("name")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "name".to_string(),
                                        })?;
                                    let var_name = get_node_text(var_name_node, source_code)
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "identifier".to_string(),
                                            field_name: "text".to_string(),
                                        })?;
                                    let type_node_opt = statement_node.child_by_field_name("type");
                                    let value_cst_node = statement_node
                                        .child_by_field_name("value")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "variable_declaration".to_string(),
                                            field_name: "value".to_string(),
                                        })?;
                                    let hir_value_node =
                                        build_hir_expression(value_cst_node, source_code)?;
                                    let type_name_str = get_variable_type_name(
                                        type_node_opt,
                                        &hir_value_node,
                                        source_code,
                                    );
                                    let hir_var_decl = HirVariableDeclaration {
                                        name: var_name,
                                        type_name: type_name_str,
                                        value: Box::new(hir_value_node),
                                    };
                                    body_hir_nodes
                                        .push(HirNode::VariableDeclaration(Box::new(hir_var_decl)));
                                }
                                "while_statement" => {
                                    let condition_node = statement_node
                                        .child_by_field_name("condition")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "while_statement".to_string(),
                                            field_name: "condition".to_string(),
                                        })?;
                                    let condition_expr =
                                        build_hir_expression(condition_node, source_code)?;
                                    let body_block = statement_node
                                        .child_by_field_name("body")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "while_statement".to_string(),
                                            field_name: "body".to_string(),
                                        })?;
                                    let mut inner = Vec::new();
                                    let mut c2 = body_block.walk();
                                    for stmt in body_block.children(&mut c2) {
                                        match stmt.kind() {
                                            "break_statement" => inner.push(HirNode::Break),
                                            "continue_statement" => inner.push(HirNode::Continue),
                                            _ => {}
                                        }
                                    }
                                    body_hir_nodes.push(HirNode::While(Box::new(HirWhile {
                                        condition: Box::new(condition_expr),
                                        body: inner,
                                    })));
                                }
                                "if_statement" => {
                                    let condition_node = statement_node
                                        .child_by_field_name("condition")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "if_statement".to_string(),
                                            field_name: "condition".to_string(),
                                        })?;
                                    let condition_expr =
                                        build_hir_expression(condition_node, source_code)?;
                                    let consequence_node = statement_node
                                        .child_by_field_name("consequence")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "if_statement".to_string(),
                                            field_name: "consequence".to_string(),
                                        })?;
                                    let mut then_body = Vec::new();
                                    let mut then_cursor = consequence_node.walk();
                                    for stmt in consequence_node.children(&mut then_cursor) {
                                        if stmt.kind() == "expression_statement" {
                                            if let Some(expr_child) = stmt.named_child(0) {
                                                then_body.push(build_hir_expression(
                                                    expr_child,
                                                    source_code,
                                                )?);
                                            }
                                        } else if stmt.kind() == "variable_declaration" {
                                            let var_name_node = stmt
                                                .child_by_field_name("name")
                                                .ok_or_else(|| HirBuildError::MissingField {
                                                    node_type: "variable_declaration".to_string(),
                                                    field_name: "name".to_string(),
                                                })?;
                                            let var_name =
                                                get_node_text(var_name_node, source_code)
                                                    .ok_or_else(|| HirBuildError::MissingField {
                                                        node_type: "identifier".to_string(),
                                                        field_name: "text".to_string(),
                                                    })?;
                                            let type_node = stmt
                                                .child_by_field_name("type")
                                                .ok_or_else(|| HirBuildError::MissingField {
                                                    node_type: "variable_declaration".to_string(),
                                                    field_name: "type".to_string(),
                                                })?;
                                            let type_name_str =
                                                get_node_text(type_node, source_code);
                                            let value_cst_node = stmt
                                                .child_by_field_name("value")
                                                .ok_or_else(|| HirBuildError::MissingField {
                                                    node_type: "variable_declaration".to_string(),
                                                    field_name: "value".to_string(),
                                                })?;
                                            let hir_value_node =
                                                build_hir_expression(value_cst_node, source_code)?;
                                            let hir_var_decl = HirVariableDeclaration {
                                                name: var_name,
                                                type_name: type_name_str,
                                                value: Box::new(hir_value_node),
                                            };
                                            then_body.push(HirNode::VariableDeclaration(Box::new(
                                                hir_var_decl,
                                            )));
                                        }
                                    }
                                    let else_body = if let Some(alternative_node) =
                                        statement_node.child_by_field_name("alternative")
                                    {
                                        let mut else_stmts = Vec::new();
                                        let mut else_cursor = alternative_node.walk();
                                        for stmt in alternative_node.children(&mut else_cursor) {
                                            if stmt.kind() == "expression_statement" {
                                                if let Some(expr_child) = stmt.named_child(0) {
                                                    else_stmts.push(build_hir_expression(
                                                        expr_child,
                                                        source_code,
                                                    )?);
                                                }
                                            } else if stmt.kind() == "variable_declaration" {
                                                let var_name_node = stmt
                                                    .child_by_field_name("name")
                                                    .ok_or_else(|| HirBuildError::MissingField {
                                                        node_type: "variable_declaration"
                                                            .to_string(),
                                                        field_name: "name".to_string(),
                                                    })?;
                                                let var_name =
                                                    get_node_text(var_name_node, source_code)
                                                        .ok_or_else(|| {
                                                            HirBuildError::MissingField {
                                                                node_type: "identifier".to_string(),
                                                                field_name: "text".to_string(),
                                                            }
                                                        })?;
                                                let type_node = stmt
                                                    .child_by_field_name("type")
                                                    .ok_or_else(|| HirBuildError::MissingField {
                                                        node_type: "variable_declaration"
                                                            .to_string(),
                                                        field_name: "type".to_string(),
                                                    })?;
                                                let type_name_str =
                                                    get_node_text(type_node, source_code);
                                                let value_cst_node = stmt
                                                    .child_by_field_name("value")
                                                    .ok_or_else(|| HirBuildError::MissingField {
                                                        node_type: "variable_declaration"
                                                            .to_string(),
                                                        field_name: "value".to_string(),
                                                    })?;
                                                let hir_value_node = build_hir_expression(
                                                    value_cst_node,
                                                    source_code,
                                                )?;
                                                let hir_var_decl = HirVariableDeclaration {
                                                    name: var_name,
                                                    type_name: type_name_str,
                                                    value: Box::new(hir_value_node),
                                                };
                                                else_stmts.push(HirNode::VariableDeclaration(
                                                    Box::new(hir_var_decl),
                                                ));
                                            }
                                        }
                                        Some(else_stmts)
                                    } else {
                                        None
                                    };
                                    body_hir_nodes.push(HirNode::If(Box::new(HirIf {
                                        condition: Box::new(condition_expr),
                                        then_body,
                                        else_body,
                                    })));
                                }
                                "break_statement" => body_hir_nodes.push(HirNode::Break),
                                "continue_statement" => body_hir_nodes.push(HirNode::Continue),
                                "return_statement" => {
                                    let return_expr = if statement_node.child_count() > 1 {
                                        if let Some(expr_node) = statement_node.child(1) {
                                            if expr_node.kind() != ";" {
                                                Some(Box::new(build_hir_expression(
                                                    expr_node,
                                                    source_code,
                                                )?))
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    } else {
                                        None
                                    };
                                    body_hir_nodes.push(HirNode::Return(Box::new(HirReturn {
                                        expression: return_expr,
                                    })));
                                }
                                "for_in_statement" => {
                                    let iter_node = statement_node
                                        .child_by_field_name("iterator")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "for_in_statement".to_string(),
                                            field_name: "iterator".to_string(),
                                        })?;
                                    let iterator_name =
                                        get_node_text(iter_node, source_code).unwrap_or_default();
                                    let iterable_node = statement_node
                                        .child_by_field_name("iterable")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "for_in_statement".to_string(),
                                            field_name: "iterable".to_string(),
                                        })?;
                                    let iterable_expr =
                                        build_hir_expression(iterable_node, source_code)?;
                                    let body_block = statement_node
                                        .child_by_field_name("body")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "for_in_statement".to_string(),
                                            field_name: "body".to_string(),
                                        })?;
                                    let mut inner_body = Vec::new();
                                    let mut inner_cursor = body_block.walk();
                                    for stmt in body_block.children(&mut inner_cursor) {
                                        match stmt.kind() {
                                            "break_statement" => inner_body.push(HirNode::Break),
                                            "continue_statement" => {
                                                inner_body.push(HirNode::Continue)
                                            }
                                            _ => {}
                                        }
                                    }
                                    body_hir_nodes.push(HirNode::ForIn(Box::new(HirForIn {
                                        iterator: iterator_name,
                                        iterable: Box::new(iterable_expr),
                                        body: inner_body,
                                    })));
                                }
                                "match_statement" => {
                                    let value_node = statement_node
                                        .child_by_field_name("value")
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "match_statement".to_string(),
                                            field_name: "value".to_string(),
                                        })?;
                                    let value_expr = build_hir_expression(value_node, source_code)?;
                                    let mut arms = Vec::new();
                                    let mut match_cursor = statement_node.walk();
                                    for child_arm in statement_node.children(&mut match_cursor) {
                                        if child_arm.kind() == "match_arm" {
                                            let pattern_node = child_arm
                                                .child_by_field_name("pattern")
                                                .ok_or_else(|| HirBuildError::MissingField {
                                                    node_type: "match_arm".to_string(),
                                                    field_name: "pattern".to_string(),
                                                })?;
                                            let pattern_expr =
                                                build_hir_expression(pattern_node, source_code)?;
                                            let mut body_stmts = Vec::new();
                                            if let Some(body_node) = child_arm.named_child(2) {
                                                if body_node.kind() == "block_statement" {
                                                    let mut bcur = body_node.walk();
                                                    for stmt in body_node.children(&mut bcur) {
                                                        if stmt.kind() == "expression_statement" {
                                                            if let Some(expr_child) =
                                                                stmt.named_child(0)
                                                            {
                                                                body_stmts.push(
                                                                    build_hir_expression(
                                                                        expr_child,
                                                                        source_code,
                                                                    )?,
                                                                );
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    let hir_expr = build_hir_expression(
                                                        body_node,
                                                        source_code,
                                                    )?;
                                                    body_stmts.push(hir_expr);
                                                }
                                            }
                                            arms.push(HirMatchArm {
                                                pattern: Box::new(pattern_expr),
                                                body: body_stmts,
                                            });
                                        }
                                    }
                                    body_hir_nodes.push(HirNode::Match(Box::new(HirMatch {
                                        value: Box::new(value_expr),
                                        arms,
                                    })));
                                }
                                "expression_statement" => {
                                    if let Some(expr_child) = statement_node.named_child(0) {
                                        body_hir_nodes
                                            .push(build_hir_expression(expr_child, source_code)?);
                                    }
                                }
                                _ => {}
                            }
                        }

                        // Parameters
                        let impl_parameters =
                            if let Some(params_node) = child.child_by_field_name("parameters") {
                                parse_function_parameters(params_node, source_code)
                            } else {
                                Vec::new()
                            };
                        // Async flag
                        let is_async = child.child_by_field_name("async_modifier").is_some();

                        let impl_fn = HirFunctionDefinition {
                            name: fn_name,
                            parameters: impl_parameters,
                            type_parameters,
                            return_type: return_type_text,
                            body: body_hir_nodes,
                            visibility: HirVisibility::Private,
                            is_async,
                        };
                        methods.push(impl_fn);
                    }
                }
                let impl_hir = HirImplBlock {
                    target: target_name,
                    methods,
                };
                hir_nodes.push(HirNode::ImplBlock(Box::new(impl_hir)));
            }
            "ask_expression" => {
                // Handle ask expressions as top-level statements
                let hir_expr = build_hir_expression(top_level_node, source_code)?;
                hir_nodes.push(hir_expr);
            }
            "comment" => {
                // Skip comments - they don't need to be in the HIR
                continue;
            }
            ";" => {
                // Skip stray semicolons at top-level (empty statements)
                continue;
            }
            "expression_statement" => {
                if let Some(expr_child) = top_level_node.named_child(0) {
                    let hir_expr = build_hir_expression(expr_child, source_code)?;
                    hir_nodes.push(hir_expr);
                }
            }
            "if_statement" => {
                let condition_node =
                    top_level_node
                        .child_by_field_name("condition")
                        .ok_or_else(|| HirBuildError::MissingField {
                            node_type: "if_statement".to_string(),
                            field_name: "condition".to_string(),
                        })?;
                let condition_expr = build_hir_expression(condition_node, source_code)?;
                let consequence_node = top_level_node
                    .child_by_field_name("consequence")
                    .ok_or_else(|| HirBuildError::MissingField {
                        node_type: "if_statement".to_string(),
                        field_name: "consequence".to_string(),
                    })?;

                let mut then_body = Vec::new();
                let mut then_cursor = consequence_node.walk();
                for stmt in consequence_node.children(&mut then_cursor) {
                    if stmt.kind() == "expression_statement" {
                        if let Some(expr_child) = stmt.named_child(0) {
                            let hir_expr = build_hir_expression(expr_child, source_code)?;
                            then_body.push(hir_expr);
                        }
                    }
                }

                // Handle optional else block
                let else_body = None;
                let hir_if = HirIf {
                    condition: Box::new(condition_expr),
                    then_body,
                    else_body,
                };
                hir_nodes.push(HirNode::If(Box::new(hir_if)));
            }
            "function_call" => {
                let hir_func_call = build_hir_expression(top_level_node, source_code)?;
                hir_nodes.push(hir_func_call);
            }
            "while_statement" => {
                let condition_node =
                    top_level_node
                        .child_by_field_name("condition")
                        .ok_or_else(|| HirBuildError::MissingField {
                            node_type: "while_statement".to_string(),
                            field_name: "condition".to_string(),
                        })?;
                let condition_expr = build_hir_expression(condition_node, source_code)?;

                let body_node = top_level_node.child_by_field_name("body").ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "while_statement".to_string(),
                        field_name: "body".to_string(),
                    }
                })?;

                let mut body_vec = Vec::new();
                let mut body_cursor = body_node.walk();
                for stmt in body_node.children(&mut body_cursor) {
                    eprintln!("DEBUG HIR while body stmt.kind(): {}", stmt.kind());
                    match stmt.kind() {
                        "expression_statement" => {
                            if let Some(expr_child) = stmt.named_child(0) {
                                let hir_expr = build_hir_expression(expr_child, source_code)?;
                                body_vec.push(hir_expr);
                            }
                        }
                        "variable_declaration" => {
                            let var_name_node =
                                stmt.child_by_field_name("name").ok_or_else(|| {
                                    HirBuildError::MissingField {
                                        node_type: "variable_declaration".to_string(),
                                        field_name: "name".to_string(),
                                    }
                                })?;
                            let var_name =
                                get_node_text(var_name_node, source_code).ok_or_else(|| {
                                    HirBuildError::MissingField {
                                        node_type: "identifier".to_string(),
                                        field_name: "text".to_string(),
                                    }
                                })?;
                            let type_node = stmt.child_by_field_name("type").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "variable_declaration".to_string(),
                                    field_name: "type".to_string(),
                                }
                            })?;
                            let type_name_str = get_node_text(type_node, source_code);
                            let value_cst_node =
                                stmt.child_by_field_name("value").ok_or_else(|| {
                                    HirBuildError::MissingField {
                                        node_type: "variable_declaration".to_string(),
                                        field_name: "value".to_string(),
                                    }
                                })?;
                            let hir_value_node = build_hir_expression(value_cst_node, source_code)?;
                            let hir_var_decl = HirVariableDeclaration {
                                name: var_name,
                                type_name: type_name_str,
                                value: Box::new(hir_value_node),
                            };
                            body_vec.push(HirNode::VariableDeclaration(Box::new(hir_var_decl)));
                        }
                        "assignment_statement" => {
                            eprintln!("DEBUG HIR while body: processing assignment_statement");
                            // Handle assignment: counter = counter + 1;
                            if let Some(var_node) = stmt.child_by_field_name("left") {
                                if let Some(value_node) = stmt.child_by_field_name("right") {
                                    let var_name = get_node_text(var_node, source_code)
                                        .ok_or_else(|| HirBuildError::MissingField {
                                            node_type: "assignment_statement".to_string(),
                                            field_name: "variable_name".to_string(),
                                        })?;
                                    let value_expr = build_hir_expression(value_node, source_code)?;
                                    let assignment = HirAssignment {
                                        left: var_name.clone(),
                                        right: Box::new(value_expr),
                                    };
                                    eprintln!(
                                        "DEBUG HIR while body: created assignment {} = ...",
                                        var_name
                                    );
                                    body_vec.push(HirNode::Assignment(Box::new(assignment)));
                                }
                            }
                        }
                        // Skip braces, comments, and other non-statement tokens
                        "{" | "}" | "comment" => {}
                        _ => {
                            eprintln!(
                                "Warning: Unhandled while body statement type: {}",
                                stmt.kind()
                            );
                        }
                    }
                }

                let hir_while = HirWhile {
                    condition: Box::new(condition_expr),
                    body: body_vec,
                };
                hir_nodes.push(HirNode::While(Box::new(hir_while)));
            }
            "assignment_statement" => {
                // Handle top-level assignment: counter = counter + 1;
                if let Some(var_node) = top_level_node.child_by_field_name("left") {
                    if let Some(value_node) = top_level_node.child_by_field_name("right") {
                        let var_name = get_node_text(var_node, source_code).ok_or_else(|| {
                            HirBuildError::MissingField {
                                node_type: "assignment_statement".to_string(),
                                field_name: "variable_name".to_string(),
                            }
                        })?;
                        let value_expr = build_hir_expression(value_node, source_code)?;
                        let assignment = HirAssignment {
                            left: var_name,
                            right: Box::new(value_expr),
                        };
                        hir_nodes.push(HirNode::Assignment(Box::new(assignment)));
                    }
                }
            }
            "match_statement" => {
                // Handle match statement: match value { patterns... }
                let value_node = top_level_node.child_by_field_name("value").ok_or_else(|| {
                    HirBuildError::MissingField {
                        node_type: "match_statement".to_string(),
                        field_name: "value".to_string(),
                    }
                })?;
                let value_expr = build_hir_expression(value_node, source_code)?;

                let mut arms = Vec::new();
                let mut cursor = top_level_node.walk();
                for child in top_level_node.children(&mut cursor) {
                    if child.kind() == "match_arm" {
                        let pattern_node =
                            child.child_by_field_name("pattern").ok_or_else(|| {
                                HirBuildError::MissingField {
                                    node_type: "match_arm".to_string(),
                                    field_name: "pattern".to_string(),
                                }
                            })?;
                        let pattern_expr = build_hir_expression(pattern_node, source_code)?;

                        // Find the body (either block_statement or expression)
                        let body_exprs = if let Some(block_node) = child.named_child(1) {
                            if block_node.kind() == "block_statement" {
                                // Extract expressions from block
                                let mut block_exprs = Vec::new();
                                let mut block_cursor = block_node.walk();
                                for stmt in block_node.children(&mut block_cursor) {
                                    if stmt.kind() == "expression_statement" {
                                        if let Some(expr_child) = stmt.named_child(0) {
                                            block_exprs.push(build_hir_expression(
                                                expr_child,
                                                source_code,
                                            )?);
                                        }
                                    }
                                }
                                block_exprs
                            } else {
                                vec![build_hir_expression(block_node, source_code)?]
                            }
                        } else {
                            return Err(HirBuildError::MissingField {
                                node_type: "match_arm".to_string(),
                                field_name: "body".to_string(),
                            });
                        };

                        let arm = HirMatchArm {
                            pattern: Box::new(pattern_expr),
                            body: body_exprs,
                        };
                        arms.push(arm);
                    }
                }

                let match_expr = HirMatch {
                    value: Box::new(value_expr),
                    arms,
                };
                hir_nodes.push(HirNode::Match(Box::new(match_expr)));
            }
            _ => {
                if top_level_node.kind() == "for_in_statement" {
                    let iter_node = top_level_node.child_by_field_name("iterator").unwrap();
                    let iterator_name = get_node_text(iter_node, source_code).unwrap_or_default();
                    let iterable_node = top_level_node.child_by_field_name("iterable").unwrap();
                    let iterable_expr = build_hir_expression(iterable_node, source_code)?;
                    let body_node = top_level_node.child_by_field_name("body").unwrap();
                    let mut inner_body = Vec::new();
                    let mut inner_cursor = body_node.walk();
                    for stmt in body_node.children(&mut inner_cursor) {
                        match stmt.kind() {
                            "break_statement" => inner_body.push(HirNode::Break),
                            "continue_statement" => inner_body.push(HirNode::Continue),
                            "expression_statement" => {
                                if let Some(expr_child) = stmt.named_child(0) {
                                    if let Ok(expr_hir) =
                                        build_hir_expression(expr_child, source_code)
                                    {
                                        inner_body.push(expr_hir);
                                    }
                                }
                            }
                            "{" | "}" | "comment" => {}
                            _ => {}
                        }
                    }
                    let for_in_hir = HirForIn {
                        iterator: iterator_name,
                        iterable: Box::new(iterable_expr),
                        body: inner_body,
                    };
                    hir_nodes.push(HirNode::ForIn(Box::new(for_in_hir)));
                } else {
                    eprintln!(
                        "Unsupported top-level node type during HIR build: {}",
                        top_level_node.kind()
                    );
                }
            }
        }
    }
    Ok(hir_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;
    // HirAskExpression is now part of HirNode::Ask, AskAttribute is used for asserts on HirAskExpression
    use crate::ask_processor::AskAttribute;

    extern "C" {
        fn tree_sitter_lexon() -> tree_sitter::Language;
    }

    fn setup_parser() -> Parser {
        let mut parser = Parser::new();
        let language = unsafe { tree_sitter_lexon() };
        parser
            .set_language(&language)
            .expect("Error loading Lexon language");
        parser
    }

    #[test]
    fn test_build_hir_for_simple_ask_var_decl() {
        let source_code = r#"
let result: Dummy = ask {
    system: "You are helpful.";
    user: "Give me data.";
    schema: MyDataSchema;
};
        "#;
        let mut parser = setup_parser();
        let tree = parser
            .parse(source_code, None)
            .expect("Failed to parse for HIR test");
        let root_node = tree.root_node();

        let hir_result = build_hir_from_cst(root_node, source_code);
        assert!(
            hir_result.is_ok(),
            "build_hir_from_cst failed: {:?}",
            hir_result.err()
        );
        let hir_nodes = hir_result.unwrap();

        assert_eq!(hir_nodes.len(), 1, "Expected 1 HIR node");
        match &hir_nodes[0] {
            HirNode::VariableDeclaration(var_decl) => {
                assert_eq!(var_decl.name, "result");
                assert_eq!(var_decl.type_name, Some("Dummy".to_string()));
                match &*var_decl.value {
                    HirNode::Ask(hir_ask) => {
                        assert_eq!(hir_ask.system_prompt, Some("You are helpful.".to_string()));
                        assert_eq!(hir_ask.user_prompt, Some("Give me data.".to_string()));
                        assert_eq!(hir_ask.output_schema_name, Some("MyDataSchema".to_string()));
                        assert!(hir_ask.attributes.is_empty());
                    }
                    _ => panic!(
                        "Expected var_decl.value to be HirNode::Ask, got {:?}",
                        var_decl.value
                    ),
                }
            }
            _ => panic!(
                "Expected HirNode::VariableDeclaration, got {:?}",
                hir_nodes[0]
            ),
        }
    }

    #[test]
    fn test_build_hir_for_ask_with_attributes_in_var_decl() {
        let source_code = r#"
let result: Dummy = ask @model("model-x") @temperature(0.5) {
    user: "Query.";
};
        "#;
        let mut parser = setup_parser();
        let tree = parser
            .parse(source_code, None)
            .expect("Failed to parse for HIR test");
        let root_node = tree.root_node();
        let hir_nodes = build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        assert_eq!(hir_nodes.len(), 1);
        match &hir_nodes[0] {
            HirNode::VariableDeclaration(var_decl) => {
                assert_eq!(var_decl.name, "result");
                assert_eq!(var_decl.type_name, Some("Dummy".to_string()));
                match &*var_decl.value {
                    HirNode::Ask(hir_ask) => {
                        assert_eq!(hir_ask.user_prompt, Some("Query.".to_string()));
                        assert_eq!(hir_ask.attributes.len(), 2);
                        assert!(hir_ask.attributes.contains(&AskAttribute {
                            name: "model".to_string(),
                            value: Some("model-x".to_string())
                        }));
                        assert!(hir_ask.attributes.contains(&AskAttribute {
                            name: "temperature".to_string(),
                            value: Some("0.5".to_string())
                        }));
                    }
                    _ => panic!("Expected var_decl.value to be HirNode::Ask"),
                }
            }
            _ => panic!("Expected HirNode::VariableDeclaration"),
        }
    }

    #[test]
    fn test_build_hir_for_string_literal_var_decl() {
        let source_code = r#"let message: String = "Hello, World!";"#;
        let mut parser = setup_parser();
        let tree = parser.parse(source_code, None).expect("Parse failed");
        let root_node = tree.root_node();
        let hir_nodes = build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        assert_eq!(hir_nodes.len(), 1);
        match &hir_nodes[0] {
            HirNode::VariableDeclaration(var_decl) => {
                assert_eq!(var_decl.name, "message");
                assert_eq!(var_decl.type_name, Some("String".to_string()));
                match &*var_decl.value {
                    HirNode::Literal(HirLiteral::String(s)) => {
                        assert_eq!(s, "Hello, World!");
                    }
                    _ => panic!("Expected string literal value"),
                }
            }
            _ => panic!("Expected VariableDeclaration"),
        }
    }

    #[test]
    fn test_build_hir_for_integer_literal_var_decl() {
        let source_code = r#"let count: int = 123;"#;
        let mut parser = setup_parser();
        let tree = parser.parse(source_code, None).expect("Parse failed");
        let root_node = tree.root_node();
        let hir_nodes = build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        assert_eq!(hir_nodes.len(), 1);
        match &hir_nodes[0] {
            HirNode::VariableDeclaration(var_decl) => {
                assert_eq!(var_decl.name, "count");
                assert_eq!(var_decl.type_name, Some("int".to_string())); // Assuming 'int' type from grammar for now
                match &*var_decl.value {
                    HirNode::Literal(HirLiteral::Integer(i)) => {
                        assert_eq!(*i, 123);
                    }
                    _ => panic!("Expected integer literal value"),
                }
            }
            _ => panic!("Expected VariableDeclaration"),
        }
    }

    #[test]
    fn test_build_hir_skips_other_top_level_statements() {
        let source_code = r#"
fn my_func() {}
schema MySchema { id: int; }
let greeting: String = "Hi"; // This should be processed
        "#;
        let mut parser = setup_parser();
        let tree = parser
            .parse(source_code, None)
            .expect("Failed to parse for HIR test");
        let root_node = tree.root_node();
        let hir_nodes = build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        // Expect 3 nodes now: function, schema, and the variable declaration
        assert_eq!(
            hir_nodes.len(),
            3,
            "Expected 3 HIR nodes (function, schema, variable)"
        );

        // Unordered check: ensure each expected node type exists
        let mut has_func = false;
        let mut has_schema = false;
        let mut has_var = false;
        for node in &hir_nodes {
            match node {
                HirNode::FunctionDefinition(fd) => {
                    has_func = true;
                    assert_eq!(fd.name, "my_func");
                }
                HirNode::SchemaDefinition(sd) => {
                    has_schema = true;
                    assert_eq!(sd.name, "MySchema");
                }
                HirNode::VariableDeclaration(vd) => {
                    has_var = true;
                    assert_eq!(vd.name, "greeting");
                }
                _ => {}
            }
        }

        assert!(
            has_func && has_schema && has_var,
            "Did not find all expected node types"
        );
    }

    #[test]
    fn test_build_hir_for_function_definition_with_body() {
        let source_code = r#"fn myFunc() {
    let x: int = 1;
}"#;
        let mut parser = setup_parser();
        let tree = parser.parse(source_code, None).expect("Parse failed");
        let root_node = tree.root_node();
        let hir_nodes = build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        assert_eq!(hir_nodes.len(), 1);
        match &hir_nodes[0] {
            HirNode::FunctionDefinition(func_def) => {
                assert_eq!(func_def.name, "myFunc");
                assert!(func_def.return_type.is_none());
                assert_eq!(func_def.body.len(), 1);
                match &func_def.body[0] {
                    HirNode::VariableDeclaration(vd) => {
                        assert_eq!(vd.name, "x");
                    }
                    _ => panic!("Expected variable declaration inside function body"),
                }
            }
            _ => panic!("Expected FunctionDefinition node"),
        }
    }

    #[test]
    fn test_build_hir_for_schema_definition() {
        let source_code = r#"schema Person {
    name: string;
    age?: int = 30;
}"#;
        let mut parser = setup_parser();
        let tree = parser.parse(source_code, None).expect("Parse failed");
        let root_node = tree.root_node();
        let hir_nodes = build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        assert_eq!(hir_nodes.len(), 1);
        match &hir_nodes[0] {
            HirNode::SchemaDefinition(schema_def) => {
                assert_eq!(schema_def.name, "Person");
                assert_eq!(schema_def.fields.len(), 2);
                let field1 = &schema_def.fields[0];
                assert_eq!(field1.name, "name");
                assert_eq!(field1.type_name, "string");
                assert!(!field1.is_optional);
                assert!(field1.default_value.is_none());
                let field2 = &schema_def.fields[1];
                assert_eq!(field2.name, "age");
                assert_eq!(field2.type_name, "int");
                assert!(field2.is_optional);
                assert!(field2.default_value.is_some());
            }
            _ => panic!("Expected SchemaDefinition node"),
        }
    }

    #[test]
    fn test_build_hir_for_for_in_statement() {
        let source_code = r#"fn loopTest() {
    for x in numbers {
        break;
    }
}"#;
        let mut parser = setup_parser();
        let tree = parser.parse(source_code, None).expect("Parse failed");
        let root_node = tree.root_node();
        let hir_nodes =
            super::build_hir_from_cst(root_node, source_code).expect("HIR build failed");

        assert_eq!(hir_nodes.len(), 1);
        match &hir_nodes[0] {
            HirNode::FunctionDefinition(func_def) => {
                assert_eq!(func_def.name, "loopTest");
                assert_eq!(func_def.body.len(), 1);
                match &func_def.body[0] {
                    HirNode::ForIn(for_in) => {
                        assert_eq!(for_in.iterator, "x");
                    }
                    _ => panic!("Expected ForIn node"),
                }
            }
            _ => panic!("Expected FunctionDefinition node"),
        }
    }
}
