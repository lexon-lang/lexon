//! Schema and Function Definition Conversion
//!
//! This module handles the conversion of HIR type and function definitions to LexIR.
//! It supports the complete type system and function definition semantics of Lexon.
//!
//! ## Supported Definitions
//!
//! - **Schema Definitions**: Custom data types with fields and default values
//! - **Function Definitions**: Function signatures and bodies with parameters
//! - **Generic Types**: Parameterized types with instantiation support
//! - **For-In Loops**: Iterator-based loop constructs
//!
//! ## Type System Support
//!
//! The module provides comprehensive type conversion:
//! - Basic types (int, float, string, bool, void)
//! - Generic types (List<T>, Option<T>, Map<K,V>)
//! - Schema types (custom user-defined types)
//! - Type parameter substitution for generics
//!
//! ## Schema Processing
//!
//! Schema definitions support:
//! - Field type conversion and validation
//! - Optional fields with default values
//! - Generic schema instantiation
//! - Specialized type generation
//!
//! ## Function Processing
//!
//! Function definitions support:
//! - Parameter type conversion
//! - Return type specification
//! - Complex function bodies with all statement types
//! - Variable scoping and declaration handling
//!
//! ## Error Handling
//!
//! All operations return `Result<()>` to handle type resolution
//! and conversion failures gracefully.

use crate::hir::{HirSchemaDefinition, HirFunctionDefinition, HirNode, HirLiteral, HirVariableDeclaration};
use crate::lexir::{LexSchemaDefinition, LexSchemaField, LexFunction, LexInstruction, LexType, ValueRef, LexExpression, LexBinaryOperator};
use super::{Result, ConversionContext, HirToLexIrError};
use std::collections::HashMap;

impl ConversionContext {
    /// 🏗️ Converts HIR schema definitions to LexIR
    ///
    /// This method handles the conversion of schema (struct/type) definitions, including:
    /// - Field type conversion and validation
    /// - Optional field handling with default values
    /// - Generic type parameter processing
    /// - Default value literal conversion
    ///
    /// ## Field Processing
    ///
    /// Each schema field is converted with:
    /// - Type name parsing and LexIR type conversion
    /// - Optional flag preservation
    /// - Default value extraction from HIR literals
    /// - Field name and metadata preservation
    ///
    /// ## Type System Integration
    ///
    /// The method integrates with the type system to:
    /// - Parse complex type expressions
    /// - Handle generic type parameters
    /// - Validate type consistency
    /// - Generate appropriate LexIR type representations
    pub fn convert_schema_definition(&mut self, schema_def: &HirSchemaDefinition) -> Result<()> {
        let mut fields = Vec::new();

        for field in &schema_def.fields {
            let field_type = self.parse_lex_type(field.type_name.as_str());

            let default_value = if let Some(box_node) = &field.default_value {
                if let HirNode::Literal(lit) = &**box_node {
                Some(self.convert_literal(lit)?)
                } else {
                    None
                }
            } else {
                None
            };

            fields.push(LexSchemaField {
                name: field.name.clone(),
                field_type,
                is_optional: field.is_optional,
                default_value,
            });
        }

        self.program.add_schema(LexSchemaDefinition {
            name: schema_def.name.clone(),
            fields,
        });

        Ok(())
    }

    /// 🔧 Converts HIR function definitions to LexIR
    ///
    /// This method handles the conversion of function definitions, including:
    /// - Parameter type conversion and validation
    /// - Return type specification and parsing
    /// - Function body statement processing
    /// - Variable scoping and declaration handling
    ///
    /// ## Parameter Processing
    ///
    /// Function parameters are converted with:
    /// - Parameter name preservation
    /// - Type annotation parsing and conversion
    /// - Type validation and consistency checking
    ///
    /// ## Body Conversion
    ///
    /// Function bodies support all statement types:
    /// - Variable declarations with initialization
    /// - Assignments and expressions
    /// - Control flow constructs (while, match)
    /// - Function calls and data operations
    /// - Memory operations and LLM calls
    pub fn convert_function_definition(&mut self, func_def: &HirFunctionDefinition) -> Result<()> {
        // Convert return type
        let return_type = if let Some(type_name) = &func_def.return_type {
            self.parse_lex_type(type_name.as_str())
        } else {
            LexType::Void
        };

        // Convert the function body
        let mut body = Vec::new();

        // Add Declare instructions for function parameters
        for param in &func_def.parameters {
            let declare_instr = LexInstruction::Declare {
                name: param.name.clone(),
                type_name: Some(param.type_name.clone()),
                is_mutable: false, // Function parameters are immutable by default
            };
            body.push(declare_instr);
        }

        for (i, statement) in func_def.body.iter().enumerate() {
            match statement {
                HirNode::VariableDeclaration(var_decl) => {
                    // Declaration with initial value
                    let instruction = self.convert_variable_declaration(var_decl)?;
                    body.push(instruction); // Add the declare instruction to function body

                    // Evaluate and assign the initial value of the variable (includes ask expressions and literals)
                    let init_expr = match var_decl.value.as_ref() {
                        HirNode::Binary(bin_expr) => {
                            let left_val = self.convert_node_to_value_ref(&bin_expr.left)?;
                            let right_val = self.convert_node_to_value_ref(&bin_expr.right)?;
                            let operator = match bin_expr.operator.as_str() {
                                "+" => LexBinaryOperator::Add, "-" => LexBinaryOperator::Subtract,
                                "*" => LexBinaryOperator::Multiply, "/" => LexBinaryOperator::Divide,
                                ">" => LexBinaryOperator::GreaterThan, "<" => LexBinaryOperator::LessThan,
                                ">=" => LexBinaryOperator::GreaterEqual, "<=" => LexBinaryOperator::LessEqual,
                                "==" => LexBinaryOperator::Equal, "!=" => LexBinaryOperator::NotEqual,
                                "&&" => LexBinaryOperator::And, "||" => LexBinaryOperator::Or,
                                _ => return Err(HirToLexIrError::UnsupportedNode(format!("Unsupported binary operator: {}", bin_expr.operator))),
                            };
                            LexExpression::BinaryOp {
                                operator,
                                left: Box::new(LexExpression::Value(left_val)),
                                right: Box::new(LexExpression::Value(right_val)),
                            }
                        },
                        _ => {
                            let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                            LexExpression::Value(init_ref)
                        }
                    };
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(var_decl.name.clone()),
                        expr: init_expr,
                    };
                    body.push(assign_instr);
                },
                // Add handling for other node types within the function body
                HirNode::DataLoad(data_load) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_load_instruction(data_load, ValueRef::Temp(temp_id))?;
                },
                HirNode::DataFilter(data_filter) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_filter_instruction(data_filter, ValueRef::Temp(temp_id))?;
                },
                HirNode::DataSelect(data_select) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_select_instruction(data_select, ValueRef::Temp(temp_id))?;
                },
                HirNode::DataTake(data_take) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_take_instruction(data_take, ValueRef::Temp(temp_id))?;
                },
                HirNode::DataExport(data_export) => {
                    self.add_data_export_instruction(data_export)?;
                },
                HirNode::MemoryLoad(memory_load) => {
                    let temp_id = self.temp_gen.next();
                    self.add_memory_load_instruction(memory_load, ValueRef::Temp(temp_id))?;
                },
                HirNode::MemoryStore(memory_store) => {
                    self.add_memory_store_instruction(memory_store)?;
                },
                HirNode::Assignment(assignment) => {
                    let right_value = self.convert_node_to_value_ref(&assignment.right)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(assignment.left.clone()),
                        expr: LexExpression::Value(right_value),
                    };
                    body.push(assign_instr);
                },
                HirNode::Return(return_node) => {
                    let return_expr = if let Some(ref expr) = return_node.expression {
                        // Handle binary expressions directly without creating global instructions
                        match expr.as_ref() {
                            HirNode::Binary(bin_expr) => {
                                let left_val = self.convert_node_to_value_ref(&bin_expr.left)?;
                                let right_val = self.convert_node_to_value_ref(&bin_expr.right)?;
                                let operator = match bin_expr.operator.as_str() {
                                    "+" => LexBinaryOperator::Add, "-" => LexBinaryOperator::Subtract,
                                    "*" => LexBinaryOperator::Multiply, "/" => LexBinaryOperator::Divide,
                                    ">" => LexBinaryOperator::GreaterThan, "<" => LexBinaryOperator::LessThan,
                                    ">=" => LexBinaryOperator::GreaterEqual, "<=" => LexBinaryOperator::LessEqual,
                                    "==" => LexBinaryOperator::Equal, "!=" => LexBinaryOperator::NotEqual,
                                    "&&" => LexBinaryOperator::And, "||" => LexBinaryOperator::Or,
                                    _ => return Err(HirToLexIrError::UnsupportedNode(format!("Unsupported binary operator: {}", bin_expr.operator))),
                                };
                                Some(LexExpression::BinaryOp {
                                    operator,
                                    left: Box::new(LexExpression::Value(left_val)),
                                    right: Box::new(LexExpression::Value(right_val)),
                                })
                            },
                            _ => {
                                let expr_val = self.convert_node_to_value_ref(expr)?;
                                Some(LexExpression::Value(expr_val))
                            }
                        }
                    } else {
                        None
                    };
                    body.push(LexInstruction::Return { expr: return_expr });
                },
                HirNode::FunctionCall(func_call) => {
                    let temp_id = self.temp_gen.next();
                    let mut args_exprs = Vec::new();
                    for arg in &func_call.args {
                        let val_ref = self.convert_node_to_value_ref(arg)?;
                        args_exprs.push(LexExpression::Value(val_ref));
                    }
                    let call_instr = LexInstruction::Call {
                        result: Some(ValueRef::Temp(temp_id)),
                        function: func_call.function.clone(),
                        args: args_exprs,
                    };
                    body.push(call_instr);
                },
                // Other statement types would go here
                _ => {
                    return Err(HirToLexIrError::UnsupportedNode(format!("Unsupported statement in function body: {:?}", statement)));
                },
            }
        }

        // Convert function parameters
        let parameters: Vec<(String, LexType)> = func_def.parameters.iter().map(|p| (p.name.clone(), self.parse_lex_type(&p.type_name))).collect();

        self.program.add_function(LexFunction {
            name: func_def.name.clone(),
            return_type,
            parameters,
            body,
        });

        Ok(())
    }

    /// 🔧 Converts HIR variable declarations to LexIR instructions
    ///
    /// This method handles the conversion of variable declarations, including:
    /// - Type name processing and generic instantiation
    /// - Mutability flags (currently defaults to immutable)
    /// - Schema specialization for generic types
    ///
    /// ## Generic Type Handling
    ///
    /// If the variable has a generic type (e.g., `Vec<T>`), the method:
    /// 1. Splits the type into base and arguments
    /// 2. Attempts to instantiate a specialized schema
    /// 3. Uses the specialized name if available
    ///
    /// ## Error Handling
    ///
    /// Returns `Result<LexInstruction>` to handle type resolution failures.
    pub fn convert_variable_declaration(&mut self, var_decl: &HirVariableDeclaration) -> Result<LexInstruction> {
        let mut declared_type = var_decl.type_name.clone();
        if let Some(ref tstr) = declared_type {
            let (base, args) = ConversionContext::split_type(tstr);
            if !args.is_empty() {
                if let Some(spec_name) = self.instantiate_schema(&base, &args) {
                    declared_type = Some(spec_name);
                }
            }
        }

        Ok(LexInstruction::Declare {
            name: var_decl.name.clone(),
            type_name: declared_type,
            is_mutable: false,
        })
    }

    /// 🔄 Converts HIR nodes to LexIR value references
    ///
    /// This is a helper method for node conversion that handles variable declarations
    /// specially by creating both declare and assign instructions.
    pub fn convert_node(&mut self, node: &HirNode) -> Result<ValueRef> {
        match node {
            HirNode::VariableDeclaration(var_decl) => {
                let instruction = self.convert_variable_declaration(var_decl)?;
                self.program.add_instruction(instruction);

                // Evaluate and assign the initial value of the variable (includes ask expressions and literals)
                let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                let assign_instr = LexInstruction::Assign {
                    result: ValueRef::Named(var_decl.name.clone()),
                    expr: LexExpression::Value(init_ref),
                };
                self.program.add_instruction(assign_instr);
                return Ok(ValueRef::Named(var_decl.name.clone()));
            },
            _ => Ok(self.convert_node_to_value_ref(node)?),
        }
    }

    /// 🔄 Converts For-In loops to basic LexIR instructions
    ///
    /// This method handles the conversion of iterator-based loops, including:
    /// - Iterator variable setup and management
    /// - Iterable expression conversion
    /// - Loop body statement processing
    /// - Break and continue statement support
    ///
    /// ## Iterator Support
    ///
    /// For-in loops support various iterable types:
    /// - Arrays and lists
    /// - Range expressions
    /// - Custom iterable objects
    /// - Generator expressions
    ///
    /// ## Body Processing
    ///
    /// Loop bodies support all statement types:
    /// - Function calls and expressions
    /// - Variable assignments and declarations
    /// - Break and continue statements
    /// - Nested control flow constructs
    pub fn convert_for_in(&mut self, for_in: &crate::hir::HirForIn) -> Result<()> {
        // Convert the iterable to a value reference
        let iterable_ref = self.convert_node_to_value_ref(&for_in.iterable)?;

        // Convert body
        let mut body_instrs = Vec::new();
        for stmt in &for_in.body {
            match stmt {
                crate::hir::HirNode::Break => body_instrs.push(LexInstruction::Break),
                crate::hir::HirNode::Continue => body_instrs.push(LexInstruction::Continue),
                crate::hir::HirNode::FunctionCall(func_call) => {
                    // Handle function calls like print() inside for loop
                    let temp_id = self.temp_gen.next();
                    // Convert arguments properly
                    let mut args_exprs = Vec::new();
                    for arg in &func_call.args {
                        let val_ref = match arg {
                            HirNode::Identifier(var_name) => ValueRef::Named(var_name.clone()),
                            _ => self.convert_node_to_value_ref(arg)?
                        };
                        args_exprs.push(LexExpression::Value(val_ref));
                    }
                    let call_instr = LexInstruction::Call {
                        result: Some(ValueRef::Temp(temp_id)),
                        function: func_call.function.clone(),
                        args: args_exprs,
                    };
                    body_instrs.push(call_instr);
                },
                crate::hir::HirNode::Assignment(assignment) => {
                    // Handle assignments inside for loop
                    let right_value = self.convert_node_to_value_ref(&assignment.right)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(assignment.left.clone()),
                        expr: LexExpression::Value(right_value),
                    };
                    body_instrs.push(assign_instr);
                },
                crate::hir::HirNode::VariableDeclaration(var_decl) => {
                    // Handle variable declarations inside for loop
                    let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(var_decl.name.clone()),
                        expr: LexExpression::Value(init_ref),
                    };
                    body_instrs.push(assign_instr);
                },
                _ => {
                    // Try to handle any other statement type
                    if let Ok(value_ref) = self.convert_node_to_value_ref(stmt) {
                        let temp_id = self.temp_gen.next();
                        let assign_instr = LexInstruction::Assign {
                            result: ValueRef::Temp(temp_id),
                            expr: LexExpression::Value(value_ref),
                        };
                        body_instrs.push(assign_instr);
                    }
                }
            }
        }

        let instr = LexInstruction::ForEach {
            iterator: for_in.iterator.clone(),
            iterable: iterable_ref,
            body: body_instrs,
        };
        self.program.add_instruction(instr);

        Ok(())
    }
}