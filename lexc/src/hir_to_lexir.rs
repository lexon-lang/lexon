// lexc/src/hir_to_lexir.rs
//
// 🔄 HIR to LexIR Converter - Core Compilation Pipeline
//
// This module implements the critical conversion from the High-Level Intermediate Representation (HIR)
// to the Lexon Intermediate Representation (LexIR), which is closer to the target executable code.
//
// ## Architecture Overview
//
// The conversion process follows a two-phase approach:
// 1. **Schema and Function Processing**: Converts type definitions and function signatures
// 2. **Instruction Generation**: Converts HIR nodes to executable LexIR instructions
//
// ## Key Components
//
// - `ConversionContext`: Manages state during conversion including temporary ID generation
// - `TempIdGenerator`: Provides unique temporary variable IDs for intermediate values
// - Node conversion methods: Handle specific HIR node types (literals, expressions, statements)
// - Type system integration: Bridges HIR types with LexIR type system
//
// ## Supported HIR Node Types
//
// - **Literals**: String, Integer, Float, Boolean, Array
// - **Expressions**: Binary operations, function calls, method calls
// - **Statements**: Variable declarations, assignments, control flow
// - **LLM Operations**: Ask expressions, memory operations, data operations
// - **Control Flow**: If/else, while loops, pattern matching
// - **Advanced Features**: Async/await, generics, type inference
//
// ## Error Handling
//
// The conversion process uses Result types to handle errors gracefully:
// - `UnsupportedNode`: For HIR nodes not yet implemented
// - `InvalidExpression`: For malformed expressions
// - `MissingType`: For type resolution failures

use crate::hir::{
    HirAskExpression, HirFunctionDefinition, HirLiteral, HirMatch, HirNode, HirSchemaDefinition,
    HirVariableDeclaration, HirWhile,
};
use crate::hir::{
    HirDataExport, HirDataFilter, HirDataLoad, HirDataSelect, HirDataTake, HirMemoryLoad,
    HirMemoryStore,
};
use crate::lexir::{
    LexExpression, LexFunction, LexInstruction, LexLiteral, LexProgram, LexSchemaDefinition,
    LexSchemaField, LexType, ValueRef,
};
use std::collections::HashMap;

/// 🚨 Error types for HIR to LexIR conversion
///
/// These errors represent different failure modes during the conversion process:
/// - Unsupported language features that haven't been implemented yet
/// - Invalid expressions that can't be converted to valid LexIR
/// - Type system failures where types can't be resolved
#[derive(Debug)]
pub enum HirToLexIrError {
    /// A HIR node type that is not yet supported in the conversion process
    UnsupportedNode(String),
    /// An expression that is malformed or cannot be converted to valid LexIR
    InvalidExpression(String),
    /// A type that cannot be resolved or is missing required information
    MissingType(String),
}

/// 🔧 Result type alias for HIR to LexIR conversion operations
pub type Result<T> = std::result::Result<T, HirToLexIrError>;

/// 🔢 Temporary ID generator for intermediate values during conversion
///
/// During HIR to LexIR conversion, we often need to create temporary variables
/// to hold intermediate computation results. This generator ensures unique IDs.
///
/// ## Usage
///
/// ```rust
/// let mut temp_gen = TempIdGenerator::new();
/// let temp1 = temp_gen.next(); // TempId(0)
/// let temp2 = temp_gen.next(); // TempId(1)
/// ```
struct TempIdGenerator {
    next_id: u32,
}

impl TempIdGenerator {
    /// Creates a new temporary ID generator starting from 0
    fn new() -> Self {
        TempIdGenerator { next_id: 0 }
    }

    /// Generates the next unique temporary ID
    fn next(&mut self) -> crate::lexir::TempId {
        let id = self.next_id;
        self.next_id += 1;
        crate::lexir::TempId(id)
    }
}

/// 🏗️ Conversion context for HIR to LexIR transformation
///
/// This structure maintains all the state needed during the conversion process:
/// - Temporary ID generation for intermediate values
/// - The target LexIR program being built
/// - Generic type and function definitions for instantiation
///
/// ## Architecture
///
/// The conversion context acts as a stateful converter that processes HIR nodes
/// and accumulates LexIR instructions in the target program. It handles:
///
/// - **Temporary Management**: Generates unique IDs for intermediate values
/// - **Program Building**: Accumulates instructions and definitions
/// - **Generic Resolution**: Manages generic schemas and functions
/// - **Type Conversion**: Bridges HIR and LexIR type systems
struct ConversionContext {
    /// Generator for unique temporary variable IDs
    temp_gen: TempIdGenerator,
    /// The target LexIR program being constructed
    program: LexProgram,
    /// Generic schema definitions available for instantiation
    generic_schemas: HashMap<String, HirSchemaDefinition>,
    /// Generic function definitions available for instantiation
    generic_functions: HashMap<String, HirFunctionDefinition>,
    /// Current module prefix (normalized with __), empty means root
    module_prefix: String,
    /// Module alias map: alias -> full module path with '::' (e.g., "math" -> "lib::math")
    module_aliases: HashMap<String, String>,
    /// Item alias map: alias -> full item path with '::' (e.g., "dbl" -> "lib::math::double")
    item_aliases: HashMap<String, String>,
}

impl ConversionContext {
    /// 🏗️ Creates a new conversion context with empty state
    ///
    /// Initializes all components needed for HIR to LexIR conversion:
    /// - Fresh temporary ID generator
    /// - Empty LexIR program
    /// - Empty generic type registries
    fn new() -> Self {
        ConversionContext {
            temp_gen: TempIdGenerator::new(),
            program: LexProgram::new(),
            generic_schemas: HashMap::new(),
            generic_functions: HashMap::new(),
            module_prefix: String::new(),
            module_aliases: HashMap::new(),
            item_aliases: HashMap::new(),
        }
    }

    /// Expand leading module alias in a qualified name, if present.
    /// Examples:
    ///  - "math::double" with alias math="lib::math" -> "lib::math::double"
    ///  - "User" or non-qualified names remain unchanged
    fn expand_module_aliases(&self, qualified: &str) -> String {
        let parts: Vec<&str> = qualified.split("::").collect();
        if let Some(first) = parts.first().cloned() {
            if let Some(full) = self.module_aliases.get(first) {
                // Replace first segment with full path segments
                let mut expanded: Vec<String> = full.split("::").map(|s| s.to_string()).collect();
                if parts.len() > 1 {
                    expanded.extend(parts.iter().skip(1).map(|s| s.to_string()));
                }
                return expanded.join("::");
            }
        }
        qualified.to_string()
    }

    /// Expand item alias (unqualified names) to full path if present.
    /// Example: "dbl" with alias "dbl"="lib::math::double" -> "lib::math::double"
    fn expand_item_alias(&self, name: &str) -> String {
        if let Some(full) = self.item_aliases.get(name) {
            full.clone()
        } else {
            name.to_string()
        }
    }

    /// Expand either module or item alias depending on the form
    fn expand_any_alias(&self, name: &str) -> String {
        if name.contains("::") {
            self.expand_module_aliases(name)
        } else {
            self.expand_item_alias(name)
        }
    }

    /// 🔤 Converts HIR literals to LexIR literals
    ///
    /// This method handles the conversion of all supported literal types:
    /// - String literals (including multi-line strings)
    /// - Numeric literals (integers and floats)
    /// - Boolean literals
    /// - Array literals (with recursive element conversion)
    ///
    /// ## Array Conversion
    ///
    /// For arrays, we recursively convert each element:
    /// - Literal elements are converted directly
    /// - Non-literal elements are converted to string representation
    ///
    /// ## Error Handling
    ///
    /// Returns `Result<LexLiteral>` to handle conversion failures gracefully.
    #[allow(clippy::only_used_in_recursion)]
    fn convert_literal(&self, literal: &HirLiteral) -> Result<LexLiteral> {
        match literal {
            HirLiteral::String(s) => Ok(LexLiteral::String(s.clone())),
            HirLiteral::MultiLineString(s) => Ok(LexLiteral::String(s.clone())),
            HirLiteral::Integer(i) => Ok(LexLiteral::Integer(*i)),
            HirLiteral::Float(f) => Ok(LexLiteral::Float(*f)),
            HirLiteral::Boolean(b) => Ok(LexLiteral::Boolean(*b)),
            HirLiteral::Array(elements) => {
                // Convert array elements to LexLiteral::Array
                let mut lex_elements = Vec::new();
                for element in elements {
                    match element {
                        HirNode::Literal(lit) => {
                            lex_elements.push(self.convert_literal(lit)?);
                        }
                        _ => {
                            // For non-literal elements, convert to string representation
                            lex_elements.push(LexLiteral::String(format!("{:?}", element)));
                        }
                    }
                }
                Ok(LexLiteral::Array(lex_elements))
            }
        }
    }

    /// 🔄 Converts HIR nodes to LexIR value references
    ///
    /// This is the core conversion method that handles all HIR node types and converts
    /// them to appropriate LexIR value references. It's responsible for:
    ///
    /// - **Literal Conversion**: Direct mapping of literal values
    /// - **Expression Evaluation**: Converting complex expressions to temporary values
    /// - **LLM Operations**: Handling ask expressions and memory operations
    /// - **Data Operations**: Processing data load/filter/select operations
    /// - **Binary Operations**: Converting mathematical and logical operations
    /// - **Function Calls**: Handling both regular and method calls
    ///
    /// ## Value Reference Types
    ///
    /// The method returns different types of value references:
    /// - `ValueRef::Literal`: For direct literal values
    /// - `ValueRef::Temp`: For computed expressions requiring temporary storage
    /// - `ValueRef::Named`: For variable references
    ///
    /// ## Temporary Value Generation
    ///
    /// For complex expressions, the method generates temporary variables and
    /// adds corresponding instructions to the program. This ensures proper
    /// evaluation order and intermediate value storage.
    ///
    /// ## Error Handling
    ///
    /// Returns `Result<ValueRef>` to handle unsupported nodes or conversion failures.
    fn convert_node_to_value_ref(&mut self, node: &HirNode) -> Result<ValueRef> {
        match node {
            // 🔤 Direct literal conversion
            HirNode::Literal(lit) => {
                let lex_lit = self.convert_literal(lit)?;
                Ok(ValueRef::Literal(lex_lit))
            }
            // 🤖 LLM ask expression handling
            HirNode::Ask(ask_expr) => {
                // For ask expressions, we generate additional instructions and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_ask_instruction(ask_expr, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            // 📊 Data processing operations
            HirNode::DataLoad(data_load) => {
                // For data load operations, we generate the instruction and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_data_load_instruction(data_load, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            HirNode::DataFilter(data_filter) => {
                // For filter operations, we generate the instruction and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_data_filter_instruction(data_filter, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            HirNode::DataSelect(data_select) => {
                // For select operations, we generate the instruction and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_data_select_instruction(data_select, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            HirNode::DataTake(data_take) => {
                // For take operations, we generate the instruction and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_data_take_instruction(data_take, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            // 🧠 Memory operations
            HirNode::MemoryLoad(memory_load) => {
                // For memory load operations, we generate the instruction and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_memory_load_instruction(memory_load, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            HirNode::MemoryStore(memory_store) => {
                // For memory store operations, we generate the instruction
                // It doesn't return a value, but we need to process the node
                self.add_memory_store_instruction(memory_store)?;
                // We return a boolean literal as placeholder
                Ok(ValueRef::Literal(LexLiteral::Boolean(false)))
            }
            // ➕ Binary operations (mathematical and logical)
            HirNode::Binary(bin_expr) => {
                // Evaluate operands recursively
                let left_val = self.convert_node_to_value_ref(&bin_expr.left)?;
                let right_val = self.convert_node_to_value_ref(&bin_expr.right)?;

                use crate::lexir::{LexBinaryOperator as LB, LexExpression};

                let operator = match bin_expr.operator.as_str() {
                    "+" => LB::Add,
                    "-" => LB::Subtract,
                    "*" => LB::Multiply,
                    "/" => LB::Divide,
                    ">" => LB::GreaterThan,
                    "<" => LB::LessThan,
                    ">=" => LB::GreaterEqual,
                    "<=" => LB::LessEqual,
                    "==" => LB::Equal,
                    "!=" => LB::NotEqual,
                    "&&" => LB::And,
                    "||" => LB::Or,
                    _ => {
                        return Err(HirToLexIrError::UnsupportedNode(format!(
                            "Unsupported binary operator: {}",
                            bin_expr.operator
                        )))
                    }
                };

                let expr = LexExpression::BinaryOp {
                    operator,
                    left: Box::new(LexExpression::Value(left_val.clone())),
                    right: Box::new(LexExpression::Value(right_val.clone())),
                };

                let temp_id = self.temp_gen.next();
                self.program.add_instruction(LexInstruction::Assign {
                    result: ValueRef::Temp(temp_id.clone()),
                    expr,
                });
                Ok(ValueRef::Temp(temp_id))
            }
            // 🔄 Data export operations
            HirNode::DataExport(data_export) => {
                // For export operations, we generate the instruction
                // It doesn't return a value, but we need to process the node
                self.add_data_export_instruction(data_export)?;
                // We return a boolean literal as placeholder
                Ok(ValueRef::Literal(LexLiteral::Boolean(true)))
            }
            // 📞 Method call operations
            HirNode::MethodCall(method_call) => {
                // Heuristic: static if receiver is identifier starting uppercase or builtin struct/enum
                let is_type = match &*method_call.target {
                    HirNode::Identifier(name) => {
                        let first = name.chars().next();
                        let stdmods = [
                            "struct", "enum", "encoding", "strings", "math", "regex", "time",
                            "number", "crypto", "json",
                        ];
                        stdmods.contains(&name.as_str())
                            || first.map(|c| c.is_ascii_uppercase()).unwrap_or(false)
                    }
                    _ => false,
                };

                let temp_id = self.temp_gen.next();
                if is_type {
                    let target_name = if let HirNode::Identifier(name) = &*method_call.target {
                        name.clone()
                    } else {
                        String::new()
                    };
                    let target_expanded = self.expand_any_alias(&target_name);
                    let target_norm = target_expanded.replace("::", "__").replace('.', "__");
                    let mut args_exprs = Vec::new();
                    for arg in &method_call.args {
                        let val_ref = match arg {
                            HirNode::Identifier(var_name) => ValueRef::Named(var_name.clone()),
                            _ => self.convert_node_to_value_ref(arg)?,
                        };
                        args_exprs.push(LexExpression::Value(val_ref));
                    }
                    let instr = LexInstruction::Call {
                        result: Some(ValueRef::Temp(temp_id.clone())),
                        function: format!("{}__{}", target_norm, method_call.method),
                        args: args_exprs,
                    };
                    self.program.add_instruction(instr);
                } else {
                    // Instance dispatch via runtime helper: method.call(receiver, method, ...args)
                    let mut args_exprs = Vec::new();
                    // receiver expression
                    let recv = self.convert_node_to_value_ref(&method_call.target)?;
                    args_exprs.push(LexExpression::Value(recv));
                    // method name
                    args_exprs.push(LexExpression::Value(ValueRef::Literal(LexLiteral::String(
                        method_call.method.clone(),
                    ))));
                    // remaining args
                    for arg in &method_call.args {
                        let v = self.convert_node_to_value_ref(arg)?;
                        args_exprs.push(LexExpression::Value(v));
                    }
                    self.program.add_instruction(LexInstruction::Call {
                        result: Some(ValueRef::Temp(temp_id.clone())),
                        function: "method.call".to_string(),
                        args: args_exprs,
                    });
                }
                Ok(ValueRef::Temp(temp_id))
            }
            // 🔧 Function call operations with generic instantiation support
            HirNode::FunctionCall(func_call) => {
                // Detect special memory functions
                let fn_name_expanded = self.expand_any_alias(&func_call.function);
                let fn_name_norm = fn_name_expanded.replace("::", "__").replace('.', "__");
                match fn_name_norm.as_str() {
                    "memory_store" => {
                        // Convert arguments for memory_store
                        if func_call.args.len() < 2 {
                            return Err(HirToLexIrError::InvalidExpression(
                                "memory_store requires at least 2 arguments".to_string(),
                            ));
                        }

                        // Extract scope (first argument)
                        let _scope_val = self.convert_node_to_value_ref(&func_call.args[0])?;

                        // Extract key (second argument)
                        let _key_val = self.convert_node_to_value_ref(&func_call.args[1])?;

                        // Extract value (third argument)
                        let value_val = self.convert_node_to_value_ref(&func_call.args[2])?;

                        // Generate memory store instruction
                        let instruction = LexInstruction::MemoryStore {
                            scope: match &func_call.args[0] {
                                HirNode::Literal(HirLiteral::String(s)) => s.clone(),
                                _ => {
                                    return Err(HirToLexIrError::InvalidExpression(
                                        "memory_store scope must be a string literal".to_string(),
                                    ))
                                }
                            },
                            value: value_val,
                            key: match &func_call.args[1] {
                                HirNode::Literal(HirLiteral::String(s)) => Some(s.clone()),
                                _ => None,
                            },
                            options: HashMap::new(),
                        };

                        self.program.add_instruction(instruction);
                        Ok(ValueRef::Literal(LexLiteral::Boolean(true)))
                    }
                    "memory_load" => {
                        // Convert arguments for memory_load
                        if func_call.args.len() < 2 {
                            return Err(HirToLexIrError::InvalidExpression(
                                "memory_load requires at least 2 arguments".to_string(),
                            ));
                        }

                        // Extract scope (first argument)
                        let _scope_val = self.convert_node_to_value_ref(&func_call.args[0])?;

                        // Extract key (second argument)
                        let key_val = self.convert_node_to_value_ref(&func_call.args[1])?;

                        // Generate memory load instruction
                        let temp_id = self.temp_gen.next();
                        let instruction = LexInstruction::MemoryLoad {
                            result: ValueRef::Temp(temp_id.clone()),
                            scope: match &func_call.args[0] {
                                HirNode::Literal(HirLiteral::String(s)) => s.clone(),
                                _ => {
                                    return Err(HirToLexIrError::InvalidExpression(
                                        "memory_load scope must be a string literal".to_string(),
                                    ))
                                }
                            },
                            source: Some(key_val),
                            strategy: "buffer".to_string(),
                            options: HashMap::new(),
                        };

                        self.program.add_instruction(instruction);
                        Ok(ValueRef::Temp(temp_id))
                    }
                    _ => {
                        // Handle generic function calls with potential instantiation
                        let specialized_name = if let Some(generic_func) =
                            self.generic_functions.get(&fn_name_norm)
                        {
                            let spec_name = format!("{}_{}", fn_name_norm, self.temp_gen.next().0);

                            // Create a monomorphized version
                            let mono = generic_func.clone();

                            // Store the specialized function
                            if !self.generic_functions.contains_key(&spec_name) {
                                self.generic_functions
                                    .insert(spec_name.clone(), mono.clone());

                                // Generate the specialized function
                                // TODO: future substitution of type parameters inside body
                                self.convert_function_definition(&mono)?;
                            }
                            spec_name
                        } else {
                            fn_name_norm.clone()
                        };

                        // Convert arguments
                        let mut args_exprs = Vec::new();
                        for arg in &func_call.args {
                            let val_ref = match arg {
                                HirNode::Identifier(var_name) => ValueRef::Named(var_name.clone()),
                                _ => self.convert_node_to_value_ref(arg)?,
                            };
                            args_exprs.push(LexExpression::Value(val_ref));
                        }
                        let temp_id = self.temp_gen.next();
                        let instr = LexInstruction::Call {
                            result: Some(ValueRef::Temp(temp_id.clone())),
                            function: specialized_name,
                            args: args_exprs,
                        };
                        self.program.add_instruction(instr);
                        Ok(ValueRef::Temp(temp_id))
                    }
                }
            }
            // 🔍 Type introspection operations
            HirNode::TypeOf(typeof_expr) => {
                // For typeof expressions, we evaluate the argument and return its type as a string
                let arg_val = self.convert_node_to_value_ref(&typeof_expr.argument)?;

                // Generate a typeof instruction that will be handled by the executor
                let temp_id = self.temp_gen.next();
                let instr = LexInstruction::Call {
                    result: Some(ValueRef::Temp(temp_id.clone())),
                    function: "typeof".to_string(),
                    args: vec![LexExpression::Value(arg_val)],
                };
                self.program.add_instruction(instr);
                Ok(ValueRef::Temp(temp_id))
            }
            // 🏷️ Variable identifier references
            HirNode::Identifier(name) => Ok(ValueRef::Named(name.clone())),
            // ⏳ Async await operations
            HirNode::Await(await_expr) => {
                // For await expressions, we process the inner expression
                self.convert_node_to_value_ref(&await_expr.expression)
            }
            // 🛡️ Anti-hallucination ask_safe expressions
            HirNode::AskSafe(ask_safe_expr) => {
                // For ask_safe expressions, generate additional instructions and return a temporary
                let temp_id = self.temp_gen.next();
                self.add_ask_safe_instruction(ask_safe_expr, ValueRef::Temp(temp_id.clone()))?;
                Ok(ValueRef::Temp(temp_id))
            }
            // Other node types here
            _ => Err(HirToLexIrError::UnsupportedNode(format!(
                "Unsupported node as value: {:?}",
                node
            ))),
        }
    }

    /// 📝 Converts HIR variable declarations to LexIR instructions
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
    fn convert_variable_declaration(
        &mut self,
        var_decl: &HirVariableDeclaration,
    ) -> Result<LexInstruction> {
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

    /// 🤖 Converts HIR ask expressions to LexIR ask instructions
    ///
    /// This method handles the conversion of LLM ask expressions, including:
    /// - Attribute processing and extraction
    /// - Model and temperature parameter handling
    /// - Schema validation setup
    /// - System and user prompt configuration
    ///
    /// ## Attribute Processing
    ///
    /// The method extracts common LLM parameters:
    /// - `model`: Specific LLM model to use
    /// - `temperature`: Response randomness control
    /// - `max_tokens`: Maximum response length
    /// - `schema`: Output format validation
    ///
    /// ## Debug Information
    ///
    /// Includes debug output for troubleshooting attribute extraction and processing.
    fn add_ask_instruction(&mut self, ask: &HirAskExpression, result: ValueRef) -> Result<()> {
        let mut attributes = HashMap::new();

        // Convert attributes from HIR to HashMap
        for attr in &ask.attributes {
            if let Some(value) = &attr.value {
                attributes.insert(attr.name.clone(), value.clone());
            } else {
                attributes.insert(attr.name.clone(), "true".to_string());
            }
        }

        println!(
            "🔍 DEBUG HIR->LEXIR: attributes before extraction: {:?}",
            attributes
        );

        // Extract LLM parameters from attributes
        let model = attributes.remove("model");
        let temperature = attributes
            .get("temperature")
            .and_then(|t| t.parse::<f64>().ok());

        println!(
            "🔍 DEBUG HIR->LEXIR: extracted model: {:?}, temperature: {:?}",
            model, temperature
        );

        let max_tokens = attributes
            .get("max_tokens")
            .and_then(|t| t.parse::<u32>().ok());

        // Create the Ask instruction with all parameters
        let instruction = LexInstruction::Ask {
            result,
            system_prompt: ask.system_prompt.clone(),
            user_prompt: ask.user_prompt.clone(),
            model,
            temperature,
            max_tokens,
            schema: ask.output_schema_name.clone(),
            attributes,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 🛡️ Converts HIR ask_safe expressions to LexIR anti-hallucination instructions
    ///
    /// This method handles the conversion of ask_safe expressions with advanced anti-hallucination
    /// validation features, including:
    /// - Basic LLM parameter extraction (model, temperature, max_tokens)
    /// - Anti-hallucination validation strategy configuration
    /// - Confidence threshold and retry logic setup
    /// - Cross-reference model validation
    /// - Fact-checking integration
    ///
    /// ## Anti-Hallucination Features
    ///
    /// The method extracts specialized validation parameters:
    /// - `validation_strategy`: Type of validation (basic, ensemble, fact_check, comprehensive)
    /// - `confidence_threshold`: Minimum confidence score required (0.0-1.0)
    /// - `max_attempts`: Maximum retry attempts for low-confidence responses
    /// - `cross_reference_models`: List of models for cross-validation
    /// - `use_fact_checking`: Enable external fact-checking services
    ///
    /// ## Validation Strategies
    ///
    /// - **Basic**: Simple confidence scoring
    /// - **Ensemble**: Multi-model consensus validation
    /// - **Fact Check**: External fact verification
    /// - **Comprehensive**: All validation methods combined
    ///
    /// ## Debug Information
    ///
    /// Includes comprehensive debug output for troubleshooting validation setup.
    fn add_ask_safe_instruction(
        &mut self,
        ask_safe: &crate::hir::HirAskSafeExpression,
        result: ValueRef,
    ) -> Result<()> {
        let mut attributes = HashMap::new();

        // Convert basic attributes from HIR to HashMap
        for attr in &ask_safe.attributes {
            if let Some(value) = &attr.value {
                attributes.insert(attr.name.clone(), value.clone());
            } else {
                attributes.insert(attr.name.clone(), "true".to_string());
            }
        }

        println!(
            "🛡️ DEBUG HIR->LEXIR: ask_safe attributes before extraction: {:?}",
            attributes
        );

        // Extract basic LLM parameters
        let model = attributes.remove("model");
        let temperature = attributes
            .get("temperature")
            .and_then(|t| t.parse::<f64>().ok());
        let max_tokens = attributes
            .get("max_tokens")
            .and_then(|t| t.parse::<u32>().ok());

        // Extract anti-hallucination validation attributes
        let validation_strategy = ask_safe
            .validation_strategy
            .clone()
            .or_else(|| attributes.remove("validation"));
        let confidence_threshold = ask_safe.confidence_threshold.or_else(|| {
            attributes
                .get("confidence_threshold")
                .and_then(|t| t.parse::<f64>().ok())
        });
        let max_attempts = ask_safe.max_attempts.or_else(|| {
            attributes
                .get("max_attempts")
                .and_then(|t| t.parse::<u32>().ok())
        });
        let use_fact_checking = attributes
            .get("use_fact_checking")
            .and_then(|t| t.parse::<bool>().ok());

        // Extract cross-reference models for ensemble validation
        let mut cross_reference_models = ask_safe.cross_reference_models.clone();
        if cross_reference_models.is_empty() {
            if let Some(models_str) = attributes.remove("cross_reference_models") {
                cross_reference_models = models_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }

        println!("🛡️ DEBUG HIR->LEXIR: validation_strategy: {:?}, confidence_threshold: {:?}, max_attempts: {:?}", 
                validation_strategy, confidence_threshold, max_attempts);
        println!(
            "🛡️ DEBUG HIR->LEXIR: cross_reference_models: {:?}",
            cross_reference_models
        );

        // Create the AskSafe instruction with comprehensive anti-hallucination validation
        let instruction = LexInstruction::AskSafe {
            result,
            system_prompt: ask_safe.system_prompt.clone(),
            user_prompt: ask_safe.user_prompt.clone(),
            model,
            temperature,
            max_tokens,
            schema: ask_safe.output_schema_name.clone(),
            attributes,
            validation_strategy,
            confidence_threshold,
            max_attempts,
            cross_reference_models,
            use_fact_checking,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 📊 Data Processing Methods
    ///
    /// The following methods handle the conversion of HIR data processing operations
    /// to LexIR instructions. These support the data pipeline functionality of Lexon.

    /// 📥 Converts HIR data load operations to LexIR instructions
    ///
    /// This method handles the conversion of data loading operations, including:
    /// - Source file or database specification
    /// - Schema validation and type checking
    /// - Loading options and configuration
    /// - Result value reference generation
    ///
    /// ## Supported Data Sources
    ///
    /// - CSV files with automatic schema inference
    /// - JSON files with structured data parsing
    /// - Database connections with query support
    /// - API endpoints with authentication
    ///
    /// ## Options Processing
    ///
    /// Converts HIR literal options to LexIR format for:
    /// - Delimiter specification for CSV files
    /// - Encoding settings for text files
    /// - Authentication credentials for APIs
    /// - Query parameters for databases
    fn add_data_load_instruction(
        &mut self,
        data_load: &HirDataLoad,
        result: ValueRef,
    ) -> Result<()> {
        // Convert options from HIR literals to LexIR format
        let mut options = HashMap::new();
        for (key, lit) in &data_load.options {
            options.insert(key.clone(), self.convert_literal(lit)?);
        }

        // Create the DATA_LOAD instruction with all parameters
        let instruction = LexInstruction::DataLoad {
            result,
            source: data_load.source.clone(),
            schema: data_load.schema_name.clone(),
            options,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 🔍 Converts HIR data filter operations to LexIR instructions
    ///
    /// This method handles the conversion of data filtering operations, including:
    /// - Input dataset reference conversion
    /// - Predicate expression processing
    /// - Lazy evaluation support
    /// - Result value reference generation
    ///
    /// ## Filter Predicates
    ///
    /// Supports complex filtering conditions:
    /// - Comparison operations (==, !=, <, >, <=, >=)
    /// - Logical operations (&&, ||, !)
    /// - String matching and regex patterns
    /// - Null and empty value checks
    ///
    /// ## Performance Optimization
    ///
    /// - Lazy evaluation for large datasets
    /// - Index-based filtering when available
    /// - Predicate pushdown for database sources
    fn add_data_filter_instruction(
        &mut self,
        data_filter: &HirDataFilter,
        result: ValueRef,
    ) -> Result<()> {
        // Convert the input dataset reference
        let input = self.convert_node_to_value_ref(&data_filter.input)?;

        // Convert the filtering condition to a LexIR expression
        let predicate_value_ref = self.convert_node_to_value_ref(&data_filter.condition)?;
        let predicate = LexExpression::Value(predicate_value_ref);

        // Create the DATA_FILTER instruction with predicate
        let instruction = LexInstruction::DataFilter {
            result,
            input,
            predicate,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 📋 Converts HIR data select operations to LexIR instructions
    ///
    /// This method handles the conversion of data selection operations, including:
    /// - Input dataset reference conversion
    /// - Field selection and projection
    /// - Column renaming support
    /// - Result value reference generation
    ///
    /// ## Field Selection
    ///
    /// Supports various selection patterns:
    /// - Explicit field names
    /// - Wildcard patterns (*)
    /// - Computed columns with expressions
    /// - Nested field access for JSON data
    ///
    /// ## Performance Features
    ///
    /// - Column pruning for large datasets
    /// - Projection pushdown for database sources
    /// - Memory-efficient streaming for large files
    fn add_data_select_instruction(
        &mut self,
        data_select: &HirDataSelect,
        result: ValueRef,
    ) -> Result<()> {
        // Convert the input dataset reference
        let input = self.convert_node_to_value_ref(&data_select.input)?;

        // Create the DATA_SELECT instruction with field list
        let instruction = LexInstruction::DataSelect {
            result,
            input,
            fields: data_select.fields.clone(),
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 📏 Converts HIR data take operations to LexIR instructions
    ///
    /// This method handles the conversion of data limiting operations, including:
    /// - Input dataset reference conversion
    /// - Count parameter processing
    /// - Offset support for pagination
    /// - Result value reference generation
    ///
    /// ## Limiting Strategies
    ///
    /// Supports different approaches:
    /// - Head/tail operations for ordered data
    /// - Random sampling for statistical analysis
    /// - Pagination with offset and limit
    /// - Top-K selection with ordering
    ///
    /// ## Memory Management
    ///
    /// - Streaming evaluation for large datasets
    /// - Early termination when limit is reached
    /// - Memory-efficient buffering
    fn add_data_take_instruction(
        &mut self,
        data_take: &HirDataTake,
        result: ValueRef,
    ) -> Result<()> {
        // Convert the input dataset reference
        let input = self.convert_node_to_value_ref(&data_take.input)?;

        // Convert the count parameter
        let count = self.convert_node_to_value_ref(&data_take.count)?;

        // Create the DATA_TAKE instruction with count
        let instruction = LexInstruction::DataTake {
            result,
            input,
            count,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 📤 Converts HIR data export operations to LexIR instructions
    ///
    /// This method handles the conversion of data export operations, including:
    /// - Input dataset reference conversion
    /// - Output path and format specification
    /// - Export options and configuration
    /// - File system operations
    ///
    /// ## Supported Export Formats
    ///
    /// - CSV with configurable delimiters
    /// - JSON with pretty printing options
    /// - Parquet for efficient storage
    /// - Database tables with schema creation
    ///
    /// ## Export Options
    ///
    /// Configurable parameters:
    /// - Encoding settings (UTF-8, ASCII, etc.)
    /// - Compression options (gzip, bzip2, etc.)
    /// - Header inclusion for CSV files
    /// - Batch size for large datasets
    fn add_data_export_instruction(&mut self, data_export: &HirDataExport) -> Result<()> {
        // Convert the input dataset reference
        let input = self.convert_node_to_value_ref(&data_export.input)?;

        // Convert export options from HIR literals to LexIR format
        let mut options = HashMap::new();
        for (key, lit) in &data_export.options {
            options.insert(key.clone(), self.convert_literal(lit)?);
        }

        // Create the DATA_EXPORT instruction with all parameters
        let instruction = LexInstruction::DataExport {
            input,
            path: data_export.path.clone(),
            format: data_export.format.clone(),
            options,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 🧠 Memory Management Methods
    ///
    /// The following methods handle the conversion of HIR memory operations
    /// to LexIR instructions. These support the persistent memory functionality of Lexon.

    /// 📥 Converts HIR memory load operations to LexIR instructions
    ///
    /// This method handles the conversion of memory loading operations, including:
    /// - Scope-based memory access
    /// - Source specification for external memory
    /// - Loading strategy configuration
    /// - Options processing for memory systems
    ///
    /// ## Memory Scopes
    ///
    /// Supports different memory scopes:
    /// - Session: Temporary memory for current session
    /// - Global: Persistent memory across sessions
    /// - User: User-specific memory storage
    /// - System: System-wide memory access
    ///
    /// ## Loading Strategies
    ///
    /// - **Buffer**: Load entire memory into buffer
    /// - **Stream**: Streaming access for large memories
    /// - **Lazy**: Load on demand with caching
    /// - **Snapshot**: Point-in-time memory snapshot
    ///
    /// ## Options Processing
    ///
    /// Configurable parameters:
    /// - Cache size and eviction policies
    /// - Compression settings for storage
    /// - Encryption keys for secure memory
    /// - Timeout settings for remote memory
    fn add_memory_load_instruction(
        &mut self,
        memory_load: &HirMemoryLoad,
        result: ValueRef,
    ) -> Result<()> {
        // Convert the source reference if present
        let source = if let Some(src) = &memory_load.source {
            Some(self.convert_node_to_value_ref(src)?)
        } else {
            None
        };

        // Convert memory loading options from HIR literals to LexIR format
        let mut options = HashMap::new();
        for (key, lit) in &memory_load.options {
            options.insert(key.clone(), self.convert_literal(lit)?);
        }

        // Create the MemoryLoad instruction with all parameters
        let instruction = LexInstruction::MemoryLoad {
            result,
            scope: memory_load.scope.clone(),
            source,
            strategy: memory_load.strategy.clone(),
            options,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 💾 Converts HIR memory store operations to LexIR instructions
    ///
    /// This method handles the conversion of memory storage operations, including:
    /// - Scope-based memory access
    /// - Value serialization and storage
    /// - Key-based organization
    /// - Storage options configuration
    ///
    /// ## Memory Organization
    ///
    /// Supports different organization patterns:
    /// - Key-value pairs for structured access
    /// - Hierarchical namespaces for organization
    /// - Tagged memory for categorization
    /// - Timestamped entries for versioning
    ///
    /// ## Storage Features
    ///
    /// - Automatic serialization of complex types
    /// - Compression for large values
    /// - Encryption for sensitive data
    /// - Atomic operations for consistency
    ///
    /// ## Debug Information
    ///
    /// Includes debug output for troubleshooting memory operations.
    fn add_memory_store_instruction(&mut self, memory_store: &HirMemoryStore) -> Result<()> {
        eprintln!(
            "DEBUG: add_memory_store_instruction called with scope: {}",
            memory_store.scope
        );

        // Convert the value to be stored
        let value = self.convert_node_to_value_ref(&memory_store.value)?;

        // Convert storage options from HIR literals to LexIR format
        let options = HashMap::new();

        // Create the MemoryStore instruction with all parameters
        let instruction = LexInstruction::MemoryStore {
            scope: memory_store.scope.clone(),
            value,
            key: memory_store.key.clone(),
            options,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    fn convert_schema_definition(&mut self, schema_def: &HirSchemaDefinition) -> Result<()> {
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

    #[allow(clippy::only_used_in_recursion)]
    fn parse_lex_type(&mut self, type_str: &str) -> LexType {
        match type_str {
            "int" => LexType::Int,
            "float" => LexType::Float,
            "string" => LexType::String,
            "bool" => LexType::Bool,
            "void" => LexType::Void,
            _ => {
                let (base, args) = Self::split_type(type_str);
                match base.as_str() {
                    "List" if args.len() == 1 => {
                        let inner = self.parse_lex_type(&args[0]);
                        LexType::List(Box::new(inner))
                    }
                    "Option" if args.len() == 1 => {
                        let inner = self.parse_lex_type(&args[0]);
                        LexType::Option(Box::new(inner))
                    }
                    "Map" if args.len() == 2 => {
                        let k = self.parse_lex_type(&args[0]);
                        let v = self.parse_lex_type(&args[1]);
                        LexType::Map(Box::new(k), Box::new(v))
                    }
                    _ => LexType::Schema(type_str.to_string()),
                }
            }
        }
    }

    fn split_type(type_str: &str) -> (String, Vec<String>) {
        // Split a type string like "Map<string,int>" into base "Map" and args ["string", "int"]
        let trimmed = type_str.trim();
        if let Some(start) = trimmed.find('<') {
            if trimmed.ends_with('>') {
                let base = trimmed[..start].to_string();
                let inside = &trimmed[start + 1..trimmed.len() - 1];
                let args = inside
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>();
                return (base, args);
            }
        }
        (trimmed.to_string(), Vec::new())
    }

    /// Instantiates a generic schema (if it exists) with the supplied type arguments and returns the specialized name.
    /// For example, `Result` with args `["int"]` will generate `Result__int`.
    fn instantiate_schema(&mut self, base: &str, args: &[String]) -> Option<String> {
        // We check if the generic definition exists
        let generic = self.generic_schemas.get(base)?.clone();

        // Arity must match
        if generic.type_parameters.len() != args.len() {
            return None;
        }

        let specialized_name = format!("{}__{}", base, args.join("_"));

        // If already generated before, do not repeat
        if self.program.schemas.contains_key(&specialized_name) {
            return Some(specialized_name);
        }

        // Build specialized fields
        let mut spec_fields = Vec::<LexSchemaField>::new();
        for field in &generic.fields {
            // Simple substitution of generic parameters in the field type
            let mut field_type_name = field.type_name.clone();
            for (param, arg) in generic.type_parameters.iter().zip(args.iter()) {
                if field_type_name == *param {
                    field_type_name = arg.clone();
                }
            }

            let lex_type = self.parse_lex_type(&field_type_name);
            let default_value = if let Some(boxed) = &field.default_value {
                if let HirNode::Literal(lit) = &**boxed {
                    self.convert_literal(lit).ok()
                } else {
                    None
                }
            } else {
                None
            };

            spec_fields.push(LexSchemaField {
                name: field.name.clone(),
                field_type: lex_type,
                is_optional: field.is_optional,
                default_value,
            });
        }

        // Register the new specialized schema
        self.program.add_schema(LexSchemaDefinition {
            name: specialized_name.clone(),
            fields: spec_fields,
        });

        Some(specialized_name)
    }

    /// Converts a For-In loop to basic LexIR instructions.
    /// Simplified implementation: declares the iterator variable and assigns the iterable.
    fn convert_for_in(&mut self, for_in: &crate::hir::HirForIn) -> Result<()> {
        // Convert the iterable to a value reference
        let iterable_ref = self.convert_node_to_value_ref(&for_in.iterable)?;

        // Convert the body
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
                            _ => self.convert_node_to_value_ref(arg)?,
                        };
                        args_exprs.push(LexExpression::Value(val_ref));
                    }
                    let fn_name_expanded = self.expand_any_alias(&func_call.function);
                    let fn_name_norm = fn_name_expanded.replace("::", "__").replace('.', "__");
                    let call_instr = LexInstruction::Call {
                        result: Some(ValueRef::Temp(temp_id)),
                        function: fn_name_norm,
                        args: args_exprs,
                    };
                    body_instrs.push(call_instr);
                }
                crate::hir::HirNode::Assignment(assignment) => {
                    // Handle assignments inside for loop
                    let right_value = self.convert_node_to_value_ref(&assignment.right)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(assignment.left.clone()),
                        expr: LexExpression::Value(right_value),
                    };
                    body_instrs.push(assign_instr);
                }
                crate::hir::HirNode::VariableDeclaration(var_decl) => {
                    // Handle variable declarations inside for loop
                    let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(var_decl.name.clone()),
                        expr: LexExpression::Value(init_ref),
                    };
                    body_instrs.push(assign_instr);
                }
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

    fn convert_function_definition(&mut self, func_def: &HirFunctionDefinition) -> Result<()> {
        // Convert the return type
        let return_type = if let Some(type_name) = &func_def.return_type {
            self.parse_lex_type(type_name.as_str())
        } else {
            LexType::Void
        };

        // Convert the function body
        let mut body = Vec::new();

        // Helper: convert an expression node into a LexExpression while ensuring
        // any required instructions (e.g., function/method calls) are appended to this local body
        fn node_to_expr_local(
            ctx: &mut ConversionContext,
            node: &HirNode,
            body: &mut Vec<LexInstruction>,
        ) -> Result<LexExpression> {
            match node {
                HirNode::Literal(lit) => {
                    let lex_lit = ctx.convert_literal(lit)?;
                    Ok(LexExpression::Value(ValueRef::Literal(lex_lit)))
                }
                HirNode::Identifier(name) => {
                    Ok(LexExpression::Value(ValueRef::Named(name.clone())))
                }
                HirNode::Binary(bin) => {
                    use crate::lexir::LexBinaryOperator as LB;
                    let op = match bin.operator.as_str() {
                        "+" => LB::Add,
                        "-" => LB::Subtract,
                        "*" => LB::Multiply,
                        "/" => LB::Divide,
                        ">" => LB::GreaterThan,
                        "<" => LB::LessThan,
                        ">=" => LB::GreaterEqual,
                        "<=" => LB::LessEqual,
                        "==" => LB::Equal,
                        "!=" => LB::NotEqual,
                        "&&" => LB::And,
                        "||" => LB::Or,
                        other => {
                            return Err(HirToLexIrError::UnsupportedNode(format!(
                                "Unsupported binary operator in function: {}",
                                other
                            )))
                        }
                    };
                    let left_expr = node_to_expr_local(ctx, &bin.left, body)?;
                    let right_expr = node_to_expr_local(ctx, &bin.right, body)?;
                    Ok(LexExpression::BinaryOp {
                        operator: op,
                        left: Box::new(left_expr),
                        right: Box::new(right_expr),
                    })
                }
                HirNode::FunctionCall(func_call) => {
                    // Build args
                    let fn_name_expanded = ctx.expand_any_alias(&func_call.function);
                    let fn_name_norm = fn_name_expanded.replace("::", "__").replace('.', "__");
                    let specialized_name = if let Some(generic_func) =
                        ctx.generic_functions.get(&func_call.function)
                    {
                        let spec_name = format!("{}_{}", fn_name_norm, ctx.temp_gen.next().0);
                        let mono = generic_func.clone();
                        if !ctx.generic_functions.contains_key(&spec_name) {
                            ctx.generic_functions
                                .insert(spec_name.clone(), mono.clone());
                            ctx.convert_function_definition(&mono)?;
                        }
                        spec_name
                    } else {
                        fn_name_norm.clone()
                    };

                    let mut args_exprs = Vec::new();
                    for arg in &func_call.args {
                        // Convert nested values to expressions (no global temps)
                        let e = node_to_expr_local(ctx, arg, body)?;
                        args_exprs.push(e);
                    }
                    let temp_id = ctx.temp_gen.next();
                    body.push(LexInstruction::Call {
                        result: Some(ValueRef::Temp(temp_id.clone())),
                        function: specialized_name,
                        args: args_exprs,
                    });
                    Ok(LexExpression::Value(ValueRef::Temp(temp_id)))
                }
                HirNode::MethodCall(method_call) => {
                    let is_type = match &*method_call.target {
                        HirNode::Identifier(name) => {
                            let first = name.chars().next();
                            let stdmods = [
                                "struct", "enum", "encoding", "strings", "math", "regex", "time",
                                "number", "crypto", "json",
                            ];
                            stdmods.contains(&name.as_str())
                                || first.map(|c| c.is_ascii_uppercase()).unwrap_or(false)
                        }
                        _ => false,
                    };
                    let temp_id = ctx.temp_gen.next();
                    if is_type {
                        let target_name = if let HirNode::Identifier(name) = &*method_call.target {
                            name.clone()
                        } else {
                            String::new()
                        };
                        let target_expanded = ctx.expand_any_alias(&target_name);
                        let target_norm = target_expanded.replace("::", "__").replace('.', "__");
                        let mut args_exprs = Vec::new();
                        for arg in &method_call.args {
                            let e = node_to_expr_local(ctx, arg, body)?;
                            args_exprs.push(e);
                        }
                        body.push(LexInstruction::Call {
                            result: Some(ValueRef::Temp(temp_id.clone())),
                            function: format!("{}__{}", target_norm, method_call.method),
                            args: args_exprs,
                        });
                    } else {
                        // Instance call
                        let mut args_exprs = Vec::new();
                        let recv_expr = node_to_expr_local(ctx, &method_call.target, body)?;
                        args_exprs.push(recv_expr);
                        args_exprs.push(LexExpression::Value(ValueRef::Literal(
                            LexLiteral::String(method_call.method.clone()),
                        )));
                        for arg in &method_call.args {
                            let e = node_to_expr_local(ctx, arg, body)?;
                            args_exprs.push(e);
                        }
                        body.push(LexInstruction::Call {
                            result: Some(ValueRef::Temp(temp_id.clone())),
                            function: "method.call".to_string(),
                            args: args_exprs,
                        });
                    }
                    Ok(LexExpression::Value(ValueRef::Temp(temp_id)))
                }
                _ => {
                    // Fallback: use existing converter but beware it may emit global instructions.
                    // This handles rare nodes; for our OO smoke return path we cover the common ones above.
                    let v = ctx.convert_node_to_value_ref(node)?;
                    Ok(LexExpression::Value(v))
                }
            }
        }

        // Helper: push a statement into a local block body
        fn push_stmt_local(
            ctx: &mut ConversionContext,
            stmt: &HirNode,
            block: &mut Vec<LexInstruction>,
        ) -> Result<()> {
            match stmt {
                HirNode::VariableDeclaration(var_decl) => {
                    let init_expr = node_to_expr_local(ctx, &var_decl.value, block)?;
                    block.push(LexInstruction::Assign {
                        result: ValueRef::Named(var_decl.name.clone()),
                        expr: init_expr,
                    });
                    Ok(())
                }
                HirNode::Assignment(assignment) => {
                    let right_expr = node_to_expr_local(ctx, &assignment.right, block)?;
                    block.push(LexInstruction::Assign {
                        result: ValueRef::Named(assignment.left.clone()),
                        expr: right_expr,
                    });
                    Ok(())
                }
                HirNode::FunctionCall(_) | HirNode::MethodCall(_) | HirNode::Binary(_) => {
                    // Ensure any necessary call instructions are appended
                    let _ = node_to_expr_local(ctx, stmt, block)?;
                    Ok(())
                }
                HirNode::If(inner_if) => {
                    let cond_expr = node_to_expr_local(ctx, &inner_if.condition, block)?;
                    let mut then_block: Vec<LexInstruction> = Vec::new();
                    for s in &inner_if.then_body {
                        push_stmt_local(ctx, s, &mut then_block)?;
                    }
                    let else_block = if let Some(else_stmts) = &inner_if.else_body {
                        let mut ev: Vec<LexInstruction> = Vec::new();
                        for s in else_stmts {
                            push_stmt_local(ctx, s, &mut ev)?;
                        }
                        Some(ev)
                    } else {
                        None
                    };
                    block.push(LexInstruction::If {
                        condition: cond_expr,
                        then_block,
                        else_block,
                    });
                    Ok(())
                }
                _ => Ok(()),
            }
        }
        for statement in &func_def.body {
            match statement {
                // Allow expression-bodied functions: literal → return literal
                HirNode::Literal(_) => {
                    let val = self.convert_node_to_value_ref(statement)?;
                    body.push(LexInstruction::Return {
                        expr: Some(LexExpression::Value(val)),
                    });
                }
                HirNode::VariableDeclaration(var_decl) => {
                    // Declaration with initial value
                    let _instruction = self.convert_variable_declaration(var_decl)?;
                    // Skip adding to program - will be added to function body

                    // Evaluate and assign the variable's initial value (ensure calls are appended to local body)
                    let init_expr = match var_decl.value.as_ref() {
                        HirNode::Binary(bin_expr) => {
                            let left_expr = node_to_expr_local(self, &bin_expr.left, &mut body)?;
                            let right_expr = node_to_expr_local(self, &bin_expr.right, &mut body)?;
                            use crate::lexir::LexBinaryOperator as LB;
                            let operator = match bin_expr.operator.as_str() {
                                "+" => LB::Add,
                                "-" => LB::Subtract,
                                "*" => LB::Multiply,
                                "/" => LB::Divide,
                                ">" => LB::GreaterThan,
                                "<" => LB::LessThan,
                                ">=" => LB::GreaterEqual,
                                "<=" => LB::LessEqual,
                                "==" => LB::Equal,
                                "!=" => LB::NotEqual,
                                "&&" => LB::And,
                                "||" => LB::Or,
                                _ => {
                                    return Err(HirToLexIrError::UnsupportedNode(format!(
                                        "Unsupported binary operator: {}",
                                        bin_expr.operator
                                    )))
                                }
                            };
                            LexExpression::BinaryOp {
                                operator,
                                left: Box::new(left_expr),
                                right: Box::new(right_expr),
                            }
                        }
                        _ => node_to_expr_local(self, &var_decl.value, &mut body)?,
                    };
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(var_decl.name.clone()),
                        expr: init_expr,
                    };
                    body.push(assign_instr);
                }
                // Add handling for other node types within the function body
                HirNode::DataLoad(data_load) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_load_instruction(data_load, ValueRef::Temp(temp_id))?;
                }
                HirNode::DataFilter(data_filter) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_filter_instruction(data_filter, ValueRef::Temp(temp_id))?;
                }
                HirNode::DataSelect(data_select) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_select_instruction(data_select, ValueRef::Temp(temp_id))?;
                }
                HirNode::DataTake(data_take) => {
                    let temp_id = self.temp_gen.next();
                    self.add_data_take_instruction(data_take, ValueRef::Temp(temp_id))?;
                }
                HirNode::DataExport(data_export) => {
                    self.add_data_export_instruction(data_export)?;
                }
                HirNode::MemoryLoad(memory_load) => {
                    let temp_id = self.temp_gen.next();
                    self.add_memory_load_instruction(memory_load, ValueRef::Temp(temp_id))?;
                }
                HirNode::MemoryStore(memory_store) => {
                    self.add_memory_store_instruction(memory_store)?;
                }
                HirNode::Assignment(assignment) => {
                    let right_value = self.convert_node_to_value_ref(&assignment.right)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(assignment.left.clone()),
                        expr: LexExpression::Value(right_value),
                    };
                    body.push(assign_instr);
                }
                // Function/method calls as final expression: return their value
                HirNode::FunctionCall(_) | HirNode::MethodCall(_) => {
                    let value_ref = self.convert_node_to_value_ref(statement)?;
                    body.push(LexInstruction::Return {
                        expr: Some(LexExpression::Value(value_ref)),
                    });
                }
                // Binary expressions as final expression: return value
                HirNode::Binary(_) => {
                    let value_ref = self.convert_node_to_value_ref(statement)?;
                    body.push(LexInstruction::Return {
                        expr: Some(LexExpression::Value(value_ref)),
                    });
                }
                HirNode::While(while_node) => {
                    let instr = self.convert_while(while_node)?;
                    body.push(instr);
                }
                HirNode::Break => body.push(LexInstruction::Break),
                HirNode::Continue => body.push(LexInstruction::Continue),
                HirNode::Return(return_node) => {
                    let return_expr = if let Some(expr) = &return_node.expression {
                        Some(node_to_expr_local(self, expr, &mut body)?)
                    } else {
                        None
                    };
                    body.push(LexInstruction::Return { expr: return_expr });
                }
                HirNode::If(if_node) => {
                    let cond_expr = node_to_expr_local(self, &if_node.condition, &mut body)?;
                    let mut then_block: Vec<LexInstruction> = Vec::new();
                    for s in &if_node.then_body {
                        push_stmt_local(self, s, &mut then_block)?;
                    }
                    let else_block = if let Some(else_stmts) = &if_node.else_body {
                        let mut ev: Vec<LexInstruction> = Vec::new();
                        for s in else_stmts {
                            push_stmt_local(self, s, &mut ev)?;
                        }
                        Some(ev)
                    } else {
                        None
                    };
                    body.push(LexInstruction::If {
                        condition: cond_expr,
                        then_block,
                        else_block,
                    });
                }
                // Other statement types would go here
                _ => {
                    return Err(HirToLexIrError::UnsupportedNode(format!(
                        "Unsupported statement in function body: {:?}",
                        statement
                    )))
                }
            }
        }

        // For now, functions have no parameters (will be added later)
        let parameters: Vec<(String, LexType)> = func_def
            .parameters
            .iter()
            .map(|p| (p.name.clone(), self.parse_lex_type(&p.type_name)))
            .collect();

        let base_name = func_def.name.clone();
        let full_name = if self.module_prefix.is_empty() {
            base_name
        } else {
            format!("{}__{}", self.module_prefix, base_name)
        };
        self.program.add_function(LexFunction {
            name: full_name,
            return_type,
            parameters,
            body,
        });

        Ok(())
    }

    /// Converts a HIR node to a LexIR value reference
    fn convert_node(&mut self, node: &HirNode) -> Result<ValueRef> {
        match node {
            HirNode::VariableDeclaration(var_decl) => {
                let instruction = self.convert_variable_declaration(var_decl)?;
                self.program.add_instruction(instruction);

                // Evaluar y asignar el valor inicial de la variable (incluye expresiones ask y literales)
                let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                let assign_instr = LexInstruction::Assign {
                    result: ValueRef::Named(var_decl.name.clone()),
                    expr: LexExpression::Value(init_ref),
                };
                self.program.add_instruction(assign_instr);
                Ok(ValueRef::Named(var_decl.name.clone()))
            }
            _ => Ok(self.convert_node_to_value_ref(node)?),
        }
    }

    fn convert_while(&mut self, while_node: &HirWhile) -> Result<LexInstruction> {
        // Convert condition to expression - FIXED: Handle binary expressions dynamically
        let condition_expr = match while_node.condition.as_ref() {
            HirNode::Binary(bin_expr) => {
                let left_val = self.convert_node_to_value_ref(&bin_expr.left)?;
                let right_val = self.convert_node_to_value_ref(&bin_expr.right)?;

                use crate::lexir::LexBinaryOperator as LB;
                let operator = match bin_expr.operator.as_str() {
                    "+" => LB::Add,
                    "-" => LB::Subtract,
                    "*" => LB::Multiply,
                    "/" => LB::Divide,
                    ">" => LB::GreaterThan,
                    "<" => LB::LessThan,
                    ">=" => LB::GreaterEqual,
                    "<=" => LB::LessEqual,
                    "==" => LB::Equal,
                    "!=" => LB::NotEqual,
                    "&&" => LB::And,
                    "||" => LB::Or,
                    _ => {
                        return Err(HirToLexIrError::UnsupportedNode(format!(
                            "Unsupported binary operator in condition: {}",
                            bin_expr.operator
                        )))
                    }
                };

                LexExpression::BinaryOp {
                    operator,
                    left: Box::new(LexExpression::Value(left_val)),
                    right: Box::new(LexExpression::Value(right_val)),
                }
            }
            _ => {
                let cond_val = self.convert_node_to_value_ref(&while_node.condition)?;
                LexExpression::Value(cond_val)
            }
        };

        // Convert body statements - handle all statement types
        let mut body_instrs = Vec::new();
        for stmt in &while_node.body {
            match stmt {
                HirNode::Break => body_instrs.push(LexInstruction::Break),
                HirNode::Continue => body_instrs.push(LexInstruction::Continue),
                HirNode::FunctionCall(_) => {
                    // Handle function calls like print()
                    let temp_id = self.temp_gen.next();
                    let value_ref = self.convert_node_to_value_ref(stmt)?;
                    let instr = LexInstruction::Assign {
                        result: ValueRef::Temp(temp_id),
                        expr: LexExpression::Value(value_ref),
                    };
                    body_instrs.push(instr);
                }
                HirNode::Assignment(assignment) => {
                    // Handle assignments like counter = counter + 1
                    // FIXED: Handle binary expressions directly to avoid pre-calculating temporals
                    let right_expr = match assignment.right.as_ref() {
                        HirNode::Binary(bin_expr) => {
                            let left_val = self.convert_node_to_value_ref(&bin_expr.left)?;
                            let right_val = self.convert_node_to_value_ref(&bin_expr.right)?;

                            use crate::lexir::LexBinaryOperator as LB;
                            let operator = match bin_expr.operator.as_str() {
                                "+" => LB::Add,
                                "-" => LB::Subtract,
                                "*" => LB::Multiply,
                                "/" => LB::Divide,
                                ">" => LB::GreaterThan,
                                "<" => LB::LessThan,
                                ">=" => LB::GreaterEqual,
                                "<=" => LB::LessEqual,
                                "==" => LB::Equal,
                                "!=" => LB::NotEqual,
                                "&&" => LB::And,
                                "||" => LB::Or,
                                _ => {
                                    return Err(HirToLexIrError::UnsupportedNode(format!(
                                        "Unsupported binary operator: {}",
                                        bin_expr.operator
                                    )))
                                }
                            };

                            LexExpression::BinaryOp {
                                operator,
                                left: Box::new(LexExpression::Value(left_val)),
                                right: Box::new(LexExpression::Value(right_val)),
                            }
                        }
                        _ => {
                            let right_value = self.convert_node_to_value_ref(&assignment.right)?;
                            LexExpression::Value(right_value)
                        }
                    };

                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(assignment.left.clone()),
                        expr: right_expr,
                    };
                    body_instrs.push(assign_instr);
                }
                HirNode::VariableDeclaration(var_decl) => {
                    // Handle variable declarations
                    let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                    let assign_instr = LexInstruction::Assign {
                        result: ValueRef::Named(var_decl.name.clone()),
                        expr: LexExpression::Value(init_ref),
                    };
                    body_instrs.push(assign_instr);
                }
                _ => {
                    eprintln!(
                        "Warning: Unhandled statement type in while loop body: {:?}",
                        stmt
                    );
                }
            }
        }
        Ok(LexInstruction::While {
            condition: condition_expr,
            body: body_instrs,
        })
    }
    fn convert_match(&mut self, match_node: &HirMatch) -> Result<LexInstruction> {
        // Convert the value to be evaluated
        let value_ref = self.convert_node_to_value_ref(&match_node.value)?;
        let value_expr = LexExpression::Value(value_ref);

        // Convert the arms
        let mut lex_arms = Vec::new();
        for arm in &match_node.arms {
            // Convert the arm body
            let mut body_instrs = Vec::new();
            for stmt in &arm.body {
                match stmt {
                    HirNode::Assignment(assignment) => {
                        let right_value = self.convert_node_to_value_ref(&assignment.right)?;
                        let assign_instr = LexInstruction::Assign {
                            result: ValueRef::Named(assignment.left.clone()),
                            expr: LexExpression::Value(right_value),
                        };
                        body_instrs.push(assign_instr);
                    }
                    HirNode::VariableDeclaration(var_decl) => {
                        let init_ref = self.convert_node_to_value_ref(&var_decl.value)?;
                        let assign_instr = LexInstruction::Assign {
                            result: ValueRef::Named(var_decl.name.clone()),
                            expr: LexExpression::Value(init_ref),
                        };
                        body_instrs.push(assign_instr);
                    }
                    HirNode::FunctionCall(func_call) => {
                        // Convert function call
                        let mut args = Vec::new();
                        for arg in &func_call.args {
                            let arg_ref = self.convert_node_to_value_ref(arg)?;
                            args.push(LexExpression::Value(arg_ref));
                        }

                        let fn_name_expanded = self.expand_any_alias(&func_call.function);
                        let fn_name_norm = fn_name_expanded.replace("::", "__").replace('.', "__");
                        let call_instr = LexInstruction::Call {
                            result: None, // Do not assign a result in the match arm body
                            function: fn_name_norm,
                            args,
                        };
                        body_instrs.push(call_instr);
                    }
                    _ => {
                        eprintln!(
                            "Warning: Unhandled statement type in match arm body: {:?}",
                            stmt
                        );
                    }
                }
            }

            let lex_arm = crate::lexir::LexMatchArm {
                pattern: match &*arm.pattern {
                    HirNode::Literal(HirLiteral::String(s)) => s.clone(),
                    HirNode::Identifier(id) => id.clone(),
                    _ => "_".to_string(), // Default wildcard for complex patterns
                },
                body: body_instrs,
            };
            lex_arms.push(lex_arm);
        }

        Ok(LexInstruction::Match {
            value: value_expr,
            arms: lex_arms,
        })
    }
}

/// Converts a set of HIR nodes into a LexIR program
pub fn convert_hir_to_lexir(hir_nodes: &[HirNode]) -> Result<LexProgram> {
    let mut context = ConversionContext::new();
    use std::collections::HashMap as StdHashMap;
    #[derive(Clone)]
    struct TraitMethodSig {
        name: String,
        param_types: Vec<String>,
        return_type: Option<String>,
    }
    let mut trait_sigs: StdHashMap<String, Vec<TraitMethodSig>> = StdHashMap::new();
    let enforce = std::env::var("LEXON_TRAIT_ENFORCE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(true);

    // Collect trait signatures first
    for node in hir_nodes {
        if let HirNode::TraitDefinition(tr) = node {
            let mut sigs = Vec::new();
            for m in &tr.methods {
                let pts: Vec<String> = m.parameters.iter().map(|p| p.type_name.clone()).collect();
                sigs.push(TraitMethodSig {
                    name: m.name.clone(),
                    param_types: pts,
                    return_type: m.return_type.clone(),
                });
            }
            trait_sigs.insert(tr.name.clone(), sigs);
        }
    }

    // Pre-pass: collect module and item aliases from imports
    for node in hir_nodes {
        if let HirNode::ImportDeclaration(import) = node {
            if let Some(alias) = &import.alias {
                context
                    .module_aliases
                    .insert(alias.clone(), import.path.join("::"));
            }
            if !import.items.is_empty() {
                let base = import.path.join("::");
                for (name, alias) in &import.items {
                    let key = alias.clone().unwrap_or_else(|| name.clone());
                    context
                        .item_aliases
                        .insert(key, format!("{}::{}", base, name));
                }
            }
        }
    }
    // First process schema and function definitions
    for node in hir_nodes {
        match node {
            HirNode::ModuleDeclaration(md) => {
                context.module_prefix = md.path.join("__");
            }
            HirNode::SchemaDefinition(schema_def) => {
                if schema_def.type_parameters.is_empty() {
                    context.convert_schema_definition(schema_def)?;
                } else {
                    context
                        .generic_schemas
                        .insert(schema_def.name.clone(), (**schema_def).clone());
                }
            }
            HirNode::FunctionDefinition(func_def) => {
                if func_def.type_parameters.is_empty() {
                    context.convert_function_definition(func_def)?;
                } else {
                    context
                        .generic_functions
                        .insert(func_def.name.clone(), (**func_def).clone());
                }
            }
            HirNode::ImplBlock(impl_block) => {
                // Validate against trait of same name if present
                if let Some(required) = trait_sigs.get(&impl_block.target) {
                    for req in required {
                        match impl_block.methods.iter().find(|m| m.name == req.name) {
                            Some(m) => {
                                if m.parameters.len() != req.param_types.len() {
                                    let msg = format!(
                                        "impl {}.{} has {} params but trait requires {}",
                                        impl_block.target,
                                        req.name,
                                        m.parameters.len(),
                                        req.param_types.len()
                                    );
                                    if enforce {
                                        return Err(HirToLexIrError::InvalidExpression(msg));
                                    } else {
                                        println!("[TRAIT WARN] {}", msg);
                                    }
                                }
                                // Validate parameter types when both sides specify type names
                                for (i, p) in m.parameters.iter().enumerate() {
                                    if let Some(req_ty) = req.param_types.get(i) {
                                        let got_ty = &p.type_name;
                                        if got_ty != req_ty {
                                            let msg = format!(
                                                "impl {}.{} param {} type '{}' != trait '{}'",
                                                impl_block.target, req.name, i, got_ty, req_ty
                                            );
                                            if enforce {
                                                return Err(HirToLexIrError::InvalidExpression(
                                                    msg,
                                                ));
                                            } else {
                                                println!("[TRAIT WARN] {}", msg);
                                            }
                                        }
                                    }
                                }
                                // Validate return type when both sides specify
                                if let (Some(req_ret), Some(got_ret)) =
                                    (req.return_type.clone(), m.return_type.clone())
                                {
                                    if req_ret != got_ret {
                                        let msg = format!(
                                            "impl {}.{} return '{}' != trait '{}'",
                                            impl_block.target, req.name, got_ret, req_ret
                                        );
                                        if enforce {
                                            return Err(HirToLexIrError::InvalidExpression(msg));
                                        } else {
                                            println!("[TRAIT WARN] {}", msg);
                                        }
                                    }
                                }
                            }
                            None => {
                                let msg = format!(
                                    "impl {} missing method required by trait: {}",
                                    impl_block.target, req.name
                                );
                                if enforce {
                                    return Err(HirToLexIrError::InvalidExpression(msg));
                                } else {
                                    println!("[TRAIT WARN] {}", msg);
                                }
                            }
                        }
                    }
                }
                // Simple monomorphization: convert methods to free functions <Target>__<method>
                for method in &impl_block.methods {
                    let mut mono_method = method.clone();
                    // Support module-like prefix in target
                    let target_expanded = context.expand_module_aliases(&impl_block.target);
                    let target_norm = target_expanded.replace("::", "__").replace('.', "__");
                    mono_method.name = format!("{}__{}", target_norm, method.name);
                    context.convert_function_definition(&mono_method)?;
                }
            }
            HirNode::TraitDefinition(_) => {
                // Traits do not generate LexIR directly.
            }
            _ => {} // Ignore other nodes in this initial phase
        }
    }

    // Then process variable declarations and other top-level instructions
    for node in hir_nodes {
        match node {
            HirNode::ModuleDeclaration(md) => {
                context.module_prefix = md.path.join("__");
            }
            HirNode::VariableDeclaration(_var_decl) => {
                // Declaration with initial value
                context.convert_node(node)?;
            }
            HirNode::SchemaDefinition(_) | HirNode::FunctionDefinition(_) => {
                // Already processed
            }
            // Do not generate LexIR instructions directly in this phase
            HirNode::ImportDeclaration(_) => {}
            // Process the new nodes at root level
            HirNode::DataLoad(data_load) => {
                let temp_id = context.temp_gen.next();
                context.add_data_load_instruction(data_load, ValueRef::Temp(temp_id))?;
            }
            HirNode::DataFilter(data_filter) => {
                let temp_id = context.temp_gen.next();
                context.add_data_filter_instruction(data_filter, ValueRef::Temp(temp_id))?;
            }
            HirNode::DataSelect(data_select) => {
                let temp_id = context.temp_gen.next();
                context.add_data_select_instruction(data_select, ValueRef::Temp(temp_id))?;
            }
            HirNode::DataTake(data_take) => {
                let temp_id = context.temp_gen.next();
                context.add_data_take_instruction(data_take, ValueRef::Temp(temp_id))?;
            }
            HirNode::DataExport(data_export) => {
                context.add_data_export_instruction(data_export)?;
            }
            HirNode::MemoryLoad(memory_load) => {
                let temp_id = context.temp_gen.next();
                context.add_memory_load_instruction(memory_load, ValueRef::Temp(temp_id))?;
            }
            HirNode::MemoryStore(memory_store) => {
                context.add_memory_store_instruction(memory_store)?;
            }
            HirNode::ForIn(for_in) => {
                context.convert_for_in(for_in)?;
            }
            // removed duplicate unreachable ModuleDeclaration arm to fix warning
            HirNode::TraitDefinition(_) | HirNode::ImplBlock(_) => {
                // already handled or ignored
            }
            HirNode::FunctionCall(_func_call) => {
                // Handle function calls as top-level statements (e.g., ask(...))
                // FIXED: Don't generate double temp variables - just use the one from convert_node_to_value_ref
                let _value_ref = context.convert_node_to_value_ref(node)?;
                // The function call instruction is already added by convert_node_to_value_ref
                // No need to add another instruction here
            }
            HirNode::If(if_node) => {
                let _temp_id = context.temp_gen.next();
                let cond_val = context.convert_node_to_value_ref(&if_node.condition)?;
                let condition_expr = LexExpression::Value(cond_val);
                let if_instr = LexInstruction::If {
                    condition: condition_expr,
                    then_block: Vec::new(),
                    else_block: None,
                };
                context.program.add_instruction(if_instr);
            }
            HirNode::Ask(ask_expr) => {
                // Handle ask expressions as top-level statements
                let temp_id = context.temp_gen.next();
                context.add_ask_instruction(ask_expr, ValueRef::Temp(temp_id))?;
            }
            // 🛡️ Handling ask_safe expressions with anti-hallucination validation
            HirNode::AskSafe(ask_safe_expr) => {
                // Handle ask_safe expressions as top-level statements
                let temp_id = context.temp_gen.next();
                context.add_ask_safe_instruction(ask_safe_expr, ValueRef::Temp(temp_id))?;
            }
            HirNode::Binary(bin) => {
                // Recursively convert operands
                let left_val = context.convert_node_to_value_ref(&bin.left)?;
                let right_val = context.convert_node_to_value_ref(&bin.right)?;
                use crate::lexir::{LexBinaryOperator as LB, LexExpression};
                // Fixed: using string-based operators instead of enum
                let op = match bin.operator.as_str() {
                    "+" => LB::Add,
                    "-" => LB::Subtract,
                    "*" => LB::Multiply,
                    "/" => LB::Divide,
                    ">" => LB::GreaterThan,
                    "<" => LB::LessThan,
                    ">=" => LB::GreaterEqual,
                    "<=" => LB::LessEqual,
                    "==" => LB::Equal,
                    "!=" => LB::NotEqual,
                    "&&" => LB::And,
                    "||" => LB::Or,
                    _ => {
                        return Err(HirToLexIrError::UnsupportedNode(format!(
                            "Unsupported binary operator: {}",
                            bin.operator
                        )))
                    }
                };
                // Create expression and assign to temporary
                let expr = LexExpression::BinaryOp {
                    operator: op,
                    left: Box::new(LexExpression::Value(left_val.clone())),
                    right: Box::new(LexExpression::Value(right_val.clone())),
                };
                let temp = context.temp_gen.next();
                context.program.add_instruction(LexInstruction::Assign {
                    result: ValueRef::Temp(temp.clone()),
                    expr,
                });
            }
            HirNode::While(while_node) => {
                let while_instr = context.convert_while(while_node)?;
                context.program.add_instruction(while_instr);
            }
            HirNode::Match(match_node) => {
                let match_instr = context.convert_match(match_node)?;
                context.program.add_instruction(match_instr);
            }

            HirNode::Assignment(assignment) => {
                let right_value = context.convert_node_to_value_ref(&assignment.right)?;
                let assign_instr = LexInstruction::Assign {
                    result: ValueRef::Named(assignment.left.clone()),
                    expr: LexExpression::Value(right_value),
                };
                context.program.add_instruction(assign_instr);
            }
            HirNode::Await(await_expr) => {
                // Handle await expressions as top-level statements
                let temp_id = context.temp_gen.next();
                let inner_value = context.convert_node_to_value_ref(&await_expr.expression)?;
                let assign_instr = LexInstruction::Assign {
                    result: ValueRef::Temp(temp_id),
                    expr: LexExpression::Value(inner_value),
                };
                context.program.add_instruction(assign_instr);
            }
            // Allow harmless top-level expressions by evaluating into a temp
            // Accept harmless top-level literals and identifiers
            HirNode::Literal(_) | HirNode::Identifier(_) => {
                let temp_id = context.temp_gen.next();
                let val = context.convert_node_to_value_ref(&node)?;
                let assign_instr = LexInstruction::Assign {
                    result: ValueRef::Temp(temp_id),
                    expr: LexExpression::Value(val),
                };
                context.program.add_instruction(assign_instr);
            }
            _ => {
                return Err(HirToLexIrError::UnsupportedNode(format!(
                    "Unsupported top-level node: {:?}",
                    node
                )))
            }
        }
    }

    // Post-pass: rewrite function calls using item aliases into fully qualified flattened names
    if !context.item_aliases.is_empty() {
        // Rewrite top-level instructions
        for instr in &mut context.program.instructions {
            if let LexInstruction::Call { function, .. } = instr {
                if !function.contains("__") {
                    if let Some(full) = context.item_aliases.get(function) {
                        *function = full.replace("::", "__").replace('.', "__");
                    }
                }
            }
        }
        // Rewrite inside function bodies
        for (_name, func) in &mut context.program.functions {
            for instr in &mut func.body {
                if let LexInstruction::Call { function, .. } = instr {
                    if !function.contains("__") {
                        if let Some(full) = context.item_aliases.get(function) {
                            *function = full.replace("::", "__").replace('.', "__");
                        }
                    }
                }
            }
        }
    }

    Ok(context.program)
}
