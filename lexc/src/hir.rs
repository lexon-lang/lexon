//! # High-level Intermediate Representation (HIR)
//!
//! This module defines the High-level Intermediate Representation for LEXON programs,
//! serving as the structured representation between parsing and code generation.
//!
//! ## Overview
//!
//! HIR is a high-level, semantic representation that captures the program structure
//! after parsing but before lowering to executable code. It provides:
//!
//! - **Semantic Structure**: Preserves high-level language constructs
//! - **Type Information**: Maintains type annotations and references
//! - **LLM Integration**: First-class support for ask expressions
//! - **Data Operations**: Built-in data processing constructs
//! - **Memory Management**: Contextual memory operations
//! - **Async Support**: Native async/await constructs
//!
//! ## Architecture
//!
//! The HIR is organized into several key components:
//!
//! - [`HirNode`]: The main AST node type representing all language constructs
//! - **Declarations**: Function, schema, module, and import declarations
//! - **Expressions**: Ask expressions, literals, binary operations, and calls
//! - **Statements**: Control flow, assignments, and variable declarations
//! - **Data Operations**: Data loading, filtering, and processing
//! - **Memory Operations**: Memory load/store operations
//!
//! ## Example
//!
//! ```rust
//! use lexc::hir::*;
//!
//! let ask_expr = HirAskExpression {
//!     system_prompt: Some("You are a helpful assistant".to_string()),
//!     user_prompt: Some("What is the capital of France?".to_string()),
//!     output_schema_name: None,
//!     attributes: vec![],
//! };
//!
//! let node = HirNode::Ask(Box::new(ask_expr));
//! ```

use crate::ask_processor::{AskAttribute, AskExpressionData, AskSafeExpressionData};

// ============================================================================
// LLM EXPRESSIONS
// ============================================================================

/// High-level Intermediate Representation for an 'ask' expression.
///
/// Ask expressions represent calls to Large Language Models (LLMs) with
/// optional system prompts, user prompts, and output schema validation.
/// This is the fundamental building block for LLM integration in LEXON.
///
/// # Fields
/// - `system_prompt`: Optional system-level instructions for the LLM
/// - `user_prompt`: Optional user query or input
/// - `output_schema_name`: Optional schema name for response validation
/// - `attributes`: Additional LLM-specific attributes (model, temperature, etc.)
///
/// # Example
/// ```rust
/// use lexc::hir::HirAskExpression;
///
/// let ask = HirAskExpression {
///     system_prompt: Some("You are a helpful assistant".to_string()),
///     user_prompt: Some("What is the capital of France?".to_string()),
///     output_schema_name: None,
///     attributes: vec![],
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirAskExpression {
    /// Optional system-level instructions for the LLM
    pub system_prompt: Option<String>,
    /// Optional user query or input
    pub user_prompt: Option<String>,
    /// Optional schema name for response validation
    pub output_schema_name: Option<String>,
    /// Additional LLM-specific attributes
    pub attributes: Vec<AskAttribute>,
}

/// High-level Intermediate Representation for an 'ask_safe' expression.
///
/// Ask_safe expressions extend regular ask expressions with anti-hallucination
/// validation capabilities. They provide built-in fact-checking, confidence
/// scoring, and cross-validation with multiple models.
///
/// # Anti-hallucination Features
/// - **Validation Strategies**: Basic, ensemble, fact-check, comprehensive
/// - **Confidence Scoring**: Automatic confidence threshold checking
/// - **Cross-validation**: Multiple model verification
/// - **Retry Logic**: Automatic retry on low confidence
///
/// # Example
/// ```rust
/// use lexc::hir::HirAskSafeExpression;
///
/// let ask_safe = HirAskSafeExpression {
///     system_prompt: Some("You are a factual assistant".to_string()),
///     user_prompt: Some("What is the population of Paris?".to_string()),
///     output_schema_name: None,
///     attributes: vec![],
///     validation_strategy: Some("comprehensive".to_string()),
///     confidence_threshold: Some(0.8),
///     max_attempts: Some(3),
///     cross_reference_models: vec!["gpt-4".to_string(), "claude-3".to_string()],
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirAskSafeExpression {
    /// Optional system-level instructions for the LLM
    pub system_prompt: Option<String>,
    /// Optional user query or input
    pub user_prompt: Option<String>,
    /// Optional schema name for response validation
    pub output_schema_name: Option<String>,
    /// Additional LLM-specific attributes
    pub attributes: Vec<AskAttribute>,
    /// Validation strategy: "basic", "ensemble", "fact_check", "comprehensive"
    pub validation_strategy: Option<String>,
    /// Confidence threshold (0.0-1.0)
    pub confidence_threshold: Option<f64>,
    /// Maximum number of validation attempts
    pub max_attempts: Option<u32>,
    /// Models for cross-validation
    pub cross_reference_models: Vec<String>,
}

impl From<AskExpressionData> for HirAskExpression {
    fn from(ask_data: AskExpressionData) -> Self {
        HirAskExpression {
            system_prompt: ask_data.system_prompt,
            user_prompt: ask_data.user_prompt,
            output_schema_name: ask_data.schema_name,
            attributes: ask_data.attributes,
        }
    }
}

impl From<AskSafeExpressionData> for HirAskSafeExpression {
    fn from(ask_safe_data: AskSafeExpressionData) -> Self {
        HirAskSafeExpression {
            system_prompt: ask_safe_data.system_prompt,
            user_prompt: ask_safe_data.user_prompt,
            output_schema_name: ask_safe_data.schema_name,
            attributes: ask_safe_data.attributes,
            validation_strategy: ask_safe_data.validation_strategy,
            confidence_threshold: ask_safe_data.confidence_threshold,
            max_attempts: ask_safe_data.max_attempts,
            cross_reference_models: ask_safe_data.cross_reference_models,
        }
    }
}

// ============================================================================
// DATA OPERATIONS
// ============================================================================

/// High-level Intermediate Representation for data loading operations.
///
/// Data load operations represent loading external data sources into the
/// program. They support various formats (CSV, JSON, etc.) and optional
/// schema validation.
///
/// # Example
/// ```rust
/// use lexc::hir::{HirDataLoad, HirLiteral};
///
/// let data_load = HirDataLoad {
///     source: "data/users.csv".to_string(),
///     schema_name: Some("UserSchema".to_string()),
///     options: vec![
///         ("delimiter".to_string(), HirLiteral::String(",".to_string())),
///         ("header".to_string(), HirLiteral::Boolean(true)),
///     ],
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirDataLoad {
    /// Path or URL to load data from
    pub source: String,
    /// Optional schema to apply to the data
    pub schema_name: Option<String>,
    /// Options like format, delimiter, etc.
    pub options: Vec<(String, HirLiteral)>,
}

/// High-level Intermediate Representation for data filter operations.
///
/// Data filter operations represent filtering datasets based on conditions.
/// They take an input dataset and a boolean condition expression.
///
/// # Example
/// ```lexon
/// let filtered = users.filter(age > 18);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirDataFilter {
    /// Input dataset
    pub input: Box<HirNode>,
    /// Filter condition
    pub condition: Box<HirNode>,
}

/// High-level Intermediate Representation for data select operations.
///
/// Data select operations represent column selection from datasets.
/// They take an input dataset and a list of field names to select.
///
/// # Example
/// ```lexon
/// let names = users.select(["name", "email"]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirDataSelect {
    /// Input dataset
    pub input: Box<HirNode>,
    /// Fields to select
    pub fields: Vec<String>,
}

/// High-level Intermediate Representation for data take operations.
///
/// Data take operations represent taking the first N rows from a dataset.
/// They take an input dataset and a count expression.
///
/// # Example
/// ```lexon
/// let first_ten = users.take(10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirDataTake {
    /// Input dataset
    pub input: Box<HirNode>,
    /// Number of rows to take
    pub count: Box<HirNode>,
}

/// High-level Intermediate Representation for data export operations.
///
/// Data export operations represent exporting datasets to external files.
/// They support various output formats and options.
///
/// # Example
/// ```lexon
/// users.export("output.csv", "csv", {"header": true});
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirDataExport {
    /// Input dataset
    pub input: Box<HirNode>,
    /// Path to export to
    pub path: String,
    /// Format: "csv", "json", etc.
    pub format: String,
    /// Additional export options
    pub options: Vec<(String, HirLiteral)>,
}

// ============================================================================
// MEMORY OPERATIONS
// ============================================================================

/// High-level Intermediate Representation for memory load operations.
///
/// Memory load operations represent loading data from contextual memory.
/// They support different memory strategies and scoping mechanisms.
///
/// # Example
/// ```lexon
/// let context = memory_load("session_1", "vector");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirMemoryLoad {
    /// Memory scope identifier
    pub scope: String,
    /// Optional dataset to load
    pub source: Option<Box<HirNode>>,
    /// Memory strategy: "vector", "buffer", etc.
    pub strategy: String,
    /// Additional options
    pub options: Vec<(String, HirLiteral)>,
}

/// High-level Intermediate Representation for memory store operations.
///
/// Memory store operations represent storing data in contextual memory.
/// They support scoped storage with optional key-based access.
///
/// # Example
/// ```lexon
/// memory_store("session_1", user_data, "user_profile");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirMemoryStore {
    /// Memory scope identifier
    pub scope: String,
    /// Value to store
    pub value: Box<HirNode>,
    /// Optional key for the stored value
    pub key: Option<String>,
}

// ============================================================================
// LITERALS AND BASIC TYPES
// ============================================================================

/// Represents different types of literals in the HIR.
///
/// Literals are compile-time constant values that appear directly
/// in the source code. They are typed and immutable.
///
/// # Types
/// - **String**: UTF-8 string literals
/// - **MultiLineString**: Multi-line string literals (may have different processing)
/// - **Integer**: 64-bit signed integer literals
/// - **Float**: 64-bit floating point literals
/// - **Boolean**: Boolean literals (true/false)
/// - **Array**: Array literals containing other HIR nodes
///
/// # Example
/// ```rust
/// use lexc::hir::{HirLiteral, HirNode};
///
/// let string_lit = HirLiteral::String("Hello".to_string());
/// let int_lit = HirLiteral::Integer(42);
/// let array_lit = HirLiteral::Array(vec![
///     HirNode::Literal(HirLiteral::Integer(1)),
///     HirNode::Literal(HirLiteral::Integer(2)),
/// ]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum HirLiteral {
    /// UTF-8 string literal
    String(String),
    /// Multi-line string literal
    MultiLineString(String),
    /// 64-bit signed integer literal
    Integer(i64),
    /// 64-bit floating point literal
    Float(f64),
    /// Boolean literal
    Boolean(bool),
    /// Array literal containing HIR nodes
    Array(Vec<HirNode>),
}

// ============================================================================
// DECLARATIONS
// ============================================================================

/// High-level Intermediate Representation for a variable declaration.
///
/// Variable declarations represent the binding of a name to a value
/// with optional type annotation. They are the fundamental building
/// blocks for data storage in LEXON programs.
///
/// # Example
/// ```lexon
/// let name: string = "Alice";
/// let age = 30;  // Type inferred
/// let users = load_data("users.csv");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirVariableDeclaration {
    /// Variable name
    pub name: String,
    /// Optional type annotation
    pub type_name: Option<String>,
    /// The expression assigned to the variable
    pub value: Box<HirNode>,
}

/// Represents a field inside a schema definition.
///
/// Schema fields define the structure of user-defined data types.
/// They include type information, optionality, and default values.
///
/// # Example
/// ```lexon
/// schema User {
///     id: int,
///     name: string,
///     email?: string = "unknown@example.com",
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirSchemaField {
    /// Field name
    pub name: String,
    /// Field type name
    pub type_name: String,
    /// Whether the field is optional
    pub is_optional: bool,
    /// Default value for optional fields
    pub default_value: Option<Box<HirNode>>,
}

/// Declared visibility in source code.
///
/// Visibility controls the accessibility of declarations across
/// module boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HirVisibility {
    /// Public visibility (accessible from other modules)
    Public,
    /// Private visibility (only accessible within current module)
    Private,
}

/// Represents a function parameter in HIR.
///
/// Function parameters define the inputs to functions with their
/// names and types.
///
/// # Example
/// ```lexon
/// fn process_user(name: string, age: int) -> string {
///     // ...
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type name
    pub type_name: String,
}

/// High-level Intermediate Representation for a schema definition.
///
/// Schema definitions create user-defined data types with structured
/// fields. They support generic type parameters and visibility modifiers.
///
/// # Example
/// ```lexon
/// pub schema User<T> {
///     id: int,
///     name: string,
///     data: T,
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirSchemaDefinition {
    /// Schema name
    pub name: String,
    /// Generic type parameters
    pub type_parameters: Vec<String>,
    /// Schema fields
    pub fields: Vec<HirSchemaField>,
    /// Visibility (public/private)
    pub visibility: HirVisibility,
}

/// High-level Intermediate Representation for a function definition.
///
/// Function definitions create reusable code blocks with parameters,
/// return types, and bodies. They support generic type parameters,
/// visibility modifiers, and async/await.
///
/// # Example
/// ```lexon
/// pub async fn analyze_data<T>(data: T) -> string {
///     let result = await ask("Analyze this data: " + data);
///     return result;
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirFunctionDefinition {
    /// Function name
    pub name: String,
    /// Function parameters
    pub parameters: Vec<HirParameter>,
    /// Generic type parameters
    pub type_parameters: Vec<String>,
    /// Return type (optional for void functions)
    pub return_type: Option<String>,
    /// Function body statements
    pub body: Vec<HirNode>,
    /// Visibility (public/private)
    pub visibility: HirVisibility,
    /// Whether the function is async
    pub is_async: bool,
}

/// High-level Intermediate Representation for a module declaration.
///
/// Module declarations define the module structure and organization
/// of LEXON programs. They support hierarchical module paths.
///
/// # Example
/// ```lexon
/// module analytics.stats;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirModuleDeclaration {
    /// Module path, e.g. ["analytics", "stats"]
    pub path: Vec<String>,
}

/// High-level Intermediate Representation for an import statement.
///
/// Import declarations bring external modules and their items into
/// the current scope. They support selective imports and aliasing.
///
/// # Examples
/// ```lexon
/// import core.types;                    // Full module import
/// import core.types as types;           // Module aliasing
/// import core.types::{User, Schema};    // Selective import
/// import core.types::{User as U};       // Item aliasing
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirImportDeclaration {
    /// Base module path (e.g. ["core", "types"])
    pub path: Vec<String>,
    /// Imported items (name, optional alias). If empty means full import.
    pub items: Vec<(String, Option<String>)>,
    /// Alias for full module import (`import foo as bar`)
    pub alias: Option<String>,
}

/// High-level Intermediate Representation for a trait definition.
///
/// Trait definitions define interfaces that types can implement.
/// They support generic type parameters and method signatures.
///
/// # Example
/// ```lexon
/// pub trait Processable<T> {
///     fn process(self, input: T) -> string;
///     fn validate(self) -> bool;
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirTraitDefinition {
    /// Trait name
    pub name: String,
    /// Generic type parameters
    pub type_parameters: Vec<String>,
    /// Method signatures
    pub methods: Vec<HirFunctionSignature>,
    /// Visibility (public/private)
    pub visibility: HirVisibility,
}

/// Represents a function signature within a trait.
///
/// Function signatures define the interface of methods without
/// their implementation. They specify parameters and return types.
///
/// # Example
/// ```lexon
/// trait Analyzer {
///     fn analyze(data: string) -> Result<Analysis>;
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirFunctionSignature {
    /// Function name
    pub name: String,
    /// Function parameters
    pub parameters: Vec<HirParameter>,
    /// Return type (optional for void functions)
    pub return_type: Option<String>,
}

/// High-level Intermediate Representation for an implementation block.
///
/// Implementation blocks provide concrete implementations for traits
/// or define methods for types. They associate methods with types.
///
/// # Example
/// ```lexon
/// impl Processable<string> for User {
///     fn process(self, input: string) -> string {
///         return "Processed: " + input;
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirImplBlock {
    /// Name of the type being implemented
    pub target: String,
    /// Method implementations
    pub methods: Vec<HirFunctionDefinition>,
}

// ============================================================================
// EXPRESSIONS AND CALLS
// ============================================================================

/// High-level Intermediate Representation for method calls.
///
/// Method calls represent invoking methods on objects or types.
/// They include the receiver, method name, and arguments.
///
/// # Example
/// ```lexon
/// let result = user.process("data");
/// let valid = validator.check();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirMethodCall {
    /// Receiver type or identifier
    pub target: String,
    /// Method name
    pub method: String,
    /// Method arguments
    pub args: Vec<HirNode>,
}

/// High-level Intermediate Representation for function calls.
///
/// Function calls represent invoking functions with arguments.
/// They support generic type arguments and overloading.
///
/// # Example
/// ```lexon
/// let result = process_data<string>(input);
/// let output = transform(data, options);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirFunctionCall {
    /// Function name
    pub function: String,
    /// Function arguments
    pub args: Vec<HirNode>,
    /// Generic type arguments
    pub type_arguments: Vec<String>,
}

/// High-level Intermediate Representation for typeof expressions.
///
/// TypeOf expressions determine the type of an expression at runtime.
/// They are useful for type introspection and dynamic behavior.
///
/// # Example
/// ```lexon
/// let data_type = typeof(user_data);
/// if (typeof(value) == "string") { ... }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirTypeOf {
    /// The expression whose type we want to determine
    pub argument: Box<HirNode>,
}

/// Binary operators supported in HIR expressions.
///
/// These operators represent the fundamental binary operations
/// available in LEXON expressions, including arithmetic, comparison,
/// and logical operations.
///
/// # Categories
/// - **Arithmetic**: Add, Subtract, Multiply, Divide
/// - **Comparison**: GreaterThan, LessThan, GreaterEqual, LessEqual, Equal, NotEqual
/// - **Logical**: And, Or
///
/// # Example
/// ```lexon
/// let sum = a + b;           // Add
/// let valid = age >= 18;     // GreaterEqual
/// let result = x > 0 && y < 10;  // GreaterThan + And
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum HirBinaryOperator {
    /// Addition operator (+)
    Add,
    /// Subtraction operator (-)
    Subtract,
    /// Multiplication operator (*)
    Multiply,
    /// Division operator (/)
    Divide,
    /// Greater than operator (>)
    GreaterThan,
    /// Less than operator (<)
    LessThan,
    /// Greater than or equal operator (>=)
    GreaterEqual,
    /// Less than or equal operator (<=)
    LessEqual,
    /// Equality operator (==)
    Equal,
    /// Inequality operator (!=)
    NotEqual,
    /// Logical AND operator (&&)
    And,
    /// Logical OR operator (||)
    Or,
}

impl std::fmt::Display for HirBinaryOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl HirBinaryOperator {
    /// Returns the string representation of the operator.
    pub fn as_str(&self) -> &str {
        match self {
            HirBinaryOperator::Add => "+",
            HirBinaryOperator::Subtract => "-",
            HirBinaryOperator::Multiply => "*",
            HirBinaryOperator::Divide => "/",
            HirBinaryOperator::GreaterThan => ">",
            HirBinaryOperator::LessThan => "<",
            HirBinaryOperator::GreaterEqual => ">=",
            HirBinaryOperator::LessEqual => "<=",
            HirBinaryOperator::Equal => "==",
            HirBinaryOperator::NotEqual => "!=",
            HirBinaryOperator::And => "&&",
            HirBinaryOperator::Or => "||",
        }
    }
}

/// High-level Intermediate Representation for binary expressions.
///
/// Binary expressions represent operations between two operands
/// with a binary operator. They form the basis of arithmetic,
/// comparison, and logical operations.
///
/// # Example
/// ```lexon
/// let result = (a + b) * (c - d);
/// let valid = age >= 18 && status == "active";
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirBinaryExpression {
    /// Left operand
    pub left: Box<HirNode>,
    /// Binary operator
    pub operator: HirBinaryOperator,
    /// Right operand
    pub right: Box<HirNode>,
}

// ============================================================================
// CONTROL FLOW AND STATEMENTS
// ============================================================================

/// High-level Intermediate Representation for while loops.
///
/// While loops represent conditional iteration that continues
/// as long as the condition evaluates to true.
///
/// # Example
/// ```lexon
/// let i = 0;
/// while (i < 10) {
///     print(i);
///     i = i + 1;
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirWhile {
    /// Loop condition
    pub condition: Box<HirNode>,
    /// Loop body
    pub body: Vec<HirNode>,
}

/// High-level Intermediate Representation for if statements.
///
/// If statements provide conditional execution with optional
/// else branches for alternative execution paths.
///
/// # Example
/// ```lexon
/// if (user.age >= 18) {
///     print("Adult");
/// } else {
///     print("Minor");
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirIf {
    /// Condition to evaluate
    pub condition: Box<HirNode>,
    /// Body to execute if condition is true
    pub then_body: Vec<HirNode>,
    /// Optional body to execute if condition is false
    pub else_body: Option<Vec<HirNode>>,
}

/// High-level Intermediate Representation for for-in loops.
///
/// For-in loops represent iteration over collections, binding
/// each element to an iterator variable.
///
/// # Example
/// ```lexon
/// for user in users {
///     print(user.name);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirForIn {
    /// Iterator variable name
    pub iterator: String,
    /// Iterable expression (dataset or other collection)
    pub iterable: Box<HirNode>,
    /// Loop body
    pub body: Vec<HirNode>,
}

/// High-level Intermediate Representation for assignments.
///
/// Assignment statements bind values to variables, updating
/// the program state with new values.
///
/// # Example
/// ```lexon
/// name = "Alice";
/// age = age + 1;
/// result = ask("What is the weather?");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirAssignment {
    /// Variable name being assigned to
    pub left: String,
    /// Expression being assigned
    pub right: Box<HirNode>,
}

/// High-level Intermediate Representation for await expressions.
///
/// Await expressions represent waiting for asynchronous operations
/// to complete, typically used with async ask expressions.
///
/// # Example
/// ```lexon
/// let result = await ask("Analyze this data");
/// let processed = await process_async(data);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirAwait {
    /// The expression being awaited (typically an ask expression)
    pub expression: Box<HirNode>,
}

/// High-level Intermediate Representation for return statements.
///
/// Return statements transfer control back to the caller with
/// an optional return value.
///
/// # Example
/// ```lexon
/// fn calculate(x: int) -> int {
///     return x * 2;
/// }
///
/// fn log_message(msg: string) {
///     print(msg);
///     return;  // void return
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirReturn {
    /// Optional return value expression
    pub expression: Option<Box<HirNode>>,
}

/// High-level Intermediate Representation for match expressions.
///
/// Match expressions provide pattern matching with multiple
/// branches, enabling sophisticated control flow based on
/// value patterns.
///
/// # Example
/// ```lexon
/// match result {
///     Ok(value) => print("Success: " + value),
///     Err(error) => print("Error: " + error),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirMatch {
    /// Expression to evaluate
    pub value: Box<HirNode>,
    /// Match arms
    pub arms: Vec<HirMatchArm>,
}

/// Represents a single arm in a match expression.
///
/// Match arms contain a pattern to match against and a body
/// to execute when the pattern matches.
///
/// # Example
/// ```lexon
/// match status {
///     "active" => enable_user(),
///     "inactive" => disable_user(),
///     _ => log_unknown_status(),
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HirMatchArm {
    /// Pattern to match
    pub pattern: Box<HirNode>,
    /// Body to execute if pattern matches
    pub body: Vec<HirNode>,
}

// ============================================================================
// MAIN HIR NODE ENUM
// ============================================================================

/// The main HIR node type representing all language constructs.
///
/// This enum unifies all possible HIR nodes into a single type,
/// enabling the representation of complete LEXON programs as
/// trees of HIR nodes.
///
/// # Categories
/// - **LLM Operations**: Ask, AskSafe
/// - **Literals**: String, integer, float, boolean, array literals
/// - **Declarations**: Variable, function, schema, module, import, trait declarations
/// - **Data Operations**: DataLoad, DataFilter, DataSelect, DataTake, DataExport
/// - **Memory Operations**: MemoryLoad, MemoryStore
/// - **Control Flow**: If, While, ForIn, Match, Break, Continue, Return
/// - **Expressions**: Binary, Assignment, Await, TypeOf, FunctionCall, MethodCall
/// - **References**: Identifier for variable references
///
/// # Example
/// ```rust
/// use lexc::hir::{HirNode, HirLiteral, HirBinaryExpression, HirBinaryOperator};
///
/// let left = HirNode::Literal(HirLiteral::Integer(5));
/// let right = HirNode::Literal(HirLiteral::Integer(3));
/// let binary = HirNode::Binary(Box::new(HirBinaryExpression {
///     left: Box::new(left),
///     operator: HirBinaryOperator::Add,
///     right: Box::new(right),
/// }));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum HirNode {
    // LLM Operations
    /// Ask expression for LLM calls
    Ask(Box<HirAskExpression>),
    /// Ask_safe expression with anti-hallucination validation
    AskSafe(Box<HirAskSafeExpression>),

    // Literals
    /// Literal values (strings, integers, floats, booleans, arrays)
    Literal(HirLiteral),

    // Declarations
    /// Variable declaration
    VariableDeclaration(Box<HirVariableDeclaration>),
    /// Function definition
    FunctionDefinition(Box<HirFunctionDefinition>),
    /// Schema definition
    SchemaDefinition(Box<HirSchemaDefinition>),
    /// Module declaration
    ModuleDeclaration(Box<HirModuleDeclaration>),
    /// Import declaration
    ImportDeclaration(Box<HirImportDeclaration>),
    /// Trait definition
    TraitDefinition(Box<HirTraitDefinition>),
    /// Implementation block
    ImplBlock(Box<HirImplBlock>),

    // Data Operations
    /// Data loading operation
    DataLoad(Box<HirDataLoad>),
    /// Data filtering operation
    DataFilter(Box<HirDataFilter>),
    /// Data column selection
    DataSelect(Box<HirDataSelect>),
    /// Data row limiting
    DataTake(Box<HirDataTake>),
    /// Data export operation
    DataExport(Box<HirDataExport>),

    // Memory Operations
    /// Memory load operation
    MemoryLoad(Box<HirMemoryLoad>),
    /// Memory store operation
    MemoryStore(Box<HirMemoryStore>),

    // Control Flow
    /// If statement
    If(Box<HirIf>),
    /// While loop
    While(Box<HirWhile>),
    /// For-in loop
    ForIn(Box<HirForIn>),
    /// Match expression
    Match(Box<HirMatch>),
    /// Break statement
    Break,
    /// Continue statement
    Continue,
    /// Return statement
    Return(Box<HirReturn>),

    // Expressions
    /// Binary expression
    Binary(Box<HirBinaryExpression>),
    /// Assignment statement
    Assignment(Box<HirAssignment>),
    /// Await expression
    Await(Box<HirAwait>),
    /// TypeOf expression
    TypeOf(Box<HirTypeOf>),
    /// Function call
    FunctionCall(Box<HirFunctionCall>),
    /// Method call
    MethodCall(Box<HirMethodCall>),

    // References
    /// Reference to an existing variable
    Identifier(String),
}

// Placeholder for future detailed source span information
// #[derive(Debug, Clone, PartialEq)]
// pub struct SourceSpan {
//     pub start_byte: usize,
//     pub end_byte: usize,
//     // pub file_id: FileId, // If we manage multiple files
// }

// Placeholder for future resolved schema/type identifiers
// pub type ResolvedSchemaId = usize; // Or some other unique identifier
// pub type ResolvedTypeRef = usize;

// More HIR structures will be added here as the compiler develops.
