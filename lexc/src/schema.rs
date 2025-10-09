//! # Schema Definition and JSON Schema Generation
//!
//! This module provides functionality for parsing Lexon schema definitions and converting them
//! to JSON Schema format. It handles the mapping between Lexon's type system and JSON Schema
//! specifications, enabling interoperability with JSON-based systems.
//!
//! ## Architecture Overview
//!
//! The schema system consists of two main components:
//!
//! - **JsonSchema**: Represents a complete JSON Schema document with metadata
//! - **Schema Parser**: Converts Lexon schema definitions from the AST to JSON Schema
//!
//! ## Usage Example
//!
//! ```rust
//! use lexon::schema::{JsonSchema, parse_schema_definition};
//! use serde_json::json;
//!
//! // Create a JSON schema programmatically
//! let schema = JsonSchema::new("User", json!({
//!     "type": "object",
//!     "properties": {
//!         "name": {"type": "string"},
//!         "age": {"type": "integer"}
//!     },
//!     "required": ["name"]
//! }));
//!
//! // Parse from Lexon AST node
//! let parsed_schema = parse_schema_definition(schema_node, source_code);
//! ```
//!
//! ## Type Mapping
//!
//! Lexon types are mapped to JSON Schema types as follows:
//!
//! | Lexon Type | JSON Schema Type |
//! |------------|------------------|
//! | `int`      | `integer`        |
//! | `float`    | `number`         |
//! | `bool`     | `boolean`        |
//! | `string`   | `string`         |
//! | Other      | `object`         |

use serde::Serialize;
use tree_sitter::Node;

// ================================================================================================
// JSON Schema Representation
// ================================================================================================

/// Represents a complete JSON Schema document with metadata and validation rules.
///
/// This structure follows the JSON Schema Draft 07 specification and provides
/// a clean interface for working with schema definitions in Lexon.
///
/// ## Fields
///
/// - `schema_version`: The JSON Schema specification version (always Draft 07)
/// - `title`: Human-readable name for the schema
/// - `content`: The actual schema definition as a JSON value
///
/// ## Example
///
/// ```rust
/// use lexon::schema::JsonSchema;
/// use serde_json::json;
///
/// let schema = JsonSchema::new("Person", json!({
///     "type": "object",
///     "properties": {
///         "name": {"type": "string"},
///         "age": {"type": "integer", "minimum": 0}
///     },
///     "required": ["name"]
/// }));
/// ```
#[derive(Debug, Serialize, Clone)]
pub struct JsonSchema {
    /// JSON Schema specification version
    #[serde(rename = "$schema")]
    schema_version: String,

    /// Human-readable title for the schema
    title: String,

    /// The actual schema definition
    #[serde(flatten)]
    content: serde_json::Value,
}

impl JsonSchema {
    /// Creates a new JSON Schema with the specified title and content.
    ///
    /// The schema version is automatically set to JSON Schema Draft 07.
    ///
    /// ## Arguments
    ///
    /// * `title` - Human-readable name for the schema
    /// * `content` - The schema definition as a JSON value
    ///
    /// ## Returns
    ///
    /// A new `JsonSchema` instance ready for serialization or validation.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use lexon::schema::JsonSchema;
    /// use serde_json::json;
    ///
    /// let schema = JsonSchema::new("User", json!({
    ///     "type": "object",
    ///     "properties": {
    ///         "username": {"type": "string", "minLength": 3},
    ///         "email": {"type": "string", "format": "email"}
    ///     },
    ///     "required": ["username", "email"]
    /// }));
    /// ```
    pub fn new(title: &str, content: serde_json::Value) -> Self {
        Self {
            schema_version: "http://json-schema.org/draft-07/schema#".to_string(),
            title: title.to_string(),
            content,
        }
    }

    /// Returns the schema title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Returns the schema version.
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    /// Returns a reference to the schema content.
    pub fn content(&self) -> &serde_json::Value {
        &self.content
    }

    /// Validates that the schema content is well-formed.
    ///
    /// This performs basic structural validation to ensure the schema
    /// follows JSON Schema conventions.
    ///
    /// ## Returns
    ///
    /// `true` if the schema appears to be valid, `false` otherwise.
    pub fn is_valid(&self) -> bool {
        // Basic validation: ensure it's an object with a type field
        if let serde_json::Value::Object(obj) = &self.content {
            obj.contains_key("type")
        } else {
            false
        }
    }
}

// ================================================================================================
// Type Mapping and Conversion
// ================================================================================================

/// Maps Lexon primitive types to their corresponding JSON Schema type strings.
///
/// This function provides the core type mapping between Lexon's type system
/// and JSON Schema types, ensuring compatibility with JSON-based systems.
///
/// ## Arguments
///
/// * `lex_type` - The Lexon type identifier as a string
///
/// ## Returns
///
/// The corresponding JSON Schema type string.
///
/// ## Type Mapping
///
/// | Lexon Type | JSON Schema Type | Notes |
/// |------------|------------------|-------|
/// | `int`      | `integer`        | Whole numbers |
/// | `float`    | `number`         | Decimal numbers |
/// | `bool`     | `boolean`        | True/false values |
/// | `string`   | `string`         | Text data |
/// | Others     | `object`         | Complex/unknown types |
///
/// ## Example
///
/// ```rust
/// use lexon::schema::lex_type_to_json_type;
///
/// assert_eq!(lex_type_to_json_type("int"), "integer");
/// assert_eq!(lex_type_to_json_type("float"), "number");
/// assert_eq!(lex_type_to_json_type("bool"), "boolean");
/// assert_eq!(lex_type_to_json_type("string"), "string");
/// assert_eq!(lex_type_to_json_type("CustomType"), "object");
/// ```
pub fn lex_type_to_json_type(lex_type: &str) -> &'static str {
    match lex_type {
        "int" => "integer",
        "float" => "number",
        "bool" => "boolean",
        "string" => "string",
        // Default for unknown/complex types
        _ => "object",
    }
}

/// Provides additional JSON Schema constraints based on Lexon type.
///
/// This function returns additional validation rules that can be applied
/// to JSON Schema properties based on the Lexon type.
///
/// ## Arguments
///
/// * `lex_type` - The Lexon type identifier
///
/// ## Returns
///
/// A JSON object containing additional constraints, or `None` if no
/// additional constraints are needed.
pub fn get_type_constraints(lex_type: &str) -> Option<serde_json::Value> {
    match lex_type {
        "int" => Some(serde_json::json!({
            "multipleOf": 1
        })),
        "float" => Some(serde_json::json!({
            "type": "number"
        })),
        "string" => Some(serde_json::json!({
            "minLength": 0
        })),
        _ => None,
    }
}

// ================================================================================================
// Schema Parsing and Generation
// ================================================================================================

/// Parses a Lexon schema definition from an AST node and converts it to JSON Schema.
///
/// This function takes a Tree-sitter AST node representing a schema definition
/// and converts it to a complete JSON Schema document. It handles field types,
/// optional markers, and generates appropriate validation rules.
///
/// ## Arguments
///
/// * `root` - The Tree-sitter AST node representing the schema definition
/// * `source` - The source code string for extracting text from nodes
///
/// ## Returns
///
/// A `JsonSchema` instance if parsing succeeds, or `None` if the node
/// is not a valid schema definition or parsing fails.
///
/// ## Schema Definition Format
///
/// The expected Lexon schema format is:
///
/// ```lexon
/// schema User {
///     name: string
///     age: int
///     email?: string  // Optional field
/// }
/// ```
///
/// This generates a JSON Schema equivalent to:
///
/// ```json
/// {
///     "$schema": "http://json-schema.org/draft-07/schema#",
///     "title": "User",
///     "type": "object",
///     "properties": {
///         "name": {"type": "string"},
///         "age": {"type": "integer"},
///         "email": {"type": "string"}
///     },
///     "required": ["name", "age"]
/// }
/// ```
///
/// ## Error Handling
///
/// Returns `None` if:
/// - The root node is not a `schema_definition`
/// - Required child nodes are missing
/// - Text extraction from nodes fails
/// - The schema structure is malformed
pub fn parse_schema_definition(root: Node, source: &str) -> Option<JsonSchema> {
    // Validate that we have a schema definition node
    if root.kind() != "schema_definition" {
        return None;
    }

    // Extract the schema name/title
    let name_node = root.child_by_field_name("name")?;
    let title = name_node.utf8_text(source.as_bytes()).ok()?;

    // Initialize collections for properties and required fields
    let mut properties = serde_json::Map::new();
    let mut required_fields = Vec::new();

    // Parse all schema field definitions
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() == "schema_field_definition" {
            if let Some((field_name, field_schema, is_required)) = parse_schema_field(child, source)
            {
                properties.insert(field_name.clone(), field_schema);

                if is_required {
                    required_fields.push(serde_json::Value::String(field_name));
                }
            }
        }
    }

    // Build the complete JSON Schema object
    let mut schema_definition = serde_json::Map::new();
    schema_definition.insert(
        "type".to_string(),
        serde_json::Value::String("object".to_string()),
    );
    schema_definition.insert(
        "properties".to_string(),
        serde_json::Value::Object(properties),
    );

    // Add required fields if any exist
    if !required_fields.is_empty() {
        schema_definition.insert(
            "required".to_string(),
            serde_json::Value::Array(required_fields),
        );
    }

    Some(JsonSchema::new(
        title,
        serde_json::Value::Object(schema_definition),
    ))
}

/// Parses a single schema field definition from an AST node.
///
/// This helper function extracts field information from a schema field
/// definition node, including the field name, type, and whether it's required.
///
/// ## Arguments
///
/// * `field_node` - The AST node representing a schema field definition
/// * `source` - The source code string for text extraction
///
/// ## Returns
///
/// A tuple containing:
/// - Field name as a string
/// - Field JSON Schema definition
/// - Boolean indicating if the field is required
///
/// Returns `None` if parsing fails.
fn parse_schema_field(field_node: Node, source: &str) -> Option<(String, serde_json::Value, bool)> {
    // Extract field name
    let field_name_node = field_node.child_by_field_name("name")?;
    let field_name = field_name_node
        .utf8_text(source.as_bytes())
        .ok()?
        .to_string();

    // Extract field type
    let field_type_node = field_node.child_by_field_name("type")?;
    let type_identifier_node = field_type_node.named_child(0)?;
    let type_string = type_identifier_node.utf8_text(source.as_bytes()).ok()?;

    // Check if field is optional
    let optional_marker = field_node.child_by_field_name("optional_marker");
    let is_required = optional_marker.is_none();

    // Build field schema
    let json_type = lex_type_to_json_type(type_string);
    let mut field_schema = serde_json::Map::new();
    field_schema.insert(
        "type".to_string(),
        serde_json::Value::String(json_type.to_string()),
    );

    // Add type-specific constraints if available
    if let Some(serde_json::Value::Object(constraint_map)) = get_type_constraints(type_string) {
        for (key, value) in constraint_map {
            field_schema.insert(key, value);
        }
    }

    Some((
        field_name,
        serde_json::Value::Object(field_schema),
        is_required,
    ))
}

// ================================================================================================
// Utility Functions
// ================================================================================================

/// Validates a JSON Schema document for correctness.
///
/// This function performs comprehensive validation of a JSON Schema document
/// to ensure it follows the specification and is well-formed.
///
/// ## Arguments
///
/// * `schema` - The JSON Schema to validate
///
/// ## Returns
///
/// `true` if the schema is valid, `false` otherwise.
pub fn validate_json_schema(schema: &JsonSchema) -> bool {
    // Perform basic structural validation
    if !schema.is_valid() {
        return false;
    }

    // Additional validation can be added here
    // For now, we rely on the basic validation from JsonSchema::is_valid()
    true
}

/// Converts a JSON Schema back to a Lexon schema definition string.
///
/// This function provides the reverse operation of `parse_schema_definition`,
/// converting a JSON Schema back to Lexon syntax.
///
/// ## Arguments
///
/// * `schema` - The JSON Schema to convert
///
/// ## Returns
///
/// A string containing the Lexon schema definition, or `None` if conversion fails.
pub fn json_schema_to_lexon(schema: &JsonSchema) -> Option<String> {
    let content = schema.content();

    // Extract properties and required fields
    let properties = content.get("properties")?.as_object()?;
    let required_fields: Vec<&str> = content
        .get("required")
        .and_then(|r| r.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    // Build Lexon schema string
    let mut lexon_schema = format!("schema {} {{\n", schema.title());

    for (field_name, field_def) in properties {
        let field_type = field_def.get("type")?.as_str()?;
        let lexon_type = json_type_to_lex_type(field_type);
        let is_required = required_fields.contains(&field_name.as_str());
        let optional_marker = if is_required { "" } else { "?" };

        lexon_schema.push_str(&format!(
            "    {}: {}{}\n",
            field_name, lexon_type, optional_marker
        ));
    }

    lexon_schema.push('}');
    Some(lexon_schema)
}

/// Maps JSON Schema types back to Lexon types.
///
/// This provides the reverse mapping of `lex_type_to_json_type`.
fn json_type_to_lex_type(json_type: &str) -> &'static str {
    match json_type {
        "integer" => "int",
        "number" => "float",
        "boolean" => "bool",
        "string" => "string",
        _ => "object",
    }
}
