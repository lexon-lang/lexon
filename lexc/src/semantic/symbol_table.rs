// lexc/src/semantic/symbol_table.rs
//
// Symbol table implementation for tracking declarations and scopes

use super::type_resolver::{ResolvedType, TypeId};
use std::collections::{HashMap, HashSet};

/// Unique identifier for a declaration (variable, function, schema) in the program
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeclId(pub usize);

/// Types of declarations that can be stored in the symbol table
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Variable {
        name: String,
        type_id: Option<TypeId>,
        is_mutable: bool, // from 'let' vs 'var'
    },
    Function {
        name: String,
        return_type_id: Option<TypeId>,
        // Will add parameters later
    },
    Schema {
        name: String,
        fields: Vec<SchemaFieldInfo>,
    },
}

/// Information about a field in a schema definition
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaFieldInfo {
    pub name: String,
    pub type_id: TypeId,
    pub is_optional: bool,
    pub has_default: bool,
}

/// Manages scopes and declarations during semantic analysis
#[derive(Debug)]
pub struct SymbolTable {
    // Declarations by ID
    declarations: Vec<Declaration>,

    // Types by ID
    types: Vec<ResolvedType>,

    // Current scope stack (indices into declarations)
    scopes: Vec<HashSet<DeclId>>,

    // Mapping from name to declaration ID within the current scope chain
    name_map: HashMap<String, DeclId>,

    // Primitive type IDs for quick access
    primitive_types: HashMap<String, TypeId>,
}

impl SymbolTable {
    /// Creates a new, empty symbol table with primitive types preregistered
    pub fn new() -> Self {
        // Register primitive types later once type_resolver is implemented
        SymbolTable {
            declarations: Vec::new(),
            types: Vec::new(),
            scopes: vec![HashSet::new()], // Start with global scope
            name_map: HashMap::new(),
            primitive_types: HashMap::new(),
        }
    }

    /// Sets up primitive types in the symbol table
    pub fn initialize_primitive_types(&mut self) {
        use super::type_resolver::{PrimitiveType, ResolvedType};

        // Register primitive types
        let int_id = self.add_type(ResolvedType::Primitive(PrimitiveType::Int));
        let float_id = self.add_type(ResolvedType::Primitive(PrimitiveType::Float));
        let string_id = self.add_type(ResolvedType::Primitive(PrimitiveType::String));
        let bool_id = self.add_type(ResolvedType::Primitive(PrimitiveType::Bool));

        self.primitive_types.insert("int".to_string(), int_id);
        self.primitive_types.insert("float".to_string(), float_id);
        self.primitive_types.insert("string".to_string(), string_id);
        self.primitive_types.insert("bool".to_string(), bool_id);
    }

    /// Adds a new type to the symbol table and returns its ID
    pub fn add_type(&mut self, type_val: ResolvedType) -> TypeId {
        let id = TypeId(self.types.len());
        self.types.push(type_val);
        id
    }

    /// Adds a new declaration to the symbol table and returns its ID
    pub fn add_declaration(
        &mut self,
        decl: Declaration,
    ) -> Result<DeclId, super::analyzer::SemanticError> {
        // Extract name and make a clone to avoid ownership issues
        let name = match &decl {
            Declaration::Variable { name, .. } => name.clone(),
            Declaration::Function { name, .. } => name.clone(),
            Declaration::Schema { name, .. } => name.clone(),
        };

        // Check if the name is already declared in the current scope
        if self.name_map.contains_key(&name) {
            return Err(super::analyzer::SemanticError::NameAlreadyDeclared(name));
        }

        // Add the declaration
        let id = DeclId(self.declarations.len());
        self.declarations.push(decl);

        // Register the declaration in the current scope
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.insert(id);
        }

        // Add to the name map
        self.name_map.insert(name, id);

        Ok(id)
    }

    /// Enters a new scope for local declarations
    pub fn enter_scope(&mut self) {
        self.scopes.push(HashSet::new());
    }

    /// Exits the current scope, removing local declarations
    pub fn exit_scope(&mut self) {
        if let Some(scope) = self.scopes.pop() {
            for decl_id in scope {
                let name = match &self.declarations[decl_id.0] {
                    Declaration::Variable { name, .. } => name,
                    Declaration::Function { name, .. } => name,
                    Declaration::Schema { name, .. } => name,
                };
                self.name_map.remove(name);
            }
        }
    }

    /// Looks up a declaration by name in the current scope chain
    pub fn lookup_name(&self, name: &str) -> Option<DeclId> {
        self.name_map.get(name).copied()
    }

    /// Gets a reference to a declaration by ID
    pub fn get_declaration(&self, id: DeclId) -> Option<&Declaration> {
        self.declarations.get(id.0)
    }

    /// Gets a reference to a type by ID
    pub fn get_type(&self, id: TypeId) -> Option<&ResolvedType> {
        self.types.get(id.0)
    }

    /// Resolves a type name to a TypeId
    pub fn resolve_type_name(
        &mut self,
        type_name: &str,
    ) -> Result<TypeId, super::analyzer::SemanticError> {
        // Check for primitive types first
        if let Some(type_id) = self.primitive_types.get(type_name) {
            return Ok(*type_id);
        }

        // Check if it's a reference to a schema
        if let Some(decl_id) = self.lookup_name(type_name) {
            if let Some(Declaration::Schema { .. }) = self.get_declaration(decl_id) {
                // Create a type reference to this schema if not exists
                return Ok(self.add_type(ResolvedType::Schema(decl_id)));
            }
        }

        // Type not found
        Err(super::analyzer::SemanticError::TypeNotFound(
            type_name.to_string(),
        ))
    }

    /// Updates the fields of a schema declaration
    pub fn update_schema_fields(
        &mut self,
        schema_id: DeclId,
        fields: Vec<SchemaFieldInfo>,
    ) -> Result<(), super::analyzer::SemanticError> {
        let decl = match self.declarations.get_mut(schema_id.0) {
            Some(decl) => decl,
            None => {
                return Err(super::analyzer::SemanticError::NameNotFound(
                    "schema".to_string(),
                ))
            }
        };

        match decl {
            Declaration::Schema {
                fields: schema_fields,
                ..
            } => {
                *schema_fields = fields;
                Ok(())
            }
            _ => Err(super::analyzer::SemanticError::TypeMismatch {
                expected: "schema".to_string(),
                found: "non-schema".to_string(),
            }),
        }
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}
