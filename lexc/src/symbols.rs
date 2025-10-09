//! # Symbol Table and Name Resolution System
//!
//! This module provides the core symbol table infrastructure for the Lexon programming language.
//! The symbol table manages the hierarchical structure of modules, tracks symbol declarations,
//! and provides efficient name resolution capabilities.
//!
//! ## Architecture Overview
//!
//! The symbol table system consists of several key components:
//!
//! - **ModuleId**: Unique identifiers for modules in the compilation unit
//! - **ItemId**: Unique identifiers for symbols (functions, schemas, variables, etc.)
//! - **Symbol**: Individual entries containing metadata about declared items
//! - **ModuleInfo**: Hierarchical module structure with parent-child relationships
//! - **SymbolTable**: Global registry managing all symbols and modules
//!
//! ## Design Principles
//!
//! 1. **Hierarchical Modules**: Support for nested module structures
//! 2. **Unique Identification**: Every symbol has a unique ItemId
//! 3. **Efficient Resolution**: Fast lookup by module and name
//! 4. **Visibility Control**: Public/private symbol visibility
//! 5. **Type Safety**: Strong typing for IDs to prevent confusion
//!
//! ## Usage Example
//!
//! ```rust
//! use lexc::symbols::{SymbolTable, SymbolKind, Visibility};
//!
//! let mut table = SymbolTable::new();
//! let module_id = table.declare_module(vec!["std".to_string(), "io".to_string()]);
//! let symbol_id = table.declare_symbol(
//!     module_id,
//!     "println".to_string(),
//!     SymbolKind::Function,
//!     Visibility::Public
//! );
//! ```

use std::collections::HashMap;

// ============================================================================
// CORE IDENTIFIERS
// ============================================================================

/// Unique module identifier within the compilation unit.
///
/// Each module in the Lexon compilation gets a unique `ModuleId` that serves
/// as its primary identifier throughout the compilation process. This provides
/// type safety and prevents confusion between different types of IDs.
///
/// # Examples
///
/// ```rust
/// let root_module = ModuleId(0);
/// let std_module = ModuleId(1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ModuleId(pub u32);

/// Unique item identifier for symbols (functions, schemas, variables, etc.).
///
/// Every symbol declared in the Lexon source code gets a unique `ItemId` that
/// distinguishes it from all other symbols in the compilation. This enables
/// efficient symbol resolution and cross-referencing.
///
/// # Examples
///
/// ```rust
/// let main_function = ItemId(0);
/// let user_schema = ItemId(1);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ItemId(pub u32);

// ============================================================================
// SYMBOL CLASSIFICATION
// ============================================================================

/// Visibility modifier for symbols within their containing module.
///
/// Controls whether a symbol can be accessed from outside its declaring module.
/// This is fundamental to Lexon's module system and encapsulation.
///
/// # Variants
///
/// - `Public`: Symbol is accessible from other modules
/// - `Private`: Symbol is only accessible within its declaring module
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    /// Symbol is accessible from other modules
    Public,
    /// Symbol is only accessible within its declaring module
    Private,
}

/// Classification of different types of symbols in the Lexon language.
///
/// This enum categorizes the various kinds of items that can be declared
/// in Lexon source code. Each symbol kind has different resolution rules
/// and semantic meaning.
///
/// # Symbol Categories
///
/// - **Containers**: Module, Trait, Impl
/// - **Callables**: Function
/// - **Types**: Schema
/// - **Values**: Variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {
    /// Module declaration (container for other symbols)
    Module,
    /// Function declaration (callable item)
    Function,
    /// Schema declaration (type definition)
    Schema,
    /// Variable declaration (value binding)
    Variable,
    /// Trait declaration (interface definition)
    Trait,
    /// Implementation block (trait implementation)
    Impl,
}

impl SymbolKind {
    /// Returns true if this symbol kind represents a container type.
    ///
    /// Container types can contain other symbols and form hierarchical
    /// structures in the symbol table.
    pub fn is_container(&self) -> bool {
        matches!(
            self,
            SymbolKind::Module | SymbolKind::Trait | SymbolKind::Impl
        )
    }

    /// Returns true if this symbol kind represents a callable item.
    ///
    /// Callable items can be invoked with arguments and may return values.
    pub fn is_callable(&self) -> bool {
        matches!(self, SymbolKind::Function)
    }

    /// Returns true if this symbol kind represents a type definition.
    ///
    /// Type definitions can be used in type annotations and declarations.
    pub fn is_type(&self) -> bool {
        matches!(self, SymbolKind::Schema | SymbolKind::Trait)
    }

    /// Returns a human-readable description of the symbol kind.
    pub fn description(&self) -> &'static str {
        match self {
            SymbolKind::Module => "module",
            SymbolKind::Function => "function",
            SymbolKind::Schema => "schema",
            SymbolKind::Variable => "variable",
            SymbolKind::Trait => "trait",
            SymbolKind::Impl => "implementation",
        }
    }
}

// ============================================================================
// SYMBOL REPRESENTATION
// ============================================================================

/// Individual entry in the symbol table containing metadata about a declared item.
///
/// Each symbol represents a single named item in the Lexon source code, along
/// with its classification, location, and accessibility information.
///
/// # Fields
///
/// - `id`: Unique identifier for this symbol
/// - `name`: The declared name of the symbol
/// - `kind`: Classification of what type of symbol this is
/// - `module`: Which module contains this symbol
/// - `visibility`: Whether the symbol is public or private
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Unique identifier for this symbol
    pub id: ItemId,
    /// The declared name of the symbol
    pub name: String,
    /// Classification of what type of symbol this is
    pub kind: SymbolKind,
    /// Which module contains this symbol
    pub module: ModuleId,
    /// Whether the symbol is public or private
    pub visibility: Visibility,
}

impl Symbol {
    /// Creates a new symbol with the given parameters.
    pub fn new(
        id: ItemId,
        name: String,
        kind: SymbolKind,
        module: ModuleId,
        visibility: Visibility,
    ) -> Self {
        Self {
            id,
            name,
            kind,
            module,
            visibility,
        }
    }

    /// Returns true if this symbol is publicly accessible.
    pub fn is_public(&self) -> bool {
        matches!(self.visibility, Visibility::Public)
    }

    /// Returns true if this symbol is privately scoped.
    pub fn is_private(&self) -> bool {
        matches!(self.visibility, Visibility::Private)
    }

    /// Returns the fully qualified name of this symbol.
    ///
    /// This method requires access to the symbol table to resolve the module path.
    pub fn qualified_name(&self, table: &SymbolTable) -> String {
        if let Some(module_info) = table.modules.get(&self.module) {
            let module_path = module_info.path.join("::");
            if module_path.is_empty() {
                self.name.clone()
            } else {
                format!("{}::{}", module_path, self.name)
            }
        } else {
            self.name.clone()
        }
    }
}

// ============================================================================
// MODULE MANAGEMENT
// ============================================================================

/// Information about a module and its hierarchical relationships.
///
/// Modules in Lexon form a tree structure where each module can contain
/// child modules and symbols. This structure enables organized code
/// organization and namespace management.
///
/// # Hierarchical Structure
///
/// - **Root Module**: Has no parent (parent = None)
/// - **Nested Modules**: Have a parent module
/// - **Child Modules**: Listed in the children vector
/// - **Symbols**: All symbols declared directly in this module
#[derive(Debug, Default)]
pub struct ModuleInfo {
    /// Unique identifier for this module
    pub id: ModuleId,
    /// Full path components (e.g., ["std", "io"] for std::io)
    pub path: Vec<String>,
    /// Parent module (None for root module)
    pub parent: Option<ModuleId>,
    /// Direct child modules
    pub children: Vec<ModuleId>,
    /// All symbols declared in this module
    pub symbols: Vec<ItemId>,
}

impl ModuleInfo {
    /// Returns true if this is the root module.
    pub fn is_root(&self) -> bool {
        self.parent.is_none()
    }

    /// Returns the depth of this module in the hierarchy.
    pub fn depth(&self) -> usize {
        self.path.len()
    }

    /// Returns the simple name of this module (last path component).
    pub fn simple_name(&self) -> Option<&str> {
        self.path.last().map(|s| s.as_str())
    }

    /// Returns the full qualified path as a string.
    pub fn qualified_path(&self) -> String {
        self.path.join("::")
    }
}

// ============================================================================
// SYMBOL TABLE IMPLEMENTATION
// ============================================================================

/// Global symbol table managing all modules and symbols in the compilation.
///
/// The symbol table is the central registry for all named items in a Lexon
/// compilation. It maintains the hierarchical module structure and provides
/// efficient name resolution capabilities.
///
/// # Key Features
///
/// - **Hierarchical Modules**: Support for nested module structures
/// - **Unique Identification**: Every symbol gets a unique ItemId
/// - **Efficient Lookup**: Fast resolution by module and name
/// - **Incremental Construction**: Symbols can be added during compilation
/// - **Cross-References**: Symbols can reference other symbols by ID
///
/// # Usage Pattern
///
/// 1. Create a new symbol table
/// 2. Declare modules as they are encountered
/// 3. Declare symbols within their containing modules
/// 4. Resolve names during semantic analysis
#[derive(Debug, Default)]
pub struct SymbolTable {
    /// Counter for generating unique module IDs
    next_module_id: u32,
    /// Counter for generating unique item IDs
    next_item_id: u32,
    /// Map of full module path → ModuleId for fast lookup
    modules_by_path: HashMap<Vec<String>, ModuleId>,
    /// All module information indexed by ModuleId
    modules: HashMap<ModuleId, ModuleInfo>,
    /// Map of (module, name) → ItemId for symbol resolution
    items: HashMap<(ModuleId, String), ItemId>,
    /// Information about each symbol indexed by ItemId
    symbols: HashMap<ItemId, Symbol>,
}

impl SymbolTable {
    /// Creates a new empty symbol table.
    ///
    /// The symbol table starts with no modules or symbols. The root module
    /// is typically created when the first module is declared.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a new module, returning its `ModuleId`.
    ///
    /// If the module already exists, returns the existing ID. Otherwise,
    /// creates a new module and ensures all parent modules exist in the
    /// hierarchy.
    ///
    /// # Arguments
    ///
    /// * `path` - Full path components (e.g., vec!["std", "io"])
    ///
    /// # Returns
    ///
    /// The `ModuleId` for the declared module (new or existing).
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut table = SymbolTable::new();
    /// let std_io = table.declare_module(vec!["std".to_string(), "io".to_string()]);
    /// ```
    pub fn declare_module(&mut self, path: Vec<String>) -> ModuleId {
        if let Some(id) = self.modules_by_path.get(&path) {
            return *id;
        }

        let id = ModuleId(self.next_module_id);
        self.next_module_id += 1;

        // Ensure parent module exists
        let parent = if path.len() > 1 {
            let parent_path = path[..path.len() - 1].to_vec();
            Some(self.declare_module(parent_path))
        } else {
            None
        };

        // Update parent's children list
        if let Some(parent_id) = parent {
            if let Some(parent_info) = self.modules.get_mut(&parent_id) {
                parent_info.children.push(id);
            }
        }

        let info = ModuleInfo {
            id,
            path: path.clone(),
            parent,
            children: Vec::new(),
            symbols: Vec::new(),
        };

        self.modules_by_path.insert(path, id);
        self.modules.insert(id, info);
        id
    }

    /// Declares a symbol within a module.
    ///
    /// Creates a new symbol entry and registers it in the symbol table.
    /// The symbol becomes available for resolution within its module
    /// and potentially from other modules (depending on visibility).
    ///
    /// # Arguments
    ///
    /// * `module` - The module containing this symbol
    /// * `name` - The declared name of the symbol
    /// * `kind` - The type of symbol being declared
    /// * `visibility` - Whether the symbol is public or private
    ///
    /// # Returns
    ///
    /// The unique `ItemId` assigned to the new symbol.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let mut table = SymbolTable::new();
    /// let module_id = table.declare_module(vec!["main".to_string()]);
    /// let symbol_id = table.declare_symbol(
    ///     module_id,
    ///     "hello".to_string(),
    ///     SymbolKind::Function,
    ///     Visibility::Public
    /// );
    /// ```
    pub fn declare_symbol(
        &mut self,
        module: ModuleId,
        name: String,
        kind: SymbolKind,
        visibility: Visibility,
    ) -> ItemId {
        let item_id = ItemId(self.next_item_id);
        self.next_item_id += 1;

        let symbol = Symbol::new(item_id, name.clone(), kind, module, visibility);

        self.symbols.insert(item_id, symbol);
        self.items.insert((module, name), item_id);

        // Add symbol to module's symbol list
        if let Some(module_info) = self.modules.get_mut(&module) {
            module_info.symbols.push(item_id);
        }

        item_id
    }

    /// Attempts to resolve a simple name within a specific module.
    ///
    /// Looks up a symbol by name within the given module. This is the
    /// fundamental operation for name resolution during compilation.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to search in
    /// * `name` - The symbol name to resolve
    ///
    /// # Returns
    ///
    /// `Some(ItemId)` if the symbol is found, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let symbol_id = table.resolve_in_module(module_id, "hello");
    /// ```
    pub fn resolve_in_module(&self, module: ModuleId, name: &str) -> Option<ItemId> {
        self.items.get(&(module, name.to_string())).copied()
    }

    /// Returns the ModuleId associated with a full path.
    ///
    /// Converts a module path (e.g., ["std", "io"]) to its corresponding
    /// ModuleId, if the module has been declared.
    ///
    /// # Arguments
    ///
    /// * `path` - Slice of path components
    ///
    /// # Returns
    ///
    /// `Some(ModuleId)` if the module exists, `None` otherwise.
    pub fn module_id(&self, path: &[String]) -> Option<ModuleId> {
        self.modules_by_path.get(path).copied()
    }

    /// Returns reference to a symbol by its ItemId.
    ///
    /// Retrieves the full symbol information for a given ItemId.
    /// This is used to access symbol metadata during compilation.
    ///
    /// # Arguments
    ///
    /// * `id` - The ItemId to look up
    ///
    /// # Returns
    ///
    /// `Some(&Symbol)` if the symbol exists, `None` otherwise.
    pub fn symbol(&self, id: ItemId) -> Option<&Symbol> {
        self.symbols.get(&id)
    }

    /// Returns reference to module information by ModuleId.
    ///
    /// Retrieves the full module information for a given ModuleId.
    /// This provides access to the module's hierarchy and contents.
    ///
    /// # Arguments
    ///
    /// * `id` - The ModuleId to look up
    ///
    /// # Returns
    ///
    /// `Some(&ModuleInfo)` if the module exists, `None` otherwise.
    pub fn module_info(&self, id: ModuleId) -> Option<&ModuleInfo> {
        self.modules.get(&id)
    }

    /// Returns the total number of declared symbols.
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    /// Returns the total number of declared modules.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    /// Returns all symbols in a given module.
    ///
    /// Retrieves all symbols that were declared directly in the specified module.
    /// This is useful for module-level analysis and code generation.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to query
    ///
    /// # Returns
    ///
    /// Vector of references to symbols in the module.
    pub fn symbols_in_module(&self, module: ModuleId) -> Vec<&Symbol> {
        if let Some(module_info) = self.modules.get(&module) {
            module_info
                .symbols
                .iter()
                .filter_map(|&id| self.symbols.get(&id))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Returns all public symbols in a given module.
    ///
    /// Filters symbols to only include those with public visibility.
    /// This is useful for determining what symbols are available for import.
    ///
    /// # Arguments
    ///
    /// * `module` - The module to query
    ///
    /// # Returns
    ///
    /// Vector of references to public symbols in the module.
    pub fn public_symbols_in_module(&self, module: ModuleId) -> Vec<&Symbol> {
        self.symbols_in_module(module)
            .into_iter()
            .filter(|symbol| symbol.is_public())
            .collect()
    }
}
