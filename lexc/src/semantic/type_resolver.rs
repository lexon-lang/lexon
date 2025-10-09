// lexc/src/semantic/type_resolver.rs
//
// Type resolution and checking for Lexon

use super::symbol_table::DeclId;

/// Unique identifier for a resolved type in the program
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeId(pub usize);

/// Represents a resolved type in the program
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedType {
    Primitive(PrimitiveType),
    Schema(DeclId), // Reference to a schema declaration
    Generic {
        base_type: Box<ResolvedType>,
        params: Vec<ResolvedType>,
    },
    Future(Box<ResolvedType>), // Future<T> type for async operations
    // Will add more complex types later (e.g., function types, union types)
    Unknown, // For errors or types that couldn't be resolved
}

/// Primitive types supported by Lexon
#[derive(Debug, Clone, PartialEq)]
pub enum PrimitiveType {
    Int,
    Float,
    String,
    Bool,
    // Will add more primitive types later as needed
}

impl std::fmt::Display for ResolvedType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedType::Primitive(p) => match p {
                PrimitiveType::Int => write!(f, "int"),
                PrimitiveType::Float => write!(f, "float"),
                PrimitiveType::String => write!(f, "string"),
                PrimitiveType::Bool => write!(f, "bool"),
            },
            ResolvedType::Schema(_) => write!(f, "schema"),
            ResolvedType::Generic { base_type, params } => {
                let base = base_type.to_string();
                let param_strs: Vec<String> = params.iter().map(|p| p.to_string()).collect();
                write!(f, "{}<{}>", base, param_strs.join(", "))
            }
            ResolvedType::Future(inner) => write!(f, "Future<{}>", inner),
            ResolvedType::Unknown => write!(f, "unknown"),
        }
    }
}

impl ResolvedType {
    /// Check if two types are compatible (for assignment, etc.)
    pub fn is_compatible_with(&self, other: &ResolvedType) -> bool {
        match (self, other) {
            // Same primitive types are compatible
            (ResolvedType::Primitive(a), ResolvedType::Primitive(b)) => a == b,

            // Same schema types are compatible
            (ResolvedType::Schema(a), ResolvedType::Schema(b)) => a == b,

            // Generic types are compatible if base type and all params are compatible
            (
                ResolvedType::Generic {
                    base_type: base_a,
                    params: params_a,
                },
                ResolvedType::Generic {
                    base_type: base_b,
                    params: params_b,
                },
            ) => {
                if !base_a.is_compatible_with(base_b) || params_a.len() != params_b.len() {
                    return false;
                }

                // Check each parameter type
                for (a, b) in params_a.iter().zip(params_b.iter()) {
                    if !a.is_compatible_with(b) {
                        return false;
                    }
                }

                true
            }

            // Future types are compatible if inner types are compatible
            (ResolvedType::Future(inner_a), ResolvedType::Future(inner_b)) => {
                inner_a.is_compatible_with(inner_b)
            }

            // Unknown type is compatible with anything (for error recovery)
            (ResolvedType::Unknown, _) | (_, ResolvedType::Unknown) => true,

            // Otherwise, not compatible
            _ => false,
        }
    }

    /// Check if this type is a Future<T>
    pub fn is_future(&self) -> bool {
        matches!(self, ResolvedType::Future(_))
    }

    /// Extract the inner type from a Future<T>, if applicable
    pub fn future_inner(&self) -> Option<&ResolvedType> {
        match self {
            ResolvedType::Future(inner) => Some(inner),
            _ => None,
        }
    }
}
