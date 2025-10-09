// Only using HirNode here; remove unused import
use crate::hir::HirNode;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum LintWarning {
    MissingAwait {
        function_name: String,
        line: usize,
        message: String,
    },
    BlockingIoInAsync {
        operation: String,
        line: usize,
        message: String,
    },
    AsyncFunctionNotAwaited {
        function_name: String,
        line: usize,
        message: String,
    },
    SyncCallInAsyncContext {
        function_name: String,
        line: usize,
        message: String,
    },
}

#[derive(Debug)]
pub struct LintResult {
    pub warnings: Vec<LintWarning>,
    pub errors: Vec<String>,
}

pub struct Linter {
    warnings: Vec<LintWarning>,
    errors: Vec<String>,
    async_functions: HashMap<String, bool>,
    current_context_is_async: bool,
    current_line: usize,
}

impl Linter {
    pub fn new() -> Self {
        Self {
            warnings: Vec::new(),
            errors: Vec::new(),
            async_functions: HashMap::new(),
            current_context_is_async: false,
            current_line: 1,
        }
    }

    pub fn lint_hir(&mut self, nodes: &[HirNode]) -> LintResult {
        // First pass: collect all async functions
        self.collect_async_functions(nodes);

        // Second pass: lint the code
        for node in nodes {
            self.lint_node(node);
        }

        LintResult {
            warnings: self.warnings.clone(),
            errors: self.errors.clone(),
        }
    }

    fn collect_async_functions(&mut self, nodes: &[HirNode]) {
        for node in nodes {
            if let HirNode::FunctionDefinition(func_def) = node {
                self.async_functions
                    .insert(func_def.name.clone(), func_def.is_async);
            }
        }
    }

    fn lint_node(&mut self, node: &HirNode) {
        match node {
            HirNode::FunctionDefinition(func_def) => {
                let previous_context = self.current_context_is_async;
                self.current_context_is_async = func_def.is_async;

                // Lint function body
                for body_node in &func_def.body {
                    self.lint_node(body_node);
                }

                self.current_context_is_async = previous_context;
            }
            HirNode::FunctionCall(func_call) => {
                self.lint_function_call(&func_call.function, &func_call.args);

                // Lint arguments
                for arg in &func_call.args {
                    self.lint_node(arg);
                }
            }
            HirNode::Ask(_) => {
                if self.current_context_is_async {
                    self.warnings.push(LintWarning::MissingAwait {
                        function_name: "ask".to_string(),
                        line: self.current_line,
                        message:
                            "Consider using 'await ask' in async context for better performance"
                                .to_string(),
                    });
                }
            }
            HirNode::AskSafe(_) => {
                if self.current_context_is_async {
                    self.warnings.push(LintWarning::MissingAwait {
                        function_name: "ask_safe".to_string(),
                        line: self.current_line,
                        message: "Consider using 'await ask_safe' in async context for better performance".to_string(),
                    });
                }
            }
            HirNode::Assignment(assignment) => {
                self.lint_node(&assignment.right);
            }
            HirNode::If(if_stmt) => {
                self.lint_node(&if_stmt.condition);
                for node in &if_stmt.then_body {
                    self.lint_node(node);
                }
                if let Some(else_body) = &if_stmt.else_body {
                    for node in else_body {
                        self.lint_node(node);
                    }
                }
            }
            HirNode::While(while_stmt) => {
                self.lint_node(&while_stmt.condition);
                for node in &while_stmt.body {
                    self.lint_node(node);
                }
            }
            HirNode::ForIn(for_stmt) => {
                self.lint_node(&for_stmt.iterable);
                for node in &for_stmt.body {
                    self.lint_node(node);
                }
            }
            HirNode::Binary(binary_expr) => {
                self.lint_node(&binary_expr.left);
                self.lint_node(&binary_expr.right);
            }
            HirNode::VariableDeclaration(var_decl) => {
                self.lint_node(&var_decl.value);
            }
            HirNode::Await(await_expr) => {
                // This is good - proper await usage
                self.lint_node(&await_expr.expression);
            }
            HirNode::Match(match_expr) => {
                self.lint_node(&match_expr.value);
                for arm in &match_expr.arms {
                    self.lint_node(&arm.pattern);
                    for node in &arm.body {
                        self.lint_node(node);
                    }
                }
            }
            HirNode::Return(return_stmt) => {
                if let Some(expr) = &return_stmt.expression {
                    self.lint_node(expr);
                }
            }
            HirNode::TypeOf(typeof_expr) => {
                self.lint_node(&typeof_expr.argument);
            }
            HirNode::MethodCall(method_call) => {
                for arg in &method_call.args {
                    self.lint_node(arg);
                }
            }
            // Data operations
            HirNode::DataLoad(_)
            | HirNode::DataFilter(_)
            | HirNode::DataSelect(_)
            | HirNode::DataTake(_)
            | HirNode::DataExport(_) => {
                if self.current_context_is_async {
                    self.warnings.push(LintWarning::BlockingIoInAsync {
                        operation: "data_operation".to_string(),
                        line: self.current_line,
                        message: "Data operations may be blocking in async context".to_string(),
                    });
                }
            }
            // Memory operations
            HirNode::MemoryLoad(_) | HirNode::MemoryStore(_) => {
                // Memory operations are typically async-safe
            }
            // Literals and identifiers don't need linting
            HirNode::Literal(_) | HirNode::Identifier(_) | HirNode::Break | HirNode::Continue => {}
            // Module-level constructs
            HirNode::SchemaDefinition(_)
            | HirNode::ModuleDeclaration(_)
            | HirNode::ImportDeclaration(_)
            | HirNode::TraitDefinition(_)
            | HirNode::ImplBlock(_) => {}
        }
    }

    fn lint_function_call(&mut self, name: &str, _args: &[HirNode]) {
        // Check for blocking I/O operations in async context
        if self.current_context_is_async {
            match name {
                "read_file" | "write_file" | "load_file" | "save_file" => {
                    self.warnings.push(LintWarning::BlockingIoInAsync {
                        operation: name.to_string(),
                        line: self.current_line,
                        message: format!("Blocking I/O operation '{}' in async context. Consider using async version.", name),
                    });
                }
                "execute" => {
                    self.warnings.push(LintWarning::BlockingIoInAsync {
                        operation: name.to_string(),
                        line: self.current_line,
                        message: "Blocking command execution in async context. Consider using async version.".to_string(),
                    });
                }
                "data_load" | "data_filter" | "data_select" | "data_take" | "data_export" => {
                    self.warnings.push(LintWarning::BlockingIoInAsync {
                        operation: name.to_string(),
                        line: self.current_line,
                        message: format!(
                            "Data operation '{}' may be blocking in async context",
                            name
                        ),
                    });
                }
                _ => {}
            }
        }

        // Check for async functions called without await
        if let Some(&is_async) = self.async_functions.get(name) {
            if is_async && self.current_context_is_async {
                self.warnings.push(LintWarning::AsyncFunctionNotAwaited {
                    function_name: name.to_string(),
                    line: self.current_line,
                    message: format!(
                        "Async function '{}' should be awaited in async context",
                        name
                    ),
                });
            }
        }

        // Check for known async operations
        match name {
            "ask" | "ask_safe" | "ask_parallel" | "ask_ensemble" => {
                if self.current_context_is_async {
                    self.warnings.push(LintWarning::MissingAwait {
                        function_name: name.to_string(),
                        line: self.current_line,
                        message: format!("LLM operation '{}' should use 'await' in async context for optimal performance", name),
                    });
                }
            }
            _ => {}
        }
    }

    pub fn print_warnings(&self) {
        if self.warnings.is_empty() && self.errors.is_empty() {
            println!("✅ No linting issues found");
            return;
        }

        if !self.warnings.is_empty() {
            println!("⚠️  Linting Warnings:");
            for warning in &self.warnings {
                match warning {
                    LintWarning::MissingAwait {
                        function_name,
                        line,
                        message,
                    } => {
                        println!(
                            "  Warning: Missing await for '{}' at line {}",
                            function_name, line
                        );
                        println!("    {}", message);
                    }
                    LintWarning::BlockingIoInAsync {
                        operation,
                        line,
                        message,
                    } => {
                        println!("  Warning: Blocking I/O '{}' at line {}", operation, line);
                        println!("    {}", message);
                    }
                    LintWarning::AsyncFunctionNotAwaited {
                        function_name,
                        line,
                        message,
                    } => {
                        println!(
                            "  Warning: Async function '{}' not awaited at line {}",
                            function_name, line
                        );
                        println!("    {}", message);
                    }
                    LintWarning::SyncCallInAsyncContext {
                        function_name,
                        line,
                        message,
                    } => {
                        println!(
                            "  Warning: Sync call '{}' in async context at line {}",
                            function_name, line
                        );
                        println!("    {}", message);
                    }
                }
            }
        }

        if !self.errors.is_empty() {
            println!("❌ Linting Errors:");
            for error in &self.errors {
                println!("  Error: {}", error);
            }
        }
    }
}
impl Default for Linter {
    fn default() -> Self {
        Self::new()
    }
}

// Convenience function for linting HIR
pub fn lint_hir(nodes: &[HirNode]) -> LintResult {
    let mut linter = Linter::new();
    linter.lint_hir(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::*;

    #[test]
    fn test_missing_await_detection() {
        let mut linter = Linter::new();
        linter.current_context_is_async = true;

        // Create a simple ask node
        let ask_node = HirNode::Ask(Box::new(HirAskExpression {
            system_prompt: None,
            user_prompt: Some("test".to_string()),
            output_schema_name: None,
            attributes: vec![],
        }));

        linter.lint_node(&ask_node);

        assert_eq!(linter.warnings.len(), 1);
        if let LintWarning::MissingAwait { function_name, .. } = &linter.warnings[0] {
            assert_eq!(function_name, "ask");
        }
    }

    #[test]
    fn test_blocking_io_detection() {
        let mut linter = Linter::new();
        linter.current_context_is_async = true;

        linter.lint_function_call("read_file", &[]);

        assert_eq!(linter.warnings.len(), 1);
        if let LintWarning::BlockingIoInAsync { operation, .. } = &linter.warnings[0] {
            assert_eq!(operation, "read_file");
        }
    }

    #[test]
    fn test_no_warnings_in_sync_context() {
        let mut linter = Linter::new();
        linter.current_context_is_async = false;

        linter.lint_function_call("read_file", &[]);

        assert_eq!(linter.warnings.len(), 0);
    }
}
