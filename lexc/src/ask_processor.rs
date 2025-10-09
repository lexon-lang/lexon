// lexc/src/ask_processor.rs

use tree_sitter::Node;

#[derive(Debug, Clone, PartialEq)]
pub struct AskAttribute {
    pub name: String,
    pub value: Option<String>, // Value is optional
}

#[derive(Debug, Clone, PartialEq)]
pub struct AskExpressionData {
    pub system_prompt: Option<String>,
    pub user_prompt: Option<String>,
    pub schema_name: Option<String>,
    pub attributes: Vec<AskAttribute>,
}

// 🛡️ Structure for ask_safe expressions with anti-hallucination validation
#[derive(Debug, Clone, PartialEq)]
pub struct AskSafeExpressionData {
    pub system_prompt: Option<String>,
    pub user_prompt: Option<String>,
    pub schema_name: Option<String>,
    pub attributes: Vec<AskAttribute>,
    // Specific validation parameters
    pub validation_strategy: Option<String>,
    pub confidence_threshold: Option<f64>,
    pub max_attempts: Option<u32>,
    pub cross_reference_models: Vec<String>,
    pub use_fact_checking: Option<bool>,
}

// Helper functions (private)
fn get_node_text<'a>(node: Node<'a>, source_code: &'a str) -> Option<String> {
    source_code
        .get(node.start_byte()..node.end_byte())
        .map(String::from)
}

/// Extracts the value of a literal node as String.
/// For strings, removes quotes. For other literals (numbers, etc.), returns their textual representation.
fn get_literal_node_value_as_string<'a>(
    literal_node: Node<'a>,
    source_code: &'a str,
) -> Option<String> {
    // Debug: Print the type of node being processed
    // println!("get_literal_node_value_as_string: processing node kind: {}, text: \"{}\"", literal_node.kind(), get_node_text(literal_node, source_code).unwrap_or_default());

    match literal_node.kind() {
        "string_literal" => {
            let text = get_node_text(literal_node, source_code)?;
            if text.starts_with("\"") && text.ends_with("\"") && text.len() >= 2 {
                Some(text[1..text.len() - 1].to_string())
            } else {
                eprintln!("Warning: malformed string_literal found: {}", text);
                None
            }
        }
        "multiline_string_literal" => {
            let text = get_node_text(literal_node, source_code)?;
            eprintln!("DEBUG multiline_string_literal raw text: [{}]", text); // DEBUG
                                                                              // The regex for multiline_string_literal in grammar.js is: token(seq('"""', repeat(choice(/[^\"]+/, seq('"', /"?[^"]/))), '"""'))
                                                                              // This implies that there shouldn't be escape sequences like \n directly in the node text if the grammar handles them.
                                                                              // The unescaping of \n to real \n should occur here.
            if text.starts_with("\"\"\"") && text.ends_with("\"\"\"") && text.len() >= 6 {
                let inner_text = &text[3..text.len() - 3];
                eprintln!(
                    "DEBUG multiline_string_literal inner text: [{}]",
                    inner_text
                ); // DEBUG
                   // Step 1: Replace literal \n with actual newlines (in case explicit escapes exist)
                let replaced = inner_text.replace("\\n", "\n");
                // Step 2: Trim leading/trailing whitespace and remove common indentation
                let trimmed = replaced.trim_matches('\n'); // remove extreme newlines
                                                           // Detect common indentation (spaces or tabs) in all non-empty lines
                let lines: Vec<&str> = trimmed.lines().collect();
                let common_prefix_len = lines
                    .iter()
                    .filter(|l| !l.trim().is_empty())
                    .map(|l| l.chars().take_while(|c| *c == ' ' || *c == '\t').count())
                    .min()
                    .unwrap_or(0);

                let cleaned_lines: Vec<String> = lines
                    .iter()
                    .map(|l| {
                        if l.len() >= common_prefix_len {
                            l[common_prefix_len..].to_string()
                        } else {
                            l.to_string()
                        }
                    })
                    .collect();

                Some(cleaned_lines.join("\n").trim_end().to_string())
            } else {
                eprintln!(
                    "Warning: malformed multiline_string_literal found: {}",
                    text
                );
                None
            }
        }
        "number_literal" | "float_literal" => {
            // Assuming the grammar defines them
            get_node_text(literal_node, source_code)
        }
        "integer_literal" => {
            // Handle integer literals by converting to string
            get_node_text(literal_node, source_code)
        }
        "boolean_literal" => {
            // Handle boolean literals by converting to string
            get_node_text(literal_node, source_code)
        }
        "binary_expression" => {
            // Handle binary expressions by getting their text representation
            // This is used when expressions like "Hello " + variable are in prompts
            get_node_text(literal_node, source_code)
        }
        // Add more literal types here if necessary (null_literal)
        _ => {
            eprintln!(
                "Warning: Literal type not supported for direct extraction as string: {}",
                literal_node.kind()
            );
            get_node_text(literal_node, source_code) // Fallback to raw text if not a known string
        }
    }
}

/// Parses an `ask_expression` node from the Tree-sitter tree and extracts its information.
pub fn parse_ask_expression(node: Node, source_code: &str) -> Option<AskExpressionData> {
    if node.kind() != "ask_expression" {
        eprintln!(
            "Error: Node provided to parse_ask_expression is not an ask_expression. Type: {}",
            node.kind()
        );
        return None;
    }

    println!("🔍 DEBUG ASK_PROCESSOR: Starting parse_ask_expression");
    println!(
        "🔍 DEBUG ASK_PROCESSOR: Node structure: {:?}",
        node.to_sexp()
    );

    let mut system_prompt: Option<String> = None;
    let mut user_prompt: Option<String> = None;
    let mut schema_name: Option<String> = None;
    let mut attributes: Vec<AskAttribute> = Vec::new();

    // Check if this is function syntax: ask("prompt") or ask("prompt", "model")
    let prompt_field = node.child_by_field_name("prompt");
    if let Some(prompt_node) = prompt_field {
        // This is function syntax: ask("prompt") or ask("prompt", "model")
        user_prompt = get_literal_node_value_as_string(prompt_node, source_code);
        println!(
            "🔍 DEBUG ASK_PROCESSOR: Function syntax detected, user_prompt: {:?}",
            user_prompt
        );

        // Check for optional model field
        if let Some(model_node) = node.child_by_field_name("model") {
            if let Some(model_str) = get_literal_node_value_as_string(model_node, source_code) {
                println!(
                    "🔍 DEBUG ASK_PROCESSOR: Model field detected: {:?}",
                    model_str
                );
                attributes.push(AskAttribute {
                    name: "model".to_string(),
                    value: Some(model_str),
                });
            }
        }

        return Some(AskExpressionData {
            system_prompt,
            user_prompt,
            schema_name,
            attributes,
        });
    }

    // Otherwise, parse as block syntax: ask { ... }
    let mut cursor = node.walk();
    for child_node in node.children(&mut cursor) {
        if !child_node.is_named() {
            continue;
        }

        println!(
            "🔍 DEBUG ASK_PROCESSOR: Processing child node: kind={}, text={:?}",
            child_node.kind(),
            get_node_text(child_node, source_code)
        );

        match child_node.kind() {
            "attribute" => {
                println!("🔍 DEBUG ASK_PROCESSOR: Found attribute node");
                let name_node = child_node.child_by_field_name("name");

                if let Some(name_n) = name_node {
                    let name_text = get_node_text(name_n, source_code).unwrap_or_else(|| {
                        eprintln!("Warning: Unable to get text for attribute name.");
                        String::new()
                    });

                    let mut attr_value: Option<String> = None;
                    if let Some(args_node) = child_node.child_by_field_name("arguments") {
                        if let Some(attribute_arg_node) = args_node.named_child(0) {
                            // attribute_arg_node is the 'attribute_arg' node. Its child is the actual literal.
                            if let Some(actual_literal_node) = attribute_arg_node.named_child(0) {
                                attr_value = get_literal_node_value_as_string(
                                    actual_literal_node,
                                    source_code,
                                );
                            } else {
                                // If attribute_arg does not have a named child, it could be an error or an unexpected node type.
                                // The rule is attribute_arg: $ => choice($._literal, $.identifier),
                                // and _literal and identifier are named. So this should not happen with a valid grammar.
                                eprintln!("Warning: The attribute_arg node does not have a named literal child. Node attribute_arg: {:?}", attribute_arg_node.to_sexp());
                            }
                        } else {
                            eprintln!("Warning: No named child (attribute_arg) found inside attribute_args for attribute '{}'. args structure: {:?}", name_text, args_node.to_sexp());
                        }
                    }
                    attributes.push(AskAttribute {
                        name: name_text,
                        value: attr_value,
                    });
                } else {
                    eprintln!(
                        "Warning: Malformed attribute found (missing 'name' node). Node: {:?}",
                        child_node.to_sexp()
                    );
                }
            }
            "ask_kv_pair" => {
                println!("🔍 DEBUG ASK_PROCESSOR: Found ask_kv_pair node");
                let key_node = child_node.child_by_field_name("key");
                let value_node = child_node.child_by_field_name("value");

                if let (Some(key_n), Some(value_n)) = (key_node, value_node) {
                    let key_text = get_node_text(key_n, source_code).unwrap_or_default();
                    println!(
                        "🔍 DEBUG ASK_PROCESSOR: ask_kv_pair key={}, value_kind={}",
                        key_text,
                        value_n.kind()
                    );
                    match key_text.as_str() {
                        "system" => {
                            system_prompt = get_literal_node_value_as_string(value_n, source_code);
                        }
                        "user" => {
                            user_prompt = get_literal_node_value_as_string(value_n, source_code);
                        }
                        "schema" => {
                            // The schema value is an identifier, not a string literal
                            if value_n.kind() == "identifier" {
                                schema_name = get_node_text(value_n, source_code);
                            } else {
                                eprintln!("Warning: The value for the 'schema' key is not an identifier: {}", value_n.kind());
                            }
                        }
                        "model" | "temperature" | "max_tokens" => {
                            // Add these keys as attributes
                            if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                println!(
                                    "🔍 DEBUG ASK_PROCESSOR: Adding {} = {} to attributes",
                                    key_text, value_str
                                );
                                attributes.push(AskAttribute {
                                    name: key_text.clone(),
                                    value: Some(value_str),
                                });
                            }
                        }
                        // 🛡️ NEW ATTRIBUTES FOR ANTI-HALLUCINATION SYSTEM
                        "validation"
                        | "confidence_threshold"
                        | "max_attempts"
                        | "use_fact_checking" => {
                            if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                println!(
                                    "🛡️ DEBUG ASK_PROCESSOR: Adding validation attribute {} = {}",
                                    key_text, value_str
                                );
                                attributes.push(AskAttribute {
                                    name: key_text.clone(),
                                    value: Some(value_str),
                                });
                            }
                        }
                        "cross_reference_models" => {
                            // Handle array of models for cross-validation
                            if value_n.kind() == "array_literal" {
                                let mut models = Vec::new();
                                let mut cursor = value_n.walk();
                                for array_child in value_n.children(&mut cursor) {
                                    if array_child.is_named()
                                        && array_child.kind() == "string_literal"
                                    {
                                        if let Some(model_name) = get_literal_node_value_as_string(
                                            array_child,
                                            source_code,
                                        ) {
                                            models.push(model_name);
                                        }
                                    }
                                }
                                let models_str = models.join(",");
                                println!(
                                    "🛡️ DEBUG ASK_PROCESSOR: Adding cross_reference_models = {}",
                                    models_str
                                );
                                attributes.push(AskAttribute {
                                    name: "cross_reference_models".to_string(),
                                    value: Some(models_str),
                                });
                            } else if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                attributes.push(AskAttribute {
                                    name: key_text.clone(),
                                    value: Some(value_str),
                                });
                            }
                        }
                        _ => {
                            eprintln!("Warning: Unknown key in ask_kv_pair: {}", key_text);
                        }
                    }
                } else {
                    eprintln!("Warning: Malformed ask_kv_pair found (missing key or value).");
                }
            }
            _ => {
                // Other named child nodes that are not attribute nor ask_kv_pair (if any)
                println!(
                    "🔍 DEBUG ASK_PROCESSOR: Skipping unknown child node: {}",
                    child_node.kind()
                );
            }
        }
    }

    println!("🔍 DEBUG ASK_PROCESSOR: Final attributes: {:?}", attributes);

    Some(AskExpressionData {
        system_prompt,
        user_prompt,
        schema_name,
        attributes,
    })
}

// 🛡️ Function to parse ask_safe expressions with anti-hallucination validation
pub fn parse_ask_safe_expression(node: Node, source_code: &str) -> Option<AskSafeExpressionData> {
    if node.kind() != "ask_safe_expression" {
        eprintln!("Error: Node provided to parse_ask_safe_expression is not an ask_safe_expression. Type: {}", node.kind());
        return None;
    }

    println!("🛡️ DEBUG ASK_SAFE_PROCESSOR: Starting parse_ask_safe_expression");
    println!(
        "🛡️ DEBUG ASK_SAFE_PROCESSOR: Node structure: {:?}",
        node.to_sexp()
    );

    let mut system_prompt: Option<String> = None;
    let mut user_prompt: Option<String> = None;
    let mut schema_name: Option<String> = None;
    let mut attributes: Vec<AskAttribute> = Vec::new();

    // Specific validation parameters
    let mut validation_strategy: Option<String> = None;
    let mut confidence_threshold: Option<f64> = None;
    let mut max_attempts: Option<u32> = None;
    let mut cross_reference_models: Vec<String> = Vec::new();
    let mut use_fact_checking: Option<bool> = None;

    // Check if this is function syntax: ask_safe("prompt", validation: "basic")
    let prompt_field = node.child_by_field_name("prompt");
    if let Some(prompt_node) = prompt_field {
        // This is function syntax: ask_safe("prompt", ...)
        user_prompt = get_literal_node_value_as_string(prompt_node, source_code);

        // Parse validation parameters
        let mut cursor = node.walk();
        for child_node in node.children(&mut cursor) {
            if child_node.kind() == "ask_safe_parameter" {
                let name_node = child_node.child_by_field_name("name");
                let value_node = child_node.child_by_field_name("value");

                if let (Some(name_n), Some(value_n)) = (name_node, value_node) {
                    let param_name = get_node_text(name_n, source_code).unwrap_or_default();
                    match param_name.as_str() {
                        "validation" => {
                            validation_strategy =
                                get_literal_node_value_as_string(value_n, source_code);
                        }
                        "confidence_threshold" => {
                            if let Some(val_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                confidence_threshold = val_str.parse::<f64>().ok();
                            }
                        }
                        "max_attempts" => {
                            if let Some(val_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                max_attempts = val_str.parse::<u32>().ok();
                            }
                        }
                        "use_fact_checking" => {
                            if let Some(val_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                use_fact_checking = val_str.parse::<bool>().ok();
                            }
                        }
                        "cross_reference_models" => {
                            // Handle array of models
                            if value_n.kind() == "array_literal" {
                                let mut models = Vec::new();
                                let mut cursor = value_n.walk();
                                for array_child in value_n.children(&mut cursor) {
                                    if array_child.is_named()
                                        && array_child.kind() == "string_literal"
                                    {
                                        if let Some(model_name) = get_literal_node_value_as_string(
                                            array_child,
                                            source_code,
                                        ) {
                                            models.push(model_name);
                                        }
                                    }
                                }
                                cross_reference_models = models;
                            }
                        }
                        _ => {
                            // Add unknown parameters as attributes
                            if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                attributes.push(AskAttribute {
                                    name: param_name,
                                    value: Some(value_str),
                                });
                            }
                        }
                    }
                }
            }
        }

        println!(
            "🛡️ DEBUG ASK_SAFE_PROCESSOR: Function syntax detected, user_prompt: {:?}",
            user_prompt
        );
        return Some(AskSafeExpressionData {
            system_prompt,
            user_prompt,
            schema_name,
            attributes,
            validation_strategy,
            confidence_threshold,
            max_attempts,
            cross_reference_models,
            use_fact_checking,
        });
    }

    // Otherwise, parse as block syntax: ask_safe { ... }
    let mut cursor = node.walk();
    for child_node in node.children(&mut cursor) {
        if !child_node.is_named() {
            continue;
        }

        println!(
            "🛡️ DEBUG ASK_SAFE_PROCESSOR: Processing child node: kind={}, text={:?}",
            child_node.kind(),
            get_node_text(child_node, source_code)
        );

        match child_node.kind() {
            "attribute" => {
                println!("🛡️ DEBUG ASK_SAFE_PROCESSOR: Found attribute node");
                let name_node = child_node.child_by_field_name("name");

                if let Some(name_n) = name_node {
                    let name_text = get_node_text(name_n, source_code).unwrap_or_else(|| {
                        eprintln!("Warning: Unable to get text for attribute name.");
                        String::new()
                    });

                    let mut attr_value: Option<String> = None;
                    if let Some(args_node) = child_node.child_by_field_name("arguments") {
                        if let Some(attribute_arg_node) = args_node.named_child(0) {
                            if let Some(actual_literal_node) = attribute_arg_node.named_child(0) {
                                attr_value = get_literal_node_value_as_string(
                                    actual_literal_node,
                                    source_code,
                                );
                            } else {
                                eprintln!("Warning: The attribute_arg node does not have a named literal child. Node attribute_arg: {:?}", attribute_arg_node.to_sexp());
                            }
                        } else {
                            eprintln!("Warning: No named child (attribute_arg) found inside attribute_args for attribute '{}'. args structure: {:?}", name_text, args_node.to_sexp());
                        }
                    }
                    attributes.push(AskAttribute {
                        name: name_text,
                        value: attr_value,
                    });
                } else {
                    eprintln!(
                        "Warning: Malformed attribute found (missing 'name' node). Node: {:?}",
                        child_node.to_sexp()
                    );
                }
            }
            "ask_safe_kv_pair" => {
                println!("🛡️ DEBUG ASK_SAFE_PROCESSOR: Found ask_safe_kv_pair node");
                let key_node = child_node.child_by_field_name("key");
                let value_node = child_node.child_by_field_name("value");

                if let (Some(key_n), Some(value_n)) = (key_node, value_node) {
                    let key_text = get_node_text(key_n, source_code).unwrap_or_default();
                    println!(
                        "🛡️ DEBUG ASK_SAFE_PROCESSOR: ask_safe_kv_pair key={}, value_kind={}",
                        key_text,
                        value_n.kind()
                    );

                    match key_text.as_str() {
                        "system" => {
                            system_prompt = get_literal_node_value_as_string(value_n, source_code);
                        }
                        "user" => {
                            user_prompt = get_literal_node_value_as_string(value_n, source_code);
                        }
                        "schema" => {
                            if value_n.kind() == "identifier" {
                                schema_name = get_node_text(value_n, source_code);
                            } else {
                                eprintln!("Warning: The value for the 'schema' key is not an identifier: {}", value_n.kind());
                            }
                        }
                        "model" | "temperature" | "max_tokens" => {
                            if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                println!(
                                    "🛡️ DEBUG ASK_SAFE_PROCESSOR: Adding {} = {} to attributes",
                                    key_text, value_str
                                );
                                attributes.push(AskAttribute {
                                    name: key_text.clone(),
                                    value: Some(value_str),
                                });
                            }
                        }
                        // 🛡️ NEW ATTRIBUTES FOR ANTI-HALLUCINATION SYSTEM
                        "validation" => {
                            validation_strategy =
                                get_literal_node_value_as_string(value_n, source_code);
                        }
                        "confidence_threshold" => {
                            if let Some(val_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                confidence_threshold = val_str.parse::<f64>().ok();
                            }
                        }
                        "max_attempts" => {
                            if let Some(val_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                max_attempts = val_str.parse::<u32>().ok();
                            }
                        }
                        "use_fact_checking" => {
                            if let Some(val_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                use_fact_checking = val_str.parse::<bool>().ok();
                            }
                        }
                        "cross_reference_models" => {
                            // Handle array of models for cross-validation
                            if value_n.kind() == "array_literal" {
                                let mut models = Vec::new();
                                let mut cursor = value_n.walk();
                                for array_child in value_n.children(&mut cursor) {
                                    if array_child.is_named()
                                        && array_child.kind() == "string_literal"
                                    {
                                        if let Some(model_name) = get_literal_node_value_as_string(
                                            array_child,
                                            source_code,
                                        ) {
                                            models.push(model_name);
                                        }
                                    }
                                }
                                cross_reference_models = models;
                                println!("🛡️ DEBUG ASK_SAFE_PROCESSOR: Added cross_reference_models: {:?}", cross_reference_models);
                            } else if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                // Single model as string
                                cross_reference_models = vec![value_str];
                            }
                        }
                        _ => {
                            eprintln!("Warning: Unknown key in ask_safe_kv_pair: {}", key_text);
                            // Add as attribute for flexibility
                            if let Some(value_str) =
                                get_literal_node_value_as_string(value_n, source_code)
                            {
                                attributes.push(AskAttribute {
                                    name: key_text.clone(),
                                    value: Some(value_str),
                                });
                            }
                        }
                    }
                } else {
                    eprintln!("Warning: Malformed ask_safe_kv_pair found (missing key or value).");
                }
            }
            _ => {
                println!(
                    "🛡️ DEBUG ASK_SAFE_PROCESSOR: Skipping unknown child node: {}",
                    child_node.kind()
                );
            }
        }
    }

    println!(
        "🛡️ DEBUG ASK_SAFE_PROCESSOR: Final validation_strategy: {:?}",
        validation_strategy
    );
    println!(
        "🛡️ DEBUG ASK_SAFE_PROCESSOR: Final confidence_threshold: {:?}",
        confidence_threshold
    );
    println!(
        "🛡️ DEBUG ASK_SAFE_PROCESSOR: Final cross_reference_models: {:?}",
        cross_reference_models
    );
    println!(
        "🛡️ DEBUG ASK_SAFE_PROCESSOR: Final attributes: {:?}",
        attributes
    );

    Some(AskSafeExpressionData {
        system_prompt,
        user_prompt,
        schema_name,
        attributes,
        validation_strategy,
        confidence_threshold,
        max_attempts,
        cross_reference_models,
        use_fact_checking,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirAskExpression;
    use tree_sitter::Parser;

    // Necessary for tests to find tree_sitter_lexon
    // Ensure build.rs is configured to compile the grammar.
    extern "C" {
        fn tree_sitter_lexon() -> tree_sitter::Language;
    }

    fn setup_parser() -> Parser {
        let mut parser = Parser::new();
        let language = unsafe { tree_sitter_lexon() }; // Use the extern function
        parser
            .set_language(&language)
            .expect("Error loading Lexon language");
        parser
    }

    #[test]
    fn test_parse_simple_ask_expression() {
        let mut parser = setup_parser();
        let source_code = r#"
let my_ask_result: MySchema = ask {
    system: "You are an assistant.";
    user: "Generate an instance.";
    schema: MySchema;
};
        "#;
        let tree = parser
            .parse(source_code, None)
            .expect("Failed to parse source code");
        let root_node = tree.root_node();

        let var_decl_node = root_node
            .child(0)
            .expect("No child node for root (expected variable_declaration)");
        assert_eq!(var_decl_node.kind(), "variable_declaration");

        let expression_node = var_decl_node
            .child_by_field_name("value")
            .expect("No value field for variable_declaration");
        assert_eq!(
            expression_node.kind(),
            "ask_expression",
            "Expected ask_expression, got: {}",
            expression_node.kind()
        );

        let ask_data_opt = parse_ask_expression(expression_node, source_code);

        assert!(ask_data_opt.is_some(), "parse_ask_expression returned None");
        let data = ask_data_opt.unwrap();

        // Convert to HIR & assert HIR
        let hir_ask: HirAskExpression = data.clone().into();
        assert_eq!(
            hir_ask.system_prompt,
            Some("You are an assistant.".to_string())
        );
        assert_eq!(
            hir_ask.user_prompt,
            Some("Generate an instance.".to_string())
        );
        assert_eq!(hir_ask.output_schema_name, Some("MySchema".to_string()));
        assert!(
            hir_ask.attributes.is_empty(),
            "There should be no attributes in this simple example."
        );

        // Original assertions on 'data' can also be kept if desired, or removed if HIR covers all.
        // For now, let's assume HIR assertions are sufficient and the original assert_eq!(data.system_prompt...); for this test was simplified earlier.
    }

    #[test]
    fn test_parse_ask_with_attributes() {
        let mut parser = setup_parser();
        let source_code = r#"
let my_ask_result: MySchema = ask @model("gpt-4") {
    user: "Hello";
};
        "#;
        let tree = parser
            .parse(source_code, None)
            .expect("Failed to parse source code");
        let root_node = tree.root_node();
        let var_decl_node = root_node.child(0).unwrap();

        println!(
            "S-expression for var_decl_node (attributes test): {:?}",
            var_decl_node.to_sexp()
        );

        let ask_expr_node_opt = var_decl_node.child_by_field_name("value");
        assert!(
            ask_expr_node_opt.is_some(),
            "Did not find 'value' field in var_decl_node for test_parse_ask_with_attributes"
        );
        let ask_expr_node = ask_expr_node_opt.unwrap();

        println!(
            "S-expression for node passed to parse_ask_expression (attributes test): {:?}",
            ask_expr_node.to_sexp()
        );

        let data_opt = parse_ask_expression(ask_expr_node, source_code);
        assert!(
            data_opt.is_some(),
            "parse_ask_expression returned None for test_parse_ask_with_attributes"
        );
        let data = data_opt.unwrap();

        // Convert to HIR
        let hir_ask: HirAskExpression = data.clone().into();

        assert_eq!(hir_ask.user_prompt, Some("Hello".to_string()));
        assert_eq!(hir_ask.attributes.len(), 1);
        assert_eq!(hir_ask.attributes[0].name, "model");
        assert_eq!(hir_ask.attributes[0].value, Some("gpt-4".to_string()));
        assert!(hir_ask.system_prompt.is_none());
        assert!(hir_ask.output_schema_name.is_none());
    }

    #[test]
    fn test_parse_ask_multiline_strings() {
        let mut parser = setup_parser();
        let source_code = r#"
let complex_prompt: String = ask {
    system: """
    Multi-line
    system.
    """;
    user: """User as well.""";
};
        "#;
        let tree = parser.parse(source_code, None);
        assert!(
            tree.is_some(),
            "Initial code parse failed for multiline_strings test"
        );
        let unwrapped_tree = tree.unwrap();
        let root_node = unwrapped_tree.root_node();

        assert!(
            root_node.child_count() > 0,
            "Root node has no children in multiline_strings test"
        );
        let var_decl_node_opt = root_node.child(0);
        assert!(
            var_decl_node_opt.is_some(),
            "First child (variable_declaration) not found in multiline_strings test"
        );
        let var_decl_node = var_decl_node_opt.unwrap();
        println!(
            "S-expression for var_decl_node (multiline test): {:?}",
            var_decl_node.to_sexp()
        );

        let ask_expr_node_opt = var_decl_node.child_by_field_name("value");
        assert!(ask_expr_node_opt.is_some(), "Did not find 'value' field in var_decl_node for multiline_strings test. S-expr var_decl: {:?}", var_decl_node.to_sexp());
        let ask_expr_node = ask_expr_node_opt.unwrap();

        println!(
            "S-expression for node passed to parse_ask_expression (multiline test): {:?}",
            ask_expr_node.to_sexp()
        );

        let data_opt = parse_ask_expression(ask_expr_node, source_code);
        assert!(
            data_opt.is_some(),
            "parse_ask_expression returned None for multiline_strings test. S-expr ask_node: {:?}",
            ask_expr_node.to_sexp()
        );
        let data = data_opt.unwrap();

        // Convert to HIR
        let hir_ask: HirAskExpression = data.clone().into();

        assert_eq!(
            hir_ask.system_prompt,
            Some("Multi-line\nsystem.".to_string())
        );
        assert_eq!(hir_ask.user_prompt, Some("User as well.".to_string()));
        assert!(hir_ask.attributes.is_empty());
        assert!(hir_ask.output_schema_name.is_none());
    }

    // TODO: Add more exhaustive tests for:
    // - Ask expressions with only some fields (e.g., only user, only system, no schema)
    // - Attribute combinations with and without value.
    // - Attributes with different literal types (when better supported than just string).
    // - Error cases or minor malformations inside an ask_expression (e.g., incomplete kv_pair).
    // - Nodes that are not ask_expression passed to the function.
}
