// This will be the main file for the Lexon grammar using tree-sitter. 

module.exports = grammar({
    name: 'lexon',

    extras: $ => [
        /\s+/, // Whitespace (one or more)
        $.comment
    ],

    rules: {
        // Main rule: a source file is a sequence of top-level statements
        source_file: $ => repeat($._top_level_statement),

        // Top-level statements
        _top_level_statement: $ => choice(
            $.module_declaration,
            $.import_statement,
            $.variable_declaration,
            $.function_definition,
            $.schema_definition,
            $.trait_definition,
            $.impl_block,
            $.if_statement,
            $.while_statement,
            $.for_in_statement,
            $.match_statement,
            $.expression_statement
        ),

        // Comments (including doc-comments)
        comment: $ => token(choice(
            seq('//', /.*/),    // Single-line comment
            seq('///', /.*/),   // Documentation comment
            seq('/*', /[^*]*\*+([^/*][^*]*\*+)*\//) // C-style block comment
        )),

        // Identifiers (variable, function, type names, etc.)
        identifier: $ => /[a-zA-Z_][a-zA-Z0-9_]*/,

        // Dotted identifier for module paths (e.g., core.llm.text)
        dotted_identifier: $ => seq($.identifier, repeat(seq('.', $.identifier))),

        // Module declaration: `module analytics.stats;`
        module_declaration: $ => seq(
            'module',
            field('path', $.dotted_identifier),
            ';'
        ),

        // Imports – form 1: `import core.llm;` or `import core.llm as llm;`
        simple_import: $ => seq(
            'import',
            field('path', $.dotted_identifier),
            optional(seq('as', field('alias', $.identifier))),
            ';'
        ),

        // Item list in a selective import
        import_item: $ => seq($.identifier, optional(seq('as', $.identifier))),
        import_item_list: $ => seq($.import_item, repeat(seq(',', $.import_item))),

        // Imports – form 2: `from core.types import string, int as alias;`
        from_import: $ => seq(
            'from', field('path', $.dotted_identifier),
            'import', field('items', $.import_item_list),
            ';'
        ),

        import_statement: $ => choice($.simple_import, $.from_import),

        // Types (simplified as an identifier for now)
        type_identifier: $ => $.identifier,

        // --- Type Definitions (S-2) ---
        type_identifier_with_generics: $ => choice(
            seq(field('name', $.identifier), field('parameters', $.generic_type_parameters)), // e.g., List<string>
            $.identifier // e.g., int, string, UserProfile (schema name)
        ),

        generic_type_parameters: $ => seq(
            '<',
            $.type_identifier_with_generics,
            repeat(seq(',', $.type_identifier_with_generics)),
            '>'
        ),

        // Literals (basic for now)
        _literal: $ => choice(
            $.string_literal,
            $.multiline_string_literal,
            $.integer_literal,
            $.float_literal,
            $.boolean_literal,
            $.array_literal
        ),
        string_literal: $ => seq('\"', repeat(choice(/[^"\\]+/, /\\./)), '\"'),
        integer_literal: $ => /-?[0-9]+/, // Support negative integers
        float_literal: $ => /-?[0-9]+\.[0-9]+/, // Support negative floats
        boolean_literal: $ => choice('true', 'false'),
        multiline_string_literal: $ => token(seq('"""', /([^"]|"[^"]|""[^"])*/, '"""')),

        // Array literal: [1, 2, 3] or ["a", "b", "c"] or []
        array_literal: $ => seq(
            '[',
            optional(seq(
                field('element', $._expression),
                repeat(seq(',', field('element', $._expression)))
            )),
            ']'
        ),

        attribute: $ => seq(
            '@',
            field('name', $.identifier),
            optional(seq(
                '(',
                field('arguments', $.attribute_args),
                ')'
            ))
        ),
        attribute_args: $ => seq(
            field('value', $.attribute_arg),
            repeat(seq(',', field('value', $.attribute_arg)))
        ),
        attribute_arg: $ => choice($._literal, $.identifier),

        // Method call: Target.method(args)
        method_call: $ => seq(
            field('target', $.identifier),
            '.',
            field('method', $.identifier),
            '(',
            optional(seq(
                field('arg', $._expression),
                repeat(seq(',', field('arg', $._expression)))
            )),
            ')'
        ),

        ask_expression: $ => choice(
            // Block syntax: ask { user: "prompt"; }
            seq(
                repeat($.attribute),  // Attributes before 'ask'
                'ask',
                repeat($.attribute),  // Attributes after 'ask'
                '{',
                repeat($.ask_kv_pair),
                '}'
            ),
            // Function syntax: ask("prompt") or ask("prompt", "model")
            seq(
                'ask',
                '(',
                field('prompt', $._expression),
                optional(seq(',', field('model', $._expression))),
                ')'
            )
        ),

        // New ask_safe expression with anti-hallucination validation
        ask_safe_expression: $ => choice(
            // Block syntax: ask_safe { user: "prompt"; validation: "basic"; }
            seq(
                repeat($.attribute),  // Attributes before 'ask_safe'
                'ask_safe',
                repeat($.attribute),  // Attributes after 'ask_safe'
                '{',
                repeat($.ask_safe_kv_pair),
                '}'
            ),
            // Function syntax: ask_safe("prompt", validation: "ensemble")
            seq(
                'ask_safe',
                '(',
                field('prompt', $._expression),
                optional(seq(',', repeat($.ask_safe_parameter))),
                ')'
            )
        ),

        ask_kv_pair: $ => seq(
            field('key', choice('system', 'user', 'schema', 'model', 'temperature', 'max_tokens')),
            ':',
            field('value', choice($._literal, $.identifier)),
            ';'
        ),

        // Key-value pairs for ask_safe with validation parameters
        ask_safe_kv_pair: $ => prec(1, seq(
            field('key', choice(
                'system', 'user', 'schema', 'model', 'temperature', 'max_tokens',
                // Anti-hallucination validation parameters
                'validation', 'confidence_threshold', 'max_attempts', 
                'cross_reference_models', 'use_fact_checking'
            )),
            ':',
            field('value', choice($._literal, $.identifier)),
            ';'
        )),

        // Named parameters for ask_safe function syntax
        ask_safe_parameter: $ => seq(
            field('name', choice(
                'validation', 'confidence_threshold', 'max_attempts',
                'cross_reference_models', 'use_fact_checking'
            )),
            ':',
            field('value', choice($._literal, $.identifier))
        ),

        typeof_expression: $ => seq(
            'typeof',
            '(',
            field('argument', $._expression),
            ')'
        ),

        // Generic type arguments list e.g. <int, string>
        generic_type_arguments: $ => seq(
            '<',
            $.type_identifier_with_generics,
            repeat(seq(',', $.type_identifier_with_generics)),
            '>'
        ),

        // Function call: foo<types>(args) or foo(args)
        function_call: $ => seq(
            field('function', $.identifier),
            optional(field('type_args', $.generic_type_arguments)),
            '(',
            optional(seq(
                field('arg', $._expression),
                repeat(seq(',', field('arg', $._expression)))
            )),
            ')'
        ),

        // Special functions for the anti-hallucination system
        ask_ensemble_call: $ => seq(
            'ask_ensemble',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        ask_parallel_call: $ => seq(
            'ask_parallel',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        ask_consensus_call: $ => seq(
            'ask_consensus',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        ask_with_fallback_call: $ => seq(
            'ask_with_fallback',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        memory_store_call: $ => seq(
            'memory_store',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        memory_load_call: $ => seq(
            'memory_load',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        // New functional iterator functions
        enumerate_call: $ => seq(
            'enumerate',
            '(',
            field('arg', $._expression),
            ')'
        ),

        range_call: $ => seq(
            'range',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        map_call: $ => seq(
            'map',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        filter_call: $ => seq(
            'filter',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        reduce_call: $ => seq(
            'reduce',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        // New I/O functions
        read_file_call: $ => seq(
            'read_file',
            '(',
            field('arg', $._expression),
            ')'
        ),

        write_file_call: $ => seq(
            'write_file',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        load_csv_call: $ => seq(
            'load_csv',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        save_json_call: $ => seq(
            'save_json',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        save_file_call: $ => seq(
            'save_file',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        load_file_call: $ => seq(
            'load_file',
            '(',
            field('arg', $._expression),
            ')'
        ),

        execute_call: $ => seq(
            'execute',
            '(',
            field('arg', $._expression),
            ')'
        ),

        // New global configuration functions (Sprint B)
        set_default_model_call: $ => seq(
            'set_default_model',
            '(',
            field('arg', $._expression),
            ')'
        ),

        get_provider_default_call: $ => seq(
            'get_provider_default',
            '(',
            field('arg', $._expression),
            ')'
        ),

        // New anti-hallucination validation functions (Sprint C)
        confidence_score_call: $ => seq(
            'confidence_score',
            '(',
            field('arg', $._expression),
            ')'
        ),

        validate_response_call: $ => seq(
            'validate_response',
            '(',
            field('arg', $._expression),
            ',',
            field('arg', $._expression),
            ')'
        ),

        // New indexed memory functions (Sprint D)
        memory_index_ingest_call: $ => seq(
            'memory_index.ingest',
            '(',
            field('arg', $._expression),
            ')'
        ),

        memory_index_vector_search_call: $ => seq(
            'memory_index.vector_search',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        auto_rag_context_call: $ => seq(
            'auto_rag_context',
            '(',
            ')'
        ),

        // 📦 Multioutput functions - Multiple outputs system
        ask_multioutput_call: $ => seq(
            'ask_multioutput',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        save_binary_file_call: $ => seq(
            'save_binary_file',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        load_binary_file_call: $ => seq(
            'load_binary_file',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        get_multioutput_text_call: $ => seq(
            'get_multioutput_text',
            '(',
            field('arg', $._expression),
            ')'
        ),

        get_multioutput_files_call: $ => seq(
            'get_multioutput_files',
            '(',
            field('arg', $._expression),
            ')'
        ),

        get_multioutput_metadata_call: $ => seq(
            'get_multioutput_metadata',
            '(',
            field('arg', $._expression),
            ')'
        ),

        save_multioutput_file_call: $ => seq(
            'save_multioutput_file',
            '(',
            field('arg', $._expression),
            repeat(seq(',', field('arg', $._expression))),
            ')'
        ),

        // Expressions - UPDATED VERSION
        _expression: $ => choice(
            $.assignment_expression,
            $.binary_expression,
            $.await_expression,
            $.method_call,
            $.function_call,
            $.ask_expression,
            $.ask_safe_expression,     // New ask_safe expression
            $.ask_ensemble_call,       // Function ask_ensemble
            $.ask_parallel_call,       // Function ask_parallel
            $.ask_consensus_call,      // Function ask_consensus
            $.ask_with_fallback_call,  // Function ask_with_fallback
            $.memory_store_call,       // Function memory_store
            $.memory_load_call,        // Function memory_load
            $.enumerate_call,          // Function enumerate
            $.range_call,              // Function range
            $.map_call,                // Function map
            $.filter_call,             // Function filter
            $.reduce_call,             // Function reduce
            $.read_file_call,          // Function read_file
            $.write_file_call,         // Function write_file
            $.save_file_call,          // Function save_file
            $.load_file_call,          // Function load_file
            $.execute_call,            // Function execute
            $.set_default_model_call,  // Function set_default_model
            $.get_provider_default_call, // Function get_provider_default
            $.confidence_score_call,   // Function confidence_score
            $.validate_response_call,  // Function validate_response
            $.memory_index_ingest_call, // Function memory_index.ingest
            $.memory_index_vector_search_call, // Function memory_index.vector_search
            $.auto_rag_context_call,   // Function auto_rag_context
            $.ask_multioutput_call,    // Function ask_multioutput
            $.save_binary_file_call,   // Function save_binary_file
            $.load_binary_file_call,   // Function load_binary_file
            $.get_multioutput_text_call, // Function get_multioutput_text
            $.get_multioutput_files_call, // Function get_multioutput_files
            $.get_multioutput_metadata_call, // Function get_multioutput_metadata
            $.save_multioutput_file_call, // Function save_multioutput_file
            $.typeof_expression,
            $.identifier,
            $._literal
        ),

        // Binary expressions - FIXED VERSION WITH LOGICAL OPERATORS
        binary_expression: $ => choice(
            // Logical operators (lowest precedence)
            prec.left(0, seq(field('left', choice($.identifier, $._literal, $.binary_expression)), '||', field('right', choice($.identifier, $._literal, $.binary_expression)))),
            prec.left(0, seq(field('left', choice($.identifier, $._literal, $.binary_expression)), '&&', field('right', choice($.identifier, $._literal, $.binary_expression)))),
            // Comparison operators
            prec.left(1, seq(field('left', $.identifier), '>', field('right', choice($.identifier, $._literal)))),
            prec.left(1, seq(field('left', $.identifier), '<', field('right', choice($.identifier, $._literal)))),
            prec.left(1, seq(field('left', $.identifier), '>=', field('right', choice($.identifier, $._literal)))),
            prec.left(1, seq(field('left', $.identifier), '<=', field('right', choice($.identifier, $._literal)))),
            prec.left(1, seq(field('left', $.identifier), '==', field('right', choice($.identifier, $._literal)))),
            prec.left(1, seq(field('left', $.identifier), '!=', field('right', choice($.identifier, $._literal)))),
            // Arithmetic operators - FIXED: Only allow identifiers and binary_expressions on left side for +/-
            prec.left(2, seq(field('left', choice($.identifier, $.binary_expression)), '+', field('right', choice($.identifier, $._literal, $.binary_expression)))),
            prec.left(2, seq(field('left', choice($.identifier, $.binary_expression)), '-', field('right', choice($.identifier, $._literal, $.binary_expression)))),
            // Multiplication and division can still use literals on both sides for numeric operations
            prec.left(3, seq(field('left', choice($.identifier, $._literal)), '*', field('right', choice($.identifier, $._literal)))),
            prec.left(3, seq(field('left', choice($.identifier, $._literal)), '/', field('right', choice($.identifier, $._literal))))
        ),

        // Assignment expressions
        assignment_expression: $ => seq(
            field("left", $.identifier),
            "=",
            field("right", $._expression)
        ),

        // Assignment statements (assignments with semicolon) - HIGH PRECEDENCE
        assignment_statement: $ => prec(1, seq(
            field("left", $.identifier),
            "=", 
            field("right", $._expression),
            ";"
        )),

        // An expression followed by a semicolon as a statement (e.g., a standalone function call)
        await_expression: $ => seq('await', field('expression', choice(
            $.ask_expression, 
            $.ask_safe_expression, 
            $.ask_ensemble_call,
            $.ask_parallel_call,
            $.ask_consensus_call,
            $.ask_with_fallback_call,
            $.memory_store_call,
            $.memory_load_call,
            $.enumerate_call,
            $.range_call,
            $.map_call,
            $.filter_call,
            $.reduce_call,
            $.read_file_call,
            $.write_file_call,
            $.load_csv_call,
            $.save_json_call,
            $.save_file_call,
            $.load_file_call,
            $.execute_call,
            $.set_default_model_call,
            $.get_provider_default_call,
            $.confidence_score_call,
            $.validate_response_call,
            $.memory_index_ingest_call,
            $.memory_index_vector_search_call,
            $.auto_rag_context_call,
            $.ask_multioutput_call,
            $.save_binary_file_call,
            $.load_binary_file_call,
            $.get_multioutput_text_call,
            $.get_multioutput_files_call,
            $.get_multioutput_metadata_call,
            $.save_multioutput_file_call,
            $.function_call, 
            $.method_call, 
            $.identifier
        ))),
        expression_statement: $ => seq(
            $._expression,
            ';'
        ),

        // Variable Declaration
        // Variable Declaration - UPDATED: Optional type annotation for type inference
        variable_declaration: $ => seq(
            choice('let', 'var'),
            field('name', $.identifier),
            optional(seq(
                ':',
                field('type', $.type_identifier_with_generics)
            )),
            '=',
            field('value', $._expression),
            ';'
        ),

        // Function parameter list
        parameter_list: $ => seq(
            $.parameter,
            repeat(seq(',', $.parameter))
        ),

        // Individual parameter: name: type
        parameter: $ => seq(
            field('name', $.identifier),
            ':',
            field('type', $.type_identifier_with_generics)
        ),

                // Function Definition
        function_definition: $ => seq(
            optional(field('visibility', choice('pub', 'private'))),
            optional(field('async_modifier', 'async')),
            'fn',
            field('name', $.identifier),
            optional(field('type_params', $.generic_parameter_list)),
            '(',
            optional(field('parameters', $.parameter_list)),
            ')',
            optional(seq(
                '->',
                field('return_type', $.type_identifier_with_generics)
            )),
            field('body', $.block_statement)
        ),

        // Block of statements (function body, if, etc.)
        block_statement: $ => seq(
            '{',
            repeat($._statement_in_block),
            '}'
        ),

        _statement_in_block: $ => choice(
            $.variable_declaration,
            $.assignment_statement,
            $.if_statement,
            $.while_statement,
            $.for_in_statement,
            $.match_statement,
            $.break_statement,
            $.continue_statement,
            $.return_statement,
            $.expression_statement
        ),

        // --- Schema Definition (S-2) ---
        schema_definition: $ => seq(
            'schema',
            field('name', $.identifier),
            optional(field('type_params', $.generic_parameter_list)),
            '{',
            repeat($.schema_field_definition),
            '}'
        ),

        schema_field_definition: $ => seq(
            field('name', $.identifier),
            field('optional_marker', optional('?')),
            ':',
            field('type', $.type_identifier_with_generics),
            optional(seq('=', field('default_value', $._expression))),
            ';'
        ),

        // --- FIXED CONTROL FLOW CONSTRUCTS ---
        if_statement: $ => seq(
            'if',
            field('condition', $._expression),
            field('consequence', $.block_statement),
            repeat(seq(
                'else', 'if',
                field('condition', $._expression),
                field('consequence', $.block_statement)
            )),
            optional(seq('else', field('alternative', $.block_statement)))
        ),

        while_statement: $ => seq(
            'while',
            field('condition', $._expression),
            field('body', $.block_statement)
        ),

        for_in_statement: $ => seq(
            'for',
            field('iterator', $.identifier),
            'in',
            field('iterable', $._expression),
            field('body', $.block_statement)
        ),

        match_statement: $ => seq(
            'match',
            field('value', $._expression),
            '{',
            repeat($.match_arm),
            '}'
        ),

        match_arm: $ => seq(
            field('pattern', $._expression),
            '=>',
            choice($.block_statement, $._expression),
            optional(',')
        ),

        break_statement: $ => seq('break', ';'),
        continue_statement: $ => seq('continue', ';'),
        return_statement: $ => seq('return', optional($._expression), ';'),

        // --- TRAIT & IMPL ---
        trait_definition: $ => seq(
            'trait',
            field('name', $.identifier),
            optional(field('type_params', $.generic_parameter_list)),
            '{',
            repeat($.function_signature),
            '}',
        ),

        function_signature: $ => seq(
            'fn', field('name', $.identifier),
            '(',
            optional(field('parameters', $.parameter_list)),
            ')',
            optional(seq('->', field('return_type', $.type_identifier_with_generics))),
            ';'
        ),

        impl_block: $ => seq(
            'impl',
            field('target', $.identifier),
            '{',
            repeat($.function_definition),
            '}',
        ),

        // Generic parameter list e.g. <T, U>
        generic_parameter_list: $ => seq(
            '<',
            field('param', $.identifier),
            repeat(seq(',', field('param', $.identifier))),
            '>'
        ),
    }
}); 
