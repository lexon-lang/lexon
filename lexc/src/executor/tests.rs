// lexc/src/executor/tests.rs
//
// Unit tests for the executor

// Tests for executor module
#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use crate::executor::data_processor::DataProcessor;
    use crate::executor::execute;
    // use crate::executor::Result;
    use crate::lexir::{LexExpression, LexInstruction, LexLiteral, LexProgram, ValueRef};
    use std::collections::HashMap;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;
    use tempfile::TempDir;

    fn setup_test_csv() -> (TempDir, PathBuf) {
        // Create a temporary directory for test files
        let dir = tempdir().expect("Failed to create temp directory");
        let csv_path = dir.path().join("test.csv");

        // Test content
        let csv_content = r#"id,name,age
1,John Doe,30
2,Jane Smith,25
3,Bob Johnson,40
4,Alice Brown,35
5,Charlie Wilson,22
"#;

        std::fs::write(&csv_path, csv_content).expect("Failed to write test CSV");

        (dir, csv_path)
    }

    #[test]
    fn test_data_operations() {
        // Prepare the test CSV file
        let (_temp_dir, csv_path) = setup_test_csv();

        // Create a LexIR program with data operations
        let mut program = LexProgram::new();

        // 1. DATA_LOAD
        let load_result = ValueRef::Named("contacts".to_string());
        let load_instruction = LexInstruction::DataLoad {
            result: load_result.clone(),
            source: csv_path.to_string_lossy().to_string(),
            schema: None,
            options: HashMap::new(),
        };
        program.add_instruction(load_instruction);

        // 2. DATA_FILTER (age > 30)
        let filter_result = ValueRef::Named("adults".to_string());
        let filter_condition = LexExpression::Value(ValueRef::Literal(LexLiteral::String(
            "age > 30".to_string(),
        )));
        let filter_instruction = LexInstruction::DataFilter {
            result: filter_result.clone(),
            input: load_result.clone(),
            predicate: filter_condition,
        };
        program.add_instruction(filter_instruction);

        // 3. DATA_SELECT (only name and age)
        let select_result = ValueRef::Named("selected".to_string());
        let select_fields = vec!["name".to_string(), "age".to_string()];
        let select_instruction = LexInstruction::DataSelect {
            result: select_result.clone(),
            input: filter_result.clone(),
            fields: select_fields,
        };
        program.add_instruction(select_instruction);

        // 4. DATA_EXPORT
        let export_path = csv_path.parent().unwrap().join("export_test.csv");
        let export_instruction = LexInstruction::DataExport {
            input: select_result.clone(),
            path: export_path.to_string_lossy().to_string(),
            format: "csv".to_string(),
            options: HashMap::new(),
        };
        program.add_instruction(export_instruction);

        // Execute the program
        let result = execute(&program);
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());

        // Verify that the exported file exists
        assert!(export_path.exists(), "Exported file doesn't exist");

        // Read the exported file content
        let export_content =
            std::fs::read_to_string(&export_path).expect("Failed to read exported file");

        // Verify content (should have only 2 rows with age > 30, with name and age)
        assert!(
            export_content.contains("name,age"),
            "Header not found in exported file"
        );
        assert!(
            export_content.contains("Bob Johnson,40"),
            "Expected row not found"
        );
        assert!(
            export_content.contains("Alice Brown,35"),
            "Expected row not found"
        );
        assert!(!export_content.contains("John Doe"), "Unexpected row found");
        assert!(!export_content.contains("id"), "Unexpected column found");

        // Cleanup
        let _ = std::fs::remove_file(&export_path);
    }

    #[test]
    fn test_json_data_operations() {
        // Create a temporary directory for files
        let temp_dir = tempdir().expect("Failed to create temporary directory");

        // Create a test JSON file
        let json_path = temp_dir.path().join("test_data.json");
        let json_content = r#"[
            {"id": 1, "name": "Juan", "city": "Madrid", "age": 30},
            {"id": 2, "name": "Maria", "city": "Barcelona", "age": 25},
            {"id": 3, "name": "Pedro", "city": "Valencia", "age": 40},
            {"id": 4, "name": "Ana", "city": "Madrid", "age": 35}
        ]"#;
        std::fs::write(&json_path, json_content).expect("Failed to write test JSON file");

        // Create a test NDJSON file
        let ndjson_path = temp_dir.path().join("test_data.ndjson");
        let ndjson_content = r#"{"id": 1, "name": "Juan", "city": "Madrid", "age": 30}
{"id": 2, "name": "Maria", "city": "Barcelona", "age": 25}
{"id": 3, "name": "Pedro", "city": "Valencia", "age": 40}
{"id": 4, "name": "Ana", "city": "Madrid", "age": 35}"#;
        std::fs::write(&ndjson_path, ndjson_content).expect("Failed to write test NDJSON file");

        // Initialize the processor
        let mut processor = DataProcessor::new();

        // Load JSON
        let json_dataset = processor
            .load_data(json_path.to_str().unwrap(), &HashMap::new())
            .expect("Failed to load JSON data");

        // Verify it loaded correctly
        let df = json_dataset
            .collect()
            .expect("Failed to collect JSON dataset");
        assert_eq!(df.height(), 4, "Expected 4 rows in JSON dataset");
        assert_eq!(df.width(), 4, "Expected 4 columns in JSON dataset");

        // Load NDJSON
        let ndjson_dataset = processor
            .load_data(ndjson_path.to_str().unwrap(), &HashMap::new())
            .expect("Failed to load NDJSON data");

        // Verify it loaded correctly
        let df = ndjson_dataset
            .collect()
            .expect("Failed to collect NDJSON dataset");
        assert_eq!(df.height(), 4, "Expected 4 rows in NDJSON dataset");
        assert_eq!(df.width(), 4, "Expected 4 columns in NDJSON dataset");

        // Test filtering on the JSON dataset
        let json_filtered = processor
            .filter_data(
                std::sync::Arc::new(json_dataset.clone()),
                super::super::RuntimeValue::String("city = Madrid".to_string()),
            )
            .expect("Failed to filter JSON dataset");

        // Verify filtering
        let df = json_filtered
            .collect()
            .expect("Failed to collect filtered dataset");
        assert_eq!(df.height(), 2, "Expected 2 rows after filtering for Madrid");

        // Export to different formats

        // 1. Export to standard JSON
        let export_json_path = temp_dir.path().join("export.json");
        processor
            .export_data(
                std::sync::Arc::new(json_filtered.clone()),
                export_json_path.to_str().unwrap(),
                "json",
            )
            .expect("Failed to export to JSON");

        // Verify the file exists
        assert!(export_json_path.exists(), "JSON export file not created");

        // 2. Export to NDJSON
        let export_ndjson_path = temp_dir.path().join("export.ndjson");
        processor
            .export_data(
                std::sync::Arc::new(json_filtered.clone()),
                export_ndjson_path.to_str().unwrap(),
                "json",
            )
            .expect("Failed to export to NDJSON");

        // Verify the file exists
        assert!(
            export_ndjson_path.exists(),
            "NDJSON export file not created"
        );

        // 3. Roundtrip test: export and re-import
        let roundtrip_dataset = processor
            .load_data(export_json_path.to_str().unwrap(), &HashMap::new())
            .expect("Failed to load exported JSON data");

        // Verify content matches expectations
        let df = roundtrip_dataset
            .collect()
            .expect("Failed to collect roundtrip dataset");
        assert_eq!(df.height(), 2, "Expected 2 rows in roundtrip dataset");

        // Cleanup
        temp_dir
            .close()
            .expect("Failed to clean up temporary directory");
    }

    #[test]
    fn test_nested_json_operations() {
        // No temporary directory needed since we use existing files

        // Use the nested JSON file created earlier
        let json_path = "./data/users_nested.json"; // Adjust relative path if needed

        // Ensure the file exists
        if !std::path::Path::new(json_path).exists() {
            // If it does not exist, skip the test
            println!("Test file {} not found, skipping test", json_path);
            return;
        }

        // Initialize the processor
        let mut processor = DataProcessor::new();

        // Load the JSON with nested structures
        let nested_dataset = processor
            .load_data(json_path, &HashMap::new())
            .expect("Failed to load nested JSON data");

        // Verify it was loaded correctly
        let df = nested_dataset
            .collect()
            .expect("Failed to collect nested JSON dataset");
        assert_eq!(df.height(), 4, "Expected 4 rows in nested JSON dataset");

        // 1. Test filtering with a nested field
        let filtered_by_city = processor
            .filter_data(
                std::sync::Arc::new(nested_dataset.clone()),
                super::super::RuntimeValue::String("contact.address.city = Madrid".to_string()),
            )
            .expect("Failed to filter by nested field");

        // Verify filtering by nested city
        let df = filtered_by_city
            .collect()
            .expect("Failed to collect filtered dataset");
        assert_eq!(
            df.height(),
            2,
            "Expected 2 rows after filtering for Madrid in nested field"
        );

        // 2. Test selecting nested fields
        let selected_nested = processor
            .select_fields(
                std::sync::Arc::new(nested_dataset.clone()),
                &[
                    "name".to_string(),
                    "contact.address.city".to_string(),
                    "preferences.language".to_string(),
                ],
            )
            .expect("Failed to select nested fields");

        // Verify selection
        let df = selected_nested
            .collect()
            .expect("Failed to collect with nested field selection");
        assert_eq!(
            df.width(),
            3,
            "Expected 3 columns after selecting nested fields"
        );

        // 3. Test flattening JSON
        let flattened = processor
            .flatten_json(std::sync::Arc::new(nested_dataset.clone()), None)
            .expect("Failed to flatten nested JSON");

        // Verify flattening
        let df = flattened
            .collect()
            .expect("Failed to collect flattened dataset");
        // The number of columns should increase after flattening structures
        let orig_df = nested_dataset
            .collect()
            .expect("Failed to collect original dataset");
        assert!(
            df.width() >= orig_df.width(),
            "Expected more columns after flattening"
        );
    }

    #[test]
    fn test_invalid_json_error() {
        let dir = tempdir().expect("create temp dir");
        let invalid_path = dir.path().join("invalid.json");
        let mut file = std::fs::File::create(&invalid_path).unwrap();
        writeln!(file, "{{ invalid json").unwrap();

        let mut processor = DataProcessor::new();
        let result = processor.load_data(invalid_path.to_str().unwrap(), &HashMap::new());

        assert!(result.is_err(), "Expected error for invalid JSON");
        let err = format!("{}", result.err().unwrap());
        assert!(
            err.contains("Error reading JSON"),
            "Unexpected error message: {}",
            err
        );
    }

    #[test]
    fn test_invalid_jsonpath_error() {
        let (_dir, csv_path) = setup_test_csv();

        let mut processor = DataProcessor::new();
        let dataset = processor
            .load_data(csv_path.to_str().unwrap(), &HashMap::new())
            .unwrap();

        // Use invalid JSONPath
        let result = processor.filter_jsonpath(std::sync::Arc::new(dataset), "$[invalid jsonpath");

        assert!(result.is_err(), "Expected error for invalid JSONPath");
        let err = format!("{}", result.err().unwrap());
        assert!(err.contains("path error"), "Unexpected error: {}", err);
    }

    #[test]
    fn test_schema_validation_error() {
        let json_content = r#"[{"id":1,"name":"A"}]"#;
        let dir = tempdir().unwrap();
        let json_path = dir.path().join("data.json");
        std::fs::write(&json_path, json_content).unwrap();

        let mut processor = DataProcessor::new();
        let dataset = processor
            .load_data(json_path.to_str().unwrap(), &HashMap::new())
            .unwrap();

        // Schema requires missing field "age"
        let schema: serde_json::Value = serde_json::json!({
            "type": "object",
            "required": ["id", "name", "age"],
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        });

        let validation_result = dataset.validate_against_schema(&schema);
        assert!(
            validation_result.is_err(),
            "Expected schema validation error"
        );
        let msg = format!("{}", validation_result.err().unwrap());
        assert!(
            msg.contains("Schema validation error"),
            "Unexpected message: {}",
            msg
        );
    }

    #[test]
    fn test_jsonpath_select() {
        let json_content = r#"[
            {"id":1,"items":[{"price":10},{"price":20}]},
            {"id":2,"items":[{"price":30}]}
        ]"#;
        let dir = tempdir().unwrap();
        let path = dir.path().join("data.json");
        std::fs::write(&path, json_content).unwrap();

        let mut processor = DataProcessor::new();
        let dataset = processor
            .load_data(path.to_str().unwrap(), &HashMap::new())
            .unwrap();

        let extracted = processor
            .select_jsonpath(std::sync::Arc::new(dataset), "$[*].items[*].price")
            .unwrap();
        let df = extracted.collect().unwrap();
        // Should have 3 prices
        assert_eq!(df.height(), 3);
    }

    #[test]
    fn test_schema_inference() {
        // Create a simple dataset to infer schema
        let json_content = r#"[
            {"id":1,"name":"Item 1","price":10.5,"in_stock":true},
            {"id":2,"name":"Item 2","price":20.0,"in_stock":false},
            {"id":3,"name":"Item 3","price":30.5,"in_stock":true}
        ]"#;

        let dir = tempdir().unwrap();
        let path = dir.path().join("products.json");
        std::fs::write(&path, json_content).unwrap();

        // Load dataset
        let mut processor = DataProcessor::new();
        let dataset = processor
            .load_data(path.to_str().unwrap(), &HashMap::new())
            .unwrap();
        let dataset_arc = std::sync::Arc::new(dataset);

        // Infer schema
        let schema_value = dataset_arc.infer_json_schema().unwrap();

        // Verify schema has the correct properties
        if let serde_json::Value::Object(schema_obj) = &schema_value {
            if let Some(serde_json::Value::Object(props)) = schema_obj.get("properties") {
                // Verify required fields exist
                assert!(props.contains_key("id"), "Schema should have 'id' property");
                assert!(
                    props.contains_key("name"),
                    "Schema should have 'name' property"
                );
                assert!(
                    props.contains_key("price"),
                    "Schema should have 'price' property"
                );
                assert!(
                    props.contains_key("in_stock"),
                    "Schema should have 'in_stock' property"
                );

                // Verify types
                if let Some(serde_json::Value::Object(id_prop)) = props.get("id") {
                    if let Some(serde_json::Value::String(type_val)) = id_prop.get("type") {
                        assert_eq!(type_val, "integer", "Id should be integer type");
                    } else {
                        panic!("Id property missing type");
                    }
                }

                if let Some(serde_json::Value::Object(price_prop)) = props.get("price") {
                    if let Some(serde_json::Value::String(type_val)) = price_prop.get("type") {
                        assert_eq!(type_val, "number", "Price should be number type");
                    } else {
                        panic!("Price property missing type");
                    }
                }
            } else {
                panic!("Schema missing properties");
            }
        } else {
            panic!("Schema is not an object");
        }
    }

    #[test]
    #[ignore]
    fn test_incremental_validation() {
        let mut processor = DataProcessor::new();

        // Create test data with some valid and some invalid rows
        let temp_dir = tempdir().expect("Failed to create temporary directory");
        let json_path = temp_dir.path().join("validation_test.json");
        let json_content = r#"[
            {"name": "Valid User", "age": 25, "email": "valid@example.com"},
            {"name": "Invalid User", "age": "not_a_number", "email": "invalid_email"},
            {"name": "Another Valid", "age": 30, "email": "another@example.com"}
        ]"#;
        std::fs::write(&json_path, json_content).expect("Failed to write test JSON file");

        // Load data
        let dataset = processor
            .load_data(json_path.to_str().unwrap(), &HashMap::new())
            .expect("Failed to load JSON data");

        // Create a JSON schema for validation
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"}
            },
            "required": ["name", "age", "email"]
        });

        // Perform incremental validation
        let validation_result = dataset
            .validate_incremental(&schema)
            .expect("Failed to perform incremental validation");

        // Verify results
        let df = validation_result
            .collect()
            .expect("Failed to collect validation results");

        // It should have columns for original data + validation_errors
        assert!(
            df.get_column_names().contains(&"validation_errors"),
            "validation_errors column not found"
        );

        // Verify there are errors for the invalid row
        let validation_errors = df
            .column("validation_errors")
            .expect("Failed to get validation_errors column");

        // At least one row should have errors (the one with invalid age)
        let has_errors = validation_errors.iter().any(|val| !val.is_null());
        assert!(has_errors, "Expected at least one validation error");

        temp_dir
            .close()
            .expect("Failed to clean up temporary directory");
    }

    #[test]
    fn test_consolidated_executor_data_pipeline() {
        // This test verifies the consolidated executor can run
        // data operations using LexInstruction

        use crate::executor::{ExecutionEnvironment, ExecutorConfig};
        use crate::lexir::{LexInstruction, LexLiteral, LexProgram, TempId, ValueRef};

        // paths (use workspace samples to avoid missing test assets)
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let src_path = manifest_dir
            .join("..")
            .join("samples")
            .join("triage")
            .join("tickets.csv");
        let src = src_path.to_string_lossy().to_string();
        let out = "/tmp/contacts_pipeline_test.csv";

        // Build program
        let mut prog = LexProgram::new();
        let t0 = ValueRef::Temp(TempId(0));
        let t1 = ValueRef::Temp(TempId(1));
        let t2 = ValueRef::Temp(TempId(2));

        prog.add_instruction(LexInstruction::DataLoad {
            result: t0.clone(),
            source: src,
            schema: None,
            options: Default::default(),
        });
        prog.add_instruction(LexInstruction::DataSelect {
            result: t1.clone(),
            input: t0.clone(),
            fields: vec!["subject".into()],
        });
        prog.add_instruction(LexInstruction::DataTake {
            result: t2.clone(),
            input: t1.clone(),
            count: ValueRef::Literal(LexLiteral::Integer(2)),
        });
        prog.add_instruction(LexInstruction::DataExport {
            input: t2.clone(),
            path: out.into(),
            format: "csv".into(),
            options: Default::default(),
        });

        // Execute
        let mut env = ExecutionEnvironment::new(ExecutorConfig::default());
        let result = env.execute_program(&prog);

        // Check result
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());

        // Check output file exists
        assert!(
            std::path::Path::new(out).exists(),
            "Output file was not created"
        );

        // Verify content
        let content = std::fs::read_to_string(out).expect("Failed to read output file");
        assert!(content.contains("subject"), "Header not found");
        // Should contain first 2 subjects from tickets.csv
        assert!(
            content.contains("Payment failed"),
            "Expected subject not found"
        );
        assert!(
            content.contains("Login issue"),
            "Expected subject not found"
        );

        // Cleanup
        let _ = std::fs::remove_file(out);
    }
}
