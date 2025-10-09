// lexc/src/executor/data_functions.rs
//
// Data processing functions extracted from executor/mod.rs
// This module handles all data-related operations including:
// - load_csv, save_csv, load_json, save_json
// - create_schema, validate_data
// - filter_data, transform_data

use crate::lexir::{LexExpression, ValueRef};
use std::sync::Arc;

use super::{ExecutionEnvironment, ExecutorError, RuntimeValue};
use crate::telemetry::trace_data_operation;

impl ExecutionEnvironment {
    /// Handle load_csv function - Load CSV data into memory
    pub fn handle_load_csv(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // load_csv(path) -> Dataset - Load CSV file using data processor
        if args.len() != 1 {
            return Err(ExecutorError::ArgumentError(
                "load_csv requires exactly 1 argument: path".to_string(),
            ));
        }

        let path_value = self.evaluate_expression(args[0].clone())?;
        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "load_csv path must be a string".to_string(),
                ))
            }
        };

        println!("📊 load_csv: Loading CSV file '{}'", path);
        let _span = trace_data_operation("load_csv", 0);
        _span.record_event("Reading CSV");

        let dataset = self
            .data_processor
            .load_data(&path, &std::collections::HashMap::new())?;

        println!("✅ load_csv: CSV loaded successfully");

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Dataset(Arc::new(dataset)))?;
        }
        Ok(())
    }

    /// Handle save_json function - Save data as JSON file
    pub fn handle_save_json(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // save_json(data, path) -> success - Save data as JSON
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "save_json requires exactly 2 arguments: data, path".to_string(),
            ));
        }

        let data_value = self.evaluate_expression(args[0].clone())?;
        let path_value = self.evaluate_expression(args[1].clone())?;

        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "save_json path must be a string".to_string(),
                ))
            }
        };

        println!("💾 save_json: Saving data to '{}'", path);
        let _span = trace_data_operation("save_json", 0);
        _span.record_event("Saving JSON");

        match data_value {
            RuntimeValue::Dataset(dataset) => {
                self.data_processor.export_data(dataset, &path, "json")?;
            }
            RuntimeValue::Json(json_value) => {
                let json_string = serde_json::to_string_pretty(&json_value).map_err(|e| {
                    ExecutorError::DataError(format!("JSON serialization error: {}", e))
                })?;
                std::fs::write(&path, json_string)
                    .map_err(|e| ExecutorError::DataError(format!("File write error: {}", e)))?;
            }
            _ => {
                // Convert other types to JSON
                let json_value = match data_value {
                    RuntimeValue::String(s) => serde_json::Value::String(s),
                    RuntimeValue::Integer(i) => serde_json::Value::Number(i.into()),
                    RuntimeValue::Float(f) => serde_json::Number::from_f64(f)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null),
                    RuntimeValue::Boolean(b) => serde_json::Value::Bool(b),
                    RuntimeValue::Null => serde_json::Value::Null,
                    _ => serde_json::Value::String(format!("{:?}", data_value)),
                };
                let json_string = serde_json::to_string_pretty(&json_value).map_err(|e| {
                    ExecutorError::DataError(format!("JSON serialization error: {}", e))
                })?;
                std::fs::write(&path, json_string)
                    .map_err(|e| ExecutorError::DataError(format!("File write error: {}", e)))?;
            }
        }

        println!("✅ save_json: Data saved successfully");

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Boolean(true))?;
        }
        Ok(())
    }

    /// Handle save_csv function - Save data as CSV file
    pub fn handle_save_csv(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // save_csv(data, path) -> success - Save dataset as CSV
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "save_csv requires exactly 2 arguments: data, path".to_string(),
            ));
        }

        let data_value = self.evaluate_expression(args[0].clone())?;
        let path_value = self.evaluate_expression(args[1].clone())?;

        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "save_csv path must be a string".to_string(),
                ))
            }
        };

        println!("💾 save_csv: Saving data to '{}'", path);
        let _span = trace_data_operation("save_csv", 0);
        _span.record_event("Saving CSV");

        match data_value {
            RuntimeValue::Dataset(dataset) => {
                self.data_processor.export_data(dataset, &path, "csv")?;
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "save_csv requires a Dataset value".to_string(),
                ));
            }
        }

        println!("✅ save_csv: Data saved successfully");

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Boolean(true))?;
        }
        Ok(())
    }

    /// Handle load_json function - Load JSON data into memory
    pub fn handle_load_json(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // load_json(path) -> Json - Load JSON file and store as Json RuntimeValue
        if args.len() != 1 {
            return Err(ExecutorError::ArgumentError(
                "load_json requires exactly 1 argument: path".to_string(),
            ));
        }

        let path_value = self.evaluate_expression(args[0].clone())?;
        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "load_json path must be a string".to_string(),
                ))
            }
        };

        println!("📥 load_json: Loading JSON file '{}'", path);
        let _span = trace_data_operation("load_json", 0);
        _span.record_event("Reading JSON");
        let content = std::fs::read_to_string(&path)
            .map_err(|e| ExecutorError::DataError(format!("File read error: {}", e)))?;
        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| ExecutorError::DataError(format!("JSON parse error: {}", e)))?;

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(json_value))?;
        }
        println!("✅ load_json: JSON loaded successfully");
        Ok(())
    }

    /// Handle create_schema function - Create data schema
    pub fn handle_create_schema(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // create_schema(dataset) -> Json (inferred JSON schema)
        if args.len() != 1 {
            return Err(ExecutorError::ArgumentError(
                "create_schema requires exactly 1 argument: dataset".to_string(),
            ));
        }

        let dataset_value = self.evaluate_expression(args[0].clone())?;
        let dataset = match dataset_value {
            RuntimeValue::Dataset(ds) => ds,
            _ => {
                return Err(ExecutorError::TypeError(
                    "create_schema expects a Dataset".to_string(),
                ))
            }
        };

        let schema = dataset.infer_json_schema().or_else(|_| {
            // Fallback: convert dataset rows to JSON and build a minimal schema
            let _rows = dataset.to_json_rows()?;
            let schema = serde_json::json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": { "type": "object" }
            });
            Ok(schema)
        })?;

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(schema))?;
        }
        Ok(())
    }

    /// Handle validate_data function - Validate data against schema
    pub fn handle_validate_data(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // validate_data(dataset, schema_json, mode?) -> bool or Dataset (if incremental)
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "validate_data requires at least 2 arguments: dataset, schema".to_string(),
            ));
        }

        let dataset_value = self.evaluate_expression(args[0].clone())?;
        let schema_value = self.evaluate_expression(args[1].clone())?;
        let mode = if args.len() > 2 {
            match self.evaluate_expression(args[2].clone())? {
                RuntimeValue::String(s) => s,
                RuntimeValue::Boolean(true) => "incremental".to_string(),
                _ => "strict".to_string(),
            }
        } else {
            "strict".to_string()
        };

        let dataset = match dataset_value {
            RuntimeValue::Dataset(ds) => ds,
            _ => {
                return Err(ExecutorError::TypeError(
                    "validate_data expects a Dataset as first argument".to_string(),
                ))
            }
        };

        let schema_json = match schema_value {
            RuntimeValue::Json(v) => v,
            RuntimeValue::String(s) => serde_json::from_str(&s)
                .map_err(|e| ExecutorError::DataError(format!("Invalid schema string: {}", e)))?,
            _ => {
                return Err(ExecutorError::TypeError(
                    "validate_data expects a JSON schema as second argument".to_string(),
                ))
            }
        };

        if mode.eq_ignore_ascii_case("incremental") {
            let validation_ds = dataset.validate_incremental(&schema_json)?;
            if let Some(res) = result {
                self.store_value(res, RuntimeValue::Dataset(Arc::new(validation_ds)))?;
            }
        } else {
            dataset.validate_against_schema(&schema_json)?;
            if let Some(res) = result {
                self.store_value(res, RuntimeValue::Boolean(true))?;
            }
        }
        Ok(())
    }

    /// Handle filter_data function - Filter data based on criteria
    pub fn handle_filter_data(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // filter_data(dataset, condition_string) -> Dataset
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "filter_data requires exactly 2 arguments: dataset, condition".to_string(),
            ));
        }

        let dataset_value = self.evaluate_expression(args[0].clone())?;
        let condition_value = self.evaluate_expression(args[1].clone())?;

        let dataset = match dataset_value {
            RuntimeValue::Dataset(ds) => ds,
            _ => {
                return Err(ExecutorError::TypeError(
                    "filter_data expects a Dataset as first argument".to_string(),
                ))
            }
        };

        let filtered = self.data_processor.filter_data(dataset, condition_value)?;
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Dataset(Arc::new(filtered)))?;
        }
        Ok(())
    }

    /// Handle transform_data function - Transform data structure
    pub fn handle_transform_data(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // transform_data(dataset, options_json) -> Dataset
        // Supported options: { select: ["field1","field2"], take: 100, flatten_json: true }
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "transform_data requires exactly 2 arguments: dataset, options".to_string(),
            ));
        }

        let dataset_value = self.evaluate_expression(args[0].clone())?;
        let options_value = self.evaluate_expression(args[1].clone())?;

        let dataset = match dataset_value {
            RuntimeValue::Dataset(ds) => ds,
            _ => {
                return Err(ExecutorError::TypeError(
                    "transform_data expects a Dataset as first argument".to_string(),
                ))
            }
        };

        let mut current = (*dataset).clone();

        if let RuntimeValue::Json(serde_json::Value::Object(map)) = options_value {
            // select fields
            if let Some(sel) = map.get("select") {
                if let Some(arr) = sel.as_array() {
                    let mut fields: Vec<String> = Vec::new();
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            fields.push(s.to_string());
                        }
                    }
                    let ds = self
                        .data_processor
                        .select_fields(Arc::new(current.clone()), &fields)?;
                    current = ds;
                }
            }

            // take rows
            if let Some(take_v) = map.get("take") {
                if let Some(n) = take_v.as_u64() {
                    let ds = self
                        .data_processor
                        .take_rows(Arc::new(current.clone()), n as usize)?;
                    current = ds;
                }
            }

            // flatten nested json (best-effort)
            if let Some(flatten_v) = map.get("flatten_json") {
                if flatten_v.as_bool().unwrap_or(false) {
                    let ds = self
                        .data_processor
                        .flatten_json(Arc::new(current.clone()), None)?;
                    current = ds;
                }
            }
        } else {
            return Err(ExecutorError::TypeError(
                "transform_data expects a JSON object for options".to_string(),
            ));
        }

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Dataset(Arc::new(current)))?;
        }
        Ok(())
    }

    /// Handle read_file(path) -> String
    pub fn handle_read_file(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.len() != 1 {
            return Err(ExecutorError::ArgumentError(
                "read_file requires exactly 1 argument: file path".to_string(),
            ));
        }

        let path_value = self.evaluate_expression(args[0].clone())?;
        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "read_file path must be a string".to_string(),
                ))
            }
        };

        println!("📁 read_file: Reading file '{}'", path);
        let _span = trace_data_operation("read_file", 0);
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                println!(
                    "✅ read_file: Successfully read {} bytes from '{}'",
                    content.len(),
                    path
                );
                if let Some(res) = result {
                    self.store_value(res, RuntimeValue::String(content))?;
                }
            }
            Err(e) => {
                let error_msg = format!("Failed to read file '{}': {}", path, e);
                println!("❌ read_file: {}", error_msg);
                return Err(ExecutorError::RuntimeError(error_msg));
            }
        }
        Ok(())
    }

    /// Handle write_file(path, content) -> void
    pub fn handle_write_file(
        &mut self,
        args: &[LexExpression],
        _result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "write_file requires exactly 2 arguments: path and content".to_string(),
            ));
        }

        let path_value = self.evaluate_expression(args[0].clone())?;
        let content_value = self.evaluate_expression(args[1].clone())?;

        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "write_file path must be a string".to_string(),
                ))
            }
        };

        let content = match content_value {
            RuntimeValue::String(s) => s,
            RuntimeValue::Json(json) => json.to_string(),
            _ => format!("{:?}", content_value),
        };

        println!("📝 write_file: Writing to file '{}'", path);
        let _span = trace_data_operation("write_file", content.len());
        match std::fs::write(&path, content) {
            Ok(_) => {
                println!("✅ write_file: Successfully wrote to '{}'", path);
            }
            Err(e) => {
                let error_msg = format!("Failed to write file '{}': {}", path, e);
                println!("❌ write_file: {}", error_msg);
                return Err(ExecutorError::RuntimeError(error_msg));
            }
        }
        Ok(())
    }

    /// Handle save_file(content, path) -> void (alias)
    pub fn handle_save_file(
        &mut self,
        args: &[LexExpression],
        _result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "save_file requires exactly 2 arguments: content and path".to_string(),
            ));
        }

        let content_value = self.evaluate_expression(args[0].clone())?;
        let path_value = self.evaluate_expression(args[1].clone())?;

        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "save_file path must be a string".to_string(),
                ))
            }
        };

        let content = match content_value {
            RuntimeValue::String(s) => s,
            RuntimeValue::Json(json) => json.to_string(),
            _ => format!("{:?}", content_value),
        };

        println!("💾 save_file: Writing to file '{}'", path);
        let _span = trace_data_operation("save_file", content.len());
        match std::fs::write(&path, content) {
            Ok(_) => {
                println!("✅ save_file: Successfully wrote to '{}'", path);
            }
            Err(e) => {
                let error_msg = format!("Failed to write file '{}': {}", path, e);
                println!("❌ save_file: {}", error_msg);
                return Err(ExecutorError::RuntimeError(error_msg));
            }
        }
        Ok(())
    }

    /// Handle save_binary_file(multioutput, path) -> void
    pub fn handle_save_binary_file(
        &mut self,
        args: &[LexExpression],
        _result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.len() != 2 {
            return Err(ExecutorError::ArgumentError(
                "save_binary_file requires exactly 2 arguments: binary_file and path".to_string(),
            ));
        }

        let file_value = self.evaluate_expression(args[0].clone())?;
        let path_value = self.evaluate_expression(args[1].clone())?;

        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "save_binary_file path must be a string".to_string(),
                ))
            }
        };

        match file_value {
            RuntimeValue::MultiOutput { binary_files, .. } => {
                if binary_files.is_empty() {
                    return Err(ExecutorError::ArgumentError(
                        "No binary files found in MultiOutput".to_string(),
                    ));
                }
                self.save_binary_file(&binary_files[0], &path)?;
            }
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "save_binary_file requires a MultiOutput with binary files".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Handle load_binary_file(path, [name]) -> MultiOutput
    pub fn handle_load_binary_file(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() || args.len() > 2 {
            return Err(ExecutorError::ArgumentError(
                "load_binary_file requires 1-2 arguments: path and optional name".to_string(),
            ));
        }

        let path_value = self.evaluate_expression(args[0].clone())?;
        let path = match path_value {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::ArgumentError(
                    "load_binary_file path must be a string".to_string(),
                ))
            }
        };

        let name = if args.len() > 1 {
            let name_value = self.evaluate_expression(args[1].clone())?;
            match name_value {
                RuntimeValue::String(s) => Some(s),
                _ => None,
            }
        } else {
            None
        };

        let binary_file = self.load_binary_file(&path, name)?;
        let multioutput = self.create_multioutput("".to_string(), vec![binary_file]);
        if let Some(res) = result {
            self.store_value(res, multioutput)?;
        }
        Ok(())
    }

    /// Handle load_file(path) -> String (alias of read_file)
    pub fn handle_load_file(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        self.handle_read_file(args, result)
    }
}
