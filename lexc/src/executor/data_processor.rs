// lexc/src/executor/data_processor.rs
//
// Data processor for the executor
// Implements data operations using Polars as the backend

use super::{ExecutorError, Result, RuntimeValue};
use crate::lexir::LexLiteral;
use polars::prelude::BooleanChunked;
use polars::prelude::*;
use polars_io::json::JsonFormat;
use polars_io::json::JsonReader;
use polars_io::parquet::ParquetWriter;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::iter::Iterator;
// use std::path::Path;
use std::sync::Arc;

/// Wrapper for Polars DataFrame
#[derive(Clone)]
pub struct Dataset {
    /// Dataset name
    name: String,
    /// Polars LazyFrame
    df: LazyFrame,
    /// Additional metadata
    metadata: HashMap<String, String>,
}

impl Dataset {
    /// Creates a new empty dataset
    fn new(name: &str, df: LazyFrame) -> Self {
        Dataset {
            name: name.to_string(),
            df,
            metadata: HashMap::new(),
        }
    }

    /// Executes and materializes the LazyFrame
    pub fn collect(&self) -> std::result::Result<DataFrame, PolarsError> {
        self.df.clone().collect()
    }

    /// Converts the Dataset to JSON format
    #[allow(dead_code)]
    fn to_json(&self) -> Result<String> {
        let _df = self
            .collect()
            .map_err(|e| ExecutorError::DataError(format!("Error collecting dataframe: {}", e)))?;

        // Use serde_json functionality for cleaner serialization
        let json_rows = self.to_json_rows()?;
        serde_json::to_string_pretty(&json_rows)
            .map_err(|e| ExecutorError::DataError(format!("Error serializing to JSON: {}", e)))
    }

    /// Converts the Dataset to an array of JSON objects
    pub fn to_json_rows(&self) -> Result<Vec<Value>> {
        let df = self
            .collect()
            .map_err(|e| ExecutorError::DataError(format!("Error collecting dataframe: {}", e)))?;

        let mut rows: Vec<Value> = Vec::with_capacity(df.height());
        let columns = df.get_columns();

        for row_idx in 0..df.height() {
            let mut map = serde_json::Map::with_capacity(columns.len());
            for col in columns {
                // get() now returns a Result<AnyValue, PolarsError>
                let av_result = col.get(row_idx);

                // Convert the value to JSON or use null on error
                let json_val = match av_result {
                    Ok(av) => self.anyvalue_to_json(&av),
                    Err(_) => Value::Null,
                };

                map.insert(col.name().to_string(), json_val);
            }
            rows.push(Value::Object(map));
        }

        Ok(rows)
    }

    /// Converts a Polars `AnyValue` into a nested `serde_json::Value`.
    #[allow(clippy::only_used_in_recursion)]
    fn anyvalue_to_json(&self, av: &AnyValue) -> Value {
        match av {
            AnyValue::Null => Value::Null,
            AnyValue::Int64(i) => Value::Number((*i).into()),
            AnyValue::Int32(i) => Value::Number((*i).into()),
            AnyValue::UInt64(u) => Value::Number(serde_json::Number::from(*u)),
            AnyValue::UInt32(u) => Value::Number(serde_json::Number::from(*u)),
            AnyValue::Float64(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            AnyValue::Float32(f) => serde_json::Number::from_f64(*f as f64)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            AnyValue::String(s) => Value::String(s.to_string()),
            AnyValue::Boolean(b) => Value::Bool(*b),
            AnyValue::List(series) => {
                let mut arr = Vec::with_capacity(series.len());
                for av in series.iter() {
                    arr.push(self.anyvalue_to_json(&av));
                }
                Value::Array(arr)
            }
            AnyValue::Struct(..) => Value::String(format!("{:?}", av)),
            _ => Value::String(format!("{:?}", av)),
        }
    }

    /// Returns a textual representation of the dataset
    pub fn to_string(&self) -> Result<String> {
        let df = self
            .collect()
            .map_err(|e| ExecutorError::DataError(format!("Error collecting dataframe: {}", e)))?;

        let mut result = format!(
            "Dataset: {} ({} rows, {} columns)\n",
            self.name,
            df.height(),
            df.width()
        );

        // Add schema
        result.push_str("Schema:\n");
        for field in df.schema().iter_fields() {
            result.push_str(&format!("  {}: {}\n", field.name(), field.data_type()));
        }

        // Show the first rows (head)
        let sample_size = 5.min(df.height());
        if sample_size > 0 {
            result.push_str("\nSample rows:\n");
            let head_df = df.head(Some(sample_size));
            result.push_str(&format!("{}", head_df));
        }

        if df.height() > sample_size {
            result.push_str(&format!(
                "\n... and {} more rows\n",
                df.height() - sample_size
            ));
        }

        Ok(result)
    }

    /// Exports the dataset to a file
    pub fn export(&self, path: &str, format: &str) -> Result<()> {
        let df = self
            .collect()
            .map_err(|e| ExecutorError::DataError(format!("Error collecting dataframe: {}", e)))?;

        match format.to_lowercase().as_str() {
            "csv" => {
                let mut file = std::fs::File::create(path).map_err(|e| {
                    ExecutorError::DataError(format!("Error creating file {}: {}", path, e))
                })?;

                let mut df_mut = df.clone();

                CsvWriter::new(&mut file)
                    .with_separator(b',')
                    .finish(&mut df_mut)
                    .map_err(|e| ExecutorError::DataError(format!("Error writing CSV: {}", e)))?;
            }
            "parquet" => {
                let mut file = std::fs::File::create(path).map_err(|e| {
                    ExecutorError::DataError(format!("Error creating file {}: {}", path, e))
                })?;
                let mut df_mut = df.clone();
                ParquetWriter::new(&mut file)
                    .finish(&mut df_mut)
                    .map_err(|e| {
                        ExecutorError::DataError(format!("Error writing Parquet: {}", e))
                    })?;
            }
            "json" => {
                // Serialize to JSON lines (NDJSON, one JSON object per line)
                let json_rows = self.to_json_rows()?;

                // Decide between pretty and minified based on extension
                if path.ends_with(".pretty.json") {
                    // Pretty JSON for readability
                    let json_string = serde_json::to_string_pretty(&json_rows).map_err(|e| {
                        ExecutorError::DataError(format!("Error serializing JSON: {}", e))
                    })?;
                    std::fs::write(path, json_string).map_err(|e| {
                        ExecutorError::DataError(format!("Error writing JSON: {}", e))
                    })?;
                } else if path.ends_with(".ndjson") || path.ends_with(".jsonl") {
                    // NDJSON/JSONL (one line per object)
                    let mut content = String::new();
                    for row in json_rows {
                        let line = serde_json::to_string(&row).map_err(|e| {
                            ExecutorError::DataError(format!("Error serializing JSON line: {}", e))
                        })?;
                        content.push_str(&line);
                        content.push('\n');
                    }
                    std::fs::write(path, content).map_err(|e| {
                        ExecutorError::DataError(format!("Error writing NDJSON: {}", e))
                    })?;
                } else {
                    // Regular JSON array (minified by default)
                    let json_string = serde_json::to_string(&json_rows).map_err(|e| {
                        ExecutorError::DataError(format!("Error serializing JSON: {}", e))
                    })?;
                    std::fs::write(path, json_string).map_err(|e| {
                        ExecutorError::DataError(format!("Error writing JSON: {}", e))
                    })?;
                }
            }
            _ => {
                return Err(ExecutorError::DataError(format!(
                    "Unsupported export format: {}",
                    format
                )));
            }
        }

        Ok(())
    }

    /// Loads a parquet file into a Dataset (lazy)
    pub fn load_parquet(&self, name: &str, path: &str) -> Result<Dataset> {
        let lf = LazyFrame::scan_parquet(path, Default::default()).map_err(|e| {
            ExecutorError::DataError(format!("Error scanning parquet {}: {}", path, e))
        })?;
        Ok(Dataset::new(name, lf))
    }

    /// Validates the dataset against a provided JSON Schema.
    pub fn validate_against_schema(&self, schema: &Value) -> Result<()> {
        use jsonschema::JSONSchema;

        // Compile the schema
        let compiled = JSONSchema::compile(schema)
            .map_err(|e| ExecutorError::DataError(format!("Invalid JSON schema: {}", e)))?;

        // Convert dataset to JSON rows and validate each
        let rows = self.to_json_rows()?;
        for (idx, row) in rows.iter().enumerate() {
            if let Err(errors) = compiled.validate(row) {
                let first = errors.into_iter().next();
                if let Some(err) = first {
                    return Err(ExecutorError::DataError(format!(
                        "Schema validation error at row {}: {}",
                        idx, err
                    )));
                } else {
                    return Err(ExecutorError::DataError(format!(
                        "Schema validation error at row {}",
                        idx
                    )));
                }
            }
        }

        Ok(())
    }

    /// Infers a JSON schema from a dataset.
    ///
    /// The inferred schema contains the detected types and properties in the dataset.
    /// Useful as a starting point to create a more complete validation schema.
    pub fn infer_json_schema(&self) -> Result<Value> {
        println!("🔍 Inferring JSON schema from dataset");

        // Materialize the dataset
        let df = self
            .collect()
            .map_err(|e| ExecutorError::DataError(format!("Error collecting dataframe: {}", e)))?;

        // Map for schema properties
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        // Analyze each column
        for col in df.get_columns() {
            let col_name = col.name();

            // Determine JSON type based on Polars type
            let json_type = match col.dtype() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => "integer",
                DataType::Float32 | DataType::Float64 => "number",
                DataType::Boolean => "boolean",
                DataType::String => "string",
                DataType::Struct(_) => "object",
                DataType::List(_) => "array",
                DataType::Null => continue, // Skip completely null columns
                _ => "string",              // Default to string
            };

            // Create property
            let mut prop = serde_json::Map::new();
            prop.insert("type".to_string(), Value::String(json_type.to_string()));

            // If the column has null values, it is not required
            // Using AnyValue::is_null() for compatibility with Polars 0.39
            let has_nulls = (0..col.len()).any(|i| match col.get(i) {
                Ok(av) => matches!(av, AnyValue::Null),
                Err(_) => false,
            });

            if !has_nulls {
                required.push(Value::String(col_name.to_string()));
            }

            // For structs, try to infer nested properties
            if let DataType::Struct(fields) = col.dtype() {
                // TODO: Implement recursive inference for structs
                let mut nested_props = serde_json::Map::new();

                for field in fields {
                    // Simple type detection without full recursion
                    let nested_type = match field.data_type() {
                        DataType::Int8
                        | DataType::Int16
                        | DataType::Int32
                        | DataType::Int64
                        | DataType::UInt8
                        | DataType::UInt16
                        | DataType::UInt32
                        | DataType::UInt64 => "integer",
                        DataType::Float32 | DataType::Float64 => "number",
                        DataType::Boolean => "boolean",
                        DataType::String => "string",
                        DataType::Struct(_) => "object",
                        DataType::List(_) => "array",
                        _ => "string",
                    };

                    let mut nested_prop = serde_json::Map::new();
                    nested_prop.insert("type".to_string(), Value::String(nested_type.to_string()));
                    nested_props.insert(field.name().to_string(), Value::Object(nested_prop));
                }

                prop.insert("properties".to_string(), Value::Object(nested_props));
            }

            properties.insert(col_name.to_string(), Value::Object(prop));
        }

        // Build the complete JSON schema
        let mut schema = serde_json::Map::new();
        schema.insert(
            "$schema".to_string(),
            Value::String("http://json-schema.org/draft-07/schema#".to_string()),
        );
        schema.insert("type".to_string(), Value::String("object".to_string()));

        let prop_count = properties.len();
        schema.insert("properties".to_string(), Value::Object(properties));
        schema.insert("required".to_string(), Value::Array(required));

        println!("   Schema inferred with {} properties", prop_count);

        Ok(Value::Object(schema))
    }

    /// Performs incremental validation against a JSON schema.
    ///
    /// Unlike validate_against_schema, this function:
    /// 1. Does not fail at the first error; it collects all
    /// 2. Returns a dataset with the errors found
    /// 3. Includes detailed debugging information
    pub fn validate_incremental(&self, schema: &Value) -> Result<Dataset> {
        use jsonschema::JSONSchema;

        println!("🔍 Performing incremental schema validation");

        // Compile the schema
        let compiled = JSONSchema::compile(schema)
            .map_err(|e| ExecutorError::DataError(format!("Invalid JSON schema: {}", e)))?;

        // Convert dataset to JSON rows
        let rows = self.to_json_rows()?;

        // Columns for the results dataset
        let mut row_indices = Vec::new();
        let mut error_paths = Vec::new();
        let mut error_messages = Vec::new();

        // Validate each row
        for (idx, row) in rows.iter().enumerate() {
            if let Err(errors) = compiled.validate(row) {
                for error in errors {
                    row_indices.push(idx as i64);
                    error_paths.push(error.instance_path.to_string());
                    error_messages.push(error.to_string());
                }
            }
        }

        // Create DataFrame with errors
        let df = DataFrame::new(vec![
            Series::new("row", row_indices),
            Series::new("path", error_paths),
            Series::new("message", error_messages),
        ])
        .map_err(|e| {
            ExecutorError::DataError(format!("Error creating validation results: {}", e))
        })?;

        println!("   Found {} validation errors", df.height());

        // Convert to LazyFrame and create Dataset
        let lazy_df = df.lazy();
        let validation_dataset = Dataset::new("validation_results", lazy_df);

        Ok(validation_dataset)
    }

    /// Returns an iterator of JSON batches of size `batch_size` for streaming
    pub fn stream_json_rows(&self, batch_size: usize) -> DataStream {
        DataStream {
            dataset: Arc::new(self.clone()),
            batch_size,
            offset: 0,
        }
    }
}

// Implement Debug manually because LazyFrame doesn't implement it
impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dataset")
            .field("name", &self.name)
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Iterator that produces dataset rows in batches (streaming)
pub struct DataStream {
    dataset: Arc<Dataset>,
    batch_size: usize,
    offset: usize,
}

// Ensure DataProcessor exposes parquet loader regardless of earlier impl placement
impl DataProcessor {
    /// Loads a parquet file into a Dataset (lazy)
    pub fn load_parquet(&self, name: &str, path: &str) -> Result<Dataset> {
        let lf = LazyFrame::scan_parquet(path, Default::default()).map_err(|e| {
            ExecutorError::DataError(format!("Error scanning parquet {}: {}", path, e))
        })?;
        Ok(Dataset::new(name, lf))
    }
}

impl Iterator for DataStream {
    type Item = Result<Vec<Value>>;

    fn next(&mut self) -> Option<Self::Item> {
        let df_res = self
            .dataset
            .df
            .clone()
            .slice(self.offset as i64, self.batch_size as u32)
            .collect();
        match df_res {
            Ok(df) => {
                if df.height() == 0 {
                    return None;
                }
                // build JSON rows
                let mut rows: Vec<Value> = Vec::with_capacity(df.height());
                let columns = df.get_columns();
                for row_idx in 0..df.height() {
                    let mut map = serde_json::Map::with_capacity(columns.len());
                    for col in columns {
                        let av = col.get(row_idx).unwrap_or(AnyValue::Null);
                        let json_val = self.dataset.anyvalue_to_json(&av);
                        map.insert(col.name().to_string(), json_val);
                    }
                    rows.push(Value::Object(map));
                }
                self.offset += df.height();
                Some(Ok(rows))
            }
            Err(e) => Some(Err(ExecutorError::DataError(format!(
                "Streaming error: {}",
                e
            )))),
        }
    }
}

/// Data processor
pub struct DataProcessor {
    /// Loaded datasets (cached)
    datasets: HashMap<String, Dataset>,
}

impl DataProcessor {
    /// Creates a new data processor
    pub fn new() -> Self {
        DataProcessor {
            datasets: HashMap::new(),
        }
    }
}

// Default implementation for DataProcessor (must be at module scope, not inside impl block)
impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DataProcessor {
    /// Loads a dataset from a source
    pub fn load_data(
        &mut self,
        source: &str,
        options: &HashMap<String, LexLiteral>,
    ) -> Result<Dataset> {
        println!("📊 Loading data from: {}", source);

        // Verify if it's already loaded
        if let Some(cached) = self.datasets.get(source) {
            println!("   Using cached dataset");
            return Ok(cached.clone());
        }

        // Common options for loading
        let has_header = options
            .get("header")
            .map(|lit| match lit {
                LexLiteral::Boolean(b) => *b,
                _ => true, // Assume there is a header by default
            })
            .unwrap_or(true);

        let delimiter = options
            .get("delimiter")
            .map(|lit| match lit {
                LexLiteral::String(s) => s.chars().next().unwrap_or(','),
                _ => ',',
            })
            .unwrap_or(',');

        // Detect streaming option
        let streaming = options
            .get("streaming")
            .map(|lit| matches!(lit, LexLiteral::Boolean(true)))
            .unwrap_or(false);

        // Resolve path (support tests that pass "lexc/tests/..." even when CWD is crate root)
        let mut resolved_path = std::path::PathBuf::from(source);
        if !resolved_path.exists() {
            if let Some(stripped) = source.strip_prefix("lexc/") {
                let alt = std::path::PathBuf::from(stripped);
                if alt.exists() {
                    resolved_path = alt;
                }
            }
        }

        // Determine extension lowercase for branching
        let ext_lower = resolved_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Detect JSON extensions quickly
        let is_json = matches!(ext_lower.as_str(), "json" | "ndjson" | "jsonl");

        // Always return a LazyFrame for consistency
        let lazy_df: LazyFrame = if ext_lower == "csv" {
            if streaming {
                LazyCsvReader::new(resolved_path.to_string_lossy().to_string())
                    .with_separator(delimiter as u8)
                    .has_header(has_header)
                    .finish()
                    .map_err(|e| ExecutorError::DataError(format!("CSV streaming error: {}", e)))?
            } else {
                let file = std::fs::File::open(&resolved_path).map_err(|e| {
                    ExecutorError::DataError(format!(
                        "Error opening file {}: {}",
                        resolved_path.display(),
                        e
                    ))
                })?;
                {
                    let df_eager = CsvReader::new(file)
                        .has_header(has_header)
                        .finish()
                        .map_err(|e| {
                            ExecutorError::DataError(format!("Error reading CSV: {}", e))
                        })?;
                    df_eager.lazy()
                }
            }
        } else if is_json {
            // JSON loading (array of objects or JSON Lines)
            let file = std::fs::File::open(&resolved_path).map_err(|e| {
                ExecutorError::DataError(format!(
                    "Error opening file {}: {}",
                    resolved_path.display(),
                    e
                ))
            })?;

            let json_format = match ext_lower.as_str() {
                "json" => JsonFormat::Json,
                _ => JsonFormat::JsonLines,
            };

            let df_eager = JsonReader::new(file)
                .with_json_format(json_format)
                .finish()
                .map_err(|e| ExecutorError::DataError(format!("Error reading JSON: {}", e)))?;
            df_eager.lazy()
        } else {
            return Err(ExecutorError::DataError(format!(
                "Unsupported file format: {}",
                source
            )));
        };

        // Create the dataset
        let dataset = Dataset::new(source, lazy_df);

        // Show information
        match dataset.collect() {
            Ok(df) => {
                println!("   Loaded {} rows with {} columns", df.height(), df.width());
            }
            Err(e) => {
                println!("   Warning: Could not collect dataframe stats: {}", e);
            }
        }

        // Save to cache
        self.datasets.insert(source.to_string(), dataset.clone());

        Ok(dataset)
    }

    /// Filters a dataset based on a condition
    pub fn filter_data(
        &mut self,
        dataset: Arc<Dataset>,
        condition: RuntimeValue,
    ) -> Result<Dataset> {
        let condition_str = match condition {
            RuntimeValue::String(s) => s,
            _ => {
                return Err(ExecutorError::TypeError(
                    "Expected string condition for filter".to_string(),
                ))
            }
        };

        println!("🔍 Filtering dataset with condition: {}", condition_str);

        // Support composite conditions with simple AND/OR.
        // Example: "age > 30 AND city = Madrid"
        // Also support nested fields with dot notation
        // Example: "contact.address.city = Madrid"

        // Split first by uppercase/lowercase AND/OR keeping delimiters
        let tokens: Vec<&str> = condition_str.split_whitespace().collect();

        // Build expressions
        let mut expr_stack: Vec<Expr> = Vec::new();
        let mut op_stack: Vec<String> = Vec::new();

        let mut i = 0;
        while i < tokens.len() {
            let token = tokens[i];
            match token.to_uppercase().as_str() {
                "AND" | "OR" => {
                    op_stack.push(token.to_uppercase());
                    i += 1;
                }
                _ => {
                    // Expect triple: field operator value
                    if i + 2 >= tokens.len() {
                        return Err(ExecutorError::DataError(format!(
                            "Invalid filter condition near '{}'",
                            token
                        )));
                    }
                    let field = tokens[i];
                    let operator = tokens[i + 1];
                    let value = tokens[i + 2];

                    // Handle nested fields if they contain dots
                    let subexpr = if field.contains('.') {
                        // We have a nested field; use Polars struct syntax
                        let field_parts: Vec<&str> = field.split('.').collect();

                        // Build the expression using struct.field syntax
                        let expr = self.build_nested_field_expr(field_parts)?;

                        match operator {
                            "=" | "==" => expr.eq(lit(value)),
                            "!=" => expr.neq(lit(value)),
                            ">" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    expr.gt(lit(num))
                                } else {
                                    expr.gt(lit(value))
                                }
                            }
                            "<" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    expr.lt(lit(num))
                                } else {
                                    expr.lt(lit(value))
                                }
                            }
                            ">=" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    expr.gt_eq(lit(num))
                                } else {
                                    expr.gt_eq(lit(value))
                                }
                            }
                            "<=" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    expr.lt_eq(lit(num))
                                } else {
                                    expr.lt_eq(lit(value))
                                }
                            }
                            _ => {
                                return Err(ExecutorError::DataError(format!(
                                    "Unsupported operator: {}",
                                    operator
                                )));
                            }
                        }
                    } else {
                        // Regular (non-nested) field
                        match operator {
                            "=" | "==" => col(field).eq(lit(value)),
                            "!=" => col(field).neq(lit(value)),
                            ">" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    col(field).gt(lit(num))
                                } else {
                                    col(field).gt(lit(value))
                                }
                            }
                            "<" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    col(field).lt(lit(num))
                                } else {
                                    col(field).lt(lit(value))
                                }
                            }
                            ">=" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    col(field).gt_eq(lit(num))
                                } else {
                                    col(field).gt_eq(lit(value))
                                }
                            }
                            "<=" => {
                                if let Ok(num) = value.parse::<f64>() {
                                    col(field).lt_eq(lit(num))
                                } else {
                                    col(field).lt_eq(lit(value))
                                }
                            }
                            _ => {
                                return Err(ExecutorError::DataError(format!(
                                    "Unsupported operator: {}",
                                    operator
                                )));
                            }
                        }
                    };

                    expr_stack.push(subexpr);
                    i += 3;
                }
            }
        }

        // Combine expressions following left associativity
        if expr_stack.is_empty() {
            return Err(ExecutorError::DataError("Empty filter expression".into()));
        }

        let mut filter_expr = expr_stack[0].clone();
        let mut expr_index = 1;
        for logic_op in op_stack {
            if expr_index >= expr_stack.len() {
                return Err(ExecutorError::DataError(
                    "Malformed filter expression".into(),
                ));
            }
            let next_expr = expr_stack[expr_index].clone();
            filter_expr = match logic_op.as_str() {
                "AND" => filter_expr.and(next_expr),
                "OR" => filter_expr.or(next_expr),
                _ => unreachable!(),
            };
            expr_index += 1;
        }

        // Apply the filter
        let filtered_df = dataset.df.clone().filter(filter_expr);

        // Create a new dataset
        let filtered_name = format!("{}_filtered", dataset.name);
        let filtered_dataset = Dataset::new(&filtered_name, filtered_df);

        println!("   Filter applied. Execution deferred (lazy evaluation)");

        Ok(filtered_dataset)
    }

    /// Builds an expression to access a nested field
    fn build_nested_field_expr(&self, field_parts: Vec<&str>) -> Result<Expr> {
        if field_parts.is_empty() {
            return Err(ExecutorError::DataError("Empty field name".into()));
        }

        // The first element is the main column
        let mut expr = col(field_parts[0]);

        // If there are more parts, they are nested field accesses
        for part in field_parts.iter().skip(1) {
            // Use the struct operator to access nested fields
            // Since Polars 0.39, accessing struct fields is done through
            // the `struct_()` namespace followed by `field_by_name()`.
            expr = expr.struct_().field_by_name(part);
        }

        Ok(expr)
    }

    /// Selects specific fields
    pub fn select_fields(&mut self, dataset: Arc<Dataset>, fields: &[String]) -> Result<Dataset> {
        println!("🔍 Selecting fields: {:?}", fields);

        // Convert field names to Polars columns
        let columns: Vec<Expr> = fields
            .iter()
            .map(|field| {
                if field.contains('.') {
                    // Nested field
                    let field_parts: Vec<&str> = field.split('.').collect();
                    // Use alias to rename the resulting column
                    match self.build_nested_field_expr(field_parts.clone()) {
                        Ok(expr) => expr.alias(field),
                        Err(e) => {
                            // If there is an error, return an expression that won't fail
                            // but will produce an error later
                            println!("Warning: Error in nested field '{}': {:?}", field, e);
                            col("__error_column__").alias(field)
                        }
                    }
                } else {
                    // Simple field
                    col(field)
                }
            })
            .collect();

        // Apply selection
        let selected_df = dataset.df.clone().select(columns);

        // Create a new dataset
        let selected_name = format!("{}_selected", dataset.name);
        let selected_dataset = Dataset::new(&selected_name, selected_df);

        println!("   Selection applied. Execution deferred (lazy evaluation)");

        Ok(selected_dataset)
    }

    /// Takes a limited number of rows
    pub fn take_rows(&mut self, dataset: Arc<Dataset>, count: usize) -> Result<Dataset> {
        println!("🔍 Taking {} rows from dataset", count);

        // Apply the limit
        let limited_df = dataset.df.clone().limit(count as u32);

        // Create a new dataset
        let limited_name = format!("{}_take_{}", dataset.name, count);
        let limited_dataset = Dataset::new(&limited_name, limited_df);

        println!("   Limit applied. Execution deferred (lazy evaluation)");

        Ok(limited_dataset)
    }

    /// Exports a dataset to a file
    pub fn export_data(&mut self, dataset: Arc<Dataset>, path: &str, format: &str) -> Result<()> {
        dataset.export(path, format)
    }

    /// Flattens a nested JSON dataset to facilitate access to all fields
    pub fn flatten_json(
        &mut self,
        dataset: Arc<Dataset>,
        separator: Option<&str>,
    ) -> Result<Dataset> {
        println!("🔍 Flattening nested JSON structures");

        // Use dot as the default separator
        let sep = separator.unwrap_or(".");

        // First materialize the dataframe to work with columns
        let df =
            dataset.df.clone().collect().map_err(|e| {
                ExecutorError::DataError(format!("Error collecting dataframe: {}", e))
            })?;

        // List of expressions to flatten
        let mut flatten_exprs: Vec<Expr> = Vec::new();

        // For each column in the dataframe
        for col_name in df.get_column_names() {
            let series = df.column(col_name).map_err(|e| {
                ExecutorError::DataError(format!("Error getting column {}: {}", col_name, e))
            })?;

            match series.dtype() {
                // If the data type is struct, flatten
                DataType::Struct(_) => {
                    println!("   Flattening struct column: {}", col_name);
                    // Instead of unnest, we use a more direct approach.
                    // `alias` expects &str, so we pass a reference to the temporary String.
                    let alias_name = format!("{}{}flattened", col_name, sep);
                    flatten_exprs.push(col(col_name).alias(&alias_name));
                }
                // If it's a list, we might implement specific behavior in the future
                DataType::List(_) => {
                    println!("   Keeping list column as is: {}", col_name);
                    flatten_exprs.push(col(col_name));
                }
                // For other types, keep as is
                _ => {
                    flatten_exprs.push(col(col_name));
                }
            }
        }

        // Create a new dataframe with the flattened expressions
        let flattened_df = df.clone().lazy().select(flatten_exprs);

        // Create a new dataset
        let flattened_name = format!("{}_flattened", dataset.name);
        let flattened_dataset = Dataset::new(&flattened_name, flattened_df);

        println!("   Flattening applied. Execution deferred (lazy evaluation)");

        Ok(flattened_dataset)
    }

    /// Filters rows in the dataset using a JSONPath expression. Keeps any row where the expression returns at least one result.
    pub fn filter_jsonpath(&mut self, dataset: Arc<Dataset>, path: &str) -> Result<Dataset> {
        println!("🔍 Filtering dataset with JSONPath: {}", path);

        // Materialize the dataframe to process row by row
        let df = dataset
            .collect()
            .map_err(|e| ExecutorError::DataError(format!("Error collecting dataframe: {}", e)))?;

        // Convert each row to Value and evaluate JSONPath
        let rows_json = dataset.to_json_rows()?;

        // Create boolean mask by evaluating JSONPath per row
        let mut mask: Vec<bool> = Vec::with_capacity(rows_json.len());
        for row in &rows_json {
            match jsonpath_lib::select(row, path) {
                Ok(matches) => mask.push(!matches.is_empty()),
                Err(e) => {
                    return Err(ExecutorError::DataError(format!(
                        "JSONPath evaluation error: {}",
                        e
                    )))
                }
            }
        }

        let bool_chunk = BooleanChunked::from_slice("mask", &mask);

        let filtered_df = df
            .filter(&bool_chunk)
            .map_err(|e| ExecutorError::DataError(format!("Error filtering dataframe: {}", e)))?;

        let filtered_lazy = filtered_df.lazy();
        let filtered_name = format!("{}_jp_filtered", dataset.name);
        let filtered_dataset = Dataset::new(&filtered_name, filtered_lazy);

        Ok(filtered_dataset)
    }

    /// Selects values using JSONPath and returns a new Dataset with a "result" column (string)
    pub fn select_jsonpath(&mut self, dataset: Arc<Dataset>, path: &str) -> Result<Dataset> {
        println!("🔍 Selecting values with JSONPath: {}", path);

        // If the dataset name points to a JSON file, try to read it
        let path_obj = std::path::Path::new(&dataset.name);
        let root: serde_json::Value = if path_obj.exists() && path_obj.extension().is_some() {
            match path_obj
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_lowercase()
                .as_str()
            {
                "json" => {
                    let content = std::fs::read_to_string(path_obj).map_err(|e| {
                        ExecutorError::DataError(format!("Error reading JSON file: {}", e))
                    })?;
                    serde_json::from_str(&content).map_err(|e| {
                        ExecutorError::DataError(format!("Error parsing JSON: {}", e))
                    })?
                }
                "ndjson" | "jsonl" => {
                    let content = std::fs::read_to_string(path_obj).map_err(|e| {
                        ExecutorError::DataError(format!("Error reading NDJSON file: {}", e))
                    })?;
                    let mut arr = Vec::new();
                    for line in content.lines() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        let val: serde_json::Value = serde_json::from_str(line).map_err(|e| {
                            ExecutorError::DataError(format!("Error parsing NDJSON line: {}", e))
                        })?;
                        arr.push(val);
                    }
                    serde_json::Value::Array(arr)
                }
                _ => serde_json::Value::Array(dataset.to_json_rows()?),
            }
        } else {
            serde_json::Value::Array(dataset.to_json_rows()?)
        };

        let mut matches: Vec<String> = Vec::new();
        match jsonpath_lib::select(&root, path) {
            Ok(vals) => {
                for v in vals {
                    let s = serde_json::to_string(v).unwrap_or_else(|_| "null".to_string());
                    matches.push(s);
                }
            }
            Err(e) => {
                return Err(ExecutorError::DataError(format!(
                    "JSONPath evaluation error: {}",
                    e
                )))
            }
        }

        // Build DataFrame
        let series = Series::new("result", matches);
        let df = DataFrame::new(vec![series])
            .map_err(|e| ExecutorError::DataError(format!("Error building DataFrame: {}", e)))?;

        let lazy = df.lazy();
        let new_name = format!("{}_jp_select", dataset.name);
        Ok(Dataset::new(&new_name, lazy))
    }
}
