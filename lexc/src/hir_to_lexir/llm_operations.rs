//! LLM Operations Conversion
//!
//! This module handles the conversion of HIR LLM-related operations to LexIR instructions.
//! It supports both basic ask expressions and advanced ask_safe expressions with
//! anti-hallucination validation features.
//!
//! ## Supported Operations
//!
//! - **Ask**: Basic LLM queries with model, temperature, and schema parameters
//! - **Ask Safe**: Advanced queries with anti-hallucination validation
//!
//! ## Ask Safe Features
//!
//! The ask_safe operation includes sophisticated validation:
//! - Multiple validation strategies (basic, ensemble, fact_check, comprehensive)
//! - Confidence thresholds and retry logic
//! - Cross-reference model validation
//! - External fact-checking integration
//!
//! ## Attribute Processing
//!
//! Both operations support flexible attribute processing:
//! - Model selection and configuration
//! - Temperature and token limits
//! - Output schema validation
//! - Custom parameters and flags
//!
//! ## Error Handling
//!
//! All operations return `Result<()>` to handle attribute processing
//! and instruction generation failures gracefully.

use crate::hir::{HirAskExpression};
use crate::lexir::{LexInstruction, ValueRef};
use super::{Result, ConversionContext};
use std::collections::HashMap;
use crate::lexir::LexExpression;
use crate::lexir::LexLiteral;

impl ConversionContext {
    /// 🤖 Converts HIR ask expressions to LexIR ask instructions
    ///
    /// This method handles the conversion of LLM ask expressions, including:
    /// - Attribute processing and extraction
    /// - Model and temperature parameter handling
    /// - Schema validation setup
    /// - System and user prompt configuration
    ///
    /// ## Attribute Processing
    ///
    /// The method extracts common LLM parameters:
    /// - `model`: Specific LLM model to use
    /// - `temperature`: Response randomness control
    /// - `max_tokens`: Maximum response length
    /// - `schema`: Output format validation
    ///
    /// ## Debug Information
    ///
    /// Includes debug output for troubleshooting attribute extraction and processing.
    pub fn add_ask_instruction(&mut self, ask: &HirAskExpression, result: ValueRef) -> Result<()> {
        let mut attributes = HashMap::new();

        // Convert attributes from HIR to HashMap
        for attr in &ask.attributes {
            if let Some(value) = &attr.value {
                attributes.insert(attr.name.clone(), value.clone());
            } else {
                attributes.insert(attr.name.clone(), "true".to_string());
            }
        }

        println!("🔍 DEBUG HIR->LEXIR: attributes before extraction: {:?}", attributes);

        // Extract LLM parameters from attributes
        let model = attributes.remove("model");
        let temperature = attributes.get("temperature").and_then(|t| t.parse::<f64>().ok());

        println!("🔍 DEBUG HIR->LEXIR: extracted model: {:?}, temperature: {:?}", model, temperature);

        let max_tokens = attributes.get("max_tokens").and_then(|t| t.parse::<u32>().ok());

        // Convert user_prompt to LexExpression if present
        let user_prompt_expr = if let Some(prompt_node) = &ask.user_prompt {
            // Convert the HirNode to LexExpression
            Some(self.convert_node_to_lex_expression(prompt_node)?)
        } else {
            None
        };

        // Create the Ask instruction with all parameters
        let instruction = LexInstruction::Ask {
            result,
            system_prompt: ask.system_prompt.clone(),
            user_prompt: user_prompt_expr,
            model,
            temperature,
            max_tokens,
            schema: ask.output_schema_name.clone(),
            attributes,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }

    /// 🛡️ Converts HIR ask_safe expressions to LexIR anti-hallucination instructions
    ///
    /// This method handles the conversion of ask_safe expressions with advanced anti-hallucination
    /// validation features, including:
    /// - Basic LLM parameter extraction (model, temperature, max_tokens)
    /// - Anti-hallucination validation strategy configuration
    /// - Confidence threshold and retry logic setup
    /// - Cross-reference model validation
    /// - Fact-checking integration
    ///
    /// ## Anti-Hallucination Features
    ///
    /// The method extracts specialized validation parameters:
    /// - `validation_strategy`: Type of validation (basic, ensemble, fact_check, comprehensive)
    /// - `confidence_threshold`: Minimum confidence score required (0.0-1.0)
    /// - `max_attempts`: Maximum retry attempts for low-confidence responses
    /// - `cross_reference_models`: List of models for cross-validation
    /// - `use_fact_checking`: Enable external fact-checking services
    ///
    /// ## Validation Strategies
    ///
    /// - **Basic**: Simple confidence scoring
    /// - **Ensemble**: Multi-model consensus validation
    /// - **Fact Check**: External fact verification
    /// - **Comprehensive**: All validation methods combined
    ///
    /// ## Debug Information
    ///
    /// Includes comprehensive debug output for troubleshooting validation setup.
    pub fn add_ask_safe_instruction(&mut self, ask_safe: &crate::hir::HirAskSafeExpression, result: ValueRef) -> Result<()> {
        let mut attributes = HashMap::new();

        // Convert basic attributes from HIR to HashMap
        for attr in &ask_safe.attributes {
            if let Some(value) = &attr.value {
                attributes.insert(attr.name.clone(), value.clone());
            } else {
                attributes.insert(attr.name.clone(), "true".to_string());
            }
        }

        println!("🛡️ DEBUG HIR->LEXIR: ask_safe attributes before extraction: {:?}", attributes);

        // Extract basic LLM parameters
        let model = attributes.remove("model");
        let temperature = attributes.get("temperature").and_then(|t| t.parse::<f64>().ok());
        let max_tokens = attributes.get("max_tokens").and_then(|t| t.parse::<u32>().ok());

        // Extract anti-hallucination validation attributes
        let validation_strategy = ask_safe.validation_strategy.clone()
            .or_else(|| attributes.remove("validation"));
        let confidence_threshold = ask_safe.confidence_threshold
            .or_else(|| attributes.get("confidence_threshold").and_then(|t| t.parse::<f64>().ok()));
        let max_attempts = ask_safe.max_attempts
            .or_else(|| attributes.get("max_attempts").and_then(|t| t.parse::<u32>().ok()));
        let use_fact_checking = attributes.get("use_fact_checking")
            .and_then(|t| t.parse::<bool>().ok());

        // Extract cross-reference models for ensemble validation
        let mut cross_reference_models = ask_safe.cross_reference_models.clone();
        if cross_reference_models.is_empty() {
            if let Some(models_str) = attributes.remove("cross_reference_models") {
                cross_reference_models = models_str.split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }

        println!("🛡️ DEBUG HIR->LEXIR: validation_strategy: {:?}, confidence_threshold: {:?}, max_attempts: {:?}",
                validation_strategy, confidence_threshold, max_attempts);
        println!("🛡️ DEBUG HIR->LEXIR: cross_reference_models: {:?}", cross_reference_models);

        // Create the AskSafe instruction with comprehensive anti-hallucination validation
        let instruction = LexInstruction::AskSafe {
            result,
            system_prompt: ask_safe.system_prompt.clone(),
            user_prompt: ask_safe.user_prompt.clone(),
            model,
            temperature,
            max_tokens,
            schema: ask_safe.output_schema_name.clone(),
            attributes,
            validation_strategy,
            confidence_threshold,
            max_attempts,
            cross_reference_models,
            use_fact_checking,
        };

        self.program.add_instruction(instruction);
        Ok(())
    }
}