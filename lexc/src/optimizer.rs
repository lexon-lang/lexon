// lexc/src/optimizer.rs
//
// LexIR Optimizer
// This module implements optimizations in the LexIR intermediate representation

use crate::lexir::{
    LexBinaryOperator, LexExpression, LexInstruction, LexLiteral, LexProgram, LexUnaryOperator,
    ValueRef,
};
use std::collections::{HashMap, HashSet};

/// Error during optimization
#[derive(Debug)]
pub enum OptimizerError {
    UnsupportedOptimization(String),
}

pub type Result<T> = std::result::Result<T, OptimizerError>;

/// Available optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Optimization {
    /// Eliminates dead code (instructions whose result is not used)
    DeadCodeElimination,
    /// Performs constant propagation
    ConstantPropagation,
    /// Evaluates constant operations at compile time
    ConstantFolding,
    /// Fuses data operations
    DataOperationFusion,
    /// Eliminates redundant assignments
    RedundantAssignmentElimination,
    /// Inlines small free functions
    InlineFunction,
}

/// Optimizer configuration
pub struct OptimizerConfig {
    /// Enabled optimizations
    pub enabled_optimizations: Vec<Optimization>,
    /// Verbosity level for optimization logs
    pub verbose: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enabled_optimizations: vec![
                Optimization::DeadCodeElimination,
                Optimization::ConstantPropagation,
                Optimization::ConstantFolding,
                Optimization::DataOperationFusion,
                Optimization::RedundantAssignmentElimination,
                Optimization::InlineFunction,
            ],
            verbose: false,
        }
    }
}

/// Optimizer context that maintains information between passes
struct OptimizerContext {
    /// Live variable analysis
    live_vars: HashSet<ValueRef>,
    /// Constant propagation map
    constant_map: HashMap<ValueRef, ValueRef>,
    /// Counter of applied optimizations
    optimization_count: HashMap<Optimization, usize>,
    /// Verbose mode
    verbose: bool,
}

impl OptimizerContext {
    fn new(verbose: bool) -> Self {
        Self {
            live_vars: HashSet::new(),
            constant_map: HashMap::new(),
            optimization_count: HashMap::new(),
            verbose,
        }
    }

    fn record_optimization(&mut self, opt: Optimization) {
        *self.optimization_count.entry(opt).or_insert(0) += 1;
        if self.verbose {
            println!("Applied optimization: {:?}", opt);
        }
    }

    fn get_stats(&self) -> HashMap<Optimization, usize> {
        self.optimization_count.clone()
    }
}

/// Optimizer for LexIR programs
pub struct Optimizer {
    config: OptimizerConfig,
}

impl Optimizer {
    pub fn new(config: OptimizerConfig) -> Self {
        Self { config }
    }

    /// Optimizes a LexIR program by applying configured optimizations
    pub fn optimize(&self, program: &mut LexProgram) -> Result<HashMap<Optimization, usize>> {
        let mut context = OptimizerContext::new(self.config.verbose);

        // Apply optimizations according to configuration
        for &optimization in &self.config.enabled_optimizations {
            match optimization {
                Optimization::DeadCodeElimination => {
                    self.eliminate_dead_code(program, &mut context)?;
                }
                Optimization::ConstantPropagation => {
                    self.propagate_constants(program, &mut context)?;
                }
                Optimization::ConstantFolding => {
                    self.constant_folding(program, &mut context)?;
                }
                Optimization::DataOperationFusion => {
                    self.fuse_data_operations(program, &mut context)?;
                }
                Optimization::RedundantAssignmentElimination => {
                    self.eliminate_redundant_assignments(program, &mut context)?;
                }
                Optimization::InlineFunction => {
                    self.inline_functions(program, &mut context)?;
                }
            }
        }

        Ok(context.get_stats())
    }

    /// Eliminates dead code (instructions whose result is not used)
    fn eliminate_dead_code(
        &self,
        program: &mut LexProgram,
        context: &mut OptimizerContext,
    ) -> Result<()> {
        // First, identify all variables that are used
        self.analyze_live_variables(program, context);

        // Then, filter instructions to keep only those that produce live variables
        let mut live_instructions = Vec::new();

        for instruction in &program.instructions {
            match instruction {
                LexInstruction::Assign { result, .. } => {
                    if context.live_vars.contains(result) {
                        live_instructions.push(instruction.clone());
                    } else {
                        context.record_optimization(Optimization::DeadCodeElimination);
                    }
                }
                // Always keep instructions that have side effects
                LexInstruction::DataExport { .. } | LexInstruction::Call { .. } => {
                    live_instructions.push(instruction.clone());
                }
                // For other instructions, check if the result is used
                LexInstruction::DataLoad { result, .. }
                | LexInstruction::DataFilter { result, .. }
                | LexInstruction::DataSelect { result, .. }
                | LexInstruction::DataTake { result, .. }
                | LexInstruction::DataFlatten { result, .. }
                | LexInstruction::DataFilterJsonPath { result, .. }
                | LexInstruction::DataInferSchema { result, .. }
                | LexInstruction::DataValidateIncremental { result, .. } => {
                    if context.live_vars.contains(result) {
                        live_instructions.push(instruction.clone());
                    } else {
                        context.record_optimization(Optimization::DeadCodeElimination);
                    }
                }
                // Keep all other instructions
                _ => live_instructions.push(instruction.clone()),
            }
        }

        // Update the program with live instructions
        program.instructions = live_instructions;

        Ok(())
    }

    /// Analyzes live variables in the program
    fn analyze_live_variables(&self, program: &LexProgram, context: &mut OptimizerContext) {
        // First, identify variables used in instructions with side effects
        for instruction in &program.instructions {
            match instruction {
                LexInstruction::DataExport { input, .. } => {
                    context.live_vars.insert(input.clone());
                }
                LexInstruction::Call { args, .. } => {
                    for arg in args {
                        self.add_live_vars_from_expression(arg, context);
                    }
                }
                _ => {}
            }
        }

        // Then, propagate backwards to find all live variables
        let mut changed = true;
        while changed {
            changed = false;

            for instruction in &program.instructions {
                match instruction {
                    LexInstruction::Assign { result, expr } => {
                        if context.live_vars.contains(result) {
                            changed |= self.add_live_vars_from_expression(expr, context);
                        }
                    }
                    LexInstruction::DataFilter {
                        result,
                        input,
                        predicate,
                    } => {
                        if context.live_vars.contains(result) {
                            changed |= context.live_vars.insert(input.clone());
                            changed |= self.add_live_vars_from_expression(predicate, context);
                        }
                    }
                    LexInstruction::DataSelect { result, input, .. }
                    | LexInstruction::DataTake { result, input, .. }
                    | LexInstruction::DataFlatten { result, input, .. }
                    | LexInstruction::DataFilterJsonPath { result, input, .. }
                    | LexInstruction::DataInferSchema { result, input, .. } => {
                        if context.live_vars.contains(result) {
                            changed |= context.live_vars.insert(input.clone());
                        }
                    }
                    LexInstruction::DataValidateIncremental {
                        result,
                        input,
                        schema,
                    } => {
                        if context.live_vars.contains(result) {
                            changed |= context.live_vars.insert(input.clone());
                            changed |= context.live_vars.insert(schema.clone());
                        }
                    }
                    LexInstruction::If { condition, .. } => {
                        // Conditions are always used
                        changed |= self.add_live_vars_from_expression(condition, context);
                    }
                    LexInstruction::ForEach {
                        iterator: _,
                        iterable,
                        body: _,
                    } => {
                        changed |= context.live_vars.insert(iterable.clone());
                    }
                    _ => {}
                }
            }
        }
    }

    /// Adds live variables from an expression
    #[allow(clippy::only_used_in_recursion)]
    fn add_live_vars_from_expression(
        &self,
        expr: &LexExpression,
        context: &mut OptimizerContext,
    ) -> bool {
        let mut changed = false;

        match expr {
            LexExpression::Value(ValueRef::Temp(_))
            | LexExpression::Value(ValueRef::Literal(_))
            | LexExpression::Value(ValueRef::Named(_)) => {}
            LexExpression::BinaryOp { left, right, .. } => {
                changed |= self.add_live_vars_from_expression(left, context);
                changed |= self.add_live_vars_from_expression(right, context);
            }
            LexExpression::UnaryOp { operand, .. } => {
                changed |= self.add_live_vars_from_expression(operand, context);
            }
            LexExpression::FieldAccess { base, .. } => {
                changed |= self.add_live_vars_from_expression(base, context);
            }
        }

        changed
    }

    /// Performs constant propagation
    #[allow(clippy::collapsible_match)]
    fn propagate_constants(
        &self,
        program: &mut LexProgram,
        context: &mut OptimizerContext,
    ) -> Result<()> {
        // Identify constants
        for instruction in &program.instructions {
            if let LexInstruction::Assign { result, expr } = instruction {
                if let LexExpression::Value(ValueRef::Literal(lit)) = expr {
                    context
                        .constant_map
                        .insert(result.clone(), ValueRef::Literal(lit.clone()));
                    context.record_optimization(Optimization::ConstantPropagation);
                }
            }
        }

        // Propagate constants in all expressions
        let mut new_instructions = Vec::new();

        for instruction in &program.instructions {
            let new_instruction = match instruction {
                LexInstruction::Assign { result, expr } => {
                    let new_expr = self.propagate_in_expression(expr, context);
                    LexInstruction::Assign {
                        result: result.clone(),
                        expr: new_expr,
                    }
                }
                LexInstruction::DataFilter {
                    result,
                    input,
                    predicate,
                } => LexInstruction::DataFilter {
                    result: result.clone(),
                    input: self.propagate_in_value_ref(input, context),
                    predicate: self.propagate_in_expression(predicate, context),
                },
                LexInstruction::DataSelect {
                    result,
                    input,
                    fields,
                } => LexInstruction::DataSelect {
                    result: result.clone(),
                    input: self.propagate_in_value_ref(input, context),
                    fields: fields.clone(),
                },
                LexInstruction::DataTake {
                    result,
                    input,
                    count,
                } => LexInstruction::DataTake {
                    result: result.clone(),
                    input: self.propagate_in_value_ref(input, context),
                    count: self.propagate_in_value_ref(count, context),
                },
                LexInstruction::ForEach {
                    iterator,
                    iterable,
                    body,
                } => LexInstruction::ForEach {
                    iterator: iterator.clone(),
                    iterable: self.propagate_in_value_ref(iterable, context),
                    body: body.clone(),
                },
                // Propagate in other instructions as needed
                _ => instruction.clone(),
            };

            new_instructions.push(new_instruction);
        }

        program.instructions = new_instructions;

        Ok(())
    }

    /// Propagates constants in an expression
    fn propagate_in_expression(
        &self,
        expr: &LexExpression,
        context: &mut OptimizerContext,
    ) -> LexExpression {
        match expr {
            LexExpression::Value(value_ref) => {
                LexExpression::Value(self.propagate_in_value_ref(value_ref, context))
            }
            LexExpression::BinaryOp {
                operator,
                left,
                right,
            } => {
                // Get the propagated subexpressions
                let new_left = self.propagate_in_expression(left, context);
                let new_right = self.propagate_in_expression(right, context);

                LexExpression::BinaryOp {
                    operator: *operator,
                    left: Box::new(new_left),
                    right: Box::new(new_right),
                }
            }
            LexExpression::UnaryOp { operator, operand } => LexExpression::UnaryOp {
                operator: *operator,
                operand: Box::new(self.propagate_in_expression(operand, context)),
            },
            LexExpression::FieldAccess { base, field } => LexExpression::FieldAccess {
                base: Box::new(self.propagate_in_expression(base, context)),
                field: field.clone(),
            },
        }
    }

    /// Propagates constants in a value reference
    fn propagate_in_value_ref(
        &self,
        value_ref: &ValueRef,
        context: &mut OptimizerContext,
    ) -> ValueRef {
        if let Some(constant) = context.constant_map.get(value_ref).cloned() {
            context.record_optimization(Optimization::ConstantPropagation);
            constant
        } else {
            value_ref.clone()
        }
    }

    /// Fuses sequential data operations to reduce intermediate operations
    fn fuse_data_operations(
        &self,
        program: &mut LexProgram,
        context: &mut OptimizerContext,
    ) -> Result<()> {
        // Value usage map: which instruction uses each value
        let mut value_uses: HashMap<ValueRef, Vec<usize>> = HashMap::new();

        // Build usage map
        for (idx, instruction) in program.instructions.iter().enumerate() {
            match instruction {
                LexInstruction::DataFilter { input, .. }
                | LexInstruction::DataSelect { input, .. }
                | LexInstruction::DataTake { input, .. }
                | LexInstruction::DataFlatten { input, .. }
                | LexInstruction::DataFilterJsonPath { input, .. }
                | LexInstruction::DataInferSchema { input, .. }
                | LexInstruction::DataValidateIncremental { input, .. }
                | LexInstruction::DataExport { input, .. } => {
                    value_uses.entry(input.clone()).or_default().push(idx);
                }
                _ => {}
            }

            // Process special cases for other fields
            if let LexInstruction::DataTake { count, .. } = instruction {
                value_uses.entry(count.clone()).or_default().push(idx);
            }

            if let LexInstruction::DataValidateIncremental { schema, .. } = instruction {
                value_uses.entry(schema.clone()).or_default().push(idx);
            }
        }

        // Try to fuse operations
        let mut fused_instructions = program.instructions.clone();
        let mut removed_indices = HashSet::new();

        #[allow(clippy::needless_range_loop)]
        for i in 0..program.instructions.len() {
            if removed_indices.contains(&i) {
                continue;
            }

            // Only try to fuse if the result is used only once
            if let LexInstruction::DataFilter {
                result: res1,
                input: _in1,
                ..
            }
            | LexInstruction::DataSelect {
                result: res1,
                input: _in1,
                ..
            }
            | LexInstruction::DataTake {
                result: res1,
                input: _in1,
                ..
            } = &program.instructions[i]
            {
                if let Some(uses) = value_uses.get(res1) {
                    if uses.len() == 1 {
                        let next_idx = uses[0];
                        if next_idx > i && !removed_indices.contains(&next_idx) {
                            // Try to fuse with the next operation
                            if let Some(fused) = self.try_fuse_operations(
                                &program.instructions[i],
                                &program.instructions[next_idx],
                                res1.clone(),
                            ) {
                                fused_instructions[i] = fused;
                                removed_indices.insert(next_idx);
                                context.record_optimization(Optimization::DataOperationFusion);
                            }
                        }
                    }
                }
            }
        }

        // Rebuild the program without the removed instructions
        program.instructions = fused_instructions
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !removed_indices.contains(idx))
            .map(|(_, instr)| instr)
            .collect();

        Ok(())
    }

    /// Attempts to fuse two data operations
    fn try_fuse_operations(
        &self,
        first: &LexInstruction,
        second: &LexInstruction,
        intermediate: ValueRef,
    ) -> Option<LexInstruction> {
        match (first, second) {
            // Fuse filter + filter
            (
                LexInstruction::DataFilter {
                    result: _,
                    input,
                    predicate: pred1,
                },
                LexInstruction::DataFilter {
                    result,
                    predicate: pred2,
                    ..
                },
            ) => {
                // Create a new predicate combining both with AND
                let new_predicate = LexExpression::BinaryOp {
                    operator: crate::lexir::LexBinaryOperator::And,
                    left: Box::new(pred1.clone()),
                    right: Box::new(pred2.clone()),
                };

                Some(LexInstruction::DataFilter {
                    result: result.clone(),
                    input: input.clone(),
                    predicate: new_predicate,
                })
            }

            // Fuse select + select
            (
                LexInstruction::DataSelect {
                    result: _,
                    input,
                    fields: fields1,
                },
                LexInstruction::DataSelect {
                    result,
                    fields: fields2,
                    ..
                },
            ) => {
                // Create a new set of fields that is the intersection
                let final_fields: Vec<String> = fields2
                    .iter()
                    .filter(|f| fields1.contains(f))
                    .cloned()
                    .collect();

                Some(LexInstruction::DataSelect {
                    result: result.clone(),
                    input: input.clone(),
                    fields: final_fields,
                })
            }

            // Fuse select + filter
            (
                LexInstruction::DataSelect {
                    result: _,
                    input,
                    fields: _,
                },
                LexInstruction::DataFilter {
                    result: _,
                    predicate,
                    ..
                },
            ) => {
                // Filter first, then select
                Some(LexInstruction::DataFilter {
                    result: intermediate.clone(), // Temporal
                    input: input.clone(),
                    predicate: predicate.clone(),
                })

                // NOTE: This merge is more complex and requires two instructions,
                // so we can't do it directly here.
                // In a more complete implementation, we would need to handle
                // instruction sequences.
            }

            // Not fusable
            _ => None,
        }
    }

    /// Eliminates redundant assignments (a = b; c = a; -> c = b;)
    fn eliminate_redundant_assignments(
        &self,
        program: &mut LexProgram,
        context: &mut OptimizerContext,
    ) -> Result<()> {
        // Map of simple assignments: a = b
        let mut simple_assignments: HashMap<ValueRef, ValueRef> = HashMap::new();

        // Collect simple assignments
        for instruction in &program.instructions {
            if let LexInstruction::Assign {
                result,
                expr: LexExpression::Value(value_ref),
            } = instruction
            {
                simple_assignments.insert(result.clone(), value_ref.clone());
            }
        }

        // Find original values (complete resolution of assignment chain)
        let mut resolved_values = HashMap::new();
        for key in simple_assignments.keys() {
            let original = self.resolve_to_original_value(key, &simple_assignments);
            if original != *key {
                resolved_values.insert(key.clone(), original);
                context.record_optimization(Optimization::RedundantAssignmentElimination);
            }
        }

        // Replace references in all instructions
        let mut new_instructions = Vec::new();

        for instruction in &program.instructions {
            match instruction {
                LexInstruction::Assign { result, expr } => {
                    // If it's a simple assignment (a = b), and b has an original value,
                    // we can replace it directly
                    if let LexExpression::Value(value_ref) = expr {
                        if let Some(original) = resolved_values.get(value_ref) {
                            let new_expr = LexExpression::Value(original.clone());
                            new_instructions.push(LexInstruction::Assign {
                                result: result.clone(),
                                expr: new_expr,
                            });
                            continue;
                        }
                    }

                    // If it's not a simple assignment, process the expression
                    let new_expr = self.replace_refs_in_expression(expr, &resolved_values);
                    new_instructions.push(LexInstruction::Assign {
                        result: result.clone(),
                        expr: new_expr,
                    });
                }
                // Process other instructions as needed
                _ => new_instructions.push(instruction.clone()),
            }
        }

        program.instructions = new_instructions;

        Ok(())
    }

    /// Resolves a value to its original reference in the assignment chain
    fn resolve_to_original_value(
        &self,
        value_ref: &ValueRef,
        simple_assignments: &HashMap<ValueRef, ValueRef>,
    ) -> ValueRef {
        // If it's already a literal, no need to follow the chain
        if let ValueRef::Literal(_) = value_ref {
            return value_ref.clone();
        }

        let mut current = value_ref;
        let mut visited = HashSet::new();

        // Follow the assignment chain until the original value is found
        // or a cycle is detected
        while let Some(next) = simple_assignments.get(current) {
            // If next is a literal, return that value
            if let ValueRef::Literal(_) = next {
                return next.clone();
            }

            if visited.contains(next) {
                // Cycle detected, stop
                break;
            }
            visited.insert(next);
            current = next;
        }

        current.clone()
    }

    /// Replaces references in an expression
    #[allow(clippy::only_used_in_recursion)]
    fn replace_refs_in_expression(
        &self,
        expr: &LexExpression,
        resolved_values: &HashMap<ValueRef, ValueRef>,
    ) -> LexExpression {
        match expr {
            LexExpression::Value(value_ref) => {
                if let Some(original) = resolved_values.get(value_ref) {
                    LexExpression::Value(original.clone())
                } else {
                    expr.clone()
                }
            }
            LexExpression::BinaryOp {
                operator,
                left,
                right,
            } => LexExpression::BinaryOp {
                operator: *operator,
                left: Box::new(self.replace_refs_in_expression(left, resolved_values)),
                right: Box::new(self.replace_refs_in_expression(right, resolved_values)),
            },
            LexExpression::UnaryOp { operator, operand } => LexExpression::UnaryOp {
                operator: *operator,
                operand: Box::new(self.replace_refs_in_expression(operand, resolved_values)),
            },
            LexExpression::FieldAccess { base, field } => LexExpression::FieldAccess {
                base: Box::new(self.replace_refs_in_expression(base, resolved_values)),
                field: field.clone(),
            },
        }
    }

    /// Evaluates binary/unary expressions with literals at compile time
    fn constant_folding(
        &self,
        program: &mut LexProgram,
        context: &mut OptimizerContext,
    ) -> Result<()> {
        fn fold_expr(expr: &LexExpression) -> LexExpression {
            match expr {
                LexExpression::BinaryOp {
                    operator,
                    left,
                    right,
                } => {
                    let l_fold = fold_expr(left);
                    let r_fold = fold_expr(right);
                    if let (
                        LexExpression::Value(ValueRef::Literal(lit_l)),
                        LexExpression::Value(ValueRef::Literal(lit_r)),
                    ) = (&l_fold, &r_fold)
                    {
                        if let Some(result) = eval_binary(*operator, lit_l.clone(), lit_r.clone()) {
                            return LexExpression::Value(ValueRef::Literal(result));
                        }
                    }
                    LexExpression::BinaryOp {
                        operator: *operator,
                        left: Box::new(l_fold),
                        right: Box::new(r_fold),
                    }
                }
                LexExpression::UnaryOp { operator, operand } => {
                    let op_fold = fold_expr(operand);
                    if let LexExpression::Value(ValueRef::Literal(lit)) = &op_fold {
                        if let Some(result) = eval_unary(*operator, lit.clone()) {
                            return LexExpression::Value(ValueRef::Literal(result));
                        }
                    }
                    LexExpression::UnaryOp {
                        operator: *operator,
                        operand: Box::new(op_fold),
                    }
                }
                _ => expr.clone(),
            }
        }

        fn eval_binary(
            op: LexBinaryOperator,
            left: LexLiteral,
            right: LexLiteral,
        ) -> Option<LexLiteral> {
            use LexBinaryOperator::*;
            match op {
                Add => match (left, right) {
                    (LexLiteral::Integer(a), LexLiteral::Integer(b)) => {
                        Some(LexLiteral::Integer(a + b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Float(b)) => Some(LexLiteral::Float(a + b)),
                    (LexLiteral::Integer(a), LexLiteral::Float(b)) => {
                        Some(LexLiteral::Float(a as f64 + b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Integer(b)) => {
                        Some(LexLiteral::Float(a + b as f64))
                    }
                    (LexLiteral::String(mut s1), LexLiteral::String(s2)) => {
                        s1.push_str(&s2);
                        Some(LexLiteral::String(s1))
                    }
                    _ => None,
                },
                Subtract => match (left, right) {
                    (LexLiteral::Integer(a), LexLiteral::Integer(b)) => {
                        Some(LexLiteral::Integer(a - b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Float(b)) => Some(LexLiteral::Float(a - b)),
                    (LexLiteral::Integer(a), LexLiteral::Float(b)) => {
                        Some(LexLiteral::Float(a as f64 - b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Integer(b)) => {
                        Some(LexLiteral::Float(a - b as f64))
                    }
                    _ => None,
                },
                Multiply => match (left, right) {
                    (LexLiteral::Integer(a), LexLiteral::Integer(b)) => {
                        Some(LexLiteral::Integer(a * b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Float(b)) => Some(LexLiteral::Float(a * b)),
                    (LexLiteral::Integer(a), LexLiteral::Float(b)) => {
                        Some(LexLiteral::Float(a as f64 * b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Integer(b)) => {
                        Some(LexLiteral::Float(a * b as f64))
                    }
                    _ => None,
                },
                Divide => match (left, right) {
                    (LexLiteral::Integer(a), LexLiteral::Integer(b)) if b != 0 => {
                        Some(LexLiteral::Integer(a / b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Float(b)) if b != 0.0 => {
                        Some(LexLiteral::Float(a / b))
                    }
                    (LexLiteral::Integer(a), LexLiteral::Float(b)) if b != 0.0 => {
                        Some(LexLiteral::Float(a as f64 / b))
                    }
                    (LexLiteral::Float(a), LexLiteral::Integer(b)) if b != 0 => {
                        Some(LexLiteral::Float(a / b as f64))
                    }
                    _ => None,
                },
                GreaterThan => cmp_bool(left, right, |a, b| a > b),
                LessThan => cmp_bool(left, right, |a, b| a < b),
                GreaterEqual => cmp_bool(left, right, |a, b| a >= b),
                LessEqual => cmp_bool(left, right, |a, b| a <= b),
                Equal => Some(LexLiteral::Boolean(literals_equal(&left, &right))),
                NotEqual => Some(LexLiteral::Boolean(!literals_equal(&left, &right))),
                And => match (left, right) {
                    (LexLiteral::Boolean(a), LexLiteral::Boolean(b)) => {
                        Some(LexLiteral::Boolean(a && b))
                    }
                    _ => None,
                },
                Or => match (left, right) {
                    (LexLiteral::Boolean(a), LexLiteral::Boolean(b)) => {
                        Some(LexLiteral::Boolean(a || b))
                    }
                    _ => None,
                },
            }
        }

        fn cmp_bool(
            left: LexLiteral,
            right: LexLiteral,
            cmp: fn(f64, f64) -> bool,
        ) -> Option<LexLiteral> {
            match (left, right) {
                (LexLiteral::Integer(a), LexLiteral::Integer(b)) => {
                    Some(LexLiteral::Boolean(cmp(a as f64, b as f64)))
                }
                (LexLiteral::Float(a), LexLiteral::Float(b)) => {
                    Some(LexLiteral::Boolean(cmp(a, b)))
                }
                (LexLiteral::Integer(a), LexLiteral::Float(b)) => {
                    Some(LexLiteral::Boolean(cmp(a as f64, b)))
                }
                (LexLiteral::Float(a), LexLiteral::Integer(b)) => {
                    Some(LexLiteral::Boolean(cmp(a, b as f64)))
                }
                _ => None,
            }
        }

        fn literals_equal(a: &LexLiteral, b: &LexLiteral) -> bool {
            a == b
        }

        fn eval_unary(op: LexUnaryOperator, lit: LexLiteral) -> Option<LexLiteral> {
            use LexUnaryOperator::*;
            match op {
                Negate => match lit {
                    LexLiteral::Integer(i) => Some(LexLiteral::Integer(-i)),
                    LexLiteral::Float(f) => Some(LexLiteral::Float(-f)),
                    _ => None,
                },
                Not => match lit {
                    LexLiteral::Boolean(b) => Some(LexLiteral::Boolean(!b)),
                    _ => None,
                },
            }
        }

        // Traverse instructions and replace expressions
        for instr in &mut program.instructions {
            match instr {
                LexInstruction::Assign { expr, .. } => {
                    let folded = fold_expr(expr);
                    if &folded != expr {
                        *expr = folded;
                        context.record_optimization(Optimization::ConstantFolding);
                    }
                }
                LexInstruction::DataFilter { predicate, .. } => {
                    let folded = fold_expr(predicate);
                    if &folded != predicate {
                        *predicate = folded;
                        context.record_optimization(Optimization::ConstantFolding);
                    }
                }
                LexInstruction::If { condition, .. } => {
                    let folded = fold_expr(condition);
                    if &folded != condition {
                        *condition = folded;
                        context.record_optimization(Optimization::ConstantFolding);
                    }
                }
                LexInstruction::While { condition, .. } => {
                    let folded = fold_expr(condition);
                    if &folded != condition {
                        *condition = folded;
                        context.record_optimization(Optimization::ConstantFolding);
                    }
                }
                // TODO: fold inside other instructions if needed
                _ => {}
            }
        }

        Ok(())
    }

    /// Inlines small functions (<=3 instructions without control flow) in Call invocations
    fn inline_functions(
        &self,
        program: &mut LexProgram,
        context: &mut OptimizerContext,
    ) -> Result<()> {
        // Step 1: identify inlineable functions
        let inlineable: Vec<String> = program
            .functions
            .iter()
            .filter(|(_, f)| {
                f.body.len() <= 3
                    && !f.body.iter().any(|instr| {
                        matches!(
                            instr,
                            LexInstruction::Call { .. }
                                | LexInstruction::While { .. }
                                | LexInstruction::ForEach { .. }
                                | LexInstruction::If { .. }
                        )
                    })
            })
            .map(|(name, _)| name.clone())
            .collect();

        if inlineable.is_empty() {
            return Ok(());
        }

        // Step 2: replace calls
        let mut new_insts = Vec::new();
        for instr in &program.instructions {
            match instr {
                LexInstruction::Call {
                    result,
                    function,
                    args,
                } if inlineable.contains(function) => {
                    let func = &program.functions[function];
                    // Parameter -> args mapping (we only support Value expr)
                    let mut param_map: std::collections::HashMap<String, LexExpression> =
                        std::collections::HashMap::new();
                    for (idx, (pname, _ptype)) in func.parameters.iter().enumerate() {
                        if let Some(arg_expr) = args.get(idx) {
                            param_map.insert(pname.clone(), arg_expr.clone());
                        }
                    }

                    // clone body and do simple substitution
                    for finst in &func.body {
                        let mut cloned = finst.clone();
                        substitute_exprs(&mut cloned, &param_map);
                        match (&result, cloned.clone()) {
                            (
                                Some(res_ref),
                                LexInstruction::Assign {
                                    result: inner_res,
                                    expr,
                                },
                            ) if &inner_res == res_ref => {
                                // ok keep as is with same result
                                new_insts.push(LexInstruction::Assign {
                                    result: res_ref.clone(),
                                    expr,
                                });
                            }
                            _ => new_insts.push(cloned),
                        }
                    }
                    context.record_optimization(Optimization::InlineFunction);
                }
                _ => new_insts.push(instr.clone()),
            }
        }

        program.instructions = new_insts;

        fn substitute_exprs(
            instr: &mut LexInstruction,
            map: &std::collections::HashMap<String, LexExpression>,
        ) {
            fn subst_expr(
                expr: &mut LexExpression,
                map: &std::collections::HashMap<String, LexExpression>,
            ) {
                match expr {
                    LexExpression::Value(ValueRef::Named(name)) => {
                        if let Some(rep) = map.get(name) {
                            *expr = rep.clone();
                        }
                    }
                    LexExpression::BinaryOp { left, right, .. } => {
                        subst_expr(left, map);
                        subst_expr(right, map);
                    }
                    LexExpression::UnaryOp { operand, .. } => {
                        subst_expr(operand, map);
                    }
                    LexExpression::FieldAccess { base, .. } => {
                        subst_expr(base, map);
                    }
                    _ => {}
                }
            }

            match instr {
                LexInstruction::Assign { expr, .. } => subst_expr(expr, map),
                LexInstruction::DataFilter { predicate, .. } => subst_expr(predicate, map),
                LexInstruction::If { condition, .. } => subst_expr(condition, map),
                LexInstruction::While { condition, .. } => subst_expr(condition, map),
                _ => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
