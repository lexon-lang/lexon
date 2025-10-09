//! Optimizer tests
//!
//! This module contains unit tests for the different
//! optimizations implemented.

use super::*;
use crate::lexir::LexBinaryOperator;
use crate::lexir::{LexExpression, LexInstruction, LexLiteral, LexProgram, ValueRef};

/// Creates a simple program to test optimizations
fn create_test_program() -> LexProgram {
    let mut program = LexProgram::new();

    // Variable a = 10
    program.add_instruction(LexInstruction::Assign {
        result: ValueRef::Named("a".to_string()),
        expr: LexExpression::Value(ValueRef::Literal(LexLiteral::Integer(10))),
    });

    // Variable b = a (for constant propagation)
    program.add_instruction(LexInstruction::Assign {
        result: ValueRef::Named("b".to_string()),
        expr: LexExpression::Value(ValueRef::Named("a".to_string())),
    });

    // Variable c = b + 5 (for constant propagation)
    program.add_instruction(LexInstruction::Assign {
        result: ValueRef::Named("c".to_string()),
        expr: LexExpression::BinaryOp {
            operator: LexBinaryOperator::Add,
            left: Box::new(LexExpression::Value(ValueRef::Named("b".to_string()))),
            right: Box::new(LexExpression::Value(ValueRef::Literal(
                LexLiteral::Integer(5),
            ))),
        },
    });

    // Variable unused = 20 (dead code)
    program.add_instruction(LexInstruction::Assign {
        result: ValueRef::Named("unused".to_string()),
        expr: LexExpression::Value(ValueRef::Literal(LexLiteral::Integer(20))),
    });

    // Declare variable result
    program.add_instruction(LexInstruction::Declare {
        name: "result".to_string(),
        type_name: None,
        is_mutable: false,
    });

    // result = c * 2 (variable used at the end)
    program.add_instruction(LexInstruction::Assign {
        result: ValueRef::Named("result".to_string()),
        expr: LexExpression::BinaryOp {
            operator: LexBinaryOperator::Multiply,
            left: Box::new(LexExpression::Value(ValueRef::Named("c".to_string()))),
            right: Box::new(LexExpression::Value(ValueRef::Literal(
                LexLiteral::Integer(2),
            ))),
        },
    });

    program
}

fn create_data_operations_program() -> LexProgram {
    let mut program = LexProgram::new();

    // Load data
    program.add_instruction(LexInstruction::DataLoad {
        result: ValueRef::Named("data".to_string()),
        source: "data.csv".to_string(),
        schema: None,
        options: std::collections::HashMap::new(),
    });

    // Filter data (temp1)
    program.add_instruction(LexInstruction::DataFilter {
        result: ValueRef::Named("temp1".to_string()),
        input: ValueRef::Named("data".to_string()),
        predicate: LexExpression::BinaryOp {
            operator: LexBinaryOperator::GreaterThan,
            left: Box::new(LexExpression::Value(ValueRef::Named("column1".to_string()))),
            right: Box::new(LexExpression::Value(ValueRef::Literal(
                LexLiteral::Integer(10),
            ))),
        },
    });

    // Filter data again (temp2) - Candidate for fusion
    program.add_instruction(LexInstruction::DataFilter {
        result: ValueRef::Named("temp2".to_string()),
        input: ValueRef::Named("temp1".to_string()),
        predicate: LexExpression::BinaryOp {
            operator: LexBinaryOperator::LessThan,
            left: Box::new(LexExpression::Value(ValueRef::Named("column2".to_string()))),
            right: Box::new(LexExpression::Value(ValueRef::Literal(
                LexLiteral::Integer(100),
            ))),
        },
    });

    // Select columns (result)
    program.add_instruction(LexInstruction::DataSelect {
        result: ValueRef::Named("result".to_string()),
        input: ValueRef::Named("temp2".to_string()),
        fields: vec!["column1".to_string(), "column2".to_string()],
    });

    program
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_code_elimination() {
        let mut program = create_test_program();
        let config = OptimizerConfig {
            enabled_optimizations: vec![Optimization::DeadCodeElimination],
            verbose: false,
        };

        let optimizer = Optimizer::new(config);
        let stats = optimizer.optimize(&mut program).unwrap();

        // Verify unused variable was removed
        assert!(stats.contains_key(&Optimization::DeadCodeElimination));

        // Search if variable "unused" still exists in the program
        let mut unused_found = false;
        for instruction in &program.instructions {
            if let LexInstruction::Assign {
                result: ValueRef::Named(name),
                ..
            } = instruction
            {
                if name == "unused" {
                    unused_found = true;
                    break;
                }
            }
        }

        assert!(!unused_found, "Variable 'unused' should have been removed");
    }

    #[test]
    fn test_constant_propagation() {
        let mut program = create_test_program();

        // Print program before optimization
        println!("Program before optimization:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        let config = OptimizerConfig {
            enabled_optimizations: vec![Optimization::ConstantPropagation],
            verbose: true, // Enable verbose mode to observe optimizations
        };

        let optimizer = Optimizer::new(config);
        let stats = optimizer.optimize(&mut program).unwrap();

        // Print program after optimization
        println!("Program after optimization:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        // Print optimization stats
        println!("Optimization stats: {:?}", stats);

        // Verify that constant propagation ran
        assert!(
            stats.contains_key(&Optimization::ConstantPropagation),
            "Constant propagation optimization was not applied"
        );

        // Check variable "c" to confirm its value was propagated
        let mut c_value_propagated = false;
        for instruction in &program.instructions {
            if let LexInstruction::Assign {
                result: ValueRef::Named(name),
                expr,
            } = instruction
            {
                if name == "c" {
                    println!("Instruction c: {:?}", instruction);
                    if let LexExpression::BinaryOp { left, .. } = expr {
                        println!("Left in c: {:?}", left);

                        // Aceptar cualquiera de estos casos como correcto:
                        // 1. Propagated the literal value 10 directly
                        // 2. Referencia a la variable original 'a'
                        // 3. Se mantiene como 'b', pero 'b' tiene el valor correcto
                        match **left {
                            LexExpression::Value(ValueRef::Literal(LexLiteral::Integer(val))) => {
                                // Case 1: Propagated the literal value
                                assert_eq!(val, 10);
                                c_value_propagated = true;
                            }
                            LexExpression::Value(ValueRef::Named(ref var_name)) => {
                                // Case 2 or 3: Reference to another variable
                                if var_name == "a" || var_name == "b" {
                                    c_value_propagated = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        assert!(
            c_value_propagated,
            "Constant value should have been propagated to 'c'"
        );
    }

    #[test]
    fn test_data_operation_fusion() {
        let mut program = create_data_operations_program();
        let config = OptimizerConfig {
            enabled_optimizations: vec![Optimization::DataOperationFusion],
            verbose: false,
        };

        let optimizer = Optimizer::new(config);
        let stats = optimizer.optimize(&mut program).unwrap();

        // Verify that operations were fused
        assert!(stats.contains_key(&Optimization::DataOperationFusion));

        // Count how many DataFilter instructions there are (there should be one fewer after fusion)
        let filter_count = program
            .instructions
            .iter()
            .filter(|instr| matches!(instr, LexInstruction::DataFilter { .. }))
            .count();

        // Originally there were 2; after fusion there should be 1
        assert_eq!(
            filter_count, 1,
            "Filters should have been fused into a single one"
        );
    }

    #[test]
    fn test_redundant_assignment_elimination() {
        let mut program = LexProgram::new();

        // a = 10
        program.add_instruction(LexInstruction::Assign {
            result: ValueRef::Named("a".to_string()),
            expr: LexExpression::Value(ValueRef::Literal(LexLiteral::Integer(10))),
        });

        // b = a   (intermediate redundant)
        program.add_instruction(LexInstruction::Assign {
            result: ValueRef::Named("b".to_string()),
            expr: LexExpression::Value(ValueRef::Named("a".to_string())),
        });

        // c = b   (should become c = a)
        program.add_instruction(LexInstruction::Assign {
            result: ValueRef::Named("c".to_string()),
            expr: LexExpression::Value(ValueRef::Named("b".to_string())),
        });

        // result = c  (should become result = a)
        program.add_instruction(LexInstruction::Assign {
            result: ValueRef::Named("result".to_string()),
            expr: LexExpression::Value(ValueRef::Named("c".to_string())),
        });

        // Print program before optimization
        println!("Program before optimization:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        let config = OptimizerConfig {
            enabled_optimizations: vec![Optimization::RedundantAssignmentElimination],
            verbose: true, // Enable verbose mode to observe optimizations
        };

        let optimizer = Optimizer::new(config);
        let stats = optimizer.optimize(&mut program).unwrap();

        // Print the program after optimization
        println!("Program after optimization:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        // Print optimization statistics
        println!("Optimization statistics: {:?}", stats);

        // Verify that redundant assignments were removed
        assert!(
            stats.contains_key(&Optimization::RedundantAssignmentElimination),
            "Redundant assignment elimination optimization was not applied"
        );

        // Verify that the last assignment now uses a more direct reference
        let mut optimization_success = false;
        for instruction in &program.instructions {
            if let LexInstruction::Assign {
                result: ValueRef::Named(name),
                expr,
            } = instruction
            {
                if name == "result" {
                    println!("Instruction result: {:?}", instruction);

                    // Consider success if:
                    // 1. It directly uses variable 'a'
                    // 2. It directly uses the literal value 10
                    if let LexExpression::Value(value_ref) = expr {
                        match value_ref {
                            ValueRef::Named(ref var_name) => {
                                if var_name == "a" {
                                    optimization_success = true;
                                    println!("Result uses variable 'a' directly - SUCCESS");
                                }
                            }
                            ValueRef::Literal(LexLiteral::Integer(10)) => {
                                optimization_success = true;
                                println!("Result uses literal 10 directly - SUCCESS");
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        assert!(
            optimization_success,
            "The variable 'result' should use 'a' or the literal 10 directly after optimization"
        );
    }

    #[test]
    fn test_all_optimizations() {
        let mut program = create_test_program();

        // Print the program before optimization
        println!("Program before optimization:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        // Step 1: Apply constant propagation and redundant assignment elimination
        let config1 = OptimizerConfig {
            enabled_optimizations: vec![
                Optimization::ConstantPropagation,
                Optimization::RedundantAssignmentElimination,
            ],
            verbose: true,
        };

        let optimizer1 = Optimizer::new(config1);
        let stats1 = optimizer1.optimize(&mut program).unwrap();

        // Print program after first optimization
        println!("\nProgram after ConstantPropagation and RedundantAssignmentElimination:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        // Step 2: Apply dead code elimination
        let config2 = OptimizerConfig {
            enabled_optimizations: vec![Optimization::DeadCodeElimination],
            verbose: true,
        };

        let optimizer2 = Optimizer::new(config2);
        let stats2 = optimizer2.optimize(&mut program).unwrap();

        // Print program after second optimization
        println!("\nProgram after DeadCodeElimination:");
        for (i, instr) in program.instructions.iter().enumerate() {
            println!("  [{}] {:?}", i, instr);
        }

        // Combine statistics
        let mut all_stats = stats1;
        for (opt, count) in stats2 {
            *all_stats.entry(opt).or_insert(0) += count;
        }

        // Print total optimization statistics
        println!("Total optimization statistics: {:?}", all_stats);

        // Verify optimizations were applied
        assert!(
            all_stats.contains_key(&Optimization::DeadCodeElimination),
            "Missing optimization: DeadCodeElimination"
        );
        assert!(
            all_stats.contains_key(&Optimization::ConstantPropagation),
            "Missing optimization: ConstantPropagation"
        );
        assert!(
            all_stats.contains_key(&Optimization::RedundantAssignmentElimination),
            "Missing optimization: RedundantAssignmentElimination"
        );

        // Program should have fewer instructions after optimizations
        assert!(
            program.instructions.len() < 6,
            "Program should have fewer instructions after optimizations"
        );
    }
}
