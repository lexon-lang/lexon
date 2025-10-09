use crate::lexir::LexProgram;
use async_trait::async_trait;

#[async_trait]
pub trait LexExecutor {
    /// Executes a LexIR program. Returns Err(String) if it fails.
    async fn execute_program_generic(&mut self, program: &LexProgram) -> Result<(), String>;
}

/// Executor that abstracts between synchronous mode (ExecutionEnvironment) and asynchronous (Runtime).
pub enum HybridExecutor {
    Sync(crate::executor::ExecutionEnvironment),
    Async(crate::runtime::Runtime),
}

#[async_trait]
impl LexExecutor for HybridExecutor {
    async fn execute_program_generic(&mut self, program: &LexProgram) -> Result<(), String> {
        match self {
            HybridExecutor::Sync(env) => {
                env.execute_program(program).map_err(|e| format!("{:?}", e))
            }
            HybridExecutor::Async(rt) => rt
                .execute_program(program)
                .await
                .map_err(|e| format!("{:?}", e)),
        }
    }
}
