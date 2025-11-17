// lexc/src/executor/agents.rs
// Agents: create/run/chain/parallel with budgets, deadlines (timeouts), and simple parallelism

use super::{ExecutionEnvironment, ExecutorError, RuntimeValue, ValueRef};
use crate::lexir::LexExpression;
use std::collections::HashMap;
use std::time::Duration;

impl ExecutionEnvironment {
    pub(crate) fn handle_agent_create(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        // agent_create(name: string, opts?: json)
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "agent_create requires at least a name".to_string(),
            ));
        }
        let name_val = self.evaluate_expression(args[0].clone())?;
        let name = match name_val {
            RuntimeValue::String(s) => s,
            _ => format!("{:?}", name_val),
        };

        let mut model = self
            .config
            .llm_model
            .clone()
            .unwrap_or_else(|| "simulated".to_string());
        let mut budget_usd: Option<f64> = None;
        let mut deadline_ms: Option<u64> = None;

        if args.len() > 1 {
            let opts_val = self.evaluate_expression(args[1].clone())?;
            if let RuntimeValue::Json(serde_json::Value::Object(map)) = opts_val {
                if let Some(m) = map.get("model").and_then(|v| v.as_str()) {
                    model = m.to_string();
                }
                if let Some(b) = map.get("budget_usd").and_then(|v| v.as_f64()) {
                    budget_usd = Some(b);
                }
                if let Some(d) = map.get("deadline_ms").and_then(|v| v.as_u64()) {
                    deadline_ms = Some(d);
                }
            }
        }

        self.agent_registry.insert(
            name.clone(),
            super::AgentState {
                model,
                budget_usd,
                deadline_ms,
            },
        );
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(name.clone()))?;
        }
        Ok(())
    }

    pub(crate) fn handle_agent_run(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        #[cfg(feature = "otel")]
        let _span_guard = {
            use tracing::info_span;
            info_span!("agent_run").entered()
        };
        // agent_run(name: string, prompt: string)
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "agent_run requires name and prompt".to_string(),
            ));
        }
        let name = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::String(s) => s,
            v => format!("{:?}", v),
        };
        let prompt = match self.evaluate_expression(args[1].clone())? {
            RuntimeValue::String(s) => s,
            v => format!("{:?}", v),
        };

        let agent = self
            .agent_registry
            .get(&name)
            .cloned()
            .ok_or_else(|| ExecutorError::NameError(format!("Unknown agent '{}'", name)))?;
        if let Some(b) = agent.budget_usd {
            if b <= 0.0 {
                return Err(ExecutorError::RuntimeError(
                    "Agent budget exhausted".to_string(),
                ));
            }
        }

        let system = None::<&str>;
        let temperature = Some(0.3);
        let model_name = Some(agent.model.as_str());
        let mut llm_adapter = self.llm_adapter.clone();

        let _agent_key = name.clone();
        // Unified retries/backoff with LLM defaults as fallback
        let max_retries: usize = std::env::var("LEXON_AGENT_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| {
                std::env::var("LEXON_LLM_RETRIES")
                    .ok()
                    .and_then(|v| v.parse().ok())
            })
            .unwrap_or(1);
        let backoff_ms: u64 = std::env::var("LEXON_AGENT_BACKOFF_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .or_else(|| {
                std::env::var("LEXON_LLM_BACKOFF_MS")
                    .ok()
                    .and_then(|v| v.parse().ok())
            })
            .unwrap_or(300);
        let est_cost: f64 = std::env::var("LEXON_AGENT_EST_COST_USD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.001);
        let response = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let params: HashMap<String, String> = HashMap::new();
                let mut attempt: usize = 0;
                loop {
                    attempt += 1;
                    // Budget-aware model selection: error if exceeded
                    if !self
                        .llm_adapter
                        .consume_budget_for_model(agent.model.as_str(), est_cost)
                    {
                        return Err(ExecutorError::RuntimeError(
                            "Agent provider budget exceeded".to_string(),
                        ));
                    }
                    let fut = llm_adapter.call_llm_async(
                        model_name,
                        temperature,
                        system,
                        Some(&prompt),
                        None,
                        None,
                        &params,
                    );
                    let out = if let Some(ms) = agent.deadline_ms {
                        match tokio::time::timeout(Duration::from_millis(ms), fut).await {
                            Ok(r) => r,
                            Err(_) => Err(ExecutorError::RuntimeError(
                                "Agent run timed out".to_string(),
                            )),
                        }
                    } else {
                        fut.await
                    };
                    match out {
                        Ok(s) => break Ok(s),
                        Err(e) => {
                            // Update supervisor state
                            self.agent_status
                                .insert(name.clone(), format!("error: {}", e));
                            if attempt >= max_retries {
                                break Err(e);
                            }
                            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                            continue;
                        }
                    }
                }
            })
        })?;
        // Success state
        self.agent_status.insert(name.clone(), "ok".to_string());

        if let Some(res) = result {
            self.store_value(res, RuntimeValue::String(response))?;
        }
        Ok(())
    }

    pub(crate) fn handle_agent_chain(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        #[cfg(feature = "otel")]
        let _span_guard = {
            use tracing::info_span;
            info_span!("agent_chain").entered()
        };
        // agent_chain(name: string, steps: array-of-prompts)
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "agent_chain requires name and steps".to_string(),
            ));
        }
        let name = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::String(s) => s,
            v => format!("{:?}", v),
        };
        let steps_val = self.evaluate_expression(args[1].clone())?;
        let steps: Vec<String> = match steps_val {
            RuntimeValue::Json(serde_json::Value::Array(arr)) => arr
                .into_iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            RuntimeValue::String(s) => vec![s],
            _ => vec![],
        };

        let agent = self
            .agent_registry
            .get(&name)
            .cloned()
            .ok_or_else(|| ExecutorError::NameError(format!("Unknown agent '{}'", name)))?;
        let est_cost: f64 = std::env::var("LEXON_AGENT_EST_COST_USD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.001);
        let mut transcript = Vec::new();
        for prompt in steps.iter() {
            if *self.agent_cancelled.get(&name).unwrap_or(&false) {
                self.agent_status
                    .insert(name.clone(), "cancelled".to_string());
                return Err(ExecutorError::RuntimeError("Agent cancelled".to_string()));
            }
            let mut llm_adapter = self.llm_adapter.clone();
            if !self
                .llm_adapter
                .consume_budget_for_model(agent.model.as_str(), est_cost)
            {
                return Err(ExecutorError::RuntimeError(
                    "Agent provider budget exceeded".to_string(),
                ));
            }
            let model_name = Some(agent.model.as_str());
            let system = None::<&str>;
            let temperature = Some(0.3);
            let resp = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let params: HashMap<String, String> = HashMap::new();
                    let fut = llm_adapter.call_llm_async(
                        model_name,
                        temperature,
                        system,
                        Some(prompt),
                        None,
                        None,
                        &params,
                    );
                    if let Some(ms) = agent.deadline_ms {
                        match tokio::time::timeout(Duration::from_millis(ms), fut).await {
                            Ok(r) => r,
                            Err(_) => Err(ExecutorError::RuntimeError(
                                "Agent chain step timed out".to_string(),
                            )),
                        }
                    } else {
                        fut.await
                    }
                })
            })?;
            transcript.push(resp);
        }
        let out = serde_json::Value::Array(
            transcript
                .into_iter()
                .map(serde_json::Value::String)
                .collect(),
        );
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(out))?;
        }
        Ok(())
    }

    pub(crate) fn handle_agent_parallel(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        #[cfg(feature = "otel")]
        let _span_guard = {
            use tracing::info_span;
            info_span!("agent_parallel").entered()
        };
        // agent_parallel(name: string, tasks: array-of-prompts)
        if args.len() < 2 {
            return Err(ExecutorError::ArgumentError(
                "agent_parallel requires name and tasks".to_string(),
            ));
        }
        let name = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::String(s) => s,
            v => format!("{:?}", v),
        };
        let tasks_val = self.evaluate_expression(args[1].clone())?;
        let prompts: Vec<String> = match tasks_val {
            RuntimeValue::Json(serde_json::Value::Array(arr)) => arr
                .into_iter()
                .map(|v| v.as_str().unwrap_or("").to_string())
                .collect(),
            RuntimeValue::String(s) => vec![s],
            _ => vec![],
        };
        let agent = self
            .agent_registry
            .get(&name)
            .cloned()
            .ok_or_else(|| ExecutorError::NameError(format!("Unknown agent '{}'", name)))?;
        let max_conc: usize = std::env::var("LEXON_AGENT_MAX_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        use std::sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        };
        let cancelled = Arc::new(AtomicBool::new(false));
        if *self.agent_cancelled.get(&name).unwrap_or(&false) {
            cancelled.store(true, Ordering::Relaxed);
        }
        let est_cost: f64 = std::env::var("LEXON_AGENT_EST_COST_USD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.001);

        let results = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                use tokio::sync::Semaphore;
                let sem = Arc::new(Semaphore::new(max_conc));
                let mut handles = Vec::new();
                for p in prompts.into_iter() {
                    let permit = sem.clone().acquire_owned().await.expect("semaphore");
                    let mut adapter = self.llm_adapter.clone();
                    let model_name = agent.model.clone();
                    let deadline = agent.deadline_ms;
                    let cancelled_flag = cancelled.clone();
                    handles.push(tokio::spawn(async move {
                        let _permit = permit;
                        let params: HashMap<String, String> = HashMap::new();
                        if cancelled_flag.load(Ordering::Relaxed) {
                            return Err(ExecutorError::RuntimeError("Agent cancelled".to_string()));
                        }
                        if !adapter.consume_budget_for_model(&model_name, est_cost) {
                            return Err(ExecutorError::RuntimeError(
                                "Agent provider budget exceeded".to_string(),
                            ));
                        }
                        let fut = adapter.call_llm_async(
                            Some(model_name.as_str()),
                            Some(0.3),
                            None,
                            Some(&p),
                            None,
                            None,
                            &params,
                        );
                        let r = if let Some(ms) = deadline {
                            match tokio::time::timeout(Duration::from_millis(ms), fut).await {
                                Ok(x) => x,
                                Err(_) => Err(ExecutorError::RuntimeError(
                                    "Agent parallel task timed out".to_string(),
                                )),
                            }
                        } else {
                            fut.await
                        };
                        r
                    }));
                }
                let mut joined_results = Vec::new();
                for h in handles {
                    joined_results.push(h.await);
                }
                let mut out = Vec::new();
                for j in joined_results {
                    match j {
                        Ok(Ok(s)) => out.push(serde_json::Value::String(s)),
                        Ok(Err(e)) => out.push(serde_json::json!({"error": format!("{}", e)})),
                        Err(_) => out.push(serde_json::json!({"error": "join error"})),
                    }
                }
                Ok(out)
            })
        });
        let out_json = serde_json::Value::Array(results?);
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Json(out_json))?;
        }
        Ok(())
    }

    pub(crate) fn handle_agent_cancel(
        &mut self,
        args: &[LexExpression],
        result: Option<&ValueRef>,
    ) -> Result<(), ExecutorError> {
        if args.is_empty() {
            return Err(ExecutorError::ArgumentError(
                "agent_cancel requires name".to_string(),
            ));
        }
        let name = match self.evaluate_expression(args[0].clone())? {
            RuntimeValue::String(s) => s,
            v => format!("{:?}", v),
        };
        self.agent_cancelled.insert(name, true);
        if let Some(res) = result {
            self.store_value(res, RuntimeValue::Boolean(true))?;
        }
        Ok(())
    }
}
