// lexc/src/executor/memory.rs
//
// Unified memory wrapper: reuse runtime::memory to avoid duplication in executor.

use super::{ExecutorError, Result, RuntimeValue};
use crate::lexir::LexLiteral;
use crate::runtime::{memory as rt_memory, RuntimeConfig, RuntimeValue as RtValue};
use std::collections::HashMap;

pub struct MemoryManager {
    inner: rt_memory::MemoryManager,
}

impl MemoryManager {
    pub fn new() -> Self {
        let cfg = RuntimeConfig::default();
        Self {
            inner: rt_memory::MemoryManager::new(&cfg),
        }
    }

    fn to_rt(v: RuntimeValue) -> RtValue {
        match v {
            RuntimeValue::String(s) => RtValue::String(s),
            RuntimeValue::Integer(i) => RtValue::Integer(i),
            RuntimeValue::Float(f) => RtValue::Float(f),
            RuntimeValue::Boolean(b) => RtValue::Boolean(b),
            RuntimeValue::Json(j) => RtValue::String(j.to_string()),
            RuntimeValue::Dataset(_)
            | RuntimeValue::Result { .. }
            | RuntimeValue::MultiOutput { .. } => RtValue::String(format!("{:?}", v)),
            RuntimeValue::Null => RtValue::Null,
        }
    }

    fn from_rt(v: RtValue) -> RuntimeValue {
        match v {
            RtValue::String(s) => RuntimeValue::String(s),
            RtValue::Integer(i) => RuntimeValue::Integer(i),
            RtValue::Float(f) => RuntimeValue::Float(f),
            RtValue::Boolean(b) => RuntimeValue::Boolean(b),
            RtValue::List(list) => RuntimeValue::String(format!("{:?}", list)),
            RtValue::Map(map) => RuntimeValue::String(format!("{:?}", map)),
            RtValue::Dataset(ds) => RuntimeValue::String(ds),
            RtValue::Future(f) => RuntimeValue::String(format!("Future({})", f)),
            RtValue::Null => RuntimeValue::Null,
        }
    }

    pub fn load_memory(
        &mut self,
        scope: &str,
        source: Option<RuntimeValue>,
        strategy_str: &str,
        options: &HashMap<String, LexLiteral>,
    ) -> Result<RuntimeValue> {
        let strategy = match strategy_str.to_lowercase().as_str() {
            "vector" => rt_memory::MemoryStrategy::Vector,
            "summary" => rt_memory::MemoryStrategy::Summary,
            "hybrid" => rt_memory::MemoryStrategy::Hybrid,
            _ => rt_memory::MemoryStrategy::Buffer,
        };
        let capacity = options
            .get("capacity")
            .and_then(|lit| {
                matches!(lit, LexLiteral::Integer(_)).then(|| match lit {
                    LexLiteral::Integer(i) => *i as usize,
                    _ => 0,
                })
            })
            .filter(|v| *v > 0);
        let _ = self
            .inner
            .get_or_create_scope(scope, Some(strategy), capacity)
            .map_err(|e| ExecutorError::MemoryError(format!("{}", e)))?;

        if let Some(val) = source {
            let key = options.get("key").and_then(|lit| match lit {
                LexLiteral::String(s) => Some(s.as_str()),
                _ => None,
            });
            self.inner
                .store(scope, Self::to_rt(val), key, None)
                .map_err(|e| ExecutorError::MemoryError(format!("{}", e)))?;
        }

        let key = options.get("key").and_then(|lit| match lit {
            LexLiteral::String(s) => Some(s.as_str()),
            _ => None,
        });
        let vals = self
            .inner
            .retrieve(scope, key, None)
            .map_err(|e| ExecutorError::MemoryError(format!("{}", e)))?;
        let json_values: Vec<serde_json::Value> = vals
            .into_iter()
            .map(Self::from_rt)
            .map(|v| match v {
                RuntimeValue::String(s) => serde_json::Value::String(s),
                RuntimeValue::Integer(i) => serde_json::Value::Number(serde_json::Number::from(i)),
                RuntimeValue::Float(f) => serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null),
                RuntimeValue::Boolean(b) => serde_json::Value::Bool(b),
                RuntimeValue::Json(j) => j,
                RuntimeValue::Null => serde_json::Value::Null,
                other => serde_json::Value::String(format!("{:?}", other)),
            })
            .collect();
        Ok(RuntimeValue::Json(serde_json::Value::Array(json_values)))
    }

    pub fn store_memory(
        &mut self,
        scope: &str,
        value: RuntimeValue,
        key: Option<&str>,
    ) -> Result<()> {
        self.inner
            .store(scope, Self::to_rt(value), key, None)
            .map_err(|e| ExecutorError::MemoryError(format!("{}", e)))
    }

    pub fn load_value_by_key(&self, scope: &str, key: &str) -> Result<Option<RuntimeValue>> {
        match self.inner.retrieve(scope, Some(key), Some(1)) {
            Ok(mut v) => Ok(v.pop().map(Self::from_rt)),
            Err(_) => Ok(None),
        }
    }

    pub fn get_session_history(&self, session_id: &str) -> Result<String> {
        let key = format!("session_history_{}", session_id);
        match self.load_value_by_key("sessions", &key)? {
            Some(RuntimeValue::String(history)) => Ok(history),
            Some(_) => Ok(format!(
                "Session '{}' exists but contains non-text data",
                session_id
            )),
            None => Err(ExecutorError::MemoryError(format!(
                "Session '{}' not found",
                session_id
            ))),
        }
    }

    pub fn update_session_history(&mut self, session_id: &str, new_history: &str) -> Result<()> {
        let key = format!("session_history_{}", session_id);
        let _ = self
            .inner
            .get_or_create_scope("sessions", Some(rt_memory::MemoryStrategy::Buffer), None)
            .map_err(|e| ExecutorError::MemoryError(format!("{}", e)))?;
        self.store_memory(
            "sessions",
            RuntimeValue::String(new_history.to_string()),
            Some(&key),
        )
    }

    pub fn add_to_session_history(&mut self, session_id: &str, message: &str) -> Result<()> {
        let key = format!("session_history_{}", session_id);
        let existing = match self.load_value_by_key("sessions", &key)? {
            Some(RuntimeValue::String(h)) => h,
            _ => String::new(),
        };
        let updated = if existing.is_empty() {
            message.to_string()
        } else {
            format!("{}\n{}", existing, message)
        };
        self.update_session_history(session_id, &updated)
    }

    pub fn create_session(&mut self, session_id: &str) -> Result<()> {
        self.update_session_history(session_id, "")
    }

    pub fn session_exists(&self, session_id: &str) -> bool {
        let key = format!("session_history_{}", session_id);
        self.load_value_by_key("sessions", &key)
            .ok()
            .flatten()
            .is_some()
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
