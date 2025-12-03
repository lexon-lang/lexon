mod basic;
mod hybrid;
mod patricia;
mod raptor;

use crate::executor::structured_memory::{MemoryObject, MemorySpaceFile, RecallOptions};

pub trait MemoryBackend: Send + Sync {
    fn id(&self) -> &'static str;
    fn order_for_topic<'a>(
        &self,
        space: &'a MemorySpaceFile,
        topic: &str,
        opts: &RecallOptions,
    ) -> Option<Vec<&'a MemoryObject>>;
    fn order_for_kind<'a>(
        &self,
        space: &'a MemorySpaceFile,
        kind: &str,
        opts: &RecallOptions,
    ) -> Option<Vec<&'a MemoryObject>>;
}

pub fn build_backend(name: &str) -> Result<Box<dyn MemoryBackend>, String> {
    match name.to_ascii_lowercase().as_str() {
        "basic" => Ok(Box::new(basic::BasicBackend::default())),
        "patricia" => Ok(Box::new(patricia::PatriciaBackend::default())),
        "hybrid" | "graphrag" => Ok(Box::new(hybrid::HybridBackend::default())),
        "raptor" => Ok(Box::new(raptor::RaptorBackend::default())),
        other => Err(format!("unknown backend '{}'", other)),
    }
}

