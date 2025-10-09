use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3_asyncio::tokio as aio_tokio;
use tree_sitter::Parser;
extern "C" {
    fn tree_sitter_lexon() -> tree_sitter::Language;
}

use lexc::lexir::LexProgram;
use lexc::{Runtime, RuntimeConfig};

#[pyclass]
struct PyRuntime {
    inner: Runtime,
}

#[pymethods]
impl PyRuntime {
    #[new]
    fn new() -> Self {
        PyRuntime {
            inner: Runtime::new(RuntimeConfig::default()),
        }
    }

    /// Executes a LexIR program provided as a JSON string.
    fn execute_json(&mut self, lexir_json: &str) -> PyResult<()> {
        let program = LexProgram::from_json(lexir_json)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        // Execute in Tokio runtime provided by pyo3_asyncio
        aio_tokio::get_runtime()
            .block_on(self.inner.execute_program(&program))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn compile_lx(source_code: &str) -> PyResult<String> {
    // Parse source into CST using tree-sitter
    let mut parser = Parser::new();
    let language = unsafe { tree_sitter_lexon() };
    parser
        .set_language(&language)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    let tree = parser
        .parse(source_code, None)
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Failed to parse source"))?;

    // Build HIR
    let hir_nodes = lexc::hir_builder::build_hir_from_cst(tree.root_node(), source_code)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("HIR error: {e:?}")))?;

    // LexIR
    let program = lexc::hir_to_lexir::convert_hir_to_lexir(&hir_nodes)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("LexIR error: {e:?}")))?;

    let json = program
        .to_json()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(json)
}

#[pymodule]
fn lexon_py(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Tokio runtime will be lazily initialized by pyo3_asyncio when first accessed

    m.add_class::<PyRuntime>()?;
    m.add_function(wrap_pyfunction!(compile_lx, m)?)?;

    // Expose version
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
