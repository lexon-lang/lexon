Lexon Python Binding (lexon_py)

Overview
- Python bindings to compile and run Lexon programs from Python.
- Exposes:
  - PyRuntime: minimal runtime wrapper
  - compile_lx(source: str) -> str: compile .lx source to LexIR JSON

Requirements
- Python 3.9+ (tested on 3.11)
- Rust toolchain (pinned by repository to 1.82.0)
- pip and maturin

Recommended setup (virtualenv)
1) python3 -m venv .venv && source .venv/bin/activate
2) python3 -m pip install --upgrade pip
3) python3 -m pip install maturin

Editable install (development)
From this directory:

python3 -m maturin develop

This builds the extension module and installs it into the active Python environment.

Smoke test
python3 - << 'PY'
import lexon_py
print('lexon_py version:', getattr(lexon_py, '__version__', '?'))
print('exports:', [n for n in dir(lexon_py) if n in ('PyRuntime','compile_lx')])
PY

Example usage
Compile .lx source and execute via runtime:

python3 - << 'PY'
from lexon_py import compile_lx, PyRuntime

source = """
let x: int = 2 + 3;
print(x);
"""

lexir_json = compile_lx(source)
rt = PyRuntime()
rt.execute_json(lexir_json)
PY

Building a wheel (distribution)
python3 -m maturin build --release
The wheel will be written under target/wheels/.

Troubleshooting (macOS)
- If you see undefined Python symbols at link time, maturin/pyo3 should handle proper flags.
- This crate includes a build.rs that passes -Wl,-undefined,dynamic_lookup on macOS for extra safety.

Uninstall
python3 -m pip uninstall -y lexon_py


