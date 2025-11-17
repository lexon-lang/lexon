import os
import sys
import subprocess
from pathlib import Path

def _bin_path(name: str) -> str:
    here = Path(__file__).resolve().parent
    # Binaries will be copied at build time into lexon_cli/bin
    p = here / "bin" / name
    if not p.exists():
        # Fallback: try PATH
        return name
    # Ensure executable bit
    try:
        os.chmod(p, 0o755)
    except Exception:
        pass
    return str(p)

def _exec(name: str):
    exe = _bin_path(name)
    try:
        proc = subprocess.Popen([exe] + sys.argv[1:])
        proc.wait()
        sys.exit(proc.returncode)
    except FileNotFoundError:
        print(f"{name} not found; ensure wheel bundles binaries or install native binaries.", file=sys.stderr)
        sys.exit(127)

def run_lexc():
    _exec("lexc")

def run_lexc_cli():
    _exec("lexc-cli")





