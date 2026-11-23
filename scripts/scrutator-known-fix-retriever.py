#!/usr/bin/env python3
"""Executable shim for the Datarim `DATARIM_KNOWN_FIX_RETRIEVER` contract (SRCH-0026 / LTM-0022).

`known-fix-memory.py` requires an ABSOLUTE, regular, non-symlink, executable file and strips the
child environment to `PATH`, so neither `PYTHONPATH` nor an activated venv survives the hop. This
shim therefore locates the package from its own resolved location before importing.

Point the framework at it with:

    export DATARIM_KNOWN_FIX_RETRIEVER=/abs/path/to/scripts/scrutator-known-fix-retriever

Alternatively, install the package into a venv and use that venv's console script — the shebang
carries the interpreter, so a stripped environment resolves the package either way.
"""

import sys
from pathlib import Path

try:
    from scrutator.tools.known_fix_retriever import main
except ModuleNotFoundError:  # not installed — fall back to the in-repo source tree
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from scrutator.tools.known_fix_retriever import main

if __name__ == "__main__":
    raise SystemExit(main())
