"""Make harness.py importable as a plain module — it's a standalone script (stdlib-only,
runs on a bare mesh runner), not part of the scrutator package."""

import sys
from pathlib import Path

_SCRUTATOR_BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRUTATOR_BENCH_DIR))
sys.path.insert(0, str(_SCRUTATOR_BENCH_DIR / "tools"))
sys.path.insert(0, str(_SCRUTATOR_BENCH_DIR / "live"))
