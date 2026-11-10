#!/usr/bin/env python3
"""Preview-only bootstrap declaration for fi10-public / fi10-private.

Live creation is refused. Production enablement requires the paired Feeder
grant (FI10-T5.4) and an accepted generation tag (FI10-T2.8). This script only
documents the intended namespaces and verifies name shape against grant_namespace.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NAMESPACES = ("fi10-public", "fi10-private")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="refused until Feeder+generation binds exist")
    args = parser.parse_args(argv)

    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    # Reuse the existing grant tool's validator so names stay compatible.
    import importlib.util

    grant_path = Path(__file__).resolve().parent / "grant_namespace.py"
    spec = importlib.util.spec_from_file_location("grant_namespace", grant_path)
    grant = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(grant)

    for name in NAMESPACES:
        grant.validate_namespace(name)

    if args.live:
        print(
            "FI10_NAMESPACE_BOOTSTRAP_REFUSED reason=live_enablement_held "
            "namespaces=fi10-public,fi10-private "
            "(requires T2.8 generation tag + T5.4 Feeder dry-run receipt; use synthetic harness)",
            file=sys.stderr,
        )
        return 1

    print(
        "FI10_NAMESPACE_BOOTSTRAP_PREVIEW state=declared "
        f"namespaces={','.join(NAMESPACES)} "
        "live=held synthetic_harness=scripts/fi10_namespace_readback.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
