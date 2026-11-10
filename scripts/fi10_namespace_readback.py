#!/usr/bin/env python3
"""Run the synthetic FI10 namespace readback + withdrawal harness (no live enablement).

Dry-run / synthetic only. Refuses --live. Does not grant Feeder scopes, does not
write to a production Scrutator database, and does not perform a production purge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("fi10-namespace-harness-receipt.json"),
        help="path to write the synthetic harness receipt",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="refused: live namespace enablement / production purge is HELD",
    )
    args = parser.parse_args(argv)
    if args.live:
        print(
            "FI10_NAMESPACE_READBACK_REFUSED reason=live_enablement_held "
            "(synthetic harness only; no Feeder grant; no production purge)",
            file=sys.stderr,
        )
        return 1

    # Import after argv parse so --help works without installing the package.
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from scrutator.fi10.harness import write_receipt

    receipt = write_receipt(args.receipt)
    print(json.dumps({"ok": receipt["all_pass"], "receipt": str(args.receipt), "sha256": receipt["receipt_sha256"]}))
    return 0 if receipt["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
