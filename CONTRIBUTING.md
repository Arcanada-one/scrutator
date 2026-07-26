# Contributing

Scrutator accepts focused pull requests against `main`.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
PYTHONPATH=src .venv/bin/pytest tests/ -v
PYTHONPATH=src:benchmark/scrutator .venv/bin/pytest benchmark/scrutator/tests/ -v
.venv/bin/ruff check src/ tests/ benchmark/scrutator/
.venv/bin/ruff format --check src/ tests/ benchmark/scrutator/harness.py benchmark/scrutator/tests/
```

Add a regression test for every behavior change. Keep namespace authorization,
machine-capability checks, deterministic ordering, and fail-closed error paths
intact.

## Pull requests

Explain the user-visible change, test evidence, and any deployment or rollback
impact. Do not commit secrets or production data. Report security findings
through [SECURITY.md](SECURITY.md), not a public issue.
