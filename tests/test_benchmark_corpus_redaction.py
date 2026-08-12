"""Regression tests for the benchmark corpus secret-redaction pass (SEC-0051).

The ml-vs-llm benchmark samples chunks from the private knowledge base and
commits them to this PUBLIC repository. A live OpenRouter key travelled that
exact path once (sampler -> golden/corpus_100.json -> three derived files),
which forced a key rotation and a full history rewrite. These tests pin the
redaction module so the sampler can never republish a credential-shaped
string, and pin that the sampler actually calls it.

The benchmark directory name contains a hyphen, so the module is loaded by
file path rather than imported as a package.
"""

import importlib.util
import pathlib

_REDACT_PATH = pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "ml-vs-llm" / "utils" / "redact.py"
_spec = importlib.util.spec_from_file_location("bench_redact", _REDACT_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

redact_secrets = _mod.redact_secrets
REDACTION = _mod.REDACTION

# Deliberately fake, credential-SHAPED fixtures: the point of the test is that
# anything of this shape must not survive redaction.
FAKE_OPENROUTER = "sk-or-v1-" + "a1b2c3d4e5" * 6
FAKE_OPENAI = "sk-" + "Zz9Yy8Xx7Ww6Vv5Uu4Tt3"
FAKE_GITHUB = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
FAKE_AWS = "AKIA" + "ABCDEFGHIJKLMNOP"


def test_openrouter_key_is_redacted():
    text = f'{{"api_base": "https://openrouter.ai", "key": "{FAKE_OPENROUTER}"}}'
    out = redact_secrets(text)
    assert FAKE_OPENROUTER not in out
    assert REDACTION in out


def test_openai_key_is_redacted():
    out = redact_secrets(f"OPENAI_API_KEY={FAKE_OPENAI}")
    assert FAKE_OPENAI not in out


def test_github_token_is_redacted():
    out = redact_secrets(f"remote set-url https://x:{FAKE_GITHUB}@github.com/o/r")
    assert FAKE_GITHUB not in out


def test_aws_key_id_is_redacted():
    out = redact_secrets(f"aws_access_key_id = {FAKE_AWS}")
    assert FAKE_AWS not in out


def test_authorization_header_keeps_prefix():
    out = redact_secrets("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345")
    assert "Authorization: Bearer" in out
    assert "abcdefghijklmnopqrstuvwxyz012345" not in out


def test_private_key_block_is_redacted():
    pem = "-----BEGIN PRIVATE KEY-----\nMIIEvFAKEFAKEFAKE\n-----END PRIVATE KEY-----"
    out = redact_secrets(f"config:\n{pem}\nrest")
    assert "MIIEvFAKEFAKEFAKE" not in out
    assert out.count(REDACTION) == 1


def test_plain_prose_untouched():
    text = (
        "Scrutator search retrieval engine uses PostgreSQL with pgvector; "
        "the API listens on port 8310 and skips short chunks (<50 chars). "
        "See docs/architecture.md section sk-learn is not a secret."
    )
    assert redact_secrets(text) == text


def test_sampler_wires_redaction_in():
    sampler = (
        pathlib.Path(__file__).resolve().parents[1] / "benchmark" / "ml-vs-llm" / "scripts" / "01_sample_corpus.py"
    ).read_text(encoding="utf-8")
    assert "from utils.redact import redact_secrets" in sampler
    assert "redact_secrets(content)" in sampler
