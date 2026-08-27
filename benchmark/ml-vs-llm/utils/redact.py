"""Secret redaction for benchmark corpus content.

The benchmark corpus is built by sampling live knowledge-base chunks
(scripts/01_sample_corpus.py). The knowledge base is private and may contain
credentials; the sampled corpus is committed to a PUBLIC repository. Every
chunk therefore passes through redact_secrets() before it is written —
otherwise regenerating the corpus republishes whatever the source held.
That is exactly how a live OpenRouter key ended up tracked in
golden/corpus_100.json and three downstream files (SEC-0051): the sampler
copied it verbatim out of a KB chunk.

Patterns are deliberately shape-based (no verification round-trips): a false
positive costs one redacted token in a benchmark corpus, a false negative
publishes a credential.
"""

import re

REDACTION = "[REDACTED-SECRET]"

SECRET_PATTERNS: list[re.Pattern[str]] = [
    # OpenAI / OpenRouter style keys (sk-..., sk-or-v1-..., sk-proj-...)
    re.compile(r"sk-[A-Za-z0-9-]{4,32}-[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    # GitHub tokens (classic + fine-grained)
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,255}"),
    # AWS access key id
    re.compile(r"AKIA[0-9A-Z]{16}"),
    # Slack tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    # Authorization headers carrying an opaque token
    re.compile(r"(?i)(Authorization\s*:\s*(?:Bearer|Basic|token)\s+)[A-Za-z0-9._~+/=-]{16,}"),
    # Private key material — redact the whole PEM body conservatively
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    # key/token/secret/password assignments with a long opaque value
    re.compile(
        r"(?i)((?:api[_-]?key|access[_-]?token|client[_-]?secret|password|secret[_-]?key)"
        r"\s*[:=]\s*[\"']?)[A-Za-z0-9._~+/=-]{16,}"
    ),
]


def redact_secrets(text: str) -> str:
    """Replace credential-shaped substrings with a redaction marker.

    Patterns with a capture group keep the captured prefix (e.g. the
    ``Authorization:`` header name or the ``api_key=`` assignment) so the
    surrounding text stays readable; the secret value itself is replaced.
    """
    for pattern in SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda m: m.group(1) + REDACTION, text)
        else:
            text = pattern.sub(REDACTION, text)
    return text
