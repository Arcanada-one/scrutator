# Skill ingest trust boundary

## Purpose

The `skills` namespace stores executable-plan candidates. A valid `SkillPlan`
body is therefore not proof that the plan has been reviewed or promoted. The
ingest path separates three concerns:

- the caller's JSON is validated for compatibility with the ARAS Rust wire
  type;
- indexed maturity is derived from a protected, repository-tracked registry;
- only a validated semantic projection is chunked and embedded.

The exact source remains available for hash-bound fetch without becoming model
context during indexing.

## Promotion registry

The registry is immutable application input shipped in the Scrutator image.
Each entry contains:

- a normalized relative POSIX `source_path`;
- the `sha256:` hash of the exact UTF-8 source bytes;
- an approved maturity of `validated` or `production`.

Protected pull-request review and CI are the authorization workflow for a
registry change. There is no runtime promotion endpoint, promotion secret,
mutable database policy, or environment override.

The registry loader validates its complete input deterministically. Absolute,
empty, dot, dot-dot, backslash-containing, NUL-containing, or
non-deterministically normalized paths are unsafe. Malformed hashes,
unsupported maturities, duplicate normalized identities, and conflicting
entries fail application startup before the registry can be used. At ingest,
an exact path-and-hash match yields the approved maturity. A missing entry or
any path/hash mismatch yields `draft`; it can never elevate a document.

The body `maturity` field remains required and enum-validated because it is part
of the ARAS `SkillPlan` wire format. It is non-authoritative and is never copied
to searchable maturity metadata.

## Semantic projection and exact source

Scrutator parses the plan using the existing Rust-parity rules, drops struct
fields that the Rust consumer ignores, and serializes the resulting known-field
semantic value into deterministic JSON. That projection is the only content
sent to chunking and dense or sparse embedding.

The original UTF-8 string is still used for the whole-document SHA-256 and is
stored only in the isolated `source_documents` exact-source table. Fetch
therefore preserves byte equality while ignored fields cannot smuggle content
into embeddings.

## Fail-closed content checks

Before chunking, Scrutator recursively checks decoded strings and object keys in
the semantic projection. It rejects unsafe control and Unicode format
characters with a typed, non-echoing `422` error. Tab, line feed, and carriage
return remain permitted inside semantic text; other C0/C1 controls and format
controls, including bidirectional overrides such as `U+202E`, are rejected.

Scrutator then scans the deterministic semantic JSON for strong prompt-role and
instruction-override markers. The role-marker set covers the supported common
chat templates, including Gemma `<start_of_turn>` and `<end_of_turn>` markers.
A flagged skill plan is rejected before chunking, embedding, namespace creation,
or persistence. Non-skill namespaces retain the existing non-blocking
observability scan.

Errors identify the contract class and, where useful, a field path constructed
only from known schema segments and bounded numeric indices. They never echo
caller-controlled object keys, raw paths, or content. Unknown-key, unsafe
control, and injection-marker errors use fixed bounded reason text.

## Verification invariants

Tests must prove:

1. A body declaring `production` indexes as `draft` without an exact registry
   match.
2. Only an exact normalized path plus exact raw-content hash receives its
   registry-approved maturity.
3. Missing or mismatched registry data never elevates a plan, while malformed,
   duplicate, conflicting, or unsafe registry data fails loader startup.
4. Ignored-field markers do not reach chunking or embedding.
5. Markers in semantic fields, including Gemma markers, are rejected before
   embedding.
6. `ESC`, C1 controls, and `U+202E` are rejected with typed non-echoing errors.
7. The raw source and its document hash remain byte-exact for fetch.
8. Existing non-skill ingest behavior remains unchanged.
9. Absolute, empty, dot, dot-dot, backslash-containing, NUL-containing, and
   non-deterministically normalized registry paths are rejected, including
   duplicate normalized identities under the documented Unicode policy.
10. No typed error echoes an untrusted key, raw source path, or source content.

## Rejected alternatives

A runtime environment registry would allow deployment drift outside the
protected source change. A privileged promotion API would add mutable state,
another credential, and a new authorization surface. Both are unnecessary for
the current low-volume promotion workflow and weaken the audit trail.
