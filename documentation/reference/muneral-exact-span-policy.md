# Exact Muneral field/span policy candidate

`tools/muneral_sync/span_policy.py::scan_task_field` scans the complete field with the existing Python scanner and gitleaks before classifying any finding. Only `generic-entropy` may become INFO; every named critical or gitleaks finding remains blocking. No regex or whole-field allowance is accepted. The exact task UUID, field, line, entire field SHA256 and matched span SHA256 must match a single proven entry.

The shipped 136-entry catalog is **disabled and not admitted**. It contains 132 private immutable task-definition identifiers and four historical Orca terminal identifiers. Values and descriptions are absent; immutable source references and sanitized historical accounting hashes supply provenance. The original 132-entry proposal and full-scan FAIL remain in the existing digest task evidence.

An enabled policy requires the caller to provide its exact canonical policy digest only after native admission; this module does not authenticate that admission or fetch source proof. An absent, candidate, altered or unbound enabled policy refuses. Source/schema validation is not independent admission. Existing SECURITY must review the source and historical references; existing VERIFIER must execute negative controls before any admitted policy publication or full scan.

This PR adds a field-scanner entry point and disabled catalog. It does not install it in the scheduled sync, modify the existing outbound wire scanner, enable the catalog, rewrite corpus/history, rotate secrets or renew any digest grant. Runtime integration and the trusted native-admission caller remain separate unmeasured obligations. Do not use a scan of a serialized envelope as the original field context.

Required controls cover changed task/field/line/whole-field/span/source, a missing admission digest, near misses, synthetic credential assignments to metadata keys, adjacent provider and entropy secrets, named critical and gitleaks findings. Original critical findings remain in the output; a permitted entropy classification changes severity to INFO, never erases the finding.

Reverse/refuse if policy scope broadens, a named/gitleaks finding is reduced, raw protected values escape, corpus history is edited, a candidate is treated as admitted, or grant renewal precedes exact independent F1/F2/A4 acceptance.

### Assignment identity and ambiguity refusal

Classification requires the actual scanner assignment key to match the proven
metadata key. If a line contains repeated qualifying values with the same hash,
all findings in that line/hash group remain critical, even for identical keys.
The classifier reuses the scanner regex and entropy threshold. Named rules and
gitleaks remain blocking. This disabled seam is not integrated into the digest
consumer: final serialized-wire scanning remains required. Provenance reference
shape does not authenticate an issuer; native caller admission is separate.
