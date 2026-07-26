# Downstream consumers

This benchmark owns the versioned golden set and the pass/fail contract for
Scrutator's `/v1/search` retrieval path. It is intentionally separate from
benchmarks that measure the LTM recall API or extraction models.

| Consumer | Relationship | Status |
|---|---|---|
| ML-versus-LLM extraction benchmark | Reuses the human-reviewed golden-set governance pattern, not this retrieval harness | Available in `benchmark/ml-vs-llm/` |
| End-to-end recall pipeline | May contribute reranking results as candidates for a future cross-encoder baseline | Separate benchmark; no runtime dependency |
| Multi-tenant search | Supplies the namespace and reader-authorization boundary that every live harness call must obey | Enforced by the production API |
| Search recall baseline | Provides the original 33-row seed, reference script, and methodology report | Preserved as frozen historical evidence |
| Quantized-embedding research | May consume a future promoted golden-set release for comparative recall measurements | Planned consumer; not wired |
| Search recall CI | Consumes the harness exit-code contract: success, threshold failure, or infrastructure failure | Manual workflow available |
| Index-freshness monitoring | Complements this harness by detecting corpus drift that liveness filtering can only report | Separate operational concern |

Historical reports and result bundles retain their original provenance identifiers. They are
frozen evidence, not current operator guidance.
