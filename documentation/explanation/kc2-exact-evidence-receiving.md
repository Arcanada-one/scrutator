# Namespace-bounded exact evidence receiving

An ingest hash describes the original document. It does not establish that lossy chunk reconstruction returns identical bytes. Exact evidence receiving uses the existing isolated raw-content table; approximate fallback remains explicitly labelled `content_exact=false`.

## Scope and integrity

`SCRUTATOR_EVIDENCE_EXACT_BYTES` remains false by default. The optional JSON list `SCRUTATOR_EVIDENCE_EXACT_NAMESPACES` applies the same predicate to single ingest, batch ingest and fetch. Omitted/null scope preserves historical global-on behavior; an explicit empty list selects none; a list of namespace names selects those namespaces. Invalid names, wrong shapes and duplicates fail settings validation. Storage scope does not grant reader or writer authority. Skills retain their independent exact source, size cap and missing-source 409 behavior.

Before labelling returned bytes exact, the fetcher verifies the complete raw UTF-8 body against both the raw-row digest and every current chunk's ingest digest. Missing, malformed, stale or corrupt rows degrade to approximate reconstruction. The original ingest hash stays unchanged. Offset responses retain the whole-document hash; a slice is not expected to hash to that whole-document value.

A read-only repeatable-read transaction binds raw lookup to the selected namespace/path and checks the selected chunk identifiers, timestamps and stamps against the current generation. A concurrent replacement causes approximate fallback, even when body bytes stayed equal. Chunk-manifest offsets retain their existing reconstructed-character coordinate meaning; they are not claimed to locate chunks in raw content.

## Additive population

`POST /v1/index/evidence-exact` uses the existing dedicated feeder credential and IndexRequest/IndexResponse models. The namespace comes exclusively from the explicit singleton server scope and writer membership. The payload namespace cannot redirect the effect. Missing credentials return 401; insufficient writer scope returns 403; missing/global/empty/multiple server scope returns 409.

The operation verifies full original UTF-8 bytes against every existing chunk's document identifier and digest under the source advisory lock and chunk row locks. It inserts only an absent matching raw row, refuses conflicting existing rows, and performs no embeddings, chunk replacement, namespace creation or graph writes. The response reports `chunks_indexed=0` and strategy `evidence_exact_created` or `evidence_exact_present`.

`SCRUTATOR_EVIDENCE_POPULATION_MAX_BYTES` defaults to 262144 and accepts a strict positive integer at most 1048576. The bound is checked before database access. This bounds a single population request; aggregate storage, WAL, backup, latency and retention budgets require separate qualification.

## Deployment and rollback

The canonical deploy transaction preserves the environment, so a code merge alone does not enable exact evidence. Activation requires qualified installed identity, effective explicit scope, existing writer/reader authority, authoritative originals, generation/graph preservation, measured operational limits and genuine reader byte comparisons. A few successful fetches prove only those samples.

On integrity, scope, generation or operational failures, disable the scoped operation through canonical deployment/configuration custody, quiesce the affected writer, and verify approximate fallback. Disabling reads does not erase retained raw bodies, WAL or backups. Separately qualified retention and restoration procedures account for those data. Exact bytes confer no execution authority.

The raw-only route also bounds its complete wire body before JSON decoding to six times the configured UTF-8 content budget plus 65,536 bytes for the JSON envelope. Declared oversize, malformed framing and a stream crossing that bound are refused before the handler or database. The independent UTF-8 content bound still applies after decoding.
