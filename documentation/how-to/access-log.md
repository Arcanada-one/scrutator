# Read the access log: who uses the knowledge base, which namespaces, how

Scrutator writes one structured record per request, so that "who uses this base and how" can be answered from facts.

## What is recorded

One JSON line per request in `access.jsonl` inside `SCRUTATOR_ACCESS_LOG_DIR` (default `/app/logs/access`, on the
`scrutator-logs` volume, so it survives a container recreate and a Deploy). Empty setting = log off.

| Field | Meaning |
|---|---|
| `ts` | UTC time, millisecond precision |
| `identity` | `reader:<principal id>` for a verified reader, or a fixed machine kind: `capability:feeder`, `capability:ltm-writer`, `capability:rollback`, `capability:operator-rollback`, `capability:capability-projection`; `unverified` when a credential was refused; `anonymous` in the dual-auth grace window; `none` when no credential was sent |
| `auth` | `ok`, `denied` (a credential was presented and refused) or `none` |
| `mode` | `read` or `write`, set by the dependency that authorised the request |
| `src` | source address of the connection |
| `method`, `route` | HTTP method and the route template (`unmatched` for a 404) |
| `namespaces` | namespaces the request named or was checked against; `[]` for routes that act on the whole allowed set |
| `allowed_n` | for a reader, how many namespaces it could reach |
| `status`, `ms` | result code and duration |
| `qs_len`, `q_len`, `q_sha16` | length of the URL query string; for search-like routes the length and a 16-hex hash of the query text |

## What is never recorded

Credential values, token headers, request bodies, query text and URL query strings. The middleware does not read the
`Authorization` header or any token header; the identity is the principal id that the verifier returned or the fixed
capability kind. `GET /health` (the container probe) is not recorded. A test sends a unique secret in every credential
header and fails if it appears in the file.

## Retention and failure

Daily files (midnight UTC), 35 kept (`SCRUTATOR_ACCESS_LOG_BACKUPS`), so at least 30 days. A write error never fails a
request; it is reported in the process log at most once a minute.

## Report

```
docker exec <scrutator-container> python -m scrutator.tools.access_report --days 1
docker exec <scrutator-container> python -m scrutator.tools.access_report --days 30 --json
```

Read-only. For each identity: requests, reads, writes, denied, namespaces with counts, source addresses, status codes,
first and last seen; then the known identities with zero use in the window (the machine kinds above plus the principals
in `principal_namespace_grants`; use `--no-db` to skip the database).

## Limits

- A shared secret is one identity: every feeder lane shares `capability:feeder`, and every agent that uses the same reader
  principal shows as that principal. Source address and route mix are the only way to tell them apart.
- Routes that read by document or chunk id, stats, the namespace list and edges act on the whole allowed set; they log
  `namespaces: []` and the report counts them as `(unscoped)`.
- There is no history before the first record.
