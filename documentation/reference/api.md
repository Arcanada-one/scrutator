# API Routes

Scrutator exposes a Tailscale-only FastAPI service. `GET /health` is
unauthenticated. Every `/v1/*` route requires either a tenant reader context or
the route-specific machine capability enforced by the application.

The test suite compares this inventory with the generated OpenAPI schema, so a
route addition or removal must update this page in the same pull request.

| Route | Purpose |
|---|---|
| `GET /health` | Service status and version |
| `POST /v1/chunk` | Chunk a document without indexing it |
| `GET /v1/index/capability` | Read the effective namespace scope of the authenticated feeder credential |
| `GET /v1/index/rollback-capability` | Read the effective namespace scope and operator flag of the authenticated rollback credential |
| `POST /v1/index` | Index one document |
| `DELETE /v1/index` | Delete one indexed source |
| `POST /v1/index/batch` | Index a bounded same-namespace batch |
| `POST /v1/search` | Run hybrid retrieval |
| `POST /v1/fetch` | Fetch a document or bounded range by opaque ID |
| `GET /v1/navigate/outline` | Build a source outline |
| `GET /v1/navigate/section` | Read section context for one chunk |
| `GET /v1/chunks` | Look up chunks |
| `POST /v1/namespaces` | Create or update a namespace |
| `GET /v1/namespaces` | List visible namespaces |
| `GET /v1/stats` | Read index statistics |
| `POST /v1/dream/analyze` | Analyze knowledge structure |
| `POST /v1/edges` | Create graph edges |
| `DELETE /v1/edges` | Delete graph edges by creator |
| `GET /v1/edges/{chunk_id}` | List edges for a chunk |
| `POST /v1/edges/by-path` | Create graph edges from source paths |
| `POST /v1/memories` | Index one memory |
| `DELETE /v1/memories` | Delete memories for a session |
| `POST /v1/memories/bulk` | Index a bounded memory batch |
| `POST /v1/memories/recall` | Recall memories |
| `GET /v1/memories/stats` | Read memory statistics |
| `POST /v1/ltm/ingest` | Ingest structured long-term-memory facts |
| `DELETE /v1/ltm/source` | Delete one long-term-memory source |
| `GET /v1/ltm/jobs/{job_id}` | Read an ingest job |
| `POST /v1/ltm/recall` | Recall long-term-memory facts |
| `GET /v1/ltm/entities` | List visible entities |
| `GET /v1/ltm/graph` | Read the entity graph |
| `POST /v1/ltm/reflect` | Run bounded reflection |
| `GET /v1/ltm/meta_facts` | List reflected meta-facts |
| `GET /v1/ltm/events` | List temporal entity events |

FastAPI also serves interactive discovery at `/docs`, `/redoc`, and
`/openapi.json` on the same mesh-only origin.
