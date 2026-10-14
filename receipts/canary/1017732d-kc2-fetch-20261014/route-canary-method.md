# KC2 fetch route canary method

A loopback-only Uvicorn instance imported the candidate Scrutator FastAPI app. The focused `/v1/fetch` instance supplied deterministic `kc2-store` chunk rows from the repository test fixture and disabled exact-byte mode, so it exercised the actual route and fetcher refusal path without a database or production access. The same harness ran against candidate head `aa35e8cd2a97227733f3fde4610bb5aace29a8be` and parent `ecb255a594fd17fa76ee8090eb6ce142144604a9`; the expected response was HTTP 409. Candidate returned 409; parent returned HTTP 200 from lossy reassembly (`content_exact=false` in the parent implementation).

A separate candidate instance enforced authentication and had no dependency overrides. The committed route-presence plan issued its unauthenticated probes; all mutating routes were refused with HTTP 401, and all 36 probes produced verified route-presence observations (health 200, routes 401 with matching OpenAPI declarations).

These are candidate-local HTTP observations. They do not establish database-backed exact-row retrieval, production deployment, or installed runtime behavior.
