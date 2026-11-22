# Root production image evidence: scrutator PR #202 r2

## Source binding

- Measured source commit: `72234ac81ed70fdbc31942b7d0c6b79634020ecf`
- Base commit: `d08e14f2f0efcacd7ee74d4afb4d0d0c9d100b8d`
- Dockerfile plus `requirements-dev.txt` diff SHA-256: `dc060a68023e7043ac7d2f6d8f6ab7c62bceecf1381c78d26064c67cb305663f`
- Registry reference: `docker.io/library/python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928`
- Registry API query returned this OCI index digest for tag `python:3.15-slim` on 2026-11-22.

## Build and smoke result

Root measured this exact commit on 2026-11-22T18:34:18Z with `docker build --pull --target production`:

- Build exit: `0`
- Image ID: `sha256:4916cbe725c81a26af7fbefc39461891b8d924bdc5042dac95da7729fb978fd4`
- Python: `3.15.2`
- In-image import sweep, network disabled: `63` imported, `0` failed
- Container `/health`: HTTP `200`, `{"status":"ok","service":"Scrutator","version":"0.3.0"}`
- Root build log SHA-256: `13cdffa53a784f13e65cabb23de81d7ef9156f6376972791253ffbc6dccaca47`

The source report supplied by Root had full-file SHA-256 `48d50a014f931fa5bffbf290dff456b4fa446213f8becaf476f396b053388a6b`; its embedded `DONE sha256` field is `c813b9854083498656c710b587d0cfd23efcf7599b32e31cb7750a080ed95bca`, matching the report content with that self-hash line omitted. This committed summary omits private workspace paths from the source report.

## Limits

Root measured a local image only: it was not pushed or deployed, external services were not exercised, and Root did not run the test suite. The full test suite was measured separately on this source tree under lane-managed CPython 3.15.2 and is recorded in the PR readiness report.
