FROM python:3.14-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base AS production
COPY src/ src/
ENV PYTHONPATH=/app/src
EXPOSE 8310
# INFRA-0120: bind host/port are env-driven (SCRUTATOR_HOST / SCRUTATOR_PORT) instead of
# hardcoded, so prod can tighten the bind without a code change. network_mode:host means the
# bind lands directly on the host interface.
# exposure: justified expires=2026-10-07 — default SCRUTATOR_HOST=0.0.0.0 kept to preserve
#   current behaviour (prod .env may or may not set it; a 127.0.0.1 default would break mesh
#   reachability). Public :8310 is firewall-closed on the production host (re-verified
#   2026-09-04); ingress is Tailscale-only and API data access requires JWT namespace grants.
#   docker-compose.yml records the bounded Tier-3 exposure contract. Tier-2 upgrade path:
#   set SCRUTATOR_HOST=<host Tailscale IP, e.g. 100.70.137.104 (arcana-kb)> in prod .env to drop
#   the wildcard. Ref: network-exposure-baseline (Tier-3 justified).
CMD ["sh", "-c", "uvicorn scrutator.health:app --host \"${SCRUTATOR_HOST:-0.0.0.0}\" --port \"${SCRUTATOR_PORT:-8310}\""]
