"""Application configuration via environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Scrutator"
    app_version: str = "0.3.0"
    host: str = "0.0.0.0"
    port: int = 8310
    debug: bool = False

    embedding_api_url: str = "http://localhost:8300"
    embedding_timeout: float = 30.0
    embedding_max_retries: int = 3

    database_url: str = "postgresql://scrutator:scrutator@localhost:5432/scrutator"
    database_pool_min: int = 2
    database_pool_max: int = 10

    search_limit_default: int = 10
    search_timeout_ms: int = 5000

    dream_dedup_threshold: float = 0.92
    dream_crossref_threshold: float = 0.7
    dream_stale_days: int = 90
    dream_max_results: int = 50
    dream_analysis_timeout_ms: int = 30000

    memory_default_importance: float = 0.5
    memory_decay_days: int = 180
    memory_max_bulk_size: int = 100

    ltm_connector: str = "openrouter"
    ltm_model: str = "google/gemini-2.5-flash"
    ltm_mc_url: str = "https://connector.arcanada.ai"
    ltm_mc_api_key: str = ""
    ltm_max_entities_per_chunk: int = 10
    ltm_dedup_similarity: float = 0.85
    ltm_rerank_top_n: int = 5

    # LTM-0012 temporal layer
    ltm_temporal_enabled: bool = True
    ltm_auto_invalidate: bool = True
    ltm_temporal_boost: float = 0.3
    ltm_max_events_per_chunk: int = 10

    # LTM-0013 reflect layer (R in TEMPR)
    ltm_reflect_enabled: bool = True
    ltm_reflect_max_chunks_per_run: int = 50
    ltm_reflect_max_meta_facts_per_chunk: int = 5
    ltm_reflect_budget_usd: float = 0.01
    ltm_reflect_budget_req_count: int = 100
    ltm_reflect_max_depth: int = 1
    ltm_recall_include_meta_facts: bool = False
    ltm_recall_meta_fact_score_factor: float = 0.5

    # LTM-0018 reflect grouping primitive
    ltm_reflect_grouping: Literal["entity", "cosine"] = "cosine"
    ltm_reflect_cosine_threshold: float = 0.85

    # SRCH-0029 M2: ColBERT late-interaction rerank (default OFF — measure-first per consilium)
    rerank_enabled: bool = False
    rerank_pool_multiplier: int = 4  # fetch_limit = limit * multiplier when rerank ON
    rerank_colbert_max_pool: int = 30  # hard cap on candidates sent to ColBERT

    # SRCH-0023: tenant isolation — Auth Arcana identity + authorization
    auth_arcana_jwks_url: str = "https://auth.arcanada.ai/.well-known/jwks.json"
    # Generic profiles fail closed until their exact resource contract is set.
    # The dedicated LTM M2M profile below remains pinned and independently usable.
    auth_service_audience: str = ""
    auth_service_scope: str = ""
    auth_oidc_issuer: str = ""
    auth_oidc_audience: str = ""
    auth_oidc_scope: str = ""
    # LTM-0026 dedicated M2M reader profile. Literals make environment drift
    # fail at startup instead of silently widening the accepted trust domain.
    auth_ltm_issuer: Literal["https://auth.arcanada.ai"] = "https://auth.arcanada.ai"
    auth_ltm_audience: Literal["urn:arcanada:scrutator:ltm"] = "urn:arcanada:scrutator:ltm"
    auth_ltm_scope: Literal["kb:ltm.read"] = "kb:ltm.read"
    auth_ltm_client_id: Literal["muneral-kb-sync"] = "muneral-kb-sync"
    auth_ltm_observer_client_id: Literal["kb-observer"] = "kb-observer"
    auth_ltm_agent_client_id: Literal["arcana-agent-kb-reader"] = "arcana-agent-kb-reader"
    auth_ltm_max_token_lifetime_seconds: Literal[300] = 300
    auth_arcana_introspect_url: str = ""  # arc_api_* service-token introspection; [to-be-confirmed]
    auth_arcana_openfga_url: str = ""  # OpenFGA base URL; [to-be-confirmed] — empty = FK-cache fallback only
    auth_arcana_openfga_store_id: str = ""

    # Dual-auth grace (30-day window, Auth Arcana mandate §8): False = advisory would-deny audit
    # log only, never rejects. MUST stay False until the operator explicitly flips it in prod.
    # env_prefix below makes this SCRUTATOR_AUTH_ENFORCE.
    auth_enforce: bool = False
    # LTM-0025 structured/generic ingest credential. This is separate from
    # both reader grants and the /v1/index Feeder credential.
    ltm_writer_token: str = ""
    ltm_writer_namespaces: str = ""
    # JSON object mapping each writable namespace to one or more protected
    # source URI prefixes accepted by DELETE /v1/ltm/source.
    ltm_writer_source_prefixes: str = ""
    # SRCH-0048 co-located Feeder tombstone credential. This is separate from
    # reader grants and is accepted only by DELETE /v1/index.
    feeder_token: str = ""
    feeder_namespaces: str = ""
    rollback_token: str = ""
    rollback_namespaces: str = ""
    operator_rollback_token: str = ""

    # Postgres RLS defense-in-depth (Phase 6, operator-gated) — inert until the migration lands.
    rls_enabled: bool = False

    model_config = {"env_prefix": "SCRUTATOR_"}


settings = Settings()
