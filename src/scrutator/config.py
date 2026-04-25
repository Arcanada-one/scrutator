"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Scrutator"
    app_version: str = "0.1.0"
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

    ltm_connector: str = "cursor"
    ltm_model: str = "auto"
    ltm_mc_url: str = "http://100.121.155.54:3900"
    ltm_mc_api_key: str = ""
    ltm_max_entities_per_chunk: int = 10
    ltm_dedup_similarity: float = 0.85
    ltm_rerank_top_n: int = 5

    model_config = {"env_prefix": "SCRUTATOR_"}


settings = Settings()
