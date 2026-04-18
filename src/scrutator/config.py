"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Scrutator"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8310
    debug: bool = False

    embedding_api_url: str = "http://localhost:8300"

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

    model_config = {"env_prefix": "SCRUTATOR_"}


settings = Settings()
