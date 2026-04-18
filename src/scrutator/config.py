"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Scrutator"
    app_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8310
    debug: bool = False

    embedding_api_url: str = "http://localhost:8300"

    database_url: str = "postgresql+asyncpg://scrutator:scrutator@localhost:5432/scrutator"

    model_config = {"env_prefix": "SCRUTATOR_"}


settings = Settings()
