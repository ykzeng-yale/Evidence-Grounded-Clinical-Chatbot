"""Centralized settings loaded from .env."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # NCBI
    ncbi_api_key: str = ""
    pubmed_email: str = "anonymous@example.com"

    # Tavily
    tavily_api_key: str = ""

    # Supabase
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    use_supabase_cache: bool = False

    # Limits
    max_pubmed: int = 5
    max_trials: int = 5
    max_preprints: int = 3
    use_web_search: bool = False


settings = Settings()
