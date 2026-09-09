"""
Application configuration — loaded from .env file via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # IBM watsonx (optional)
    watsonx_api_key: str = ""
    watsonx_project_id: str = ""
    watsonx_url: str = "https://us-south.ml.cloud.ibm.com"

    # OpenAI (primary for students)
    openai_api_key: str = ""

    # Academic APIs
    semantic_scholar_api_key: str = ""

    # Storage
    upload_dir: str = "uploads"
    chroma_persist_dir: str = "chroma_db"

    # Processing
    max_upload_size_mb: int = 50
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Server
    port: int = 8000
    debug: bool = True
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def has_watsonx(self) -> bool:
        return bool(self.watsonx_api_key and self.watsonx_project_id)

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key)


settings = Settings()

# Ensure required directories exist
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_dir, exist_ok=True)
