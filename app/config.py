from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    anthropic_model: str = "claude-sonnet-5-5"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "documents"
    upload_dir: str = "./data/uploads"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
