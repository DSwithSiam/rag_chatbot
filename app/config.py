from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    max_chunk_size: int = 900
    chunk_overlap: int = 120
    top_k: int = 4
    min_similarity: float = 0.40
    max_history_turns: int = 8

    documents_dir: str = "data/documents"
    index_dir: str = "data/index"

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"


settings = Settings()
