from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = ""
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "banco_web"
    chunk_size: int = 512
    chunk_overlap: int = 64
    history_window_n: int = 6
    top_k: int = 8
    rerank_top_k: int = 3
    scrape_base_url: str = "https://www.bbva.com.co/"
    scrape_max_pages: int = 50
    db_path: str = "conversations.db"

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    """Singleton: lru_cache garantiza una única instancia."""
    return Settings()
