import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application Settings and Configuration.
    Loaded automatically from environment variables or .env file.
    """
    APP_NAME: str = "Intelligent Document Research Assistant"
    DEBUG: bool = True
    CORS_ORIGINS: str = "*"
    
    # LLM Settings
    LLM_PROVIDER: str = "gemini"  # "gemini" or "openai"
    GEMINI_MODEL: str = "gemini-3.6-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # Embedding Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CHROMA_DB_DIR: str = str(BASE_DIR / "storage" / "chroma_db")
    UPLOAD_DIR: str = str(BASE_DIR / "storage" / "uploads")
    
    # Text Chunking Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    # Relevance Filtering — raise this in .env to filter low-relevance chunks.
    # FlashRank scores are model-dependent; very small scores indicate no evidence.
    # Set to 0.0 to disable post-rerank filtering.
    RERANK_RELEVANCE_THRESHOLD: float = 0.001

    # LLM temperature — 0.0 = fully deterministic, factual output.
    LLM_TEMPERATURE: float = 0.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Singleton settings instance
settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
