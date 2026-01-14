"""Web application configuration."""

from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Aneurysm Detection System"
    DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production-use-a-secure-random-key"

    # Database
    DATABASE_URL: str = "sqlite:///./webapp/aneurysm_detection.db"

    # Session
    SESSION_EXPIRE_HOURS: int = 24
    SESSION_COOKIE_NAME: str = "session_token"

    # File Upload
    UPLOAD_DIR: Path = Path("webapp/uploads")
    VISUALIZATION_DIR: Path = Path("webapp/uploads/visualizations")
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    # Pipeline
    PIPELINE_CONFIG_PATH: str = "config.yaml"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
