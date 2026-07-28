"""
Application settings, loaded from environment variables / a .env file.
Copy .env.example to .env and adjust values for your machine.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Path to the trained pipeline exported from the notebook (Phase 2.6)
    MODEL_PATH: str = "models/house_price.pkl"

    # Path to the list of allowed locations exported from the notebook
    LOCATIONS_PATH: str = "models/locations.json"

    # Comma-separated list of origins allowed to call this API
    ALLOWED_ORIGINS: str = "http://localhost:5173"

    APP_NAME: str = "House Price Prediction API"
    APP_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Cached so the .env file is only parsed once per process."""
    return Settings()
